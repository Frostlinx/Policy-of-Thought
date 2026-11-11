"""Per-problem LoRA training with HF likelihoods and optional vLLM generation.

The default objective minimizes the clipped GRPO loss with reference KL .02.
Full-path output tokens are targets; historical outputs use a fixed pre-update
scoring snapshot. See docs/METHOD.md for objective variants and replay details."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
import inspect
import math
from pathlib import Path
import re
import shutil
import time
from typing import Any, Sequence
import uuid


@dataclass
class BackendConfig:
    model_path: str
    generation_backend: str = "vllm"
    device: str = "cuda:0"
    dtype: str = "bfloat16"
    vllm_url: str = "http://127.0.0.1:8000"
    vllm_model_name: str = "pot-base"
    adapter_dir: str = "work/adapters"
    learning_rate: float = 1e-4
    epochs: int = 3
    clip_ratio: float = 0.3
    loss_variant: str = "paper"
    loss_scope: str = "trajectory"
    beta: float = 0.02
    kl_coefficient: float = 0.005
    entropy_coefficient: float | None = None
    base_kl_coefficient: float | None = None
    weight_decay: float = 0.01
    reference_sync_steps: int = 10
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    disable_policy_dropout: bool = True
    target_modules: list[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj"
    ])
    temperature: float = 0.7
    enable_thinking: bool = False
    advantage_epsilon: float = 1e-8
    advantage_clip: float = 10.0
    max_grad_norm: float = 1.0
    max_context_tokens: int = 16384
    gradient_checkpointing: bool = True
    seed: int = 42
    request_timeout: float = 600.0
    trust_remote_code: bool = False
    keep_adapter_checkpoints: bool = False

    def __post_init__(self):
        if self.loss_scope not in {"trajectory", "completion"}:
            raise ValueError("loss_scope must be trajectory or completion")
        if self.loss_variant not in {"paper", "policy_scaled", "dual_kl"}:
            raise ValueError("loss_variant must be paper, policy_scaled or dual_kl")
        if self.entropy_coefficient is None:
            self.entropy_coefficient = 0.01 if self.loss_variant == "policy_scaled" else 0.0
        if self.base_kl_coefficient is None:
            self.base_kl_coefficient = 0.005 if self.loss_variant == "dual_kl" else 0.0
        if self.loss_variant == "policy_scaled" and self.base_kl_coefficient:
            raise ValueError("policy_scaled has no second base KL; set base_kl_coefficient=0")
        if self.generation_backend not in {"vllm", "hf"}:
            raise ValueError("generation_backend must be 'vllm' or 'hf'")
        if self.dtype not in {"bfloat16", "float16", "float32"}:
            raise ValueError("unsupported dtype")
        if self.epochs < 1 or self.lora_rank < 1 or self.lora_alpha < 1:
            raise ValueError("epochs, lora_rank and lora_alpha must be positive")
        if not 0 < self.clip_ratio < 1 or self.temperature <= 0:
            raise ValueError("clip_ratio must lie in (0,1); temperature must be >0")
        if self.beta < 0 or self.base_kl_coefficient < 0 or self.learning_rate <= 0:
            raise ValueError("KL coefficients must be >=0; learning_rate must be >0")
        if min(self.kl_coefficient, self.entropy_coefficient, self.weight_decay) < 0:
            raise ValueError("loss coefficients and weight_decay must be nonnegative")
        if not 0 <= self.lora_dropout < 1:
            raise ValueError("lora_dropout must lie in [0,1)")
        if self.reference_sync_steps < 0 or self.max_context_tokens < 2:
            raise ValueError("invalid reference synchronization/context limit")
        if self.advantage_epsilon <= 0 or self.advantage_clip <= 0:
            raise ValueError("advantage epsilon and clip must be positive")

    def loss_coefficients(self) -> dict[str, float]:
        return {"policy_coefficient": self.beta if self.loss_variant == "policy_scaled" else 1.0,
                "kl_coefficient": self.kl_coefficient if self.loss_variant == "policy_scaled" else self.beta,
                "base_kl_coefficient": self.base_kl_coefficient,
                "entropy_coefficient": self.entropy_coefficient}


@dataclass
class Sample:
    text: str
    token_ids: list[int] = field(default_factory=list)
    prompt_ids: list[int] = field(default_factory=list)
    old_logprobs: list[float] = field(default_factory=list)
    policy_version: int = 0
    temperature: float = 0.7
    finish_reason: str | None = None
    instance_id: str = ""
    trajectory_prefix: list[Sample] = field(default_factory=list, repr=False, compare=False)

    @property
    def tokens(self) -> list[int]:
        return self.token_ids

    @property
    def prior(self) -> float:
        """Length-normalized probability for tree priors, avoiding underflow."""
        if not self.old_logprobs:
            return 0.0
        return math.exp(sum(self.old_logprobs) / len(self.old_logprobs))


def group_advantages(rewards: Sequence[float], epsilon: float = 1e-8,
                     clip: float = 10.0) -> list[float]:
    """Population-standardized sibling rewards; constant groups have no signal."""
    if not rewards or any(not math.isfinite(float(r)) for r in rewards):
        raise ValueError("rewards must be a nonempty finite sequence")
    if epsilon <= 0 or clip <= 0:
        raise ValueError("epsilon and clip must be positive")
    mean = sum(rewards) / len(rewards)
    std = math.sqrt(sum((r - mean) ** 2 for r in rewards) / len(rewards))
    if std <= epsilon:
        return [0.0] * len(rewards)
    return [max(-clip, min(clip, (r - mean) / max(std, epsilon))) for r in rewards]


def completion_token_ids(choice: dict[str, Any]) -> list[int]:
    """Read vLLM token IDs without lossy text -> tokenizer roundtrips."""
    ids = choice.get("token_ids")
    if ids is None:
        tokens = (choice.get("logprobs") or {}).get("tokens")
        if tokens is None:
            raise RuntimeError("vLLM omitted token IDs; require return_token_ids or token_id:N logprobs")
        ids = []
        for token in tokens:
            match = re.fullmatch(r"token_id:(\d+)", token)
            if match is None:
                raise RuntimeError("vLLM did not honor return_tokens_as_token_ids; refusing retokenization")
            ids.append(int(match.group(1)))
    if any(not isinstance(x, int) or isinstance(x, bool) or x < 0 for x in ids):
        raise ValueError("invalid generated token IDs")
    return list(ids)


def grpo_loss(current, old, reference, advantage: float, clip_ratio: float = 0.3,
              beta: float = 0.02, base=None, base_kl_coefficient: float = 0.005,
              *, policy_coefficient: float = 1.0, entropy=None, entropy_coefficient: float = 0.0):
    """Token-mean clipped GRPO loss over completion tokens only.

    Inputs are 1D chosen-token log probabilities from the same temperature-
    transformed policy. KL uses the nonnegative k3 sampled estimator. Its
    argument is ref/current, so ``exp(ref-current)-(ref-current)-1``.
    All non-current tensors are detached to prevent reference/old gradients.
    """
    import torch

    if current.ndim != 1 or not current.numel():
        raise ValueError("current must contain completion-token log probabilities")
    if old.shape != current.shape or reference.shape != current.shape:
        raise ValueError("log probability arrays must have identical shape")
    if base is not None and base.shape != current.shape:
        raise ValueError("base log probability array has wrong shape")
    ratio = torch.exp(current - old.detach())
    surrogate = torch.minimum(ratio * advantage,
                              ratio.clamp(1 - clip_ratio, 1 + clip_ratio) * advantage)
    delta = reference.detach() - current
    kl = torch.expm1(delta) - delta
    base_kl = torch.zeros_like(kl)
    if base is not None and base_kl_coefficient:
        base_delta = base.detach() - current
        base_kl = torch.expm1(base_delta) - base_delta
    if entropy is not None and entropy.shape != current.shape:
        raise ValueError("entropy must have one value per completion token")
    if entropy_coefficient and entropy is None:
        raise ValueError("entropy statistics are required when entropy_coefficient is nonzero")
    token_entropy = torch.zeros_like(kl) if entropy is None else entropy
    loss = (-policy_coefficient * surrogate + beta * kl + base_kl_coefficient * base_kl
            - entropy_coefficient * token_entropy).mean()
    metrics = {"loss": loss.detach(), "surrogate": surrogate.mean().detach(),
               "kl": kl.mean().detach(), "base_kl": base_kl.mean().detach(),
               "entropy": token_entropy.mean().detach(),
               "clip_fraction": ((ratio < 1 - clip_ratio) | (ratio > 1 + clip_ratio)).float().mean().detach(),
               "ratio_mean": ratio.mean().detach()}
    return loss, metrics


def trajectory_segments(prepared, scope: str):
    """Factor equal-group, equal-trajectory, output-token-mean losses.

    A shared ancestor is forwarded once but retains each descendant's distinct
    advantage and clipping operation. No averaging of advantages before clipping.
    Prompt tokens (including feedback) never become targets in these segments.
    """
    segments, memberships, indices, lengths = [], [], {}, []
    for group in prepared:
        for leaf, advantage in group:
            path = [*leaf.trajectory_prefix, leaf] if scope == "trajectory" else [leaf]
            length = sum(len(s.token_ids) for s in path)
            if not length:
                raise ValueError("cannot optimize a trajectory without output tokens")
            lengths.append(length)
            for segment in path:
                if not segment.token_ids:
                    continue
                key = id(segment)
                if key not in indices:
                    indices[key] = len(segments)
                    segments.append(segment)
                    memberships.append([])
                weight = len(segment.token_ids) / (length * len(prepared) * len(group))
                memberships[indices[key]].append((advantage, weight))
    return segments, memberships, lengths


class Backend:
    """One sequential trainer; vLLM server is an independently launched process.

    ``reset_instance`` must be called for each new problem. ``update`` accepts
    fresh sibling leaves with their recorded ancestor outputs. In trajectory
    mode ancestors are replayed under a frozen pre-update probability snapshot;
    they are not claimed to be fresh on-policy samples. All model output tokens
    enter the loss, while user/system/feedback tokens are conditioning only.
    """

    def __init__(self, config: BackendConfig):
        import torch
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
        from peft import LoraConfig, TaskType, get_peft_model

        self.config = config
        self.torch = torch
        self._version = 0
        self._instance_counter = 0
        self._run_id = uuid.uuid4().hex[:12]
        self._instance_id = ""
        self._generation_calls = 0
        self.optimizer_steps = 0
        self.reference_version = 0
        self._loaded_lora_name: str | None = None
        self._loaded_lora_path: Path | None = None
        self._serving_dirty = False
        self._adapter_root = Path(config.adapter_dir).resolve() / f"run-{self._run_id}"
        self._adapter_root.mkdir(parents=True, exist_ok=False)
        self._session = None
        if config.generation_backend == "vllm":
            import requests
            self._session = requests.Session()
            self._session.trust_env = False
        torch.manual_seed(config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.seed)
        dtype = getattr(torch, config.dtype)
        load_args = {"local_files_only": True, "trust_remote_code": config.trust_remote_code}
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_path, **load_args)
        model_config = AutoConfig.from_pretrained(config.model_path, **load_args)
        model_cls = AutoModelForCausalLM
        if model_config.model_type in {"qwen3_5", "qwen3_5_moe"}:
            from transformers import AutoModelForImageTextToText
            model_cls = AutoModelForImageTextToText
        base_model = model_cls.from_pretrained(
            config.model_path, torch_dtype=dtype, attn_implementation="sdpa", **load_args
        ).to(config.device)
        base_model.config.use_cache = False
        self._supports_logits_to_keep = "logits_to_keep" in inspect.signature(base_model.forward).parameters
        lora_config = LoraConfig(r=config.lora_rank, lora_alpha=config.lora_alpha,
                                 target_modules=config.target_modules, lora_dropout=config.lora_dropout,
                                 bias="none", task_type=TaskType.CAUSAL_LM)
        self.model = get_peft_model(base_model, lora_config)
        self.model.add_adapter("reference", lora_config)
        self._activate_policy()
        if config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            self.model.enable_input_require_grads()
        self._initial_adapter = self._adapter_state("default")
        self._initial_base_parameter_versions = {
            n: p._version for n, p in self.model.named_parameters() if ".lora_" not in n
        }
        self.reset_instance("initial")

    @property
    def version(self) -> int:
        return self._version

    def _adapter_state(self, name: str) -> dict:
        from peft import get_peft_model_state_dict
        return {k: v.detach().cpu().clone() for k, v in
                get_peft_model_state_dict(self.model, adapter_name=name).items()}

    def _load_adapter_state(self, state: dict, name: str):
        from peft import set_peft_model_state_dict
        set_peft_model_state_dict(self.model, state, adapter_name=name)

    def _activate_policy(self):
        self.model.set_adapter("default")
        for name, param in self.model.named_parameters():
            param.requires_grad_(".lora_" in name and ".default." in name)

    def _assert_base_frozen(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and not (".lora_" in name and ".default." in name):
                raise RuntimeError(f"non-policy parameter is trainable: {name}")
            if name in self._initial_base_parameter_versions:
                if param._version != self._initial_base_parameter_versions[name]:
                    raise RuntimeError(f"frozen base parameter was modified: {name}")

    def reset_instance(self, instance_id: str | None = None):
        """Restore zero-output LoRA initialization, reference and optimizer state."""
        self._unload_served_adapter()
        self._instance_counter += 1
        self._instance_id = f"{self._instance_counter}:{instance_id or 'problem'}"
        self._version = 0
        self._generation_calls = 0
        self.optimizer_steps = 0
        self.reference_version = 0
        self._load_adapter_state(self._initial_adapter, "default")
        self._load_adapter_state(self._initial_adapter, "reference")
        self._activate_policy()
        self.optimizer = self.torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=self.config.learning_rate, weight_decay=self.config.weight_decay
        )
        self.optimizer.zero_grad(set_to_none=True)
        self._serving_dirty = False
        self.torch.manual_seed(self.config.seed)
        if self.torch.cuda.is_available():
            self.torch.cuda.manual_seed_all(self.config.seed)
        self.model.eval()
        self._assert_base_frozen()

    def render_prompt(self, messages: list[dict[str, str]]) -> str:
        return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                                  enable_thinking=self.config.enable_thinking)

    def _post(self, endpoint: str, payload: dict) -> Any:
        response = self._session.post(self.config.vllm_url.rstrip("/") + endpoint,
                                      json=payload, timeout=self.config.request_timeout)
        if not response.ok:
            raise RuntimeError(f"vLLM {endpoint}: HTTP {response.status_code}: {response.text[:1500]}")
        if "application/json" in response.headers.get("content-type", ""):
            return response.json()
        return response.text

    def _remove_checkpoint(self, path: Path | None):
        if path is not None and not self.config.keep_adapter_checkpoints and path.exists():
            # Only checkpoints created inside this backend's unique run directory.
            path.resolve().relative_to(self._adapter_root)
            shutil.rmtree(path)

    def _unload_served_adapter(self):
        if self._loaded_lora_name is not None:
            self._post("/v1/unload_lora_adapter", {"lora_name": self._loaded_lora_name})
            self._loaded_lora_name = None
            self._remove_checkpoint(self._loaded_lora_path)
            self._loaded_lora_path = None

    def _publish_adapter(self):
        if self.config.generation_backend != "vllm":
            self._serving_dirty = False
            return
        name = f"pot-{self._run_id}-i{self._instance_counter}-v{self.version}"
        path = self._adapter_root / name
        self.model.save_pretrained(str(path), selected_adapters=["default"], safe_serialization=True)
        # Sequential requests allow a max_loras=1 server; never reuse a cached ID.
        self._unload_served_adapter()
        self._post("/v1/load_lora_adapter", {"lora_name": name, "lora_path": str(path)})
        self._loaded_lora_name, self._loaded_lora_path = name, path
        self._serving_dirty = False

    def _logprobs(self, sample: Sample, with_entropy: bool = False):
        torch = self.torch
        if not sample.prompt_ids or not sample.token_ids:
            raise ValueError("sample must contain both prompt and completion tokens")
        # Historical greedy outputs use a separate positive-temperature scoring
        # view in trajectory replay, preserving their generation provenance.
        if sample.temperature <= 0:
            raise ValueError("GRPO requires a stochastic positive-temperature sample")
        ids = torch.tensor([sample.prompt_ids + sample.token_ids], dtype=torch.long,
                           device=self.config.device)
        completion_len = len(sample.token_ids)
        kwargs = {"input_ids": ids, "attention_mask": torch.ones_like(ids), "use_cache": False}
        if self._supports_logits_to_keep:
            kwargs["logits_to_keep"] = completion_len + 1
        logits = self.model(**kwargs).logits
        if self._supports_logits_to_keep:
            action_logits = logits[0, :-1, :]
        else:
            start = len(sample.prompt_ids) - 1
            action_logits = logits[0, start:start + completion_len, :]
        targets = ids[0, -completion_len:]
        # Chunk the vocabulary normalization's fp32 temporaries. Autograd retains
        # the model's logits, but not a full sequence x vocabulary log_softmax.
        selected, entropies = [], []
        def token_statistics(chunk, labels):
            part = chunk.float() / sample.temperature
            normalizer = part.logsumexp(-1)
            chosen = part.gather(-1, labels[:, None]).squeeze(-1) - normalizer
            if with_entropy:
                probabilities = (part - normalizer[:, None]).exp()
                entropy = normalizer - (probabilities * part).sum(-1)
                return chosen, entropy
            return chosen
        for start in range(0, completion_len, 128):
            chunk = action_logits[start:start + 128]
            labels = targets[start:start + 128]
            if with_entropy and torch.is_grad_enabled():
                # Recompute fp32 vocabulary statistics in backward, keeping the
                # full-vocabulary entropy bonus from ballooning retained memory.
                from torch.utils.checkpoint import checkpoint
                chosen, entropy = checkpoint(token_statistics, chunk, labels, use_reentrant=False)
                entropies.append(entropy)
            elif with_entropy:
                chosen, entropy = token_statistics(chunk, labels)
                entropies.append(entropy)
            else:
                chosen = token_statistics(chunk, labels)
            selected.append(chosen)
        return (torch.cat(selected), torch.cat(entropies)) if with_entropy else torch.cat(selected)

    def generate(self, prompt: str, n: int = 3, max_tokens: int = 2048,
                 temperature: float | None = None) -> list[Sample]:
        temperature = self.config.temperature if temperature is None else temperature
        if n < 1 or max_tokens < 1 or temperature < 0:
            raise ValueError("invalid generation n, max_tokens or temperature")
        if self._serving_dirty:
            raise RuntimeError("adapter publication failed; generation is blocked until it is republished")
        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        if not prompt_ids:
            raise ValueError("empty prompt")
        if len(prompt_ids) + max_tokens > self.config.max_context_tokens:
            raise ValueError("prompt + output budget exceeds max_context_tokens; reduce history explicitly")
        self._activate_policy()
        self.model.eval()
        seed = self.config.seed + self._generation_calls
        self._generation_calls += 1
        if self.config.generation_backend == "vllm":
            payload = {"model": self._loaded_lora_name or self.config.vllm_model_name,
                       "prompt": prompt_ids, "n": n, "max_tokens": max_tokens,
                       "temperature": temperature, "top_p": 1.0, "top_k": -1,
                       "min_p": 0.0, "repetition_penalty": 1.0,
                       "presence_penalty": 0.0, "frequency_penalty": 0.0,
                       "seed": seed, "logprobs": 0, "return_tokens_as_token_ids": True,
                       "return_token_ids": True, "skip_special_tokens": True,
                       "add_special_tokens": False, "stream": False}
            result = self._post("/v1/completions", payload)
            echoed_prompt = result.get("prompt_token_ids")
            if echoed_prompt is not None and echoed_prompt != prompt_ids:
                raise RuntimeError("vLLM changed the prompt token IDs")
            choices = sorted(result["choices"], key=lambda choice: choice["index"])
            if len(choices) != n:
                raise RuntimeError("vLLM returned an unexpected number of choices")
            for choice in choices:
                echoed_prompt = choice.get("prompt_token_ids")
                if echoed_prompt is not None and echoed_prompt != prompt_ids:
                    raise RuntimeError("vLLM changed the prompt token IDs")
            samples = [Sample(choice["text"], completion_token_ids(choice), list(prompt_ids),
                              policy_version=self.version, temperature=temperature,
                              finish_reason=choice.get("finish_reason"), instance_id=self._instance_id)
                       for choice in choices]
        else:
            samples = self._generate_hf(prompt_ids, n, max_tokens, temperature, seed)
        # Score under the unchanged trainer policy before the caller can update.
        # HF and vLLM kernels may introduce small numerical differences; IDs,
        # temperature and all other sampling transforms are otherwise identical.
        if temperature > 0:
            with self.torch.no_grad():
                for sample in samples:
                    if sample.token_ids:
                        sample.old_logprobs = self._logprobs(sample).float().cpu().tolist()
        return samples

    def _generate_hf(self, prompt_ids, n, max_tokens, temperature, seed):
        from transformers import GenerationConfig
        torch = self.torch
        torch.manual_seed(seed)
        input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=self.config.device)
        eos_ids = self.model.generation_config.eos_token_id
        if eos_ids is None:
            eos_ids = self.tokenizer.eos_token_id
        eos_set = set(eos_ids if isinstance(eos_ids, list) else [eos_ids])
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = eos_ids[0] if isinstance(eos_ids, list) else eos_ids
        generation_config = GenerationConfig(
            max_new_tokens=max_tokens, do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else 1.0,
            top_p=1.0, top_k=0, repetition_penalty=1.0,
            eos_token_id=eos_ids, pad_token_id=pad_id, use_cache=True,
        )
        samples = []
        with torch.no_grad():
            for _ in range(n):
                output = self.model.generate(input_ids=input_ids, attention_mask=torch.ones_like(input_ids),
                                             generation_config=generation_config)
                tokens = output[0, len(prompt_ids):].cpu().tolist()
                # Each call is unpadded; preserve EOS itself as a sampled action.
                finish = "stop" if tokens and tokens[-1] in eos_set else "length"
                samples.append(Sample(self.tokenizer.decode(tokens, skip_special_tokens=True),
                                      tokens, list(prompt_ids), policy_version=self.version,
                                      temperature=temperature, finish_reason=finish,
                                      instance_id=self._instance_id))
        return samples

    def _score_fixed_policy(self, samples: list[Sample], adapter: str | None):
        torch = self.torch
        self.model.eval()
        if adapter is not None:
            self.model.set_adapter(adapter)
        try:
            with torch.no_grad():
                if adapter is None:
                    with self.model.disable_adapter():
                        return [self._logprobs(sample).float().cpu() for sample in samples]
                return [self._logprobs(sample).float().cpu() for sample in samples]
        finally:
            self._activate_policy()

    def update(self, groups: list[list[tuple[Sample, float]]]) -> dict[str, Any]:
        if self._serving_dirty:
            raise RuntimeError("previous update failed after modifying the policy; reset the instance before continuing")
        started = time.monotonic()
        metrics: dict[str, Any] = {"version_before": self.version, "version_after": self.version,
            "groups_total": len(groups), "groups_used": 0, "groups_zero_variance": 0,
            "samples": 0, "completion_tokens": 0, "optimizer_steps": 0, "epoch_metrics": [],
            "loss_variant": self.config.loss_variant, **self.config.loss_coefficients(),
            "loss_scope": self.config.loss_scope,
            "old_policy_scope": "frozen_pre_update_snapshot" if self.config.loss_scope == "trajectory" else "leaf_generation",
            "policy_distribution": "softmax(logits / sample_temperature); no top-k/top-p truncation"}
        prepared = []
        seen_samples = set()
        for group in groups:
            if not group:
                continue
            advantages = group_advantages([float(r) for _, r in group],
                                          self.config.advantage_epsilon, self.config.advantage_clip)
            for sample, _ in group:
                if sample.policy_version != self.version or sample.instance_id != self._instance_id:
                    raise ValueError("stale policy or cross-instance GRPO sample")
                if not sample.prompt_ids or not sample.token_ids:
                    raise ValueError("cannot optimize an empty sample")
                if sample.temperature <= 0 or len(sample.old_logprobs) != len(sample.token_ids):
                    raise ValueError("sample lacks stochastic old-policy log probabilities")
                if any(not math.isfinite(x) for x in sample.old_logprobs):
                    raise ValueError("nonfinite old policy log probability")
                if sample.prompt_ids != group[0][0].prompt_ids:
                    raise ValueError("each GRPO group must share one parent prompt")
                if self.config.loss_scope == "trajectory":
                    if sample.temperature != self.config.temperature:
                        raise ValueError("trajectory leaves must use the configured scoring temperature")
                    prefix = sample.trajectory_prefix
                    if [id(s) for s in prefix] != [id(s) for s in group[0][0].trajectory_prefix]:
                        raise ValueError("siblings must share the same trajectory prefix")
                    path_ids = [id(s) for s in prefix] + [id(sample)]
                    if len(set(path_ids)) != len(path_ids):
                        raise ValueError("cyclic or duplicate trajectory segment")
                    for index, ancestor in enumerate(prefix):
                        if ancestor.instance_id != self._instance_id:
                            raise ValueError("cross-instance trajectory ancestor")
                        if not 0 <= ancestor.policy_version <= sample.policy_version:
                            raise ValueError("future or invalid ancestor policy version")
                        if [id(s) for s in ancestor.trajectory_prefix] != path_ids[:index]:
                            raise ValueError("inconsistent ancestor chain")
                        if not ancestor.prompt_ids or not math.isfinite(ancestor.temperature) or ancestor.temperature < 0:
                            raise ValueError("invalid ancestor generation record")
                if id(sample) in seen_samples:
                    raise ValueError("a sample may appear only once in an update")
                seen_samples.add(id(sample))
            if not any(advantages):
                metrics["groups_zero_variance"] += 1
                continue
            prepared.append([(sample, advantage) for (sample, _), advantage in zip(group, advantages)])
        if not prepared:
            metrics.update(skipped=True, reason="no_nonconstant_sibling_groups", seconds=time.monotonic()-started)
            return metrics
        leaves = [sample for group in prepared for sample, _ in group]
        segments, memberships, lengths = trajectory_segments(prepared, self.config.loss_scope)
        # Score original generated IDs at their original conditioning contexts.
        # Do not retokenize or duplicate response suffixes. Historical greedy
        # outputs are replay targets, not samples drawn at temperature 0.7.
        samples = [replace(s, temperature=self.config.temperature) for s in segments] if self.config.loss_scope == "trajectory" else segments
        leaf_ids = {id(s) for s in leaves}
        historical = [i for i, s in enumerate(segments) if id(s) not in leaf_ids]
        terminal_tokens = sum(len(s.token_ids) for s in leaves)
        metrics.update(groups_used=len(prepared), samples=len(leaves),
                       completion_tokens=terminal_tokens, skipped=False,
                       trajectory_output_tokens=sum(lengths), trajectory_lengths=lengths,
                       history_output_tokens=sum(lengths)-terminal_tokens,
                       unique_output_tokens=sum(len(s.token_ids) for s in segments),
                       unique_segments=len(segments), historical_segments=len(historical),
                       historical_policy_versions=sorted({segments[i].policy_version for i in historical}),
                       historical_greedy_segments=sum(segments[i].temperature == 0 for i in historical),
                       scoring_temperature=self.config.temperature)
        # Current leaves already carry pre-update scores. Replay ancestors may
        # come from older policies or greedy commitments, so freeze their scores
        # under the current policy BEFORE any of this update's optimizer steps.
        replay_old = self._score_fixed_policy([samples[i] for i in historical], "default") if historical else []
        replay_by_index = dict(zip(historical, replay_old))
        reference = self._score_fixed_policy(samples, "reference")
        base = (self._score_fixed_policy(samples, None) if self.config.base_kl_coefficient else None)
        torch = self.torch
        coefficients = self.config.loss_coefficients()
        old = [replay_by_index[i].to(self.config.device) if i in replay_by_index
               else torch.tensor(s.old_logprobs, device=self.config.device) for i, s in enumerate(samples)]
        trainable = [p for p in self.model.parameters() if p.requires_grad]
        self._assert_base_frozen()
        for epoch in range(self.config.epochs):
            self.model.train()
            # Base model dropout is usually zero; force deterministic policy
            # likelihoods even for an otherwise compatible model with dropout.
            if self.config.disable_policy_dropout:
                for module in self.model.modules():
                    if isinstance(module, torch.nn.Dropout):
                        module.eval()
            self.optimizer.zero_grad(set_to_none=True)
            totals: dict[str, float] = {}
            for index, sample in enumerate(samples):
                entropy = None
                if coefficients["entropy_coefficient"]:
                    current, entropy = self._logprobs(sample, with_entropy=True)
                else:
                    current = self._logprobs(sample)
                weighted_loss = 0
                for advantage, weight in memberships[index]:
                    loss, stats = grpo_loss(current, old[index], reference[index].to(self.config.device),
                        advantage, self.config.clip_ratio, coefficients["kl_coefficient"],
                        base[index].to(self.config.device) if base is not None else None,
                        coefficients["base_kl_coefficient"], policy_coefficient=coefficients["policy_coefficient"],
                        entropy=entropy, entropy_coefficient=coefficients["entropy_coefficient"])
                    weighted_loss = weighted_loss + loss * weight
                    for key, value in stats.items():
                        totals[key] = totals.get(key, 0.0) + float(value.cpu()) * weight
                if not torch.isfinite(weighted_loss).item():
                    self.optimizer.zero_grad(set_to_none=True)
                    raise FloatingPointError("nonfinite GRPO loss; no optimizer step taken for this epoch")
                weighted_loss.backward()
                del current, entropy, loss, stats, weighted_loss
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable, self.config.max_grad_norm,
                                                       error_if_nonfinite=True)
            # Any failure after this point must block stale server generation.
            self._serving_dirty = True
            self.optimizer.step()
            self.optimizer_steps += 1
            metrics["optimizer_steps"] += 1
            totals.update(epoch=epoch + 1, grad_norm=float(grad_norm.cpu()),
                          global_optimizer_step=self.optimizer_steps)
            metrics["epoch_metrics"].append(totals)
            if self.config.reference_sync_steps and self.optimizer_steps % self.config.reference_sync_steps == 0:
                self._load_adapter_state(self._adapter_state("default"), "reference")
                self.reference_version = self.optimizer_steps
                # Subsequent epochs must see the newly synchronized reference.
                if epoch + 1 < self.config.epochs:
                    reference = self._score_fixed_policy(samples, "reference")
        self.optimizer.zero_grad(set_to_none=True)
        self.model.eval()
        self._version += 1
        self._assert_base_frozen()
        self._serving_dirty = True
        self._publish_adapter()
        metrics.update(version_after=self.version, reference_optimizer_step=self.reference_version,
                       total_optimizer_steps=self.optimizer_steps, seconds=time.monotonic()-started)
        return metrics

    def describe(self) -> dict[str, Any]:
        return {"config": asdict(self.config), "trainable_parameters": sum(
            p.numel() for p in self.model.parameters() if p.requires_grad),
            "base_parameters_frozen": True, "version": self.version,
            "instance_id": self._instance_id, "adapter_name": self._loaded_lora_name,
            "reference_optimizer_step": self.reference_version,
            "effective_loss_coefficients": self.config.loss_coefficients(),
            "loss_scope": self.config.loss_scope,
            "trajectory_likelihood": "all recorded model outputs; original token contexts; user/system/feedback excluded",
            "history_probability_rule": "pre-update snapshot at configured temperature; historical outputs are replay, including greedy ancestors",
            "source_repairs": ["next-token alignment; no asymmetric logits clamping",
                "positive k3 KL replaces signed ref-current pseudo-KL; configured reference synchronization applied",
                "independent reference adapter; full optimizer/adapter reset per instance",
                "token means and equal group weights; population reward std; advantage cap=10",
                "temperature-adjusted probabilities; dropout disabled by default during likelihood optimization",
                "policy_scaled: beta scales policy; separate KL and entropy coefficients; no dual KL",
                "vLLM/HF kernel numerical differences remain"]}

    def close(self):
        self._unload_served_adapter()
        if self._session is not None:
            self._session.close()
