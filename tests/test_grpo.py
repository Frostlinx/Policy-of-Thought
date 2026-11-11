"""GRPO invariants, token fidelity, and optional differentiable loss checks."""
import math

import pytest

from pot.backend import Backend, BackendConfig, Sample, completion_token_ids, group_advantages, grpo_loss, trajectory_segments


def test_group_advantages_normalize_siblings_and_skip_constant_groups():
    advantages = group_advantages([0, 0.25, 0.5])
    assert sum(advantages) == pytest.approx(0)
    assert sum(x * x for x in advantages) / 3 == pytest.approx(1)
    assert advantages[0] < 0 < advantages[2]
    assert group_advantages([0, 0, 0]) == [0, 0, 0]
    assert group_advantages([1]) == [0]
    assert max(group_advantages([0, 0, 1], clip=0.5)) == 0.5


def test_invalid_rewards_rejected():
    for rewards in ([], [0, float("nan")], [float("inf")]):
        with pytest.raises(ValueError):
            group_advantages(rewards)


def test_completion_ids_preserve_ids_and_eos_without_retokenization():
    assert completion_token_ids({"token_ids": [11, 12, 151645]}) == [11, 12, 151645]
    assert completion_token_ids({"logprobs": {"tokens": ["token_id:11", "token_id:151645"]}}) == [11, 151645]
    with pytest.raises(RuntimeError):
        completion_token_ids({"text": "hello", "logprobs": {"tokens": ["hello"]}})
    with pytest.raises(ValueError):
        completion_token_ids({"token_ids": [-1]})


def test_length_normalized_prior_and_config_validation():
    assert Sample("x", old_logprobs=[-1, -3]).prior == pytest.approx(math.exp(-2))
    with pytest.raises(ValueError):
        BackendConfig("local", temperature=0)
    with pytest.raises(ValueError):
        BackendConfig("local", clip_ratio=2)


def test_optional_and_paper_loss_variants_are_explicit():
    default = BackendConfig("local")
    assert default.loss_variant == "paper"
    assert default.loss_coefficients() == {"policy_coefficient": 1.0, "kl_coefficient": 0.02,
                                          "base_kl_coefficient": 0.0, "entropy_coefficient": 0.0}
    assert BackendConfig("local", base_kl_coefficient=0.005).loss_coefficients()["base_kl_coefficient"] == 0.005
    source = BackendConfig("local", loss_variant="policy_scaled")
    assert source.loss_coefficients() == {"policy_coefficient": 0.02, "kl_coefficient": 0.005,
                                         "base_kl_coefficient": 0.0, "entropy_coefficient": 0.01}
    legacy = BackendConfig("local", loss_variant="dual_kl")
    assert legacy.loss_coefficients() == {"policy_coefficient": 1.0, "kl_coefficient": 0.02,
                                        "base_kl_coefficient": 0.005, "entropy_coefficient": 0.0}
    assert set(source.target_modules) == {"q_proj", "k_proj", "v_proj", "o_proj"}
    with pytest.raises(ValueError, match="no second base KL"):
        BackendConfig("local", loss_variant="policy_scaled", base_kl_coefficient=0.005)


def test_render_prompt_passes_explicit_thinking_mode():
    backend = Backend.__new__(Backend)
    backend.config = BackendConfig("local")
    captured = {}

    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            captured.update(kwargs)
            return "rendered-prompt"

    backend.tokenizer = Tokenizer()
    assert backend.render_prompt([{"role": "user", "content": "test"}]) == "rendered-prompt"
    assert captured == {"tokenize": False, "add_generation_prompt": True, "enable_thinking": False}
    backend.config.enable_thinking = True
    backend.render_prompt([])
    assert captured["enable_thinking"] is True


def test_clipped_surrogate_signs_and_reference_gradient_isolation():
    torch = pytest.importorskip("torch")
    current = torch.tensor([math.log(1.5), math.log(0.5)], requires_grad=True)
    old = torch.zeros(2, requires_grad=True)
    reference = current.detach().clone().requires_grad_(True)
    loss, stats = grpo_loss(current, old, reference, advantage=1.0, beta=0)
    assert loss.item() == pytest.approx(-0.9)
    loss.backward()
    assert current.grad.tolist() == pytest.approx([0, -0.25])
    assert old.grad is None and reference.grad is None
    assert stats["clip_fraction"].item() == 1
    loss_negative, _ = grpo_loss(current.detach(), old, reference, advantage=-1.0, beta=0)
    assert loss_negative.item() == pytest.approx(1.1)


def test_k3_kl_is_zero_at_equality_and_positive_elsewhere():
    torch = pytest.importorskip("torch")
    current = torch.tensor([-0.2, -0.4], requires_grad=True)
    _, equal = grpo_loss(current, current, current, advantage=0, beta=0.02)
    assert equal["kl"].item() == 0
    ref = torch.tensor([-0.3, -0.1])
    loss, different = grpo_loss(current, current, ref, advantage=0, beta=0.02,
                               base=ref, base_kl_coefficient=0.005)
    assert different["kl"].item() > 0
    assert loss.item() == pytest.approx(0.025 * different["kl"].item())
    loss.backward()
    assert torch.isfinite(current.grad).all()


def test_source_policy_scale_and_entropy_bonus_have_correct_sign():
    torch = pytest.importorskip("torch")
    current = torch.tensor([-1.0, -1.0], requires_grad=True)
    entropy = torch.tensor([2.0, 4.0], requires_grad=True)
    loss, _ = grpo_loss(current, current.detach(), current.detach(), advantage=1,
                        beta=0.005, base_kl_coefficient=0, policy_coefficient=0.02,
                        entropy=entropy, entropy_coefficient=0.01)
    assert loss.item() == pytest.approx(-0.02 - 0.01 * 3)
    loss.backward()
    assert entropy.grad.tolist() == pytest.approx([-0.005, -0.005])


def test_loss_is_token_mean_not_sum():
    torch = pytest.importorskip("torch")
    current = torch.tensor([-1.2, -0.8])
    old = torch.tensor([-1.0, -1.0])
    reference = torch.tensor([-1.1, -0.9])
    short, _ = grpo_loss(current, old, reference, advantage=0.7)
    long, _ = grpo_loss(current.repeat(4), old.repeat(4), reference.repeat(4), advantage=0.7)
    assert short.item() == pytest.approx(long.item())


@pytest.fixture
def tiny_backend(monkeypatch, request):
    """Offline random Qwen fixture; exercises real PEFT/transformers on CPU."""
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    pytest.importorskip("peft")
    tmp_path = request.getfixturevalue("tmp_path")

    class TinyTokenizer:
        eos_token_id = 2
        pad_token_id = 0

        def encode(self, prompt, add_special_tokens=False):
            return [1, 4, 7]

        def decode(self, tokens, skip_special_tokens=True):
            return " ".join(map(str, tokens))

    torch.manual_seed(19)
    model_config = transformers.Qwen2Config(vocab_size=32, hidden_size=16,
        intermediate_size=32, num_hidden_layers=1, num_attention_heads=2,
        num_key_value_heads=2, max_position_embeddings=64,
        bos_token_id=1, eos_token_id=2, pad_token_id=0)
    model = transformers.Qwen2ForCausalLM(model_config)
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *a, **k: TinyTokenizer())
    monkeypatch.setattr(transformers.AutoConfig, "from_pretrained", lambda *a, **k: model_config)
    monkeypatch.setattr(transformers.AutoModelForCausalLM, "from_pretrained", lambda *a, **k: model)
    backend = Backend(BackendConfig("offline-test-model", generation_backend="hf", device="cpu",
        dtype="float32", adapter_dir=str(tmp_path / "adapters"), learning_rate=0.01,
        lora_rank=2, lora_alpha=4, target_modules=["q_proj", "v_proj"],
        gradient_checkpointing=False, max_context_tokens=64, reference_sync_steps=2))
    yield backend
    backend.close()


def _tiny_siblings(backend):
    samples = [Sample(str(i), token_ids=[8 + i, 12 + i, 2], prompt_ids=[1, 4, 7],
                      instance_id=backend._instance_id, policy_version=backend.version)
               for i in range(3)]
    with backend.torch.no_grad():
        for sample in samples:
            sample.old_logprobs = backend._logprobs(sample).tolist()
    return samples


def test_real_lora_updates_only_adapter_then_resets_without_leakage(tiny_backend):
    backend = tiny_backend
    torch = backend.torch
    frozen_before = {n: p.detach().clone() for n, p in backend.model.named_parameters()
                     if ".lora_" not in n}
    samples = _tiny_siblings(backend)
    result = backend.update([list(zip(samples, [0, 0.5, 1]))])
    assert result["optimizer_steps"] == 3
    assert result["reference_optimizer_step"] == 2
    assert backend.version == 1
    assert backend.optimizer.state
    state = backend._adapter_state("default")
    assert any(not torch.equal(value, backend._initial_adapter[name]) for name, value in state.items())
    for name, param in backend.model.named_parameters():
        if name in frozen_before:
            assert torch.equal(param, frozen_before[name]), name
    with pytest.raises(ValueError, match="stale"):
        backend.update([list(zip(samples, [0, 0.5, 1]))])
    backend.reset_instance("second-problem")
    assert backend.version == 0
    assert not backend.optimizer.state
    for name, value in backend._adapter_state("default").items():
        assert torch.equal(value, backend._initial_adapter[name])
    with pytest.raises(ValueError, match="cross-instance"):
        backend.update([list(zip(samples, [0, 0.5, 1]))])


def test_zero_variance_update_does_not_step_optimizer(tiny_backend):
    samples = _tiny_siblings(tiny_backend)
    result = tiny_backend.update([list(zip(samples, [0, 0, 0]))])
    assert result["skipped"]
    assert result["groups_zero_variance"] == 1
    assert tiny_backend.version == 0
    assert not tiny_backend.optimizer.state


def test_completion_logprobs_mask_prompt_and_align_next_token(tiny_backend):
    backend = tiny_backend
    torch = backend.torch
    sample = _tiny_siblings(backend)[0]
    ids = torch.tensor([sample.prompt_ids + sample.token_ids])
    with torch.no_grad():
        full = backend.model(input_ids=ids, use_cache=False).logits[0].float() / sample.temperature
        expected = full[len(sample.prompt_ids) - 1:-1].log_softmax(-1).gather(
            1, torch.tensor(sample.token_ids)[:, None]).squeeze(1)
        actual = backend._logprobs(sample)
    assert len(actual) == len(sample.token_ids)
    assert torch.allclose(actual, expected, atol=1e-6)
    with torch.no_grad():
        chosen, entropy = backend._logprobs(sample, with_entropy=True)
        completion_logits = full[len(sample.prompt_ids) - 1:-1]
        expected_entropy = -(completion_logits.softmax(-1) * completion_logits.log_softmax(-1)).sum(-1)
    assert torch.allclose(chosen, expected, atol=1e-6)
    assert torch.allclose(entropy, expected_entropy, atol=1e-6)


def test_partial_optimizer_failure_blocks_generation(tiny_backend, monkeypatch):
    samples = _tiny_siblings(tiny_backend)

    def broken_step():
        raise RuntimeError("simulated optimizer failure")

    monkeypatch.setattr(tiny_backend.optimizer, "step", broken_step)
    with pytest.raises(RuntimeError, match="simulated"):
        tiny_backend.update([list(zip(samples, [0, 0.5, 1]))])
    with pytest.raises(RuntimeError, match="publication failed"):
        tiny_backend.generate("test", max_tokens=3)
    tiny_backend.reset_instance("recovery")
    assert not tiny_backend._serving_dirty


def test_shared_history_factorization_matches_whole_path_loss_and_gradients():
    torch = pytest.importorskip('torch')
    ancestor = Sample('history', token_ids=[1, 2])
    leaves = [Sample('short', token_ids=[3], trajectory_prefix=[ancestor]),
              Sample('long', token_ids=[4, 5, 6], trajectory_prefix=[ancestor])]
    prepared = [[(leaves[0], -1.0), (leaves[1], 1.0)]]
    segments, memberships, lengths = trajectory_segments(prepared, 'trajectory')
    assert lengths == [3, 5] and segments == [ancestor, *leaves]
    current = [torch.tensor([.5, -.5], requires_grad=True),
               torch.tensor([.2], requires_grad=True),
               torch.tensor([-.4, .4, .1], requires_grad=True)]
    old = [torch.zeros_like(x, requires_grad=True) for x in current]
    reference = [torch.full_like(x, -.1, requires_grad=True) for x in current]
    factored = sum(grpo_loss(current[i], old[i], reference[i], advantage=a)[0] * w
                   for i, members in enumerate(memberships) for a, w in members)
    direct = sum(grpo_loss(torch.cat([current[0], current[i]]),
                          torch.cat([old[0], old[i]]), torch.cat([reference[0], reference[i]]),
                          advantage=a)[0] / 2 for i, a in [(1, -1.), (2, 1.)])
    assert torch.allclose(factored, direct)
    factored_grads = torch.autograd.grad(factored, current, retain_graph=True)
    direct_grads = torch.autograd.grad(direct, current)
    assert all(torch.allclose(a, b) for a, b in zip(factored_grads, direct_grads))
    assert factored_grads[0].abs().sum() > 0  # Ancestor output is a target, not context only.
    assert all(x.grad is None for x in old + reference)
    assert sum(w for members in memberships for _, w in members) == pytest.approx(1)


def test_segment_scoring_matches_full_transcript_output_mask(tiny_backend):
    backend, torch = tiny_backend, tiny_backend.torch
    ancestor = Sample('history', prompt_ids=[1, 4, 7], token_ids=[8, 12, 2])
    # 28 and 29 represent a public feedback message: conditioning, never targets.
    leaf = Sample('new', prompt_ids=ancestor.prompt_ids + ancestor.token_ids + [28, 29],
                  token_ids=[10, 15, 2], trajectory_prefix=[ancestor])
    full_ids = leaf.prompt_ids + leaf.token_ids
    with torch.no_grad():
        logits = backend.model(input_ids=torch.tensor([full_ids]), use_cache=False).logits[0].float()
        positions = [2, 3, 4, 7, 8, 9]
        labels = ancestor.token_ids + leaf.token_ids
        expected = (logits[positions] / .7).log_softmax(-1).gather(
            1, torch.tensor(labels)[:, None]).squeeze(1)
        actual = torch.cat([backend._logprobs(ancestor), backend._logprobs(leaf)])
    assert len(actual) == 6  # Excludes all initial prompt and feedback tokens.
    assert torch.allclose(actual, expected, atol=1e-6)


def test_real_trajectory_update_replays_old_and_greedy_history_with_frozen_old_scores(tiny_backend, monkeypatch):
    import pot.backend as implementation
    backend, torch = tiny_backend, tiny_backend.torch
    ancestor = _tiny_siblings(backend)[0]
    backend.update([list(zip(_tiny_siblings(backend), [0, .5, 1]))])
    greedy = Sample('greedy history', prompt_ids=[1, 4, 7, 8, 12, 2, 28], token_ids=[13, 2],
                    temperature=0, instance_id=backend._instance_id, policy_version=backend.version,
                    trajectory_prefix=[ancestor])
    leaves = _tiny_siblings(backend)
    for i, leaf in enumerate(leaves):
        leaf.prompt_ids = greedy.prompt_ids + greedy.token_ids + [29]
        leaf.token_ids = leaf.token_ids[:i+1]  # Unequal path lengths exercise token weights.
        leaf.trajectory_prefix = [ancestor, greedy]
        with torch.no_grad():
            leaf.old_logprobs = backend._logprobs(leaf).tolist()
    ancestor.old_logprobs = [-100.] * len(ancestor.token_ids)  # Must not become the replay denominator.
    before_ids = list(ancestor.token_ids), list(greedy.token_ids)
    recorded, scored_history, snapshots = [], [], {}
    original_logprobs, original_fixed, original_loss = backend._logprobs, backend._score_fixed_policy, implementation.grpo_loss

    def observe_logprobs(sample, with_entropy=False):
        result = original_logprobs(sample, with_entropy)
        if torch.is_grad_enabled():
            result.retain_grad()
            recorded.append((sample.text, result))
        return result

    def observe_fixed(samples, adapter):
        if adapter == 'default':
            scored_history.append([s.text for s in samples])
        return original_fixed(samples, adapter)

    def observe_loss(current, old, reference, *args, **kwargs):
        assert not old.requires_grad and not reference.requires_grad
        key = old.data_ptr()
        if key not in snapshots:
            assert torch.allclose(current.detach(), old, atol=1e-6)
            snapshots[key] = old.detach().clone()
        assert torch.equal(old, snapshots[key])
        return original_loss(current, old, reference, *args, **kwargs)

    monkeypatch.setattr(backend, '_logprobs', observe_logprobs)
    monkeypatch.setattr(backend, '_score_fixed_policy', observe_fixed)
    monkeypatch.setattr(implementation, 'grpo_loss', observe_loss)
    result = backend.update([list(zip(leaves, [0, .5, 1]))])
    assert result['loss_scope'] == 'trajectory'
    assert result['history_output_tokens'] == 15 and result['completion_tokens'] == 6
    assert result['trajectory_output_tokens'] == 21 and result['unique_output_tokens'] == 11
    assert result['historical_greedy_segments'] == 1 and result['historical_policy_versions'] == [0, 1]
    assert len(recorded) == 5 * backend.config.epochs  # Shared ancestors forward once each epoch.
    assert scored_history == [[ancestor.text, greedy.text]] and len(snapshots) == 5
    assert any(t == 'greedy history' and r.grad.abs().sum() > 0 for t, r in recorded)
    assert greedy.temperature == 0 and greedy.old_logprobs == []
    assert before_ids == (ancestor.token_ids, greedy.token_ids)


@pytest.mark.parametrize('defect,pattern', [
    ('instance', 'cross-instance'), ('future', 'future'), ('cycle', 'cyclic'),
    ('siblings', 'same trajectory prefix'), ('chain', 'inconsistent ancestor'),
])
def test_trajectory_rejects_invalid_history_before_update(tiny_backend, defect, pattern):
    ancestor = _tiny_siblings(tiny_backend)[0]
    leaves = _tiny_siblings(tiny_backend)
    for leaf in leaves:
        leaf.trajectory_prefix = [ancestor]
    if defect == 'instance':
        ancestor.instance_id = 'another-question'
    elif defect == 'future':
        ancestor.policy_version += 1
    elif defect == 'cycle':
        leaves[0].trajectory_prefix = [leaves[0]]
    elif defect == 'siblings':
        leaves[1].trajectory_prefix = []
    else:
        ancestor.trajectory_prefix = [Sample('wrong ancestor')]
    with pytest.raises(ValueError, match=pattern):
        tiny_backend.update([list(zip(leaves, [0, .5, 1]))])
    assert tiny_backend.version == 0 and not tiny_backend.optimizer.state


def test_completion_ablation_excludes_history_explicitly():
    history = Sample('history', token_ids=[1, 2])
    leaf = Sample('leaf', token_ids=[3], trajectory_prefix=[history])
    segments, memberships, lengths = trajectory_segments([[(leaf, 1)]], 'completion')
    assert segments == [leaf] and memberships == [[(1, 1.)]] and lengths == [1]
    assert BackendConfig('local').loss_scope == 'trajectory'
    with pytest.raises(ValueError, match='loss_scope'):
        BackendConfig('local', loss_scope='unknown')

