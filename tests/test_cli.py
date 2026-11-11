"""Public CLI flags, precedence, paths and labeled reward-mode integration."""
import argparse
from dataclasses import asdict
import importlib
import json
from pathlib import Path
import shutil
from types import SimpleNamespace
import uuid

import pytest

from pot.backend import BackendConfig, Sample
from pot.search import SearchConfig
from pot.evaluator import CandidateEvaluation, PublicEvaluation, PrivateScore

runner = importlib.import_module("pot.run")
ROOT = Path(__file__).resolve().parents[1] / "work" / "test-cli"
SECRET = "PRIVATE_CLI_SENTINEL_NOT_FOR_PROMPTS_42786"


@pytest.fixture
def workspace(monkeypatch):
    ROOT.mkdir(parents=True, exist_ok=True)
    path = ROOT / uuid.uuid4().hex
    path.mkdir()
    monkeypatch.chdir(path)
    try:
        yield path
    finally:
        # Leave cwd before removal (required on Windows).
        monkeypatch.chdir(ROOT)
        path.resolve().relative_to(ROOT.resolve())
        shutil.rmtree(path)


def parse(arguments):
    return runner.parameters_from_args(runner.build_parser().parse_args(arguments))


def test_help_documents_primary_cli_without_user_paths():
    help_text = runner.build_parser().format_help()
    for flag in ("--model", "--data", "--output", "--simulations", "--width", "--max-depth",
        "--max-new-tokens", "--temperature", "--c-puct", "--learning-rate", "--epochs",
        "--clip-ratio", "--beta", "--base-kl-coefficient", "--kl-coefficient", "--entropy-coefficient",
        "--loss-variant", "--lora-rank", "--lora-alpha", "--lora-dropout", "--target-modules",
        "--reference-sync-steps", "--seed", "--device", "--dtype", "--backend", "--vllm-url",
        "--vllm-name", "--max-context-tokens", "--executor", "--timeout", "--memory-mb", "--cpus",
        "--no-greedy-commitment", "--no-train", "--enable-thinking", "--feedback"):
        assert flag in help_text
    assert "--config" in help_text and "Optional legacy JSON" in help_text
    assert "C:\\Users" not in help_text and "/root/" not in help_text


def test_config_free_defaults_and_cli_paths_resolve_against_cwd(workspace):
    parameters = parse(["--model", "models/local", "--data", "data/v6", "--output", "results"])
    assert parameters["backend"]["model_path"] == str(workspace / "models" / "local")
    assert parameters["dataset"]["path"] == str(workspace / "data" / "v6")
    assert parameters["backend"]["loss_variant"] == "paper"
    assert parameters["evaluation"] == {"executor": "subprocess", "memory_mb": 1024, "timeout_seconds": 10.0}
    assert parameters["search"] == {"feedback": "public"}
    backend = BackendConfig(**parameters["backend"])
    search = SearchConfig(**parameters["search"])
    assert backend.enable_thinking is False and search.train is True


def test_flags_route_to_correct_parameters_including_false_and_zero(workspace):
    parameters = parse(["--model", "model", "--data", "dataset", "--output", "results",
        "--backend", "hf", "--device", "cpu", "--dtype", "float32", "--vllm-url", "http://localhost:9011",
        "--vllm-name", "solver", "--max-context-tokens", "4096", "--simulations", "7", "--width", "2",
        "--max-depth", "4", "--max-new-tokens", "512", "--temperature", "0.4", "--c-puct", "0",
        "--learning-rate", "0.0002", "--epochs", "2", "--clip-ratio", "0.2", "--beta", "0.03",
        "--loss-variant", "policy_scaled", "--base-kl-coefficient", "0", "--kl-coefficient", "0.006",
        "--entropy-coefficient", "0", "--lora-rank", "4", "--lora-alpha", "8", "--lora-dropout", "0",
        "--target-modules", "q_proj", "v_proj", "--reference-sync-steps", "0", "--seed", "17",
        "--executor", "docker", "--timeout", "2", "--memory-mb", "768", "--cpus", "0.5",
        "--no-train", "--no-greedy-commitment", "--enable-thinking",
        "--feedback", "none", "--offset", "0", "--limit", "0"])
    backend, search, evaluation = parameters["backend"], parameters["search"], parameters["evaluation"]
    assert backend["generation_backend"] == "hf" and backend["device"] == "cpu"
    assert backend["temperature"] == search["temperature"] == 0.4
    assert backend["target_modules"] == ["q_proj", "v_proj"]
    assert backend["reference_sync_steps"] == 0 and backend["entropy_coefficient"] == 0
    assert backend["enable_thinking"] is True
    assert search["train"] is False and search["greedy_commitment"] is False
    assert search["feedback"] == "none"
    assert evaluation == {"executor": "docker", "memory_mb": 768, "timeout_seconds": 2.0, "cpus": 0.5}
    assert parameters["run"] == {"seed": 17, "offset": 0, "limit": 0}
    BackendConfig(**backend)
    SearchConfig(**search)


def test_flags_override_json_but_unset_values_and_json_path_origin_survive(workspace):
    config_dir = workspace / "configs"
    config_dir.mkdir()
    config = config_dir / "example.json"
    config.write_text(json.dumps({"backend": {"model_path": "old-model", "beta": 0.09,
        "learning_rate": 1e-5, "adapter_dir": "adapters"},
        "dataset": {"path": "data.jsonl"}, "search": {"train": True, "width": 7},
        "evaluation": {"executor": "docker", "work_dir": "scratch"}, "run": {"seed": 3}}))
    parameters = parse(["--config", str(config), "--model", "new-model", "--output", "results",
                        "--learning-rate", "0.002", "--no-train", "--temperature", "0.8"])
    assert parameters["backend"]["model_path"] == str(workspace / "new-model")
    assert parameters["backend"]["adapter_dir"] == str(config_dir / "adapters")
    assert parameters["dataset"]["path"] == str(config_dir / "data.jsonl")
    assert parameters["evaluation"]["work_dir"] == str(config_dir / "scratch")
    assert parameters["backend"]["beta"] == 0.09 and parameters["backend"]["learning_rate"] == 0.002
    assert parameters["search"]["width"] == 7 and parameters["search"]["train"] is False
    assert parameters["backend"]["temperature"] == parameters["search"]["temperature"] == 0.8
    assert parameters["run"]["seed"] == 3


def test_missing_model_or_dataset_is_a_clear_argument_error(workspace):
    with pytest.raises(argparse.ArgumentError, match="--model, --data"):
        parse(["--output", "results"])
    with pytest.raises(argparse.ArgumentError, match="--data"):
        parse(["--model", "model", "--output", "results"])


def test_short_greedy_alias_preserves_explicit_boolean(workspace):
    parameters = parse(["--model", "model", "--data", "data", "--output", "results", "--no-greedy"])
    assert parameters["search"]["greedy_commitment"] is False
    parameters = parse(["--model", "model", "--data", "data", "--output", "results", "--greedy"])
    assert parameters["search"]["greedy_commitment"] is True


def install_runtime(monkeypatch, workspace):
    data = workspace / "data.jsonl"
    row = {"platform": "test", "question_id": "1", "question_title": "Example",
        "question_content": "Print a number.", "starter_code": "", "difficulty": "easy",
        "contest_date": "2025-02-01", "public_test_cases": [{"input": "1", "output": "1"}],
        "private_test_cases": [{"input": SECRET, "output": SECRET}], "metadata": {}}
    data.write_text(json.dumps(row) + "\n", encoding="utf-8")
    state = SimpleNamespace(prompts=[], update_calls=0, private_calls=0, final_calls=0)

    class FakeBackend:
        def __init__(self, config, private_store=None):
            self.config = config
            self.version = 0

        def reset_instance(self, instance):
            self.instance = instance

        def describe(self):
            return {"config": asdict(self.config)}

        def render_prompt(self, messages):
            state.prompts.append(messages)
            return json.dumps(messages)

        def generate(self, prompt, **kwargs):
            return [Sample("```python\nprint(1)\n```", token_ids=[2], prompt_ids=[1],
                           old_logprobs=[-1], instance_id=self.instance)] * kwargs["n"]

        def update(self, groups):
            state.update_calls += 1
            raise AssertionError("--no-train must never update the model")

        def close(self):
            pass

    class FakeEvaluator:
        def __init__(self, config, private_store=None):
            self.config = config

        def check_available(self):
            return {"runtime": "fixture"}

        def performance(self, problem_id):
            return {"test_executions": 0, "eval_workers": self.config.eval_workers}

        def evaluate_public(self, problem, code):
            return SimpleNamespace(passed=2, total=2, all_passed=True, feedback=())

        def evaluate_private(self, problem, code, store):
            state.private_calls += 1
            return SimpleNamespace(passed=1, total=2, all_passed=False)

        def evaluate_final(self, problem, code, store):
            state.final_calls += 1
            return SimpleNamespace(passed=3, total=4, all_passed=False)

        def evaluate_candidate(self, problem, code):
            return CandidateEvaluation(PublicEvaluation(True, 2, 2, ()), PrivateScore(1, 2, False))

    monkeypatch.setattr(runner, "Backend", FakeBackend)
    monkeypatch.setattr(runner, "_ReportingProcessEvaluator", FakeEvaluator)
    return data, state


def test_primary_cli_no_train_runs_without_config_or_optimizer(workspace, monkeypatch):
    data, state = install_runtime(monkeypatch, workspace)
    assert runner.main(["--model", "model", "--data", str(data), "--output", "results",
        "--backend", "hf", "--no-train", "--simulations", "2", "--width", "1"]) == 0
    assert state.update_calls == 0 and len(state.prompts) == 2
    metadata = json.loads((workspace / "results" / "metadata.json").read_text())
    assert metadata["config"]["search"]["train"] is False
    assert metadata["config"]["backend"]["adapter_dir"] == str(workspace / "results" / "adapters")
    assert metadata["evaluation_protocol"] == "all_tests_reward"
    assert metadata["heldout_tests_valid"] is False



def test_cli_always_uses_all_tests_and_never_exposes_private_details(workspace, monkeypatch):
    data, state = install_runtime(monkeypatch, workspace)
    assert runner.main(['--model','model','--data',str(data),'--output','results',
        '--no-train','--simulations','1','--width','1']) == 0
    result, = runner._read_strict_jsonl(workspace/'results/process.jsonl')
    assert result['optimization_score'] == 0.75 and result['reward_tests'] == 'all'
    assert result['stop_reason'] == 'budget_exhausted' and result['public_score'] == 1
    assert SECRET not in json.dumps(state.prompts)


@pytest.mark.parametrize('flag,value', [('--early-stop','public'),('--reward-tests','public')])
def test_removed_switches_are_rejected(flag, value):
    with pytest.raises(SystemExit):
        runner.build_parser().parse_args(['--output','results',flag,value])


def test_help_has_no_reward_or_stop_switch():
    help_text = runner.build_parser().format_help()
    assert '--early-stop' not in help_text and '--reward-tests' not in help_text
