"""Persisted runner integration tests; real search/data, fake model/execution."""
from dataclasses import asdict
import importlib
import json
from pathlib import Path
import shutil
from types import SimpleNamespace
import uuid

import pytest

from pot.backend import Sample
from pot.evaluator import _Execution

runner = importlib.import_module("pot.run")
PRIVATE_SENTINEL = "PRIVATE_TEST_CONTENT_MUST_NEVER_ENTER_LOGS_92318"
TEST_ROOT = Path(__file__).resolve().parents[1] / "work" / "test-runner"


@pytest.fixture
def environment(monkeypatch):
    # Inherited ACLs avoid Python 3.13 Windows tempfile(0700) denying the app
    # sandbox access. Every test owns a unique directory under the workspace.
    TEST_ROOT.mkdir(parents=True, exist_ok=True)
    root = TEST_ROOT / uuid.uuid4().hex
    root.mkdir()
    source = root / "dataset.jsonl"
    records = [{"platform": "atcoder", "question_id": str(index),
        "question_title": f"Problem {index}", "question_content": "Print the requested answer.",
        "starter_code": "", "difficulty": "easy", "contest_date": "2025-02-01",
        "public_test_cases": [{"input": "1\n", "output": "1\n", "testtype": "stdin"}],
        "private_test_cases": [{"input": PRIVATE_SENTINEL, "output": PRIVATE_SENTINEL,
                                "testtype": "stdin"}], "metadata": {}} for index in range(3)]
    source.write_text("".join(json.dumps(row) + "\n" for row in records), encoding="utf-8")
    config = root / "config.json"
    config_data = {"backend": {"model_path": "offline-model", "generation_backend": "hf"},
        "dataset": {"path": "dataset.jsonl", "selection": "v6"},
        "search": {"simulations": 1, "width": 1, "train": False, "greedy_commitment": False},
        "run": {"seed": 123}}
    config.write_text(json.dumps(config_data), encoding="utf-8")
    state = SimpleNamespace(prompts=[], resets=[], initializations=0, closes=0,
                            fail_on=None, private_calls=0, solve_calls=[])

    class FakeBackend:
        def __init__(self, cfg):
            self.cfg = cfg
            self.version = 0
            state.initializations += 1

        def reset_instance(self, instance):
            self.instance = instance
            state.resets.append(instance)

        def render_prompt(self, messages):
            state.prompts.append(messages)
            return json.dumps(messages)

        def generate(self, prompt, **kwargs):
            state.solve_calls.append(self.instance)
            if self.instance == state.fail_on:
                raise RuntimeError(PRIVATE_SENTINEL)
            text = "```python\nprint('bad')\n```" if self.instance.endswith("/1") else "```python\nprint('good')\n```"
            return [Sample(text=text, prompt_ids=[1], token_ids=[2], old_logprobs=[-1.0],
                           instance_id=self.instance, finish_reason="stop")]

        def describe(self):
            return {"config": asdict(self.cfg), "base_parameters_frozen": True}

        def close(self):
            state.closes += 1

    class FakeEvaluator(runner._ReportingEvaluator):
        def check_available(self):
            return {"runtime": "fake-isolated-evaluator", "image": "fixture"}

        def _test(self, candidate_path, test, func_name):
            hidden = test.input == PRIVATE_SENTINEL
            state.private_calls += int(hidden)
            code = candidate_path.read_text(encoding="utf-8")
            passed = not hidden or "'good'" in code
            return passed, _Execution(ok=True, stdout="safe-output", stderr="")

    monkeypatch.setattr(runner, "Backend", FakeBackend)
    monkeypatch.setattr(runner, "_ReportingEvaluator", FakeEvaluator)
    try:
        yield SimpleNamespace(root=root, config=config, config_data=config_data,
                              source=source, output=root / "results", state=state)
    finally:
        resolved = root.resolve()
        resolved.relative_to(TEST_ROOT.resolve())
        assert resolved.parent == TEST_ROOT.resolve()
        shutil.rmtree(resolved)


def rows(path):
    return runner._read_strict_jsonl(path)


def summary(environment):
    return json.loads((environment.output / "summary.json").read_text(encoding="utf-8"))


def test_success_outputs_scores_fractional_metrics_and_private_boundary(environment, capsys):
    env = environment
    assert runner.run(env.config, env.output) == 0
    results = rows(env.output / "process.jsonl")
    assert (env.output / "process.json").read_bytes() == (env.output / "process.jsonl").read_bytes()
    assert len(results) == 3
    assert [row["score"] for row in results] == [1, 0, 1]
    assert [row["pass_rate"] for row in results] == [1, 0.5, 1]
    assert [row["private_pass_rate"] for row in results] == [1, 0, 1]
    assert [row["public_score"] for row in results] == [1, 1, 1]
    assert all(row["status"] == "ok" for row in results)
    assert all(row["elapsed_seconds"] >= 0 and row["official_total"] == 2 for row in results)
    report = summary(env)
    assert report["status"] == "completed"
    assert report["requested_count"] == report["evaluated_count"] == report["attempted_count"] == 3
    assert report["success_count"] == 2 and report["failure_count"] == 1 and report["error_count"] == 0
    assert report["accuracy"] == pytest.approx(2 / 3)
    assert len((env.output / "summary.json").read_text().splitlines()) == 1
    assert rows(env.output / "summary.jsonl")[-1] == report
    assert json.loads((env.output / "progress.json").read_text())["status"] == "completed"
    assert env.state.resets == ["atcoder/0", "atcoder/1", "atcoder/2"]
    assert env.state.private_calls == 3, "evaluate_final must reuse selected-code private counts"
    public_outputs = "".join(path.read_text(encoding="utf-8") for path in env.output.rglob("*") if path.is_file())
    assert PRIVATE_SENTINEL not in public_outputs
    assert PRIVATE_SENTINEL not in json.dumps(env.state.prompts)
    captured = capsys.readouterr()
    assert PRIVATE_SENTINEL not in captured.out + captured.err
    assert env.state.closes == 1


def test_infrastructure_error_aborts_excludes_score_then_resume_skips_recorded_ids(environment, capsys):
    env = environment
    env.config.write_text(json.dumps(env.config_data), encoding="utf-8")
    env.state.fail_on = "atcoder/1"
    assert runner.run(env.config, env.output) == 2
    results = rows(env.output / "process.jsonl")
    assert [row["status"] for row in results] == ["ok", "infrastructure_error"]
    assert results[1]["score"] is None and results[1]["pass_rate"] is None
    assert results[1]["reward_tests"] == "all"
    assert results[1]["feedback_mode"] == "public"
    assert results[1]["stop_rule"] == "all_tests_passed"
    assert results[1]["evaluation_protocol"] == "all_tests_reward"
    assert results[1]["heldout_tests_valid"] is False
    assert env.state.solve_calls == ["atcoder/0", "atcoder/1"]
    report = summary(env)
    assert report["status"] == "incomplete" and report["remaining_count"] == 1
    assert report["error_count"] == 1 and report["evaluated_count"] == 1
    assert report["failure_count"] == 0 and report["accuracy"] == 1
    assert PRIVATE_SENTINEL not in capsys.readouterr().err
    env.state.fail_on = None
    assert runner.run(env.config, env.output, resume=True) == 2
    assert summary(env)["status"] == "completed_with_errors"
    assert summary(env)["attempted_count"] == 3
    assert env.state.solve_calls == ["atcoder/0", "atcoder/1", "atcoder/2"]
    assert len({row["problem_id"] for row in rows(env.output / "process.jsonl")}) == 3


def test_complete_resume_does_not_load_model_or_repeat_questions(environment):
    env = environment
    assert runner.run(env.config, env.output) == 0
    before = (env.output / "process.jsonl").read_bytes()
    assert runner.run(env.config, env.output, resume=True) == 0
    assert (env.output / "process.jsonl").read_bytes() == before
    assert env.state.initializations == 1


@pytest.mark.parametrize("changed", ["config", "dataset", "source"])
def test_resume_rejects_changed_identity(environment, monkeypatch, changed):
    env = environment
    assert runner.run(env.config, env.output) == 0
    if changed == "config":
        env.config_data["run"]["seed"] += 1
        env.config.write_text(json.dumps(env.config_data), encoding="utf-8")
    elif changed == "dataset":
        with env.source.open("a", encoding="utf-8") as stream:
            stream.write("\n")
    else:
        original = runner._hash_file
        monkeypatch.setattr(runner, "_hash_file", lambda path: "changed-source-hash"
                            if path.name == "backend.py" else original(path))
    with pytest.raises(ValueError, match="identity differs"):
        runner.run(env.config, env.output, resume=True)
    assert env.state.initializations == 1


def test_resume_rejects_disagreeing_or_corrupt_logs(environment):
    env = environment
    assert runner.run(env.config, env.output) == 0
    with (env.output / "process.jsonl").open("a", encoding="utf-8") as stream:
        stream.write("{}\n")
    with pytest.raises(ValueError, match="copies disagree"):
        runner.run(env.config, env.output, resume=True)
    (env.output / "process.jsonl").write_bytes((env.output / "process.json").read_bytes())
    with (env.output / "trace.jsonl").open("a", encoding="utf-8") as stream:
        stream.write("incomplete")
    with pytest.raises(ValueError, match="Incomplete JSONL"):
        runner.run(env.config, env.output, resume=True)


def test_model_load_failure_leaves_initialized_empty_results(environment, monkeypatch, capsys):
    env = environment

    def unavailable(config):
        raise RuntimeError(PRIVATE_SENTINEL)

    monkeypatch.setattr(runner, "Backend", unavailable)
    assert runner.run(env.config, env.output) == 2
    assert rows(env.output / "process.jsonl") == []
    assert rows(env.output / "process.json") == []
    report = summary(env)
    assert report["status"] == "incomplete" and report["attempted_count"] == 0
    assert report["accuracy"] is None
    assert PRIVATE_SENTINEL not in capsys.readouterr().err


def test_runner_accepts_current_search_trace_events(environment, monkeypatch):
    original_solve = runner.solve

    def solve_with_events(problem, backend, evaluator, cfg, **kwargs):
        emit = kwargs["emit"]
        emit({"event": "context_pruned", "problem_id": problem.problem_id, "node_id": 0})
        emit({"event": "update", "problem_id": problem.problem_id,
              "skipped": True, "reason": "empty_completion", "optimizer_steps": 0})
        return original_solve(problem, backend, evaluator, cfg, **kwargs)

    monkeypatch.setattr(runner, "solve", solve_with_events)
    assert runner.run(environment.config, environment.output, limit=1) == 0
    assert {row["event"] for row in rows(environment.output / "trace.jsonl")} == {
        "context_pruned", "update", "candidate"}


def test_offset_limit_keep_absolute_problem_indices(environment):
    env = environment
    assert runner.run(env.config, env.output, offset=1, limit=1) == 0
    result, = rows(env.output / "process.jsonl")
    assert result["problem_index"] == 1 and result["question_id"] == "1"
    assert (env.output / "solutions" / "00001.py").is_file()
    assert summary(env)["requested_count"] == 1


def test_subprocess_executor_config_is_preserved_and_source_hashed(environment, monkeypatch):
    env = environment
    env.config_data["evaluation"] = {"executor": "subprocess"}
    env.config.write_text(json.dumps(env.config_data), encoding="utf-8")
    # Route to the existing fake execution fixture; this test validates runner
    # configuration/routing, while test_process_eval exercises POSIX processes.
    monkeypatch.setattr(runner, "_ReportingProcessEvaluator", runner._ReportingEvaluator)
    monkeypatch.setattr(runner._ReportingEvaluator, "performance", lambda self, problem_id: {}, raising=False)
    assert runner.run(env.config, env.output, limit=1) == 0
    metadata = json.loads((env.output / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["config"]["evaluation"]["executor"] == "subprocess"
    assert metadata["config"]["evaluation"]["memory_mb"] == 1024
    assert "process_evaluator.py" in metadata["identity"]["source_sha256"]


@pytest.mark.parametrize('scope', ['trajectory', 'completion'])
def test_loss_scope_cli_is_preserved_in_effective_run_metadata(environment, scope):
    env = environment
    args = runner.build_parser().parse_args([
        '--config', str(env.config), '--output', str(env.output), '--loss-scope', scope])
    parameters = runner.parameters_from_args(args)
    assert parameters['backend']['loss_scope'] == scope
    assert runner.run(parameters, env.output, limit=1) == 0
    metadata = json.loads((env.output / 'metadata.json').read_text())
    assert metadata['config']['backend']['loss_scope'] == scope
