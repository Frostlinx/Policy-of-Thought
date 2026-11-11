"""Benchmark interfaces, native assertions, privacy and persisted result semantics."""
from dataclasses import asdict, replace
import json
import os

import pytest

from pot.backend import Sample
from pot.benchmarks import normalize_humaneval, normalize_mbpp, benchmark_protocol
from pot.data import PrivateTestStore, load_dataset, save_dataset
from pot.evaluator import DockerConfig
from pot.process_evaluator import ProcessEvaluator
from pot.search import Node, problem_messages


def human_record():
    return dict(task_id="HumanEval/0", prompt='from typing import List\ndef double(x: int) -> int:\n    """Double x. Example: double(2) == 4."""\n',
        entry_point="double", canonical_solution="    return 'REFERENCE_SENTINEL'\n",
        test="def check(candidate):\n    for x in [2, 5, 8]:\n        assert candidate(x) == 2*x, 'PRIVATE_SENTINEL'\n")


def mbpp_record(task_id=11):
    return dict(task_id=task_id, prompt="Return a pair containing x and x squared.",
        code="REFERENCE_SENTINEL", test_imports=["import math"],
        test_list=["assert pair(2) == (2, 4)", "assert pair(3) == (3, 9)",
                   "assert pair(4) == (4, math.isqrt(256))"])


def evaluator(tmp_path, store):
    return ProcessEvaluator(DockerConfig(timeout_seconds=1, memory_mb=1024,
        work_dir=str(tmp_path), eval_workers=3), store)


def test_prompt_interfaces_and_reference_exclusion():
    human, hidden = normalize_humaneval([human_record()])
    mbpp, empty = normalize_mbpp([mbpp_record()])
    for p in human + mbpp:
        text = json.dumps(problem_messages(p, Node(0)))
        assert "REFERENCE_SENTINEL" not in text and "PRIVATE_SENTINEL" not in text
        assert "Implement the specified method in class Solution" not in text
        assert "complete Python module" in text
        assert "REFERENCE_SENTINEL" not in json.dumps(p.to_dict())
    assert "double(x: int)" in json.dumps(human[0].to_dict())
    assert "pair(3)" in json.dumps(problem_messages(mbpp[0], Node(0)))
    assert len(hidden._for_evaluation(human[0].problem_id)) == 2
    assert empty._for_evaluation(mbpp[0].problem_id) == ()


@pytest.mark.parametrize("normalizer,records", [
    (normalize_humaneval, [human_record(), human_record()]),
    (normalize_mbpp, [mbpp_record(), mbpp_record()]),
    (normalize_mbpp, [mbpp_record(1)]), (normalize_mbpp, [mbpp_record(511)]),
])
def test_invalid_splits_and_duplicate_ids_rejected(normalizer, records):
    with pytest.raises(ValueError):
        normalizer(records)


def test_dataset_roundtrip_and_selection_mismatch(tmp_path):
    public, private = normalize_mbpp([mbpp_record()])
    save_dataset(tmp_path, public, private, dict(selection="mbpp-sanitized", benchmark_protocol=benchmark_protocol("mbpp-sanitized")))
    loaded, store = load_dataset(tmp_path, "mbpp-sanitized")
    assert loaded == public and store._for_evaluation(loaded[0].problem_id) == ()
    with pytest.raises(ValueError):
        load_dataset(tmp_path, "humaneval")
    with pytest.raises(ValueError):
        load_dataset(tmp_path, "mbpp-sanitized", start_date="2025-01-01")
    (tmp_path / "manifest.json").write_text(json.dumps(dict(selection="mbpp-sanitized")))
    with pytest.raises(ValueError, match="protocol changed"):
        load_dataset(tmp_path, "mbpp-sanitized")


def test_sanitized_mbpp_retains_extra_published_tests():
    record = mbpp_record()
    record["test_list"] += ["assert pair(5) == (5, 25)", "assert pair(6) == (6, 36)"]
    public, _ = normalize_mbpp([record])
    assert len(public[0].public_tests) == 5


@pytest.mark.skipif(os.name != "posix", reason="POSIX child resource limits")
def test_humaneval_checker_is_whole_private_suite(tmp_path):
    problems, store = normalize_humaneval([human_record()])
    e = evaluator(tmp_path, store)
    wrong = e.evaluate_candidate(problems[0], "def double(x): return 4")
    assert not wrong.all_passed and wrong.reward == 0
    assert wrong.private.total == 1 and wrong.public.total == 0
    assert "PRIVATE_SENTINEL" not in repr(wrong)
    assert wrong.public.feedback == ()
    correct = "def double(x: int) -> int:\n return x*2"
    assert e.evaluate_candidate(problems[0], correct).all_passed
    assert e.evaluate_final(problems[0], correct, store).all_passed


@pytest.mark.skipif(os.name != "posix", reason="POSIX child resource limits")
def test_humaneval_fractional_reward_retains_original_gate_and_loop_groups(tmp_path):
    record = human_record()
    record["test"] = "def check(candidate):\n    assert candidate(2) == 4\n    for x in [5, 8]:\n        assert candidate(x) == 2*x\n"
    problems, store = normalize_humaneval([record])
    e = evaluator(tmp_path, store)
    wrong = e.evaluate_candidate(problems[0], "def double(x): return 4")
    # Whole suite gates success only; 1 of 2 assertion groups passes: 1/2.
    assert wrong.private.total == 2 and wrong.reward == pytest.approx(1/2)
    assert wrong.private.gate_total == 1 and wrong.private.gate_passed == 0
    assert not wrong.all_passed and wrong.public.feedback == ()
    assert e.evaluate_candidate(problems[0], "def double(x): return 2*x").all_passed
    original = json.loads(store._for_evaluation(problems[0].problem_id)[0].input)["test"]
    assert original == record["test"] + "\ncheck(double)\n"


def test_success_gate_can_block_full_assertion_reward():
    from pot.evaluator import CandidateEvaluation, PublicEvaluation, PrivateScore
    result = CandidateEvaluation(PublicEvaluation(False, 0, 0, ()), PrivateScore(3, 3, False, 0, 1))
    assert result.reward == 1.0 and not result.all_passed


@pytest.mark.skipif(os.name != "posix", reason="POSIX child resource limits")
def test_humaneval_probes_preserve_nonselected_assertion_calls(tmp_path):
    record = human_record()
    record["test"] = "def check(candidate):\n    state = []\n    assert candidate(state) == 1\n    assert candidate(state) == 2\n"
    problems, store = normalize_humaneval([record])
    correct = "def double(state):\n state.append(1)\n return len(state)"
    assert evaluator(tmp_path, store).evaluate_candidate(problems[0], correct).all_passed


def test_humaneval_try_checkers_are_not_rewritten():
    record = human_record()
    record["test"] = "def check(candidate):\n    try:\n        assert candidate(2) == 4\n    except AssertionError:\n        raise\n"
    problems, store = normalize_humaneval([record])
    assert len(store._for_evaluation(problems[0].problem_id)) == 1


@pytest.mark.skipif(os.name != "posix", reason="POSIX child resource limits")
def test_mbpp_native_tuples_public_feedback_and_fractional_reward(tmp_path):
    problems, store = normalize_mbpp([mbpp_record()])
    p = problems[0]
    e = evaluator(tmp_path, store)
    wrong = e.evaluate_candidate(p, "def pair(x): return (2, 4)")
    assert wrong.public.passed == 1 and wrong.reward == pytest.approx(1/3)
    assert wrong.private.total == 0 and not wrong.all_passed
    assert "pair(3)" in wrong.public.feedback[0]["input"]
    good = "print('debug')\ndef pair(x): return x, x*x"
    assert e.evaluate_candidate(p, good).all_passed
    assert e.evaluate_final(p, good, store).all_passed
    assert not e.evaluate_candidate(p, "def pair(x): return [x, x*x]").all_passed
    # No loophole for missing private LCB/HumanEval data or entirely empty suites.
    with pytest.raises(ValueError, match="nonempty private"):
        e.evaluate_candidate(replace(p, benchmark="lcb"), good)
    assert not e.evaluate_final(replace(p, benchmark="lcb"), good, store).all_passed
    with pytest.raises(ValueError, match="nonempty tests"):
        e.evaluate_candidate(replace(p, public_tests=()), good)


@pytest.mark.skipif(os.name != "posix", reason="POSIX child resource limits")
@pytest.mark.parametrize("code", ["import sys; sys.exit(0)", "import os; os._exit(0)",
    "def pair(x):\n while True: pass", "def pair(: pass", "class Solution:\n def pair(self,x): return x,x*x"])
def test_incomplete_execution_timeout_and_wrong_interface_never_pass(tmp_path, code):
    problems, store = normalize_mbpp([mbpp_record()])
    assert not evaluator(tmp_path, store).evaluate_candidate(problems[0], code).all_passed


@pytest.mark.skipif(os.name != "posix", reason="POSIX child resource limits")
def test_mbpp_results_and_resume_without_private_tests(tmp_path):
    from pot.run import run
    public, private = normalize_mbpp([mbpp_record()])
    data = tmp_path / "data"
    save_dataset(data, public, private, dict(selection="mbpp-sanitized", benchmark_protocol=benchmark_protocol("mbpp-sanitized")))

    class FakeBackend:
        def __init__(self, cfg): self.cfg = cfg; self.version = 0
        def reset_instance(self, instance): self.instance = instance
        def render_prompt(self, messages): return json.dumps(messages)
        def generate(self, prompt, **kwargs):
            return [Sample(text="```python\ndef pair(x): return x, x*x\n```",
                token_ids=[2], prompt_ids=[1], old_logprobs=[-1.0], instance_id=self.instance)]
        def describe(self): return dict(config=asdict(self.cfg))
        def close(self): pass

    config = dict(backend=dict(model_path="unused", generation_backend="hf"),
        dataset=dict(path=str(data), selection="mbpp-sanitized"),
        evaluation=dict(executor="subprocess", timeout_seconds=2),
        search=dict(simulations=1, width=1, train=False, greedy_commitment=False))
    output = tmp_path / "run"
    assert run(config, output, backend_factory=FakeBackend) == 0
    row = json.loads((output / "process.json").read_text())
    summary = json.loads((output / "summary.json").read_text())
    assert row["score"] == 1 and row["private_pass_rate"] is None
    assert row["official_total"] == 3 and row["private_total"] == 0
    assert row["test_unit"] == "published_assertion"
    assert summary["mean_private_pass_rate"] is None
    assert not summary["benchmark_protocol"]["private_tests_available"]
    assert run(config, output, resume=True, backend_factory=FakeBackend) == 0
