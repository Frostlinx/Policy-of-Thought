"""Real local-process execution checks. These require POSIX resource limits."""
import json
import os
from pathlib import Path
import shutil
import time
import uuid

import pytest

from pot.data import PrivateTestStore, PublicProblem, TestCase as Case
from pot.evaluator import DockerConfig
from pot.process_evaluator import ProcessEvaluator


pytestmark = pytest.mark.skipif(os.name != "posix", reason="POSIX process resource limits required")
ROOT = Path(__file__).resolve().parents[1] / "work" / "test-process-eval"


@pytest.fixture
def evaluator():
    ROOT.mkdir(parents=True, exist_ok=True)
    path = ROOT / uuid.uuid4().hex
    path.mkdir()
    evaluator = ProcessEvaluator(DockerConfig(work_dir=str(path), timeout_seconds=0.5,
        startup_grace_seconds=1, memory_mb=1024, max_output_bytes=4096))
    try:
        yield evaluator
    finally:
        path.resolve().relative_to(ROOT.resolve())
        shutil.rmtree(path)


def problem(tests, func_name=None):
    return PublicProblem("test/1", "1", "test", "Example", "Solve", "", "easy", "2025-02-01",
                         tuple(tests), func_name)


def execute(evaluator, code, test=None, func_name=None):
    with evaluator._candidate(code) as directory:
        path = Path(directory) / "candidate.py"
        path.write_text(code, encoding="utf-8")
        return evaluator._run(path, test or Case("", ""), func_name)


def test_subprocess_stdin_sum_matches_official_comparison(evaluator):
    p = problem([Case("2 7\n", "9\n"), Case("-3 5\n", "2\n")])
    result = evaluator.evaluate_public(p, "a,b=map(int,input().split()); print(a+b)")
    assert result.all_passed and result.passed == result.total == 2
    assert list(Path(evaluator.config.work_dir).iterdir()) == []


def test_functional_solution_and_return_protocol(evaluator):
    p = problem([Case("[1, 2, 3]\n4", "10", "functional")], func_name="calculate")
    code = "class Solution:\n    def calculate(self, values: List[int], addition: int):\n        return sum(values)+addition"
    result = evaluator.evaluate_public(p, code)
    assert result.all_passed


def test_timeout_and_output_limit_are_bounded(evaluator):
    before = time.monotonic()
    timed_out = execute(evaluator, "while True: pass")
    assert not timed_out.ok and timed_out.status == "timeout"
    assert time.monotonic() - before < 4
    overflow = execute(evaluator, "while True: print('x'*4096)")
    assert not overflow.ok and overflow.status == "output_limit"
    assert len(overflow.stdout.encode()) <= evaluator.config.max_output_bytes
    assert len(overflow.stderr.encode()) <= evaluator.config.max_output_bytes


def test_private_execution_never_returns_test_or_exception_details(evaluator):
    secret = "SECRET_PRIVATE_INPUT_74410"
    p = problem([Case("public", "expected")])
    store = PrivateTestStore({p.problem_id: (Case(secret, secret),)})
    score = evaluator.evaluate_private(p, "raise RuntimeError(input())", store)
    assert score.passed == 0 and score.total == 1 and not score.all_passed
    assert secret not in repr(score)
    assert not hasattr(score, "feedback") and not hasattr(score, "stderr")


def test_child_environment_has_no_parent_credentials_and_only_candidate_home(evaluator, monkeypatch):
    monkeypatch.setenv("POT_TEST_SECRET_CREDENTIAL", "DO_NOT_INHERIT_1865")
    result = execute(evaluator, "import os,json; print(json.dumps(dict(os.environ),sort_keys=True))")
    assert result.ok, result.stderr
    environment = json.loads(result.stdout)
    assert "POT_TEST_SECRET_CREDENTIAL" not in environment
    assert set(environment) <= {"PATH", "LANG", "LC_ALL", "HOME", "TMPDIR", "TMP", "TEMP",
                                "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"}
    assert all(environment[key] == '1' for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'))
    assert Path(environment["HOME"]).parent == Path(evaluator.config.work_dir)
    assert environment["HOME"] == environment["TMPDIR"]


def test_candidate_scratch_files_are_cleaned_and_execution_is_not_claimed_sandboxed(evaluator):
    # The stdio compatibility wrapper mocks builtins.open; os.open still writes
    # a real candidate scratch file and verifies cleanup of auxiliary files.
    result = execute(evaluator, "import os; f=os.open('scratch.tmp', os.O_CREAT|os.O_WRONLY,0o600); os.write(f,b'hello'); os.close(f)")
    assert result.ok
    assert list(Path(evaluator.config.work_dir).iterdir()) == []
    report = evaluator.check_available()
    assert report["runtime"] == "subprocess" and report["security_sandbox"] is False
    assert report["memory_mb"] == 1024
