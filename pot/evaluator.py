"""Docker-only execution and host-side LiveCodeBench comparison.

No fallback executes candidate code on the host. Expected outputs stay outside
the container. Private evaluation returns counts/booleans only, never test
inputs, outputs, traces, stderr, or exception text.
"""
from __future__ import annotations

import json
import ast
import hashlib
import os
import subprocess
import tempfile
import threading
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Sequence

from .data import PrivateTestStore, PublicProblem, TestCase


@dataclass(frozen=True)
class DockerConfig:
    image: str = field(default_factory=lambda: os.environ.get("POT_DOCKER_IMAGE", "python:3.11-slim-bookworm"))
    command: tuple[str, ...] = field(default_factory=lambda: _docker_command())
    host: str | None = field(default_factory=lambda: os.environ.get("DOCKER_HOST"))
    config_dir: str | None = field(default_factory=lambda: os.environ.get("DOCKER_CONFIG"))
    timeout_seconds: float = 6.0
    startup_grace_seconds: float = 8.0
    memory_mb: int = 512
    cpus: float = 1.0
    pids_limit: int = 64
    max_output_bytes: int = 1024 * 1024
    work_dir: str | None = None
    eval_workers: int = 1
    eval_global_workers: int = 32
    eval_slot_dir: str | None = None
    ojbench_python: str | None = None
    ojbench_pypy: str | None = None
    ojbench_gpp: str = "/usr/bin/g++"

    def __post_init__(self):
        if (self.timeout_seconds <= 0 or self.startup_grace_seconds < 0 or self.memory_mb < 64
                or self.cpus <= 0 or self.pids_limit < 1 or self.max_output_bytes < 1024):
            raise ValueError("Invalid Docker resource limit")
        if not self.command or not self.image:
            raise ValueError("Docker command and image are required")
        if any(not isinstance(n, int) or isinstance(n, bool) or n < 1
               for n in (self.eval_workers, self.eval_global_workers)):
            raise ValueError("Evaluation concurrency limits must be positive integers")


def _docker_command() -> tuple[str, ...]:
    value = os.environ.get("POT_DOCKER_COMMAND", "docker")
    if value.startswith("["):
        parsed = json.loads(value)
        if not isinstance(parsed, list) or not parsed or not all(isinstance(v, str) for v in parsed):
            raise ValueError("POT_DOCKER_COMMAND must be a path or JSON argv list")
        return tuple(parsed)
    return (value,)


@dataclass(frozen=True)
class PublicEvaluation:
    all_passed: bool
    passed: int
    total: int
    feedback: tuple[dict, ...]


@dataclass(frozen=True)
class PrivateScore:
    """Aggregate counts for rewards and reporting; no private diagnostics."""
    passed: int
    total: int
    all_passed: bool
    gate_passed: int = 0
    gate_total: int = 0


@dataclass(frozen=True)
class CandidateEvaluation:
    """Public feedback plus private counts, never private execution details."""
    public: PublicEvaluation
    private: PrivateScore

    @property
    def passed(self):
        return self.public.passed + self.private.passed

    @property
    def total(self):
        return self.public.total + self.private.total

    @property
    def reward(self):
        return self.passed / self.total if self.total else 0.0

    @property
    def all_passed(self):
        return (self.total > 0 and self.passed == self.total
                and self.private.gate_passed == self.private.gate_total)


@dataclass(frozen=True)
class _Execution:
    ok: bool
    stdout: str = ""
    stderr: str = ""
    status: str = "ok"
    elapsed_seconds: float = 0.0
    queue_wait_seconds: float = 0.0


class SandboxUnavailable(RuntimeError):
    pass


def compare_stdio(prediction: str, expected: str) -> bool:
    """Official LCB stripped lines, then exact Decimal token equality.

    There is deliberately no np.isclose tolerance and no whitespace-insensitive
    string token fallback. This follows testing_util.py grade_stdio.
    """
    actual_lines = [line.strip() for line in prediction.strip().split("\n")]
    expected_lines = [line.strip() for line in expected.strip().split("\n")]
    if len(actual_lines) != len(expected_lines):
        return False
    for actual, wanted in zip(actual_lines, expected_lines):
        if actual == wanted:
            continue
        try:
            if [Decimal(x) for x in actual.split()] == [Decimal(x) for x in wanted.split()]:
                continue
        except (InvalidOperation, ValueError):
            pass
        return False
    return True


def compare_functional(prediction, expected_json: str) -> bool:
    if isinstance(prediction, tuple):
        prediction = list(prediction)
    return prediction == json.loads(expected_json)


def _decode_prediction(value):
    """Decode JSON transport while preserving nested tuples and dictionary keys.

    Ordinary JSON would silently convert nested tuples to lists and integer dict
    keys to strings, incorrectly accepting outputs rejected by official LCB.
    """
    kind = value["kind"]
    items = value["value"]
    if kind == "scalar":
        if items is not None and not isinstance(items, (str, int, float, bool)):
            raise ValueError("Invalid scalar result")
        return items
    if kind == "list":
        return [_decode_prediction(item) for item in items]
    if kind == "tuple":
        return tuple(_decode_prediction(item) for item in items)
    if kind == "dict":
        return {_decode_prediction(key): _decode_prediction(item) for key, item in items}
    raise ValueError("Unsupported functional return type")


# Executed in the evaluator child. LCB inputs are streamed through stdin, with
# comparisons in the parent. Python assertion benchmarks also stream the test
# program to the child; private diagnostics never enter model prompts.
# LCB preimports and stdin AST wrapping are reproduced to support omitted typing
# imports, Solution classes, __main__ guards, and top-level return statements.
_RUNNER = r'''
import ast, contextlib, io, json, signal, sys, types
from unittest.mock import mock_open, patch
def timeout_handler(*args):
    raise TimeoutError('execution timeout')
signal.signal(signal.SIGALRM, timeout_handler)
signal.setitimer(signal.ITIMER_REAL, float(sys.argv[3]))
prelude = ''' + repr("from string import *\nfrom re import *\nfrom datetime import *\nfrom collections import *\nfrom heapq import *\nfrom bisect import *\nfrom copy import *\nfrom math import *\nfrom random import *\nfrom statistics import *\nfrom itertools import *\nfrom functools import *\nfrom operator import *\nfrom io import *\nfrom sys import *\nfrom json import *\nfrom builtins import *\nfrom typing import *\nimport string, re, datetime, collections, heapq, bisect, copy, math, random, statistics, itertools, functools, operator, io, sys, json\nsys.setrecursionlimit(50000)\n") + r'''
source = open('/candidate.py', encoding='utf-8').read()
def pack(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return {'kind': 'scalar', 'value': value}
    if isinstance(value, (list, tuple)):
        return {'kind': 'tuple' if isinstance(value, tuple) else 'list', 'value': [pack(item) for item in value]}
    if isinstance(value, dict):
        return {'kind': 'dict', 'value': [[pack(key), pack(item)] for key, item in value.items()]}
    raise TypeError('Unsupported functional return type')
mode, fn_name = sys.argv[1], sys.argv[2]
data = sys.stdin.read()
scope = {'__name__': '__main__'}
if mode == 'python_assert':
    payload = json.loads(data)
    # Full modules, native Python assertions and the original checker namespace.
    # No LCB preimports, JSON result coercion or rewriting of check(candidate).
    # Prints do not affect grading; a clean exit before the tests is NOT success.
    with contextlib.redirect_stdout(sys.stderr):
        exec(compile(payload.get('prefix', ''), '<public-interface>', 'exec'), scope)
        exec(compile(source, '/candidate.py', 'exec'), scope)
        exec(compile(payload.get('setup', ''), '<test-setup>', 'exec'), scope)
        exec(compile(payload['test'], '<test>', 'exec'), scope)
    print('POT_ASSERTIONS_COMPLETED')
elif mode == 'functional':
    # Candidate debug prints may not corrupt the return-value protocol.
    with contextlib.redirect_stdout(sys.stderr):
        exec(compile(prelude + '\n' + source, '/candidate.py', 'exec'), scope)
        target = scope['Solution']() if 'class Solution' in source else types.SimpleNamespace(**scope)
        method = getattr(target, fn_name)
        args = [json.loads(line) for line in data.split('\n')]
        result = method(*args)
        if isinstance(result, tuple):
            result = list(result)
    print(json.dumps(pack(result), ensure_ascii=False))
else:
    try:
        tree = ast.parse(source)
        if tree.body and isinstance(tree.body[-1], ast.If) and ast.unparse(tree.body[-1].test).strip() == "__name__ == '__main__'":
            tree.body = tree.body[:-1] + tree.body[-1].body
        imports = [node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))]
        statements = [node for node in tree.body if not isinstance(node, (ast.Import, ast.ImportFrom))]
        wrapped = ast.FunctionDef(name='wrapped_function', args=ast.arguments(posonlyargs=[], args=[], kwonlyargs=[], kw_defaults=[], defaults=[]), body=statements or [ast.Pass()], decorator_list=[])
        tree.body = imports + [wrapped]
        ast.fix_missing_locations(tree)
        source = ast.unparse(tree)
    except Exception:
        pass
    exec(compile(prelude + '\n' + source, '/candidate.py', 'exec'), scope)
    class Buffer:
        def __init__(self, text): self.text = text.encode('utf-8')
        def read(self, *args): return self.text
        def readline(self, *args): return self.text.split(b'\n')[0] + b'\n'
    class Stdin(io.StringIO):
        def __init__(self, text):
            super().__init__(text)
            self.buffer = Buffer(text)
    line_iterator = iter(data.split('\n'))
    with patch('builtins.open', mock_open(read_data=data)), patch('sys.stdin', Stdin(data)), patch('sys.stdin.readline', lambda *args: next(line_iterator)), patch('sys.stdin.readlines', lambda *args: data.split('\n')), patch('sys.stdin.read', lambda *args: data):
        try:
            scope['wrapped_function']()
        except SystemExit:
            pass
'''


class DockerEvaluator:
    def __init__(self, config: DockerConfig | None = None, private_store: PrivateTestStore | None = None):
        self.config = config or DockerConfig()
        self.private_store = private_store
        self._cached_problem = None
        self._candidate_scores = {}

    def evaluate_candidate(self, problem: PublicProblem, code: str) -> CandidateEvaluation:
        """Run all tests once; reuse the same counts for reward, stop and final score.

        Only public diagnostics leave the evaluator. The cache is cleared on a
        problem change and stores no private inputs, expectations or errors.
        """
        if self.private_store is None:
            raise ValueError("All-test evaluation requires private tests")
        hidden = self.private_store._for_evaluation(problem.problem_id)
        if not hidden and problem.benchmark != 'mbpp-sanitized':
            raise ValueError("All-test evaluation requires nonempty private tests")
        if not hidden and not problem.public_tests:
            raise ValueError("Evaluation requires nonempty tests")
        if self._cached_problem != problem.problem_id:
            self._cached_problem = problem.problem_id
            self._candidate_scores.clear()
        key = hashlib.sha256(code.encode()).hexdigest()
        if key not in self._candidate_scores:
            try:
                if not code.strip():
                    raise SyntaxError
                ast.parse(code)
            except SyntaxError:
                result = CandidateEvaluation(
                    PublicEvaluation(False, 0, len(problem.public_tests), ()),
                    PrivateScore(0, sum(t.role == 'reward' for t in hidden), False,
                                 0, sum(t.role == 'gate' for t in hidden)))
            else:
                result = CandidateEvaluation(self.evaluate_public(problem, code),
                    self.evaluate_private(problem, code, self.private_store))
            self._candidate_scores[key] = result
        return self._candidate_scores[key]

    def _base_command(self) -> list[str]:
        command = list(self.config.command)
        if self.config.host:
            command += ["--host", self.config.host]
        if self.config.config_dir:
            command += ["--config", self.config.config_dir]
        return command

    def check_available(self) -> dict:
        """Fail closed before generating candidates; never silently pull an image."""
        try:
            daemon = subprocess.run(self._base_command() + ["info", "--format", "{{.OSType}}"], capture_output=True, text=True, timeout=20)
            image = subprocess.run(self._base_command() + ["image", "inspect", self.config.image, "--format", "{{.Id}}"], capture_output=True, text=True, timeout=20)
        except (OSError, subprocess.TimeoutExpired):
            raise SandboxUnavailable("Docker daemon is unavailable; candidate execution is disabled") from None
        if daemon.returncode != 0 or daemon.stdout.strip() != "linux":
            raise SandboxUnavailable("A running Linux Docker daemon is required")
        if image.returncode != 0:
            raise SandboxUnavailable("Configured Docker image is unavailable; pre-pull it before running")
        return {"runtime": "docker", "image": self.config.image, "image_id": image.stdout.strip()}

    def _command(self, candidate_path: Path, name: str, mode: str, func_name: str | None) -> list[str]:
        cfg = self.config
        return self._base_command() + [
            "run", "--rm", "--pull", "never", "--name", name, "--network", "none",
            "--read-only", "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
            "--pids-limit", str(cfg.pids_limit), "--memory", f"{cfg.memory_mb}m",
            "--memory-swap", f"{cfg.memory_mb}m", "--cpus", str(cfg.cpus),
            "--ulimit", "nofile=64:64", "--ulimit", "fsize=1048576:1048576",
            "--user", "65534:65534", "--tmpfs", "/tmp:rw,noexec,nosuid,size=32m",
            "--mount", f"type=bind,src={candidate_path.resolve()},dst=/candidate.py,readonly",
            "--env", "PYTHONDONTWRITEBYTECODE=1", "--env", "OMP_NUM_THREADS=1",
            "--log-driver", "none", "-i", cfg.image, "python", "-I", "-c", _RUNNER,
            mode, func_name or "", str(cfg.timeout_seconds),
        ]

    def _stop_container(self, name: str) -> None:
        try:
            subprocess.run(self._base_command() + ["rm", "-f", name], stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL, timeout=10)
        except (OSError, subprocess.TimeoutExpired):
            pass

    def _run(self, candidate_path: Path, test: TestCase, func_name: str | None) -> _Execution:
        cfg = self.config
        name = "pot-eval-" + uuid.uuid4().hex
        try:
            proc = subprocess.Popen(self._command(candidate_path, name, test.testtype, func_name),
                                    stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except OSError:
            raise SandboxUnavailable("Docker executable is unavailable") from None
        chunks = {"stdout": bytearray(), "stderr": bytearray()}
        overflow = threading.Event()

        def read_stream(stream, key):
            try:
                while True:
                    chunk = stream.read(8192)
                    if not chunk:
                        break
                    room = cfg.max_output_bytes - len(chunks[key])
                    chunks[key].extend(chunk[:max(room, 0)])
                    if len(chunk) > room:
                        overflow.set()
                        proc.kill()
                        break
            finally:
                stream.close()

        def write_input():
            try:
                proc.stdin.write(test.input.encode("utf-8"))
                proc.stdin.close()
            except (BrokenPipeError, OSError, ValueError):
                pass

        readers = [threading.Thread(target=read_stream, args=(getattr(proc, key), key), daemon=True) for key in chunks]
        writer = threading.Thread(target=write_input, daemon=True)
        for thread in readers + [writer]:
            thread.start()
        status = "ok"
        try:
            proc.wait(timeout=cfg.timeout_seconds + cfg.startup_grace_seconds)
        except subprocess.TimeoutExpired:
            status = "timeout"
            proc.kill()
            self._stop_container(name)
            proc.wait(timeout=10)
        if overflow.is_set():
            status = "output_limit"
            self._stop_container(name)
        for thread in readers + [writer]:
            thread.join(timeout=2)
        if proc.returncode in (125, 126, 127):
            # Runtime setup errors are not candidate failures and never include
            # arbitrary Docker stderr in the exception.
            raise SandboxUnavailable("Docker could not start the isolated evaluator")
        if proc.returncode != 0 and status == "ok":
            status = "runtime_error"
        return _Execution(ok=status == "ok", stdout=chunks["stdout"].decode("utf-8", errors="replace"),
                          stderr=chunks["stderr"].decode("utf-8", errors="replace"), status=status)

    def _test(self, candidate_path: Path, test: TestCase, func_name: str | None) -> tuple[bool, _Execution]:
        execution = self._run(candidate_path, test, func_name)
        if not execution.ok:
            return False, execution
        if test.testtype == "python_assert":
            return execution.stdout.strip() == "POT_ASSERTIONS_COMPLETED", execution
        try:
            passed = compare_functional(_decode_prediction(json.loads(execution.stdout)), test.output) if test.testtype == "functional" else compare_stdio(execution.stdout, test.output)
        except (ValueError, TypeError, OverflowError, KeyError, RecursionError):
            passed = False
        return bool(passed), execution

    @contextmanager
    def _candidate(self, code: str):
        # Python 3.13's Windows mkdir(0700) ACL can exclude the app's sandbox
        # token. A normal inherited ACL is sufficient for this candidate-only dir.
        root = Path(self.config.work_dir or tempfile.gettempdir()).resolve()
        root.mkdir(parents=True, exist_ok=True)
        directory = root / ("pot-candidate-" + uuid.uuid4().hex)
        directory.mkdir(mode=0o755 if os.name == "nt" else 0o700)
        try:
            yield str(directory)
        finally:
            candidate = directory / "candidate.py"
            if candidate.exists():
                candidate.chmod(0o600)
                candidate.unlink()
            directory.rmdir()

    def _test_cases(self, problem, code, tests, visibility):
        """Ordered results. The subprocess executor overrides this with bounded concurrency."""
        with self._candidate(code) as directory:
            path = Path(directory) / "candidate.py"
            path.write_text(code, encoding="utf-8")
            path.chmod(0o444)
            for test in tests:
                yield self._test(path, test, problem.func_name)

    def evaluate_public(self, problem: PublicProblem, code: str,
                        tests: Sequence[TestCase] | None = None) -> PublicEvaluation:
        tests = tuple(problem.public_tests if tests is None else tests)
        feedback = []
        passed = 0
        for index, (test, (ok, execution)) in enumerate(zip(tests, self._test_cases(problem, code, tests, 'public'))):
            passed += ok
            if not ok:
                actual = execution.stdout[:2000]
                if test.testtype == 'functional' and execution.ok:
                    try:
                        actual = json.dumps(_decode_prediction(json.loads(execution.stdout)), ensure_ascii=False)[:2000]
                    except (ValueError, TypeError, KeyError):
                        pass
                from .benchmarks import public_test_text
                test_input = public_test_text(test) if test.testtype == "python_assert" else test.input
                feedback.append({"test_index": index, "status": execution.status if not execution.ok else "wrong_answer",
                                 "input": test_input[:2000], "expected": "assertion must pass" if test.testtype == "python_assert" else test.output[:2000],
                                 "actual": actual, "error": execution.stderr[:2000]})
        return PublicEvaluation(bool(tests) and passed == len(tests), passed, len(tests), tuple(feedback))

    def evaluate_private(self, problem: PublicProblem, code: str, store: PrivateTestStore) -> PrivateScore:
        """Private counts for reward and reporting; never return execution details."""
        tests = store._for_evaluation(problem.problem_id)
        passed = total = gate_passed = gate_total = 0
        for test, (ok, _) in zip(tests, self._test_cases(problem, code, tests, 'private')):
            if test.role == 'gate':
                gate_passed += ok
                gate_total += 1
            else:
                passed += ok
                total += 1
        return PrivateScore(passed, total, total > 0 and passed == total and gate_passed == gate_total,
                            gate_passed, gate_total)

    def evaluate_final(self, problem: PublicProblem, code: str, store: PrivateTestStore) -> PrivateScore:
        """Official solve requires every public AND private test to pass."""
        public = self.evaluate_public(problem, code)
        private = self.evaluate_private(problem, code, store)
        return PrivateScore(public.passed + private.passed, public.total + private.total,
                            (private.total > 0 or problem.benchmark == 'mbpp-sanitized')
                            and public.total + private.total > 0
                            and public.passed + private.passed == public.total + private.total
                            and private.gate_passed == private.gate_total,
                            private.gate_passed, private.gate_total)
