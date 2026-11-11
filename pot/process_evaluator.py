"""Fast local POSIX subprocess evaluator, explicitly not a security sandbox.

Candidates run as the evaluator's OS user. A clean environment, temporary cwd,
resource limits and process-group cleanup reduce accidents and overhead; they
do not prevent hostile filesystem access, system calls, or escaping a process
group. Use DockerEvaluator when operating-system isolation is required.
Private-test secrecy in the model/logging pipeline is inherited unchanged from
DockerEvaluator: LCB expected outputs remain in the parent; assertion benchmarks
execute the test program in the child. Private evaluation exposes counts only.
"""
from __future__ import annotations

from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import uuid

from .evaluator import DockerConfig, DockerEvaluator, SandboxUnavailable, _Execution, _RUNNER


class ProcessEvaluator(DockerEvaluator):
    def __init__(self, config: DockerConfig | None = None, private_store=None):
        super().__init__(config or DockerConfig(memory_mb=1024), private_store)
        self.timing_log = None
        self._timing_lock = threading.Lock()
        self._performance = {}
        self._slot_cursor = 0

    @contextmanager
    def _cpu_slot(self):
        """Cross-process admission. A child retains the lock if its parent exits."""
        import fcntl
        directory = Path(self.config.eval_slot_dir or
                         str(Path(tempfile.gettempdir()) / f'pot-cpu-slots-{os.getuid()}'))
        directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        limit = self.config.eval_global_workers
        with self._timing_lock:
            start = self._slot_cursor % limit
            self._slot_cursor += 1
        while True:
            for offset in range(limit):
                handle = (directory / f'{(start + offset) % limit:04d}.lock').open('a')
                try:
                    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    handle.close()
                    continue
                try:
                    yield handle
                finally:
                    fcntl.flock(handle, fcntl.LOCK_UN)
                    handle.close()
                return
            time.sleep(0.02)

    def _test_cases(self, problem, code, tests, visibility):
        tests = tuple(tests)
        if not tests:
            return
        digest = hashlib.sha256(code.encode()).hexdigest()
        def execute(item):
            index, test = item
            started = time.monotonic()
            # Scratch files and Python state cannot race between parallel tests.
            with self._candidate(code) as directory:
                path = Path(directory) / 'candidate.py'
                path.write_text(code, encoding='utf-8')
                path.chmod(0o444)
                ok, execution = self._test(path, test, problem.func_name)
            row = {'at': datetime.now(timezone.utc).isoformat(), 'problem_id': problem.problem_id,
                'code_sha256': digest, 'visibility': visibility, 'test_index': index,
                'passed': bool(ok), 'status': execution.status if not execution.ok else 'passed' if ok else 'wrong_answer',
                'execution_seconds': execution.elapsed_seconds,
                'queue_wait_seconds': execution.queue_wait_seconds, 'wall_seconds': time.monotonic() - started}
            with self._timing_lock:
                metrics = self._performance.setdefault(problem.problem_id, dict(test_executions=0,
                    test_timeouts=0, test_execution_seconds=0.0, test_queue_wait_seconds=0.0))
                metrics['test_executions'] += 1
                metrics['test_timeouts'] += execution.status == 'timeout'
                metrics['test_execution_seconds'] += execution.elapsed_seconds
                metrics['test_queue_wait_seconds'] += execution.queue_wait_seconds
                if self.timing_log:
                    # Explicit allowlist above: never write private I/O or exception text.
                    with Path(self.timing_log).open('a', encoding='utf-8') as stream:
                        stream.write(json.dumps(row, ensure_ascii=False, separators=(',', ':')) + '\n')
            return ok, execution
        count = min(self.config.eval_workers, self.config.eval_global_workers, len(tests))
        if count == 1:
            yield from map(execute, enumerate(tests))
        else:
            with ThreadPoolExecutor(max_workers=count, thread_name_prefix='pot-case') as pool:
                # Completion can be out of order; feedback/test indices stay deterministic.
                yield from pool.map(execute, enumerate(tests))

    def performance(self, problem_id):
        with self._timing_lock:
            return {**self._performance.get(problem_id, {}),
                    'eval_workers': self.config.eval_workers,
                    'eval_global_workers': self.config.eval_global_workers}

    def check_available(self) -> dict:
        if os.name != "posix":
            raise SandboxUnavailable("Subprocess resource limits require a POSIX host")
        try:
            import resource
        except ImportError:
            raise SandboxUnavailable("POSIX resource module is unavailable") from None
        return {"runtime": "subprocess", "security_sandbox": False,
                "python": sys.executable, "python_version": sys.version.split()[0],
                "memory_mb": self.config.memory_mb, "timeout_seconds": self.config.timeout_seconds,
                "max_output_bytes": self.config.max_output_bytes,
                "eval_workers": self.config.eval_workers,
                "eval_global_workers": self.config.eval_global_workers,
                "resource_limits": ["RLIMIT_AS", "RLIMIT_CPU", "RLIMIT_NOFILE", "RLIMIT_FSIZE",
                                    "RLIMIT_CORE", "RLIMIT_NPROC"],
                "limitations": "Same OS user; filesystem and system calls are not isolated"}

    @contextmanager
    def _candidate(self, code: str):
        root = Path(self.config.work_dir or tempfile.gettempdir()).resolve()
        root.mkdir(parents=True, exist_ok=True)
        directory = root / ("pot-process-" + uuid.uuid4().hex)
        directory.mkdir(mode=0o700)
        try:
            yield str(directory)
        finally:
            if directory.exists():
                # Remove only this evaluator-owned directory, including scratch
                # files produced by ordinary candidate programs. Never follow a
                # candidate-replaced directory symlink outside the scratch root.
                if directory.is_symlink() or directory.resolve().parent != root:
                    raise RuntimeError("Candidate directory changed; refusing unsafe cleanup")
                shutil.rmtree(directory)

    def _environment(self, candidate_path: Path) -> dict[str, str]:
        directory = str(candidate_path.parent.resolve())
        return {"PATH": os.defpath, "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8",
                "HOME": directory, "TMPDIR": directory, "TMP": directory, "TEMP": directory,
                "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}

    def _bootstrap(self, candidate_path: Path) -> str:
        cfg = self.config
        # repr handles arbitrary quotes/backslashes in an explicitly configured
        # scratch path; replace complete Python string literals, not raw paths.
        source = _RUNNER.replace("'/candidate.py'", repr(str(candidate_path.resolve())))
        # Exit immediately on the trusted wall timer. Candidate stdin wrapper
        # catches SystemExit, so raising it would incorrectly accept a timeout.
        source = "import os\n" + source.replace("raise TimeoutError('execution timeout')", "os._exit(124)")
        limits = {
            "RLIMIT_AS": (cfg.memory_mb * 1024 * 1024, cfg.memory_mb * 1024 * 1024),
            "RLIMIT_CPU": (max(1, math.ceil(cfg.timeout_seconds)), max(2, math.ceil(cfg.timeout_seconds) + 1)),
            "RLIMIT_NOFILE": (64, 64), "RLIMIT_FSIZE": (cfg.max_output_bytes, cfg.max_output_bytes),
            "RLIMIT_CORE": (0, 0), "RLIMIT_NPROC": (cfg.pids_limit, cfg.pids_limit),
        }
        # Setting limits after exec avoids preexec_fn deadlocks when the parent
        # has heartbeat and pipe-reader threads. The code is stdlib-only.
        return ("import os, resource\n"
                f"limits = {limits!r}\n"
                "try:\n"
                "    for name, values in limits.items():\n"
                "        if hasattr(resource, name):\n"
                "            resource.setrlimit(getattr(resource, name), values)\n"
                "except BaseException:\n"
                "    os._exit(125)\n"
                f"exec(compile({source!r}, '<pot-local-evaluator>', 'exec'))\n")

    @staticmethod
    def _kill_group(proc: subprocess.Popen) -> None:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass

    def _run(self, candidate_path, test, func_name):
        if os.name != "posix":
            raise SandboxUnavailable("Subprocess evaluation requires a POSIX host")
        queued = time.monotonic()
        with self._cpu_slot() as slot:
            started = time.monotonic()
            execution = self._run_admitted(candidate_path, test, func_name, slot.fileno())
            return replace(execution, elapsed_seconds=time.monotonic() - started,
                           queue_wait_seconds=started - queued)

    def _run_admitted(self, candidate_path, test, func_name, slot_fd):
        cfg = self.config
        candidate_path = Path(candidate_path).resolve()
        command = [sys.executable, "-I", "-S", "-c", self._bootstrap(candidate_path),
                   test.testtype, func_name or "", str(cfg.timeout_seconds)]
        try:
            proc = subprocess.Popen(command, cwd=candidate_path.parent,
                env=self._environment(candidate_path), stdin=subprocess.PIPE,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True,
                close_fds=True, pass_fds=(slot_fd,))
        except OSError:
            raise SandboxUnavailable("Could not launch the local Python evaluator") from None
        chunks = {"stdout": bytearray(), "stderr": bytearray()}
        overflow = threading.Event()

        def read_stream(stream, key):
            try:
                while True:
                    chunk = stream.read(8192)
                    if not chunk:
                        break
                    room = cfg.max_output_bytes - len(chunks[key])
                    chunks[key].extend(chunk[:max(0, room)])
                    if len(chunk) > room:
                        overflow.set()
                        self._kill_group(proc)
                        break
            finally:
                stream.close()

        def write_input():
            try:
                proc.stdin.write(test.input.encode("utf-8"))
                proc.stdin.close()
            except (BrokenPipeError, OSError, ValueError):
                pass

        readers = [threading.Thread(target=read_stream, args=(getattr(proc, key), key), daemon=True)
                   for key in chunks]
        writer = threading.Thread(target=write_input, daemon=True)
        status = "ok"
        try:
            for thread in readers + [writer]:
                thread.start()
            try:
                proc.wait(timeout=cfg.timeout_seconds + min(1.0, cfg.startup_grace_seconds))
            except subprocess.TimeoutExpired:
                status = "timeout"
                self._kill_group(proc)
                proc.wait(timeout=5)
        finally:
            # Also reap descendants holding output pipes after the main
            # candidate returns normally. An intentional setsid escape is not
            # contained, one reason this executor is not a security sandbox.
            self._kill_group(proc)
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
            for thread in readers + [writer]:
                if thread.is_alive():
                    thread.join(timeout=2)
        if proc.returncode == 125:
            raise SandboxUnavailable("Could not apply local evaluator resource limits")
        if overflow.is_set():
            status = "output_limit"
        elif proc.returncode in (124, -signal.SIGXCPU):
            status = "timeout"
        elif proc.returncode != 0 and status == "ok":
            status = "runtime_error"
        return _Execution(ok=status == "ok",
            stdout=chunks["stdout"].decode("utf-8", errors="replace"),
            stderr=chunks["stderr"].decode("utf-8", errors="replace"), status=status)

