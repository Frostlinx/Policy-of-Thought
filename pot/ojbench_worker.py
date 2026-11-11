"""Separate official DMOJ judge process; JSON protocol contains no hidden I/O.

Requires the OJBench-pinned DMOJ installation. No upstream source is vendored.
Every official case is graded, including interactive cases, without short circuit.
"""
from __future__ import annotations
from contextlib import contextmanager, redirect_stdout, redirect_stderr
from concurrent.futures import ProcessPoolExecutor
import fcntl
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
import time

_settings = None


def initialize(settings, self_test=False):
    global _settings
    _settings = settings
    import dmoj
    from .ojbench import DMOJ_CORE_SHA256, sha256
    package_root = Path(dmoj.__file__).resolve().parent.parent
    if any(sha256(package_root / name) != expected for name, expected in DMOJ_CORE_SHA256.items()):
        raise RuntimeError('Installed DMOJ grading code differs from the OJBench-pinned revision')
    from dmoj import judgeenv
    from dmoj.executors import load_executors, executors
    from dmoj.contrib import load_contrib_modules
    judgeenv._problem_dirs_cache = [Path(settings['judge_root'])]
    judgeenv.only_executors = {'CPP17', 'PYPY3'}
    judgeenv.skip_self_test = not self_test
    # Allow only the interpreter installation, never its parent's data directory.
    pypy_prefix = Path(settings['pypy']).resolve().parent.parent
    extra = [] if str(pypy_prefix) == '/usr' else [{'recursive_dir': str(pypy_prefix)}]
    judgeenv.env.update({'runtime': {'g++17': settings['gpp'], 'pypy3': settings['pypy']},
                         'extra_fs': {'PYPY3': extra}})
    with open(os.devnull, 'w') as quiet, redirect_stdout(quiet), redirect_stderr(quiet):
        load_executors()
        load_contrib_modules()
    if not {'CPP17', 'PYPY3'} <= executors.keys():
        raise RuntimeError('Required official judge executors did not initialize')


@contextmanager
def cpu_slot():
    root = Path(_settings['slot_dir'])
    root.mkdir(parents=True, exist_ok=True)
    count = _settings['global_workers']
    while True:
        for i in range(count):
            f = (root / f'{(i + os.getpid()) % count:04d}.lock').open('a')
            try:
                fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                f.close()
                continue
            try:
                yield
            finally:
                f.close()
            return
        time.sleep(.02)


def grade_chunk(job):
    from dmoj.problem import Problem
    from dmoj.error import CompileError
    pid, source, indices, public, count = job
    problem = Problem(problem_id=pid, time_limit=_settings['timeout'],
                      memory_limit=_settings['memory_mb'] * 1024, meta={})
    # Official per-problem limits/checkers take precedence. They are included in
    # the pinned dataset identity. PyPy runs the complete program, without LCB's
    # AST wrapping or preimports.
    cases = problem.cases()
    if len(cases) != count:
        raise RuntimeError('Official case count differs from the prepared dataset')
    if any(hasattr(c, 'batched_cases') for c in cases):
        raise RuntimeError('Unsupported batched case')
    try:
        with open(Path(_settings['slot_dir']) / 'oj-compile.lock', 'a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            grader = problem.grader_class(judge=None, problem=problem, language='PYPY3', source=source)
    except CompileError:
        return [{'index': i, 'passed': False, 'status': 'CE', 'seconds': 0., 'queue_seconds': 0.} for i in indices]
    rows = []
    for i in indices:
        queued = time.monotonic()
        with cpu_slot():
            started = time.monotonic()
            result = grader.grade(cases[i])
            elapsed = time.monotonic() - started
        code = result.readable_codes()[0]
        if code in {'IE', 'SC'}:
            raise RuntimeError('Official judge infrastructure failure')
        row = {'index': i, 'passed': result.result_flag == 0, 'status': code,
               'seconds': elapsed, 'queue_seconds': started - queued}
        if public:
            def text(value):
                return (value.decode('utf-8', 'replace') if isinstance(value, bytes) else str(value or ''))[:2000]
            row.update(actual=text(result.output), error=text(result.feedback))
        # Hidden filenames, inputs, expected/actual outputs, stderr, checker
        # feedback and interactive transcripts are never put on the IPC channel.
        rows.append(row)
    return rows


def serve():
    settings = json.loads(sys.stdin.readline())
    Path(settings['slot_dir']).mkdir(parents=True, exist_ok=True)
    initialize(settings, self_test=True)
    from dmoj.executors import executors
    versions = {k: executors[k].Executor.get_runtime_versions() for k in ('PYPY3', 'CPP17')}
    import importlib.metadata
    print(json.dumps({'ready': True, 'versions': versions, 'dmoj_version': importlib.metadata.version('dmoj')}), flush=True)
    count = min(settings['workers'], settings['global_workers'])
    with ProcessPoolExecutor(max_workers=count, mp_context=mp.get_context('spawn'),
                             initializer=initialize, initargs=(settings,)) as pool:
        for line in sys.stdin:
            request = json.loads(line)
            if request.get('action') == 'close':
                break
            n = request['count']
            jobs = [(request['problem_id'], request['source'], list(range(i, n, min(n, count))), request['public'], n)
                    for i in range(min(n, count))]
            try:
                rows = [row for chunk in pool.map(grade_chunk, jobs) for row in chunk]
                print(json.dumps({'results': sorted(rows, key=lambda r: r['index'])}), flush=True)
            except Exception as exc:
                # Do not stringify judge exceptions: they may contain test data.
                print(json.dumps({'error': type(exc).__name__}), flush=True)


if __name__ == '__main__':
    try:
        serve()
    except Exception as exc:
        print(json.dumps({'error': type(exc).__name__}), flush=True)
        raise SystemExit(1)
