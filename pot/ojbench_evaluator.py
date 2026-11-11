"""PoT interface to the separate pinned OJBench/DMOJ judge service."""
from __future__ import annotations
import atexit
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import select
import subprocess
import time

from .evaluator import SandboxUnavailable, _Execution
from .ojbench import DMOJ_REVISION, OJ_CODE_REVISION, OJPrivateStore
from .process_evaluator import ProcessEvaluator


class OJBenchEvaluator(ProcessEvaluator):
    def __init__(self, config, private_store):
        if not isinstance(private_store, OJPrivateStore):
            raise ValueError('OJBench requires prepared official judge data')
        super().__init__(config, private_store)
        self._service = None
        atexit.register(self.close)

    def _reply(self, timeout):
        ready, _, _ = select.select([self._service.stdout], [], [], timeout)
        if not ready:
            self.close()
            raise SandboxUnavailable('OJBench judge service timed out')
        try:
            reply = json.loads(self._service.stdout.readline())
        except (ValueError, OSError):
            raise SandboxUnavailable('OJBench judge service returned an invalid response') from None
        if 'error' in reply:
            raise SandboxUnavailable('OJBench judge failed; no candidate score was recorded')
        return reply

    def check_available(self):
        cfg = self.config
        if os.name != 'posix' or not cfg.ojbench_python or not cfg.ojbench_pypy:
            raise SandboxUnavailable('OJBench requires Linux, --ojbench-python and --ojbench-pypy')
        self.private_store.verify()
        env = {k: v for k, v in os.environ.items() if k not in {'HF_TOKEN', 'HUGGING_FACE_HUB_TOKEN'}}
        env['PYTHONPATH'] = str(Path(__file__).resolve().parent.parent)
        self._service = subprocess.Popen([cfg.ojbench_python, '-u', '-m', 'pot.ojbench_worker'],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            text=True, encoding='utf-8', env=env, start_new_session=True)
        settings = {'judge_root': str(self.private_store.judge_root), 'pypy': cfg.ojbench_pypy,
            'gpp': cfg.ojbench_gpp, 'timeout': cfg.timeout_seconds, 'memory_mb': cfg.memory_mb,
            'workers': cfg.eval_workers, 'global_workers': cfg.eval_global_workers,
            'slot_dir': cfg.eval_slot_dir or str(Path(cfg.work_dir) / 'cpu-slots')}
        self._service.stdin.write(json.dumps(settings) + '\n')
        self._service.stdin.flush()
        reply = self._reply(120)
        if reply.get('ready') is not True:
            raise SandboxUnavailable('OJBench judge did not initialize')
        return {'runtime': 'ojbench-dmoj', 'dmoj_revision': DMOJ_REVISION,
                'ojbench_reference_revision': OJ_CODE_REVISION, 'versions': reply['versions'],
                'dmoj_version': reply['dmoj_version'], 'dmoj_core_hashes_verified': True,
                'output_limits': 'official per-problem config/DMOJ defaults; generic max_output_bytes is not applied',
                'timeout_seconds': cfg.timeout_seconds, 'memory_mb': cfg.memory_mb,
                'eval_workers': cfg.eval_workers, 'eval_global_workers': cfg.eval_global_workers,
                'private_diagnostics': 'discarded_in_judge_process'}

    def _test_cases(self, problem, code, tests, visibility):
        tests = tuple(tests)
        if not tests:
            return
        if self._service is None:
            self.check_available()
        public = visibility == 'public'
        pid = problem.question_id + ('-public' if public else '')
        self._service.stdin.write(json.dumps({'problem_id': pid, 'source': code,
            'count': len(tests), 'public': public}) + '\n')
        self._service.stdin.flush()
        reply = self._reply(max(180, len(tests) * (3 * self.config.timeout_seconds + 15)))
        rows = reply.get('results', [])
        if [r.get('index') for r in rows] != list(range(len(tests))):
            raise SandboxUnavailable('Incomplete official judge results')
        metrics = self._performance.setdefault(problem.problem_id, dict(test_executions=0,
                    test_timeouts=0, test_execution_seconds=0., test_queue_wait_seconds=0.))
        digest = hashlib.sha256(code.encode()).hexdigest()
        for row in rows:
            metrics['test_executions'] += 1
            metrics['test_timeouts'] += row['status'] == 'TLE'
            metrics['test_execution_seconds'] += row['seconds']
            metrics['test_queue_wait_seconds'] += row['queue_seconds']
            if self.timing_log:
                safe = {'at': datetime.now(timezone.utc).isoformat(), 'problem_id': problem.problem_id,
                    'code_sha256': digest, 'visibility': visibility, 'test_index': row['index'],
                    'passed': row['passed'], 'status': row['status'], 'execution_seconds': row['seconds'],
                    'queue_wait_seconds': row['queue_seconds']}
                with Path(self.timing_log).open('a', encoding='utf-8') as out:
                    out.write(json.dumps(safe) + '\n')
            yield row['passed'], _Execution(row['status'] in {'AC', 'WA'},
                row.get('actual', '') if public else '', row.get('error', '') if public else '',
                row['status'], row['seconds'], row['queue_seconds'])

    def close(self):
        proc, self._service = self._service, None
        if proc is None:
            return
        try:
            proc.stdin.write('{"action":"close"}\n')
            proc.stdin.flush()
            proc.wait(timeout=15)
        except (OSError, ValueError, subprocess.TimeoutExpired):
            self._kill_group(proc)
            proc.wait(timeout=10)
        finally:
            for stream in (proc.stdin, proc.stdout):
                stream.close()
