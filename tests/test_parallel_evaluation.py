"""Real subprocess equivalence, private-data isolation and shared CPU limits."""
import json
import multiprocessing
import os
from pathlib import Path
import time

import pytest

from pot.data import PublicProblem, PrivateTestStore, TestCase as Case
from pot.evaluator import DockerConfig
from pot.process_evaluator import ProcessEvaluator

pytestmark = pytest.mark.skipif(os.name != 'posix', reason='POSIX executor')


def evaluator(root, workers, store=None, limit=8, timeout=2):
    root.mkdir(parents=True, exist_ok=True)
    obj = ProcessEvaluator(DockerConfig(work_dir=str(root/'scratch'), eval_workers=workers,
        eval_global_workers=limit, eval_slot_dir=str(root/'slots'), timeout_seconds=timeout,
        memory_mb=1024), store)
    obj.timing_log = root/'timing.jsonl'
    return obj


@pytest.mark.parametrize('functional', [False, True])
def test_parallel_matches_serial_and_never_logs_private_details(tmp_path, functional):
    kind = 'functional' if functional else 'stdin'
    p = PublicProblem('test/1', '1', 'test', 'Double', 'Double', '', 'easy', '2025-01-01',
                      (Case('1', '2', kind), Case('2', '4', kind)), 'double' if functional else None)
    secret = 'PRIVATE_EXCEPTION_72891'
    store = PrivateTestStore({p.problem_id: (Case('3', '6', kind), Case('4', '8', kind), Case('5', '10', kind))})
    body = "import time\ntime.sleep((6-n)*0.01)\nif n==4: raise RuntimeError(%r)\nreturn n*2+int(n==2)" % secret
    code = 'class Solution:\n def double(self,n):\n  ' + body.replace('\n','\n  ') if functional else (
        "import time\nn=int(input())\ntime.sleep((6-n)*0.01)\nif n==4: raise RuntimeError(%r)\nprint(n*2+int(n==2))" % secret)
    serial = evaluator(tmp_path/'serial', 1, store)
    parallel = evaluator(tmp_path/'parallel', 4, store)
    expected, actual = serial.evaluate_candidate(p, code), parallel.evaluate_candidate(p, code)
    assert actual == expected
    assert actual.reward == pytest.approx(3/5) and not actual.all_passed
    assert [f['test_index'] for f in actual.public.feedback] == [1]
    rows = [json.loads(line) for line in parallel.timing_log.read_text().splitlines()]
    assert len(rows) == 5 and sum(r['visibility']=='private' for r in rows) == 3
    assert secret not in parallel.timing_log.read_text() and secret not in repr(actual)
    allowed = {'at','problem_id','code_sha256','visibility','test_index','passed','status',
               'execution_seconds','queue_wait_seconds','wall_seconds'}
    assert all(set(r) == allowed for r in rows)
    assert parallel.evaluate_candidate(p, code) is actual
    assert len(parallel.timing_log.read_text().splitlines()) == 5


def test_parallel_tests_do_not_share_scratch_files(tmp_path):
    tests = tuple(Case(str(i), str(i)) for i in range(8))
    p = PublicProblem('test/1','1','test','Echo','Echo','','easy','2025-01-01',tests)
    code = """import os,time
x=input()
fd=os.open('shared-name',os.O_CREAT|os.O_WRONLY|os.O_TRUNC,0o600)
os.write(fd,x.encode());os.close(fd)
time.sleep(0.1)
fd=os.open('shared-name',os.O_RDONLY)
print(os.read(fd,100).decode());os.close(fd)
"""
    obj = evaluator(tmp_path, 8)
    assert obj.evaluate_public(p, code).all_passed
    assert list((tmp_path/'scratch').iterdir()) == []


def test_timeouts_do_not_cancel_other_cases_or_produce_false_success(tmp_path):
    p = PublicProblem('test/1','1','test','Echo','Echo','','easy','2025-01-01',(Case('0','0'),))
    store = PrivateTestStore({p.problem_id:tuple(Case(str(i),str(i)) for i in range(1,6))})
    obj = evaluator(tmp_path, 4, store, timeout=0.5)
    result = obj.evaluate_candidate(p, 'x=int(input())\nwhile x==2: pass\nprint(x)')
    assert result.passed == 5 and result.total == 6 and not result.all_passed
    rows = [json.loads(line) for line in obj.timing_log.read_text().splitlines()]
    assert len(rows) == 6 and sum(r['status']=='timeout' for r in rows) == 1
    assert obj.performance(p.problem_id)['test_timeouts'] == 1


def _exercise_slots(root, events):
    from concurrent.futures import ThreadPoolExecutor
    obj = ProcessEvaluator(DockerConfig(eval_global_workers=2, eval_slot_dir=root))
    def task(_):
        with obj._cpu_slot():
            events.put((time.monotonic(), 1))
            time.sleep(0.1)
            events.put((time.monotonic(), -1))
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(task, range(4)))


def test_global_cpu_limit_applies_across_workers(tmp_path):
    ctx = multiprocessing.get_context('spawn')
    events = ctx.Queue()
    processes = [ctx.Process(target=_exercise_slots, args=(str(tmp_path/'slots'), events)) for _ in range(2)]
    for p in processes: p.start()
    history = [events.get(timeout=20) for _ in range(16)]
    for p in processes:
        p.join(20)
        assert p.exitcode == 0
    active, maximum = 0, 0
    for _, delta in sorted(history):
        active += delta
        maximum = max(maximum, active)
    assert active == 0 and maximum == 2
