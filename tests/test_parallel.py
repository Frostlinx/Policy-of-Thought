"""Queue ownership and recovery must not duplicate completed problems."""
import json
import os
import multiprocessing
import pytest

pytest.importorskip('fcntl')
from pot.parallel import Queue, refresh_jobs, DeviceLease, eligible_gpus, worker_name, worker_port


def _claim(root, gpu, slot=0):
    Queue(root).claim(gpu, slot)


def test_concurrent_workers_claim_different_questions(tmp_path):
    (tmp_path/'queue.json').write_text(json.dumps({'indices':list(range(8)),'jobs':[]}))
    ctx=multiprocessing.get_context('spawn')
    children=[ctx.Process(target=_claim,args=(str(tmp_path),gpu)) for gpu in range(4)]
    for p in children:p.start()
    for p in children:p.join(20);assert p.exitcode==0
    jobs=json.loads((tmp_path/'queue.json').read_text())['jobs']
    assert len({x['index'] for x in jobs})==4


def test_completed_records_survive_worker_exit_and_are_not_reclaimed(tmp_path):
    (tmp_path/'queue.json').write_text(json.dumps({'indices':[0,1],'jobs':[]}))
    q=Queue(tmp_path);first=q.claim(0)
    directory=__import__('pathlib').Path(first['directory']);directory.mkdir(parents=True)
    (directory/'process.jsonl').write_text(json.dumps({'status':'ok','problem_index':0})+'\n')
    state=refresh_jobs(tmp_path)
    assert state['jobs'][0]['status']=='completed'
    assert q.claim(1)['index']==1


def test_failed_question_retries_are_bounded(tmp_path):
    (tmp_path/'queue.json').write_text(json.dumps({'indices':[0,1],'jobs':[]}))
    q=Queue(tmp_path)
    for _ in range(3):
        j=q.claim(0);assert j['index']==0;q.finish(j,2)
    assert q.claim(0)['index']==1


def test_same_gpu_slots_have_independent_questions_and_resources(tmp_path):
    (tmp_path/'queue.json').write_text(json.dumps({'indices': [0, 1], 'jobs': []}))
    ctx = multiprocessing.get_context('spawn')
    children = [ctx.Process(target=_claim, args=(str(tmp_path), 1, slot)) for slot in range(2)]
    for child in children:
        child.start()
    for child in children:
        child.join(20)
        assert child.exitcode == 0
    jobs = json.loads((tmp_path/'queue.json').read_text())['jobs']
    assert {j['index'] for j in jobs} == {0, 1}
    assert {(j['gpu'], j['slot']) for j in jobs} == {(1, 0), (1, 1)}
    assert len({worker_name(j['gpu'], j['slot']) for j in jobs}) == 2
    assert len({worker_port(19700, j['gpu'], j['slot']) for j in jobs}) == 2


def test_admission_shares_only_owned_gpu_and_reserves_free_memory():
    devices = {0: dict(free=50000, pids=[999]), 1: dict(free=40000, pids=[10, 11]),
               2: dict(free=30000, pids=[]), 3: dict(free=80000, pids=[]),
               4: dict(free=40000, pids=[10, 999])}
    active = [dict(pid=10, server_pid=11)]
    assert eligible_gpus(devices, range(5), 36000, active,
                         owned=lambda pid, roots: pid in roots) == {1, 3}
    # Explicit permission never overrides free-memory or allowed-GPU limits.
    assert eligible_gpus(devices, range(5), 36000, active,
                         owned=lambda pid, roots: pid in roots, shared=[0, 2]) == {0, 1, 3}
    assert eligible_gpus(devices, [1, 3], 36000, active,
                         owned=lambda pid, roots: pid in roots, shared=[0, 4]) == {1, 3}


def test_nested_training_lease_releases_once_even_on_exception(tmp_path):
    import fcntl
    releases = []
    path = tmp_path/'gpu.lock'
    lease = DeviceLease(path, lambda: releases.append('released'))
    with pytest.raises(RuntimeError):
        with lease.hold():
            with lease.hold():
                assert not releases
                raise RuntimeError('training failed')
    assert releases == ['released']
    with path.open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)


def _locked_increment(root):
    import time
    from pathlib import Path
    root = Path(root)
    with DeviceLease(root/'gpu.lock').hold():
        counter = root/'counter'
        value = int(counter.read_text())
        time.sleep(0.05)
        counter.write_text(str(value + 1))


def test_training_lease_serializes_different_processes(tmp_path):
    (tmp_path/'counter').write_text('0')
    ctx = multiprocessing.get_context('spawn')
    children = [ctx.Process(target=_locked_increment, args=(str(tmp_path),)) for _ in range(4)]
    for child in children:
        child.start()
    for child in children:
        child.join(20)
        assert child.exitcode == 0
    assert (tmp_path/'counter').read_text() == '4'
