"""Persistent workers sharing a resumable queue of LCB problems.

Run inside tmux. Each worker keeps its inference server and trainer between
problems. Workers claim disjoint questions under a file lock; completed
results are never overwritten. Newly idle GPUs join automatically.
"""
from __future__ import annotations

import copy
from contextlib import contextmanager
from dataclasses import asdict
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.request

from . import run as runner
from .data import load_dataset


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    runner._atomic_json(Path(path), value)


def alive(pid):
    try:
        os.kill(pid, 0)
        return Path(f'/proc/{pid}/stat').read_text().split(') ', 1)[1][0] != 'Z'
    except (OSError, ProcessLookupError):
        return False


class Queue:
    def __init__(self, root):
        self.root = Path(root)

    def update(self, callback):
        with (self.root / 'queue.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            state = read(self.root / 'queue.json')
            result = callback(state)
            write(self.root / 'queue.json', state)
            return result

    def claim(self, gpu, slot=0):
        def take(state):
            completed = {x['index'] for x in state['jobs'] if x['status'] == 'completed'}
            occupied = {x['index'] for x in state['jobs'] if x['status'] == 'running'}
            failures = {}
            for job in state['jobs']:
                if job['status'] == 'failed':
                    failures[job['index']] = failures.get(job['index'], 0) + 1
            pending = [i for i in state['indices'] if i not in completed | occupied
                       and failures.get(i, 0) < 3]
            if not pending:
                return None
            number = len(state['jobs']) + 1
            job = {'id': number, 'index': pending[0], 'gpu': gpu, 'slot': slot, 'pid': os.getpid(),
                   'status': 'running', 'started_at': runner._now(),
                   'directory': str(self.root / 'jobs' / f'{number:05d}-q{pending[0]:05d}')}
            state['jobs'].append(job)
            return job
        return self.update(take)

    def finish(self, job, code):
        def finish(state):
            current = next(j for j in state['jobs'] if j['id'] == job['id'])
            current.update(status='completed' if code == 0 else 'failed',
                           exit_code=code, finished_at=runner._now())
        self.update(finish)


def refresh_jobs(root):
    def refresh(state):
        for job in state['jobs']:
            if job['status'] != 'running':
                continue
            p = Path(job['directory']) / 'process.jsonl'
            rows = [json.loads(line) for line in p.read_bytes().splitlines(keepends=True)
                    if line.endswith(b'\n')] if p.exists() else []
            if rows and rows[-1]['status'] == 'ok':
                job.update(status='completed', exit_code=0, finished_at=runner._now())
            elif not alive(job['pid']):
                job.update(status='failed', exit_code=99, finished_at=runner._now())
        return state
    return Queue(root).update(refresh)


def publish(root, state, metadata):
    records, failures = {}, []
    for job in state['jobs']:
        directory = Path(job['directory'])
        path = directory / 'process.jsonl'
        if not path.exists():
            continue
        data = path.read_bytes()
        for line in data.splitlines(keepends=True):
            if not line.endswith(b'\n'):
                continue
            row = json.loads(line)
            if row['status'] != 'ok':
                failures.append(row)
                continue
            index = row['problem_index']
            if index in records:
                raise ValueError('Duplicate completed question')
            row['code_file'] = str((directory / row['code_file']).relative_to(root))
            row['source_run'] = str(directory.relative_to(root))
            records[index] = row
    rows = [records[k] for k in sorted(records)]
    text = ''.join(runner._json(row) + '\n' for row in rows)
    for name in ['process.json', 'process.jsonl']:
        p = root / name
        temp = p.with_suffix(p.suffix + '.tmp')
        temp.write_text(text)
        os.replace(temp, p)
    active = []
    for path in (root / 'workers').glob('*/state.json'):
        w = read(path)
        if w['status'] != 'exited' and alive(w['pid']):
            if w.get('job_directory'):
                p = Path(w['job_directory']) / 'progress.json'
                if p.exists():
                    w['progress'] = read(p)
            active.append(w)
    total = len(state['indices'])
    exhausted = all(sum(j['index'] == i and j['status'] == 'failed' for j in state['jobs']) >= 3
                    for i in state['indices'] if i not in records)
    status = ('completed' if len(rows) == total else 'running' if active else
              'needs_attention' if exhausted else 'waiting_for_gpu')
    summary = runner._summary(rows, total, metadata['dataset_count'], status,
                              metadata['started_at'], metadata['config_sha256'], metadata['dataset_sha256'])
    summary.update(evaluation_protocol='all_tests_reward', reward_tests='all',
                   benchmark_protocol=runner.benchmark_protocol(metadata['config']['dataset'].get('selection', 'v6')),
                   stop_rule='all_tests_passed', heldout_tests_valid=False,
                   feedback_mode=metadata['config']['search'].get('feedback', 'public'),
                   worker_failure_attempts=sum(j['status'] == 'failed' for j in state['jobs']),
                   elapsed_seconds_definition='sum of problem times; parallel times overlap')
    write(root / 'summary.json', summary)
    runner._append(root / 'summary.jsonl', summary)
    write(root / 'progress.json', {'status': status, 'heartbeat_at': runner._now(),
        'completed_count': len(rows), 'requested_count': total, 'active_workers': active})
    write(root / 'execution_failures.json', {'attempts': failures})
    return status


def gpu_inventory():
    result = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.free,memory.total',
        '--format=csv,noheader,nounits'], text=True, timeout=15)
    devices = {int(index): dict(uuid=uuid, free=int(free), total=int(total), pids=[])
        for index, uuid, free, total in ([s.strip() for s in line.split(',')]
        for line in result.splitlines())}
    apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
        '--format=csv,noheader,nounits'], text=True, timeout=15)
    by_uuid = {v['uuid']: v for v in devices.values()}
    for line in apps.splitlines():
        uuid, pid = [s.strip() for s in line.split(',')]
        if uuid in by_uuid:
            by_uuid[uuid]['pids'].append(int(pid))
    return devices


def belongs_to_worker(pid, roots):
    """Allow sharing only with this run's workers and their descendants."""
    for _ in range(64):
        if pid in roots:
            return True
        if pid <= 1:
            return False
        try:
            pid = int(Path(f'/proc/{pid}/stat').read_text().split(') ', 1)[1].split()[1])
        except (OSError, ValueError):
            return False
    return False


def eligible_gpus(devices, allowed, minimum, active, owned=belongs_to_worker, *, shared=()):
    roots = {pid for w in active for pid in (w['pid'], w.get('server_pid')) if pid}
    return {gpu for gpu, device in devices.items() if gpu in allowed
            and device['free'] >= minimum
            and (gpu in shared or all(owned(pid, roots) for pid in device['pids']))}


class DeviceLease:
    """Reentrant, process-shared lock for HF training peaks and server startup."""
    def __init__(self, path, release=None):
        self.path, self.release = Path(path), release
        self.depth = 0
        self.thread_lock = threading.RLock()

    @contextmanager
    def hold(self):
        with self.thread_lock:
            outer = self.depth == 0
            if outer:
                self.handle = self.path.open('a')
                fcntl.flock(self.handle, fcntl.LOCK_EX)
            self.depth += 1
            try:
                yield
            finally:
                self.depth -= 1
                if outer:
                    try:
                        if self.release:
                            self.release()
                    finally:
                        fcntl.flock(self.handle, fcntl.LOCK_UN)
                        self.handle.close()


class SharedGPUBackend(runner.Backend):
    """Keep independent policies; serialize only HF scoring/training on a GPU."""
    def __init__(self, config, lock_path, memory_path):
        self.memory_path = Path(memory_path)
        self.device_lease = DeviceLease(lock_path, self._release_memory)
        with self.device_lease.hold():
            super().__init__(config)

    def _release_memory(self):
        torch = getattr(self, 'torch', None)
        if torch is not None and torch.cuda.is_initialized():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            write(self.memory_path, {'updated_at': runner._now(), 'pid': os.getpid(),
                'allocated_mib': torch.cuda.memory_allocated() / 2**20,
                'reserved_mib': torch.cuda.memory_reserved() / 2**20,
                'peak_allocated_mib': torch.cuda.max_memory_allocated() / 2**20,
                'peak_reserved_mib': torch.cuda.max_memory_reserved() / 2**20})

    def reset_instance(self, *args, **kwargs):
        with self.device_lease.hold():
            return super().reset_instance(*args, **kwargs)

    def _logprobs(self, *args, **kwargs):
        # Preserve exact saved tensors on host RAM instead of retaining them on GPU.
        # This changes storage only; trajectory tokens and loss remain unchanged.
        with self.device_lease.hold(), self.torch.autograd.graph.save_on_cpu(pin_memory=True):
            return super()._logprobs(*args, **kwargs)

    def update(self, *args, **kwargs):
        with self.device_lease.hold():
            return super().update(*args, **kwargs)


def worker_name(gpu, slot):
    return f'gpu{gpu}-slot{slot}'


def worker_port(base, gpu, slot):
    return base + gpu * 32 + slot


def server_health(port):
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(f'http://127.0.0.1:{port}/health', timeout=2) as response:
            return response.status == 200
    except OSError:
        return False


def worker(args):
    root, gpu, slot = Path(args.output).resolve(), args.worker_gpu, args.worker_slot
    os.environ.update(CUDA_DEVICE_ORDER='PCI_BUS_ID', CUDA_VISIBLE_DEVICES=str(gpu),
        HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_ENDPOINT='https://hf-mirror.com',
        VLLM_ALLOW_RUNTIME_LORA_UPDATING='True', VLLM_NO_USAGE_STATS='1', HF_HUB_DISABLE_TELEMETRY='1')
    os.environ.pop('HF_TOKEN', None)
    directory = root / 'workers' / worker_name(gpu, slot)
    directory.mkdir(parents=True, exist_ok=True)
    config = copy.deepcopy(read(root / 'metadata.json')['config'])
    config['backend']['adapter_dir'] = str(directory / 'adapters')
    config['evaluation']['work_dir'] = str(directory / 'sandbox')
    if not config['evaluation'].get('eval_slot_dir'):
        config['evaluation']['eval_slot_dir'] = str(root / 'cpu-slots')
    port = args.adopt_port or worker_port(args.base_port, gpu, slot)
    config['backend']['vllm_url'] = f'http://127.0.0.1:{port}'
    state = dict(status='starting_model', gpu=gpu, slot=slot, pid=os.getpid(), started_at=runner._now(),
                 port=port, adopted_server=bool(args.adopt_server))
    write(directory / 'state.json', state)
    server = None
    server_pid = args.adopt_server
    managed_server = False
    backend = None
    job = None
    exit_code = 2
    def backend_factory(cfg):
        nonlocal backend
        if backend is None:
            backend = SharedGPUBackend(cfg, root / f'gpu{gpu}.compute.lock', directory / 'memory.json')
        elif asdict(backend.config) != asdict(cfg):
            raise ValueError('Persistent trainer configuration changed')
        return backend
    try:
        if server_pid:
            proc = Path(f'/proc/{server_pid}')
            command = (proc / 'cmdline').read_bytes()
            if proc.stat().st_uid != os.getuid() or b'vllm.entrypoints.openai.api_server' not in command:
                raise ValueError('Cannot adopt a server with another owner or process identity')
            if not server_health(port):
                raise RuntimeError('Adopted server is not healthy')
            managed_server = True
        else:
            cmd = [sys.executable, '-m', 'vllm.entrypoints.openai.api_server',
                '--model', config['backend']['model_path'], '--served-model-name', 'pot-base',
                '--host', '127.0.0.1', '--port', str(port), '--gpu-memory-utilization', str(args.memory_utilization),
                '--max-model-len', str(config['backend'].get('max_context_tokens', 16384)),
                '--enable-lora', '--max-lora-rank', str(config['backend'].get('lora_rank', 8)),
                '--dtype', config['backend'].get('dtype', 'bfloat16'), '--enforce-eager', '--generation-config', 'vllm',
                '--max-num-seqs', str(max(1, config['search'].get('width', 3))),
                '--max-num-batched-tokens', '2048']
            if args.kv_cache_gib is not None:
                cmd += ['--kv-cache-memory-bytes', str(int(args.kv_cache_gib * 2**30))]
            with DeviceLease(root / f'gpu{gpu}.compute.lock').hold():
                with socket.socket() as sock:
                    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                    sock.bind(('127.0.0.1', port))
                with (directory / 'vllm.log').open('a') as log:
                    server = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                server_pid = server.pid
                managed_server = True
                state.update(server_pid=server_pid)
                write(directory / 'state.json', state)
                for _ in range(120):
                    if server.poll() is not None:
                        raise RuntimeError('Inference server exited')
                    if server_health(port):
                        break
                    time.sleep(5)
                else:
                    raise TimeoutError('Inference server startup timed out')
        state.update(server_pid=server_pid)
        while True:
            job = Queue(root).claim(gpu, slot)
            if job is None:
                exit_code = 0
                break
            state.update(status='running', problem_index=job['index'], job_directory=job['directory'], heartbeat_at=runner._now())
            write(directory / 'state.json', state)
            code = runner.run(config, job['directory'], offset=job['index'], limit=1, backend_factory=backend_factory)
            Queue(root).finish(job, code)
            job = None
            if code:
                exit_code = code
                break
    except BaseException as exc:
        state['error_type'] = type(exc).__name__
        print(runner._json({'event': 'worker_failed', 'error_type': type(exc).__name__}), flush=True)
        if job:
            Queue(root).finish(job, 2)
    finally:
        if backend is not None:
            try:
                backend.close()
            except Exception:
                pass
        if server_pid and managed_server:
            try:
                os.killpg(server_pid, signal.SIGTERM)
                if server is not None:
                    try:
                        server.wait(timeout=20)
                    except subprocess.TimeoutExpired:
                        os.killpg(server_pid, signal.SIGKILL)
                        server.wait()
            except ProcessLookupError:
                pass
        state.update(status='exited', exit_code=exit_code, finished_at=runner._now())
        write(directory / 'state.json', state)
    return exit_code


def coordinator(args):
    root = Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    for folder in ['workers', 'jobs']:
        (root / folder).mkdir(exist_ok=True)
    lock = (root / 'coordinator.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    config = runner.parameters_from_args(args)
    config['backend']['generation_backend'] = 'vllm'
    config['backend']['vllm_model_name'] = 'pot-base'
    if {'reward_tests', 'early_stop'} & config['search'].keys():
        raise ValueError('Legacy reward/stop switches are not accepted')
    dataset = config['dataset']
    problems, _ = load_dataset(dataset['path'], selection=dataset.get('selection', 'v6'),
        start_date=dataset.get('start_date'), end_date=dataset.get('end_date'))
    start = args.offset or 0
    indices = list(range(len(problems)))[start:None if args.limit is None else start + args.limit]
    dataset_hash = runner._dataset_identity(Path(dataset['path']), dataset.get('selection', 'v6'))['sha256']
    config_hash = hashlib.sha256(runner._json(config).encode()).hexdigest()
    source_hash = {p.name: runner._hash_file(p) for p in Path(__file__).parent.glob('*.py')}
    meta_path = root / 'metadata.json'
    if meta_path.exists():
        metadata = read(meta_path)
        if (metadata['config_sha256'] != config_hash or metadata['dataset_sha256'] != dataset_hash
                or read(root/'queue.json')['indices'] != indices):
            raise ValueError('Resume code, dataset, configuration or selection changed')
        if metadata['source_sha256'] != source_hash:
            changed = {name for name in source_hash.keys() | metadata['source_sha256'].keys()
                       if source_hash.get(name) != metadata['source_sha256'].get(name)}
            if not args.resume_scheduling_change or changed != {'parallel.py'}:
                raise ValueError('Resume source changed outside an explicitly accepted scheduler-only revision')
            metadata.setdefault('scheduling_revisions', []).append({'at': runner._now(),
                'previous_source_sha256': metadata['source_sha256'], 'changed_files': sorted(changed)})
            metadata['source_sha256'] = source_hash
    else:
        metadata = {'started_at': runner._now(), 'config': config, 'config_sha256': config_hash,
            'dataset_sha256': dataset_hash, 'source_sha256': source_hash, 'dataset_count': len(problems),
            'requested_count': len(indices), 'reward_tests': 'all', 'stop_rule': 'all_tests_passed',
            'evaluation_protocol': 'all_tests_reward', 'heldout_tests_valid': False,
            'benchmark_protocol': runner.benchmark_protocol(dataset.get('selection', 'v6')),
            'worker_policy': 'persistent server and trainer; one problem per locked queue claim',
            'gpus': args.gpus, 'versions': runner._versions()}
        write(meta_path, metadata)
        write(root / 'queue.json', {'indices': indices, 'jobs': []})
    runtime = {name: getattr(args, name) for name in ['workers_per_gpu', 'memory_utilization',
        'kv_cache_gib', 'minimum_free_mib', 'base_port', 'gpus', 'shared_gpus']}
    if metadata.get('runtime') != runtime:
        metadata.setdefault('runtime_history', []).append({'at': runner._now(), 'settings': runtime})
        metadata['runtime'] = runtime
    write(meta_path, metadata)
    adopt = read(args.adopt_file) if args.adopt_file else []
    previous, last_query = set(), 0.0
    children = {}
    while True:
        state = refresh_jobs(root)
        status = publish(root, state, metadata)
        if status == 'completed':
            return 0
        if status == 'needs_attention':
            return 2
        if time.monotonic() - last_query >= args.poll_seconds:
            last_query = time.monotonic()
            try:
                active = [read(path) for path in (root / 'workers').glob('*/state.json')]
                active = [w for w in active if w['status'] != 'exited' and alive(w['pid'])]
                devices = gpu_inventory()
                free = eligible_gpus(devices, args.gpus, args.minimum_free_mib, active, shared=args.shared_gpus)
            except (subprocess.SubprocessError, OSError):
                time.sleep(5)
                continue
            candidates = sorted((free & previous) | {a['gpu'] for a in adopt})
            previous = free
            active_slots = {(w['gpu'], w.get('slot', 0)) for w in active}
            active_slots |= {key for key, child in children.items() if child.poll() is None}
            for gpu in candidates:
                if any(w['gpu'] == gpu and w['status'] == 'starting_model' for w in active):
                    continue
                slot = next((s for s in range(args.workers_per_gpu) if (gpu, s) not in active_slots), None)
                if slot is None:
                    continue
                ad = next((a for a in adopt if a['gpu'] == gpu and a.get('slot', 0) == slot), None)
                cmd = [sys.executable, '-u', '-m', 'pot.parallel', '--output', str(root), '--worker-gpu', str(gpu),
                    '--worker-slot', str(slot), '--base-port', str(args.base_port),
                    '--memory-utilization', str(args.memory_utilization)]
                if args.kv_cache_gib is not None:
                    cmd += ['--kv-cache-gib', str(args.kv_cache_gib)]
                if ad:
                    cmd += ['--adopt-server', str(ad['pid']), '--adopt-port', str(ad['port'])]
                directory = root / 'workers' / worker_name(gpu, slot)
                directory.mkdir(exist_ok=True)
                with (directory / 'worker.log').open('a') as log:
                    children[gpu, slot] = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                if ad:
                    adopt.remove(ad)
        time.sleep(5)


def main():
    parser = runner.build_parser()
    parser.description = __doc__
    parser.add_argument('--gpus', type=int, nargs='+', default=list(range(8)))
    parser.add_argument('--shared-gpus', type=int, nargs='+', default=[],
        help='Explicitly allow coexistence with other jobs on these GPUs, subject to the free-memory threshold')
    parser.add_argument('--minimum-free-mib', type=int, default=36000)
    parser.add_argument('--poll-seconds', type=float, default=30)
    parser.add_argument('--base-port', type=int, default=19700)
    parser.add_argument('--memory-utilization', type=float, default=0.24)
    parser.add_argument('--kv-cache-gib', type=float, help='Explicit KV cache size; overrides automatic vLLM cache sizing')
    parser.add_argument('--workers-per-gpu', type=int, default=1)
    parser.add_argument('--resume-scheduling-change', action='store_true',
        help='Accept and record a parallel.py-only revision; all experiment code/configuration must match')
    parser.add_argument('--worker-gpu', type=int, help=__import__('argparse').SUPPRESS)
    parser.add_argument('--worker-slot', type=int, default=0, help=__import__('argparse').SUPPRESS)
    parser.add_argument('--adopt-file', help=__import__('argparse').SUPPRESS)
    parser.add_argument('--adopt-server', type=int, help=__import__('argparse').SUPPRESS)
    parser.add_argument('--adopt-port', type=int, help=__import__('argparse').SUPPRESS)
    args = parser.parse_args()
    if not set(args.shared_gpus) <= set(args.gpus):
        parser.error('shared-gpus must be a subset of gpus')
    if args.poll_seconds < 5:
        parser.error('poll-seconds must be at least 5')
    if not 1 <= args.workers_per_gpu <= 16 or not 0 <= args.worker_slot < 16:
        parser.error('workers-per-gpu must be 1..16 and worker-slot must be 0..15')
    if not 0 < args.memory_utilization <= 1 or args.minimum_free_mib <= 0:
        parser.error('invalid GPU memory budget')
    if args.kv_cache_gib is not None and args.kv_cache_gib <= 0:
        parser.error('kv-cache-gib must be positive')
    return worker(args) if args.worker_gpu is not None else coordinator(args)


if __name__ == '__main__':
    raise SystemExit(main())
