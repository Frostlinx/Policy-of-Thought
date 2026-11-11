"""Generate independent LCB candidates to measure output truncation, without training or judging.

Reads public.jsonl only. Each instance owns its server process group and releases
it on completion. Run the coordinator inside tmux; results are resumable.
"""
from __future__ import annotations

import argparse
import ast
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import statistics
import subprocess
import sys
import threading
import time
from collections import deque

import requests

from .data import PublicProblem, TestCase, read_jsonl
from .search import Node, problem_messages, extract_code


def now():
    return datetime.now(timezone.utc).isoformat()


def write(path, obj):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, separators=(',', ':')) + '\n', encoding='utf-8')
    os.replace(tmp, path)


def inspect_choice(choice, tokenizer, output):
    text = choice['text']
    token_ids = choice.get('token_ids')
    if token_ids is None:
        # Do not silently substitute retokenized text for the exact output count.
        raise ValueError('vLLM did not return completion token_ids')
    output.write_text(text, encoding='utf-8')
    code = None
    error = None
    try:
        code = extract_code(text)
    except ValueError as exc:
        error = str(exc)
    syntax_ok = False
    if code is not None:
        try:
            ast.parse(code)
            syntax_ok = True
            output.with_suffix('.py').write_text(code, encoding='utf-8')
        except SyntaxError as exc:
            error = 'syntax_error: ' + exc.msg
    opening = text.find('```')
    analysis = text[:opening].strip() if opening >= 0 else text.strip()
    return {
        'candidate_index': choice['index'], 'finish_reason': choice['finish_reason'],
        'output_tokens': len(token_ids), 'truncated': choice['finish_reason'] == 'length',
        'has_analysis_before_code': bool(analysis),
        'analysis_text_tokens': len(tokenizer.encode(analysis, add_special_tokens=False)),
        'code_text_tokens': len(tokenizer.encode(code or '', add_special_tokens=False)),
        'component_token_note': 'Text segments are retokenized; output_tokens uses exact server token IDs.',
        'code_extractable': code is not None, 'python_syntax_valid': syntax_ok,
        'format_or_syntax_error': error, 'response_file': str(output),
        'score': None, 'pass_rate': None,
    }


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', required=True)
    p.add_argument('--data', required=True, help='Normalized dataset directory; only public.jsonl is read')
    p.add_argument('--output', required=True)
    p.add_argument('--gpus', nargs='+', type=int, default=list(range(8)))
    p.add_argument('--shared-gpus', nargs='*', type=int, default=[], help='Explicitly allow coexistence with other GPU jobs')
    p.add_argument('--workers-per-gpu', type=int, default=2)
    p.add_argument('--max-workers', type=int, default=6)
    p.add_argument('--minimum-free-mib', type=int, default=20000)
    p.add_argument('--kv-cache-gib', type=float, default=4)
    p.add_argument('--base-port', type=int, default=20500)
    p.add_argument('--candidates', type=int, choices=[3, 4], default=4)
    p.add_argument('--max-tokens', type=int, default=2048)
    p.add_argument('--max-context-tokens', type=int, default=16384)
    p.add_argument('--temperature', type=float, default=0.7)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--limit', type=int, default=175)
    p.add_argument('--poll-seconds', type=float, default=10)
    p.add_argument('--request-timeout', type=float, default=900)
    return p


def main():
    args = parser().parse_args()
    if args.max_workers < 1 or args.workers_per_gpu < 1 or args.limit < 1:
        raise ValueError('Worker counts and limit must be positive')
    from transformers import AutoTokenizer
    from .parallel import gpu_inventory, belongs_to_worker
    import fcntl
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    lock_file = (out/'coordinator.lock').open('a')
    fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    for name in ['questions', 'responses', 'workers']:
        (out/name).mkdir(exist_ok=True)
    source = Path(args.data)/'public.jsonl'
    public = []
    for i, row in enumerate(read_jsonl(source)):
        if i >= args.limit:
            break
        fields = {k: row[k] for k in PublicProblem.__dataclass_fields__ if k in row and k != 'public_tests'}
        fields['public_tests'] = tuple(TestCase(**t) for t in row['public_tests'])
        public.append((i, PublicProblem(**fields)))
    if len(public) != args.limit:
        raise ValueError('Dataset has fewer questions than requested')
    tok = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    fingerprint = {
        'model': args.model, 'data_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
        'prompt_source_sha256': hashlib.sha256(Path(__file__).with_name('search.py').read_bytes()).hexdigest(),
        'prompt_templates_sha256': hashlib.sha256(Path(__file__).with_name('prompts.py').read_bytes()).hexdigest(),
        'candidates': args.candidates, 'max_tokens': args.max_tokens, 'temperature': args.temperature,
        'seed': args.seed, 'limit': args.limit, 'max_context_tokens': args.max_context_tokens,
    }
    if (out/'metadata.json').exists():
        metadata = json.loads((out/'metadata.json').read_text())
        if metadata['fingerprint'] != fingerprint:
            raise ValueError('Cannot mix results from different probe configurations')
    else:
        metadata = {'started_at': now(), 'fingerprint': fingerprint,
                    'protocol': 'initial_prompt_only; base model; no LoRA training; no test execution',
                    'private_test_access': False, 'score_note': 'Scores and pass rates are null because no tests are executed.'}
    metadata['runtime'] = vars(args)
    write(out/'metadata.json', metadata)
    records = {z['problem_index']: z for f in (out/'questions').glob('*.json')
               if (z := json.loads(f.read_text())).get('status') == 'ok'}
    pending = deque((i, p) for i, p in public if i not in records)
    mutex = threading.Lock()
    stop = threading.Event()
    states, failures, attempts = {}, [], {}
    for sig in [signal.SIGTERM, signal.SIGINT]:
        signal.signal(sig, lambda *_: stop.set())

    def worker(gpu, slot):
        key = f'gpu{gpu}-slot{slot}'
        port = args.base_port + gpu*8 + slot
        directory = out/'workers'/key
        directory.mkdir(exist_ok=True)
        state = states[key]
        server = None
        job = None
        try:
            with socket.socket() as sock:
                sock.bind(('127.0.0.1', port))
            env = dict(os.environ, CUDA_DEVICE_ORDER='PCI_BUS_ID', CUDA_VISIBLE_DEVICES=str(gpu))
            cmd = [sys.executable, '-m', 'vllm.entrypoints.openai.api_server',
                   '--model', args.model, '--served-model-name', 'pot-probe',
                   '--host', '127.0.0.1', '--port', str(port), '--dtype', 'bfloat16',
                   '--max-model-len', str(args.max_context_tokens), '--enforce-eager',
                   '--generation-config', 'vllm', '--gpu-memory-utilization', '0.24',
                   '--kv-cache-memory-bytes', str(int(args.kv_cache_gib*2**30)),
                   '--max-num-seqs', str(args.candidates), '--max-num-batched-tokens', '2048',
                   '--enable-prefix-caching']
            with (directory/'vllm.log').open('a') as log:
                server = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
            state.update(server_pid=server.pid, port=port)
            session = requests.Session()
            ready = False
            deadline = time.monotonic()+600
            while not stop.is_set() and time.monotonic() < deadline:
                if server.poll() is not None:
                    raise RuntimeError('vLLM server exited during startup')
                try:
                    ready = session.get(f'http://127.0.0.1:{port}/health', timeout=2).ok
                except requests.RequestException:
                    pass
                if ready:
                    break
                stop.wait(2)
            if not ready:
                raise RuntimeError('vLLM startup did not complete')
            while not stop.is_set():
                with mutex:
                    if not pending:
                        break
                    job = pending.popleft()
                index, problem = job
                state.update(phase='generating', problem_index=index, problem_id=problem.problem_id, updated_at=now())
                messages = problem_messages(problem, Node(0))
                prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                ids = tok.encode(prompt, add_special_tokens=False)
                if len(ids)+args.max_tokens > args.max_context_tokens:
                    raise ValueError('Prompt plus output budget exceeds context length')
                payload = {'model': 'pot-probe', 'prompt': ids, 'n': args.candidates,
                           'max_tokens': args.max_tokens, 'temperature': args.temperature,
                           'top_p': 1.0, 'top_k': -1, 'min_p': 0.0, 'repetition_penalty': 1.0,
                           'presence_penalty': 0.0, 'frequency_penalty': 0.0,
                           'seed': args.seed+index, 'return_token_ids': True,
                           'add_special_tokens': False, 'skip_special_tokens': True, 'stream': False}
                started = time.monotonic()
                response = session.post(f'http://127.0.0.1:{port}/v1/completions', json=payload, timeout=args.request_timeout)
                response.raise_for_status()
                result = response.json()
                elapsed = time.monotonic()-started
                choices = sorted(result['choices'], key=lambda z: z['index'])
                if len(choices) != args.candidates:
                    raise RuntimeError('Unexpected number of candidates')
                candidates = [inspect_choice(c, tok, out/'responses'/f'{index:05d}-c{c["index"]}.txt') for c in choices]
                row = {'problem_index': index, 'problem_id': problem.problem_id, 'question_id': problem.question_id,
                       'platform': problem.platform, 'difficulty': problem.difficulty,
                       'interface': 'functional' if problem.starter_code else 'stdin',
                       'status': 'ok', 'finished_at': now(), 'elapsed_seconds': elapsed,
                       'time_definition': 'Wall time for the batch of four/three candidates; overlapping across workers.',
                       'gpu': gpu, 'slot': slot, 'prompt_tokens': len(ids),
                       'output_tokens': sum(c['output_tokens'] for c in candidates),
                       'truncated_candidates': sum(c['truncated'] for c in candidates),
                       'score': None, 'pass_rate': None, 'candidates': candidates}
                write(out/'questions'/f'{index:05d}.json', row)
                with mutex:
                    records[index] = row
                job = None
                state.update(phase='ready', updated_at=now())
        except Exception as exc:
            with mutex:
                failures.append({'at': now(), 'worker': key, 'problem_index': job[0] if job else None,
                                 'type': type(exc).__name__, 'message': str(exc)[:300]})
                if job is not None:
                    attempts[job[0]] = attempts.get(job[0], 0)+1
                    if attempts[job[0]] < 3:
                        pending.append(job)
            state['error_type'] = type(exc).__name__
        finally:
            if server is not None and server.poll() is None:
                try:
                    os.killpg(server.pid, signal.SIGTERM)
                    server.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(server.pid, signal.SIGKILL)
                    server.wait(timeout=10)
                except ProcessLookupError:
                    pass
            state.update(phase='exited', updated_at=now())
            write(directory/'state.json', state)

    def summarize(status):
        with mutex:
            rows = [records[i] for i in sorted(records)]
            failed = list(failures)
        candidates = [c for row in rows for c in row['candidates']]
        n = len(candidates)
        tokens = sorted(c['output_tokens'] for c in candidates)
        summary = {'status': status, 'started_at': metadata['started_at'], 'updated_at': now(),
                   'requested_questions': len(public), 'completed_questions': len(rows),
                   'remaining_questions': len(public)-len(rows), 'requested_candidates': len(public)*args.candidates,
                   'generated_candidates': n, 'truncated_candidates': sum(c['truncated'] for c in candidates),
                   'truncation_rate': sum(c['truncated'] for c in candidates)/n if n else None,
                   'questions_with_truncation': sum(row['truncated_candidates']>0 for row in rows),
                   'extractable_candidates': sum(c['code_extractable'] for c in candidates),
                   'syntax_valid_candidates': sum(c['python_syntax_valid'] for c in candidates),
                   'candidates_with_analysis': sum(c['has_analysis_before_code'] for c in candidates),
                   'mean_output_tokens': statistics.mean(tokens) if tokens else None,
                   'p95_output_tokens': tokens[min(n-1, int(n*.95))] if n else None,
                   'accuracy': None, 'mean_score': None, 'judging_performed': False, 'training_performed': False,
                   'failure_attempts': len(failed), 'runtime_workers': list(states.values()), 'by_group': {}}
        for field in ['platform', 'difficulty', 'interface']:
            for value in sorted({row[field] for row in rows}):
                subset = [c for row in rows if row[field] == value for c in row['candidates']]
                summary['by_group'][f'{field}:{value}'] = {'candidates': len(subset),
                    'truncated': sum(c['truncated'] for c in subset),
                    'extractable': sum(c['code_extractable'] for c in subset),
                    'syntax_valid': sum(c['python_syntax_valid'] for c in subset)}
        write(out/'summary.json', summary)
        write(out/'summary.jsonl', summary)
        text = ''.join(json.dumps(row, ensure_ascii=False, separators=(',', ':'))+'\n' for row in rows)
        for name in ['process.json', 'process.jsonl']:
            tmp = out/(name+'.tmp'); tmp.write_text(text, encoding='utf-8'); os.replace(tmp, out/name)
        write(out/'execution_failures.json', failed)

    pool = ThreadPoolExecutor(max_workers=args.max_workers)
    active = {}
    launches = {}
    try:
        while not stop.is_set():
            for key, future in list(active.items()):
                if future.done():
                    future.result()
                    del active[key]
            if len(records) == len(public):
                break
            devices = gpu_inventory()
            roots = {s['server_pid'] for s in states.values() if s.get('server_pid') and s['phase'] != 'exited'}
            for gpu in args.gpus:
                if not pending or len(active) >= args.max_workers:
                    break
                device = devices.get(gpu)
                if not device or device['free'] < args.minimum_free_mib:
                    continue
                if gpu not in args.shared_gpus and not all(belongs_to_worker(pid, roots) for pid in device['pids']):
                    continue
                if any(states[k]['gpu'] == gpu and states[k]['phase'] == 'starting' for k in active):
                    continue
                for slot in range(args.workers_per_gpu):
                    key = f'gpu{gpu}-slot{slot}'
                    if key in active or launches.get(key, 0) >= 3:
                        continue
                    states[key] = {'gpu': gpu, 'slot': slot, 'phase': 'starting', 'started_at': now()}
                    launches[key] = launches.get(key, 0)+1
                    active[key] = pool.submit(worker, gpu, slot)
                    break  # Recheck actual free memory after this server starts.
            if not active and not pending:
                summarize('needs_attention')
                return 2
            summarize('running' if active else 'waiting_for_gpu')
            stop.wait(args.poll_seconds)
    finally:
        stop.set()
        pool.shutdown(wait=True)
        summarize('completed' if len(records) == len(public) else 'interrupted')
    return 0 if len(records) == len(public) else 2


if __name__ == '__main__':
    raise SystemExit(main())
