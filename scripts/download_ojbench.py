#!/usr/bin/env python3
"""Download only OJBench's 73 ICPC Python tasks through hf-mirror.

python scripts/download_ojbench.py --output data/ojbench-icpc --no-proxy
Offline: --raw-dir RAW, containing prompts/full.jsonl and ICPC/<id>/ files.
No dataset loader scripts run; downloaded validators run later inside DMOJ.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pot.ojbench import OJ_REPO, OJ_REVISION, DMOJ_REVISION, STATEMENT_ERRATA, normalize_ojbench, sha256, pinned_files, verify_raw_file
from pot.data import HF_MIRROR, read_jsonl, save_dataset
from pot.benchmarks import benchmark_protocol
from scripts.mirror_transfer import _MirrorRedirects, aria2_download


def download(name, destination, no_proxy=False, use_aria2=False):
    destination = Path(destination)
    spec = pinned_files()[name]
    if verify_raw_file(destination, spec):
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    url = f'{HF_MIRROR}/datasets/{OJ_REPO}/resolve/{OJ_REVISION}/{name}?download=true'
    headers = {'User-Agent': 'pot-benchmark-adapter/0.5', 'Accept-Encoding': 'identity'}
    if os.environ.get('HF_TOKEN'):
        headers['Authorization'] = 'Bearer ' + os.environ['HF_TOKEN']
    for attempt in range(4):
        handlers = [_MirrorRedirects()]
        if no_proxy:
            handlers.append(urllib.request.ProxyHandler({}))
        partial = destination.with_name(destination.name + '.part')
        try:
            if use_aria2:
                aria2_download(url, partial, no_proxy=no_proxy)
            else:
                with urllib.request.build_opener(*handlers).open(urllib.request.Request(url, headers=headers), timeout=180) as response, partial.open('wb') as out:
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        out.write(chunk)
            if not verify_raw_file(partial, spec):
                raise ValueError('Incomplete or corrupt mirror transfer')
            partial.chmod(0o600)
            partial.replace(destination)
            return
        except Exception:
            if attempt == 3:
                raise RuntimeError('OJBench mirror transfer failed; no alternate hub was contacted') from None
            time.sleep(1 + attempt)


def prepare(output, raw_dir=None, no_proxy=False, workers=6, use_aria2=False):
    output = Path(output)
    if any((output / n).exists() for n in ('manifest.json', 'public.jsonl', 'judge')):
        raise ValueError('Use a new normalized output directory; raw downloads can be reused')
    raw = Path(raw_dir) if raw_dir else output / 'raw'
    if raw_dir is None:
        download('prompts/full.jsonl', raw / 'prompts/full.jsonl', no_proxy, use_aria2)
    rows = list(read_jsonl(raw / 'prompts/full.jsonl'))
    ids = {str(r['id']) for r in rows if str(r.get('dataset', '')).lower() == 'icpc' and r.get('language') == 'python'}
    if len(ids) != 73:
        raise ValueError('Expected 73 official ICPC Python prompts')
    if raw_dir is None:
        files = [f for f in pinned_files() if f.startswith('ICPC/') and f.split('/')[1] in ids]
        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(lambda name: download(name, raw / name, no_proxy, use_aria2), files))
    output.mkdir(parents=True, exist_ok=True)
    public, private, stats = normalize_ojbench(rows, raw, output)
    manifest = {'schema_version': 3, 'dataset': OJ_REPO, 'revision': OJ_REVISION,
        'endpoint': HF_MIRROR, 'selection': 'ojbench-icpc', 'problem_count': len(public),
        'created_at': datetime.now(timezone.utc).isoformat(), 'benchmark_protocol': benchmark_protocol('ojbench-icpc'),
        'problem_ids': [p.problem_id for p in public], 'dmoj_revision': DMOJ_REVISION,
        'prompt_sha256': sha256(raw / 'prompts/full.jsonl'),
        'judge_manifest_sha256': sha256(output / 'judge_manifest.json'),
        'judge_cases': sum(s['judge_cases'] for s in stats),
        'statement_samples': sum(s['statement_samples'] for s in stats),
        'interactive_problems': [s['problem_id'] for s in stats if s['interactive']],
        'statement_errata': STATEMENT_ERRATA,
        'selection_rule': 'Official prompts/full.jsonl dataset=icpc and language=python, preserving order. No additional filtering.',
        'visibility_rule': 'Only statement samples are public; all official judge files are evaluator-only. This is not an upstream public/private split.',
        'reward_rule': 'Unweighted fraction of passed statement samples and all official judge cases. No partial-suite early stop.',
        'heldout_tests_valid': False, 'offline_source': raw_dir is not None}
    save_dataset(output, public, private, manifest)
    return {k: manifest[k] for k in ('problem_count', 'judge_cases', 'statement_samples', 'interactive_problems')}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--raw-dir', type=Path)
    p.add_argument('--no-proxy', action='store_true')
    p.add_argument('--aria2', action='store_true', help='Use verified resumable multi-connection transfers')
    p.add_argument('--download-workers', type=int, default=6)
    args = p.parse_args()
    if not 1 <= args.download_workers <= 16:
        p.error('--download-workers must be between 1 and 16')
    print(json.dumps(prepare(args.output, args.raw_dir, args.no_proxy, args.download_workers, args.aria2)))


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print('OJBench preparation failed: ' + type(exc).__name__, file=sys.stderr)
        raise SystemExit(1)
