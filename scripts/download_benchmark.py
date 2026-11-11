#!/usr/bin/env python3
"""Prepare pinned HumanEval or MBPP sanitized test through hf-mirror only.

python scripts/download_benchmark.py --selection humaneval --output data/humaneval --no-proxy
Offline: add --raw-file /path/to/the/pinned/test.parquet. No dataset scripts run.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pot.benchmarks import normalize_humaneval, normalize_mbpp, benchmark_protocol
from pot.data import HF_MIRROR, save_dataset
from scripts.mirror_transfer import _MirrorRedirects

SPECS = {
    "humaneval": dict(repo="openai/openai_humaneval", revision="7dce6050a7d6d172f3cc5c32aa97f52fa1a2e544",
        filename="openai_humaneval/test-00000-of-00001.parquet", count=164,
        sha256="2f2871a15fbc95b6c683043359f4ed8e144c5a1c4f24f25f66bc51f598dfcfb6",
        upstream="https://github.com/openai/human-eval", license="MIT"),
    "mbpp-sanitized": dict(repo="google-research-datasets/mbpp", revision="4bb6404fdc6cacfda99d4ac4205087b89d32030c",
        filename="sanitized/test-00000-of-00001.parquet", count=257,
        sha256="e9e9efa2c0d59ef5e55537a9d126b8f875d5ac010a8d75628d76824884e15850",
        upstream="https://github.com/google-research/google-research/tree/master/mbpp", license="CC-BY-4.0"),
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_records(path):
    try:
        import pyarrow.parquet as pq
    except ImportError:
        raise RuntimeError("Parquet loading requires pyarrow; install the project's [data] extra") from None
    return pq.read_table(path).to_pylist()


def prepare(selection, output, raw_file=None, no_proxy=False):
    spec = SPECS[selection]
    output = Path(output)
    if any((output / name).exists() for name in ("public.jsonl", "private_tests.jsonl", "manifest.json")):
        raise ValueError("Dataset already prepared; use a new output directory")
    path = Path(raw_file) if raw_file else output / "raw" / "test.parquet"
    url = f"{HF_MIRROR}/datasets/{spec['repo']}/resolve/{spec['revision']}/{spec['filename']}"
    if not path.exists():
        if raw_file:
            raise ValueError("Offline raw file does not exist")
        path.parent.mkdir(parents=True, exist_ok=True)
        handlers = [_MirrorRedirects()]
        if no_proxy or os.environ.get("POT_DISABLE_PROXY") == "1":
            handlers.append(urllib.request.ProxyHandler({}))
        headers = {"User-Agent": "pot-benchmark-adapter/0.4", "Accept-Encoding": "identity"}
        if os.environ.get("HF_TOKEN"):
            headers["Authorization"] = "Bearer " + os.environ["HF_TOKEN"]
        try:
            with urllib.request.build_opener(*handlers).open(
                urllib.request.Request(url + "?download=true", headers=headers), timeout=120) as response:
                payload = response.read(16 * 1024 * 1024)
            if hashlib.sha256(payload).hexdigest() != spec["sha256"]:
                raise ValueError("Integrity check failed")
            partial = path.with_suffix(".part")
            partial.write_bytes(payload)
            partial.chmod(0o600)
            partial.replace(path)
        except Exception:
            # Signed URLs and authorization headers must never enter logs.
            raise RuntimeError("Mirror transfer/integrity check failed; no alternate hub was contacted") from None
    if digest(path) != spec["sha256"]:
        raise ValueError("Raw file differs from the pinned benchmark checksum")
    rows = read_records(path)
    public, private = (normalize_humaneval if selection == "humaneval" else normalize_mbpp)(rows)
    if len(public) != spec["count"]:
        raise ValueError("Unexpected benchmark problem count")
    if selection == "humaneval" and {p.problem_id for p in public} != {f"HumanEval/{i}" for i in range(164)}:
        raise ValueError("Unexpected HumanEval task IDs")
    save_dataset(output, public, private, {
        "schema_version": 2, "dataset": spec["repo"], "endpoint": HF_MIRROR,
        "revision": spec["revision"], "selection": selection, "problem_count": len(public),
        "created_at": datetime.now(timezone.utc).isoformat(), "benchmark_protocol": benchmark_protocol(selection),
        "files": [dict(filename=spec["filename"], sha256=spec["sha256"], bytes=path.stat().st_size, source=url)],
        "upstream": spec["upstream"], "dataset_license": spec["license"],
        "problem_ids": [p.problem_id for p in public], "offline_source": raw_file is not None,
        "reference_solutions_in_normalized_data": False,
        "private_data_policy": "Private checker contents/diagnostics are evaluator-only. All available test units contribute to reward and success; heldout_tests_valid=false.",
    })
    return {"output": str(output.resolve()), "selection": selection, "problems": len(public),
            "benchmark_protocol": benchmark_protocol(selection)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", required=True, choices=tuple(SPECS))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--raw-file", type=Path, help="Already downloaded pinned parquet; no network")
    parser.add_argument("--no-proxy", action="store_true")
    args = parser.parse_args(argv)
    print(json.dumps(prepare(args.selection, args.output, args.raw_file, args.no_proxy)))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, RuntimeError, OSError):
        print("Dataset preparation failed; check mirror connectivity, dependencies and pinned input file.", file=sys.stderr)
        raise SystemExit(1)
