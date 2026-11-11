#!/usr/bin/env python3
"""Download LCB JSONL via hf-mirror, without deprecated dataset scripts.

Example: python scripts/download_lcb.py --output data/lcb_v6 --no-proxy --aria2
Offline: python scripts/download_lcb.py --raw-dir /path/to/jsonl --output data/lcb_v6
Tokens are read only from HF_TOKEN. They never appear in manifests or output.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pot.data import (EXPECTED_COUNTS, HF_MIRROR, LCB_REPO, SELECTION_FILES,
                      read_jsonl, save_dataset, split_records)
from scripts.mirror_transfer import _MirrorRedirects, aria2_download

# Immutable upstream revision observed through hf-mirror. The actual raw data
# checksum is independently validated; updates require an explicit revision.
DEFAULT_REVISION = "0fe84c3912ea0c4d4a78037083943e8f0c4dd505"
V5_SHA256 = "7f77571c2a6df0c2a72a3277650309f67e01e0008e18117e624633df53f81214"
V6_SHA256 = "bb4c364f71921c4495a6ad15abe1a927350b720009f4933e2e71f8af0f6fd1f5"

PINNED_SHA256 = {"test5.jsonl": V5_SHA256, "test6.jsonl": V6_SHA256}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_file(filename: str, destination: Path, revision: str, no_proxy: bool = False, use_aria2: bool = False) -> str:
    if filename not in {f for files in SELECTION_FILES.values() for f in files}:
        raise ValueError("Unsupported dataset filename")
    if not revision or "/" in revision or "?" in revision or "#" in revision:
        raise ValueError("Revision must be a commit hash or simple branch/tag name")
    url = f"{HF_MIRROR}/datasets/{LCB_REPO}/resolve/{urllib.parse.quote(revision, safe='')}/{filename}?download=true"
    partial = destination.with_suffix(destination.suffix + ".part")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if use_aria2:
        aria2_download(url, partial, no_proxy=no_proxy)
        checksum = file_sha256(partial)
        if filename in PINNED_SHA256 and checksum != PINNED_SHA256[filename]:
            raise RuntimeError("LCB delta checksum differs from the pinned dataset; partial data retained")
        partial.replace(destination)
        return checksum
    headers = {"User-Agent": "pot-reproduction/1.0", "Accept-Encoding": "identity"}
    offset = partial.stat().st_size if partial.exists() else 0
    if offset:
        headers["Range"] = f"bytes={offset}-"
    request = urllib.request.Request(url, headers=headers)
    token = os.environ.get("HF_TOKEN")
    if token:
        request.add_header("Authorization", "Bearer " + token)
    handlers = [_MirrorRedirects()]
    if no_proxy or os.environ.get("POT_DISABLE_PROXY") == "1":
        handlers.append(urllib.request.ProxyHandler({}))
    opener = urllib.request.build_opener(*handlers)
    try:
        with opener.open(request, timeout=180) as response:
            append = offset > 0 and response.status == 206
            if append and not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-"):
                raise RuntimeError("Download range mismatch")
            expected = int(response.headers.get("Content-Length", 0))
            received = 0
            with partial.open("ab" if append else "wb") as output:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    output.write(chunk)
                    received += len(chunk)
            if expected and received != expected:
                raise RuntimeError("Incomplete transfer")
        checksum = file_sha256(partial)
        if filename in PINNED_SHA256 and checksum != PINNED_SHA256[filename]:
            raise RuntimeError("LCB delta checksum differs from the pinned dataset")
        partial.replace(destination)
        return checksum
    except Exception as exc:
        if isinstance(exc, urllib.error.HTTPError):
            raise RuntimeError(f"Mirror download failed with HTTP {exc.code}; partial data retained") from None
        # Network exception strings may contain a signed URL: never echo them.
        raise RuntimeError("Mirror transfer or integrity check failed; partial data retained; retry with --aria2 or --no-proxy") from None


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="data/lcb_v6")
    parser.add_argument("--selection", choices=tuple(SELECTION_FILES), default="v6")
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--raw-dir", type=Path, help="Use already downloaded official JSONL; no network")
    parser.add_argument("--no-proxy", action="store_true", help="Ignore HTTP(S)_PROXY for mirror downloads")
    parser.add_argument("--aria2", action="store_true", help="Use quiet 8-stream aria2 transfers with resume and length checks")
    args = parser.parse_args(argv)
    output = Path(args.output)
    raw_dir = args.raw_dir or output / "raw"
    provenance = []
    for filename in SELECTION_FILES[args.selection]:
        path = raw_dir / filename
        if path.exists():
            checksum = file_sha256(path)
            if filename in PINNED_SHA256 and checksum != PINNED_SHA256[filename]:
                raise RuntimeError("Existing LCB delta does not match the pinned upstream checksum")
        elif args.raw_dir:
            raise RuntimeError(f"Missing raw file: {filename}")
        else:
            print(f"Downloading {filename} through hf-mirror.com", flush=True)
            checksum = download_file(filename, path, args.revision, args.no_proxy, use_aria2=args.aria2)
        provenance.append({"filename": filename, "sha256": checksum, "bytes": path.stat().st_size,
                           "source": f"{HF_MIRROR}/datasets/{LCB_REPO}/resolve/{args.revision}/{filename}"})
    public, private = split_records(row for entry in provenance for row in read_jsonl(raw_dir / entry["filename"]))
    if len(public) != EXPECTED_COUNTS[args.selection]:
        raise RuntimeError(f"Expected {EXPECTED_COUNTS[args.selection]} problems for {args.selection}, found {len(public)}")
    manifest = {
        "schema_version": 1, "dataset": LCB_REPO, "endpoint": HF_MIRROR, "revision": args.revision,
        "selection": args.selection, "problem_count": len(public),
        "selection_meaning": (f"{len(public)} new problems in test{args.selection[1:]}.jsonl"
                              if args.selection in ("v5", "v6") else f"cumulative {args.selection} ({len(public)} problems)"),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "contest_date_min": min(p.contest_date for p in public), "contest_date_max": max(p.contest_date for p in public),
        "problem_ids": [p.problem_id for p in public], "files": provenance,
        "private_data_policy": "private_tests.jsonl is evaluator-owned; private records and diagnostics never enter model prompts; aggregate private counts contribute to all-test rewards, selection, stopping and final scoring; heldout_tests_valid=false",
        "comparison_source": "https://github.com/LiveCodeBench/LiveCodeBench/blob/main/lcb_runner/evaluation/testing_util.py",
        "offline_source": args.raw_dir is not None,
    }
    save_dataset(output, public, private, manifest)
    print(json.dumps({"output": str(output.resolve()), "selection": args.selection, "problems": len(public)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, ValueError, OSError) as exc:
        print(f"Dataset preparation failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
