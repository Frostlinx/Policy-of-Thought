#!/usr/bin/env python3
"""Check adapter compatibility using dataset reference solutions, NOT model accuracy.

Run in tmux on a POSIX host. Reference solutions are executed only in resource-
limited child processes, never passed to a model, stored in prompts or solutions.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pot.data import load_dataset
from pot.benchmarks import benchmark_protocol
from pot.evaluator import DockerConfig
from pot.process_evaluator import ProcessEvaluator
from scripts.download_benchmark import SPECS, digest, read_records


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", choices=tuple(SPECS), required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--raw-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--test-timeout", type=float, default=10)
    args = parser.parse_args(argv)
    if args.workers < 1:
        parser.error("workers must be positive")
    if digest(args.raw_file) != SPECS[args.selection]["sha256"]:
        raise ValueError("Reference data differs from pinned revision")
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("Use an empty validation output directory")
    public, private = load_dataset(args.data, args.selection)
    raw = read_records(args.raw_file)
    by_id = {str(r["task_id"]) if args.selection == "humaneval" else f"MBPP/{r['task_id']}": r for r in raw}
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()

    def check(item):
        index, problem = item
        row = by_id[problem.problem_id]
        code = row["prompt"] + row["canonical_solution"] if args.selection == "humaneval" else row["code"]
        e = ProcessEvaluator(DockerConfig(timeout_seconds=args.test_timeout, memory_mb=1024,
            work_dir=str(args.output / "scratch"), eval_workers=1, eval_global_workers=args.workers), private)
        before = time.monotonic()
        score = e.evaluate_candidate(problem, code)
        return dict(problem_index=index, problem_id=problem.problem_id, question_id=problem.question_id,
            score=int(score.all_passed), pass_rate=score.reward, passed=score.passed, total=score.total,
            public_passed=score.public.passed, public_total=score.public.total,
            private_passed=score.private.passed, private_total=score.private.total,
            success_gate_passed=score.private.gate_passed, success_gate_total=score.private.gate_total,
            elapsed_seconds=time.monotonic()-before, purpose="reference_harness_validation_not_model_accuracy")

    rows = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for result in pool.map(check, enumerate(public)):
            rows.append(result)
            for name in ("process.json", "process.jsonl"):
                with (args.output / name).open("a", encoding="utf-8") as f:
                    f.write(json.dumps(result) + "\n")
    summary = dict(status="completed", purpose="reference_harness_validation_not_model_accuracy",
        selection=args.selection, count=len(rows), reference_pass_count=sum(r["score"] for r in rows),
        reference_failure_ids=[r["problem_id"] for r in rows if not r["score"]],
        elapsed_seconds=time.monotonic()-started, benchmark_protocol=benchmark_protocol(args.selection),
        raw_sha256=digest(args.raw_file), finished_at=datetime.now(timezone.utc).isoformat())
    for name in ("summary.json", "summary.jsonl"):
        (args.output / name).write_text(json.dumps(summary) + "\n", encoding="utf-8")
    print(json.dumps(summary), flush=True)
    return int(summary["reference_pass_count"] != len(rows))


if __name__ == "__main__":
    raise SystemExit(main())
