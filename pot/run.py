"""Sequential, resumable PoT evaluation from explicit command-line parameters.

Usage: python -m pot.run --model ./models/solver --data ./data/lcb-v6 --output ./runs/lcb-v6
CLI and dictionary paths resolve against the working directory. Optional legacy
JSON configuration paths resolve against that configuration's directory.
process.json and process.jsonl both contain one JSON object per line. Resume
never reruns a recorded problem, including an infrastructure-error record;
retry a failed problem in a new output directory with --offset and --limit.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import sys
import threading
import time
import uuid
from typing import Any

from . import __version__
from .backend import Backend, BackendConfig
from .data import SELECTION_FILES, SELECTIONS, load_dataset
from .benchmarks import benchmark_protocol
from .evaluator import DockerConfig, DockerEvaluator
from .process_evaluator import ProcessEvaluator
from .search import SearchConfig, solve


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json(row: Any) -> str:
    return json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, row: dict) -> None:
    temporary = path.with_name(path.name + ".tmp-" + uuid.uuid4().hex)
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            stream.write(_json(row) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _append(path: Path, row: dict) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(_json(row) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _read_strict_jsonl(path: Path) -> list[dict]:
    raw = path.read_bytes()
    if raw and not raw.endswith(b"\n"):
        raise ValueError(f"Incomplete JSONL record in {path.name}")
    rows = []
    for index, line in enumerate(raw.splitlines(), 1):
        try:
            row = json.loads(line, parse_constant=lambda _: (_ for _ in ()).throw(ValueError()))
            if not isinstance(row, dict):
                raise ValueError
        except (ValueError, UnicodeDecodeError):
            raise ValueError(f"Corrupt JSONL record in {path.name}, line {index}") from None
        rows.append(row)
    return rows


def _safe_config(value):
    """Keep configuration reproducibility without copying possible credentials."""
    from urllib.parse import urlsplit, urlunsplit
    if isinstance(value, dict):
        return {key: ("[redacted]" if any(part in key.lower() for part in
                ("password", "secret", "credential", "api_key", "access_token")) else _safe_config(item))
                for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe_config(item) for item in value]
    if isinstance(value, str) and "://" in value:
        parsed = urlsplit(value)
        host = parsed.hostname or ""
        if ":" in host:
            host = "[" + host + "]"
        if parsed.port:
            host += ":" + str(parsed.port)
        return urlunsplit((parsed.scheme, host, parsed.path, "", ""))
    return value


def _versions() -> dict:
    versions = {"pot": __version__, "python": platform.python_version(),
                "platform": platform.platform()}
    for package in ("torch", "transformers", "peft", "vllm", "requests", "safetensors"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def _error_details(error: BaseException) -> dict:
    frames = []
    traceback = error.__traceback__
    while traceback is not None:
        code = traceback.tb_frame.f_code
        frames.append({"file": Path(code.co_filename).name, "function": code.co_name,
                       "line": traceback.tb_lineno})
        traceback = traceback.tb_next
    return {"error_type": type(error).__name__, "frames": frames}


def _resolve(value: str, base: Path) -> str:
    path = Path(value).expanduser()
    return str((base / path).resolve() if not path.is_absolute() else path.resolve())


def _executable_path(value: str, base: Path) -> str:
    # Resolving venv/bin/python's symlink changes sys.prefix and loses the
    # installed judge dependencies. Make it absolute without dereferencing it.
    return os.path.abspath(os.path.join(str(base), os.path.expanduser(value)))


def _dataset_identity(path: Path, selection: str) -> dict:
    if path.is_file():
        files = [path]
    elif (path / "public.jsonl").is_file():
        files = [path / "public.jsonl", path / "private_tests.jsonl"]
        if (path / "manifest.json").is_file():
            files.append(path / "manifest.json")
        if selection == "ojbench-icpc":
            files.append(path / "judge_manifest.json")
    else:
        if selection not in SELECTION_FILES:
            raise ValueError("Prepare this benchmark with scripts/download_benchmark.py first")
        files = [path / name for name in SELECTION_FILES[selection]]
    # Only byte counts and one-way digests of hidden files are recorded.
    entries = [{"name": file.name, "bytes": file.stat().st_size,
                "sha256": _hash_file(file)} for file in files]
    return {"selection": selection, "files": entries,
            "sha256": hashlib.sha256(_json(entries).encode()).hexdigest()}


class _ReportingEvaluator(DockerEvaluator):
    pass


class _ReportingProcessEvaluator(ProcessEvaluator):
    pass


class _Progress:
    def __init__(self, path: Path):
        self.path = path
        self.state = {"status": "initializing", "started_at": _now(), "pid": os.getpid()}
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._heartbeat, daemon=True)

    def set(self, **values):
        with self.lock:
            self.state.update(values)
            self.state["heartbeat_at"] = _now()
            _atomic_json(self.path, self.state)

    def _heartbeat(self):
        while not self.stop_event.wait(15):
            try:
                self.set()
            except OSError:
                # Main-thread writes remain authoritative and fail the run.
                pass

    def start(self):
        self.set()
        self.thread.start()

    def close(self):
        self.stop_event.set()
        if self.thread.is_alive():
            self.thread.join(timeout=2)


def _summary(rows: list[dict], requested: int, dataset_count: int, status: str,
             started: str, config_hash: str, dataset_hash: str) -> dict:
    scored = [row for row in rows if row.get("status") == "ok"]
    errors = [row for row in rows if row.get("status") == "infrastructure_error"]
    successes = sum(row["score"] for row in scored)
    private_scored = [row for row in scored if row.get("private_pass_rate") is not None]
    return {"schema_version": 1, "status": status, "started_at": started, "updated_at": _now(),
        "requested_count": requested, "dataset_count": dataset_count, "attempted_count": len(rows),
        "evaluated_count": len(scored), "error_count": len(errors), "success_count": successes,
        "failure_count": len(scored) - successes, "remaining_count": max(0, requested-len(rows)),
        "accuracy": successes / len(scored) if scored else None,
        "accuracy_percent": 100 * successes / len(scored) if scored else None,
        "accuracy_denominator": "successfully_evaluated_problems; infrastructure_errors_excluded",
        "mean_pass_rate": sum(row["pass_rate"] for row in scored) / len(scored) if scored else None,
        "mean_private_pass_rate": sum(row["private_pass_rate"] for row in private_scored) / len(private_scored) if private_scored else None,
        "elapsed_seconds": sum(row.get("elapsed_seconds", 0) for row in rows),
        "generated_tokens": sum(row.get("generated_tokens", 0) for row in scored),
        "training_updates": sum(row.get("training_updates", 0) for row in scored),
        "optimizer_steps": sum(row.get("optimizer_steps", 0) for row in scored),
        "config_sha256": config_hash, "dataset_sha256": dataset_hash}


def _validate_resume(output: Path, identity: dict, selected: list) -> tuple[dict, list[dict]]:
    metadata = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
    if metadata.get("identity") != identity:
        raise ValueError("Resume config, selection, code version or dataset identity differs")
    for name in ("process.json", "process.jsonl", "summary.jsonl", "trace.jsonl"):
        if not (output / name).is_file():
            raise ValueError("Resume requires every original log file")
    if (output / "process.json").read_bytes() != (output / "process.jsonl").read_bytes():
        raise ValueError("Process log copies disagree; refusing ambiguous resume")
    rows = _read_strict_jsonl(output / "process.jsonl")
    _read_strict_jsonl(output / "summary.jsonl")
    _read_strict_jsonl(output / "trace.jsonl")
    _read_strict_jsonl(output / "summary.json")
    if len(rows) > len(selected):
        raise ValueError("Process log contains more rows than requested")
    seen = set()
    for row, (index, problem) in zip(rows, selected):
        if row.get("problem_id") in seen:
            raise ValueError("Duplicate problem in process log")
        seen.add(row.get("problem_id"))
        if row.get("problem_id") != problem.problem_id or row.get("problem_index") != index:
            raise ValueError("Process log order or problem identity differs from dataset")
        if row.get("status") not in {"ok", "infrastructure_error"}:
            raise ValueError("Invalid process record status")
        if row["status"] == "ok":
            if row.get("score") not in (0, 1) or not all(
                isinstance(row.get(key), (int, float)) and math.isfinite(row[key]) and 0 <= row[key] <= 1
                for key in ("pass_rate", "public_score")):
                raise ValueError("Invalid process result metrics")
            private_rate = row.get("private_pass_rate")
            if not (private_rate is None and row.get("private_total") == 0) and not (
                isinstance(private_rate, (int, float)) and math.isfinite(private_rate) and 0 <= private_rate <= 1):
                raise ValueError("Invalid private pass rate")
            code_path = output / "solutions" / f"{index:05d}.py"
            if not code_path.is_file() or _hash_file(code_path) != row.get("code_sha256"):
                raise ValueError("Selected solution file is missing or changed")
    return metadata, rows


def evaluation_protocol(config: SearchConfig) -> str:
    return "all_tests_reward"


def run(config_path: str | Path | dict, output_path: str | Path, *, limit: int | None = None,
        offset: int | None = None, resume: bool = False, backend_factory=None) -> int:
    output = Path(output_path).resolve()
    if isinstance(config_path, dict):
        raw = copy.deepcopy(config_path)
        path_base = Path.cwd()
    else:
        config_path = Path(config_path).resolve()
        raw = json.loads(config_path.read_text(encoding="utf-8"))
        path_base = config_path.parent
    if not isinstance(raw, dict) or "backend" not in raw or "dataset" not in raw:
        raise ValueError("Configuration requires backend and dataset objects")
    backend_values = dict(raw["backend"])
    dataset = dict(raw["dataset"])
    run_values = dict(raw.get("run", {}))
    limit = limit if limit is not None else run_values.get("limit")
    offset = offset if offset is not None else run_values.get("offset", 0)
    if not isinstance(offset, int) or offset < 0 or (limit is not None and
            (not isinstance(limit, int) or limit < 0)):
        raise ValueError("offset and limit must be nonnegative integers")
    backend_values["model_path"] = _resolve(backend_values["model_path"], path_base)
    backend_values["adapter_dir"] = _resolve(backend_values.get("adapter_dir", str(output / "adapters")), path_base)
    backend_values["seed"] = int(run_values.get("seed", backend_values.get("seed", 42)))
    backend_config = BackendConfig(**backend_values)
    search_values = raw.get("search", {})
    if {'early_stop', 'reward_tests'} & search_values.keys():
        raise ValueError("Remove early_stop and reward_tests: all tests always determine reward and success")
    search_config = SearchConfig(**search_values)
    evaluator_values = dict(raw.get("evaluation", {}))
    executor = evaluator_values.pop("executor", "docker")
    if dataset.get("selection") == "ojbench-icpc" and executor == "subprocess":
        executor = "ojbench"
    if (executor == "ojbench") != (dataset.get("selection") == "ojbench-icpc"):
        raise ValueError("OJBench requires the official ojbench executor")
    if executor not in {"docker", "subprocess", "ojbench"}:
        raise ValueError("evaluation.executor must be docker, subprocess or ojbench")
    if executor in {"subprocess", "ojbench"}:
        evaluator_values.setdefault("memory_mb", 1024)
    if evaluator_values.get("work_dir") is None:
        evaluator_values["work_dir"] = str(output / "sandbox")
    evaluator_values["work_dir"] = _resolve(evaluator_values["work_dir"], path_base)
    if evaluator_values.get('eval_slot_dir'):
        evaluator_values['eval_slot_dir'] = _resolve(evaluator_values['eval_slot_dir'], path_base)
    if "command" in evaluator_values:
        evaluator_values["command"] = tuple(evaluator_values["command"])
    dataset["path"] = _resolve(dataset["path"], path_base)
    dataset.setdefault("selection", "v6")
    if dataset["selection"] not in SELECTIONS:
        raise ValueError("Unsupported dataset selection")
    for key in ("ojbench_python", "ojbench_pypy", "ojbench_gpp"):
        if evaluator_values.get(key):
            evaluator_values[key] = _executable_path(evaluator_values[key], path_base)
    evaluator_config = DockerConfig(**evaluator_values)
    dataset_identity = _dataset_identity(Path(dataset["path"]), dataset["selection"])
    problems, private_store = load_dataset(dataset["path"], selection=dataset["selection"],
        start_date=dataset.get("start_date"), end_date=dataset.get("end_date"))
    if len({problem.problem_id for problem in problems}) != len(problems):
        raise ValueError("Dataset contains duplicate problem IDs")
    selected = list(enumerate(problems))[offset:None if limit is None else offset + limit]
    effective = {"backend": asdict(backend_config), "search": asdict(search_config),
                 "evaluation": {"executor": executor, **asdict(evaluator_config)}, "dataset": dataset,
                 "run": {"seed": backend_config.seed, "offset": offset, "limit": limit}}
    config_hash = hashlib.sha256(_json(effective).encode()).hexdigest()
    source_hashes = {name: _hash_file(Path(__file__).parent / name) for name in
                     ("run.py", "backend.py", "search.py", "prompts.py", "data.py", "benchmarks.py", "evaluator.py")}
    if executor in {"subprocess", "ojbench"}:
        source_hashes["process_evaluator.py"] = _hash_file(Path(__file__).parent / "process_evaluator.py")
    if executor == "ojbench":
        for name in ("ojbench.py", "ojbench_evaluator.py", "ojbench_worker.py"):
            source_hashes[name] = _hash_file(Path(__file__).parent / name)
    identity = {"config_sha256": config_hash, "dataset_sha256": dataset_identity["sha256"],
                "selected_ids_sha256": hashlib.sha256(_json([p.problem_id for _, p in selected]).encode()).hexdigest(),
                "source_sha256": source_hashes}
    if resume:
        metadata, rows = _validate_resume(output, identity, selected)
    else:
        if output.exists() and any(output.iterdir()):
            raise ValueError("Output directory is not empty; choose a new output or use --resume")
        output.mkdir(parents=True, exist_ok=True)
        (output / "solutions").mkdir()
        for name in ("process.json", "process.jsonl", "summary.jsonl", "trace.jsonl"):
            (output / name).touch(exist_ok=False)
        metadata = {"schema_version": 1, "started_at": _now(), "identity": identity,
            "config": _safe_config(effective), "dataset": dataset_identity, "versions": _versions(),
            "requested_count": len(selected), "dataset_count": len(problems),
            "private_data_policy": "Public+private pass counts determine reward and success. Only public execution details enter model prompts; private inputs, outputs and errors never do.",
            "evaluation_protocol": evaluation_protocol(search_config),
            "reward_tests": "all", "stop_rule": "all_tests_passed",
            "benchmark_protocol": benchmark_protocol(dataset["selection"]),
            "heldout_tests_valid": False}
        _atomic_json(output / "metadata.json", metadata)
        rows = []
    progress = _Progress(output / "progress.json")
    backend = None
    evaluator = None
    current = None
    current_started = None
    status = "running"

    def save_summary(new_status):
        snapshot = _summary(rows, len(selected), len(problems), new_status,
                            metadata["started_at"], config_hash, dataset_identity["sha256"])
        snapshot.update(evaluation_protocol=evaluation_protocol(search_config),
                        benchmark_protocol=benchmark_protocol(dataset["selection"]),
                        heldout_tests_valid=False,
                        reward_tests="all",
                        feedback_mode=getattr(search_config, "feedback", "none"),
                        stop_rule="all_tests_passed")
        _atomic_json(output / "summary.json", snapshot)
        _append(output / "summary.jsonl", snapshot)
        return snapshot

    def append_result(row):
        _append(output / "process.jsonl", row)
        _append(output / "process.json", row)
        rows.append(row)

    def trace(row):
        # Search emits public execution details and configured reward scalars.
        # Never pass private test contents or arbitrary exceptions here.
        if row.get("event") not in {"candidate", "update", "context_pruned"}:
            raise ValueError("Unexpected search trace event")
        _append(output / "trace.jsonl", {"at": _now(), "problem_index": current[0], **row})
        progress.set(last_event=row["event"], phase="search", policy_version=row.get("policy_version"))

    try:
        save_summary("initializing")
        progress.start()
        if len(rows) < len(selected):
            if executor == "ojbench":
                from .ojbench_evaluator import OJBenchEvaluator
                evaluator = OJBenchEvaluator(evaluator_config, private_store)
            else:
                evaluator = (_ReportingProcessEvaluator if executor == "subprocess" else _ReportingEvaluator)(evaluator_config, private_store)
            if executor in {"subprocess", "ojbench"}:
                evaluator.timing_log = output / 'evaluation_timing.jsonl'
            progress.set(phase="checking_evaluator")
            metadata["sandbox"] = evaluator.check_available()
            _atomic_json(output / "metadata.json", metadata)
            progress.set(phase="loading_model")
            backend = (backend_factory or Backend)(backend_config)
            metadata["backend"] = backend.describe()
            metadata["backend"]["config"] = _safe_config(metadata["backend"]["config"])
            _atomic_json(output / "metadata.json", metadata)
        for current in selected[len(rows):]:
            index, problem = current
            current_started = time.monotonic()
            progress.set(status="running", phase="search", problem_index=index,
                         problem_id=problem.problem_id, attempted_count=len(rows),
                         requested_count=len(selected), problem_started_at=_now())
            result = solve(problem, backend, evaluator, search_config, emit=trace)
            code = result.pop("code")
            code_path = output / "solutions" / f"{index:05d}.py"
            code_path.write_text(code, encoding="utf-8", newline="\n")
            progress.set(phase="final_evaluation")
            final_started = time.monotonic()
            official = evaluator.evaluate_candidate(problem, code)
            private = official.private
            row = {"schema_version": 1, "status": "ok", "problem_index": index,
                "problem_id": problem.problem_id, "question_id": problem.question_id,
                "platform": problem.platform, "title": problem.question_title, "difficulty": problem.difficulty,
                **result, "score": int(official.all_passed),
                "benchmark": problem.benchmark,
                "test_unit": benchmark_protocol(dataset["selection"])["test_unit"],
                "heldout_tests_valid": False,
                "pass_rate": official.passed / official.total if official.total else 0.0,
                "private_pass_rate": private.passed / private.total if private.total else None,
                "official_passed": official.passed, "official_total": official.total,
                "private_passed": private.passed, "private_total": private.total,
                "success_gate_passed": private.gate_passed, "success_gate_total": private.gate_total,
                "final_evaluation_seconds": time.monotonic() - final_started,
                "search_elapsed_seconds": result["elapsed_seconds"],
                "elapsed_seconds": time.monotonic() - current_started,
                "code_file": str(code_path.relative_to(output)), "code_sha256": _hash_file(code_path),
                "finished_at": _now()}
            if executor in {"subprocess", "ojbench"}:
                row['evaluation_performance'] = evaluator.performance(problem.problem_id)
            append_result(row)
            snapshot = save_summary("running")
            print(_json({"problem_index": index, "problem_id": problem.problem_id,
                         "score": row["score"], "attempted_count": len(rows),
                         "requested_count": len(selected), "accuracy": snapshot["accuracy"]}), flush=True)
            current, current_started = None, None
        status = "completed_with_errors" if any(row["status"] != "ok" for row in rows) else "completed"
        save_summary(status)
        progress.set(status=status, phase="finished", attempted_count=len(rows), requested_count=len(selected))
        return 0 if status == "completed" else 2
    except BaseException as error:
        status = "interrupted" if isinstance(error, (KeyboardInterrupt, SystemExit)) else "incomplete"
        if current is not None and not any(row["problem_id"] == current[1].problem_id for row in rows):
            index, problem = current
            append_result({"schema_version": 1, "status": "infrastructure_error",
                "problem_index": index, "problem_id": problem.problem_id, "question_id": problem.question_id,
                "platform": problem.platform, "title": problem.question_title, "difficulty": problem.difficulty,
                "score": None, "pass_rate": None, "private_pass_rate": None, "public_score": None,
                "evaluation_protocol": evaluation_protocol(search_config),
                "reward_tests": "all",
                "feedback_mode": search_config.feedback,
                "stop_rule": "all_tests_passed",
                "heldout_tests_valid": False,
                "error_type": type(error).__name__, "phase": progress.state.get("phase"),
                "elapsed_seconds": time.monotonic() - current_started if current_started else 0,
                "finished_at": _now()})
        save_summary(status)
        progress.set(status=status, error_type=type(error).__name__)
        # Exception messages and tracebacks can contain hidden evaluator data;
        # the public log intentionally contains only the exception class/phase.
        print(_json({"status": status, **_error_details(error),
                     "phase": progress.state.get("phase")}), file=sys.stderr, flush=True)
        return 130 if status == "interrupted" else 2
    finally:
        progress.close()
        if evaluator is not None and hasattr(evaluator, 'close'):
            evaluator.close()
        if backend is not None and backend_factory is None:
            try:
                backend.close()
            except Exception as error:
                print(_json({"event": "adapter_cleanup_error", "error_type": type(error).__name__}),
                      file=sys.stderr, flush=True)


_BACKEND_FLAGS = {
    "model_path", "generation_backend", "device", "dtype", "vllm_url", "vllm_model_name",
    "adapter_dir", "learning_rate", "epochs", "clip_ratio", "loss_variant", "loss_scope", "beta",
    "kl_coefficient", "base_kl_coefficient", "entropy_coefficient", "weight_decay",
    "lora_rank", "lora_alpha", "lora_dropout", "target_modules", "reference_sync_steps",
    "max_context_tokens", "enable_thinking", "gradient_checkpointing", "disable_policy_dropout",
    "advantage_epsilon", "advantage_clip", "max_grad_norm", "request_timeout",
    "trust_remote_code", "keep_adapter_checkpoints",
}
_SEARCH_FLAGS = {"simulations", "width", "max_depth", "max_new_tokens", "c_puct",
                 "greedy_commitment", "train", "feedback"}
_EVALUATOR_FLAGS = {"executor", "timeout_seconds", "memory_mb", "cpus", "pids_limit",
                   "max_output_bytes", "startup_grace_seconds", "image", "work_dir",
                   "eval_workers", "eval_global_workers", "eval_slot_dir",
                   "ojbench_python", "ojbench_pypy", "ojbench_gpp"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Explicit flags override optional JSON settings. All effective values are saved in metadata.json.")
    io = parser.add_argument_group("inputs and run")
    io.add_argument("--config", help="Optional legacy JSON settings; no YAML configuration is required")
    io.add_argument("--model", dest="model_path", default=argparse.SUPPRESS,
                    help="Local model directory (required unless supplied by --config)")
    io.add_argument("--data", dest="data_path", default=argparse.SUPPRESS,
                    help="Local raw JSONL or normalized dataset directory (required without --config)")
    io.add_argument("--output", required=True, help="Results directory; must be empty unless --resume is set")
    io.add_argument("--selection", choices=SELECTIONS, default=argparse.SUPPRESS,
                    help="Dataset selection (default: v6)")
    io.add_argument("--start-date", default=argparse.SUPPRESS, help="Inclusive contest date YYYY-MM-DD")
    io.add_argument("--end-date", default=argparse.SUPPRESS, help="Inclusive contest date YYYY-MM-DD")
    io.add_argument("--limit", type=int, help="Number of selected problems to attempt")
    io.add_argument("--offset", type=int, help="Skip this many dataset problems (default: 0)")
    io.add_argument("--seed", type=int, default=argparse.SUPPRESS, help="Sampling and adapter initialization seed (default: 42)")
    io.add_argument("--resume", action="store_true", help="Continue a matching run without repeating recorded problems")
    model = parser.add_argument_group("model and generation")
    model.add_argument("--backend", dest="generation_backend", choices=("vllm", "hf"), default=argparse.SUPPRESS,
                       help="Generation engine (default: vllm)")
    model.add_argument("--device", default=argparse.SUPPRESS, help="HF trainer device (default: cuda:0)")
    model.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default=argparse.SUPPRESS)
    model.add_argument("--vllm-url", default=argparse.SUPPRESS, help="vLLM server origin (default: http://127.0.0.1:8000)")
    model.add_argument("--vllm-name", "--vllm-model-name", dest="vllm_model_name", default=argparse.SUPPRESS,
                       help="Server's base model name (default: pot-base)")
    model.add_argument("--adapter-dir", default=argparse.SUPPRESS, help="Shared adapter directory (default: OUTPUT/adapters)")
    model.add_argument("--max-context-tokens", type=int, default=argparse.SUPPRESS, help="Prompt plus completion limit (default: 16384)")
    model.add_argument("--request-timeout", type=float, default=argparse.SUPPRESS, help="HTTP request timeout in seconds")
    model.add_argument("--enable-thinking", action=argparse.BooleanOptionalAction, default=argparse.SUPPRESS,
                       help="Enable template-specific reasoning mode (default: disabled)")
    model.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=argparse.SUPPRESS)
    model.add_argument("--keep-adapter-checkpoints", action=argparse.BooleanOptionalAction, default=argparse.SUPPRESS)
    search = parser.add_argument_group("search and feedback")
    for flag, kind, help_text in (
        ("simulations", int, "Maximum tree expansions (default: 20)"),
        ("width", int, "Sampled siblings per expansion (default: 3)"),
        ("max-depth", int, "Maximum search depth (default: 8)"),
        ("max-new-tokens", int, "Completion token limit (default: 2048)"),
        ("temperature", float, "Sampling temperature for generation and GRPO probabilities (default: 0.7)"),
        ("c-puct", float, "PUCT exploration coefficient (default: 1.414)"),
    ):
        search.add_argument("--" + flag, type=kind, default=argparse.SUPPRESS, help=help_text)
    search.add_argument("--greedy-commitment", "--greedy", dest="greedy_commitment",
                        action=argparse.BooleanOptionalAction, default=argparse.SUPPRESS,
                        help="Generate a greedy commitment after an update; --no-greedy is a short alias")
    search.add_argument("--train", action=argparse.BooleanOptionalAction, default=argparse.SUPPRESS,
                        help="Enable online LoRA updates; --no-train keeps the policy fixed")
    search.add_argument("--feedback", choices=("public", "none"), default=argparse.SUPPRESS,
                        help="Execution feedback available in model prompts (default: public)")
    training = parser.add_argument_group("LoRA and GRPO")
    training.add_argument("--loss-scope", choices=("trajectory", "completion"), default=argparse.SUPPRESS,
                          help="Default: trajectory (all model outputs from root to current node); completion is a current-response-only ablation")
    training.add_argument("--loss-variant", choices=("paper", "policy_scaled", "dual_kl"),
                          default=argparse.SUPPRESS,
                          help="Default: paper (single beta KL). policy_scaled is optional; dual_kl is an optional dual-KL objective")
    for flag, kind in (("learning-rate", float), ("epochs", int), ("clip-ratio", float),
        ("beta", float), ("base-kl-coefficient", float), ("kl-coefficient", float),
        ("entropy-coefficient", float), ("weight-decay", float), ("lora-rank", int),
        ("lora-alpha", int), ("lora-dropout", float), ("reference-sync-steps", int),
        ("advantage-epsilon", float), ("advantage-clip", float), ("max-grad-norm", float)):
        explanations = {"beta": "KL coefficient for paper (default: 0.02); policy coefficient only in policy_scaled",
                        "base-kl-coefficient": "Optional additional frozen-base KL; default 0 for paper, 0.005 only in the legacy variant",
                        "kl-coefficient": "Reference KL coefficient used by policy_scaled (default: 0.005)",
                        "entropy-coefficient": "Entropy bonus coefficient; default 0 for paper, 0.01 for policy_scaled"}
        training.add_argument("--" + flag, type=kind, default=argparse.SUPPRESS, help=explanations.get(flag))
    training.add_argument("--target-modules", nargs="+", default=argparse.SUPPRESS,
                          help="LoRA projection names (default: q_proj k_proj v_proj o_proj)")
    training.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=argparse.SUPPRESS)
    training.add_argument("--disable-policy-dropout", action=argparse.BooleanOptionalAction, default=argparse.SUPPRESS)
    evaluation = parser.add_argument_group("candidate execution")
    evaluation.add_argument("--executor", choices=("subprocess", "docker", "ojbench"), default=argparse.SUPPRESS,
                            help="Execution engine (default: subprocess; resource limits, not a security sandbox)")
    evaluation.add_argument("--timeout", dest="timeout_seconds", type=float, default=argparse.SUPPRESS,
                            help="Per-test timeout in seconds (default: 10)")
    evaluation.add_argument("--memory-mb", type=int, default=argparse.SUPPRESS, help="Per-candidate memory limit (default: 1024)")
    evaluation.add_argument("--cpus", type=float, default=argparse.SUPPRESS, help="Docker CPU quota (default: 1)")
    evaluation.add_argument("--pids-limit", type=int, default=argparse.SUPPRESS)
    evaluation.add_argument("--max-output-bytes", type=int, default=argparse.SUPPRESS)
    evaluation.add_argument("--startup-grace-seconds", type=float, default=argparse.SUPPRESS)
    evaluation.add_argument("--docker-image", dest="image", default=argparse.SUPPRESS)
    evaluation.add_argument("--evaluator-work-dir", dest="work_dir", default=argparse.SUPPRESS)
    evaluation.add_argument("--eval-workers", type=int, default=argparse.SUPPRESS,
                            help="Concurrent subprocess test cases per question worker (default: 1)")
    evaluation.add_argument("--eval-global-workers", type=int, default=argparse.SUPPRESS,
                            help="Maximum concurrent test processes across workers sharing a slot directory (default: 32)")
    evaluation.add_argument("--eval-slot-dir", default=argparse.SUPPRESS,
                            help="Shared CPU admission directory; pot.parallel assigns one per run")
    evaluation.add_argument("--ojbench-python", default=argparse.SUPPRESS, help="Python environment with pinned DMOJ installed")
    evaluation.add_argument("--ojbench-pypy", default=argparse.SUPPRESS, help="PyPy3 executable for official OJBench Python submissions")
    evaluation.add_argument("--ojbench-gpp", default=argparse.SUPPRESS, help="C++17 compiler for official validators")
    return parser


def parameters_from_args(args: argparse.Namespace) -> dict:
    """Merge flags over optional JSON; normalize paths using their own origin."""
    values = vars(args)
    if args.config:
        config_path = Path(args.config).resolve()
        raw = json.loads(config_path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise argparse.ArgumentError(None, "--config must contain a JSON object")
        base = config_path.parent
    else:
        raw = {"backend": {"loss_variant": "paper"},
               "search": {"feedback": "public"},
               "evaluation": {"executor": "subprocess", "timeout_seconds": 10.0, "memory_mb": 1024},
               "dataset": {"selection": "v6"}, "run": {}}
        base = Path.cwd()
    for section in ("backend", "search", "evaluation", "dataset", "run"):
        raw.setdefault(section, {})
        if not isinstance(raw[section], dict):
            raise argparse.ArgumentError(None, f"Configuration section {section} must be an object")
    for section, field in (("backend", "model_path"), ("backend", "adapter_dir"),
                           ("dataset", "path"), ("evaluation", "work_dir"), ("evaluation", "eval_slot_dir")):
        if raw[section].get(field) is not None:
            raw[section][field] = _resolve(raw[section][field], base)
    for name, value in values.items():
        if name in _BACKEND_FLAGS:
            raw["backend"][name] = _resolve(value, Path.cwd()) if name in {"model_path", "adapter_dir"} else value
        elif name in _SEARCH_FLAGS:
            raw["search"][name] = value
        elif name in _EVALUATOR_FLAGS:
            if name in {"ojbench_python", "ojbench_pypy", "ojbench_gpp"}:
                value = _executable_path(value, Path.cwd())
            elif name in {"work_dir", "eval_slot_dir"}:
                value = _resolve(value, Path.cwd())
            raw["evaluation"][name] = value
        elif name == "temperature":
            raw["backend"][name] = raw["search"][name] = value
        elif name == "data_path":
            raw["dataset"]["path"] = _resolve(value, Path.cwd())
        elif name in {"selection", "start_date", "end_date"}:
            raw["dataset"][name] = value
        elif name == "seed" or (name in {"limit", "offset"} and value is not None):
            raw["run"][name] = value
    missing = [flag for section, field, flag in (("backend", "model_path", "--model"),
        ("dataset", "path", "--data")) if not raw[section].get(field)]
    if missing:
        raise argparse.ArgumentError(None, "Required without matching --config values: " + ", ".join(missing))
    return raw


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        parameters = parameters_from_args(args)
        return run(parameters, args.output, limit=args.limit, offset=args.offset, resume=args.resume)
    except argparse.ArgumentError as error:
        parser.error(str(error))
    except Exception as error:
        print(_json({"status": "configuration_error", **_error_details(error)}),
              file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
