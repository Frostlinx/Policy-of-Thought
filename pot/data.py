"""LiveCodeBench loading with an explicit public/private boundary.

PoT's LCB v6 is the 175-problem delta (test6.jsonl), not the 1,055-problem
cumulative release_v6. No dataset loading scripts or unrestricted pickle run.
"""
from __future__ import annotations

import base64
import io
import json
import pickle
import zlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

LCB_REPO = "livecodebench/code_generation_lite"
HF_MIRROR = "https://hf-mirror.com"
SELECTION_FILES = {
    "v5": ("test5.jsonl",),
    "release_v5": ("test.jsonl", "test2.jsonl", "test3.jsonl", "test4.jsonl", "test5.jsonl"),
    "v6": ("test6.jsonl",),
    "release_v6": ("test.jsonl", "test2.jsonl", "test3.jsonl", "test4.jsonl", "test5.jsonl", "test6.jsonl"),
}
EXPECTED_COUNTS = {"v5": 167, "release_v5": 880, "v6": 175, "release_v6": 1055}
BENCHMARK_SELECTIONS = ("humaneval", "mbpp-sanitized", "ojbench-icpc")
SELECTIONS = (*SELECTION_FILES, *BENCHMARK_SELECTIONS)


@dataclass(frozen=True)
class TestCase:
    input: str
    output: str
    testtype: str = "stdin"
    role: str = "reward"

    def __post_init__(self):
        if self.testtype not in ("stdin", "functional", "python_assert"):
            raise ValueError("Unsupported test type")
        if self.role not in ("reward", "gate") or (self.role == "gate" and self.testtype != "python_assert"):
            raise ValueError("Unsupported test role")
        if not isinstance(self.input, str) or not isinstance(self.output, str):
            raise ValueError("Test input and output must be strings")


@dataclass(frozen=True)
class PublicProblem:
    problem_id: str
    question_id: str
    platform: str
    question_title: str
    question_content: str
    starter_code: str
    difficulty: str
    contest_date: str
    public_tests: tuple[TestCase, ...]
    func_name: str | None = None
    benchmark: str = "lcb"
    interface: str = "auto"

    def to_dict(self) -> dict[str, Any]:
        """An allowlisted prompt-safe serialization; arbitrary metadata is excluded."""
        return asdict(self)

    @property
    def public_test_cases(self) -> tuple[TestCase, ...]:
        return self.public_tests


class _StringOnlyUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        raise ValueError("Dataset pickle globals are forbidden")

    def persistent_load(self, pid):
        raise ValueError("Dataset pickle references are forbidden")


def _decode_private(value: Any) -> list[dict[str, Any]]:
    """Decode the official JSON or base64(zlib(pickle(JSON-string))) envelope.

    The pickle is permitted to contain a string only; no imported globals or
    reconstruction functions may execute. Error messages never include data.
    """
    if isinstance(value, list):
        return value
    try:
        result = json.loads(value)
    except (ValueError, TypeError):
        try:
            compressed = base64.b64decode(value, validate=True)
            decoder = zlib.decompressobj()
            unpacked = decoder.decompress(compressed, 512 * 1024 * 1024)
            if decoder.unconsumed_tail or not decoder.eof:
                raise ValueError("Private test envelope exceeds limit or is incomplete")
            payload = _StringOnlyUnpickler(io.BytesIO(unpacked)).load()
            if not isinstance(payload, str):
                raise ValueError("Private envelope must contain a JSON string")
            result = json.loads(payload)
        except Exception:
            raise ValueError("Invalid private-test encoding") from None
    if not isinstance(result, list):
        raise ValueError("Private tests must be a list")
    return result


def _tests(value: Any, *, private: bool = False) -> tuple[TestCase, ...]:
    try:
        rows = _decode_private(value) if private else (json.loads(value) if isinstance(value, str) else value)
        if not isinstance(rows, list):
            raise ValueError
        return tuple(TestCase(input=t["input"], output=t["output"], testtype=t.get("testtype", "stdin"), role=t.get("role", "reward")) for t in rows)
    except Exception:
        raise ValueError("Invalid private tests" if private else "Invalid public tests") from None


class PrivateTestStore:
    """Evaluator-owned test storage. Never serialize this into model state/logs.

    This is a software boundary, not a security boundary between Python modules.
    Private records are withheld from the search/prompt API. Docker candidates
    cannot access this store's file through their mounted filesystem; ordinary
    subprocess candidates share the host user's filesystem permissions.
    """
    __slots__ = ("__tests",)

    def __init__(self, tests: Mapping[str, tuple[TestCase, ...]]):
        self.__tests = dict(tests)

    def __repr__(self) -> str:
        return "<PrivateTestStore redacted>"

    def _for_evaluation(self, problem_id: str) -> tuple[TestCase, ...]:
        if problem_id not in self.__tests:
            raise ValueError("No private tests for requested problem")
        return self.__tests[problem_id]

    def _write(self, path: Path) -> None:
        with path.open("w", encoding="utf-8", newline="\n") as stream:
            for problem_id, tests in self.__tests.items():
                stream.write(json.dumps({"problem_id": problem_id, "tests": [asdict(t) for t in tests]}, ensure_ascii=False) + "\n")
        try:
            path.chmod(0o600)
        except OSError:
            pass

    @classmethod
    def _read(cls, path: Path) -> "PrivateTestStore":
        tests = {}
        for row in read_jsonl(path):
            key = row["problem_id"]
            if key in tests:
                raise ValueError("Duplicate private problem ID")
            tests[key] = _tests(row["tests"], private=True)
        return cls(tests)


def split_records(records: Iterable[Mapping[str, Any]]) -> tuple[list[PublicProblem], PrivateTestStore]:
    public, private = [], {}
    for row in records:
        platform = str(row["platform"])
        question_id = str(row["question_id"])
        problem_id = f"{platform}/{question_id}"
        if problem_id in private:
            raise ValueError("Duplicate problem ID")
        metadata = row.get("metadata", {})
        if isinstance(metadata, str):
            metadata = json.loads(metadata)
        func_name = metadata.get("func_name")
        if func_name is not None and not isinstance(func_name, str):
            raise ValueError("Invalid function name")
        public.append(PublicProblem(
            problem_id=problem_id, question_id=question_id, platform=platform,
            question_title=str(row["question_title"]), question_content=str(row["question_content"]),
            starter_code=str(row.get("starter_code") or ""), difficulty=str(row.get("difficulty", "")),
            contest_date=str(row["contest_date"]), public_tests=_tests(row["public_test_cases"]),
            func_name=func_name,
        ))
        private[problem_id] = _tests(row["private_test_cases"], private=True)
    return public, PrivateTestStore(private)


def read_jsonl(path: str | Path) -> Iterable[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError
            except (ValueError, TypeError):
                # JSONDecodeError includes source text: suppress it at this boundary.
                raise ValueError(f"Invalid JSON object on line {number}") from None
            yield row


def save_dataset(directory: str | Path, public: list[PublicProblem], private: PrivateTestStore, manifest: Mapping[str, Any]) -> None:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "public.jsonl").open("w", encoding="utf-8", newline="\n") as stream:
        for problem in public:
            stream.write(json.dumps(problem.to_dict(), ensure_ascii=False) + "\n")
    private._write(directory / "private_tests.jsonl")
    (directory / "manifest.json").write_text(json.dumps(dict(manifest), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def load_dataset(directory: str | Path, selection: str = "v6", start_date: str | None = None,
                 end_date: str | None = None, limit: int | None = None) -> tuple[list[PublicProblem], PrivateTestStore]:
    """Load normalized files or raw official JSONL without remote code execution."""
    if selection not in SELECTIONS:
        raise ValueError("Unsupported dataset selection")
    directory = Path(directory)
    if selection in BENCHMARK_SELECTIONS and (directory.is_file() or not (directory / "public.jsonl").is_file()):
        raise ValueError("Prepare this benchmark with scripts/download_benchmark.py first")
    if selection in BENCHMARK_SELECTIONS and (start_date or end_date):
        raise ValueError("Date filters apply only to LiveCodeBench")
    if directory.is_file():
        public, private = split_records(read_jsonl(directory))
    elif (directory / "public.jsonl").exists():
        manifest_path = directory / "manifest.json"
        if selection in BENCHMARK_SELECTIONS and not manifest_path.exists():
            raise ValueError("Benchmark data requires a protocol manifest")
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("selection", selection) != selection:
                raise ValueError("Dataset manifest selection does not match requested selection")
            if selection in BENCHMARK_SELECTIONS:
                from .benchmarks import benchmark_protocol
                if manifest.get("benchmark_protocol") != benchmark_protocol(selection):
                    raise ValueError("Benchmark reward protocol changed; prepare data in a new directory")
        public = []
        for row in read_jsonl(directory / "public.jsonl"):
            # Never splat unknown fields, including accidental private fields.
            allowed = {k: row[k] for k in PublicProblem.__dataclass_fields__ if k in row and k != "public_tests"}
            allowed["public_tests"] = _tests(row["public_tests"])
            public.append(PublicProblem(**allowed))
        private = PrivateTestStore._read(directory / "private_tests.jsonl")
        if selection == "ojbench-icpc":
            from .ojbench import OJPrivateStore
            private = OJPrivateStore(directory, public, private)
    else:
        public, private = split_records(row for filename in SELECTION_FILES[selection] for row in read_jsonl(directory / filename))
    expected_benchmark = selection if selection in BENCHMARK_SELECTIONS else "lcb"
    if any(p.benchmark != expected_benchmark for p in public):
        raise ValueError("Problem benchmark does not match requested selection")
    if len({p.problem_id for p in public}) != len(public):
        raise ValueError("Duplicate problem ID")
    # Compare dates (not timestamps) so end_date includes the entire day.
    public = [p for p in public if (start_date is None or p.contest_date[:10] >= start_date)
              and (end_date is None or p.contest_date[:10] <= end_date)]
    if limit is not None:
        if limit < 0:
            raise ValueError("limit must be nonnegative")
        public = public[:limit]
    return public, private
