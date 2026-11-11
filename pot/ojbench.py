"""OJBench ICPC normalization. Statement examples are the only public tests.

The upstream archive has no public/private labels. We keep all official judge
cases evaluator-only; those names describe our feedback boundary, not an
upstream split. Judge files are never serialized into PublicProblem.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import zipfile

from .data import PrivateTestStore, PublicProblem, TestCase

OJ_REPO = "He-Ren/OJBench_testdata"
OJ_REVISION = "61cf9986f22c25d08e1657b03742124099c74353"
DMOJ_REVISION = "f098cd3a49a60186d1fadde5132329ec5f4f2213"
OJ_CODE_REVISION = "5e94480b1e135b98855cf5bc81213c256aff5b17"
STATEMENT_ERRATA = {
    'nwerc2024_J': {
        'before': '### Sample Output 1\n```\n5\n3\n```',
        'after': '### Sample Output 1\n```\n3\n5\n3\n```',
        'source': 'https://2024.nwerc.eu/main/problem-set.pdf#page=22',
        'reason': 'OJBench statement omits the first output line; official public sample is 3, 5, 3.',
    },
}

DMOJ_CORE_SHA256 = {'dmoj/problem.py': '0a0571e8700b5db5e7746be9466a20844149b5c9671d5833b77ad4cd5c42dc6f', 'dmoj/graders/standard.py': 'c61de4e7c9105ab830e9c630b0428930823e4765ec390ee618ab44451a24bb5e', 'dmoj/graders/bridged.py': '982924ec6fdcf37d279fa58148ba90e96ed5d07cf96303de98d2c1d870500ce7', 'dmoj/checkers/bridged.py': 'c7b9397b446be77762f3555a041d00c579a919c564ea2a191377041bb05ac9a3', 'dmoj/checkers/floats.py': '21198f4eb9ac80cbff9dc091ffca34b1736d31879b6eabff4ed310e2b7c505d2', 'dmoj/checkers/standard.py': '33ac5ad17dc7403552c8dba014774f31d0f075b2ed5856206f8884ba45b133f7', 'dmoj/contrib/default.py': 'e3d513f3b8f20706d5404b73c6cf788cd6511a4aac5ba67269f30b23c14c55e2', 'dmoj/contrib/base.py': '587c2d64e979977903cf6a7ea9ec80a01b8f69b61aacaaa0980429d0ae2cd717'}


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def pinned_files():
    return json.loads(Path(__file__).with_name('ojbench_checksums.json').read_text(encoding='utf-8'))


def verify_raw_file(path, spec):
    """Validate against immutable hub LFS hashes or Git blob IDs, not local hashes."""
    path = Path(path)
    if not path.is_file() or path.stat().st_size != spec['bytes']:
        return False
    digest = hashlib.sha256() if 'sha256' in spec else hashlib.sha1(
        ('blob ' + str(spec['bytes']) + '\0').encode())
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest() == spec.get('sha256', spec.get('git_blob_sha1'))


def statement_samples(text):
    """Read only explicitly labelled, fenced statement samples; fail on ambiguity."""
    pattern = re.compile(r"(?im)^[#* \t]*(?:sample[ \t]+)?(input|output)[ \t]*(\d*)[ *:#\t]*\r?\n[ \t\r\n]*```[^\n]*\n(.*?)^```[ \t]*$", re.S | re.M)
    items = list(pattern.finditer(text))
    samples = []
    if not items and re.search(r"(?im)^#+ Sample Input/Output\s*$", text):
        # Pinned swerc2024_M uses adjacent input/output fences under Sample N.
        paired = re.compile(r"(?m)^#+ Sample \d+\s*\n```[^\n]*\n(.*?)^```\s*\n```[^\n]*\n(.*?)^```[ \t]*$", re.S)
        return tuple(TestCase(m[1], m[2]) for m in paired.finditer(text))
    for i in range(0, len(items), 2):
        if i + 1 >= len(items):
            raise ValueError("Unpaired public sample")
        a, b = items[i:i+2]
        if a[1].lower() != "input" or b[1].lower() != "output" or a[2] != b[2]:
            raise ValueError("Ambiguous public sample ordering")
        samples.append(TestCase(a[3].replace("\r\n", "\n"), b[3].replace("\r\n", "\n")))
    return tuple(samples)


def correct_statement(problem_id, statement):
    erratum = STATEMENT_ERRATA.get(problem_id)
    if erratum is None:
        return statement
    if statement.count(erratum['before']) != 1:
        raise ValueError('Public statement erratum no longer matches the pinned source')
    return statement.replace(erratum['before'], erratum['after'], 1)


def _link_or_copy(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, target)
    except OSError:
        shutil.copyfile(source, target)


def normalize_ojbench(rows, raw_root, output):
    import yaml
    raw_root, output = Path(raw_root), Path(output)
    if any(not verify_raw_file(raw_root / name, spec) for name, spec in pinned_files().items()):
        raise ValueError('Raw OJBench assets differ from the pinned upstream checksums')
    selected = [r for r in rows if str(r.get("dataset", "")).lower() == "icpc" and r.get("language") == "python"]
    if len(selected) != 73 or len({r["id"] for r in selected}) != 73:
        raise ValueError("Expected exactly the official 73 ICPC Python prompts")
    public, private, stats = [], {}, []
    judge_root = output / "judge"
    for row in selected:
        pid = str(row["id"])
        if not re.fullmatch(r"(?:nwerc|swerc)20\d{2}_[A-Z]", pid):
            raise ValueError("Invalid ICPC problem ID")
        source = raw_root / "ICPC" / pid
        cfg = yaml.safe_load((source / "init.yml").read_text(encoding="utf-8"))
        cases = cfg["test_cases"]
        # This pinned subset has flat test lists. Never silently drop subtasks.
        if not cases or any("batched" in c for c in cases):
            raise ValueError("Unsupported empty or batched OJBench case list")
        if cfg.get("archive") != "data.zip":
            raise ValueError("Unexpected official archive")
        with zipfile.ZipFile(source / "data.zip") as archive:
            names = set(archive.namelist())
            if any(c.get(k) and c[k] not in names for c in cases for k in ("in", "out")):
                raise ValueError("Official test archive is missing a configured member")
        target = judge_root / pid
        target.mkdir(parents=True, exist_ok=False)
        for file in sorted(source.rglob("*")):
            if file.is_file() and not file.name.endswith(".part"):
                _link_or_copy(file, target / file.relative_to(source))
        interactive = bool(cfg.get("interactive"))
        # Keep the task wording and examples, replace only upstream answer-format
        # boilerplate with PoT's established reasoning + last-closed-block contract.
        statement = correct_statement(pid, row["prompt"].split("### Format:", 1)[0].rstrip())
        samples = () if interactive else statement_samples(statement)
        if not interactive and not samples:
            raise ValueError(f"No unambiguous statement samples for {pid}")
        if samples:
            sample_root = judge_root / (pid + "-public")
            sample_root.mkdir(parents=True)
            for file in source.rglob("*"):
                if file.is_file() and file.suffix in {".h", ".hpp", ".cpp", ".cc", ".c", ".py"}:
                    _link_or_copy(file, sample_root / file.relative_to(source))
            sample_cfg = copy.deepcopy(cfg)
            sample_cfg.pop("archive", None)
            sample_cfg["test_cases"] = []
            for i, sample in enumerate(samples):
                inp, out = f"sample-{i}.in", f"sample-{i}.ans"
                (sample_root / inp).write_text(sample.input, encoding="utf-8")
                (sample_root / out).write_text(sample.output, encoding="utf-8")
                sample_cfg["test_cases"].append({"in": inp, "out": out, "points": 1})
            (sample_root / "init.yml").write_text(yaml.safe_dump(sample_cfg), encoding="utf-8")
        public.append(PublicProblem("ICPC/" + pid, pid, "ojbench", statement.splitlines()[0].lstrip("# "),
            statement, "", str(row.get("difficulty", "")), "", samples, None, "ojbench-icpc",
            "interactive" if interactive else "stdio"))
        # Placeholders carry counts only. Actual judge inputs are read exclusively
        # by the separate official-judge process, never by the model/search.
        private["ICPC/" + pid] = tuple(TestCase("", "") for _ in cases)
        stats.append({"problem_id": "ICPC/" + pid, "judge_cases": len(cases),
                      "statement_samples": len(samples), "interactive": interactive})
    files = [{"path": f.relative_to(output).as_posix(), "bytes": f.stat().st_size, "sha256": sha256(f)}
             for f in sorted(judge_root.rglob("*")) if f.is_file()]
    (output / "judge_manifest.json").write_text(json.dumps({"files": files, "problems": stats}, indent=2) + "\n", encoding="utf-8")
    return public, PrivateTestStore(private), stats


_verified_assets = {}


class OJPrivateStore(PrivateTestStore):
    """Evaluator-only paths, not part of prompt-safe problem records."""
    def __init__(self, directory, public, store):
        super().__init__({p.problem_id: store._for_evaluation(p.problem_id) for p in public})
        self.judge_root = Path(directory).resolve() / "judge"
        self.manifest_path = Path(directory).resolve() / "judge_manifest.json"

    def verify(self):
        root = self.judge_root.parent
        manifest_digest = sha256(self.manifest_path)
        items = json.loads(self.manifest_path.read_text(encoding="utf-8"))["files"]
        stamps = []
        for item in items:
            file = (root / item["path"]).resolve()
            if not file.is_relative_to(self.judge_root) or not file.is_file():
                raise ValueError("OJBench judge data is missing or outside its root")
            stat = file.stat()
            stamps.append((stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns, stat.st_ino))
        key = (str(self.manifest_path), manifest_digest)
        if _verified_assets.get(key) == stamps:
            return
        for item in items:
            file = (root / item["path"]).resolve()
            if not file.is_relative_to(self.judge_root) or not file.is_file() or file.stat().st_size != item["bytes"] or sha256(file) != item["sha256"]:
                raise ValueError("OJBench judge data differs from the prepared manifest")
        _verified_assets[key] = stamps
