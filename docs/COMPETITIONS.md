# LiveCodeBench and OJBench ICPC

LCB uses standard input/output or function interfaces. OJBench ICPC uses its
official DMOJ judging environment. Both share the PoT search and online
training implementation.

## Dataset scope

| Selection | Scope | Judge |
|---|---|---|
| `v5` | 167 new tasks from LCB `test5.jsonl` | Same function/stdio evaluator as v6 |
| `release_v5` | 880 cumulative tasks from releases 1–5 | Not the paper's 167-task v5 increment |
| `ojbench-icpc` | 73 entries with `dataset=icpc` and `language=python` in official `prompts/full.jsonl` | Official DMOJ, preserving test manifests, floating-point tolerances, custom checkers and interactors |

The ICPC task list follows the official prompt file order. Extra problem
directories absent from that list are not added. The 73 tasks contain 4,643
official test cases; six tasks are interactive.

Non-interactive tasks run as complete Python programs under PyPy3, without
LCB's AST wrapping or pre-imports. Output limits follow each problem's official
settings, falling back to DMOJ defaults. The generic executor's
`--max-output-bytes` does not apply to ICPC. Interactive programs must follow
the statement's read/write and flush protocol; sample conversations are not
replayed as fixed standard input.

## Rewards and feedback

OJBench does not define a uniform public/private split. This implementation
extracts explicit input/output examples only from **problem statements**.
Their outputs and errors may be shown to the model. No additional examples
are selected from judge archives for model feedback.

Statements come from the official prompt file. Only the final response-format
template is replaced with PoT's reasoning-and-code prompt. One public statement
erratum is applied: OJBench omits the first output line of sample 1 in
`nwerc2024_J`. The sample is restored to the three lines `3`, `5`, `3`, following
the [official NWERC statement (PDF page 22, printed page 20)](https://2024.nwerc.eu/main/problem-set.pdf#page=22).
The correction applies to the prompt and public example and is recorded in
`manifest.json`. Raw downloads and official judge tests remain unchanged.
No further algorithm hints or test information are added to statements.

Official judge tests run in a separate evaluation process. Their inputs,
expected answers, actual outputs, errors, and interaction logs are not returned
to search or model context. Per-test statuses and counts support scalar rewards
and timing logs; they are not placed in prompts. ICPC `private_*` fields mean
judge-only official tests, not an upstream dataset split.

```text
reward = (passed statement examples + passed official judge tests)
         / (statement examples + official judge tests)
```

Every test has equal weight; contest points do not weight rewards. Statement
examples are evaluated separately. If an official archive also contains the
same example, both sets are retained. Interactive sample conversations provide
context only, with no additional replayable public tests; rewards come from
the actual interactive judge tests.

Success requires every test to pass. Partial 1/8, 1/4 or 1/2 test lists from
upstream tools do not trigger early stopping. Tests may run concurrently, but
remaining tests are not skipped after a failure. This protocol uses judge tests
for training, selection and stopping, and records `heldout_tests_valid=false`.
Report results under this adaptive protocol rather than as held-out pass@1.
See [method correspondence](METHOD.md) for the paper's split-test protocol.

## Prepare data and environments

HF downloads use hf-mirror. Supply credentials through `HF_TOKEN`, not source
files. Revisions are pinned. LCB v5/v6 files have fixed SHA-256 checks; all ICPC
files are verified against upstream LFS SHA-256 or Git blob IDs and lengths.

```bash
python -m pip install -e '.[data]' --no-deps  # Preserve an existing training environment.
# Install pyarrow and PyYAML separately if missing.
python scripts/download_lcb.py --selection v5 --output ./data/lcb-v5 --no-proxy --aria2
python scripts/download_ojbench.py --output ./data/ojbench-icpc --no-proxy --aria2 --download-workers 6
```

Offline inputs are supported by `download_lcb.py --raw-dir DIR` and
`download_ojbench.py --raw-dir DIR`. The OJBench raw directory must contain
`prompts/full.jsonl` and the complete official files under `ICPC/<problem-id>/`.
The prepared `judge_manifest.json` records hashes for all evaluation assets.
They are checked before execution and bound to dataset identity on resume.
Raw assets remain in the data directory; the source package does not bundle
statements, tests, models or run logs.

OJBench requires Linux, a C++17 compiler, PyPy3, and a **separate Python judge
environment**. Follow the [official OJBench installation instructions](https://github.com/He-Ren/OJBench#-installation).
DMOJ is pinned to `f098cd3a49a60186d1fadde5132329ec5f4f2213`. The adapter calls
that revision's API, using OJBench's `PYPY3` configuration, a 10-second CPU
limit, a default 1 GiB memory limit, and the original checkers/interactors.
Paths are CLI arguments; no user-authored YAML configuration is required.
Dataset `init.yml` files are official judge metadata and are retained.

```bash
python3 -m venv ./judge-venv
./judge-venv/bin/pip install 'cython<3' wheel 'setuptools<81'
# Install the official system build dependencies first, including Python and libseccomp headers.
git clone https://github.com/DMOJ/judge-server.git ./judge-server
git -C ./judge-server checkout f098cd3a49a60186d1fadde5132329ec5f4f2213
./judge-venv/bin/pip install --no-build-isolation ./judge-server
```

Install PyPy in a dedicated directory, with no test data under its installation
prefix. That directory is granted read access for interpreter dependencies.
The validated runtime is PyPy 7.3.17 / Python 3.10.14; older system PyPy versions
may only support Python 3.8.

## Run from the command line

Start tmux first. Replace GPU 0 with an available device you may use. Add
`--shared-gpus` only where sharing existing workloads is intended.

```bash
tmux new-session -s pot-lcb-v5
python -m pot.parallel \
  --model ./models/Qwen3-4B-Instruct-2507 \
  --data ./data/lcb-v5 --selection v5 --output ./runs/lcb-v5 \
  --gpus 0 --workers-per-gpu 1 --minimum-free-mib 28000 \
  --memory-utilization 0.20 --kv-cache-gib 2.5 \
  --eval-workers 16 --eval-global-workers 64 \
  --simulations 20 --width 3 --max-depth 8 --max-new-tokens 2048 --seed 42
```

```bash
tmux new-session -s pot-ojbench-icpc
python -m pot.parallel \
  --model ./models/Qwen3-4B-Instruct-2507 \
  --data ./data/ojbench-icpc --selection ojbench-icpc --output ./runs/ojbench-icpc \
  --executor ojbench --ojbench-python ./judge-venv/bin/python \
  --ojbench-pypy /path/to/pypy/bin/pypy3 --ojbench-gpp /usr/bin/g++ \
  --gpus 0 --workers-per-gpu 1 --minimum-free-mib 28000 \
  --memory-utilization 0.20 --kv-cache-gib 2.5 \
  --eval-workers 16 --eval-global-workers 64 --timeout 10 --memory-mb 1024 \
  --simulations 20 --width 3 --max-depth 8 --max-new-tokens 2048 --seed 42
```

For ICPC, the default `subprocess` selection automatically resolves to the
official `ojbench` executor. It cannot fall back to LCB's comparator or the
generic Docker comparator. The judge reuses CPU worker processes during each
problem's search; shared slots limit total CPU concurrency across model workers.
`--limit 3 --simulations 1 --no-train` is an interface smoke run, not a full PoT
accuracy experiment.

## Outputs

`summary.json` contains aggregates and `process.json` contains one JSON object
per completed problem, with JSONL aliases also available. Per-problem records
include the ID, all-tests-pass status, joint pass rate, public/judge test counts,
elapsed time, generation count and update count.

`evaluation_timing.jsonl` records problem IDs, test indices, pass status, and
execution/queue times, without judge-test contents. Judge initialization,
auxiliary compilation and test-data failures are infrastructure errors, not
model wrong answers. Report completed tasks, solved tasks and infrastructure
failures separately from each run's summary and process files.
