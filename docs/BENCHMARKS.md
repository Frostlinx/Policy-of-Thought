# HumanEval and MBPP

Both benchmarks require a complete Python module, including imports, functions,
and helpers, using the function names and signatures required by the problem.

## Data and rewards

| Property | HumanEval | MBPP sanitized test |
|---|---|---|
| Selection | `humaneval` | `mbpp-sanitized` |
| Tasks | 164 | 257 |
| Model sees | Function signature and docstring examples | Statement, published tests, setup/imports |
| Executable public tests | None; docstring examples are not converted to assertions | Each `test_list` entry |
| Reward | Passed assertion-location groups / total groups | Passed test entries / total entries |
| Success | Original full check and all assertion groups pass | All published tests pass |

HumanEval retains the original `test + check(entry_point)` check. For each
assertion location, an additional reward probe runs the full checking program,
but only a failed assertion at the selected location raises an error. The
conditions at other locations are still evaluated. Repeated execution of one
assertion in a loop counts as one group.
An earlier runtime exception can still cause later probes to fail.

The full check determines success; assertion groups determine reward. The pinned dataset
has 1,181 assertion reward units and 164 full checks. Checks containing
`try`/`except` are not rewritten; they fall back to the binary full-check reward.
Each unit executes independently.

MBPP uses the sanitized test split: 257 tasks and 776 published test entries,
with 3–5 entries per task. Multiline entries remain intact. The private pass
rate is `null` because this split supplies public tests only.

Both adapters use all-test rewards and record `heldout_tests_valid=false`.

## Download and run

```bash
python scripts/download_benchmark.py --selection humaneval --output ./data/humaneval
python scripts/download_benchmark.py --selection mbpp-sanitized --output ./data/mbpp-sanitized
tmux new-session -s pot-humaneval
bash scripts/run_benchmark.sh \
  --model ./models/Qwen3-4B-Instruct-2507 \
  --selection humaneval --data ./data/humaneval --output ./runs/humaneval \
  --gpus 0 --limit 164
```

For MBPP, change the selection, data and output paths, and use `--limit 257`.
Parquet preparation requires PyArrow. Downloads use hf-mirror with a pinned
revision and SHA-256; `--raw-file` accepts an offline file. Normalized output
contains `public.jsonl`, `private_tests.jsonl`, and `manifest.json`. Raw Parquet files remain under `raw/`.

## Validate the evaluator

```bash
python scripts/validate_benchmark.py \
  --selection humaneval --data ./data/humaneval \
  --raw-file ./data/humaneval/raw/test.parquet \
  --output ./runs/humaneval-reference --workers 8
```

This command validates the evaluator by executing the dataset's reference
solutions.

Sources: [HumanEval](https://github.com/openai/human-eval),
[MBPP](https://github.com/google-research/google-research/tree/master/mbpp).
Upstream dataset and model terms continue to apply.
