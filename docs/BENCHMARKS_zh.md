# HumanEval 与 MBPP

两个基准都要求模型返回完整 Python 模块，包括所需导入、函数和辅助函数。不要返回函数体片段或 `Solution` 类；函数名称和签名以题目接口为准。

## 数据与奖励

| 项目 | HumanEval | MBPP sanitized test |
|---|---|---|
| selection | humaneval | mbpp-sanitized |
| 题数 | 164 | 257 |
| 模型可见 | 函数签名、docstring 示例 | 题面、发布测试及 setup/imports |
| 可执行公开测试 | 无，docstring 不自动转成断言 | 每个 test_list 条目 |
| 奖励 | 通过的断言位置组 / 总组数 | 通过测试条目数 / 总条目数 |
| 成功门槛 | 原始完整检查和全部断言组都通过 | 全部发布测试通过 |

HumanEval 保留原始 `test + check(entry_point)` 完整检查。额外为每处 assert 语句生成一个奖励探针：运行完整检查程序，只让选中位置的断言失败触发错误，其余位置仍计算原条件。循环里的同一 assert 位置算一个组，不按动态调用次数拆开。较早运行时异常仍可导致后续探针失败。

完整检查作为成功门槛，不额外计入奖励分母。固定数据有 1181 个断言奖励单元和 164 个完整检查。包含 try/except 的检查程序不改写，回退为完整检查的二值奖励。每个单元独立执行。

MBPP 使用 sanitized 的 test 划分，共 257 题、776 个发布测试条目，条目数为每题 3–5 个。多行条目不再拆分；不虚构额外私有测试。结果中的 private 通过率为 null，而不是 0%。未使用 EvalPlus 扩展测试或另加 MBPP three-shot 模板。

两者均采用全测试奖励协议，记录 `heldout_tests_valid=false`。HumanEval 断言组奖励是本适配的计分约定；模型最终是否答对仍要求完整检查通过。

## 下载和运行

```bash
python scripts/download_benchmark.py --selection humaneval --output ./data/humaneval
python scripts/download_benchmark.py --selection mbpp-sanitized --output ./data/mbpp-sanitized
tmux new-session -s pot-humaneval
bash scripts/run_benchmark.sh \
  --model ./models/Qwen3-4B-Instruct-2507 \
  --selection humaneval --data ./data/humaneval --output ./runs/humaneval \
  --gpus 0 --limit 164
```

MBPP 改 selection/data/output，并设置 `--limit 257`。Parquet 准备需要 pyarrow。
下载从 hf-mirror 开始，固定 revision 和 SHA256；离线文件可用 `--raw-file`。
归一化输出包含 public.jsonl、private_tests.jsonl 和 manifest.json，不含参考实现；原始 parquet 保留在 raw 目录。

## 判题适配验证

```bash
python scripts/validate_benchmark.py \
  --selection humaneval --data ./data/humaneval \
  --raw-file ./data/humaneval/raw/test.parquet \
  --output ./runs/humaneval-reference --workers 8
```

此命令执行数据中的参考答案来检查适配器，所得通过数不是模型成绩，不得与模型评测混报。它不将参考答案注入模型提示。

来源：[HumanEval](https://github.com/openai/human-eval)、[MBPP](https://github.com/google-research/google-research/tree/master/mbpp)。上游数据与模型各自的使用条款继续适用。
