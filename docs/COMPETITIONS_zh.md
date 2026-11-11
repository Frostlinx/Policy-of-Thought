# LCB v5 与 OJBench ICPC

LCB 使用标准输入输出或函数式接口；OJBench ICPC 使用官方 DMOJ 判题环境。两者共享 PoT 搜索与在线训练实现。

## 数据范围

| 参数 | 范围 | 判题方式 |
|---|---|---|
| `--selection v5` | LCB `test5.jsonl` 新增 167 题 | 与 v6 相同的函数式/标准输入输出判题 |
| `--selection release_v5` | LCB 1–5 累计 880 题 | 不是论文中的 v5 167 题 |
| `--selection ojbench-icpc` | 官方 `prompts/full.jsonl` 中 `dataset=icpc`、`language=python` 的 73 题 | 官方 DMOJ 判题，保留测试清单、浮点容差、自定义 checker 和 interactor |

ICPC 不用目录总数推断题目数：目录里另有不在官方提示清单中的题目，不补入。
73 题有 4,643 个官方测试点，其中 6 题需要交互。题目顺序保留官方提示文件顺序。
普通题运行完整 Python 程序，使用 PyPy3；不应用 LCB 的 AST 包装或预导入。
输出上限保留官方每题设置（未设置时采用 DMOJ 默认值）；通用执行器的 `--max-output-bytes` 不用于 ICPC。
交互题按题面协议读写和 flush，不能把对话样例回放为固定 stdin。

## 奖励和反馈

OJBench 未提供统一 public/private 划分。本实现只从**题面**提取明确的输入输出样例，
这些样例可以反馈实际输出与错误；不会从判题压缩包中挑选额外样例展示给模型。
题面使用官方提示文件，只替换末尾的回答格式模板，使用 PoT 推理和代码提示。
另有一处已核实的公开题面勘误：`nwerc2024_J` 样例 1 的输出在 OJBench 中漏了第一行，
按 [NWERC 官方原题（PDF 第 22 页、印刷页码 20）](https://2024.nwerc.eu/main/problem-set.pdf#page=22)
补为 `3, 5, 3` 三行。修正同时应用于提示与公开样例，写入 `manifest.json`，原始下载文件及官方判题测试不改。
其余题面未作算法提示或测试信息补充。

所有官方判题测试均由独立评测进程使用。测试输入、答案、实际输出、错误和交互日志
不会返回给搜索或模型。返回的逐测试点状态与计数用于训练及效率日志，不进入提示词。
`private_*` 字段在 ICPC 下表示“仅供判题的官方测试”，不是官方数据划分。

```text
reward = (题面样例通过数 + 官方判题测试通过数)
         / (题面样例数 + 官方判题测试数)
```

每个测试点等权，不使用比赛 points 权重。题面样例额外评测；如果官方测试包也包含
相同样例，两个集合均保留，不隐式改变官方清单。交互题的题面对话仅作为说明，
不另计可回放公开测试；其奖励来自全部真实交互测试。
只在所有测试点通过时成功，不采用官方工具输出的 1/8、1/4、1/2 部分清单成绩早停。
测试点执行顺序可并行，但不跳过失败后的测试。

此协议会用判题测试指导训练、选择和早停，固定记录 `heldout_tests_valid=false`。
不得把结果当作论文中独立隐藏测试的 pass@1。少量测试反馈/独立留出评测属于另一个实验协议，
不属于本发布版的运行协议。

## 准备环境与数据

所有 HF 数据只从 hf-mirror 下载，凭证可通过环境变量 `HF_TOKEN` 提供，不写入源码。
数据固定到 revision；LCB v5/v6 核验固定 SHA-256，ICPC 全部文件核验上游 LFS SHA-256 或 Git blob ID 及长度。

```bash
python -m pip install -e '.[data]' --no-deps  # 已有训练环境时保留其依赖
# 如缺少数据依赖，另安装 pyarrow 和 PyYAML。
python scripts/download_lcb.py --selection v5 --output ./data/lcb-v5 --no-proxy --aria2
python scripts/download_ojbench.py --output ./data/ojbench-icpc --no-proxy --aria2 --download-workers 6
```

离线数据支持 `download_lcb.py --raw-dir DIR` 和 `download_ojbench.py --raw-dir DIR`。
后者目录必须包含 `prompts/full.jsonl` 及 `ICPC/<题号>/` 下完整官方文件。
准备后的 `judge_manifest.json` 记录所有评测文件的哈希；运行前核验，恢复运行时绑定数据标识。
原始大文件保留在数据目录，发布源码包不含题面、测试集、模型或运行日志。

OJBench 需要 Linux、C++17 编译器、PyPy3 和**独立 Python 判题环境**。
遵循 [OJBench 官方安装步骤](https://github.com/He-Ren/OJBench#-installation)，
DMOJ 固定为 `f098cd3a49a60186d1fadde5132329ec5f4f2213`。
本实现直接调用该版本 DMOJ 的 API，采用 OJBench 对 Python 的 `PYPY3` 设置、
10 秒 CPU 时限、1 GiB 内存默认值和每题原有 checker/interactor。
无需用户编写 config.yaml，路径通过命令行传入。数据自带的 `init.yml` 是官方判题元数据，
不属于用户配置文件，也不会替换成未经验证的自制比较器。

```bash
python3 -m venv ./judge-venv
./judge-venv/bin/pip install 'cython<3' wheel 'setuptools<81'
# 先按官方说明安装系统编译依赖（Python headers、libseccomp headers 等）。
git clone https://github.com/DMOJ/judge-server.git ./judge-server
git -C ./judge-server checkout f098cd3a49a60186d1fadde5132329ec5f4f2213
./judge-venv/bin/pip install --no-build-isolation ./judge-server
```

PyPy 应安装在独立目录，不要把测试文件放进其安装前缀。该目录将作为解释器依赖的只读访问范围。
验证环境采用 PyPy 7.3.17 / Python 3.10.14，避免系统旧 PyPy 只支持 Python 3.8 的差异。

## 参数式运行

先创建 tmux，再运行。下例 GPU 0 应替换为你有权使用的空闲 GPU；
`--shared-gpus` 仅在明确允许共享已有任务时添加。

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

ICPC 的 `subprocess` 默认会自动选择官方 `ojbench` 执行器；不允许回退到 LCB 比较器或 Docker 通用比较器。
判题服务在每题搜索期间复用多个 CPU 工作进程，跨模型实例用共享槽位限制总 CPU 并发。
`--limit 3 --simulations 1 --no-train` 可做接口试跑；这不是完整 PoT 正确率实验。

## 输出

保留 `summary.json` 汇总及 `process.json` 每题一行，另有 JSONL 别名。
每题包括题号、是否整题成功、联合通过率、公开/判题测试计数、用时、生成与更新次数。
`evaluation_timing.jsonl` 只含题号、测试索引、通过状态、执行/排队耗时，不含判题测试内容。
判题器初始化、编译辅助程序或测试数据故障记录为基础设施错误，不计为模型错误答案。

使用各自运行目录的 summary.json 和 process.json 报告结果，区分完成数、答对数与基础设施失败次数。
