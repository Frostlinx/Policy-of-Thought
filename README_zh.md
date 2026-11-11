# Policy of Thoughts 使用说明

PoT 在解一道题的过程中交替执行搜索与训练：先选取树上的一个节点，生成三个候选程序并运行测试；再依据同组候选的相对奖励更新 LoRA，让下一轮生成使用更新后的策略。每道题结束后重置 LoRA 和优化器，基础模型始终冻结。

[论文 v2](https://arxiv.org/abs/2601.20379v2) · [方法与论文对应](docs/METHOD.md) · [发布说明](docs/RELEASE.md)

本项目包含 LCB、HumanEval、MBPP 和 OJBench 的数据与判题适配。

## 评测协议

当前采用全测试奖励：公开与隐藏/官方测试共同计算奖励、用于选优和全对早停。模型对话只接收题面公开信息及公开样例错误详情。隐藏测试的输入、输出、错误和通过率不写入对话。

论文附录 D.1 描述的是奖励测试与最终留出测试分离的协议；当前结果应表述为“全测试奖励下的自适应解题率”。这项差异及贪心候选预算、KL、树先验等实现细节集中记录在 [METHOD.md](docs/METHOD.md)，不能把当前配置等同于论文全部实验设置。

## 安装与下载

使用 Linux、Python 3.10+ 和兼容 vLLM 0.21.0 的 CUDA 环境。已有兼容环境可以使用 `--no-deps`，避免重装训练依赖。

```bash
python -m pip install -e '.[vllm,data,test]'
python scripts/download_model.py --output ./models/Qwen3-4B-Instruct-2507
python scripts/download_lcb.py --selection v5 --output ./data/lcb-v5
```

下载仅通过 hf-mirror，认证从环境变量 `HF_TOKEN` 读取。支持 `--aria2` 和 `--no-proxy`；不用 YAML 配置。数据及模型保留各自上游条款，源码包不包含这些文件。

## 开始评测

```bash
tmux new-session -s pot-lcb-v5
bash scripts/run_benchmark.sh \
  --model ./models/Qwen3-4B-Instruct-2507 \
  --data ./data/lcb-v5 --selection v5 --output ./runs/lcb-v5 \
  --gpus 0 --workers-per-gpu 1 --limit 167
```

Ctrl-b 后按 d 退出 tmux 界面，评测继续运行。原命令和同一输出目录可以续跑；改变算法、数据或源码后使用新目录。

默认参数：20 轮、宽度 3、深度 8、输出 2048 tokens、上下文 16384 tokens、温度 0.7；LoRA rank8/alpha16，q/k/v/o 投影，学习率 1e-4、3 epochs、clip 0.3、参考 KL 0.02。整条路径上的模型输出计入损失，题面和反馈只作上下文。每次有效更新后生成一个额外贪心候选，单题最多 60 个采样候选加 20 个贪心候选，分别统计。

多 GPU 用 `--gpus 0 1 2`，同卡多个实例用 `--workers-per-gpu 2`。只有计划与已有任务共享 GPU 时才传 `--shared-gpus`。实际显存包含推理权重、KV cache、训练权重和反向传播峰值，不能只看 KV cache 大小。

CPU 测试并发由 `--eval-workers` 与 `--eval-global-workers` 控制。默认启动脚本使用每实例 16、全局 64；在较小机器上调低这两个值。

## 数据集

| selection | 题数 | 说明 |
|---|---:|---|
| v5 | 167 | LCB test5 新增题，不是累计 880 题 |
| v6 | 175 | LCB test6 新增题，不是累计 1055 题 |
| humaneval | 164 | 完整函数模块；断言位置分组奖励，原始完整检查作为成功门槛 |
| mbpp-sanitized | 257 | sanitized test 划分，按发布测试条目计分 |
| ojbench-icpc | 73 | 官方 DMOJ/PyPy 判题，包含交互题 |

具体下载、函数接口和判题环境见 [HumanEval/MBPP](docs/BENCHMARKS_zh.md) 与 [LCB/OJBench](docs/COMPETITIONS_zh.md)。

## 查看结果

- `summary.json`：完成数、答对数、正确率、平均通过率和累计耗时。
- `process.json`：JSONL，每题一行，记录编号、题目代号、得分、通过率、耗时、生成量与更新次数。
- `progress.json`：实时实例与当前题目；`metadata.json`：完整参数与源码、数据标识。
- `execution_failures.json`：运行异常，与模型答错分开记录。
- `jobs/`：逐次尝试、候选代码、公开反馈、训练指标和评测时间日志。

正确率分母为已经成功完成评测的题目数；基础设施失败不直接记作模型答错。未完成的题目须单列。累计逐题耗时是各题时间之和，与并发实验的墙钟时间不同。

## 测试与许可

```bash
python -m pytest tests -q
```

项目使用 MIT，第三方署名见 `THIRD_PARTY_NOTICES.md`。普通 subprocess 执行器有限时、限内存和输出限制，但与当前用户共享文件权限；运行生成代码应使用专用环境。OJBench 使用独立判题环境。
