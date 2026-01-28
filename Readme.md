
# Policy of Thoughts (PoT)

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

**Policy of Thoughts (PoT)** is a novel test-time reasoning framework that enables Large Language Models (LLMs) to dynamically evolve their internal reasoning policy in real-time, inspired by Popper's epistemology of "conjectures and refutations". Instead of merely discarding failed attempts, PoT uses execution feedback to perform online optimization on a transient LoRA adapter, allowing even small models to achieve state-of-the-art performance on complex reasoning tasks.

A compact **4B-parameter model** enhanced with PoT achieves **49.71% accuracy on LiveCodeBench**, outperforming much larger commercial models like GPT-4o, Claude-4-Opus, and DeepSeek-V3.

<p align="center">
  <img src="assets/framework.png" alt="PoT Framework" width="80%">
  <br>
  <em>The Policy of Thoughts (PoT) Framework: A closed-loop reasoning system.</em>
</p>

## 🚀 Key Features

- **Test-time Policy Evolution**: Transforms reasoning into a within-instance online optimization process.
- **Closed-loop Learning**: Internalizes execution feedback directly into the model's parameters using GRPO.
- **Parameter-Efficient**: Uses transient LoRA adapters for rapid, instance-specific adaptation without modifying the base model.
- **Structured Exploration**: Leverages Monte Carlo Tree Search (MCTS) for efficient and diverse trajectory generation.
- **Architecture-Agnostic**: Demonstrates strong generalization across different model families and scales (e.g., Qwen3, Phi-3).

## 📊 Results

### Code Reasoning Benchmarks (Qwen3-4B-Instruct)

| Method | HumanEval | MBPP | LCB v5 | LCB v6 | ICPC | **Overall** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **PoT (Ours)** | **98.78** | **94.94** | **57.49** | **49.71** | **19.18** | **58.98** |
| Best Baseline (LATS) | 87.80 | 90.66 | 47.51 | 43.43 | 13.69 | 53.34 |
| GPT-4o | 91.50 | 87.60 | 31.74 | 16.00 | 9.59 | 49.75 |

### Model Scale vs. Performance on LiveCodeBench v6
<p align="center">
  <img src="assets/performance_vs_scale.png" alt="Performance vs Scale" width="70%">
</p>

Our tiny 4B model with PoT significantly outperforms models that are orders of magnitude larger.

## 🛠️ Installation

> **Note**: Code will be released soon! ⏳  
> Stay tuned by watching this repository or checking back shortly.

## 📚 Citation

If you find this work useful in your research, please consider citing:

```bibtex
@article{anonymous2026policy,
  title={Policy of Thoughts: Scaling LLM Reasoning via Test-time Policy Evolution},
  author={Anonymous Authors},
  journal={Under Review},
  year={2026}
}
```

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

--- 
