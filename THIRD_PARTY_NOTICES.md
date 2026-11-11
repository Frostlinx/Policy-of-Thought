# Third-party notices

This file preserves attribution for adapted third-party material. This project's original work is licensed under MIT; see `LICENSE`.
The upstream notices below retain their own copyright holders and scope.

## LiveCodeBench evaluation utilities

Parts of `pot/evaluator.py` adapt the import prelude, stdin AST wrapping and
input/output comparison behavior from LiveCodeBench's `testing_util.py`.
`pot/process_evaluator.py` also uses the resulting runner. Although execution
is restructured and isolation is implemented here, the adaptations are treated
conservatively as substantial portions requiring the upstream notice.

Source: [LiveCodeBench testing_util.py](https://github.com/LiveCodeBench/LiveCodeBench/blob/main/lcb_runner/evaluation/testing_util.py).
License: [LiveCodeBench LICENSE](https://github.com/LiveCodeBench/LiveCodeBench/blob/main/LICENSE).

```text
MIT License

Copyright (c) 2024 LiveCodeBench

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## APPS evaluator lineage

LiveCodeBench documents its evaluator as a modification of the APPS evaluator.
The following notice is retained conservatively for that inherited lineage;
this project does not bundle the APPS benchmark dataset.

Sources: [LiveCodeBench evaluation attribution](https://github.com/LiveCodeBench/LiveCodeBench#evaluation)
and [APPS LICENSE](https://github.com/hendrycks/apps/blob/main/LICENSE).

```text
MIT License

Copyright (c) 2021 Dan Hendrycks

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## External data, model weights and dependencies

HumanEval and MBPP adapters download external benchmark data; no
benchmark records or reference implementations are bundled. Preserve the terms
and attribution in the upstream repositories and downloaded dataset cards:
[HumanEval](https://github.com/openai/human-eval) and
[MBPP](https://github.com/google-research/google-research/tree/master/mbpp).
The adapters use the checker programs from downloaded records at runtime;
they do not vendor the upstream execution harness. PyArrow is an optional
installed dependency for reading the pinned Parquet files.

The software notices above do not license LiveCodeBench problem statements,
test data, model weights or adapters. Downloaded assets retain their respective
upstream terms and must not be included in a source release by default.

Python, PyTorch, Transformers, PEFT, vLLM, Requests and optional download/test
tools are installed dependencies rather than vendored source in this package.
Their own licenses apply. Binary/container redistribution requires retaining
the notices shipped with the redistributed dependencies.

Prompt template hashes are recorded in `docs/original-prompt-provenance.json`.

## OJBench / DMOJ integration

The ICPC adapter downloads external OJBench problem assets from the pinned
dataset revision via hf-mirror. No problem data or validators are bundled.
[OJBench](https://github.com/He-Ren/OJBench) provides the reference runtime
configuration and dataset. The optional judge runs as a separate process and
imports [DMOJ judge-server](https://github.com/DMOJ/judge-server) at commit
`f098cd3a49a60186d1fadde5132329ec5f4f2213`; DMOJ is AGPL-3.0-or-later.
DMOJ source/binaries are installed separately, not relicensed under this
project's MIT license or included in this source archive. Preserve upstream
licenses/notices when distributing judge environments. PyPy and downloaded
problem/checker files retain their respective upstream terms.

## Paper figures

`assets/framework.png` and `assets/pipeline.png` reproduce figures from
[Policy of Thoughts, arXiv v2](https://arxiv.org/html/2601.20379v2),
by Zhengbo Jiao, Hongyu Xian, Qinglong Wang, Yunpu Ma, Zhebo Wang,
Zifan Zhang, Dezhang Kong and Meng Han. The paper is distributed under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/); these figures
retain that license. They have not been modified.
