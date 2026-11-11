# Method and correspondence to the paper

Reference: [Policy of Thoughts v2](https://arxiv.org/html/2601.20379v2).
This document describes the implemented algorithm and the choices needed to
run the implementation. Source locations are relative to the repository.

## Search and adaptation

1. Reset the policy LoRA, reference LoRA, optimizer and RNG for the new problem.
2. Select an expandable leaf by PUCT, using mean backed-up reward and a stored
   sibling prior. Generate three continuations at temperature 0.7.
3. Select the last closed Python block from each response and evaluate it.
   Back up the reward along its ancestor path. Return immediately on full success.
4. Standardize rewards within the sibling group, using population standard
   deviation, epsilon 1e-8 and advantage clipping to [-10, 10]. Skip a
   constant-reward group. Otherwise perform three GRPO epochs on the LoRA.
5. Publish the updated adapter and generate one greedy candidate from the same
   parent prompt. Evaluate it and continue searching if it does not solve the task.
6. On budget exhaustion, return the highest-reward evaluated candidate. Equal
   rewards favor syntactically valid code, then the earlier candidate.

The loop is in `pot/search.py`; training and adapter reset/publication are in
`pot/backend.py`. `pot/parallel.py` runs independent question workers and bounds
same-run, same-GPU training concurrency. The base model is never optimized.

## Training objective

The default `--loss-variant paper` minimizes the negative clipped policy
surrogate plus 0.02 times the sampled reference KL penalty. Token ratios use
the same positive-temperature distribution for current and fixed old scores.
The sampled KL term is `expm1(logp_ref - logp_current) - (logp_ref - logp_current)`.
Old/reference scores are detached. Advantages are constant through each update.

Loss is averaged over all model-output tokens in each root-to-leaf trajectory,
then over sibling trajectories and groups. Prompt and feedback tokens are
conditioning only. Ancestor outputs are replayed under a frozen pre-update
probability snapshot; they are not resampled at each update. See
[trajectory details](TRAJECTORY_LOSS.md).

Reference weights synchronize every 10 optimizer steps. LoRA targets the q/k/v/o
attention projections, with rank 8 and alpha 16. Learning rate is 1e-4, AdamW
weight decay 0.01, PPO clip 0.3 and gradient norm cap 1. Policy dropout is disabled
during likelihood optimization. No entropy bonus or frozen-base KL is active
in the default configuration.

The launcher selects `paper` explicitly. Optional `policy_scaled` and
`dual_kl` objectives are available through the API; neither is
enabled by default.

## Paper correspondence and explicit differences

| Topic | This implementation | Paper reference |
|---|---|---|
| Core loop | PUCT, sibling-relative GRPO, transient per-problem LoRA, greedy post-update candidate | Sections 4.2–4.4 |
| Search settings | 20 expansions, width 3, depth 8, temperature .7, 2048 output tokens, c_puct 1.414 | Appendix E.1 |
| Main training settings | rank8, lr1e-4, 3 epochs, clip .3, reference KL .02 | Appendix E.2 |
| Reward/evaluation split | All available tests affect reward, selection and stopping; only public diagnostics are textual feedback | Differs from D.1, which requires disjoint reward and final hidden tests |
| Greedy budget | Up to 60 sampled plus 20 separately counted greedy candidates | Section 4.4 describes greedy commitment; B/D.2 state a total capacity of 60 nodes without allocating an extra budget |
| Additional KL | Frozen-base KL defaults to 0 | E.2 additionally lists .005; its exact reference/objective is not resolved by Eq. 9. This release does not enable a second term |
| Objective sign | Minimize negative surrogate plus positive KL | Eq. 9 displays surrogate minus KL alongside minimizing language; code uses the usual maximizing-surrogate/minimizing-loss convention |
| Stored tree priors | Mean-token-logprob sibling softmax, fixed after node creation; a new greedy child receives half the sibling prior mass | Prior construction, greedy placement and refresh schedule are not fully specified; Section 4.3 refers to the current policy |
| Trajectory replay | Historical outputs rescored under the pre-update policy; fresh leaves use pre-update HF scores | Historical sampling/replay convention is not specified |
| Reference synchronization | Every 10 optimizer steps, including between epochs | E.2 says 10 update steps |
| Code selection | Last closed Python block only, identified and quoted in feedback | A concrete extraction convention; not best-of-all-blocks |
| Context overflow | Prune a path that cannot fit prompt plus output in 16384 tokens; record the pruning | Operational context policy |
| HumanEval reward | One probe per assertion site, plus original full checker as a success gate | Assertion-site reward decomposition is an adapter convention |
| Runtime | vLLM 0.21.0, PyTorch 2.11.0; independent workers may share GPUs | E.3 lists vLLM .13/PyTorch2.9 and one A800 |

The generation service is vLLM, while chosen-token likelihoods are rescored by
the HF trainer. Hardware, kernels, concurrency and sampling can change
trajectories even at a fixed seed.

The source follows PoT's search-and-adapt design. The full-test-reward
protocol and the implementation choices above must be specified when
comparing results with the paper.

## Reporting results

Report the model/revision, dataset/revision, seed, reward protocol, generated
sampled and greedy counts, context limit, completed/correct/failed counts and
source identity. Include both question durations and parallel wall time when
comparing speed. `--no-train` also removes post-update greedy candidates, so it
changes the generation budget as well as training.

Benchmark results should be accompanied by their own configuration and run
metadata.
