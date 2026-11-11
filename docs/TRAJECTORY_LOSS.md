# Trajectory loss

`--loss-scope trajectory` is the default. Every model output on the
root-to-current path contributes to the objective: reasoning, intermediate
code, final code and sampled special tokens. Execution uses the last closed
Python block; training uses every model-output token on the path.

System/user messages, problem statements, execution feedback and quoted code in
feedback serve as context. Each response segment is scored at its original
conditioning prompt using the token IDs recorded during generation.

For each sibling, its normalized reward advantage weights the complete path.
Losses are means over path output tokens, then over siblings and groups. Shared
ancestors are forwarded once per epoch; each descendant retains its own
advantage, clipping operation and length weight.

Leaves must come from the current policy and share the same parent prompt and
ancestor chain. Ancestor outputs may come from older adapters or greedy
decoding. Before an update, they are teacher-forced under the current adapter at
the configured stochastic temperature, and those old scores stay fixed through
all epochs. Greedy history retains its original generation temperature in the
record. These historical outputs form the replayed part of each trajectory.

`--loss-scope completion` is a separate current-response-only ablation.
Update records contain the loss scope, old-policy snapshot rule, trajectory and
history token counts, forwarded segments, historical policy versions and greedy
history count. Constant-reward groups skip optimizer updates, including KL-only
updates.
