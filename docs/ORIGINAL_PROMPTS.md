# Prompts and code selection

`pot/prompts.py` contains the initial, continuation, error-feedback and reflection
templates. Literal hashes are stored in `original-prompt-provenance.json`
for verification.

The prompt builder in `pot/search.py` adds benchmark-specific interfaces:
stdin/stdout programs, LeetCode `Solution` methods, complete function modules,
or interactive read/write protocols. Generic examples do not override the
required function name or signature.

Each response may contain reasoning and several Python blocks. Evaluation uses
only the last closed block tagged `python`, `python3` or `py`. A trailing
unfinished block is ignored. Blocks are not concatenated or repaired. Syntax
and execution determine whether the selected program works.

Feedback names the selected block, quotes its exact code, and reports public
test counts and diagnostics. Format and syntax rejection are distinguished from
executed failures. Hidden test scores and diagnostics are not interpolated into
messages. Public success alone does not stop search. If no executable public
samples exist, the prompt says that no public execution result is available.

The continuation template includes reflection questions; the separate reflection
template does not introduce another model call. Template substitutions are made
once, so Python braces and braces inside the problem are preserved.

LCB uses the general coding templates with explicit standard-input/output
or function-method interface instructions.

`pot.prompt_probe` generates initial responses without training, tree search or
code execution. Its scores are null; it measures response length and extraction,
not solution accuracy.
