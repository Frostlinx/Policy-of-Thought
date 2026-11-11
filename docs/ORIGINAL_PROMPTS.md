# Prompts and code selection

`pot/prompts.py` contains the initial, continuation, error-feedback and reflection
templates. Literal hashes are stored in `original-prompt-provenance.json`
for verification.

The prompt builder in `pot/search.py` adds benchmark-specific interfaces:
stdin/stdout programs, LeetCode `Solution` methods, complete function modules,
or interactive read/write protocols, with the required function name and
signature.

Each response may contain reasoning and several Python blocks. Evaluation uses
only the last closed block tagged `python`, `python3` or `py`. A trailing
unfinished block is ignored. The selected block is executed as written.

Feedback names the selected block, quotes its exact code, and reports public
test counts and diagnostics. Format and syntax rejection are distinguished from
executed failures. Hidden test scores and diagnostics are not interpolated into
messages. Search stops when all public and hidden tests pass. If no executable public
samples exist, the prompt says that no public execution result is available.

The continuation template includes reflection questions in the same model call. Template substitutions are made
once, so Python braces and braces inside the problem are preserved.

LCB uses the general coding templates with explicit standard-input/output
or function-method interface instructions.

`pot.prompt_probe` measures initial response lengths and code extraction. It
generates responses without training or execution and records scores as `null`.
