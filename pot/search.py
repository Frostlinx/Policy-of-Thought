"""MCTS with public-only model feedback and isolated scalar reward callbacks.

Public and private tests always jointly determine optimization rewards, but
private test records, diagnostics, and aggregate scores never enter prompts.
"""
from __future__ import annotations
import ast
import dataclasses
import hashlib
import math
import re
import time
from typing import Any, Callable
from .prompts import CODING_START_PROMPT, CODING_CONTINUE_PROMPT, CODE_ERROR_FEEDBACK_PROMPT, render_template


@dataclasses.dataclass
class SearchConfig:
    simulations: int = 20
    width: int = 3
    max_depth: int = 8
    max_new_tokens: int = 2048
    temperature: float = 0.7
    c_puct: float = 1.414
    feedback: str = "public"  # public or none; never private diagnostics
    greedy_commitment: bool = True
    train: bool = True


@dataclasses.dataclass
class Node:
    node_id: int
    parent: Any = None
    depth: int = 0
    text: str = ""
    code: str = ""
    feedback: str = ""
    reward: float = 0.0
    prior: float = 1.0
    visits: int = 0
    value_sum: float = 0.0
    children: list = dataclasses.field(default_factory=list)
    expanded: bool = False
    valid_python: bool = False
    public_reward: float = 0.0
    public_passed: int = 0
    public_total: int = 0
    sample: Any = dataclasses.field(default=None, repr=False, compare=False)
    selected_block: Any = None
    evaluation_status: str = "not_evaluated"

    @property
    def value(self):
        return self.value_sum / self.visits if self.visits else 0.0


@dataclasses.dataclass(frozen=True)
class CodeSelection:
    code: str
    python_block_index: int
    closed_python_blocks: int
    fenced_block_index: int
    ignored_unclosed_python_tail: bool


def select_code(text: str) -> CodeSelection:
    """Use the last closed Python block from the original multi-step format.

    A trailing incomplete block is not executable and is ignored. Never join
    snippets or synthesize missing code. Closure does not imply correctness.
    """
    fenced = list(re.finditer(r"```([^`\n\r]*)\r?\n(.*?)```", text, re.S))
    blocks = [(i, m) for i, m in enumerate(fenced, 1)
              if m[1].strip().lower() in {"python", "python3", "py"}]
    if not blocks:
        raise ValueError("End your response with a closed python code block containing the complete solution.")
    index, last = blocks[-1]
    code = last[2].strip()
    tail = text[fenced[-1].end():]
    return CodeSelection(code, len(blocks), len(blocks), index,
                         bool(re.search(r"```(?:python|python3|py)[ \t]*\r?\n", tail, re.I)))


def extract_code(text: str) -> str:
    code = select_code(text).code
    if not code:
        raise ValueError("The Python code block is empty; return a complete runnable solution.")
    return code


CODE_SELECTION_CONTRACT = (
    "Evaluation rule: only the last closed Python code block in each response is selected. "
    "Earlier blocks and an unfinished trailing block are not evaluated. "
    "Make the selected block a complete solution, including required imports and the required interface. "
    "Feedback identifies that block and quotes the selected code; revise that program."
)


def selected_code_feedback(item: Node) -> tuple[str, str]:
    """Identify the program independently of its public diagnostics or reward."""
    selected = item.selected_block
    if selected is None:
        return (f"Response node {item.node_id}: no Python block was selected; no code was executed.\n", "")
    description = (f"Response node {item.node_id}: selected Python block "
                   f"{selected.python_block_index} of {selected.closed_python_blocks} closed Python blocks "
                   f"(fenced block {selected.fenced_block_index} overall). "
                   "No other block from this response was evaluated.\n")
    if selected.ignored_unclosed_python_tail:
        description += "The unfinished trailing Python block was ignored.\n"
    if item.evaluation_status == "evaluated":
        description += "The public execution feedback below refers exactly to this selected program.\n"
    else:
        description += f"Status: {item.evaluation_status}; no code was executed.\n"
    # A longer fence keeps backticks inside Python strings from ending the quote.
    fence = "`" * max(3, 1 + max((len(m[0]) for m in re.finditer(r"`+", item.code)), default=0))
    quoted = f"{fence}python\n{item.code}\n{fence}"
    return description, quoted


def problem_messages(problem, node: Node, feedback: str = 'public') -> list[dict]:
    if feedback not in {'public', 'none'}:
        raise ValueError("feedback must be public or none")
    from .benchmarks import public_test_text
    public_examples = "\n\n".join(
        f"Example {i+1}\n{public_test_text(t)}"
        for i, t in enumerate(problem.public_tests)
    )
    task = problem.question_content
    if getattr(problem, 'interface', 'auto') == 'interactive':
        interface = ("This is an interactive problem. Follow the statement's interaction protocol, "
                     "read each judge reply when requested, and flush stdout after every query or answer. "
                     "Do not read all stdin at once or reproduce a sample transcript. Return the complete Python program.")
    elif getattr(problem, 'interface', 'auto') == 'module':
        interface = ("Return a complete Python module with the required top-level function(s), "
                     "including imports and any helpers. Do not use a Solution class, read stdin, "
                     "or include test calls in the final solution. ")
        if problem.starter_code:
            interface += ("Preserve the following function name, signature and behavior; "
                          "replace its stub with your implementation:\n```python\n"
                          + problem.starter_code + "\n```")
        else:
            interface += "Use the function names and positional arguments in the public assertions below."
    elif problem.starter_code:
        interface = ("Implement the specified method in class Solution. Preserve the method name "
                 "and argument order. Return the answer from the method; do not read stdin, "
                 "print the answer, or write a driver program. Use this interface:\n```python\n"
                 + problem.starter_code + "\n```")
    else:
        interface = "Read standard input and write standard output."
    task += "\n\nRequired interface:\n" + interface
    task += "\n\nPublic examples:\n" + (public_examples or (
        "See the problem statement above. Interactive transcripts are illustrative and are not replayed as fixed tests."
        if getattr(problem, 'benchmark', '') == 'ojbench-icpc' else "See the function docstring above."))
    messages = [{"role": "system", "content":
        "Solve the given problem in Python 3 using only the standard library. "
        "Follow this problem's required interface; function names in illustrative examples do not override it."},
        {"role": "user", "content": render_template(CODING_START_PROMPT, problem=task)
         + "\nRequired interface for this problem (also applies to the final solution):\n" + interface
         + "\n\n" + CODE_SELECTION_CONTRACT}]
    history = []
    while node.parent is not None:
        history.append(node)
        node = node.parent
    for item in reversed(history):
        messages.append({"role": "assistant", "content": item.text})
        # Keep the original instruction text, without duplicating the full
        # problem and previous answer already present in conversation history.
        instruction = render_template(CODING_CONTINUE_PROMPT,
            problem="See the original problem and public examples above.",
            previous_steps="See the immediately preceding assistant response.")
        if feedback == 'public':
            # Explicitly allowlist public fields. item.reward may include a
            # private-test scalar and must never be interpolated here.
            public_info, selected_code = selected_code_feedback(item)
            if item.public_total:
                public_info += f"Public tests: {item.public_passed}/{item.public_total} passed.\n"
            else:
                public_info += "No executable public tests are configured; no public test result is available.\n"
            public_failure = item.public_passed < item.public_total
            code_issue = any(t in item.feedback for t in ('format_error', 'syntax_error'))
            if item.feedback and (public_failure or code_issue):
                public_info += render_template(CODE_ERROR_FEEDBACK_PROMPT,
                    error_type="Python format/syntax issue" if code_issue else "Public test failure",
                    error_message=item.feedback,
                    pass_rate=f"{item.public_passed/item.public_total:.1%}" if item.public_total else "Not evaluated",
                    code=selected_code or "No code was selected or executed.",
                    failed_cases="See the public execution diagnostics above; no private cases are provided.")
            else:
                public_info += f"Selected program:\n{selected_code}\nPublic execution diagnostics:\n{item.feedback}\n"
            instruction = public_info + '\n' + instruction
        instruction += "\nRequired interface for this problem:\n" + interface + "\n\n" + CODE_SELECTION_CONTRACT
        messages.append({"role": "user", "content": instruction})
    return messages


def _available(node, depth):
    return node.depth < depth and (not node.expanded or any(_available(c, depth) for c in node.children))


def select_leaf(root: Node, cfg: SearchConfig):
    if not _available(root, cfg.max_depth):
        return None
    node = root
    while node.expanded:
        children = [c for c in node.children if _available(c, cfg.max_depth)]
        if not children:
            return None
        total = max(1, sum(c.visits for c in node.children))
        node = max(children, key=lambda c: c.value + cfg.c_puct * c.prior * math.sqrt(total)/(1+c.visits))
    return node


def backpropagate(node: Node, reward: float):
    while node is not None:
        node.visits += 1
        node.value_sum += reward
        node = node.parent


def solve(problem, backend, evaluator, cfg: SearchConfig, *,
          emit: Callable[[dict], None] | None = None) -> dict:
    if cfg.feedback not in {'public', 'none'}:
        raise ValueError("feedback must be public or none")
    if cfg.width < 1 or min(cfg.simulations, cfg.max_depth, cfg.max_new_tokens) < 1:
        raise ValueError("Invalid search budget")
    if not math.isfinite(cfg.temperature) or cfg.temperature <= 0:
        raise ValueError("Sampling temperature must be finite and positive")
    if not math.isfinite(cfg.c_puct) or cfg.c_puct < 0:
        raise ValueError("PUCT exploration coefficient must be finite and nonnegative")
    emit = emit or (lambda row: None)
    started = time.perf_counter()
    root = Node(0)
    nodes = [root]
    best = None
    stop_reason = "budget_exhausted"
    stats = dict(expansions=0, sampled_candidates=0, greedy_candidates=0,
                 evaluated_candidates=0, format_errors=0, syntax_errors=0,
                 generated_tokens=0, sampled_tokens=0, greedy_tokens=0,
                 training_updates=0, optimizer_steps=0, skipped_training_groups=0, context_pruned_nodes=0,
                 generation_seconds=0.0, training_seconds=0.0, reward_evaluation_seconds=0.0)
    backend.reset_instance(problem.problem_id)

    def record_generation(samples, kind):
        # A generation batch is already complete when early stopping examines
        # its first child. Count every completion, including unused siblings.
        stats[kind + '_candidates'] += len(samples)
        tokens = sum(len(sample.token_ids) for sample in samples)
        stats[kind + '_tokens'] += tokens
        stats['generated_tokens'] += tokens

    def add(sample, parent, prior, kind):
        nonlocal best, stop_reason
        code = ''
        invalid = None
        selected = None
        sample.trajectory_prefix = [] if parent.sample is None else [*parent.sample.trajectory_prefix, parent.sample]
        try:
            selected = select_code(sample.text)
            code = selected.code
            if not code:
                raise ValueError("The Python code block is empty; return a complete runnable solution.")
            ast.parse(code)
        except ValueError as error:
            stats['format_errors'] += 1
            invalid = {'status': 'format_error', 'message': str(error)}
        except SyntaxError as error:
            stats['syntax_errors'] += 1
            invalid = {'status': 'syntax_error', 'message':
                       f"Python syntax error at line {error.lineno}: {error.msg}"}
        if invalid is not None:
            passed, total, all_passed, feedback = 0, len(problem.public_tests), False, invalid
            reward = 0.0
        else:
            then = time.perf_counter()
            try:
                result = evaluator.evaluate_candidate(problem, code)
            except Exception:
                raise RuntimeError("All-test evaluation failed; private diagnostics withheld") from None
            finally:
                stats['reward_evaluation_seconds'] += time.perf_counter()-then
            public = result.public
            passed, total, feedback = public.passed, public.total, public.feedback
            reward, all_passed = result.reward, result.all_passed
        public_reward = passed/total if total else 0.0
        if not isinstance(feedback, str):
            import json
            feedback = json.dumps(feedback, ensure_ascii=False)
        child = Node(len(nodes), parent, parent.depth+1, sample.text, code,
                     feedback, reward, prior)
        child.valid_python = invalid is None
        child.sample = sample
        child.selected_block = selected
        child.evaluation_status = invalid['status'] if invalid else 'evaluated'
        child.public_reward = public_reward
        child.public_passed, child.public_total = passed, total
        nodes.append(child)
        parent.children.append(child)
        backpropagate(child, reward)
        if best is None or (child.reward, child.valid_python) > (best.reward, best.valid_python):
            best = child
        stats['evaluated_candidates'] += 1
        emit({'event': 'candidate', 'problem_id': problem.problem_id,
              'node_id': child.node_id, 'parent_id': parent.node_id, 'depth': child.depth,
              'kind': kind, 'public_reward': public_reward, 'optimization_reward': reward,
              'reward_tests': 'all', 'public_passed': passed,
              'public_total': total, 'public_evaluation_diagnostics': feedback, 'code': code,
              'code_selection': dataclasses.asdict(selected) if selected else None,
              'code_sha256': hashlib.sha256(code.encode()).hexdigest() if selected else None,
              'evaluation_status': child.evaluation_status,
              'text': sample.text, 'policy_version': sample.policy_version,
              'completion_tokens': len(sample.token_ids), 'finish_reason': sample.finish_reason})
        should_stop = all_passed
        if should_stop:
            best = child
            stop_reason = 'all_tests_passed'
        return child, should_stop

    while stats['expansions'] < cfg.simulations:
        parent = select_leaf(root, cfg)
        if parent is None:
            stop_reason = 'tree_exhausted'
            break
        prompt = backend.render_prompt(problem_messages(problem, parent, cfg.feedback))
        then = time.perf_counter()
        try:
            samples = backend.generate(prompt, n=cfg.width, max_tokens=cfg.max_new_tokens,
                                       temperature=cfg.temperature)
        except ValueError as error:
            stats['generation_seconds'] += time.perf_counter()-then
            if str(error) != 'prompt + output budget exceeds max_context_tokens; reduce history explicitly':
                raise
            # A complete path may exceed the model's configured context even
            # before the depth limit. Do not alter its recorded token history.
            parent.expanded = True
            stats['context_pruned_nodes'] += 1
            emit({'event': 'context_pruned', 'problem_id': problem.problem_id,
                  'node_id': parent.node_id, 'depth': parent.depth})
            continue
        stats['generation_seconds'] += time.perf_counter()-then
        record_generation(samples, 'sampled')
        if len(samples) != cfg.width:
            raise RuntimeError("Generation backend returned an incomplete sibling group")
        stats['expansions'] += 1
        parent.expanded = True
        logits = [sum(s.old_logprobs)/len(s.old_logprobs) if s.old_logprobs else None for s in samples]
        mx = max((value for value in logits if value is not None), default=0.0)
        weights = [math.exp(value-mx) if value is not None else 0.0 for value in logits]
        denom = sum(weights)
        if not denom:
            weights, denom = [1.0] * len(samples), len(samples)
        group = []
        halted = False
        for sample, weight in zip(samples, weights):
            child, halted = add(sample, parent, weight/denom, 'sampled')
            group.append((sample, child.reward))
            if halted:
                break
        if halted:
            break
        if cfg.train:
            then = time.perf_counter()
            if any(not sample.token_ids for sample, _ in group):
                # Do not invent EOS IDs or change the sibling normalization
                # population just to make an empty response differentiable.
                metrics = {'optimizer_steps': 0, 'skipped': True, 'reason': 'empty_completion',
                           'groups_total': 1, 'groups_used': 0}
            else:
                metrics = backend.update([group])
            stats['training_seconds'] += time.perf_counter()-then
            stats['skipped_training_groups'] += int(bool(metrics.get('skipped', False)))
            updated = metrics.get('optimizer_steps', 0) > 0
            stats['training_updates'] += int(updated)
            stats['optimizer_steps'] += metrics.get('optimizer_steps',0)
            emit({'event':'update','problem_id':problem.problem_id, 'parent_id':parent.node_id, **metrics})
            if cfg.greedy_commitment and updated:
                then = time.perf_counter()
                commitments = backend.generate(prompt,n=1,max_tokens=cfg.max_new_tokens,temperature=0.0)
                stats['generation_seconds'] += time.perf_counter()-then
                record_generation(commitments, 'greedy')
                if len(commitments) != 1:
                    raise RuntimeError("Generation backend returned an invalid commitment batch")
                greedy = commitments[0]
                # Commitment is a new child, accounted separately from the 3 sampled siblings.
                # Split prior mass equally between sampled exploration and greedy commitment.
                for child in parent.children:
                    child.prior *= 0.5
                _, halted = add(greedy,parent,0.5,'greedy')
                if halted:
                    break
    return {'code': best.code if best else '', 'public_score':best.public_reward if best else 0.0,
            'optimization_score': best.reward if best else 0.0,
            'selected_node':best.node_id if best else None, 'stop_reason':stop_reason,
            'evaluation_protocol': 'all_tests_reward', 'reward_tests': 'all',
            'feedback_mode': cfg.feedback, 'stop_rule': 'all_tests_passed',
            'unevaluated_candidates': stats['sampled_candidates'] + stats['greedy_candidates'] - stats['evaluated_candidates'],
            'elapsed_seconds':time.perf_counter()-started, **stats}
