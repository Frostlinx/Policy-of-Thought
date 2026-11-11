"""Search invariants with fake generation/execution; no model or torch needed."""
from types import SimpleNamespace

import pytest

from pot.evaluator import CandidateEvaluation, PublicEvaluation, PrivateScore
from pot.search import Node, SearchConfig, extract_code, select_code, problem_messages, select_leaf, solve


def problem():
    return SimpleNamespace(
        problem_id="atcoder/example", question_content="Print the requested number.",
        starter_code="", public_tests=[SimpleNamespace(input="1", output="1"),
                                       SimpleNamespace(input="2", output="2")],
        private_tests="THIS_SECRET_MUST_NEVER_ENTER_A_PROMPT",
    )


def sample(number, tokens=2, *, text=None, version=0):
    return SimpleNamespace(
        text=text if text is not None else f"```python\nprint({number})\n```",
        token_ids=list(range(tokens)), old_logprobs=[-1.0] * tokens,
        policy_version=version, finish_reason="stop",
    )


class FakeBackend:
    def __init__(self, batches, steps=3):
        self.batches = list(batches)
        self.calls, self.updates, self.resets = [], [], []
        self.steps = steps
        self.version = 0

    def reset_instance(self, instance):
        self.resets.append(instance)
        self.version = 0

    def render_prompt(self, messages):
        return messages

    def generate(self, prompt, **kwargs):
        self.calls.append((prompt, kwargs))
        batch = self.batches.pop(0)
        if isinstance(batch, Exception):
            raise batch
        return batch

    def update(self, groups):
        self.updates.append(groups)
        old = self.version
        self.version += int(self.steps > 0)
        # Deliberately match the real backend: there is no "updated" key.
        return {"optimizer_steps": self.steps, "skipped": not self.steps,
                "version_before": old, "version_after": self.version}


class FakeEvaluator:
    def __init__(self, scores=None, private_scores=None):
        self.scores = scores or {}
        self.private_scores = self.scores if private_scores is None else private_scores
        self.codes = []

    def evaluate_candidate(self, public_problem, code):
        self.codes.append(code)
        passed = self.scores.get(code, 0)
        hidden = self.private_scores.get(code, 0)
        return CandidateEvaluation(PublicEvaluation(passed == 2, passed, 2, f"{passed}/2 public tests passed"),
                                   PrivateScore(hidden, 2, hidden == 2))


def test_early_stop_accounts_for_entire_already_generated_batch():
    backend = FakeBackend([[sample(1, 2), sample(2, 3), sample(3, 4)]])
    evaluator = FakeEvaluator({"print(1)": 2})
    result = solve(problem(), backend, evaluator, SearchConfig())
    assert result["stop_reason"] == "all_tests_passed"
    assert result["sampled_candidates"] == 3
    assert result["evaluated_candidates"] == 1
    assert result["unevaluated_candidates"] == 2
    assert result["generated_tokens"] == result["sampled_tokens"] == 9
    assert result["greedy_tokens"] == 0
    assert backend.updates == []
    assert backend.resets == ["atcoder/example"]


def test_real_backend_metrics_trigger_separately_accounted_greedy_commitment():
    siblings = [sample(0, 2), sample(1, 3), sample(2, 4)]
    commitment = sample(3, 5, version=1)
    backend = FakeBackend([siblings, [commitment]])
    evaluator = FakeEvaluator({"print(1)": 1, "print(3)": 2})
    result = solve(problem(), backend, evaluator, SearchConfig(simulations=1))
    assert result["training_updates"] == 1
    assert result["optimizer_steps"] == 3
    assert result["sampled_candidates"] == 3
    assert result["greedy_candidates"] == 1
    assert result["generated_tokens"] == 14
    assert result["sampled_tokens"] == 9 and result["greedy_tokens"] == 5
    assert result["selected_node"] == 4
    assert result["code"] == "print(3)"
    assert backend.calls[1][1]["temperature"] == 0
    assert [pair[0] for pair in backend.updates[0][0]] == siblings
    assert [pair[1] for pair in backend.updates[0][0]] == [0, 0.5, 0]


def test_skipped_update_does_not_generate_commitment():
    backend = FakeBackend([[sample(0), sample(1), sample(2)]], steps=0)
    result = solve(problem(), backend, FakeEvaluator(), SearchConfig(simulations=1))
    assert len(backend.calls) == 1
    assert result["training_updates"] == result["optimizer_steps"] == result["greedy_candidates"] == 0


def test_format_and_syntax_errors_get_zero_reward_without_execution():
    backend = FakeBackend([[
        sample(0, text="Here is a solution: print(0)"),
        sample(1, text="```python\ndef invalid(:\n```"),
        sample(2),
    ]], steps=0)
    evaluator, events = FakeEvaluator(), []
    result = solve(problem(), backend, evaluator, SearchConfig(simulations=1), emit=events.append)
    assert evaluator.codes == ["print(2)"]
    assert result["format_errors"] == result["syntax_errors"] == 1
    assert result["public_score"] == 0
    assert result["code"] == "print(2)"  # Prefer runnable syntax when rewards tie.
    assert [r for _, r in backend.updates[0][0]] == [0, 0, 0]
    assert "format_error" in events[0]["public_evaluation_diagnostics"]
    assert "syntax_error" in events[1]["public_evaluation_diagnostics"]


def test_format_requires_one_python_block_and_prompts_repeat_the_contract():
    assert extract_code("```python\nprint(1)\n```") == "print(1)"
    for text in ("print(1)", "```javascript\nprint(1)\n```", "```python\n\n```",
                 "```python\nprint(2)"):
        with pytest.raises(ValueError):
            extract_code(text)
    root = Node(0)
    child = Node(1, parent=root, depth=1, text="bad response", feedback="format_error")
    messages = problem_messages(problem(), child)
    assert "Solve this coding problem step by step" in messages[1]["content"]
    assert "For each reasoning step, provide the corresponding Python code" in messages[1]["content"]
    assert "REFLECTION QUESTIONS to consider" in messages[-1]["content"]
    assert "THIS_SECRET" not in str(messages)


def test_reasoning_examples_do_not_replace_final_python_solution():
    text = ('Reasoning and an input example:\n```text\n1 2\n```\n'
            'An intermediate implementation:\n```python\nprint(0)\n```\n'
            'Final solution:\n```python\nprint(3)\n```')
    assert extract_code(text) == 'print(3)'


def test_empty_final_python_block_does_not_fall_back_to_earlier_code():
    with pytest.raises(ValueError):
        extract_code('```python\nprint(1)\n```\n```python\n\n```')


def test_original_incremental_format_keeps_last_closed_block_when_tail_is_cut_off():
    assert extract_code('Step 1:\n```python\nprint(1)\n```\nStep 2:\n```python\nprint(') == 'print(1)'


def test_feedback_none_hides_execution_results_across_multiple_ancestors():
    root = Node(0)
    first = Node(1, parent=root, depth=1, text="```python\nprint(1)\n```",
                 feedback="LCB_ERROR_SENTINEL: expected 999 but got 123", reward=0.5)
    second = Node(2, parent=first, depth=2, text="```python\nprint(2)\n```",
                  feedback="FORMAT_ERROR_SENTINEL: return Python", reward=0.25)
    messages = problem_messages(problem(), second, feedback='none')
    rendered = str(messages)
    assert "LCB_ERROR_SENTINEL" not in rendered
    assert "FORMAT_ERROR_SENTINEL" not in rendered
    assert "expected 999" not in rendered
    assert "0.5" not in rendered and "0.25" not in rendered
    assert "print(1)" in rendered and "print(2)" in rendered
    assert "Public examples" in rendered  # Original visible examples remain allowed.
    continuation_prompts = [message["content"] for message in messages[2:] if message["role"] == "user"]
    assert len(continuation_prompts) == 2 and continuation_prompts[0] == continuation_prompts[1]


def test_public_feedback_contains_only_public_fields_not_optimization_reward():
    root = Node(0)
    child = Node(1, parent=root, depth=1, text="```python\nprint(1)\n```",
                 feedback="PUBLIC_ERROR_SENTINEL", reward=0.87654321,
                 public_reward=0.5, public_passed=1, public_total=2)
    rendered = str(problem_messages(problem(), child, feedback='public'))
    assert "PUBLIC_ERROR_SENTINEL" in rendered and "1/2 passed" in rendered
    assert "0.87654321" not in rendered and "THIS_SECRET" not in rendered


def test_depth_limit_exhausts_search_without_expanding_terminal_nodes():
    backend = FakeBackend([[sample(0), sample(1)]])
    cfg = SearchConfig(simulations=20, width=2, max_depth=1, train=False)
    result = solve(problem(), backend, FakeEvaluator(), cfg)
    assert result["stop_reason"] == "tree_exhausted"
    assert result["expansions"] == 1 and result["sampled_candidates"] == 2
    assert not backend.updates


def test_selection_uses_puct_and_excludes_depth_capped_children():
    root = Node(0, expanded=True)
    high_prior = Node(1, parent=root, depth=1, prior=0.9, visits=1)
    low_prior = Node(2, parent=root, depth=1, prior=0.1, visits=1)
    capped = Node(3, parent=root, depth=2, value_sum=1000, visits=1)
    root.children = [low_prior, high_prior, capped]
    assert select_leaf(root, SearchConfig(max_depth=2)) is high_prior


def test_generation_must_return_the_requested_sibling_group():
    backend = FakeBackend([[sample(0)]])
    with pytest.raises(RuntimeError, match="incomplete sibling"):
        solve(problem(), backend, FakeEvaluator(), SearchConfig())


def test_context_overflow_prunes_that_path_without_consuming_expansion_budget():
    context_error = ValueError("prompt + output budget exceeds max_context_tokens; reduce history explicitly")
    backend = FakeBackend([[sample(0), sample(1)], context_error, [sample(2), sample(3)]])
    cfg = SearchConfig(simulations=2, width=2, max_depth=2, train=False)
    result = solve(problem(), backend, FakeEvaluator(), cfg)
    assert result["context_pruned_nodes"] == 1
    assert result["expansions"] == 2
    assert result["sampled_candidates"] == 4
    assert len(backend.calls) == 3


def test_empty_completion_skips_whole_group_without_inventing_tokens():
    backend = FakeBackend([[sample(0, tokens=0, text=""), sample(1), sample(2)]])
    evaluator = FakeEvaluator({"print(1)": 1})
    result = solve(problem(), backend, evaluator, SearchConfig(simulations=1))
    assert not backend.updates
    assert result["generated_tokens"] == 4
    assert result["skipped_training_groups"] == 1
    assert result["training_updates"] == result["greedy_candidates"] == 0



def test_private_counts_change_training_rewards_and_public_success_does_not_stop():
    backend = FakeBackend([[sample(0), sample(1)], [sample(2), sample(3)]])
    evaluator = FakeEvaluator({f"print({i})": 2 for i in range(4)},
                              {"print(0)": 0, "print(1)": 1, "print(2)": 2, "print(3)": 0})
    result = solve(problem(), backend, evaluator,
                   SearchConfig(simulations=2, width=2, greedy_commitment=False))
    assert [r for _, r in backend.updates[0][0]] == [0.5, 0.75]
    assert len(backend.calls) == 2
    assert result['code'] == 'print(2)' and result['stop_reason'] == 'all_tests_passed'
    assert result['optimization_score'] == 1.0 and result['reward_tests'] == 'all'
    assert '0.75' not in str(backend.calls[1][0]) and 'THIS_SECRET' not in str(backend.calls)


def test_selected_solution_uses_all_test_reward_when_public_rewards_tie():
    backend = FakeBackend([[sample(0), sample(1)]], steps=0)
    result = solve(problem(), backend, FakeEvaluator({'print(0)': 2, 'print(1)': 2},
                   {'print(0)': 0, 'print(1)': 1}), SearchConfig(simulations=1, width=2))
    assert result['code'] == 'print(1)' and result['optimization_score'] == 0.75
    assert result['stop_reason'] == 'budget_exhausted'


def test_private_success_cannot_stop_when_a_public_test_fails():
    backend = FakeBackend([[sample(0), sample(1)]], steps=0)
    result = solve(problem(), backend, FakeEvaluator({'print(0)': 1, 'print(1)': 1},
                   {'print(0)': 2, 'print(1)': 2}), SearchConfig(simulations=1, width=2))
    assert result['evaluated_candidates'] == 2 and result['stop_reason'] == 'budget_exhausted'


def test_legacy_reward_and_stop_switches_cannot_silently_restore_wrong_protocol():
    for kwargs in [{'early_stop': 'public'}, {'early_stop': 'none'}, {'reward_tests': 'public'}]:
        with pytest.raises(TypeError):
            SearchConfig(**kwargs)


def test_evaluator_errors_are_redacted_before_reaching_search_logs():
    class BrokenEvaluator:
        def evaluate_candidate(self, problem, code):
            raise ValueError('PRIVATE_TEST_SECRET')
    with pytest.raises(RuntimeError) as error:
        solve(problem(), FakeBackend([[sample(0), sample(1), sample(2)]]), BrokenEvaluator(), SearchConfig())
    assert 'PRIVATE_TEST_SECRET' not in str(error.value)


def test_functional_prompt_requires_solution_method_and_return_value():
    p = problem()
    p.starter_code = 'class Solution:\n    def solve(self, nums: List[int]) -> int:\n'
    prompt = str(problem_messages(p, Node(0)))
    assert 'class Solution' in prompt and 'Return the answer from the method' in prompt


def test_feedback_identifies_exact_selected_program_and_ignores_other_blocks():
    text = ('An example:\n```text\n1\n```\n'
            'First attempt:\n```python\nprint(999)\n```\n'
            'Final attempt:\n```python3\nprint(2)\n```\n'
            'Cut-off revision:\n```python\nprint(')
    backend = FakeBackend([[sample(0, text=text)], [sample(3)]], steps=0)
    evaluator, events = FakeEvaluator(), []
    solve(problem(), backend, evaluator, SearchConfig(width=1, simulations=2), emit=events.append)
    assert evaluator.codes == ['print(2)', 'print(3)']
    feedback = backend.calls[1][0][-1]['content']
    assert 'Python block 2 of 2' in feedback and 'fenced block 3 overall' in feedback
    assert 'unfinished trailing Python block was ignored' in feedback
    assert '```python\nprint(2)\n```' in feedback and 'print(999)' not in feedback
    assert 'THIS_SECRET' not in str(backend.calls)
    assert events[0]['evaluation_status'] == 'evaluated'
    assert events[0]['code_selection']['python_block_index'] == 2
    assert backend.updates[1][0][0][0].trajectory_prefix == [backend.updates[0][0][0][0]]


@pytest.mark.parametrize('text,status,code', [
    ('No Python here.', 'format_error', ''),
    ('```python\ndef broken(:\n```', 'syntax_error', 'def broken(:'),
    ('```python\nprint(1)\n```\n```python\n\n```', 'format_error', ''),
])
def test_nonexecuted_candidate_is_not_reported_as_executed(text, status, code):
    backend = FakeBackend([[sample(0, text=text)], [sample(1)]], steps=0)
    evaluator, events = FakeEvaluator(), []
    solve(problem(), backend, evaluator, SearchConfig(width=1, simulations=2), emit=events.append)
    feedback = backend.calls[1][0][-1]['content']
    assert 'no code was executed' in feedback
    assert events[0]['evaluation_status'] == status and events[0]['code'] == code
    assert evaluator.codes == ['print(1)']
    assert code in feedback


def test_public_success_feedback_still_quotes_selected_code_without_private_score():
    text = '```python\nprint(0)\n```\n```py\nprint(1)\n```'
    node = Node(1, parent=Node(0), text=text, code='print(1)', selected_block=select_code(text),
                evaluation_status='evaluated', feedback='All public examples passed.',
                public_passed=2, public_total=2, reward=.87654321)
    feedback = problem_messages(problem(), node)[-1]['content']
    assert 'Python block 2 of 2' in feedback and '```python\nprint(1)\n```' in feedback
    assert 'print(0)' not in feedback and '0.87654321' not in feedback
    assert 'Your previous code had execution issues' not in feedback


def test_greedy_commitment_is_retained_as_an_ancestor_of_later_training_group():
    roots = [sample(0), sample(1), sample(2)]
    greedy = sample(3, version=1)
    leaves = [sample(4, version=1), sample(5, version=1), sample(6, version=1)]
    backend = FakeBackend([roots, [greedy], leaves, [sample(7, version=2)]])
    solve(problem(), backend, FakeEvaluator({'print(3)': 1}), SearchConfig(simulations=2))
    assert len(backend.updates) == 2
    assert all(leaf.trajectory_prefix == [greedy] for leaf, _ in backend.updates[1][0])
    assert greedy.trajectory_prefix == []
