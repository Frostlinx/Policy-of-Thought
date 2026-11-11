import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from pot import prompts
from pot.search import Node, problem_messages


def problem():
    return SimpleNamespace(question_content='Return the sum of the two arguments.',
        starter_code='class Solution:\n    def add(self, a: int, b: int) -> int:\n        pass',
        public_tests=[], private_tests='PRIVATE_INPUT_SENTINEL')


def test_template_literals_match_recorded_hashes():
    path = Path(__file__).parents[1]/'docs/original-prompt-provenance.json'
    metadata = json.loads(path.read_text())
    for name, digest in metadata['template_sha256'].items():
        assert hashlib.sha256(getattr(prompts, name).encode()).hexdigest() == digest


def test_original_initial_template_and_functional_interface_are_both_present():
    messages = problem_messages(problem(), Node(0))
    text = messages[1]['content']
    before, after = prompts.CODING_START_PROMPT.split('{problem}')
    assert text.startswith(before)
    assert after in text
    assert 'def add(self, a: int, b: int)' in text
    assert 'class Solution' in text and 'do not read stdin' in text
    assert 'PRIVATE_INPUT_SENTINEL' not in str(messages)
    assert 'with no other text' not in str(messages)
    assert 'exactly one fenced code block' not in str(messages)
    assert 'seen = {}' in text


def test_problem_braces_are_not_reinterpreted_as_template_slots():
    text = prompts.render_template(prompts.CODING_START_PROMPT, problem='Print {problem} and {previous_steps}.')
    assert 'Print {problem} and {previous_steps}.' in text


def test_original_error_instructions_receive_public_fields_only():
    node = Node(1, parent=Node(0), text='A previous attempt.', feedback='PUBLIC_WA_SENTINEL',
                public_passed=1, public_total=3, reward=.87654321)
    messages = problem_messages(problem(), node)
    last = messages[-1]['content']
    assert 'Analyze the error and understand what went wrong' in last
    assert 'PUBLIC_WA_SENTINEL' in last and '33.3%' in last
    assert '.87654321' not in str(messages) and 'PRIVATE_INPUT_SENTINEL' not in str(messages)


def test_private_failure_does_not_invent_public_error_feedback():
    node = Node(1, parent=Node(0), text='A previous attempt.', feedback='All public examples passed.',
                public_passed=3, public_total=3, reward=.25)
    text = problem_messages(problem(), node)[-1]['content']
    assert 'Your previous code had execution issues' not in text
    assert 'REFLECTION QUESTIONS' in text and '.25' not in text
