from pot.prompt_probe import inspect_choice
from pot.search import extract_code


class Tokenizer:
    def encode(self, text, **kwargs):
        return list(text.encode())


def test_reasoning_then_final_code_is_extractable():
    assert extract_code('The sum is sufficient.\n```python\nprint(3)\n```') == 'print(3)'


def test_length_stop_counted_even_with_valid_code(tmp_path):
    choice = {'index': 0, 'text': 'Reasoning.\n```python\nprint(3)\n```',
              'token_ids': [1]*2048, 'finish_reason': 'length'}
    result = inspect_choice(choice, Tokenizer(), tmp_path/'candidate.txt')
    assert result['truncated'] and result['output_tokens'] == 2048
    assert result['python_syntax_valid'] and result['has_analysis_before_code']
    assert result['score'] is None and result['pass_rate'] is None


def test_cut_off_code_is_not_executed_or_repaired(tmp_path):
    choice = {'index': 1, 'text': 'Reasoning.\n```python\ndef solve(',
              'token_ids': [1]*2048, 'finish_reason': 'length'}
    result = inspect_choice(choice, Tokenizer(), tmp_path/'candidate.txt')
    assert result['truncated'] and not result['code_extractable']
    assert not (tmp_path/'candidate.py').exists()
