"""Selection, feedback boundary and real official-judge integration checks."""
from dataclasses import replace
import json
import os
from pathlib import Path
import pytest

from pot.data import SELECTION_FILES, EXPECTED_COUNTS, TestCase as Case, PublicProblem, PrivateTestStore
from pot.evaluator import DockerConfig
from pot.ojbench import statement_samples, OJPrivateStore, sha256, pinned_files, verify_raw_file, correct_statement
from pot.search import problem_messages, Node


def test_lcb_v5_is_delta_not_cumulative():
    assert SELECTION_FILES['v5'] == ('test5.jsonl',)
    assert EXPECTED_COUNTS['v5'] == 167
    assert EXPECTED_COUNTS['release_v5'] == 880
    assert 'test6.jsonl' not in SELECTION_FILES['release_v5']


def test_upstream_pins_and_truncated_download_rejected(tmp_path):
    import hashlib
    pins = pinned_files()
    assert len(pins) == 181
    assert len([p for p in pins if p.endswith('/data.zip')]) == 73
    assert 'ICPC/nwerc2023_H/init.yml' not in pins
    f = tmp_path / 'asset'
    f.write_bytes(b'ab')
    spec = {'bytes': 3, 'sha256': hashlib.sha256(b'abc').hexdigest()}
    assert not verify_raw_file(f, spec)
    f.write_bytes(b'abd')
    assert not verify_raw_file(f, spec)
    f.write_bytes(b'abc')
    assert verify_raw_file(f, spec)
    assert verify_raw_file(f, {'bytes': 3, 'git_blob_sha1': hashlib.sha1(b'blob 3\0abc').hexdigest()})


def test_only_explicit_statement_samples():
    s = '## Sample Input 1\n```\n2\n```\n## Sample Output 1\n```\n4\n```\n'
    assert statement_samples(s) == (Case('2\n', '4\n'),)
    s = '## Sample Input/Output\n\n### Sample 1\n```\nOK\n```\n```\n.-.-.-\n```\n'
    assert statement_samples(s) == (Case('OK\n', '.-.-.-\n'),)
    with pytest.raises(ValueError):
        statement_samples('## Sample Input 1\n```\n2\n```\n')


def test_public_statement_erratum_is_narrow_and_auditable():
    original = '### Sample Output 1\n```\n5\n3\n```'
    assert '3\n5\n3' in correct_statement('nwerc2024_J', original)
    assert correct_statement('another_problem', original) == original
    with pytest.raises(ValueError):
        correct_statement('nwerc2024_J', 'unexpected revision')


@pytest.mark.skipif(os.name != 'posix', reason='POSIX venv executable layout')
def test_cli_preserves_venv_interpreter_entrypoint(tmp_path):
    import sys
    from pot.run import build_parser, parameters_from_args, _executable_path
    entry = tmp_path / 'venv' / 'bin' / 'python'
    entry.parent.mkdir(parents=True)
    entry.symlink_to(sys.executable)
    args = build_parser().parse_args(['--model', str(tmp_path), '--data', str(tmp_path),
        '--output', str(tmp_path / 'run'), '--ojbench-python', str(entry)])
    settings = parameters_from_args(args)
    assert settings['evaluation']['ojbench_python'] == str(entry)
    assert _executable_path(str(entry), tmp_path) == str(entry)


def problem(pid='plain', interface='stdio'):
    return PublicProblem('ICPC/'+pid, pid, 'ojbench', 'Example', 'Read x and print 2*x.',
                         '', '', '', (), None, 'ojbench-icpc', interface)


def test_interactive_prompt_preserves_streaming_protocol():
    text = json.dumps(problem_messages(problem(interface='interactive'), Node(0)))
    assert 'flush stdout' in text and 'Do not read all stdin at once' in text
    assert 'See the function docstring' not in text


JUDGE_PYTHON = os.environ.get('POT_OJBENCH_PYTHON')
JUDGE_PYPY = os.environ.get('POT_OJBENCH_PYPY')


@pytest.mark.skipif(not (JUDGE_PYTHON and JUDGE_PYPY), reason='Optional pinned DMOJ/PyPy runtime is not configured')
def test_real_dmoj_plain_float_custom_interactive_and_private_boundary(tmp_path):
    import yaml
    from pot.ojbench_evaluator import OJBenchEvaluator
    root = tmp_path / 'judge'
    configs = {
        'plain': {'test_cases': [{'in': 'a.in', 'out': 'a.ans'}, {'in': 'b.in', 'out': 'b.ans'}]},
        'floating': {'checker': {'name': 'floats', 'args': {'precision': 6}}, 'test_cases': [{'in': 'a.in', 'out': 'a.ans'}]},
        'custom': {'checker': {'name': 'bridged', 'args': {'files': ['checker.cpp'], 'lang': 'CPP17', 'type': 'default'}}, 'test_cases': [{'in': 'a.in', 'out': 'a.ans'}]},
        'interactive': {'interactive': {'files': ['interactor.cpp'], 'lang': 'CPP17', 'type': 'default'},
                        'unbuffered': False, 'test_cases': [{'in': 'a.in', 'out': 'a.ans'}]},
    }
    for pid, config in configs.items():
        d = root / pid
        d.mkdir(parents=True)
        (d/'init.yml').write_text(yaml.safe_dump(config))
        (d/'a.in').write_text('2\n')
        (d/'a.ans').write_text('4\n')
        (d/'b.in').write_text('3\n')
        (d/'b.ans').write_text('6\n')
    # DMOJ default protocol: argv[2]=submission output, return 0/1.
    (root/'custom'/'checker.cpp').write_text('#include <iostream>\n#include <fstream>\nint main(int argc,char** argv){int x=-1; std::ifstream f(argv[2]); f>>x; std::cerr<<"PRIVATE_SENTINEL"; return x==4?0:1;}\n')
    (root/'interactive'/'interactor.cpp').write_text('#include <iostream>\nint main(){std::cout<<2<<std::endl; int x; std::cin>>x; std::cerr<<"PRIVATE_SENTINEL"; return x==4?0:1;}\n')
    files = [{'path': f.relative_to(tmp_path).as_posix(), 'bytes': f.stat().st_size, 'sha256': sha256(f)}
             for f in root.rglob('*') if f.is_file()]
    (tmp_path/'judge_manifest.json').write_text(json.dumps({'files': files}))
    problems = [problem(pid, 'interactive' if pid == 'interactive' else 'stdio') for pid in configs]
    hidden = PrivateTestStore({p.problem_id: tuple(Case('', '') for _ in configs[p.question_id]['test_cases']) for p in problems})
    store = OJPrivateStore(tmp_path, problems, hidden)
    cfg = DockerConfig(ojbench_python=JUDGE_PYTHON, ojbench_pypy=JUDGE_PYPY, work_dir=str(tmp_path/'scratch'),
                       timeout_seconds=1, memory_mb=1024, eval_workers=2, eval_global_workers=2)
    judge = OJBenchEvaluator(cfg, store)
    judge.timing_log = tmp_path/'timing.jsonl'
    try:
        info = judge.check_available()
        assert info['runtime'] == 'ojbench-dmoj'
        partial = judge.evaluate_candidate(problems[0], 'print(4)')
        assert partial.reward == .5 and not partial.all_passed
        for p in problems:
            result = judge.evaluate_candidate(p, 'x=int(input()); print(x*2, flush=True)')
            assert result.all_passed, p.question_id
            assert result.public.feedback == ()
            wrong = judge.evaluate_candidate(p, 'print(-1, flush=True)')
            assert wrong.reward == 0 and not wrong.all_passed
        assert judge.evaluate_candidate(problems[1], 'print(4.0000001)').all_passed
        assert not judge.evaluate_candidate(problems[1], 'print(4.01)').all_passed
        assert 'PRIVATE_SENTINEL' not in judge.timing_log.read_text()
    finally:
        judge.close()
