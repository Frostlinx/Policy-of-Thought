"""Exercise actual subprocess evaluation with synthetic public/private tests."""
import os
import pytest
from pot.data import PublicProblem, PrivateTestStore, TestCase as Case
from pot.evaluator import DockerConfig
from pot.process_evaluator import ProcessEvaluator

pytestmark = pytest.mark.skipif(os.name != 'posix', reason='POSIX executor')


@pytest.mark.parametrize('functional', [False, True])
def test_private_results_drive_reward_and_success_without_private_feedback(tmp_path, functional):
    kind = 'functional' if functional else 'stdin'
    p = PublicProblem('test/1','1','test','Double','Double the input','', 'easy','2025-01-01',
                      (Case('2','4',kind),), 'double' if functional else None)
    store = PrivateTestStore({p.problem_id:(Case('3','6',kind),Case('4','8',kind))})
    evaluator = ProcessEvaluator(DockerConfig(work_dir=str(tmp_path), timeout_seconds=2,memory_mb=1024),store)
    bad = 'class Solution:\n def double(self, n: int): return 4' if functional else 'print(4)'
    good = 'class Solution:\n def double(self, n: int): return n*2' if functional else 'print(int(input())*2)'
    first = evaluator.evaluate_candidate(p,bad)
    assert first.public.all_passed and first.reward == pytest.approx(1/3) and not first.all_passed
    assert not hasattr(first.private,'feedback') and first.public.feedback == ()
    passed = evaluator.evaluate_candidate(p,good)
    assert passed.reward == 1 and passed.all_passed
    assert evaluator.evaluate_candidate(p,bad) is first
    assert evaluator.evaluate_candidate(p,good) is passed


def test_private_runtime_error_is_not_exposed_and_cached(tmp_path):
    p=PublicProblem('test/1','1','test','Echo','Echo the input','', 'easy','2025-01-01',(Case('public','public'),))
    secret='PRIVATE_ERROR_CONTENT_73258'
    store=PrivateTestStore({p.problem_id:(Case(secret,secret),)})
    e=ProcessEvaluator(DockerConfig(work_dir=str(tmp_path),timeout_seconds=2,memory_mb=1024),store)
    result=e.evaluate_candidate(p,"x=input()\nif x!='public': raise RuntimeError(x)\nprint(x)")
    assert result.reward==0.5 and not result.all_passed and result.public.feedback==()
    assert secret not in repr(result)


def test_functional_public_feedback_displays_return_value(tmp_path):
    p=PublicProblem('test/1','1','test','Double','Double','', 'easy','2025-01-01',(Case('2','4','functional'),),'double')
    e=ProcessEvaluator(DockerConfig(work_dir=str(tmp_path),timeout_seconds=2,memory_mb=1024))
    result=e.evaluate_public(p,'class Solution:\n def double(self, x): return 3')
    assert result.feedback[0]['actual']=='3'
