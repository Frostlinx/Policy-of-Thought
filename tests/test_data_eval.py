import base64
import dataclasses
import json
import os
import pickle
import tempfile
import unittest
import zlib
import uuid
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

from pot.data import (EXPECTED_COUNTS, SELECTION_FILES, PrivateTestStore, TestCase,
                      _decode_private, load_dataset, save_dataset, split_records)
from pot.evaluator import (DockerConfig, DockerEvaluator, PrivateScore, _Execution,
                           _RUNNER, _decode_prediction, compare_functional, compare_stdio)


TEST_WORK = Path(__file__).resolve().parents[1] / "work" / "test-data-eval"


@contextmanager
def scratch():
    TEST_WORK.mkdir(parents=True, exist_ok=True)
    directory = TEST_WORK / uuid.uuid4().hex
    directory.mkdir()
    try:
        yield directory
    finally:
        for path in directory.iterdir():
            path.chmod(0o600)
            path.unlink()
        directory.rmdir()


def fixture(private="PRIVATE_SENTINEL_84731"):
    return {"platform": "atcoder", "question_id": "demo", "question_title": "Double",
            "question_content": "Print twice the integer.", "starter_code": "", "difficulty": "easy",
            "contest_date": "2025-02-01T12:00:00", "public_test_cases": json.dumps([
                {"input": "2\n", "output": "4\n", "testtype": "stdin"}]),
            "private_test_cases": json.dumps([
                {"input": private, "output": private, "testtype": "stdin"}]),
            "metadata": json.dumps({"func_name": None, "internal_private_hint": private})}


class FakeEvaluator(DockerEvaluator):
    def _run(self, candidate_path, test, func_name):
        return _Execution(ok=False, stdout=test.input, stderr=test.output, status="runtime_error")


class DataBoundaryTests(unittest.TestCase):
    def test_v6_is_delta(self):
        self.assertEqual(SELECTION_FILES["v6"], ("test6.jsonl",))
        self.assertEqual(EXPECTED_COUNTS["v6"], 175)
        self.assertEqual(EXPECTED_COUNTS["release_v6"], 1055)
        self.assertEqual(len(SELECTION_FILES["release_v6"]), 6)

    def test_public_serialization_never_has_private_tests_or_metadata(self):
        public, store = split_records([fixture()])
        text = json.dumps(public[0].to_dict()) + repr(public[0]) + repr(store)
        self.assertNotIn("PRIVATE_SENTINEL", text)
        self.assertNotIn("internal_private_hint", text)
        self.assertNotIn("private_test_cases", public[0].to_dict())
        self.assertFalse(hasattr(public[0], "metadata"))

    def test_official_compressed_private_envelope(self):
        content = '[{"input":"secret","output":"answer","testtype":"stdin"}]'
        envelope = base64.b64encode(zlib.compress(pickle.dumps(content))).decode()
        self.assertEqual(_decode_private(envelope), json.loads(content))

    def test_pickle_global_never_executes(self):
        # GLOBAL/REDUCE payload attempting eval; restricted unpickler must reject.
        malicious = b"cbuiltins\neval\n(S'1+1'\ntR."
        envelope = base64.b64encode(zlib.compress(malicious)).decode()
        with self.assertRaisesRegex(ValueError, "Invalid private-test encoding"):
            _decode_private(envelope)

    def test_malformed_private_error_is_redacted(self):
        row = fixture()
        row["private_test_cases"] = "PRIVATE_SENTINEL: not base64 or JSON"
        with self.assertRaises(ValueError) as caught:
            split_records([row])
        self.assertNotIn("PRIVATE_SENTINEL", str(caught.exception))

    def test_roundtrip_separate_files_and_full_day_date_filter(self):
        public, store = split_records([fixture()])
        with scratch() as tmp:
            save_dataset(tmp, public, store, {"selection": "v6"})
            self.assertNotIn("PRIVATE_SENTINEL", (Path(tmp) / "public.jsonl").read_text())
            loaded, restored = load_dataset(tmp, end_date="2025-02-01")
            self.assertEqual(loaded, public)
            self.assertEqual(len(restored._for_evaluation(public[0].problem_id)), 1)
            self.assertEqual(load_dataset(tmp, start_date="2025-02-02")[0], [])
            with self.assertRaisesRegex(ValueError, "selection"):
                load_dataset(tmp, selection="release_v6")

    def test_unexpected_private_field_in_public_json_is_not_forwarded(self):
        public, store = split_records([fixture()])
        with scratch() as tmp:
            save_dataset(tmp, public, store, {"selection": "v6"})
            path = Path(tmp) / "public.jsonl"
            row = json.loads(path.read_text())
            row["private_test_cases"] = "PRIVATE_SENTINEL"
            path.write_text(json.dumps(row))
            loaded, _ = load_dataset(tmp)
            self.assertNotIn("PRIVATE_SENTINEL", repr(loaded))

    def test_private_evaluation_and_combined_score_never_return_errors(self):
        public, store = split_records([fixture()])
        evaluator = FakeEvaluator(DockerConfig(work_dir=str(TEST_WORK)), store)
        result = evaluator.evaluate_private(public[0], "pass", store)
        self.assertEqual(result, PrivateScore(passed=0, total=1, all_passed=False))
        self.assertEqual(set(dataclasses.asdict(result)), {"passed", "total", "all_passed", "gate_passed", "gate_total"})
        self.assertNotIn("PRIVATE_SENTINEL", repr(result))
        self.assertIs(evaluator.evaluate_candidate(public[0], "pass").all_passed, False)
        self.assertNotIn("PRIVATE_SENTINEL", repr(evaluator.evaluate_public(public[0], "pass")))


class ComparisonTests(unittest.TestCase):
    def test_official_decimal_not_approximate_equality(self):
        self.assertTrue(compare_stdio("  1.00  2e0 \n", "1 2"))
        self.assertFalse(compare_stdio("50000000000000000", "50000000000000001"))
        self.assertFalse(compare_stdio("1.00000001", "1.0"))

    def test_line_count_and_string_spacing_are_preserved(self):
        self.assertTrue(compare_stdio("\n yes \n no \n", "yes\nno"))
        self.assertFalse(compare_stdio("yes  no", "yes no"))
        self.assertFalse(compare_stdio("1 2", "1\n2"))
        self.assertFalse(compare_stdio("x\n\ny", "x\ny"))

    def test_functional_json_and_tuple_semantics(self):
        self.assertTrue(compare_functional((1, 2), "[1,2]"))
        self.assertTrue(compare_functional({"a": [1]}, '{"a":[1]}'))
        self.assertFalse(compare_functional([2, 1], "[1,2]"))
        self.assertFalse(compare_functional([(1, 2)], "[[1,2]]"))

    def test_functional_transport_does_not_coerce_nested_tuples(self):
        encoded = {"kind": "list", "value": [{"kind": "tuple", "value": [
            {"kind": "scalar", "value": 1}, {"kind": "scalar", "value": 2}]}]}
        prediction = _decode_prediction(encoded)
        self.assertEqual(prediction, [(1, 2)])
        self.assertFalse(compare_functional(prediction, "[[1,2]]"))


class SandboxPolicyTests(unittest.TestCase):
    def test_wrapper_is_syntactically_valid(self):
        compile(_RUNNER, "<container-runner>", "exec")

    def test_docker_mount_and_environment_boundary(self):
        cfg = DockerConfig(command=("/usr/bin/docker",), host="unix:///tmp/docker.sock", config_dir="/tmp/docker-config")
        evaluator = DockerEvaluator(cfg)
        command = evaluator._command(Path("/tmp/candidate.py"), "test-container", "stdin", None)
        joined = " ".join(command)
        for required in ("--network none", "--read-only", "--cap-drop ALL", "no-new-privileges",
                         "--pids-limit 64", "--memory 512m", "--user 65534:65534", "--pull never"):
            self.assertIn(required, joined)
        self.assertEqual(command.count("--mount"), 1)
        self.assertIn("dst=/candidate.py,readonly", command[command.index("--mount") + 1])
        self.assertNotIn("HF_TOKEN", joined)
        self.assertNotIn("private_tests.jsonl", joined)
        self.assertNotIn("docker.sock,", joined)
        self.assertNotIn("--privileged", joined)

    def test_command_env_supports_argv_without_shell(self):
        with patch.dict(os.environ, {"POT_DOCKER_COMMAND": '["docker","--context","research"]'}):
            self.assertEqual(DockerConfig().command, ("docker", "--context", "research"))


@unittest.skipUnless(os.environ.get("POT_TEST_DOCKER") == "1", "Set POT_TEST_DOCKER=1 for actual isolated execution")
class DockerIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.evaluator = DockerEvaluator()
        self.evaluator.check_available()
        self.problem = split_records([fixture()])[0][0]

    def test_stdin_and_buffer(self):
        result = self.evaluator.evaluate_public(self.problem, "import sys\nx=int(sys.stdin.buffer.read())\nprint(x*2)")
        self.assertTrue(result.all_passed, result.feedback)

    def test_functional_starter_and_implicit_typing_imports(self):
        problem = dataclasses.replace(self.problem, func_name="twice", public_tests=(TestCase("[1,2]", "[2,4]", "functional"),))
        result = self.evaluator.evaluate_public(problem, "class Solution:\n def twice(self, nums: List[int]) -> List[int]:\n  return [x*2 for x in nums]")
        self.assertTrue(result.all_passed, result.feedback)

    def test_container_cannot_see_host_token_or_private_file(self):
        code = "import os\nassert 'HF_TOKEN' not in os.environ\nassert not os.path.exists('/private_tests.jsonl')\nprint(4)"
        with patch.dict(os.environ, {"HF_TOKEN": "HOST_SECRET_SENTINEL"}):
            self.assertTrue(self.evaluator.evaluate_public(self.problem, code).all_passed)


if __name__ == "__main__":
    unittest.main()
