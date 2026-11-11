"""HumanEval / sanitized MBPP adapters; reference solutions are never retained.

HumanEval retains the unmodified checker as a mandatory test and adds assertion-
site probes for fractional rewards. Loops stay whole. MBPP's published tests are
public units; no additional hidden MBPP cases are invented.
"""
from __future__ import annotations

import ast
import copy
from dataclasses import replace
import json
import re
from typing import Iterable, Mapping, Any

from .data import PublicProblem, PrivateTestStore, TestCase


def assertion_test(test: str, *, setup: str = "", prefix: str = "") -> TestCase:
    """Only serialize here; all executable strings run inside the evaluator child."""
    if not isinstance(test, str) or not test.strip():
        raise ValueError("Missing Python test program")
    for text in (test, setup, prefix):
        if not isinstance(text, str):
            raise ValueError("Python test fields must be strings")
        ast.parse(text)
    return TestCase(json.dumps(dict(prefix=prefix, setup=setup, test=test)), "", "python_assert")


def public_test_text(test: TestCase) -> str:
    if getattr(test, "testtype", "stdin") != "python_assert":
        return f"Input:\n{test.input}\nOutput:\n{test.output}"
    payload = json.loads(test.input)
    return "\n".join(payload.get(key, "") for key in ("prefix", "setup", "test") if payload.get(key))


def humaneval_units(test: str, entry: str, prompt: str) -> tuple[TestCase, ...]:
    """Original suite plus one probe per assertion statement (not per loop item).

    Each probe executes the entire checker with one assertion site enabled.
    Other assertions still evaluate their conditions/truthiness, preserving
    candidate calls and side effects, but do not stop on a false condition.
    Earlier runtime exceptions still fail the probe. The original unmodified
    checker is always a separate mandatory unit, so probes cannot by themselves
    make an officially failing program a success. These are reward probes, not
    claimed independent input/output cases or a held-out evaluation protocol.
    """
    call = f"\ncheck({entry})\n"
    units = [assertion_test(test + call, prefix=prompt)]
    tree = ast.parse(test)
    # A checker intentionally catching assertion exceptions may depend on their
    # control flow. Be conservative for any try/except suite; never rewrite it.
    if any(isinstance(n, (ast.Try, getattr(ast, "TryStar", ast.Try))) for n in ast.walk(tree)):
        return tuple(units)
    sites = sorted((n.lineno, n.col_offset) for n in ast.walk(tree) if isinstance(n, ast.Assert))
    if not sites:
        return tuple(units)
    # The original suite gates success but is NOT an extra rewarded assertion.
    units[0] = replace(units[0], role="gate")

    class Probe(ast.NodeTransformer):
        def __init__(self, selected): self.selected = selected
        def visit_Assert(self, node):
            if (node.lineno, node.col_offset) == self.selected:
                return node
            # `if` applies Python's original truthiness operation without calling
            # a potentially shadowed bool(). Do NOT discard condition evaluation.
            return ast.copy_location(ast.If(test=node.test, body=[ast.Pass()], orelse=[]), node)

    for site in sites:
        probe = ast.fix_missing_locations(Probe(site).visit(copy.deepcopy(tree)))
        units.append(assertion_test(ast.unparse(probe) + call, prefix=prompt))
    return tuple(units)


def normalize_humaneval(records: Iterable[Mapping[str, Any]]):
    public, private = [], {}
    for row in records:
        task_id, prompt, entry = row["task_id"], row["prompt"], row["entry_point"]
        if not isinstance(task_id, str) or not re.fullmatch(r"HumanEval/\d+", task_id):
            raise ValueError("Invalid HumanEval task ID")
        if task_id in private:
            raise ValueError("Duplicate HumanEval task ID")
        if not isinstance(entry, str) or not entry.isidentifier():
            raise ValueError("Invalid HumanEval entry point")
        tree = ast.parse(prompt)
        if entry not in {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}:
            raise ValueError("HumanEval prompt lacks its declared entry point")
        test = row["test"]
        if "check" not in {n.name for n in ast.parse(test).body if isinstance(n, ast.FunctionDef)}:
            raise ValueError("HumanEval checker is missing")
        # Prompt supplies the original imports and stub, then the generated full
        # module replaces that stub. Never append or retain canonical_solution.
        private[task_id] = humaneval_units(test, entry, prompt)
        public.append(PublicProblem(task_id, task_id.split("/")[1], "humaneval", entry,
            "Implement the function specified below. The docstring includes public examples.",
            prompt, "", "", (), entry, "humaneval", "module"))
    return public, PrivateTestStore(private)


def normalize_mbpp(records: Iterable[Mapping[str, Any]]):
    public, private = [], {}
    for row in records:
        task_id = row["task_id"]
        if isinstance(task_id, bool) or not isinstance(task_id, int) or not 11 <= task_id <= 510:
            raise ValueError("MBPP sanitized test must contain only task IDs 11 through 510")
        problem_id = f"MBPP/{task_id}"
        if problem_id in private:
            raise ValueError("Duplicate MBPP task ID")
        tests, imports = row["test_list"], row.get("test_imports", [])
        if not isinstance(tests, list) or not tests:
            raise ValueError("Expected nonempty published MBPP tests")
        if not isinstance(imports, list) or not all(isinstance(x, str) for x in imports):
            raise ValueError("Invalid MBPP test imports")
        setup = "\n".join(imports) + "\n" + (row.get("test_setup_code") or "")
        cases = []
        for test in tests:
            tree = ast.parse(test)
            if not any(isinstance(n, ast.Assert) for n in ast.walk(tree)):
                raise ValueError("MBPP test lacks an assertion")
            cases.append(assertion_test(test, setup=setup))
        # The published assertions specify the function names/signatures without
        # deriving them from row['code'], which is the reference implementation.
        public.append(PublicProblem(problem_id, str(task_id), "mbpp", f"MBPP {task_id}",
            str(row["prompt"]), "", "", "", tuple(cases), None, "mbpp-sanitized", "module"))
        private[problem_id] = ()
    return public, PrivateTestStore(private)


def benchmark_protocol(selection: str) -> dict:
    if selection == "ojbench-icpc":
        return {"benchmark": selection, "split": "icpc/python", "test_unit": "official_judge_case",
                "reward_granularity": "passed_public_and_private_cases_over_total",
                "public_examples": "statement_samples; interactive_transcripts_not_replayed",
                "private_tests_available": True, "judge": "DMOJ/OJBench",
                "success": "all_cases_full_scale", "official_point_weights_used": False,
                "statement_errata": "nwerc2024_J_sample1_missing_first_output_line"}
    if selection == "humaneval":
        return {"benchmark": selection, "split": "test", "test_unit": "assertion_site_probe",
                "reward_granularity": "passed_assertion_sites_over_total", "public_examples": "prompt_only",
                "official_suite_required": True, "probe_policy": "whole_loop_per_site; preserve_condition_calls; no_rewrite_if_try",
                "official_suite_in_reward": False, "fallback_if_no_safe_probes": "whole_suite_binary",
                "private_tests_available": True}
    if selection == "mbpp-sanitized":
        return {"benchmark": selection, "split": "sanitized/test", "test_unit": "published_assertion",
                "reward_granularity": "passed_published_tests_over_total", "public_examples": "evaluated",
                "private_tests_available": False}
    return {"benchmark": "lcb", "split": selection, "test_unit": "input_output_case",
            "reward_granularity": "passed_public_and_private_cases_over_total", "private_tests_available": True}
