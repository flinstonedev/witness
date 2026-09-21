from __future__ import annotations

import copy
import importlib.util
from pathlib import Path
import unittest

from witness_bench.runner import BenchmarkConfig, run_benchmark


spec = importlib.util.spec_from_file_location(
    "check_benchmark", Path(__file__).resolve().parents[1] / "scripts" / "check_benchmark.py"
)
assert spec is not None and spec.loader is not None
checker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checker)


class ReproducibilityChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.result = run_benchmark(BenchmarkConfig(
            profile="smoke", repetitions=1,
            task_ids=("rare_termination_exception",),
            system_names=("vector-rag-tfidf-k10",),
            bootstrap_resamples=20,
        ))

    def test_machine_and_timing_variation_does_not_change_evidence(self) -> None:
        other = copy.deepcopy(self.result)
        other["manifest"].update(created_at="different time", python="different Python", platform="different OS")
        other["runs"][0]["metrics"]["latency_ms"] += 100
        other["runs"][0]["result"]["telemetry"]["cpu_time_ms"] += 100
        checker.compare(self.result, other)

    def test_changed_evidence_metrics_and_inputs_are_detected(self) -> None:
        for section in ("evidence", "metric", "corpus", "implementation"):
            with self.subTest(section=section):
                other = copy.deepcopy(self.result)
                if section == "evidence":
                    other["runs"][0]["result"]["passage_set_hash"] = "changed"
                elif section == "metric":
                    other["runs"][0]["metrics"]["evidence_recall"] = -1
                elif section == "corpus":
                    other["manifest"]["snapshots"]["v1"]["content_hash"] = "changed"
                else:
                    other["manifest"]["implementation_fingerprint"] = "changed"
                with self.assertRaises(ValueError):
                    checker.compare(self.result, other)

    def test_missing_or_duplicated_run_cannot_pass(self) -> None:
        missing = copy.deepcopy(self.result)
        missing["runs"].pop()
        duplicate = copy.deepcopy(self.result)
        duplicate["runs"][-1] = duplicate["runs"][0]
        for other in (missing, duplicate):
            with self.assertRaises(ValueError):
                checker.compare(self.result, other)


if __name__ == "__main__":
    unittest.main()
