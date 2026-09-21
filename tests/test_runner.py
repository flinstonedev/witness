from __future__ import annotations

import unittest

from witness_bench.report import render_markdown
from witness_bench.runner import BenchmarkConfig, build_systems, run_benchmark


class RunnerTests(unittest.TestCase):
    def test_full_profile_registers_requested_sweeps_and_ablations(self) -> None:
        names = {system.name for system in build_systems("full")}
        for expected in (
            "vector-rag-tfidf-k10",
            "vector-rag-tfidf-k50",
            "vector-rag-tfidf-k100",
            "vector-rag-tfidf-k500",
            "witness-binary-no-uncertain",
            "witness-no-query-view",
            "witness-hybrid-prefilter-k10",
            "witness-hybrid-prefilter-k500",
            "graphrag-local-approx",
            "graphrag-global-approx",
            "long-context-65536",
            "diagnostic-exhaustive-lexical-map-no-cache",
        ):
            self.assertIn(expected, names)

    def test_small_run_is_machine_readable_and_refuses_superiority_claim(self) -> None:
        result = run_benchmark(
            BenchmarkConfig(
                profile="smoke",
                repetitions=1,
                task_ids=("rare_termination_exception",),
                system_names=(
                    "witness-exhaustive-offline-no-counter",
                    "vector-rag-tfidf-k10",
                ),
                bootstrap_resamples=20,
            )
        )
        self.assertEqual(result["manifest"]["claim_status"], "preliminary_offline_proxy_only")
        self.assertFalse(result["manifest"]["production_baselines"])
        self.assertEqual(result["manifest"]["benchmark_version"], "0.2.0")
        self.assertRegex(
            result["manifest"]["implementation_fingerprint"], r"^sha256:[0-9a-f]{64}$"
        )
        self.assertGreater(result["manifest"]["implementation_python_files"], 10)
        self.assertEqual(result["manifest"]["cases"], 2)
        self.assertEqual(len(result["runs"]), 4)
        report = render_markdown(result)
        self.assertIn("No production superiority claim", report)
        self.assertIn("conclusive program-execution coverage", report)
        self.assertIn("Operational distributions", report)
        self.assertIn("First p99", report)

        no_counter = result["summary"]["witness-exhaustive-offline-no-counter"]
        diagnostic = {
            "manifest": {"diagnostic_only": True, "headline_eligible": False},
            "summary": {
                "witness-aster-human-plan-development-oracle": no_counter,
                "witness-exhaustive-offline-no-counter": no_counter,
            },
        }
        diagnostic_report = render_markdown(
            result, multihop_diagnostic=diagnostic
        )
        self.assertIn("Human-plan development oracle (diagnostic only)", diagnostic_report)
        self.assertIn("fair baseline win", diagnostic_report)


if __name__ == "__main__":
    unittest.main()
