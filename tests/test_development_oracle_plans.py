from __future__ import annotations

import unittest

from witness_bench.data import build_benchmark
from witness_bench.isolation import (
    answer_key_is_absent,
    public_documents,
    public_task,
)
from witness_bench.metrics import evidence_metrics
from witness_bench.plans import (
    AsterPrismDevelopmentOracleCompiler,
    AsterPrismDevelopmentOracleSystem,
    aster_prism_depth,
)


class AsterPrismDevelopmentOracleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.dataset = build_benchmark()
        cls.tasks = tuple(
            task for task in cls.dataset.tasks if task.category == "multi_hop"
        )
        cls.evidence = cls.dataset.corpus.evidence_index()

    def test_compiler_emits_explicit_dependency_chain(self) -> None:
        question = next(
            task.question for task in self.tasks if task.metadata["hop_count"] == 6
        )
        program = AsterPrismDevelopmentOracleCompiler().compile(question)

        self.assertEqual(aster_prism_depth(question), 6)
        self.assertEqual(len(program.scans), 6)
        self.assertEqual(len(program.dependencies), 5)
        self.assertEqual(program.answer_contract["final_binding"], "owner")
        self.assertTrue(program.answer_contract["development_oracle"])
        self.assertFalse(program.answer_contract["headline_eligible"])
        self.assertEqual(
            [dependency.depends_on for dependency in program.dependencies],
            [scan.scan_id for scan in program.scans[:-1]],
        )

    def test_unsupported_question_uses_conservative_fallback(self) -> None:
        program = AsterPrismDevelopmentOracleCompiler().compile(
            "Which contract contains the exception?"
        )
        self.assertEqual(len(program.scans), 1)
        self.assertNotIn("development_oracle", program.answer_contract)

    def test_all_hop_depths_recover_every_decisive_span_without_answer_key(self) -> None:
        # One shared system also exercises demand-view and per-shard caching
        # across the nested programs.  Only public task/doc copies are passed.
        system = AsterPrismDevelopmentOracleSystem(max_workers=8)
        for task in self.tasks:
            with self.subTest(task=task.task_id):
                query = public_task(task)
                documents = public_documents(
                    self.dataset.corpus.documents_for(task.corpus_versions[0])
                )
                self.assertTrue(answer_key_is_absent(query, documents))
                result = system.retrieve(query, documents)
                catalog = {
                    evidence_id: self.evidence[evidence_id]
                    for evidence_id in task.evidence_ids
                }
                metrics = evidence_metrics(task, result, catalog)
                self.assertEqual(metrics["evidence_recall"], 1.0)
                self.assertEqual(metrics["decisive_evidence_recall"], 1.0)
                self.assertEqual(metrics["evidence_precision"], 1.0)
                self.assertEqual(len(result.passages), task.metadata["hop_count"])
                self.assertFalse(result.metadata["headline_eligible"])
                self.assertGreater(
                    result.telemetry.counters["dependency_expansions"], 0
                )

                final_binding = result.metadata["program"]["answer_contract"][
                    "final_binding"
                ]
                final_relation = result.metadata["final_relation"]
                self.assertEqual(len(final_relation), 1)
                self.assertEqual(
                    final_relation[0][final_binding],
                    task.expected_answer["entity"],
                )


if __name__ == "__main__":
    unittest.main()
