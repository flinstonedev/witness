from __future__ import annotations

import unittest

from witness_engine import QuestionCompiler, fallback_lexical_predicate
from witness_bench.data import build_benchmark
from witness_bench.controls import ExhaustiveLexicalMapControl
from witness_bench.isolation import public_documents, public_task
from witness_bench.metrics import evidence_metrics
from witness_bench.models import EvidenceRef, SourceDocument
from witness_bench.witness_system import WitnessOfflineSystem


class WitnessSystemTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.dataset = build_benchmark()
        cls.gold = cls.dataset.task("rare_termination_exception")
        cls.query = public_task(cls.gold)
        cls.documents = public_documents(cls.dataset.corpus.documents_for("v1"))
        cls.catalog = {
            evidence_id: cls.dataset.corpus.evidence_index()[evidence_id]
            for evidence_id in cls.gold.evidence_ids
        }

    def test_exhaustive_adapter_finds_decisive_span_without_gold_input(self) -> None:
        system = WitnessOfflineSystem(counter_search=False)
        result = system.retrieve(self.query, self.documents)
        metrics = evidence_metrics(self.gold, result, self.catalog)
        self.assertEqual(metrics["decisive_evidence_recall"], 1.0)
        self.assertEqual(result.telemetry.authorized_shards, len(self.documents))
        self.assertGreater(result.telemetry.uncertain_shards, 0)

    def test_control_predicate_is_exactly_the_compiler_fallback_predicate(self) -> None:
        compiled = QuestionCompiler().compile(self.query.question)
        self.assertEqual(
            compiled.scans[0].predicate,
            fallback_lexical_predicate(self.query.question),
        )

    def test_repeated_query_reads_materialized_view_without_changing_rows(self) -> None:
        system = WitnessOfflineSystem(counter_search=True)
        cold = system.retrieve(self.query, self.documents)
        warm = system.retrieve(self.query, self.documents)
        self.assertEqual(cold.passages, warm.passages)
        self.assertTrue(cold.telemetry.cold)
        self.assertFalse(warm.telemetry.cold)
        self.assertTrue(warm.metadata["materialized_query_view_hit"])
        self.assertLess(warm.telemetry.input_tokens, cold.telemetry.input_tokens)
        self.assertEqual(warm.telemetry.cache_hits, cold.telemetry.cache_hits)
        self.assertEqual(warm.telemetry.counters["query_result_cache_hits"], 1.0)

    def test_exhaustive_lexical_control_matches_no_counter_witness_coordinates(self) -> None:
        witness = WitnessOfflineSystem(
            counter_search=False,
            materialized_query_views=False,
        ).retrieve(self.query, self.documents)
        control = ExhaustiveLexicalMapControl().retrieve(self.query, self.documents)

        def coordinates(result):  # type: ignore[no-untyped-def]
            return [
                (
                    passage.document_id,
                    passage.document_version,
                    passage.start_offset,
                    passage.end_offset,
                    passage.text,
                )
                for passage in result.passages
            ]

        self.assertEqual(coordinates(control), coordinates(witness))
        self.assertEqual(
            control.telemetry.uncertain_shards,
            witness.telemetry.uncertain_shards,
        )
        self.assertEqual(
            control.telemetry.authorized_shards,
            witness.telemetry.authorized_shards,
        )
        self.assertTrue(control.metadata["diagnostic_only"])
        self.assertFalse(control.metadata["production_equivalent"])

    def test_exhaustive_lexical_control_never_caches_and_strips_gold(self) -> None:
        text = "Contract Kappa contains the moonlight termination exception."
        start = text.index("moonlight")
        leaked = EvidenceRef(
            "gold-secret-id",
            "contract-k",
            quote="moonlight termination exception",
            start_offset=start,
            end_offset=start + len("moonlight termination exception"),
        )
        enriched_document = SourceDocument(
            "contract-k",
            content=text,
            evidence_spans=(leaked,),
        )
        enriched_task = type(self.gold)(
            task_id="leak-check",
            category="rare_detail",
            question="Which contract has the moonlight termination exception?",
            expected_answer="DO-NOT-EXPOSE-THIS-ANSWER",
            evidence_ids=("gold-secret-id",),
            decisive_evidence_ids=("gold-secret-id",),
            counter_evidence_ids=("counter-secret-id",),
        )
        system = ExhaustiveLexicalMapControl()
        first = system.retrieve(enriched_task, (enriched_document,))
        second = system.retrieve(enriched_task, (enriched_document,))
        self.assertEqual(first.passages, second.passages)
        self.assertTrue(first.telemetry.cold)
        self.assertTrue(second.telemetry.cold)
        self.assertEqual(first.telemetry.cache_hits, 0)
        self.assertEqual(second.telemetry.cache_hits, 0)
        self.assertEqual(
            first.telemetry.counters["evaluator_calls"],
            first.telemetry.authorized_shards,
        )
        self.assertEqual(first.telemetry.input_tokens, second.telemetry.input_tokens)
        self.assertNotIn("DO-NOT-EXPOSE", repr(first.metadata))
        self.assertNotIn("gold-secret-id", repr(first.metadata))
        for passage in first.passages:
            self.assertEqual(
                enriched_document.content[passage.start_offset : passage.end_offset],
                passage.text,
            )


if __name__ == "__main__":
    unittest.main()
