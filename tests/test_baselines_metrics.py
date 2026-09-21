from __future__ import annotations

from dataclasses import replace
import unittest

from witness_bench.baselines import (
    AgenticRAGBaseline,
    GraphRAGBaseline,
    HybridRAGBaseline,
    LLMWikiBaseline,
    LongContextBaseline,
    RerankedRAGBaseline,
    VectorRAGBaseline,
    default_baselines,
)
from witness_bench.metrics import (
    EvaluationSuite,
    aggregation_accuracy,
    answer_correctness,
    cost_metrics,
    coverage_metrics,
    evidence_metrics,
    incremental_update_correctness,
    negative_claim_calibration,
    reproducibility_metrics,
    temporal_accuracy,
    unsupported_claim_rate,
)
from witness_bench.models import (
    BenchmarkTask,
    Claim,
    EvidenceRef,
    RetrievedPassage,
    RetrievalResult,
    RetrievalResult,
    RunTelemetry,
    SourceDocument,
    SystemAnswer,
    coerce_source_document,
    coerce_task,
)


def evidence(document_id: str, text: str, quote: str, evidence_id: str, *, kind: str = "fact") -> EvidenceRef:
    start = text.index(quote)
    return EvidenceRef(
        evidence_id=evidence_id,
        document_id=document_id,
        quote=quote,
        start_offset=start,
        end_offset=start + len(quote),
        kind=kind,
    )


class ModelTests(unittest.TestCase):
    def test_canonical_document_aliases_and_mapping_coercion(self) -> None:
        document = coerce_source_document(
            {
                "source_id": "policy-1",
                "document_version": 3,
                "raw_content": "Approval was optional.",
                "timestamps": {"valid_from": "2025-01-01"},
                "permissions": ["benchmark"],
            }
        )
        self.assertEqual(document.document_id, "policy-1")
        self.assertEqual(document.version, "3")
        self.assertEqual(document.text, document.raw_content)
        self.assertEqual(len(document.content_hash), 64)
        self.assertEqual(document.valid_from, "2025-01-01")

        task = coerce_task(
            {
                "id": "q1",
                "task_type": "lookup",
                "query": "Was approval optional?",
                "gold_evidence_ids": ["ev-1"],
            }
        )
        self.assertEqual(task.task_id, "q1")
        self.assertEqual(task.evidence_ids, ("ev-1",))
        freeform = coerce_task("What changed?")
        self.assertEqual(freeform.question, "What changed?")
        self.assertTrue(freeform.task_id.startswith("query:"))


class BaselineTests(unittest.TestCase):
    def setUp(self) -> None:
        obscure = "Contract Kappa contains the moonlight termination exception."
        self.documents = (
            SourceDocument("noise-1", content="Routine renewal terms apply to ordinary contracts."),
            SourceDocument("contract-k", content=obscure, metadata={"synthetic": True}),
            SourceDocument("noise-2", content="The support handbook discusses response times."),
        )
        self.gold = evidence(
            "contract-k",
            obscure,
            "the moonlight termination exception",
            "ev-rare",
        )
        self.task = BenchmarkTask(
            "rare",
            "rare_detail",
            "Which contract has the moonlight termination exception?",
            expected_answer="Contract Kappa",
            evidence_ids=("ev-rare",),
            decisive_evidence_ids=("ev-rare",),
        )

    def test_vector_retrieval_and_scorer_side_gold_overlap(self) -> None:
        # Documents intentionally carry no gold evidence spans.
        self.assertFalse(self.documents[1].evidence_spans)
        result = VectorRAGBaseline(top_k=1).retrieve(self.task, self.documents)
        self.assertEqual(result.passages[0].document_id, "contract-k")
        scores = evidence_metrics(self.task, result, [self.gold])
        self.assertEqual(scores["evidence_recall"], 1.0)
        self.assertEqual(scores["decisive_evidence_recall"], 1.0)
        self.assertEqual(scores["evidence_precision"], 1.0)

    def test_scorer_gold_attachment_does_not_invalidate_exact_source_claim(self) -> None:
        baseline = VectorRAGBaseline(top_k=1)
        answer = baseline.run(self.task, self.documents)
        # The isolation layer may replace synthetic citation rows with exposed
        # scorer-gold spans.  Claim support is grounded by source coordinates,
        # not by requiring annotation IDs to match generated witness IDs.
        scored = replace(answer, evidence=(self.gold,))
        self.assertEqual(
            unsupported_claim_rate(scored, corpus=self.documents),
            0.0,
        )

    def test_hybrid_and_reranked_are_deterministic_and_explicitly_approximate(self) -> None:
        for baseline in (
            HybridRAGBaseline(top_k=2),
            RerankedRAGBaseline(top_k=2, candidate_k=3),
        ):
            first = baseline.retrieve(self.task, self.documents)
            second = baseline.retrieve(self.task, self.documents)
            self.assertEqual(
                [item.passage_id for item in first.passages],
                [item.passage_id for item in second.passages],
            )
            self.assertFalse(first.metadata["production_equivalent"])
            self.assertTrue(first.metadata["limitations"])
            self.assertTrue(first.telemetry.cold)
            self.assertFalse(second.telemetry.cold)

    def test_graph_local_expands_entity_chain(self) -> None:
        docs = (
            SourceDocument("design", content="Product Zephyr was designed by Ada Vale."),
            SourceDocument("founding", content="Ada Vale founded Lumen Works."),
            SourceDocument("acquisition", content="Northstar acquired Lumen Works."),
            SourceDocument("noise", content="Quarterly catering menus were revised."),
        )
        task = BenchmarkTask(
            "hop",
            "multi_hop",
            "Who acquired the organization founded by the designer of Product Zephyr?",
        )
        result = GraphRAGBaseline(
            mode="local", top_k=3, seed_k=1, graph_hops=2
        ).retrieve(task, docs)
        self.assertEqual(
            {item.document_id for item in result.passages},
            {"design", "founding", "acquisition"},
        )

    def test_long_context_obeys_budget_and_agent_obeys_call_budget(self) -> None:
        long_context = LongContextBaseline(context_token_budget=8, top_k=10, chunk_tokens=20)
        result = long_context.retrieve(self.task, self.documents)
        self.assertLessEqual(
            int(result.telemetry.counters["retrieved_context_tokens"]), 8
        )

        agent = AgenticRAGBaseline(
            top_k=3, tool_call_budget=2, results_per_call=1
        )
        agent_result = agent.retrieve(self.task, self.documents)
        self.assertLessEqual(agent_result.telemetry.tool_calls, 2)
        self.assertEqual(
            agent_result.telemetry.tool_calls,
            int(agent_result.telemetry.counters["agent_iterations"]),
        )

    def test_wiki_compression_can_drop_detail_and_raw_fallback_recovers_it(self) -> None:
        content = (
            "Contract Kappa has conventional renewal language. "
            "The moonlight termination exception applies after an eclipse."
        )
        docs = (SourceDocument("contract-k", content=content, metadata={"wiki_topic": "Kappa"}),)
        gold = evidence(
            "contract-k",
            content,
            "The moonlight termination exception applies after an eclipse.",
            "ev-hidden",
        )
        task = BenchmarkTask(
            "wiki-loss",
            "rare_detail",
            "What moonlight termination exception applies after an eclipse?",
            evidence_ids=("ev-hidden",),
        )
        wiki_only = LLMWikiBaseline(
            top_k=1, sentences_per_source=1, chunk_tokens=100
        ).retrieve(task, docs)
        fallback = LLMWikiBaseline(
            top_k=2,
            sentences_per_source=1,
            raw_fallback=True,
            raw_fallback_k=1,
            chunk_tokens=100,
        ).retrieve(task, docs)
        self.assertEqual(evidence_metrics(task, wiki_only, [gold])["evidence_recall"], 0.0)
        self.assertEqual(evidence_metrics(task, fallback, [gold])["evidence_recall"], 1.0)
        self.assertTrue(any(item.document_id == "contract-k" for item in fallback.passages))
        self.assertLessEqual(
            fallback.telemetry.scanned_shards,
            fallback.telemetry.authorized_shards,
        )

    def test_as_of_filter_excludes_future_version(self) -> None:
        docs = (
            SourceDocument(
                "old",
                content="Manager approval was optional under the historical policy.",
                valid_from="2024-01-01",
                valid_to="2025-01-01",
            ),
            SourceDocument(
                "new",
                content="Manager approval is mandatory under the current policy.",
                valid_from="2025-01-01",
            ),
        )
        task = BenchmarkTask(
            "time", "temporal", "What approval applied?", as_of="2024-03-14"
        )
        result = VectorRAGBaseline(top_k=2).retrieve(task, docs)
        self.assertEqual([item.document_id for item in result.passages], ["old"])

    def test_default_factory_covers_all_requested_families(self) -> None:
        systems = default_baselines(top_k=2)
        families = {item.family for item in systems}
        self.assertTrue(
            {
                "classic_vector_rag",
                "hybrid_bm25_vector_rag",
                "reranked_rag",
                "graphrag",
                "long_context",
                "agentic_rag",
                "llm_maintained_wiki",
            }.issubset(families)
        )


class MetricTests(unittest.TestCase):
    def setUp(self) -> None:
        self.text = "Revenue was 12M. A correction states final revenue was 14M."
        self.ev_old = evidence("report", self.text, "Revenue was 12M.", "ev-old", kind="counter")
        self.ev_final = evidence(
            "report", self.text, "final revenue was 14M", "ev-final", kind="decisive"
        )
        self.task = BenchmarkTask(
            "revenue",
            "contradiction_aggregation",
            "What was final revenue?",
            expected_answer={"revenue_m": 14},
            evidence_ids=("ev-old", "ev-final"),
            decisive_evidence_ids=("ev-final",),
            counter_evidence_ids=("ev-old",),
            metadata={"aggregation_expected_answer": {"revenue_m": 14}},
        )
        self.passage = RetrievedPassage(
            "report@1:0-62",
            "report",
            "1",
            self.text,
            0,
            len(self.text),
        )
        self.answer = SystemAnswer(
            "test",
            "revenue",
            answer={"revenue_m": 14},
            evidence=(self.ev_final, self.ev_old),
            retrieved_passages=(self.passage,),
            claims=(Claim("c1", "Final revenue was 14M.", ("ev-final",)),),
            telemetry=RunTelemetry(
                authorized_shards=4,
                scanned_shards=1,
                soundly_skipped_shards=2,
                unresolved_shards=1,
                latency_ms=10,
            ),
        )

    def test_all_evidence_metrics(self) -> None:
        metrics = evidence_metrics(self.task, self.answer, [self.ev_old, self.ev_final])
        self.assertEqual(metrics["evidence_recall"], 1.0)
        self.assertEqual(metrics["decisive_evidence_recall"], 1.0)
        self.assertEqual(metrics["counter_evidence_recall"], 1.0)
        self.assertEqual(metrics["weighted_evidence_recall"], 1.0)
        self.assertEqual(metrics["precision_unit"], "passage")

    def test_precision_ignores_other_tasks_gold_and_forged_id_gets_no_credit(self) -> None:
        unrelated_task = BenchmarkTask(
            "other", "lookup", "Find another fact", evidence_ids=("ev-other",)
        )
        # The passage contains annotations for the revenue task, but none for
        # this task. A global catalog must not turn that into precision credit.
        metrics = evidence_metrics(
            unrelated_task, self.answer, [self.ev_old, self.ev_final]
        )
        self.assertEqual(metrics["evidence_precision"], 0.0)

        forged = SystemAnswer(
            "attacker",
            "revenue",
            evidence=(
                EvidenceRef(
                    "ev-final",
                    "wrong-document",
                    quote=self.ev_final.quote,
                    start_offset=self.ev_final.start_offset,
                    end_offset=self.ev_final.end_offset,
                ),
            ),
        )
        forged_metrics = evidence_metrics(
            self.task, forged, [self.ev_old, self.ev_final]
        )
        self.assertEqual(forged_metrics["decisive_evidence_recall"], 0.0)

        forged_passage = RetrievalResult(
            system_name="attacker",
            task_id="revenue",
            passages=(
                RetrievedPassage(
                    passage_id="forged-passage",
                    document_id=self.ev_final.document_id,
                    document_version=self.ev_final.document_version,
                    text="different bytes despite plausible coordinates",
                    start_offset=self.ev_final.start_offset,
                    end_offset=self.ev_final.end_offset,
                ),
            ),
        )
        self.assertEqual(
            evidence_metrics(
                self.task, forged_passage, [self.ev_old, self.ev_final]
            )["decisive_evidence_recall"],
            0.0,
        )

    def test_uncertain_scans_are_not_conclusive_coverage(self) -> None:
        answer = SystemAnswer(
            "witness",
            "negative",
            telemetry=RunTelemetry(
                authorized_shards=4,
                scanned_shards=3,
                soundly_skipped_shards=1,
                uncertain_shards=2,
            ),
        )
        coverage = coverage_metrics(answer)
        self.assertEqual(coverage["coverage"], 0.5)
        self.assertEqual(coverage["uncertain_shards"], 2)
        self.assertFalse(coverage["fully_covered"])

    def test_answer_specialized_and_unsupported_claim_metrics(self) -> None:
        self.assertEqual(answer_correctness(self.task, self.answer), 1.0)
        self.assertEqual(aggregation_accuracy(self.task, self.answer), 1.0)
        document = SourceDocument("report", content=self.text)
        self.assertEqual(unsupported_claim_rate(self.answer, corpus=[document]), 0.0)
        unsupported = replace(
            self.answer,
            claims=(Claim("bad", "Revenue was 99M.", ("missing",)),),
        )
        self.assertEqual(unsupported_claim_rate(unsupported, corpus=[document]), 1.0)

    def test_temporal_and_negative_metrics(self) -> None:
        temporal_task = BenchmarkTask(
            "t",
            "temporal",
            "What applied?",
            expected_answer="optional",
            metadata={"temporal_expected_answer": "optional"},
        )
        temporal_answer = SystemAnswer("x", "t", answer=" OPTIONAL ")
        self.assertEqual(temporal_accuracy(temporal_task, temporal_answer), 1.0)

        negative_task = BenchmarkTask(
            "n",
            "negative_claim",
            "Did any incident occur?",
            metadata={"negative_claim_status": "not_found"},
        )
        calibrated = SystemAnswer(
            "x", "n", answer={"status": "no evidence found"}
        )
        overclaim = SystemAnswer("x", "n", answer={"status": "does not exist"})
        self.assertEqual(negative_claim_calibration(negative_task, calibrated), 1.0)
        self.assertEqual(negative_claim_calibration(negative_task, overclaim), 0.0)

    def test_reproducibility_update_and_cost(self) -> None:
        same = replace(self.answer, telemetry=replace(self.answer.telemetry, latency_ms=20))
        reproducibility = reproducibility_metrics(
            [self.answer, same], [self.ev_old, self.ev_final]
        )
        self.assertEqual(reproducibility["reproducibility"], 1.0)

        updated_task = replace(
            self.task,
            expected_answer={"revenue_m": 15},
            evidence_ids=("ev-new",),
            decisive_evidence_ids=("ev-new",),
            counter_evidence_ids=(),
        )
        new_text = "A later correction states final revenue was 15M."
        ev_new = evidence("new-report", new_text, "final revenue was 15M", "ev-new")
        new_answer = SystemAnswer(
            "test",
            "revenue",
            answer={"revenue_m": 15},
            evidence=(ev_new,),
        )
        update = incremental_update_correctness(
            self.task,
            self.answer,
            updated_task,
            new_answer,
            before_evidence_catalog=[self.ev_old, self.ev_final],
            after_evidence_catalog=[ev_new],
        )
        self.assertEqual(update["post_update_answer_correctness"], 1.0)
        self.assertEqual(update["stale_evidence_rate"], 0.0)
        self.assertEqual(update["update_correctness"], 1.0)

        stale_answer = replace(new_answer, evidence=(ev_new, self.ev_old))
        stale_update = incremental_update_correctness(
            self.task,
            self.answer,
            updated_task,
            stale_answer,
            before_evidence_catalog=[self.ev_old, self.ev_final],
            after_evidence_catalog=[ev_new],
        )
        self.assertGreater(stale_update["stale_evidence_rate"], 0.0)
        self.assertLess(stale_update["update_correctness"], 1.0)

        cold = RunTelemetry(cold=True, input_tokens=100, model_calls=1, latency_ms=10)
        warm = RunTelemetry(cold=False, input_tokens=10, cache_hits=1, latency_ms=20)
        costs = cost_metrics([cold, warm])
        self.assertEqual(costs["input_tokens"], 110)
        self.assertEqual(costs["cold"]["runs"], 1)
        self.assertEqual(costs["warm"]["runs"], 1)
        self.assertEqual(costs["latency_ms"]["p50"], 15.0)
        self.assertEqual(costs["latency_ms"]["p95"], 19.5)

    def test_evaluation_suite_aggregates(self) -> None:
        suite = EvaluationSuite()
        report = suite.add(
            self.task,
            self.answer,
            evidence_catalog=[self.ev_old, self.ev_final],
            corpus=[SourceDocument("report", content=self.text)],
        )
        self.assertEqual(report["coverage"], 0.75)
        summary = suite.summary()
        self.assertEqual(summary["quality"]["runs"], 1)
        self.assertEqual(summary["cost"]["runs"], 1)


if __name__ == "__main__":
    unittest.main()
