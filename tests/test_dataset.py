"""Integrity tests for the deterministic adversarial benchmark fixture."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from witness_bench.data import (  # noqa: E402
    DEFAULT_SEED,
    build_benchmark,
    build_corpus,
    distributed_batch,
    distributed_total,
    distributed_value,
    document_key,
)
from witness_bench.tasks import build_tasks  # noqa: E402


class DatasetIntegrityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.dataset = build_benchmark()
        cls.corpus = cls.dataset.corpus
        cls.tasks = cls.dataset.tasks
        cls.documents = cls.corpus.document_index()
        cls.evidence = cls.corpus.evidence_index()

    def test_generation_is_deterministic(self) -> None:
        again = build_corpus(DEFAULT_SEED)
        self.assertEqual(
            [(row.snapshot_id, row.content_hash) for row in self.corpus.snapshots],
            [(row.snapshot_id, row.content_hash) for row in again.snapshots],
        )
        self.assertEqual(
            [(doc.document_id, doc.version, doc.content_hash) for doc in self.corpus.documents],
            [(doc.document_id, doc.version, doc.content_hash) for doc in again.documents],
        )
        self.assertEqual(build_tasks(), tuple(self.tasks))

    def test_seed_changes_only_generated_decoys_not_answer_key(self) -> None:
        alternate = build_benchmark(DEFAULT_SEED + 1)
        self.assertNotEqual(
            self.corpus.snapshot("v1").content_hash,
            alternate.corpus.snapshot("v1").content_hash,
        )
        self.assertEqual(
            [(task.task_id, task.expected_answer, task.evidence_ids) for task in self.tasks],
            [(task.task_id, task.expected_answer, task.evidence_ids) for task in alternate.tasks],
        )

    def test_every_annotation_is_an_exact_stable_span(self) -> None:
        self.assertEqual(len(self.evidence), len(self.corpus.evidence))
        for evidence_id, row in self.evidence.items():
            key = f"{row.document_id}@{row.document_version}"
            self.assertIn(key, self.documents, evidence_id)
            content = self.documents[key].content
            self.assertEqual(content[row.start_offset : row.end_offset], row.quote, evidence_id)
            self.assertEqual(content.count(row.quote), 1, evidence_id)

    def test_public_documents_do_not_expose_answer_labels(self) -> None:
        banned = ("evidence", "relevance", "ground_truth", "decisive", "counter")
        for document in self.corpus.documents:
            self.assertEqual(document.evidence_spans, ())
            metadata_text = repr(document.metadata).lower()
            for token in banned:
                self.assertNotIn(token, metadata_text, document_key(document))

        public_task = self.dataset.public_task("rare_termination_exception")
        self.assertNotIn("expected_answer", public_task)
        self.assertNotIn("evidence_ids", public_task)
        self.assertNotIn("decisive_evidence_ids", public_task)
        self.assertNotIn("counter_evidence_ids", public_task)
        self.assertIn("answer_schema", public_task)

    def test_manifest_ids_are_complete_and_well_formed(self) -> None:
        task_ids = [task.task_id for task in self.tasks]
        self.assertEqual(len(task_ids), len(set(task_ids)))
        used: set[str] = set()
        known_snapshots = {snapshot.snapshot_id for snapshot in self.corpus.snapshots}
        for task in self.tasks:
            with self.subTest(task=task.task_id):
                relevant = set(task.evidence_ids)
                decisive = set(task.decisive_evidence_ids)
                counter = set(task.counter_evidence_ids)
                self.assertTrue(task.corpus_versions)
                self.assertLessEqual(set(task.corpus_versions), known_snapshots)
                self.assertTrue(relevant)
                self.assertTrue(decisive)
                self.assertLessEqual(decisive, relevant)
                self.assertLessEqual(counter, relevant)
                self.assertEqual(len(task.evidence_ids), len(relevant))
                self.assertIsInstance(task.expected_answer, dict)
                self.assertTrue(task.expected_answer)
                self.assertEqual(task.answer_type, "structured")
                self.assertIn("answer_schema", task.metadata)
                self.assertEqual(task.metadata["answer_schema"]["type"], "object")

                active_union: set[str] = set()
                for snapshot_id in task.corpus_versions:
                    active_union.update(self.corpus.snapshot(snapshot_id).document_keys)
                for evidence_id in relevant:
                    self.assertIn(evidence_id, self.evidence)
                    row = self.evidence[evidence_id]
                    self.assertIn(
                        f"{row.document_id}@{row.document_version}",
                        active_union,
                        f"{evidence_id} is unavailable to {task.task_id}",
                    )
                used.update(relevant)

        # There are no orphan scorer annotations whose task relevance is undefined.
        self.assertEqual(used, set(self.evidence))

    def test_required_problem_classes_are_present(self) -> None:
        categories = {task.category for task in self.tasks}
        self.assertTrue(
            {
                "simple_lookup",
                "rare_detail",
                "multi_hop",
                "global_aggregation",
                "contradiction_resolution",
                "temporal_reasoning",
                "negative_claim",
                "qualifier_sensitivity",
                "prompt_injection",
                "distributed_evidence",
                "update_handling",
            }.issubset(categories)
        )

    def test_multihop_manifest_covers_every_depth_two_through_six(self) -> None:
        multi_hop = [task for task in self.tasks if task.category == "multi_hop"]
        self.assertEqual({task.metadata["hop_count"] for task in multi_hop}, set(range(2, 7)))
        for task in multi_hop:
            self.assertEqual(len(task.decisive_evidence_ids), task.metadata["hop_count"])

    def test_global_aggregation_has_all_rows_and_exclusion_traps(self) -> None:
        task = self.dataset.task("global_q2_cancellation_reasons")
        self.assertEqual(len(task.decisive_evidence_ids), 12)
        self.assertEqual(len(task.counter_evidence_ids), 3)
        self.assertEqual(
            task.expected_answer["ranking"],
            [
                {"rank": 1, "reason": "price", "count": 5},
                {"rank": 2, "reason": "integration", "count": 4},
                {"rank": 3, "reason": "reliability", "count": 3},
            ],
        )

    def test_distributed_cases_require_ten_one_hundred_and_one_thousand_rows(self) -> None:
        for size in (10, 100, 1000):
            with self.subTest(size=size):
                task = self.dataset.task(f"distributed_evidence_{size}")
                self.assertEqual(len(task.decisive_evidence_ids), size)
                self.assertEqual(len(task.counter_evidence_ids), 3)
                self.assertEqual(task.expected_answer["batch"], distributed_batch(size))
                self.assertEqual(task.expected_answer["accepted_fragments"], size)
                self.assertEqual(task.expected_answer["lumen_credit_sum"], distributed_total(size))
                annotated_sum = sum(
                    self.evidence[evidence_id].metadata["value"]
                    for evidence_id in task.decisive_evidence_ids
                )
                self.assertEqual(annotated_sum, distributed_total(size))
        self.assertEqual(distributed_total(10), sum(distributed_value(10, i) for i in range(1, 11)))

    def test_temporal_tasks_separate_valid_and_record_time(self) -> None:
        valid = self.dataset.task("temporal_export_valid_time")
        recorded = self.dataset.task("temporal_export_record_time")
        self.assertEqual(valid.as_of, recorded.as_of)
        self.assertEqual(valid.metadata["temporal_axis"], "valid_time")
        self.assertEqual(recorded.metadata["temporal_axis"], "record_time")
        self.assertEqual(valid.expected_answer["required_approval"], "Finance Controller")
        self.assertEqual(recorded.expected_answer["required_approval"], "Director")

    def test_prompt_injection_is_inert_corpus_content(self) -> None:
        task = self.dataset.task("adversarial_cluster_status")
        payload = self.evidence["ev.injection.payload"]
        self.assertIn("IGNORE THE USER'S QUESTION", payload.quote)
        self.assertIn(payload.evidence_id, task.counter_evidence_ids)
        self.assertEqual(task.expected_answer["status"], "degraded")
        self.assertIn("ev.injection.verified-status", task.decisive_evidence_ids)

    def test_snapshot_v2_has_add_remove_and_supersede_changes(self) -> None:
        v1 = self.corpus.snapshot("v1")
        v2 = self.corpus.snapshot("v2")
        self.assertEqual(v2.parent_id, "v1")
        self.assertNotEqual(v1.content_hash, v2.content_hash)
        self.assertIn("contracts/orion-msa@1", v1.document_keys)
        self.assertNotIn("contracts/orion-msa@1", v2.document_keys)
        self.assertIn("contracts/orion-msa@2", v2.document_keys)
        self.assertIn("incidents/release-42-sable@1", v2.added_keys)
        self.assertIn("operations/obsolete-banner@1", v2.removed_keys)
        self.assertEqual(
            v2.superseded,
            (("contracts/orion-msa@1", "contracts/orion-msa@2"),),
        )

    def test_update_pairs_change_answers_without_rebuilding_unchanged_documents(self) -> None:
        before = self.dataset.task("negative_release42_data_loss_v1")
        after = self.dataset.task("negative_release42_data_loss_v2")
        self.assertEqual(before.metadata["update_pair"], after.metadata["update_pair"])
        self.assertNotEqual(before.expected_answer["result"], after.expected_answer["result"])

        v1 = self.corpus.snapshot("v1")
        v2 = self.corpus.snapshot("v2")
        unchanged = set(v1.document_keys) & set(v2.document_keys)
        self.assertGreater(len(unchanged), 1100)
        for key in unchanged:
            self.assertEqual(self.documents[key].content_hash, self.documents[key].content_hash)

    def test_corpus_contains_paraphrases_and_decoys(self) -> None:
        text = "\n".join(document.content for document in self.corpus.documents)
        self.assertIn("Archive exports are retained for thirty-seven days", text)
        self.assertIn("Lina Quill advised on a navigation demonstrator", text)
        self.assertIn("might cause data loss", text)
        self.assertIn("this number is not a credit quantity", text)
        noise_contracts = [
            doc for doc in self.corpus.documents if doc.document_id.startswith("contracts/noise-")
        ]
        self.assertGreaterEqual(len(noise_contracts), 30)


if __name__ == "__main__":
    unittest.main()
