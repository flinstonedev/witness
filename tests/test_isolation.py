from __future__ import annotations

import unittest

from witness_bench.data import build_benchmark
from witness_bench.isolation import (
    answer_key_is_absent,
    exposed_evidence,
    public_documents,
    public_task,
)
from witness_bench.models import RetrievedPassage


class IsolationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.dataset = build_benchmark()

    def test_public_views_remove_answer_key(self) -> None:
        gold = self.dataset.task("rare_termination_exception")
        task = public_task(gold)
        documents = public_documents(self.dataset.corpus.documents_for("v1"))
        self.assertTrue(answer_key_is_absent(task, documents))
        self.assertIsNone(task.expected_answer)
        self.assertEqual(task.evidence_ids, ())

    def test_unrelated_span_gets_no_same_document_credit(self) -> None:
        ref = self.dataset.corpus.evidence_index()[
            "ev.rare.cygnus.termination-exception"
        ]
        passage = RetrievedPassage(
            passage_id="unrelated",
            document_id=ref.document_id,
            document_version=ref.document_version,
            text="unrelated header",
            start_offset=0,
            end_offset=max(1, ref.start_offset - 1),
        )
        self.assertEqual(exposed_evidence((passage,), (ref,)), ())

    def test_exact_containment_receives_span_credit(self) -> None:
        ref = self.dataset.corpus.evidence_index()[
            "ev.rare.cygnus.termination-exception"
        ]
        passage = RetrievedPassage(
            passage_id="exact",
            document_id=ref.document_id,
            document_version=ref.document_version,
            text=ref.quote,
            start_offset=ref.start_offset,
            end_offset=ref.end_offset,
        )
        self.assertEqual(exposed_evidence((passage,), (ref,)), (ref,))


if __name__ == "__main__":
    unittest.main()

