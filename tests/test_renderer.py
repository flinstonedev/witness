from __future__ import annotations

import json
import unittest

from witness_engine import (
    ClaimLedgerEntry,
    ClaimStatus,
    CoverageCertificate,
    EvidenceRole,
    LedgerAnswerRenderer,
    ValidationError,
    WitnessRow,
)


def _row(witness_id: str, role: EvidenceRole = EvidenceRole.POSITIVE) -> WitnessRow:
    quote = "Manager approval is required."
    return WitnessRow(
        witness_id=witness_id,
        predicate_id="approval",
        source_id="policy",
        source_version="1",
        source_hash="sha256:policy",
        block_id="policy:1:0",
        start_offset=0,
        end_offset=len(quote),
        quote=quote,
        evaluator_version="test-evaluator",
        role=role,
    )


def _coverage() -> CoverageCertificate:
    return CoverageCertificate(
        corpus_snapshot="sha256:snapshot",
        authorized_shards=2,
        scanned_shards=2,
        soundly_skipped_shards=0,
        unresolved_shards=0,
        uncertain_shards=0,
        positive_witnesses=1,
        counter_witnesses=1,
        program_version="wql-test",
        program_hash="sha256:program",
        evaluator_version="test-evaluator",
        verifier_version="test-verifier",
    )


class LedgerAnswerRendererTests(unittest.TestCase):
    def test_projects_ledger_text_citations_and_coverage_without_generation(self):
        text = "Manager approval is required; the exception remains unresolved."
        claim = ClaimLedgerEntry(
            claim_id="approval-claim",
            text=text,
            supporting_witness_ids=("support-1",),
            counter_witness_ids=("counter-1",),
            status=ClaimStatus.QUALIFIED,
            qualifiers=("exception remains unresolved",),
        )
        coverage = _coverage()

        rendered = LedgerAnswerRenderer().render(
            question="Is manager approval required?",
            claims=(claim,),
            witnesses=(
                _row("support-1"),
                _row("counter-1", EvidenceRole.COUNTER),
            ),
            coverage=coverage,
        )

        self.assertEqual(rendered.answer_text, text)
        self.assertIs(rendered.claims[0], claim)
        self.assertIs(rendered.coverage, coverage)
        payload = rendered.to_dict()
        self.assertEqual(payload["claims"][0]["supporting_witness_ids"], ["support-1"])
        self.assertEqual(payload["claims"][0]["counter_witness_ids"], ["counter-1"])
        self.assertEqual(payload["coverage"], coverage.to_dict())
        self.assertEqual(payload["validation"]["semantic_entailment"], "not_evaluated")
        json.dumps(payload, allow_nan=False)

    def test_preserves_claim_order_and_adds_no_connective_prose(self):
        claims = (
            ClaimLedgerEntry(
                "c1",
                "First ledger sentence.",
                supporting_witness_ids=("w1",),
                status=ClaimStatus.SUPPORTED,
            ),
            ClaimLedgerEntry(
                "c2",
                "Second ledger sentence.",
                supporting_witness_ids=("w2",),
                status=ClaimStatus.SUPPORTED,
            ),
        )

        rendered = LedgerAnswerRenderer().render(
            question="What happened?",
            claims=claims,
            witnesses=(_row("w1"), _row("w2")),
            coverage=_coverage(),
        )

        self.assertEqual(
            rendered.answer_text,
            "First ledger sentence.\nSecond ledger sentence.",
        )

    def test_unresolved_claim_remains_in_ledger_but_is_not_asserted(self):
        unresolved = ClaimLedgerEntry("c1", "An unverified proposition.")

        rendered = LedgerAnswerRenderer().render(
            question="What happened?",
            claims=(unresolved,),
            witnesses=(),
            coverage=_coverage(),
        )

        self.assertEqual(rendered.answer_text, "")
        payload = rendered.to_dict()
        self.assertEqual(payload["claims"][0]["text"], unresolved.text)
        self.assertEqual(payload["asserted_claim_ids"], [])
        self.assertEqual(payload["non_asserted_claim_ids"], ["c1"])

    def test_supported_claim_requires_a_supporting_witness(self):
        unsupported = ClaimLedgerEntry(
            "c1", "Unsupported assertion.", status=ClaimStatus.SUPPORTED
        )

        with self.assertRaisesRegex(ValidationError, "without a supporting witness"):
            LedgerAnswerRenderer().render(
                question="What happened?",
                claims=(unsupported,),
                witnesses=(),
                coverage=_coverage(),
            )

    def test_rejects_unknown_support_and_counter_citation_ids(self):
        renderer = LedgerAnswerRenderer()
        for field, unknown_id in (
            ("supporting_witness_ids", "missing-support"),
            ("counter_witness_ids", "missing-counter"),
        ):
            with self.subTest(field=field):
                kwargs = {field: (unknown_id,)}
                claim = ClaimLedgerEntry("c1", "Ledger text.", **kwargs)
                with self.assertRaisesRegex(ValidationError, unknown_id):
                    renderer.render(
                        question="Question?",
                        claims=(claim,),
                        witnesses=(_row("known"),),
                        coverage=_coverage(),
                    )


if __name__ == "__main__":
    unittest.main()
