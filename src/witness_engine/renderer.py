"""Deterministic rendering of evidence-linked claim ledgers.

This boundary deliberately performs no prose generation and no semantic
entailment judgment.  It copies claim text verbatim from the supplied ledger,
checks that every cited witness identifier exists, and carries the execution
coverage certificate into a serializable result.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterable

from .models import (
    ClaimLedgerEntry,
    ClaimStatus,
    CoverageCertificate,
    JSONValue,
    ValidationError,
    WitnessRow,
)

if TYPE_CHECKING:
    from .engine import ExecutionResult


@dataclass(frozen=True, slots=True)
class RenderedAnswer:
    """A lossless, non-generative projection of a claim ledger.

    ``answer_text`` is computed from ledger entries rather than accepted as an
    independent field, so this type cannot silently introduce prose between
    the ledger and its rendered representation.
    """

    question: str
    claims: tuple[ClaimLedgerEntry, ...]
    coverage: CoverageCertificate
    renderer_version: str = "ledger-renderer-0.1"

    @property
    def asserted_claims(self) -> tuple[ClaimLedgerEntry, ...]:
        """Claims permitted to appear as factual answer prose."""

        return tuple(
            claim
            for claim in self.claims
            if claim.status in {ClaimStatus.SUPPORTED, ClaimStatus.QUALIFIED}
        )

    @property
    def answer_text(self) -> str:
        return "\n".join(claim.text for claim in self.asserted_claims)

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "question": self.question,
            "answer_text": self.answer_text,
            "claims": [claim.to_dict() for claim in self.claims],
            "asserted_claim_ids": [claim.claim_id for claim in self.asserted_claims],
            "non_asserted_claim_ids": [
                claim.claim_id
                for claim in self.claims
                if claim not in self.asserted_claims
            ],
            "coverage": self.coverage.to_dict(),
            "renderer_version": self.renderer_version,
            "validation": {
                "claim_text": "copied_verbatim_from_ledger",
                "citation_ids": "existence_checked",
                "assertion_policy": "supported_or_qualified_with_support_only",
                "semantic_entailment": "not_evaluated",
            },
        }


class LedgerAnswerRenderer:
    """Render only ledger-authored claims with known witness citations."""

    version = "ledger-renderer-0.1"

    def render(
        self,
        *,
        question: str,
        claims: Iterable[ClaimLedgerEntry],
        witnesses: Iterable[WitnessRow],
        coverage: CoverageCertificate,
    ) -> RenderedAnswer:
        claim_rows = tuple(claims)
        known_witness_ids = {row.witness_id for row in witnesses}

        for claim in claim_rows:
            self._validate_citations(claim, known_witness_ids)
            if (
                claim.status in {ClaimStatus.SUPPORTED, ClaimStatus.QUALIFIED}
                and not claim.supporting_witness_ids
            ):
                raise ValidationError(
                    f"claim {claim.claim_id!r} cannot be asserted without a supporting witness"
                )

        return RenderedAnswer(
            question=question,
            claims=claim_rows,
            coverage=coverage,
            renderer_version=self.version,
        )

    def render_execution(self, result: ExecutionResult) -> RenderedAnswer:
        """Render an execution result without exposing unrelated relations."""

        return self.render(
            question=result.program.question,
            claims=result.claims,
            witnesses=(*result.rows, *result.counter_rows),
            coverage=result.coverage,
        )

    @staticmethod
    def _validate_citations(
        claim: ClaimLedgerEntry,
        known_witness_ids: set[str],
    ) -> None:
        support_unknown = sorted(
            set(claim.supporting_witness_ids) - known_witness_ids
        )
        counter_unknown = sorted(set(claim.counter_witness_ids) - known_witness_ids)
        if support_unknown or counter_unknown:
            details: list[str] = []
            if support_unknown:
                details.append(f"unknown supporting witness IDs: {support_unknown}")
            if counter_unknown:
                details.append(f"unknown counter witness IDs: {counter_unknown}")
            raise ValidationError(
                f"claim {claim.claim_id!r} cannot be rendered; " + "; ".join(details)
            )


__all__ = ["LedgerAnswerRenderer", "RenderedAnswer"]
