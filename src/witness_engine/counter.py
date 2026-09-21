"""Counter-witness obligation generation and claim-ledger reconciliation."""

from __future__ import annotations

import re
from typing import Iterable

from .models import (
    ClaimLedgerEntry,
    ClaimStatus,
    CounterTestSpec,
    PredicateSpec,
    WitnessProgram,
    WitnessRow,
    stable_hash,
)


_COUNTER_TERMS = (
    "not",
    "no",
    "except",
    "exception",
    "optional",
    "void",
    "voided",
    "cancelled",
    "canceled",
    "withdrawn",
    "replaced",
    "override",
    "overridden",
    "later",
    "newer",
    "backdated",
    "repealed",
    "superseded",
    "amended",
    "corrected",
    "correction",
    "estimate",
    "proposed",
    "rejected",
    "excluding",
    "however",
)
_WORDS = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{2,}")
_STOP = frozenset("the and for from with that this was were are has have into over under claim".split())


class CounterWitnessGenerator:
    GENERATOR_VERSION = "counter-generator-0.1.0"

    def generate(
        self,
        program: WitnessProgram,
        claims: Iterable[ClaimLedgerEntry] = (),
    ) -> tuple[CounterTestSpec, ...]:
        """Return explicit counter-tests plus conservative generated tests.

        Generated tests search for *potential* falsifiers.  They never declare
        a claim false by themselves; the ledger marks a supported claim as
        qualified until those source spans are resolved.
        """

        result = list(program.counter_tests)
        explicit_targets = {item.target_claim for item in result}
        for claim in claims:
            if claim.claim_id in explicit_targets:
                continue
            anchors: list[str] = []
            for token in _WORDS.findall(claim.text):
                folded = token.casefold()
                if folded not in _STOP and folded not in anchors:
                    anchors.append(folded)
                if len(anchors) == 8:
                    break
            identity = stable_hash([claim.claim_id, claim.text])[:16]
            if not anchors:
                anchors = [claim.text]
            result.append(
                CounterTestSpec(
                    counter_id=f"auto-counter-{identity}",
                    target_claim=claim.claim_id,
                    rationale=(
                        "Search for negations, exceptions, corrections, scope qualifiers, "
                        "or superseding evidence that could materially alter the claim."
                    ),
                    predicate=PredicateSpec(
                        predicate_id=f"counter_{identity}",
                        description=f"Potential falsifier or qualifier for claim: {claim.text}",
                        # One claim anchor plus one counter cue is ideal, but the
                        # lexical fallback cannot express proximity/conjunction
                        # groups. Broad alternatives preserve recall and demand
                        # downstream resolution.
                        any_term_groups=(
                            tuple(dict.fromkeys(anchors)),
                            _COUNTER_TERMS,
                        ),
                        closed_world=False,
                        polarity="potential_counter",
                        modality="source_statement",
                    ),
                )
            )
        return tuple(result)

    @staticmethod
    def reconcile(
        claims: Iterable[ClaimLedgerEntry],
        positive_rows: Iterable[WitnessRow],
        counter_rows_by_test: dict[str, Iterable[WitnessRow]],
        tests: Iterable[CounterTestSpec],
    ) -> tuple[ClaimLedgerEntry, ...]:
        positives = {row.witness_id for row in positive_rows}
        materialized_counters = {
            test_id: tuple(rows) for test_id, rows in counter_rows_by_test.items()
        }
        tests_by_target: dict[str, list[str]] = {}
        for test in tests:
            tests_by_target.setdefault(test.target_claim, []).append(test.counter_id)
        result: list[ClaimLedgerEntry] = []
        for claim in claims:
            support = tuple(item for item in claim.supporting_witness_ids if item in positives)
            counters = tuple(
                sorted(
                    {
                        row.witness_id
                        for test_id in tests_by_target.get(claim.claim_id, [])
                        for row in materialized_counters.get(test_id, ())
                    }
                )
            )
            if counters and support:
                status = ClaimStatus.QUALIFIED
                qualifiers = tuple(dict.fromkeys((*claim.qualifiers, "potential counter-evidence requires resolution")))
            elif counters:
                status = ClaimStatus.UNRESOLVED
                qualifiers = tuple(dict.fromkeys((*claim.qualifiers, "counter-evidence found without verified support")))
            elif support:
                status = ClaimStatus.SUPPORTED
                qualifiers = claim.qualifiers
            else:
                status = ClaimStatus.UNRESOLVED
                qualifiers = claim.qualifiers
            result.append(
                ClaimLedgerEntry(
                    claim_id=claim.claim_id,
                    text=claim.text,
                    supporting_witness_ids=support,
                    counter_witness_ids=counters,
                    status=status,
                    qualifiers=qualifiers,
                )
            )
        return tuple(result)
