"""Local evidence evaluation.

The bundled evaluator is deterministic and lexical.  Production deployments
can implement :class:`LocalEvidenceEvaluator` with a grammar-constrained model;
the execution, verification, caching, and reduction layers remain unchanged.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import date, datetime
import re
from typing import Protocol

from .models import (
    BindingValue,
    EvaluationStatus,
    EvidenceRole,
    LocalEvaluation,
    PredicateSpec,
    SourceBlock,
    WitnessRow,
)


class LocalEvidenceEvaluator(Protocol):
    EVALUATOR_VERSION: str

    def evaluate(
        self,
        block: SourceBlock,
        predicate: PredicateSpec,
        *,
        role: EvidenceRole = EvidenceRole.POSITIVE,
    ) -> LocalEvaluation: ...


def _term_spans(content: str, term: str, case_sensitive: bool) -> list[tuple[int, int]]:
    flags = 0 if case_sensitive else re.IGNORECASE
    return [match.span() for match in re.finditer(re.escape(term), content, flags=flags)]


def _quote_span(content: str, hit_start: int, hit_end: int, window: int) -> tuple[int, int]:
    half = max(20, window // 2)
    lower = max(0, hit_start - half)
    upper = min(len(content), hit_end + half)
    left_candidates = [content.rfind(mark, lower, hit_start) for mark in ("\n", ". ", "? ", "! ")]
    left_boundary = max(left_candidates)
    start = left_boundary + (1 if left_boundary >= 0 and content[left_boundary] == "\n" else 2)
    if left_boundary < 0:
        start = lower
    right_candidates: list[int] = []
    for mark in ("\n", ". ", "? ", "! "):
        position = content.find(mark, hit_end, upper)
        if position >= 0:
            right_candidates.append(position + (1 if mark != "\n" else 0))
    end = min(right_candidates) if right_candidates else upper
    while start < end and content[start].isspace():
        start += 1
    while end > start and content[end - 1].isspace():
        end -= 1
    return start, end


def _convert(surface: str, value_type: str):
    normalized_type = value_type.casefold()
    stripped = surface.strip()
    if normalized_type in {"str", "string", "entity", "date", "datetime"}:
        if normalized_type == "date":
            # Validation only; preserve an ISO scalar rather than a Python object.
            date.fromisoformat(stripped)
        elif normalized_type == "datetime":
            datetime.fromisoformat(stripped[:-1] + "+00:00" if stripped.endswith("Z") else stripped)
        return stripped
    if normalized_type in {"int", "integer"}:
        return int(stripped.replace(",", ""))
    if normalized_type in {"float", "number", "decimal"}:
        return float(stripped.replace(",", ""))
    if normalized_type in {"bool", "boolean"}:
        if stripped.casefold() in {"true", "yes"}:
            return True
        if stripped.casefold() in {"false", "no"}:
            return False
        raise ValueError("not a boolean surface")
    return stripped


class LexicalEvidenceEvaluator:
    """Conservative exhaustive local predicate evaluator.

    This component does *not* claim to understand semantic equivalence.  A miss
    is conclusive only for predicates explicitly marked ``closed_world``.
    """

    EVALUATOR_VERSION = "lexical-evaluator-0.2.0"

    def __init__(self, *, max_witnesses_per_block: int = 256) -> None:
        if max_witnesses_per_block < 1:
            raise ValueError("max_witnesses_per_block must be positive")
        self.max_witnesses_per_block = max_witnesses_per_block

    def evaluate(
        self,
        block: SourceBlock,
        predicate: PredicateSpec,
        *,
        role: EvidenceRole = EvidenceRole.POSITIVE,
    ) -> LocalEvaluation:
        content = block.content
        must_spans = [
            _term_spans(content, term, predicate.case_sensitive) for term in predicate.must_terms
        ]
        if any(not spans for spans in must_spans):
            return self._miss(predicate, "one or more required lexical terms were absent")

        group_spans: list[list[tuple[int, int]]] = []
        for group in predicate.any_term_groups:
            spans = [
                span
                for term in group
                for span in _term_spans(content, term, predicate.case_sensitive)
            ]
            group_spans.append(spans)
        if any(not spans for spans in group_spans):
            return self._miss(predicate, "one or more required alternative-term groups were absent")

        for term in predicate.exclude_terms:
            if _term_spans(content, term, predicate.case_sensitive):
                return self._miss(predicate, f"exclusion term was present: {term}")

        trigger_spans: list[tuple[int, int]] = []
        for term in predicate.should_terms:
            trigger_spans.extend(_term_spans(content, term, predicate.case_sensitive))
        trigger_spans.extend(span for spans in group_spans for span in spans)
        regex_flags = 0 if predicate.case_sensitive else re.IGNORECASE
        for pattern in predicate.patterns:
            trigger_spans.extend(match.span() for match in re.finditer(pattern, content, flags=regex_flags))

        has_alternatives = bool(
            predicate.should_terms or predicate.patterns or predicate.any_term_groups
        )
        if has_alternatives and not trigger_spans:
            return self._miss(predicate, "no alternative lexical or regex trigger matched")
        if not has_alternatives:
            trigger_spans = [span for spans in must_spans for span in spans]
        if not trigger_spans:
            # An empty predicate is meaningful to an external semantic judge but
            # cannot be decided by this implementation.
            return LocalEvaluation(
                EvaluationStatus.UNCERTAIN,
                reason="predicate has no lexical decision boundary",
            )

        quote_spans = sorted(
            {_quote_span(content, start, end, predicate.quote_window) for start, end in trigger_spans}
        )
        # Merge overlapping windows to avoid duplicate evidence without losing
        # any matched source bytes.
        merged: list[tuple[int, int]] = []
        for start, end in quote_spans:
            if (
                merged
                and start <= merged[-1][1]
                and max(merged[-1][1], end) - merged[-1][0] <= predicate.quote_window * 2
            ):
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))
            else:
                merged.append((start, end))

        truncated = len(merged) > self.max_witnesses_per_block
        selected = merged[: self.max_witnesses_per_block]
        witnesses: list[WitnessRow] = []
        missing_required = False
        conversion_failures: list[str] = []
        for local_start, local_end in selected:
            quote = content[local_start:local_end]
            absolute_start = block.start_offset + local_start
            absolute_end = block.start_offset + local_end
            bindings: dict[str, BindingValue] = {}
            for name, pattern in predicate.binding_patterns.items():
                match = re.search(pattern, quote, flags=regex_flags)
                if match is None:
                    if name in predicate.required_bindings:
                        missing_required = True
                    continue
                surface = match.group(0)
                binding_start = absolute_start + match.start()
                binding_end = absolute_start + match.end()
                value_type = predicate.binding_types.get(name, "string")
                try:
                    value = _convert(surface, value_type)
                except (ValueError, TypeError):
                    conversion_failures.append(name)
                    value = surface
                    value_type = "string"
                bindings[name] = BindingValue(
                    value=value,
                    value_type=value_type,
                    surface=surface,
                    start_offset=binding_start,
                    end_offset=binding_end,
                    resolver=f"regex:{pattern}",
                )
            uncertainties: list[str] = []
            if missing_required:
                uncertainties.append("one or more required bindings were not extracted")
            if conversion_failures:
                uncertainties.append(
                    "binding type conversion failed: " + ", ".join(sorted(set(conversion_failures)))
                )
            if truncated:
                uncertainties.append("witness limit reached; additional local matches exist")
            witnesses.append(
                WitnessRow.create(
                    predicate_id=predicate.predicate_id,
                    block=block,
                    start_offset=absolute_start,
                    end_offset=absolute_end,
                    quote=quote,
                    bindings=bindings,
                    polarity=predicate.polarity,
                    modality=predicate.modality,
                    evaluator_version=self.EVALUATOR_VERSION,
                    role=role,
                    uncertainties=uncertainties,
                )
            )
        if missing_required or conversion_failures or truncated:
            return LocalEvaluation(
                EvaluationStatus.UNCERTAIN,
                tuple(witnesses),
                "candidate evidence was found but extraction was incomplete",
            )
        return LocalEvaluation(EvaluationStatus.MATCH, tuple(witnesses), "predicate matched")

    @staticmethod
    def _miss(predicate: PredicateSpec, reason: str) -> LocalEvaluation:
        return LocalEvaluation(
            EvaluationStatus.NO_MATCH if predicate.closed_world else EvaluationStatus.UNCERTAIN,
            reason=reason,
        )
