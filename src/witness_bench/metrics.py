"""Evidence-first evaluation and resource aggregation for Witness experiments.

Gold evidence remains scorer-side.  Retrieval systems may return exact evidence
rows, raw passages, or both; this module maps those outputs to gold spans by
source version and containment after the run.  That separation prevents a
baseline from seeing the annotations it is evaluated against.

Several metrics can be inapplicable.  Such values are returned as ``None`` and
aggregate helpers skip them instead of quietly treating them as success or
failure.  This matters for, for example, counter-evidence recall on a task with
no annotated counter-evidence.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
import math
import re
from typing import Any, Iterable, Mapping, Sequence

from .models import (
    BenchmarkTask,
    Claim,
    EvidenceRef,
    RetrievalResult,
    RunTelemetry,
    SourceDocument,
    SystemAnswer,
    canonical_json,
    coerce_evidence_ref,
    coerce_retrieval_result,
    coerce_source_document,
    coerce_system_answer,
    coerce_task,
    coerce_telemetry,
)


_SPACE_RE = re.compile(r"\s+")
_STATUS_ALIASES = {
    "not found": "not_found",
    "no evidence found": "not_found",
    "unknown": "unresolved",
    "uncertain": "unresolved",
    "unresolved": "unresolved",
    "does not exist": "does_not_exist",
    "none exist": "does_not_exist",
    "absent": "does_not_exist",
    "exists": "exists",
    "found": "exists",
}


def _safe_recall(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _catalog(value: Mapping[str, Any] | Sequence[Any] | None) -> dict[str, EvidenceRef]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        output: dict[str, EvidenceRef] = {}
        for key, item in value.items():
            if isinstance(item, str):
                # An ID-only catalog can score explicit IDs but not passage overlap.
                output[str(key)] = EvidenceRef(str(key), "", quote=item)
            else:
                ref = coerce_evidence_ref(item)
                output[ref.evidence_id or str(key)] = ref
        return output
    output = {}
    for item in value:
        ref = coerce_evidence_ref(item)
        output[ref.evidence_id] = ref
    return output


def _output_parts(
    output: SystemAnswer | RetrievalResult | Mapping[str, Any] | Any,
) -> tuple[tuple[EvidenceRef, ...], tuple[Any, ...], tuple[Claim, ...], RunTelemetry, Any]:
    if isinstance(output, RetrievalResult) or (
        isinstance(output, Mapping)
        and "passages" in output
        and "answer" not in output
        and "claims" not in output
    ):
        result = coerce_retrieval_result(output)
        return result.evidence, result.passages, (), result.telemetry, None
    answer = coerce_system_answer(output)
    return answer.evidence, answer.retrieved_passages, answer.claims, answer.telemetry, answer.answer


def _version_matches(gold: EvidenceRef, document_id: str, version: str) -> bool:
    return gold.document_id == document_id and (
        not gold.document_version or not version or gold.document_version == version
    )


def _evidence_contains_gold(emitted: EvidenceRef, gold: EvidenceRef) -> bool:
    if emitted.evidence_id == gold.evidence_id:
        # An ID-only catalog can only score explicit identity.  With a complete
        # scorer-side reference, identity is insufficient: forged IDs must not
        # bypass immutable source coordinates and exact quoted content.
        if not gold.document_id:
            return True
        return (
            _version_matches(gold, emitted.document_id, emitted.document_version)
            and emitted.start_offset == gold.start_offset
            and emitted.end_offset == gold.end_offset
            and (not gold.quote or emitted.quote == gold.quote)
        )
    if not _version_matches(gold, emitted.document_id, emitted.document_version):
        return False
    if not (emitted.start_offset <= gold.start_offset and gold.end_offset <= emitted.end_offset):
        return False
    if not gold.quote:
        return True
    relative_start = gold.start_offset - emitted.start_offset
    relative_end = relative_start + len(gold.quote)
    return emitted.quote[relative_start:relative_end] == gold.quote


def _passage_contains_gold(passage: Any, gold: EvidenceRef) -> bool:
    document_id = str(getattr(passage, "document_id", ""))
    document_version = str(getattr(passage, "document_version", ""))
    metadata = getattr(passage, "metadata", {}) or {}
    # The benchmark labels its procedurally generated *raw corpus* as
    # ``metadata.synthetic``.  Only a derived wiki document changes the source
    # coordinate system; identify that by its reserved document ID, matching the
    # isolation layer, rather than by fixture provenance metadata.
    if document_id.startswith("wiki:"):
        source_ids = metadata.get("source_ids", ())
        if isinstance(source_ids, str):
            source_ids = (source_ids,)
        # Synthesized pages have offsets in generated text, so exact surviving
        # quote plus source lineage is the conservative scorer-side criterion.
        return gold.document_id in source_ids and bool(gold.quote and gold.quote in passage.text)
    if not _version_matches(gold, document_id, document_version):
        return False
    start = int(getattr(passage, "start_offset", 0))
    end = int(getattr(passage, "end_offset", start + len(getattr(passage, "text", ""))))
    if not (start <= gold.start_offset and gold.end_offset <= end):
        return False
    if not gold.quote:
        return True
    relative_start = gold.start_offset - start
    relative_end = relative_start + len(gold.quote)
    return passage.text[relative_start:relative_end] == gold.quote


@dataclass(slots=True)
class EvidenceExposure:
    """Scorer-side matching between emitted units and annotated gold spans."""

    exposed_ids: frozenset[str]
    unit_matches: tuple[frozenset[str], ...]
    explicit_ids: frozenset[str] = frozenset()

    @property
    def retrieved_units(self) -> int:
        return len(self.unit_matches)

    @property
    def relevant_units(self) -> int:
        return sum(bool(item) for item in self.unit_matches)


def infer_evidence_exposure(
    output: SystemAnswer | RetrievalResult | Mapping[str, Any] | Any,
    evidence_catalog: Mapping[str, Any] | Sequence[Any] | None = None,
) -> EvidenceExposure:
    """Infer exposed evidence without exposing gold annotations to the system.

    When passages exist, precision is passage-unit precision: a passage is
    relevant if it fully contains at least one gold span.  When an evidence-only
    architecture emits rows, each row is the evaluation unit.  Recall is always
    computed over unique gold evidence IDs.
    """

    catalog = _catalog(evidence_catalog)
    emitted, passages, _, _, _ = _output_parts(output)
    explicit_ids = frozenset(item.evidence_id for item in emitted)
    # Identity is diagnostic metadata, not scorer credit.  Every emitted row is
    # matched through ``_evidence_contains_gold`` below so full catalogs enforce
    # source/version/offset/quote integrity.
    exposed: set[str] = set()
    unit_matches: list[frozenset[str]] = []
    if passages:
        for passage in passages:
            matches = frozenset(
                evidence_id
                for evidence_id, gold in catalog.items()
                if _passage_contains_gold(passage, gold)
            )
            unit_matches.append(matches)
            exposed.update(matches)
        # Exact evidence rows are usually renderer citations for these same
        # passages and are not double-counted as precision units.  Evidence
        # rows outside all returned passages remain independent emitted units.
        for item in emitted:
            matches = frozenset(
                evidence_id
                for evidence_id, gold in catalog.items()
                if _evidence_contains_gold(item, gold)
            )
            exposed.update(matches)
            duplicated_by_passage = any(
                passage.document_id == item.document_id
                and passage.document_version == item.document_version
                and passage.start_offset <= item.start_offset
                and item.end_offset <= passage.end_offset
                for passage in passages
            )
            if not duplicated_by_passage:
                unit_matches.append(matches)
    else:
        for item in emitted:
            matches = frozenset(
                evidence_id
                for evidence_id, gold in catalog.items()
                if _evidence_contains_gold(item, gold)
            )
            unit_matches.append(matches)
            exposed.update(matches)
    return EvidenceExposure(
        exposed_ids=frozenset(exposed),
        unit_matches=tuple(unit_matches),
        explicit_ids=explicit_ids,
    )


def _task_relevant_ids(task: BenchmarkTask) -> frozenset[str]:
    return frozenset(
        (*task.evidence_ids, *task.decisive_evidence_ids, *task.counter_evidence_ids)
    )


def evidence_recall(
    task: BenchmarkTask | Mapping[str, Any] | Any,
    output: SystemAnswer | RetrievalResult | Mapping[str, Any] | Any,
    evidence_catalog: Mapping[str, Any] | Sequence[Any] | None = None,
) -> float | None:
    benchmark_task = coerce_task(task)
    relevant = _task_relevant_ids(benchmark_task)
    exposed = infer_evidence_exposure(output, evidence_catalog).exposed_ids
    return _safe_recall(len(relevant & exposed), len(relevant))


def evidence_precision(
    task: BenchmarkTask | Mapping[str, Any] | Any,
    output: SystemAnswer | RetrievalResult | Mapping[str, Any] | Any,
    evidence_catalog: Mapping[str, Any] | Sequence[Any] | None = None,
) -> float | None:
    benchmark_task = coerce_task(task)
    relevant = _task_relevant_ids(benchmark_task)
    exposure = infer_evidence_exposure(output, evidence_catalog)
    relevant_units = sum(bool(matches & relevant) for matches in exposure.unit_matches)
    return _safe_recall(relevant_units, exposure.retrieved_units)


def decisive_evidence_recall(
    task: BenchmarkTask | Mapping[str, Any] | Any,
    output: SystemAnswer | RetrievalResult | Mapping[str, Any] | Any,
    evidence_catalog: Mapping[str, Any] | Sequence[Any] | None = None,
) -> float | None:
    benchmark_task = coerce_task(task)
    expected = frozenset(benchmark_task.decisive_evidence_ids)
    exposed = infer_evidence_exposure(output, evidence_catalog).exposed_ids
    return _safe_recall(len(expected & exposed), len(expected))


def counter_evidence_recall(
    task: BenchmarkTask | Mapping[str, Any] | Any,
    output: SystemAnswer | RetrievalResult | Mapping[str, Any] | Any,
    evidence_catalog: Mapping[str, Any] | Sequence[Any] | None = None,
) -> float | None:
    benchmark_task = coerce_task(task)
    expected = frozenset(benchmark_task.counter_evidence_ids)
    exposed = infer_evidence_exposure(output, evidence_catalog).exposed_ids
    return _safe_recall(len(expected & exposed), len(expected))


def weighted_evidence_recall(
    task: BenchmarkTask | Mapping[str, Any] | Any,
    output: SystemAnswer | RetrievalResult | Mapping[str, Any] | Any,
    evidence_catalog: Mapping[str, Any] | Sequence[Any] | None = None,
    *,
    decisive_weight: float = 3.0,
    counter_weight: float = 2.0,
) -> float | None:
    """Recall with higher omission cost for decisive and counter evidence."""

    benchmark_task = coerce_task(task)
    ids = _task_relevant_ids(benchmark_task)
    if not ids:
        return None
    decisive = set(benchmark_task.decisive_evidence_ids)
    counters = set(benchmark_task.counter_evidence_ids)
    weights = {
        evidence_id: max(
            1.0,
            decisive_weight if evidence_id in decisive else 1.0,
            counter_weight if evidence_id in counters else 1.0,
        )
        for evidence_id in ids
    }
    exposed = infer_evidence_exposure(output, evidence_catalog).exposed_ids
    return sum(weight for key, weight in weights.items() if key in exposed) / sum(weights.values())


def evidence_metrics(
    task: BenchmarkTask | Mapping[str, Any] | Any,
    output: SystemAnswer | RetrievalResult | Mapping[str, Any] | Any,
    evidence_catalog: Mapping[str, Any] | Sequence[Any] | None = None,
) -> dict[str, Any]:
    benchmark_task = coerce_task(task)
    exposure = infer_evidence_exposure(output, evidence_catalog)
    relevant = _task_relevant_ids(benchmark_task)
    decisive = frozenset(benchmark_task.decisive_evidence_ids)
    counters = frozenset(benchmark_task.counter_evidence_ids)
    return {
        "evidence_recall": _safe_recall(len(relevant & exposure.exposed_ids), len(relevant)),
        "evidence_precision": _safe_recall(
            sum(bool(matches & relevant) for matches in exposure.unit_matches),
            exposure.retrieved_units,
        ),
        "decisive_evidence_recall": _safe_recall(
            len(decisive & exposure.exposed_ids), len(decisive)
        ),
        "counter_evidence_recall": _safe_recall(
            len(counters & exposure.exposed_ids), len(counters)
        ),
        "weighted_evidence_recall": weighted_evidence_recall(
            benchmark_task, output, evidence_catalog
        ),
        "gold_evidence_count": len(relevant),
        "exposed_gold_evidence_count": len(relevant & exposure.exposed_ids),
        "retrieved_evidence_units": exposure.retrieved_units,
        "relevant_retrieved_units": sum(
            bool(matches & relevant) for matches in exposure.unit_matches
        ),
        "exposed_evidence_ids": sorted(exposure.exposed_ids),
        "precision_unit": "passage" if _output_parts(output)[1] else "evidence_row",
    }


def _normalize_string(value: str) -> str:
    return _SPACE_RE.sub(" ", value.strip()).casefold()


def _unwrap_answer(value: Any) -> Any:
    if isinstance(value, Mapping):
        for key in ("answer", "value", "result"):
            if key in value and len(value) == 1:
                return value[key]
    return value


def _answer_equal(
    predicted: Any,
    expected: Any,
    *,
    numeric_tolerance: float = 0.0,
    order_insensitive: bool = False,
) -> bool:
    predicted = _unwrap_answer(predicted)
    expected = _unwrap_answer(expected)
    if isinstance(expected, bool) or isinstance(predicted, bool):
        return predicted is expected
    if isinstance(expected, (int, float)) and isinstance(predicted, (int, float)):
        if not (math.isfinite(float(expected)) and math.isfinite(float(predicted))):
            return predicted == expected
        return abs(float(predicted) - float(expected)) <= numeric_tolerance
    if isinstance(expected, str) and isinstance(predicted, str):
        return _normalize_string(predicted) == _normalize_string(expected)
    if isinstance(expected, Mapping) and isinstance(predicted, Mapping):
        if set(expected) != set(predicted):
            return False
        return all(
            _answer_equal(
                predicted[key],
                expected[key],
                numeric_tolerance=numeric_tolerance,
                order_insensitive=order_insensitive,
            )
            for key in expected
        )
    if isinstance(expected, (list, tuple, set, frozenset)) and isinstance(
        predicted, (list, tuple, set, frozenset)
    ):
        left, right = list(predicted), list(expected)
        if len(left) != len(right):
            return False
        if order_insensitive:
            left = sorted(left, key=canonical_json)
            right = sorted(right, key=canonical_json)
        return all(
            _answer_equal(
                item,
                gold,
                numeric_tolerance=numeric_tolerance,
                order_insensitive=order_insensitive,
            )
            for item, gold in zip(left, right, strict=True)
        )
    return predicted == expected


def answer_correctness(
    task: BenchmarkTask | Mapping[str, Any] | Any,
    output: SystemAnswer | Mapping[str, Any] | Any,
) -> float | None:
    """Deterministic exact/structured correctness; human grading is external."""

    benchmark_task = coerce_task(task)
    if benchmark_task.expected_answer is None:
        return None
    answer = coerce_system_answer(output).answer
    accepted = benchmark_task.metadata.get("accepted_answers")
    candidates = list(accepted) if accepted is not None else [benchmark_task.expected_answer]
    tolerance = float(benchmark_task.metadata.get("numeric_tolerance", 0.0))
    order_insensitive = bool(benchmark_task.metadata.get("order_insensitive", False))
    return float(
        any(
            _answer_equal(
                answer,
                expected,
                numeric_tolerance=tolerance,
                order_insensitive=order_insensitive,
            )
            for expected in candidates
        )
    )


def _category_applies(task: BenchmarkTask, *needles: str) -> bool:
    category = task.category.casefold().replace("-", "_").replace(" ", "_")
    return any(needle in category for needle in needles)


def _specialized_accuracy(
    task: BenchmarkTask,
    output: SystemAnswer | Mapping[str, Any] | Any,
    *,
    expected_metadata_key: str,
) -> float | None:
    expected = task.metadata.get(expected_metadata_key, task.expected_answer)
    if expected is None:
        return None
    answer = coerce_system_answer(output).answer
    return float(
        _answer_equal(
            answer,
            expected,
            numeric_tolerance=float(task.metadata.get("numeric_tolerance", 0.0)),
            order_insensitive=bool(task.metadata.get("order_insensitive", False)),
        )
    )


def temporal_accuracy(
    task: BenchmarkTask | Mapping[str, Any] | Any,
    output: SystemAnswer | Mapping[str, Any] | Any,
) -> float | None:
    benchmark_task = coerce_task(task)
    if not (
        _category_applies(benchmark_task, "temporal")
        or "temporal_expected_answer" in benchmark_task.metadata
    ):
        return None
    return _specialized_accuracy(
        benchmark_task, output, expected_metadata_key="temporal_expected_answer"
    )


def aggregation_accuracy(
    task: BenchmarkTask | Mapping[str, Any] | Any,
    output: SystemAnswer | Mapping[str, Any] | Any,
) -> float | None:
    benchmark_task = coerce_task(task)
    if not (
        _category_applies(benchmark_task, "aggregation", "global_synthesis", "distributed")
        or "aggregation_expected_answer" in benchmark_task.metadata
    ):
        return None
    return _specialized_accuracy(
        benchmark_task, output, expected_metadata_key="aggregation_expected_answer"
    )


def _normalize_status(value: Any) -> str | None:
    if value is None:
        return None
    normalized = _normalize_string(str(value)).replace("-", "_")
    return _STATUS_ALIASES.get(normalized, normalized.replace(" ", "_"))


def _output_negative_status(output: SystemAnswer | Mapping[str, Any] | Any) -> str | None:
    answer = coerce_system_answer(output)
    for container in (answer.metadata, answer.answer if isinstance(answer.answer, Mapping) else {}):
        for key in ("negative_claim_status", "negative_status", "status"):
            if key in container:
                return _normalize_status(container[key])
    if answer.uncertainty:
        text = answer.uncertainty.casefold()
        if "not found" in text or "no matching" in text:
            return "not_found"
        if "unresolved" in text or "uncertain" in text:
            return "unresolved"
    return None


def negative_claim_calibration(
    task: BenchmarkTask | Mapping[str, Any] | Any,
    output: SystemAnswer | Mapping[str, Any] | Any,
) -> float | None:
    """Score ``not_found`` versus ontological ``does_not_exist`` explicitly."""

    benchmark_task = coerce_task(task)
    expected = benchmark_task.metadata.get("negative_claim_status")
    if expected is None and isinstance(benchmark_task.expected_answer, Mapping):
        expected = benchmark_task.expected_answer.get("negative_claim_status") or benchmark_task.expected_answer.get(
            "status"
        )
    if expected is None:
        return None
    predicted = _output_negative_status(output)
    return float(predicted == _normalize_status(expected)) if predicted is not None else 0.0


def negative_claim_accuracy(
    task: BenchmarkTask | Mapping[str, Any] | Any,
    output: SystemAnswer | Mapping[str, Any] | Any,
) -> float | None:
    benchmark_task = coerce_task(task)
    if not (
        _category_applies(benchmark_task, "negative")
        or "negative_claim_status" in benchmark_task.metadata
    ):
        return None
    calibration = negative_claim_calibration(benchmark_task, output)
    if calibration is not None:
        return calibration
    return answer_correctness(benchmark_task, output)


def unsupported_claim_rate(
    output: SystemAnswer | Mapping[str, Any] | Any,
    *,
    support_judgments: Mapping[str, bool] | None = None,
    corpus: Sequence[SourceDocument | Mapping[str, Any] | Any] | None = None,
) -> float | None:
    """Return factual claims without a valid support mapping divided by claims.

    ``support_judgments`` should come from a human or semantic entailment grader
    when available.  Otherwise this checks structural support: a claim must cite
    an emitted evidence row.  Exact source-anchored rows are additionally
    validated against ``corpus`` when it is supplied.  Structural support alone
    does not prove entailment; evaluation reports expose that limitation.
    """

    answer = coerce_system_answer(output)
    factual = [claim for claim in answer.claims if claim.factual]
    if not factual:
        return None
    judgments = dict(support_judgments or {})
    evidence_by_id = {item.evidence_id: item for item in answer.evidence}
    documents = {
        (item.document_id, item.version): item
        for item in (coerce_source_document(value) for value in (corpus or ()))
    }

    def exact_source_anchor(claim: Claim) -> bool:
        """Validate an engine-generated citation ID by immutable coordinates.

        The scorer may replace a system's evidence list with scorer-attached gold
        spans.  A synthetic witness ID must not become unsupported merely due to
        that replacement: source/version/offset/quote, rather than ID equality,
        is the authority hierarchy.
        """

        if not claim.metadata.get("source_anchored"):
            return False
        source_id = str(claim.metadata.get("source_id", ""))
        try:
            start = int(claim.metadata["start_offset"])
            end = int(claim.metadata["end_offset"])
        except (KeyError, TypeError, ValueError):
            return False
        for passage in answer.retrieved_passages:
            if passage.document_id != source_id:
                continue
            if not (passage.start_offset <= start <= end <= passage.end_offset):
                continue
            relative_start = start - passage.start_offset
            relative_end = end - passage.start_offset
            if passage.text[relative_start:relative_end] == claim.text:
                return True
        for (document_id, _), document in documents.items():
            if (
                document_id == source_id
                and 0 <= start <= end <= len(document.content)
                and document.content[start:end] == claim.text
            ):
                return True
        return False

    def supported(claim: Claim) -> bool:
        if claim.claim_id in judgments:
            return bool(judgments[claim.claim_id])
        if "supported" in claim.metadata:
            return bool(claim.metadata["supported"])
        citations = [evidence_by_id[item] for item in claim.evidence_ids if item in evidence_by_id]
        if not citations:
            return exact_source_anchor(claim)
        if not documents:
            return True
        valid = False
        for citation in citations:
            document = documents.get((citation.document_id, citation.document_version))
            if document is None:
                continue
            if (
                0 <= citation.start_offset <= citation.end_offset <= len(document.content)
                and document.content[citation.start_offset : citation.end_offset] == citation.quote
            ):
                valid = True
                break
        return valid

    unsupported = sum(not supported(claim) for claim in factual)
    return unsupported / len(factual)


def claim_support_metrics(
    output: SystemAnswer | Mapping[str, Any] | Any,
    *,
    support_judgments: Mapping[str, bool] | None = None,
    corpus: Sequence[SourceDocument | Mapping[str, Any] | Any] | None = None,
) -> dict[str, Any]:
    answer = coerce_system_answer(output)
    factual_count = sum(claim.factual for claim in answer.claims)
    rate = unsupported_claim_rate(
        answer, support_judgments=support_judgments, corpus=corpus
    )
    return {
        "unsupported_claim_rate": rate,
        "factual_claim_count": factual_count,
        "supported_claim_count": (
            None if rate is None else int(round(factual_count * (1.0 - rate)))
        ),
        "support_grading": (
            "semantic_judgments"
            if support_judgments
            else "source_span_structural" if corpus else "citation_structural_only"
        ),
    }


def coverage_metrics(
    output: SystemAnswer | RetrievalResult | Mapping[str, Any] | Any,
) -> dict[str, Any]:
    if isinstance(output, RetrievalResult) or (
        isinstance(output, Mapping) and "passages" in output and "answer" not in output
    ):
        telemetry = coerce_retrieval_result(output).telemetry
        coverage: Mapping[str, Any] = {}
    else:
        answer = coerce_system_answer(output)
        telemetry = answer.telemetry
        coverage = answer.coverage

    def value(name: str, fallback: int) -> int:
        aliases = {
            "authorized_shards": ("authorized_shards",),
            "scanned_shards": ("scanned_shards",),
            "soundly_skipped_shards": ("soundly_skipped_shards",),
            "unresolved_shards": ("unresolved_shards",),
            "uncertain_shards": ("uncertain_shards",),
        }
        for alias in aliases[name]:
            if alias in coverage:
                return int(coverage[alias])
        return int(fallback)

    authorized = value("authorized_shards", telemetry.authorized_shards)
    scanned = value("scanned_shards", telemetry.scanned_shards)
    skipped = value("soundly_skipped_shards", telemetry.soundly_skipped_shards)
    unresolved = value("unresolved_shards", telemetry.unresolved_shards)
    uncertain = value("uncertain_shards", telemetry.uncertain_shards)
    # UNCERTAIN is physically scanned but not conclusive, particularly for a
    # negative claim.  It therefore remains outside the coverage numerator.
    conclusively_evaluated = max(0, scanned - uncertain) + skipped
    rate = conclusively_evaluated / authorized if authorized else None
    internally_consistent = (
        authorized == 0 or conclusively_evaluated + uncertain + unresolved == authorized
    )
    return {
        "coverage": rate,
        "authorized_shards": authorized,
        "scanned_shards": scanned,
        "soundly_skipped_shards": skipped,
        "unresolved_shards": unresolved,
        "uncertain_shards": uncertain,
        "conclusively_evaluated_shards": conclusively_evaluated,
        "fully_covered": bool(
            authorized
            and conclusively_evaluated == authorized
            and not unresolved
            and not uncertain
        ),
        "coverage_certificate_consistent": internally_consistent,
    }


def _agreement(values: Sequence[Any]) -> float | None:
    if len(values) < 2:
        return None
    serialized = [canonical_json(item) for item in values]
    pairs = 0
    agreements = 0
    for index, value in enumerate(serialized):
        for other in serialized[index + 1 :]:
            pairs += 1
            agreements += value == other
    return agreements / pairs if pairs else None


def reproducibility_metrics(
    outputs: Sequence[SystemAnswer | RetrievalResult | Mapping[str, Any] | Any],
    evidence_catalog: Mapping[str, Any] | Sequence[Any] | None = None,
) -> dict[str, Any]:
    """Pairwise repeat agreement for evidence, answer, citations, and uncertainty."""

    if not outputs:
        return {
            "runs": 0,
            "evidence_set_reproducibility": None,
            "answer_reproducibility": None,
            "citation_reproducibility": None,
            "uncertainty_reproducibility": None,
            "reproducibility": None,
        }
    gold_exposure_sets = [
        sorted(infer_evidence_exposure(item, evidence_catalog).exposed_ids) for item in outputs
    ]
    evidence_sets: list[Any] = []
    answers: list[Any] = []
    citations: list[Any] = []
    uncertainty: list[Any] = []
    for item in outputs:
        if isinstance(item, RetrievalResult):
            result = coerce_retrieval_result(item)
            evidence_sets.append(
                sorted(
                    (
                        passage.document_id,
                        passage.document_version,
                        passage.start_offset,
                        passage.end_offset,
                        passage.text,
                    )
                    for passage in result.passages
                )
            )
            answers.append(None)
            citations.append([passage.passage_id for passage in result.passages])
            uncertainty.append(None)
        else:
            answer = coerce_system_answer(item)
            evidence_sets.append(
                sorted(
                    (
                        evidence.document_id,
                        evidence.document_version,
                        evidence.start_offset,
                        evidence.end_offset,
                        evidence.quote,
                    )
                    for evidence in answer.evidence
                )
                if answer.evidence
                else sorted(
                    (
                        passage.document_id,
                        passage.document_version,
                        passage.start_offset,
                        passage.end_offset,
                        passage.text,
                    )
                    for passage in answer.retrieved_passages
                )
            )
            answers.append(answer.answer)
            citations.append(
                [
                    (claim.claim_id, sorted(claim.evidence_ids))
                    for claim in sorted(answer.claims, key=lambda value: value.claim_id)
                ]
            )
            uncertainty.append(answer.uncertainty)
    components = {
        "evidence_set_reproducibility": _agreement(evidence_sets),
        "gold_exposure_reproducibility": _agreement(gold_exposure_sets),
        "answer_reproducibility": _agreement(answers),
        "citation_reproducibility": _agreement(citations),
        "uncertainty_reproducibility": _agreement(uncertainty),
    }
    applicable = [value for value in components.values() if value is not None]
    return {
        "runs": len(outputs),
        **components,
        "reproducibility": sum(applicable) / len(applicable) if applicable else None,
    }


def incremental_update_correctness(
    before_task: BenchmarkTask | Mapping[str, Any] | Any,
    before_output: SystemAnswer | RetrievalResult | Mapping[str, Any] | Any,
    after_task: BenchmarkTask | Mapping[str, Any] | Any,
    after_output: SystemAnswer | RetrievalResult | Mapping[str, Any] | Any,
    *,
    before_evidence_catalog: Mapping[str, Any] | Sequence[Any] | None = None,
    after_evidence_catalog: Mapping[str, Any] | Sequence[Any] | None = None,
) -> dict[str, Any]:
    """Measure correctness and stale-state behavior after a corpus mutation."""

    old_task, new_task = coerce_task(before_task), coerce_task(after_task)
    old_answer = None if isinstance(before_output, RetrievalResult) else coerce_system_answer(before_output).answer
    new_answer = None if isinstance(after_output, RetrievalResult) else coerce_system_answer(after_output).answer
    gold_changed = canonical_json(old_task.expected_answer) != canonical_json(new_task.expected_answer)
    output_changed = canonical_json(old_answer) != canonical_json(new_answer)
    expected_change_behavior = float(output_changed == gold_changed)
    before_catalog = _catalog(before_evidence_catalog)
    after_catalog = _catalog(after_evidence_catalog)
    combined_catalog = {**before_catalog, **after_catalog}
    before_exposure = infer_evidence_exposure(before_output, before_catalog).exposed_ids
    after_exposure = infer_evidence_exposure(after_output, after_catalog).exposed_ids
    after_all_exposure = infer_evidence_exposure(after_output, combined_catalog).exposed_ids
    old_relevant = _task_relevant_ids(old_task)
    new_relevant = _task_relevant_ids(new_task)
    obsolete = old_relevant - new_relevant
    stale = obsolete & after_all_exposure
    stale_rate = len(stale) / len(obsolete) if obsolete else 0.0
    gold_evidence_changed = old_relevant != new_relevant
    output_evidence_changed = before_exposure != after_all_exposure
    expected_evidence_change_behavior = float(
        output_evidence_changed == gold_evidence_changed
    )
    after_answer_correct = (
        None if isinstance(after_output, RetrievalResult) else answer_correctness(new_task, after_output)
    )
    after_evidence_recall = evidence_recall(
        new_task, after_output, after_evidence_catalog
    )
    components = [
        expected_change_behavior,
        expected_evidence_change_behavior,
        1.0 - stale_rate,
        *(
            [after_answer_correct]
            if after_answer_correct is not None
            else []
        ),
        *(
            [after_evidence_recall]
            if after_evidence_recall is not None
            else []
        ),
    ]
    return {
        "update_correctness": sum(components) / len(components),
        "expected_change_behavior": expected_change_behavior,
        "expected_evidence_change_behavior": expected_evidence_change_behavior,
        "post_update_answer_correctness": after_answer_correct,
        "post_update_evidence_recall": after_evidence_recall,
        "stale_evidence_rate": stale_rate,
        "stale_evidence_ids": sorted(stale),
        "gold_answer_changed": gold_changed,
        "system_answer_changed": output_changed,
        "gold_evidence_changed": gold_evidence_changed,
        "system_evidence_changed": output_evidence_changed,
    }


def _percentile(values: Sequence[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def latency_metrics(
    runs: Sequence[RunTelemetry | SystemAnswer | RetrievalResult | Mapping[str, Any] | Any],
) -> dict[str, Any]:
    telemetry = [_extract_telemetry(item) for item in runs]
    values = [item.latency_ms for item in telemetry]
    cold = [item.latency_ms for item in telemetry if item.cold]
    warm = [item.latency_ms for item in telemetry if not item.cold]

    def summarize(items: Sequence[float]) -> dict[str, float | None]:
        return {
            "count": len(items),
            "p50": _percentile(items, 0.50),
            "p95": _percentile(items, 0.95),
            "p99": _percentile(items, 0.99),
            "mean": sum(items) / len(items) if items else None,
            "max": max(items) if items else None,
        }

    return {"latency_ms": summarize(values), "cold_latency_ms": summarize(cold), "warm_latency_ms": summarize(warm)}


def _extract_telemetry(
    value: RunTelemetry | SystemAnswer | RetrievalResult | Mapping[str, Any] | Any,
) -> RunTelemetry:
    if isinstance(value, RunTelemetry):
        return value
    if isinstance(value, RetrievalResult):
        return value.telemetry
    if isinstance(value, SystemAnswer):
        return value.telemetry
    if isinstance(value, Mapping) and "telemetry" in value:
        return coerce_telemetry(value["telemetry"])
    return coerce_telemetry(value)


_RESOURCE_FIELDS = (
    "input_tokens",
    "output_tokens",
    "model_calls",
    "embedding_calls",
    "reranker_calls",
    "tool_calls",
    "cpu_time_ms",
    "latency_ms",
    "cache_hits",
)


def cost_metrics(
    runs: Sequence[RunTelemetry | SystemAnswer | RetrievalResult | Mapping[str, Any] | Any],
) -> dict[str, Any]:
    """Aggregate cold, warm, storage, token, call, and timing counters."""

    telemetry = [_extract_telemetry(item) for item in runs]

    def summarize(items: Sequence[RunTelemetry]) -> dict[str, Any]:
        output: dict[str, Any] = {"runs": len(items)}
        for name in _RESOURCE_FIELDS:
            output[name] = sum(float(getattr(item, name)) for item in items)
            if name not in {"cpu_time_ms", "latency_ms"}:
                output[name] = int(output[name])
        output["storage_bytes_max"] = max((item.storage_bytes for item in items), default=0)
        output["storage_bytes_sum"] = sum(item.storage_bytes for item in items)
        output["storage_bytes"] = output["storage_bytes_max"]
        counters: dict[str, float] = defaultdict(float)
        for item in items:
            for key, value in item.counters.items():
                counters[key] += value
        output["custom_counters"] = dict(sorted(counters.items()))
        return output

    total = summarize(telemetry)
    cold = summarize([item for item in telemetry if item.cold])
    warm = summarize([item for item in telemetry if not item.cold])
    latency = latency_metrics(telemetry)
    # Direct total keys keep CSV export simple; nested sections preserve the
    # requested cold/warm distinction.
    return {
        **total,
        "cold": cold,
        "warm": warm,
        **latency,
    }


def evaluate_task(
    task: BenchmarkTask | Mapping[str, Any] | Any,
    output: SystemAnswer | RetrievalResult | Mapping[str, Any] | Any,
    *,
    evidence_catalog: Mapping[str, Any] | Sequence[Any] | None = None,
    corpus: Sequence[SourceDocument | Mapping[str, Any] | Any] | None = None,
    support_judgments: Mapping[str, bool] | None = None,
) -> dict[str, Any]:
    """Compute all applicable quality metrics for a single task/run."""

    benchmark_task = coerce_task(task)
    metrics = evidence_metrics(benchmark_task, output, evidence_catalog)
    is_retrieval_only = isinstance(output, RetrievalResult) or (
        isinstance(output, Mapping) and "passages" in output and "answer" not in output
    )
    if is_retrieval_only:
        quality = {
            "answer_correctness": None,
            "temporal_accuracy": None,
            "aggregation_accuracy": None,
            "negative_claim_accuracy": None,
            "negative_claim_calibration": None,
            "unsupported_claim_rate": None,
        }
    else:
        quality = {
            "answer_correctness": answer_correctness(benchmark_task, output),
            "temporal_accuracy": temporal_accuracy(benchmark_task, output),
            "aggregation_accuracy": aggregation_accuracy(benchmark_task, output),
            "negative_claim_accuracy": negative_claim_accuracy(benchmark_task, output),
            "negative_claim_calibration": negative_claim_calibration(benchmark_task, output),
            "unsupported_claim_rate": unsupported_claim_rate(
                output,
                support_judgments=support_judgments,
                corpus=corpus,
            ),
        }
    telemetry = _extract_telemetry(output)
    return {
        "task_id": benchmark_task.task_id,
        "category": benchmark_task.category,
        **metrics,
        **quality,
        **coverage_metrics(output),
        "input_tokens": telemetry.input_tokens,
        "output_tokens": telemetry.output_tokens,
        "model_calls": telemetry.model_calls,
        "embedding_calls": telemetry.embedding_calls,
        "reranker_calls": telemetry.reranker_calls,
        "tool_calls": telemetry.tool_calls,
        "cpu_time_ms": telemetry.cpu_time_ms,
        "latency_ms": telemetry.latency_ms,
        "storage_bytes": telemetry.storage_bytes,
        "cold": telemetry.cold,
        "metric_notes": {
            "answer_correctness": "deterministic exact/structured match; human grading is external",
            "unsupported_claim_rate": (
                "semantic judgments" if support_judgments else "structural citation validation only"
            ),
            "evidence_precision_unit": metrics["precision_unit"],
        },
    }


def aggregate_metric_reports(
    reports: Sequence[Mapping[str, Any]],
    *,
    group_key: str | None = None,
) -> dict[str, Any]:
    """Macro-average numeric quality metrics, optionally grouped by a field."""

    if group_key is not None:
        groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for report in reports:
            groups[str(report.get(group_key, "unknown"))].append(report)
        return {
            key: aggregate_metric_reports(values)
            for key, values in sorted(groups.items())
        }
    excluded = {
        "task_id",
        "category",
        "cold",
        "metric_notes",
        "exposed_evidence_ids",
        "precision_unit",
        "fully_covered",
        "coverage_certificate_consistent",
    }
    values: dict[str, list[float]] = defaultdict(list)
    for report in reports:
        for key, value in report.items():
            if key in excluded or isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            values[key].append(float(value))
    return {
        "runs": len(reports),
        "macro": {
            key: sum(items) / len(items)
            for key, items in sorted(values.items())
            if items
        },
        "applicable_counts": {key: len(items) for key, items in sorted(values.items())},
    }


@dataclass(slots=True)
class EvaluationSuite:
    """Small accumulator suitable for CLI and test harness adapters."""

    reports: list[dict[str, Any]] = field(default_factory=list)
    outputs: list[Any] = field(default_factory=list)

    def add(
        self,
        task: BenchmarkTask | Mapping[str, Any] | Any,
        output: SystemAnswer | RetrievalResult | Mapping[str, Any] | Any,
        *,
        evidence_catalog: Mapping[str, Any] | Sequence[Any] | None = None,
        corpus: Sequence[SourceDocument | Mapping[str, Any] | Any] | None = None,
        support_judgments: Mapping[str, bool] | None = None,
    ) -> dict[str, Any]:
        report = evaluate_task(
            task,
            output,
            evidence_catalog=evidence_catalog,
            corpus=corpus,
            support_judgments=support_judgments,
        )
        self.reports.append(report)
        self.outputs.append(output)
        return report

    def summary(self) -> dict[str, Any]:
        return {
            "quality": aggregate_metric_reports(self.reports),
            "by_category": aggregate_metric_reports(self.reports, group_key="category"),
            "cost": cost_metrics(self.outputs),
        }


__all__ = [
    "EvaluationSuite",
    "EvidenceExposure",
    "aggregate_metric_reports",
    "aggregation_accuracy",
    "answer_correctness",
    "claim_support_metrics",
    "cost_metrics",
    "counter_evidence_recall",
    "coverage_metrics",
    "decisive_evidence_recall",
    "evaluate_task",
    "evidence_metrics",
    "evidence_precision",
    "evidence_recall",
    "incremental_update_correctness",
    "infer_evidence_exposure",
    "latency_metrics",
    "negative_claim_accuracy",
    "negative_claim_calibration",
    "reproducibility_metrics",
    "temporal_accuracy",
    "unsupported_claim_rate",
    "weighted_evidence_recall",
]
