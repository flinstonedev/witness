"""Answer-key isolation and scorer-side evidence exposure matching."""

from __future__ import annotations

from dataclasses import replace
from typing import Iterable, Mapping, Sequence

from .models import (
    BenchmarkTask,
    EvidenceRef,
    RetrievedPassage,
    SourceDocument,
    SystemAnswer,
)


_PUBLIC_TASK_METADATA = frozenset({"answer_schema", "query_features", "temporal_axis"})


def public_task(task: BenchmarkTask) -> BenchmarkTask:
    """Return the query contract visible to a system under evaluation.

    Expected values and relevance/counter labels are deliberately blank.  The
    snapshot is selected by the harness, so the system does not need update-pair
    annotations either.
    """

    metadata = {
        key: value for key, value in task.metadata.items() if key in _PUBLIC_TASK_METADATA
    }
    return BenchmarkTask(
        task_id=task.task_id,
        category=task.category,
        question=task.question,
        corpus_versions=(),
        expected_answer=None,
        evidence_ids=(),
        decisive_evidence_ids=(),
        counter_evidence_ids=(),
        as_of=task.as_of,
        answer_type=task.answer_type,
        metadata=metadata,
    )


def public_documents(documents: Sequence[SourceDocument]) -> tuple[SourceDocument, ...]:
    """Defensively strip annotations even if a caller supplied enriched docs."""

    return tuple(replace(document, evidence_spans=()) for document in documents)


def _contained(raw: RetrievedPassage, evidence: EvidenceRef) -> bool:
    return (
        raw.document_id == evidence.document_id
        and raw.document_version == evidence.document_version
        and raw.start_offset <= evidence.start_offset
        and evidence.end_offset <= raw.end_offset
    )


def _derived_contains(raw: RetrievedPassage, evidence: EvidenceRef) -> bool:
    """Match a synthesized page only when exact source text survived compression."""

    source_ids = raw.metadata.get("source_ids", ())
    if isinstance(source_ids, str):
        source_ids = (source_ids,)
    return (
        raw.document_id.startswith("wiki:")
        and evidence.document_id in source_ids
        and bool(evidence.quote)
        and evidence.quote in raw.text
    )


def exposed_evidence(
    passages: Iterable[RetrievedPassage],
    evidence: Iterable[EvidenceRef],
) -> tuple[EvidenceRef, ...]:
    """Find gold spans physically exposed by returned context.

    This function is called after a system has returned.  It never influences
    ranking or execution.  Exact containment is used instead of same-document
    credit so a system cannot retrieve an unrelated paragraph and receive the
    decisive evidence score.
    """

    refs_by_document: dict[tuple[str, str], list[EvidenceRef]] = {}
    all_refs = tuple(evidence)
    for ref in all_refs:
        refs_by_document.setdefault((ref.document_id, ref.document_version), []).append(ref)

    found: dict[str, EvidenceRef] = {}
    for passage in passages:
        for annotated in passage.evidence:
            # Only accept an embedded annotation if it is present in the scorer's
            # own index and its immutable source coordinates agree.
            for ref in all_refs:
                if (
                    ref.evidence_id == annotated.evidence_id
                    and ref.document_id == annotated.document_id
                    and ref.document_version == annotated.document_version
                    and ref.start_offset == annotated.start_offset
                    and ref.end_offset == annotated.end_offset
                ):
                    found[ref.evidence_id] = ref
                    break
        for ref in refs_by_document.get(
            (passage.document_id, passage.document_version), ()
        ):
            if _contained(passage, ref):
                found[ref.evidence_id] = ref
        if passage.document_id.startswith("wiki:"):
            for ref in all_refs:
                if _derived_contains(passage, ref):
                    found[ref.evidence_id] = ref
    return tuple(found[key] for key in sorted(found))


def annotate_answer(
    answer: SystemAnswer,
    evidence_index: Mapping[str, EvidenceRef] | Iterable[EvidenceRef],
) -> SystemAnswer:
    """Attach scorer-derived gold exposure to a copy of a system answer."""

    refs = (
        tuple(evidence_index.values())
        if isinstance(evidence_index, Mapping)
        else tuple(evidence_index)
    )
    exposed = exposed_evidence(answer.retrieved_passages, refs)
    scorer_metadata = dict(answer.metadata)
    scorer_metadata["scorer_attached_evidence"] = True
    scorer_metadata["exposed_gold_span_count"] = len(exposed)
    return replace(answer, evidence=exposed, metadata=scorer_metadata)


def answer_key_is_absent(task: BenchmarkTask, documents: Sequence[SourceDocument]) -> bool:
    """Cheap harness assertion used before invoking any system."""

    return (
        task.expected_answer is None
        and not task.evidence_ids
        and not task.decisive_evidence_ids
        and not task.counter_evidence_ids
        and all(not document.evidence_spans for document in documents)
    )


__all__ = [
    "annotate_answer",
    "answer_key_is_absent",
    "exposed_evidence",
    "public_documents",
    "public_task",
]

