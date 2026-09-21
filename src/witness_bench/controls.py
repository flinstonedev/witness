"""Diagnostic controls that isolate benchmark mechanisms.

These are deliberately not retrieval-system proposals.  In particular, the
exhaustive lexical map below exists to measure what the bundled fallback
predicate/evaluator accomplish by themselves, before crediting WitnessQL,
mechanical verification, counter-search, reduction, or materialized views.
"""

from __future__ import annotations

from hashlib import sha256
from time import perf_counter, process_time
from typing import Any, Mapping, Sequence

from witness_engine import (
    CorpusStore,
    EvaluationStatus,
    EvidenceRole,
    LexicalEvidenceEvaluator,
    WitnessRow,
    fallback_lexical_predicate,
)

from .baselines import token_count
from .isolation import answer_key_is_absent, public_documents, public_task
from .models import (
    BenchmarkTask,
    EvidenceRef,
    RetrievedPassage,
    RetrievalResult,
    RunTelemetry,
    SourceDocument,
    coerce_source_document,
    coerce_task,
    corpus_fingerprint,
)


class ExhaustiveLexicalMapControl:
    """Map the fallback lexical evaluator over every public corpus block.

    A new in-memory corpus and a new exhaustive evaluation are created for
    every call.  There is intentionally no program execution, verifier,
    counter phase, reducer, ranking gate, index, or cache.  Minimal inline
    coordinate checks ensure that emitted passages are exact source slices;
    they are not a substitute for Witness's full mechanical verifier.
    """

    name = "diagnostic-exhaustive-lexical-map-no-cache"
    family = "diagnostic_control"

    def __init__(
        self,
        *,
        evaluator: LexicalEvidenceEvaluator | None = None,
        max_terms: int = 24,
        block_size_chars: int = 12_000,
        overlap_chars: int = 400,
    ) -> None:
        if max_terms < 1:
            raise ValueError("max_terms must be positive")
        self.evaluator = evaluator or LexicalEvidenceEvaluator()
        self.max_terms = int(max_terms)
        self.block_size_chars = int(block_size_chars)
        self.overlap_chars = int(overlap_chars)

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "implementation": "direct_exhaustive_lexical_map",
            "production_equivalent": False,
            "diagnostic_only": True,
            "answer_key_isolated": True,
            "cache_policy": "disabled_recompute_every_call",
            "execution": "one_direct_evaluator_call_per_authorized_block",
            "evaluator_version": self.evaluator.EVALUATOR_VERSION,
            "components_present": [
                "shared_fallback_predicate",
                "shared_lexical_evaluator",
                "lossless_fixed_block_partition",
                "inline_exact_coordinate_guard",
            ],
            "components_absent": [
                "WitnessQL_program_execution",
                "mechanical_verifier",
                "counter_witness_search",
                "deterministic_reducer",
                "ranking_or_prefilter",
                "shard_cache",
                "query_result_cache",
            ],
            "limitations": [
                "This is a causal diagnostic control, not a production retrieval architecture.",
                "The predicate and evaluator are lexical and cannot establish semantic completeness.",
                "Inline checks cover exact source coordinates only, not typed-binding validity.",
                "Every call rescans all authorized blocks; no warm-query or update optimization exists.",
            ],
        }

    @staticmethod
    def _passage(row: WitnessRow, rank: int) -> RetrievedPassage:
        evidence = EvidenceRef(
            evidence_id=row.witness_id,
            document_id=row.source_id,
            document_version=row.source_version,
            quote=row.quote,
            start_offset=row.start_offset,
            end_offset=row.end_offset,
            kind=row.polarity,
            metadata={
                "predicate_id": row.predicate_id,
                "role": row.role.value,
                "modality": row.modality,
                "valid_time": row.valid_time,
                "record_time": row.record_time,
                "coordinate_checked": True,
            },
        )
        return RetrievedPassage(
            passage_id=f"lexical-map:{row.witness_id}",
            document_id=row.source_id,
            document_version=row.source_version,
            text=row.quote,
            start_offset=row.start_offset,
            end_offset=row.end_offset,
            score=1.0,
            rank=rank,
            evidence=(evidence,),
            metadata={
                "predicate_id": row.predicate_id,
                "role": row.role.value,
                "exact_source_span": True,
                "coordinate_checked": True,
                "verified_by_witness_verifier": False,
            },
        )

    def retrieve(
        self,
        task: BenchmarkTask | Mapping[str, Any] | Any,
        documents: Sequence[SourceDocument | Mapping[str, Any] | Any],
    ) -> RetrievalResult:
        # Sanitize defensively even when called outside the benchmark runner.
        query = public_task(coerce_task(task))
        corpus = public_documents(
            tuple(coerce_source_document(document) for document in documents)
        )
        if not answer_key_is_absent(query, corpus):
            raise RuntimeError("answer-key isolation failed for lexical map control")

        wall_started = perf_counter()
        cpu_started = process_time()
        predicate = fallback_lexical_predicate(
            query.question,
            max_terms=self.max_terms,
        )
        store = CorpusStore(
            block_size_chars=self.block_size_chars,
            overlap_chars=self.overlap_chars,
        )
        store.add_documents(corpus)
        record_axis = (
            str(query.metadata.get("temporal_axis", "")).casefold() == "record_time"
        )
        blocks = store.blocks(
            principals=("benchmark",),
            recorded_as_of=query.as_of if record_axis else None,
        )
        source_lookup = {
            (document.document_id, document.version): document for document in corpus
        }

        rows_by_coordinate: dict[tuple[Any, ...], WitnessRow] = {}
        uncertain_blocks: set[str] = set()
        failed_blocks: set[str] = set()
        coordinate_rejections = 0
        errors: list[str] = []
        for block in blocks:
            try:
                evaluation = self.evaluator.evaluate(
                    block,
                    predicate,
                    role=EvidenceRole.POSITIVE,
                )
            except Exception as exc:  # A failed map item is unresolved, never a miss.
                failed_blocks.add(block.block_id)
                errors.append(f"{block.block_id}: {type(exc).__name__}: {exc}")
                continue
            if evaluation.status is EvaluationStatus.UNCERTAIN:
                uncertain_blocks.add(block.block_id)
            for row in evaluation.witnesses:
                document = source_lookup.get((row.source_id, row.source_version))
                coordinate_ok = (
                    row.predicate_id == predicate.predicate_id
                    and row.block_id == block.block_id
                    and row.role is EvidenceRole.POSITIVE
                    and document is not None
                    and row.source_hash
                    == sha256(document.content.encode("utf-8")).hexdigest()
                    and 0 <= row.start_offset <= row.end_offset <= len(document.content)
                    and document.content[row.start_offset : row.end_offset] == row.quote
                )
                if not coordinate_ok:
                    coordinate_rejections += 1
                    failed_blocks.add(block.block_id)
                    errors.append(
                        f"{block.block_id}/{predicate.predicate_id}: invalid source coordinates"
                    )
                    continue
                identity = (
                    row.source_id,
                    row.source_version,
                    row.start_offset,
                    row.end_offset,
                    row.predicate_id,
                    row.role.value,
                )
                rows_by_coordinate.setdefault(identity, row)

        rows = tuple(rows_by_coordinate[key] for key in sorted(rows_by_coordinate))
        passages = tuple(
            self._passage(row, rank) for rank, row in enumerate(rows, start=1)
        )
        scanned = len(blocks) - len(failed_blocks)
        uncertain = len(uncertain_blocks.difference(failed_blocks))
        corpus_tokens = sum(token_count(block.content) for block in blocks)
        storage_bytes = sum(len(document.content.encode("utf-8")) for document in corpus)
        storage_bytes += sum(len(passage.text.encode("utf-8")) for passage in passages)
        conclusive = (scanned - uncertain) / len(blocks) if blocks else 1.0
        telemetry = RunTelemetry(
            # Cold means semantic work was performed. This control is cold on
            # every invocation by construction.
            cold=True,
            input_tokens=token_count(query.question) + corpus_tokens,
            output_tokens=sum(token_count(passage.text) for passage in passages),
            cpu_time_ms=(process_time() - cpu_started) * 1000.0,
            latency_ms=(perf_counter() - wall_started) * 1000.0,
            storage_bytes=storage_bytes,
            cache_hits=0,
            authorized_shards=len(blocks),
            scanned_shards=scanned,
            soundly_skipped_shards=0,
            unresolved_shards=len(failed_blocks),
            uncertain_shards=uncertain,
            counters={
                "map_obligations": float(len(blocks)),
                "evaluator_calls": float(len(blocks)),
                "cache_disabled": 1.0,
                "coordinate_rejections": float(coordinate_rejections),
                "evaluation_errors": float(len(errors)),
                "verified_witness_rows": 0.0,
                "coordinate_checked_rows": float(len(rows)),
                "conclusive_coverage": conclusive,
            },
        )
        metadata = self.metadata
        metadata.update(
            {
                "corpus_fingerprint": corpus_fingerprint(corpus),
                "fallback_predicate": predicate.to_dict(),
                "unresolved": errors,
                "block_size_chars": self.block_size_chars,
                "overlap_chars": self.overlap_chars,
            }
        )
        return RetrievalResult(
            system_name=self.name,
            task_id=query.task_id,
            passages=passages,
            telemetry=telemetry,
            metadata=metadata,
        )


__all__ = ["ExhaustiveLexicalMapControl"]
