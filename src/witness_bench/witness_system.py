"""Benchmark adapter for the Witness execution engine."""

from __future__ import annotations

from dataclasses import replace
from time import perf_counter, process_time
from typing import Any, Mapping, Sequence

from witness_engine import (
    ClaimLedgerEntry,
    ClaimStatus,
    EvaluationStatus,
    LexicalEvidenceEvaluator,
    LocalEvaluation,
    MaterializedWitnessCache,
    QuestionCompiler,
    WitnessEngine,
)

from .baselines import HybridRAGBaseline, token_count
from .models import (
    BenchmarkTask,
    Claim,
    EvidenceRef,
    RetrievedPassage,
    RetrievalResult,
    RunTelemetry,
    SourceDocument,
    SystemAnswer,
    coerce_source_document,
    coerce_task,
    corpus_fingerprint,
)


class WitnessOfflineSystem:
    """Exhaustive Witness with conservative offline lexical semantics.

    This adapter tests execution, verification, coverage, and cache behavior.
    It does not claim that the bundled lexical evaluator is an LLM-equivalent
    semantic predicate evaluator.  The limitation is included in every result.
    """

    name = "witness-exhaustive-offline"
    family = "witness"

    def __init__(
        self,
        *,
        max_workers: int = 8,
        compiler: QuestionCompiler | None = None,
        evaluator: LexicalEvidenceEvaluator | None = None,
        cache: MaterializedWitnessCache | None = None,
        block_size_chars: int = 12_000,
        overlap_chars: int = 400,
        counter_search: bool = True,
        materialized_query_views: bool = True,
    ) -> None:
        self.max_workers = int(max_workers)
        self.compiler = compiler or QuestionCompiler()
        self.evaluator = evaluator or LexicalEvidenceEvaluator()
        self.cache = cache or MaterializedWitnessCache()
        self.block_size_chars = int(block_size_chars)
        self.overlap_chars = int(overlap_chars)
        self.counter_search = bool(counter_search)
        self.materialized_query_views = bool(materialized_query_views)
        if not self.counter_search:
            self.name = "witness-exhaustive-offline-no-counter"
        self._engines: dict[str, WitnessEngine] = {}
        self._result_cache: dict[tuple[str, str, str | None, str, str, bool], RetrievalResult] = {}

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "implementation": "stdlib_offline_conservative_lexical_evaluator",
            "production_equivalent": False,
            "execution": "exhaustive_over_all_authorized_blocks",
            "compiler_version": self.compiler.COMPILER_VERSION,
            "evaluator_version": self.evaluator.EVALUATOR_VERSION,
            "counter_search": self.counter_search,
            "materialized_query_views": self.materialized_query_views,
            "limitations": [
                "The bundled compiler emits one broad lexical evidence obligation.",
                "The local evaluator is lexical, not a production semantic model.",
                "Lexical absence remains UNCERTAIN unless a predicate is explicitly closed-world.",
                "The default output is an evidence relation, not a generative final answer.",
            ],
        }

    def _engine(
        self, documents: tuple[SourceDocument, ...]
    ) -> tuple[WitnessEngine, str, bool]:
        fingerprint = corpus_fingerprint(documents)
        existing = self._engines.get(fingerprint)
        if existing is not None:
            return existing, fingerprint, False
        # Import here to keep the benchmark adapter's public surface small.
        from witness_engine import CorpusStore

        store = CorpusStore(
            block_size_chars=self.block_size_chars,
            overlap_chars=self.overlap_chars,
        )
        engine = WitnessEngine(
            store=store,
            compiler=self.compiler,
            evaluator=self.evaluator,
            cache=self.cache,
            max_workers=self.max_workers,
        )
        engine.add_documents(documents)
        self._engines[fingerprint] = engine
        return engine, fingerprint, True

    @staticmethod
    def _row_passage(row: Any, rank: int) -> RetrievedPassage:
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
                "verified": True,
            },
        )
        return RetrievedPassage(
            passage_id=f"witness:{row.witness_id}",
            document_id=row.source_id,
            document_version=row.source_version,
            text=row.quote,
            start_offset=row.start_offset,
            end_offset=row.end_offset,
            score=1.0,
            rank=rank,
            evidence=(evidence,),
            metadata={
                "witness_id": row.witness_id,
                "predicate_id": row.predicate_id,
                "role": row.role.value,
                "bindings": {
                    name: binding.to_dict() for name, binding in row.bindings.items()
                },
                "_valid_from": row.valid_time,
                "_record_time": row.record_time,
                "exact_source_span": True,
            },
        )

    def retrieve(
        self,
        task: BenchmarkTask | Mapping[str, Any] | Any,
        documents: Sequence[SourceDocument | Mapping[str, Any] | Any],
    ) -> RetrievalResult:
        public = coerce_task(task)
        corpus = tuple(coerce_source_document(item) for item in documents)
        lookup_wall_started = perf_counter()
        cpu_started = process_time()
        fingerprint = corpus_fingerprint(corpus)
        result_key = (
            fingerprint,
            public.question,
            public.as_of,
            self.compiler.COMPILER_VERSION,
            self.evaluator.EVALUATOR_VERSION,
            self.counter_search,
        )
        maintained = self._result_cache.get(result_key) if self.materialized_query_views else None
        if maintained is not None:
            counters = dict(maintained.telemetry.counters)
            counters["query_result_cache_hits"] = counters.get(
                "query_result_cache_hits", 0.0
            ) + 1.0
            telemetry = replace(
                maintained.telemetry,
                cold=False,
                input_tokens=token_count(public.question),
                model_calls=0,
                embedding_calls=0,
                reranker_calls=0,
                tool_calls=0,
                cpu_time_ms=(process_time() - cpu_started) * 1000.0,
                latency_ms=(perf_counter() - lookup_wall_started) * 1000.0,
                # This is one exact-result memoization hit, not a replay of
                # every per-shard semantic-view lookup. Keep the two cache
                # mechanisms distinct in telemetry.
                cache_hits=maintained.telemetry.cache_hits,
                counters=counters,
            )
            metadata = dict(maintained.metadata)
            metadata["materialized_query_view_hit"] = True
            return replace(maintained, telemetry=telemetry, metadata=metadata)
        engine, fingerprint, _ = self._engine(corpus)
        record_axis = str(public.metadata.get("temporal_axis", "")).casefold() == "record_time"
        plan = engine.compile(
            public.question,
            as_of=public.as_of,
            recorded_as_of=public.as_of if record_axis else None,
        )
        primary = engine.execute(plan, principals=("benchmark",))
        executions = [primary]
        result = primary
        if self.counter_search and primary.rows:
            # The offline renderer cannot synthesize a final semantic answer, so
            # it selects the most query-overlapping verified row as a bounded
            # provisional claim.  Production deployments should generate claims
            # from the deterministic reduced result and validate their ledger.
            query_terms = set(public.question.casefold().split())
            candidate = max(
                primary.rows,
                key=lambda row: (
                    len(query_terms.intersection(row.quote.casefold().split())),
                    -len(row.quote),
                    row.witness_id,
                ),
            )
            provisional = ClaimLedgerEntry(
                claim_id="provisional-answer",
                text=candidate.quote,
                supporting_witness_ids=(candidate.witness_id,),
                status=ClaimStatus.SUPPORTED,
            )
            result = engine.execute(
                plan,
                principals=("benchmark",),
                claims=(provisional,),
            )
            executions.append(result)
        rows = (*result.rows, *result.counter_rows)
        passages = tuple(self._row_passage(row, rank) for rank, row in enumerate(rows, 1))

        blocks = engine.store.blocks(
            principals=("benchmark",),
            recorded_as_of=public.as_of if record_axis else None,
        )
        corpus_tokens = sum(token_count(block.content) for block in blocks)
        cache_misses = sum(item.telemetry.cache_misses for item in executions)
        cache_hits = sum(item.telemetry.cache_hits for item in executions)
        evaluator_calls = sum(item.telemetry.evaluator_calls for item in executions)
        elapsed_ms = sum(item.telemetry.elapsed_ms for item in executions)
        obligation_factor = cache_misses / len(blocks) if blocks else 0.0
        evaluated_tokens = round(corpus_tokens * obligation_factor)
        storage_bytes = sum(len(document.content.encode("utf-8")) for document in corpus)
        storage_bytes += sum(len(passage.text.encode("utf-8")) for passage in passages)
        coverage = result.coverage
        telemetry = RunTelemetry(
            # "Cold" is query/view coldness: a new program or changed shard
            # required evaluation.  Snapshot engine creation is recorded
            # separately because unchanged blocks may still reuse views.
            cold=bool(cache_misses),
            input_tokens=token_count(public.question) + evaluated_tokens,
            output_tokens=sum(token_count(passage.text) for passage in passages),
            model_calls=0,
            embedding_calls=0,
            reranker_calls=0,
            tool_calls=0,
            cpu_time_ms=(process_time() - cpu_started) * 1000.0,
            latency_ms=elapsed_ms,
            storage_bytes=storage_bytes,
            cache_hits=cache_hits,
            authorized_shards=coverage.authorized_shards,
            scanned_shards=coverage.scanned_shards,
            soundly_skipped_shards=coverage.soundly_skipped_shards,
            unresolved_shards=coverage.unresolved_shards,
            uncertain_shards=coverage.uncertain_shards,
            counters={
                "evaluator_calls": float(evaluator_calls),
                "cache_misses": float(cache_misses),
                "uncertain_shards": float(coverage.uncertain_shards),
                "conclusive_coverage": coverage.conclusive_fraction,
                "verified_witness_rows": float(len(result.rows)),
                "counter_witness_rows": float(len(result.counter_rows)),
                "scan_obligations": float(
                    sum(item.telemetry.scan_obligations for item in executions)
                ),
                "dependency_expansions": float(
                    sum(item.telemetry.dependency_expansions for item in executions)
                ),
            },
        )
        metadata = self.metadata
        metadata.update(
            {
                "corpus_fingerprint": fingerprint,
                "execution_fingerprint": result.telemetry.execution_fingerprint,
                "program_hash": plan.program_hash,
                "program": plan.to_dict(),
                "coverage_certificate": coverage.to_dict(),
                "unresolved": list(result.unresolved),
                "final_relation": result.final_relation,
            }
        )
        retrieval = RetrievalResult(
            system_name=self.name,
            task_id=public.task_id,
            passages=passages,
            telemetry=telemetry,
            metadata=metadata,
        )
        if self.materialized_query_views:
            self._result_cache[result_key] = retrieval
        return retrieval

    def run(
        self,
        task: BenchmarkTask | Mapping[str, Any] | Any,
        documents: Sequence[SourceDocument | Mapping[str, Any] | Any],
    ) -> SystemAnswer:
        public = coerce_task(task)
        retrieval = self.retrieve(public, documents)
        claims = tuple(
            Claim(
                claim_id=f"claim-{index}",
                text=passage.text,
                evidence_ids=passage.evidence_ids,
                factual=True,
                uncertainty=None,
                metadata={"extractive": True, "exact_source_span": True},
            )
            for index, passage in enumerate(retrieval.passages, 1)
        )
        coverage = dict(retrieval.metadata["coverage_certificate"])
        uncertainty = None
        if not retrieval.passages:
            unresolved = coverage.get("unresolved_shards", 0) + coverage.get(
                "uncertain_shards", 0
            )
            uncertainty = (
                f"No matching evidence was found; {unresolved} shards were unresolved or uncertain."
            )
        return SystemAnswer(
            system_name=self.name,
            task_id=public.task_id,
            answer={
                "mode": "verified_evidence_relation",
                "witness_count": len(retrieval.passages),
                "rows": [passage.text for passage in retrieval.passages],
            },
            claims=claims,
            evidence=tuple(
                evidence for passage in retrieval.passages for evidence in passage.evidence
            ),
            retrieved_passages=retrieval.passages,
            telemetry=retrieval.telemetry,
            uncertainty=uncertainty,
            coverage=coverage,
            metadata=dict(retrieval.metadata),
        )


class BinaryLexicalEvidenceEvaluator(LexicalEvidenceEvaluator):
    """Ablation that erases UNCERTAIN and therefore overstates negatives."""

    EVALUATOR_VERSION = "lexical-evaluator-binary-ablation-0.1.0"

    def evaluate(self, block: Any, predicate: Any, *, role: Any = None) -> LocalEvaluation:
        kwargs = {} if role is None else {"role": role}
        result = super().evaluate(block, predicate, **kwargs)
        if result.status is not EvaluationStatus.UNCERTAIN:
            return result
        if result.witnesses:
            return LocalEvaluation(
                EvaluationStatus.MATCH,
                result.witnesses,
                "binary ablation coerced UNCERTAIN with candidates to MATCH",
            )
        return LocalEvaluation(
            EvaluationStatus.NO_MATCH,
            (),
            "binary ablation coerced UNCERTAIN without candidates to NO_MATCH",
        )


class WitnessTopKPrefilterSystem(WitnessOfflineSystem):
    """Ablation that inserts an explicitly approximate hybrid top-k gate."""

    def __init__(self, *, prefilter_k: int, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if prefilter_k < 1:
            raise ValueError("prefilter_k must be positive")
        self.prefilter_k = int(prefilter_k)
        self.name = f"witness-hybrid-prefilter-k{self.prefilter_k}"
        self._prefilter = HybridRAGBaseline(
            top_k=self.prefilter_k, respect_as_of=False
        )

    @property
    def metadata(self) -> dict[str, Any]:
        output = super().metadata
        output.update(
            {
                "approximate_prefilter": "hybrid_bm25_tfidf",
                "prefilter_k": self.prefilter_k,
                "semantic_answer_invariance_guaranteed": False,
            }
        )
        return output

    def retrieve(
        self,
        task: BenchmarkTask | Mapping[str, Any] | Any,
        documents: Sequence[SourceDocument | Mapping[str, Any] | Any],
    ) -> RetrievalResult:
        public = coerce_task(task)
        corpus = tuple(coerce_source_document(item) for item in documents)
        gate = self._prefilter.retrieve(public, corpus)
        keys = {
            (passage.document_id, passage.document_version)
            for passage in gate.passages
        }
        selected = tuple(
            document
            for document in corpus
            if (document.document_id, document.version) in keys
        )
        result = super().retrieve(public, selected)
        approximate_skips = max(0, len(corpus) - len(selected))
        counters = dict(result.telemetry.counters)
        counters.update(
            {
                "approximate_prefilter_skips": float(approximate_skips),
                "prefilter_selected_documents": float(len(selected)),
            }
        )
        telemetry = replace(
            result.telemetry,
            input_tokens=result.telemetry.input_tokens + gate.telemetry.input_tokens,
            embedding_calls=result.telemetry.embedding_calls
            + gate.telemetry.embedding_calls,
            cpu_time_ms=result.telemetry.cpu_time_ms + gate.telemetry.cpu_time_ms,
            latency_ms=result.telemetry.latency_ms + gate.telemetry.latency_ms,
            storage_bytes=result.telemetry.storage_bytes + gate.telemetry.storage_bytes,
            authorized_shards=len(corpus),
            unresolved_shards=result.telemetry.unresolved_shards + approximate_skips,
            counters=counters,
        )
        metadata = dict(result.metadata)
        metadata.update(
            {
                "approximate_prefilter": True,
                "prefilter_k": self.prefilter_k,
                "approximately_skipped_shards": approximate_skips,
                "full_authorized_documents": len(corpus),
            }
        )
        return replace(result, telemetry=telemetry, metadata=metadata)


__all__ = [
    "BinaryLexicalEvidenceEvaluator",
    "WitnessOfflineSystem",
    "WitnessTopKPrefilterSystem",
]
