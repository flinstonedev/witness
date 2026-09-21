"""Reproducible, offline retrieval baselines for the Witness benchmark.

These systems are deliberately dependency-free so the benchmark can run in an
air-gapped checkout.  They are *approximations* of named architecture families,
not claims of parity with a hosted embedding model, cross encoder, Microsoft's
GraphRAG, or an LLM agent.  Every result carries this qualification in metadata.

The useful property of these implementations is experimental control: corpus
partitioning, scores, tie-breaking, query budgets, evidence exposure, and cost
proxies are deterministic and inspectable.  A benchmark runner can replace any
class with a production adapter while retaining the models and metrics.
"""

from __future__ import annotations

from collections import Counter, defaultdict, deque
from dataclasses import replace
from datetime import datetime, timezone
import math
import re
from time import perf_counter, process_time
from typing import Any, Callable, Iterable, Mapping, Sequence

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
    stable_hash,
)


_TOKEN_RE = re.compile(r"[^\W_]+(?:['\N{RIGHT SINGLE QUOTATION MARK}-][^\W_]+)*", re.UNICODE)
_ENTITY_RE = re.compile(
    r"\b(?:[A-Z][\w'-]*)(?:\s+(?:[A-Z][\w'-]*|of|the|and)){0,3}\b"
)
_SENTENCE_RE = re.compile(r"\S(?:.*?)(?:[.!?](?=\s|$)|\n+|$)", re.DOTALL)

_STOPWORDS = frozenset(
    {
        "a",
        "about",
        "after",
        "all",
        "also",
        "an",
        "and",
        "any",
        "are",
        "as",
        "at",
        "be",
        "before",
        "by",
        "did",
        "do",
        "does",
        "for",
        "from",
        "had",
        "has",
        "have",
        "how",
        "in",
        "is",
        "it",
        "its",
        "of",
        "on",
        "or",
        "that",
        "the",
        "their",
        "this",
        "to",
        "was",
        "were",
        "what",
        "when",
        "where",
        "which",
        "who",
        "why",
        "with",
    }
)
_QUALIFIERS = frozenset(
    {
        "only",
        "except",
        "before",
        "after",
        "approximately",
        "at least",
        "may",
        "must",
        "proposed",
        "rejected",
        "hypothetical",
        "estimated",
        "current",
        "currently",
        "final",
        "corrected",
        "not",
        "no",
    }
)


def tokenize(text: str) -> tuple[str, ...]:
    """Deterministic Unicode word tokenization used by every offline baseline."""

    return tuple(match.group(0).casefold() for match in _TOKEN_RE.finditer(str(text)))


def token_count(text: str) -> int:
    return len(tokenize(text))


def _content_terms(text: str) -> tuple[str, ...]:
    return tuple(token for token in tokenize(text) if token not in _STOPWORDS and len(token) > 1)


def _parse_time(value: str | None) -> datetime | None:
    if not value:
        return None
    normalized = str(value).strip()
    if normalized.endswith("Z"):
        normalized = normalized[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        try:
            parsed = datetime.fromisoformat(normalized[:10])
        except ValueError:
            return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _active_as_of(passage: RetrievedPassage, as_of: str | None) -> bool:
    if not as_of:
        return True
    instant = _parse_time(as_of)
    if instant is None:
        return True
    start = _parse_time(passage.metadata.get("_valid_from"))
    end = _parse_time(passage.metadata.get("_valid_to"))
    return (start is None or start <= instant) and (end is None or instant < end)


def _passage_copy(passage: RetrievedPassage, *, score: float, rank: int) -> RetrievedPassage:
    return replace(passage, score=float(score), rank=int(rank))


def _ranked(
    passages: Iterable[RetrievedPassage],
    scores: Mapping[str, float],
    limit: int,
) -> list[RetrievedPassage]:
    ordered = sorted(
        passages,
        key=lambda item: (-float(scores.get(item.passage_id, 0.0)), item.passage_id),
    )[: max(0, int(limit))]
    return [
        _passage_copy(item, score=float(scores.get(item.passage_id, 0.0)), rank=rank)
        for rank, item in enumerate(ordered, start=1)
    ]


def _sentence_spans(text: str) -> list[tuple[int, int, str]]:
    spans: list[tuple[int, int, str]] = []
    for match in _SENTENCE_RE.finditer(text):
        raw = match.group(0)
        leading = len(raw) - len(raw.lstrip())
        value = raw.strip()
        if value:
            start = match.start() + leading
            spans.append((start, start + len(value), value))
    if not spans and text:
        spans.append((0, len(text), text))
    return spans


def _chunk_document(
    document: SourceDocument,
    *,
    chunk_tokens: int,
    overlap_tokens: int,
) -> list[RetrievedPassage]:
    matches = list(_TOKEN_RE.finditer(document.content))
    if not matches:
        ranges = [(0, len(document.content))]
    elif len(matches) <= chunk_tokens:
        ranges = [(0, len(document.content))]
    else:
        step = max(1, chunk_tokens - min(overlap_tokens, chunk_tokens - 1))
        ranges: list[tuple[int, int]] = []
        for token_start in range(0, len(matches), step):
            token_end = min(len(matches), token_start + chunk_tokens)
            start = matches[token_start].start()
            end = matches[token_end - 1].end()
            # Include punctuation immediately following the final token.
            while end < len(document.content) and document.content[end] in " .,!?:;\"')]}\n":
                end += 1
            ranges.append((start, end))
            if token_end == len(matches):
                break

    output: list[RetrievedPassage] = []
    for start, end in ranges:
        text = document.content[start:end]
        evidence: list[EvidenceRef] = []
        for item in document.evidence_spans:
            if item.start_offset < start or item.end_offset > end:
                continue
            # Do not silently propagate corrupt gold annotations into results.
            if item.quote and document.content[item.start_offset : item.end_offset] != item.quote:
                continue
            evidence.append(item)
        passage_id = f"{document.document_id}@{document.version}:{start}-{end}"
        metadata = dict(document.metadata)
        metadata.update(
            {
                "_source_type": document.source_type,
                "_content_hash": document.content_hash,
                "_record_time": document.record_time,
                "_valid_from": document.valid_from,
                "_valid_to": document.valid_to,
                "_permissions": document.permissions,
            }
        )
        output.append(
            RetrievedPassage(
                passage_id=passage_id,
                document_id=document.document_id,
                document_version=document.version,
                text=text,
                start_offset=start,
                end_offset=end,
                evidence=tuple(evidence),
                metadata=metadata,
            )
        )
    return output


def _extract_entities(passage: RetrievedPassage) -> frozenset[str]:
    provided = passage.metadata.get("entities", ())
    if isinstance(provided, str):
        provided = (provided,)
    entities = {str(item).casefold() for item in provided}
    for match in _ENTITY_RE.finditer(passage.text):
        entity = " ".join(match.group(0).casefold().split())
        if entity not in _STOPWORDS and len(entity) > 2:
            entities.add(entity)
    return frozenset(entities)


class OfflineCorpusIndex:
    """Shared fixed-block lexical index used by the controlled baselines."""

    def __init__(
        self,
        documents: Sequence[SourceDocument],
        *,
        chunk_tokens: int,
        overlap_tokens: int,
    ) -> None:
        self.documents = tuple(documents)
        self.passages = tuple(
            passage
            for document in self.documents
            for passage in _chunk_document(
                document,
                chunk_tokens=chunk_tokens,
                overlap_tokens=overlap_tokens,
            )
        )
        self.tokens: dict[str, tuple[str, ...]] = {
            passage.passage_id: tokenize(passage.text) for passage in self.passages
        }
        self.term_counts: dict[str, Counter[str]] = {
            passage_id: Counter(tokens) for passage_id, tokens in self.tokens.items()
        }
        document_frequency: Counter[str] = Counter()
        for counts in self.term_counts.values():
            document_frequency.update(counts.keys())
        count = max(1, len(self.passages))
        self.idf = {
            term: math.log((count + 1) / (frequency + 1)) + 1.0
            for term, frequency in document_frequency.items()
        }
        self.average_length = (
            sum(len(tokens) for tokens in self.tokens.values()) / count
            if self.passages
            else 0.0
        )
        self.entities = {passage.passage_id: _extract_entities(passage) for passage in self.passages}
        self._passage_by_id = {passage.passage_id: passage for passage in self.passages}
        self._entity_members: dict[str, set[str]] = defaultdict(set)
        for passage_id, entities in self.entities.items():
            for entity in entities:
                self._entity_members[entity].add(passage_id)
        self.storage_bytes = self._estimate_storage()

    def _estimate_storage(self) -> int:
        text_bytes = sum(len(item.text.encode("utf-8")) for item in self.passages)
        postings_bytes = sum(
            len(term.encode("utf-8")) + 16
            for counts in self.term_counts.values()
            for term in counts
        )
        graph_bytes = sum(len(entity.encode("utf-8")) + 16 for entity in self._entity_members)
        return text_bytes + postings_bytes + graph_bytes

    def vector_scores(
        self,
        query: str,
        passages: Sequence[RetrievedPassage] | None = None,
    ) -> dict[str, float]:
        candidates = self.passages if passages is None else passages
        query_counts = Counter(tokenize(query))
        query_vector = {
            term: (1.0 + math.log(count)) * self.idf.get(term, 1.0)
            for term, count in query_counts.items()
            if count > 0
        }
        query_norm = math.sqrt(sum(weight * weight for weight in query_vector.values()))
        scores: dict[str, float] = {}
        for passage in candidates:
            counts = self.term_counts[passage.passage_id]
            dot = 0.0
            norm = 0.0
            for term, count in counts.items():
                weight = (1.0 + math.log(count)) * self.idf.get(term, 1.0)
                norm += weight * weight
                dot += weight * query_vector.get(term, 0.0)
            denominator = query_norm * math.sqrt(norm)
            scores[passage.passage_id] = dot / denominator if denominator else 0.0
        return scores

    def bm25_scores(
        self,
        query: str,
        passages: Sequence[RetrievedPassage] | None = None,
        *,
        k1: float = 1.2,
        b: float = 0.75,
    ) -> dict[str, float]:
        candidates = self.passages if passages is None else passages
        terms = tokenize(query)
        population = max(1, len(self.passages))
        scores: dict[str, float] = {}
        for passage in candidates:
            counts = self.term_counts[passage.passage_id]
            length = len(self.tokens[passage.passage_id])
            score = 0.0
            for term in terms:
                frequency = counts.get(term, 0)
                if not frequency:
                    continue
                # Recover document frequency from the smoothed TF-IDF value.
                idf_tfidf = self.idf.get(term, 1.0)
                doc_frequency = max(
                    0.0,
                    ((population + 1) / math.exp(idf_tfidf - 1.0)) - 1,
                )
                inverse = math.log(1.0 + (population - doc_frequency + 0.5) / (doc_frequency + 0.5))
                normalization = frequency + k1 * (
                    1.0 - b + b * length / max(1.0, self.average_length)
                )
                score += inverse * (frequency * (k1 + 1.0)) / normalization
            scores[passage.passage_id] = score
        return scores

    def neighbors(self, passage_id: str) -> frozenset[str]:
        output: set[str] = set()
        for entity in self.entities.get(passage_id, ()):
            output.update(self._entity_members.get(entity, ()))
        output.discard(passage_id)
        return frozenset(output)

    def passage(self, passage_id: str) -> RetrievedPassage:
        return self._passage_by_id[passage_id]


Answerer = Callable[[BenchmarkTask, RetrievalResult], SystemAnswer | Any]


class OfflineBaseline:
    """Base class with deterministic indexing, telemetry, and extractive output."""

    family = "offline_baseline"
    name = "offline-baseline"
    limitations: tuple[str, ...] = (
        "This is a deterministic offline architecture proxy, not a hosted production system.",
    )

    def __init__(
        self,
        *,
        top_k: int = 8,
        chunk_tokens: int = 180,
        overlap_tokens: int = 30,
        respect_as_of: bool = True,
        answerer: Answerer | None = None,
        answerer_uses_model: bool = False,
    ) -> None:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        if chunk_tokens <= 0:
            raise ValueError("chunk_tokens must be positive")
        if overlap_tokens < 0 or overlap_tokens >= chunk_tokens:
            raise ValueError("overlap_tokens must satisfy 0 <= overlap < chunk_tokens")
        self.top_k = int(top_k)
        self.chunk_tokens = int(chunk_tokens)
        self.overlap_tokens = int(overlap_tokens)
        self.respect_as_of = bool(respect_as_of)
        self.answerer = answerer
        self.answerer_uses_model = bool(answerer_uses_model)
        self._indexes: dict[str, OfflineCorpusIndex] = {}

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "implementation": "stdlib_offline_deterministic_approximation",
            "production_equivalent": False,
            "limitations": list(self.limitations),
            "top_k": self.top_k,
            "chunk_tokens": self.chunk_tokens,
            "overlap_tokens": self.overlap_tokens,
            "respect_as_of": self.respect_as_of,
            "answer_backend": (
                "caller_supplied" if self.answerer is not None else "extractive_no_llm"
            ),
        }

    def _get_index(
        self, documents: Sequence[SourceDocument]
    ) -> tuple[OfflineCorpusIndex, str, bool]:
        fingerprint = corpus_fingerprint(documents)
        index = self._indexes.get(fingerprint)
        cold = index is None
        if index is None:
            index = OfflineCorpusIndex(
                documents,
                chunk_tokens=self.chunk_tokens,
                overlap_tokens=self.overlap_tokens,
            )
            self._indexes[fingerprint] = index
        return index, fingerprint, cold

    def _retrieve(
        self,
        index: OfflineCorpusIndex,
        task: BenchmarkTask,
        passages: Sequence[RetrievedPassage],
        telemetry: RunTelemetry,
    ) -> list[RetrievedPassage]:
        raise NotImplementedError

    def retrieve(
        self,
        task: BenchmarkTask | Mapping[str, Any] | Any,
        documents: Sequence[SourceDocument | Mapping[str, Any] | Any],
    ) -> RetrievalResult:
        benchmark_task = coerce_task(task)
        corpus = tuple(coerce_source_document(item) for item in documents)
        wall_start = perf_counter()
        cpu_start = process_time()
        index, fingerprint, cold = self._get_index(corpus)
        passages = tuple(
            item
            for item in index.passages
            if not self.respect_as_of or _active_as_of(item, benchmark_task.as_of)
        )
        corpus_tokens = sum(token_count(document.content) for document in corpus) if cold else 0
        telemetry = RunTelemetry(
            cold=cold,
            input_tokens=corpus_tokens + token_count(benchmark_task.question),
            storage_bytes=index.storage_bytes,
            cache_hits=0 if cold else 1,
            authorized_shards=len(passages),
            counters={
                "indexed_tokens": float(corpus_tokens),
                "lexically_scored_shards": float(len(passages)),
            },
        )
        selected = self._retrieve(index, benchmark_task, passages, telemetry)
        raw_passage_ranges = {
            item.passage_id: (item.start_offset, item.end_offset) for item in passages
        }
        # Derived wiki/community units are not full semantic evaluations of the
        # source shards they summarize.  Count only returned raw blocks toward
        # conclusive scan coverage; this also prevents synthesized + fallback
        # units from producing impossible coverage above 100%.
        telemetry.scanned_shards = len(
            {
                item.passage_id
                for item in selected
                if raw_passage_ranges.get(item.passage_id)
                == (item.start_offset, item.end_offset)
            }
        )
        # Lexical ranking is not conclusive semantic evaluation.  Non-selected
        # blocks are unresolved, not soundly skipped.
        telemetry.unresolved_shards = max(0, len(passages) - telemetry.scanned_shards)
        context_tokens = sum(token_count(item.text) for item in selected)
        telemetry.input_tokens += context_tokens
        telemetry.counters["retrieved_context_tokens"] = float(context_tokens)
        telemetry.cpu_time_ms = (process_time() - cpu_start) * 1000.0
        telemetry.latency_ms = (perf_counter() - wall_start) * 1000.0
        result_metadata = self.metadata
        result_metadata.update(
            {
                "corpus_fingerprint": fingerprint,
                "as_of_filter_applied": bool(self.respect_as_of and benchmark_task.as_of),
            }
        )
        return RetrievalResult(
            system_name=self.name,
            task_id=benchmark_task.task_id,
            passages=tuple(selected),
            telemetry=telemetry,
            metadata=result_metadata,
        )

    def _extractive_answer(
        self,
        task: BenchmarkTask,
        retrieval: RetrievalResult,
    ) -> SystemAnswer:
        evidence: list[EvidenceRef] = []
        claims: list[Claim] = []
        snippets: list[str] = []
        seen_evidence: set[str] = set()
        for index, passage in enumerate(retrieval.passages, start=1):
            sentence_spans = _sentence_spans(passage.text)
            relative_start, relative_end, sentence = sentence_spans[0] if sentence_spans else (
                0,
                len(passage.text),
                passage.text,
            )
            source_start = passage.start_offset + relative_start
            source_end = passage.start_offset + relative_end
            citation = EvidenceRef(
                evidence_id=f"retrieved:{stable_hash([passage.passage_id, source_start, sentence])[:16]}",
                document_id=passage.document_id,
                document_version=passage.document_version,
                quote=sentence,
                start_offset=source_start,
                end_offset=source_end,
                kind="retrieved_passage",
                metadata={"passage_id": passage.passage_id, "exact_source_span": True},
            )
            evidence.append(citation)
            seen_evidence.add(citation.evidence_id)
            for annotated in passage.evidence:
                if annotated.evidence_id not in seen_evidence:
                    seen_evidence.add(annotated.evidence_id)
                    evidence.append(annotated)
            snippets.append(sentence)
            claims.append(
                Claim(
                    claim_id=f"claim-{index}",
                    text=sentence,
                    evidence_ids=(citation.evidence_id,),
                    metadata={
                        "source_anchored": True,
                        "source_id": passage.document_id,
                        "start_offset": source_start,
                        "end_offset": source_end,
                    },
                )
            )
        answer: dict[str, Any] = {
            "mode": "extractive",
            "snippets": snippets,
            "note": "No generative answer backend was configured.",
        }
        uncertainty = None if snippets else "No passage was retrieved."
        coverage = {
            "authorized_shards": retrieval.telemetry.authorized_shards,
            "scanned_shards": retrieval.telemetry.scanned_shards,
            "soundly_skipped_shards": retrieval.telemetry.soundly_skipped_shards,
            "unresolved_shards": retrieval.telemetry.unresolved_shards,
            "corpus_snapshot": retrieval.metadata.get("corpus_fingerprint"),
        }
        return SystemAnswer(
            system_name=self.name,
            task_id=task.task_id,
            answer=answer,
            claims=tuple(claims),
            evidence=tuple(evidence),
            retrieved_passages=retrieval.passages,
            telemetry=retrieval.telemetry,
            uncertainty=uncertainty,
            coverage=coverage,
            metadata=dict(retrieval.metadata),
        )

    def run(
        self,
        task: BenchmarkTask | Mapping[str, Any] | Any,
        documents: Sequence[SourceDocument | Mapping[str, Any] | Any],
    ) -> SystemAnswer:
        benchmark_task = coerce_task(task)
        retrieval = self.retrieve(benchmark_task, documents)
        if self.answerer is None:
            answer = self._extractive_answer(benchmark_task, retrieval)
        else:
            generated = self.answerer(benchmark_task, retrieval)
            if isinstance(generated, SystemAnswer):
                answer = generated
                if not answer.retrieved_passages:
                    answer.retrieved_passages = retrieval.passages
                if not answer.evidence:
                    answer.evidence = retrieval.evidence
                answer.telemetry = retrieval.telemetry.add(answer.telemetry)
            else:
                answer = SystemAnswer(
                    system_name=self.name,
                    task_id=benchmark_task.task_id,
                    answer=generated,
                    evidence=retrieval.evidence,
                    retrieved_passages=retrieval.passages,
                    telemetry=retrieval.telemetry,
                    metadata=dict(retrieval.metadata),
                )
        if self.answerer_uses_model:
            answer.telemetry.model_calls += 1
        answer.telemetry.output_tokens = token_count(str(answer.answer))
        return answer


class VectorRAGBaseline(OfflineBaseline):
    family = "classic_vector_rag"
    name = "vector-rag-tfidf"
    limitations = (
        "TF-IDF cosine is a lexical vector proxy; it is not a neural embedding model.",
        "Top-k selection can permanently exclude relevant or countervailing evidence.",
        "The default renderer is extractive and is not an LLM answer generator.",
    )

    def _retrieve(self, index, task, passages, telemetry):  # type: ignore[no-untyped-def]
        scores = index.vector_scores(task.question, passages)
        telemetry.embedding_calls += 1 + (len(passages) if telemetry.cold else 0)
        return _ranked(passages, scores, self.top_k)


class HybridRAGBaseline(OfflineBaseline):
    family = "hybrid_bm25_vector_rag"
    name = "hybrid-rag-bm25-tfidf"
    limitations = (
        "The vector arm uses TF-IDF rather than a neural embedding.",
        "Score fusion is min-max weighted fusion and is corpus-sensitive.",
        "Top-k selection is an approximate semantic gate without a completeness guarantee.",
    )

    def __init__(self, *, vector_weight: float = 0.5, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if not 0.0 <= vector_weight <= 1.0:
            raise ValueError("vector_weight must be in [0, 1]")
        self.vector_weight = float(vector_weight)

    @property
    def metadata(self) -> dict[str, Any]:
        output = super().metadata
        output["vector_weight"] = self.vector_weight
        output["fusion"] = "min_max_weighted"
        return output

    @staticmethod
    def _normalize(scores: Mapping[str, float]) -> dict[str, float]:
        if not scores:
            return {}
        low, high = min(scores.values()), max(scores.values())
        if high <= low:
            return {key: (1.0 if high > 0 else 0.0) for key in scores}
        return {key: (value - low) / (high - low) for key, value in scores.items()}

    def fused_scores(
        self,
        index: OfflineCorpusIndex,
        query: str,
        passages: Sequence[RetrievedPassage],
    ) -> dict[str, float]:
        vectors = self._normalize(index.vector_scores(query, passages))
        bm25 = self._normalize(index.bm25_scores(query, passages))
        return {
            passage.passage_id: (
                self.vector_weight * vectors.get(passage.passage_id, 0.0)
                + (1.0 - self.vector_weight) * bm25.get(passage.passage_id, 0.0)
            )
            for passage in passages
        }

    def _retrieve(self, index, task, passages, telemetry):  # type: ignore[no-untyped-def]
        scores = self.fused_scores(index, task.question, passages)
        telemetry.embedding_calls += 1 + (len(passages) if telemetry.cold else 0)
        return _ranked(passages, scores, self.top_k)


def _longest_ordered_overlap(query: Sequence[str], passage: Sequence[str]) -> float:
    if not query or not passage:
        return 0.0
    previous = [0] * (len(passage) + 1)
    longest = 0
    for query_token in query:
        current = [0]
        for index, passage_token in enumerate(passage, start=1):
            if query_token == passage_token:
                value = previous[index - 1] + 1
                current.append(value)
                longest = max(longest, value)
            else:
                current.append(0)
        previous = current
    return longest / len(query)


class RerankedRAGBaseline(HybridRAGBaseline):
    family = "reranked_rag"
    name = "reranked-rag-heuristic"
    limitations = (
        "The reranker is an inspectable lexical/qualifier heuristic, not a cross encoder or LLM.",
        "A candidate gate runs before reranking, so missed evidence cannot be recovered.",
        "Reranker scores are useful for controlled ablations, not production ranking claims.",
    )

    def __init__(self, *, candidate_k: int = 40, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if candidate_k < self.top_k:
            raise ValueError("candidate_k must be at least top_k")
        self.candidate_k = int(candidate_k)

    @property
    def metadata(self) -> dict[str, Any]:
        output = super().metadata
        output.update({"candidate_k": self.candidate_k, "reranker": "lexical_qualifier_heuristic"})
        return output

    def _retrieve(self, index, task, passages, telemetry):  # type: ignore[no-untyped-def]
        fused = self.fused_scores(index, task.question, passages)
        candidates = _ranked(passages, fused, self.candidate_k)
        query_tokens = _content_terms(task.question)
        query_set = set(query_tokens)
        query_numbers = {term for term in tokenize(task.question) if any(ch.isdigit() for ch in term)}
        query_qualifiers = {phrase for phrase in _QUALIFIERS if phrase in task.question.casefold()}
        reranked: dict[str, float] = {}
        for passage in candidates:
            passage_tokens = _content_terms(passage.text)
            passage_set = set(passage_tokens)
            coverage = len(query_set & passage_set) / max(1, len(query_set))
            ordered = _longest_ordered_overlap(query_tokens, passage_tokens)
            numbers = (
                len(query_numbers & passage_set) / len(query_numbers) if query_numbers else 0.0
            )
            passage_lower = passage.text.casefold()
            qualifier_match = (
                len({item for item in query_qualifiers if item in passage_lower})
                / len(query_qualifiers)
                if query_qualifiers
                else 0.0
            )
            exact = 1.0 if task.question.casefold() in passage_lower else 0.0
            reranked[passage.passage_id] = (
                0.35 * fused.get(passage.passage_id, 0.0)
                + 0.30 * coverage
                + 0.15 * ordered
                + 0.10 * numbers
                + 0.07 * qualifier_match
                + 0.03 * exact
            )
        telemetry.embedding_calls += 1 + (len(passages) if telemetry.cold else 0)
        telemetry.reranker_calls += len(candidates)
        telemetry.counters["reranker_candidates"] = float(len(candidates))
        return _ranked(candidates, reranked, self.top_k)


class GraphRAGBaseline(HybridRAGBaseline):
    family = "graphrag"
    limitations = (
        "This is a lexical named-entity co-occurrence graph, not Microsoft's GraphRAG implementation.",
        "Entity extraction, graph edges, and global communities are query-independent and lossy.",
        "Local seed retrieval can prevent relevant graph regions from being visited.",
        "No model-generated community summaries are used; global scoring is an extractive proxy.",
    )

    def __init__(
        self,
        *,
        mode: str = "local",
        seed_k: int = 3,
        graph_hops: int = 2,
        community_k: int = 2,
        **kwargs: Any,
    ) -> None:
        if mode not in {"local", "global"}:
            raise ValueError("mode must be 'local' or 'global'")
        super().__init__(**kwargs)
        self.mode = mode
        self.seed_k = int(seed_k)
        self.graph_hops = int(graph_hops)
        self.community_k = int(community_k)
        self.name = f"graphrag-{mode}-approx"

    @property
    def metadata(self) -> dict[str, Any]:
        output = super().metadata
        output.update(
            {
                "graph_mode": self.mode,
                "seed_k": self.seed_k,
                "graph_hops": self.graph_hops,
                "community_k": self.community_k,
                "graph_backend": "named_entity_cooccurrence",
            }
        )
        return output

    def _local(self, index, task, passages, fused):  # type: ignore[no-untyped-def]
        active_ids = {passage.passage_id for passage in passages}
        seeds = _ranked(passages, fused, min(self.seed_k, self.top_k))
        distance = {passage.passage_id: 0 for passage in seeds}
        queue = deque(passage.passage_id for passage in seeds)
        while queue:
            current = queue.popleft()
            if distance[current] >= self.graph_hops:
                continue
            for neighbor in sorted(index.neighbors(current)):
                if neighbor not in active_ids or neighbor in distance:
                    continue
                distance[neighbor] = distance[current] + 1
                queue.append(neighbor)
        candidates = [index.passage(passage_id) for passage_id in distance]
        scores = {
            passage.passage_id: fused.get(passage.passage_id, 0.0)
            + 0.25 / (1 + distance[passage.passage_id])
            for passage in candidates
        }
        return _ranked(candidates, scores, self.top_k)

    def _components(
        self, index: OfflineCorpusIndex, passages: Sequence[RetrievedPassage]
    ) -> list[list[RetrievedPassage]]:
        active = {passage.passage_id for passage in passages}
        remaining = set(active)
        components: list[list[RetrievedPassage]] = []
        while remaining:
            seed = min(remaining)
            queue = deque([seed])
            remaining.remove(seed)
            member_ids: list[str] = []
            while queue:
                current = queue.popleft()
                member_ids.append(current)
                for neighbor in sorted(index.neighbors(current)):
                    if neighbor in remaining:
                        remaining.remove(neighbor)
                        queue.append(neighbor)
            components.append([index.passage(item) for item in member_ids])
        return components

    def _global(self, index, task, passages, fused):  # type: ignore[no-untyped-def]
        query_terms = set(_content_terms(task.question))
        components = self._components(index, passages)
        component_scores: list[tuple[float, str, list[RetrievedPassage]]] = []
        for component in components:
            terms = set(token for passage in component for token in _content_terms(passage.text))
            coverage = len(query_terms & terms) / max(1, len(query_terms))
            max_retrieval = max((fused.get(item.passage_id, 0.0) for item in component), default=0.0)
            representative = min(item.passage_id for item in component)
            component_scores.append((0.6 * coverage + 0.4 * max_retrieval, representative, component))
        chosen = sorted(component_scores, key=lambda item: (-item[0], item[1]))[: self.community_k]
        candidate_scores: dict[str, float] = {}
        candidates: list[RetrievedPassage] = []
        for component_score, _, component in chosen:
            for passage in component:
                candidates.append(passage)
                candidate_scores[passage.passage_id] = (
                    0.55 * component_score + 0.45 * fused.get(passage.passage_id, 0.0)
                )
        return _ranked(candidates, candidate_scores, self.top_k)

    def _retrieve(self, index, task, passages, telemetry):  # type: ignore[no-untyped-def]
        fused = self.fused_scores(index, task.question, passages)
        telemetry.embedding_calls += 1 + (len(passages) if telemetry.cold else 0)
        telemetry.counters["graph_nodes"] = float(len(passages))
        telemetry.counters["graph_entity_keys"] = float(len(index._entity_members))
        if self.mode == "local":
            selected = self._local(index, task, passages, fused)
        else:
            selected = self._global(index, task, passages, fused)
        telemetry.counters["graph_visited_nodes"] = float(len(selected))
        return selected


def _truncate_passage(passage: RetrievedPassage, budget: int) -> RetrievedPassage | None:
    if budget <= 0:
        return None
    matches = list(_TOKEN_RE.finditer(passage.text))
    if len(matches) <= budget:
        return passage
    end_relative = matches[budget - 1].end()
    text = passage.text[:end_relative]
    absolute_end = passage.start_offset + end_relative
    evidence = tuple(item for item in passage.evidence if item.end_offset <= absolute_end)
    return replace(passage, text=text, end_offset=absolute_end, evidence=evidence)


class LongContextBaseline(OfflineBaseline):
    family = "long_context"
    name = "long-context-budgeted"
    limitations = (
        "The context window is modeled with benchmark lexical tokens, not provider tokenizer tokens.",
        "Documents are inserted in stable corpus order; no lost-in-the-middle model behavior is simulated.",
        "The default renderer is extractive and therefore understates generative synthesis capability.",
    )

    def __init__(self, *, context_token_budget: int = 4096, **kwargs: Any) -> None:
        if context_token_budget <= 0:
            raise ValueError("context_token_budget must be positive")
        kwargs.setdefault("overlap_tokens", 0)
        super().__init__(**kwargs)
        self.context_token_budget = int(context_token_budget)

    @property
    def metadata(self) -> dict[str, Any]:
        output = super().metadata
        output.update(
            {"context_token_budget": self.context_token_budget, "ordering": "corpus_order"}
        )
        return output

    def _retrieve(self, index, task, passages, telemetry):  # type: ignore[no-untyped-def]
        del index, task
        remaining = self.context_token_budget
        selected: list[RetrievedPassage] = []
        for passage in passages:
            if remaining <= 0:
                break
            candidate = _truncate_passage(passage, remaining)
            if candidate is None:
                break
            selected.append(_passage_copy(candidate, score=1.0, rank=len(selected) + 1))
            remaining -= token_count(candidate.text)
        telemetry.counters["context_budget_tokens"] = float(self.context_token_budget)
        telemetry.counters["unused_context_tokens"] = float(max(0, remaining))
        return selected


class AgenticRAGBaseline(HybridRAGBaseline):
    family = "agentic_rag"
    name = "agentic-rag-iterative-offline"
    limitations = (
        "Query reformulation is a deterministic rare-term/entity heuristic, not an LLM agent.",
        "The fixed tool-call budget can terminate before all dependent hops are discovered.",
        "Every iteration still uses top-k retrieval and has no exhaustive evidence guarantee.",
    )

    def __init__(
        self,
        *,
        tool_call_budget: int = 4,
        results_per_call: int = 4,
        expansion_terms: int = 5,
        **kwargs: Any,
    ) -> None:
        if tool_call_budget <= 0 or results_per_call <= 0:
            raise ValueError("agentic budgets must be positive")
        super().__init__(**kwargs)
        self.tool_call_budget = int(tool_call_budget)
        self.results_per_call = int(results_per_call)
        self.expansion_terms = int(expansion_terms)

    @property
    def metadata(self) -> dict[str, Any]:
        output = super().metadata
        output.update(
            {
                "tool_call_budget": self.tool_call_budget,
                "results_per_call": self.results_per_call,
                "query_reformulator": "rare_terms_and_entities",
            }
        )
        return output

    def _new_terms(
        self,
        index: OfflineCorpusIndex,
        passages: Sequence[RetrievedPassage],
        used: set[str],
    ) -> list[str]:
        candidates: dict[str, float] = {}
        for passage in passages:
            for entity in index.entities.get(passage.passage_id, ()):
                if entity not in used:
                    candidates[entity] = max(candidates.get(entity, 0.0), 4.0)
            for term in _content_terms(passage.text):
                if term not in used:
                    candidates[term] = max(candidates.get(term, 0.0), index.idf.get(term, 1.0))
        return [
            term
            for term, _ in sorted(candidates.items(), key=lambda item: (-item[1], item[0]))[
                : self.expansion_terms
            ]
        ]

    def _retrieve(self, index, task, passages, telemetry):  # type: ignore[no-untyped-def]
        query = task.question
        used_terms = set(_content_terms(query))
        seen: dict[str, tuple[RetrievedPassage, float]] = {}
        active = list(passages)
        for iteration in range(self.tool_call_budget):
            scores = self.fused_scores(index, query, active)
            hits = _ranked(active, scores, self.results_per_call)
            for hit in hits:
                score = hit.score + 0.05 / (iteration + 1)
                previous = seen.get(hit.passage_id)
                if previous is None or score > previous[1]:
                    seen[hit.passage_id] = (hit, score)
            telemetry.tool_calls += 1
            telemetry.embedding_calls += 1
            new_terms = self._new_terms(index, hits, used_terms)
            if not new_terms:
                break
            used_terms.update(new_terms)
            query = f"{task.question} {' '.join(new_terms)}"
            active = [item for item in passages if item.passage_id not in seen]
            if not active:
                break
        if telemetry.cold:
            telemetry.embedding_calls += len(passages)
        telemetry.counters["agent_iterations"] = float(telemetry.tool_calls)
        candidates = [item for item, _ in seen.values()]
        scores = {passage_id: score for passage_id, (_, score) in seen.items()}
        return _ranked(candidates, scores, self.top_k)


def _topic_for_passage(passage: RetrievedPassage) -> str:
    provided = passage.metadata.get("wiki_topic") or passage.metadata.get("topic")
    if provided:
        return str(provided).casefold()
    entities = sorted(_extract_entities(passage))
    if entities:
        return entities[0]
    terms = Counter(_content_terms(passage.text))
    if terms:
        return min(terms, key=lambda term: (-terms[term], term))
    return "miscellaneous"


class LLMWikiBaseline(HybridRAGBaseline):
    family = "llm_maintained_wiki"
    limitations = (
        "Wiki synthesis is deterministic extractive compression, not an LLM-maintained wiki.",
        "Topic assignment uses metadata, named entities, or lexical frequency and can fragment concepts.",
        "Only selected source sentences survive synthesis; omitted details cannot affect wiki-only answers.",
        "The raw-fallback variant is separately labeled and retains a top-k retrieval gate.",
    )

    def __init__(
        self,
        *,
        raw_fallback: bool = False,
        sentences_per_source: int = 1,
        wiki_top_k: int | None = None,
        raw_fallback_k: int = 3,
        **kwargs: Any,
    ) -> None:
        if sentences_per_source <= 0:
            raise ValueError("sentences_per_source must be positive")
        super().__init__(**kwargs)
        self.raw_fallback = bool(raw_fallback)
        self.sentences_per_source = int(sentences_per_source)
        self.wiki_top_k = int(wiki_top_k or self.top_k)
        self.raw_fallback_k = int(raw_fallback_k)
        self.name = "llm-wiki-extractive" + ("-raw-fallback" if raw_fallback else "")
        self._wiki_pages: dict[str, tuple[RetrievedPassage, ...]] = {}

    @property
    def metadata(self) -> dict[str, Any]:
        output = super().metadata
        output.update(
            {
                "raw_fallback": self.raw_fallback,
                "sentences_per_source": self.sentences_per_source,
                "synthesis": "query_independent_extractive_proxy",
                "model_call_accounting": "one architectural proxy call per cold wiki page",
            }
        )
        return output

    def _build_wiki(
        self, fingerprint: str, passages: Sequence[RetrievedPassage]
    ) -> tuple[RetrievedPassage, ...]:
        cached = self._wiki_pages.get(fingerprint)
        if cached is not None:
            return cached
        topics: dict[str, list[RetrievedPassage]] = defaultdict(list)
        for passage in passages:
            topics[_topic_for_passage(passage)].append(passage)
        pages: list[RetrievedPassage] = []
        for topic in sorted(topics):
            sections: list[str] = []
            included_evidence: list[EvidenceRef] = []
            source_ids: list[str] = []
            for passage in sorted(topics[topic], key=lambda item: item.passage_id):
                spans = _sentence_spans(passage.text)[: self.sentences_per_source]
                selected = " ".join(sentence for _, _, sentence in spans)
                if not selected:
                    continue
                sections.append(selected)
                source_ids.append(passage.document_id)
                selected_ranges = [
                    (passage.start_offset + start, passage.start_offset + end)
                    for start, end, _ in spans
                ]
                included_evidence.extend(
                    item
                    for item in passage.evidence
                    if any(start <= item.start_offset and item.end_offset <= end for start, end in selected_ranges)
                )
            text = "\n".join(sections)
            if not text:
                continue
            page_id = f"wiki:{stable_hash([topic, fingerprint])[:16]}"
            pages.append(
                RetrievedPassage(
                    passage_id=page_id,
                    document_id=f"wiki:{topic}",
                    document_version=fingerprint[:12],
                    text=text,
                    start_offset=0,
                    end_offset=len(text),
                    evidence=tuple(included_evidence),
                    metadata={
                        "synthetic": True,
                        "topic": topic,
                        "source_ids": tuple(source_ids),
                    },
                )
            )
        cached = tuple(pages)
        self._wiki_pages[fingerprint] = cached
        return cached

    @staticmethod
    def _score_pages(query: str, pages: Sequence[RetrievedPassage]) -> dict[str, float]:
        query_counts = Counter(_content_terms(query))
        scores: dict[str, float] = {}
        for page in pages:
            counts = Counter(_content_terms(page.text))
            overlap = sum(min(count, counts.get(term, 0)) for term, count in query_counts.items())
            denominator = math.sqrt(sum(query_counts.values()) * max(1, sum(counts.values())))
            scores[page.passage_id] = overlap / denominator if denominator else 0.0
        return scores

    def _retrieve(self, index, task, passages, telemetry):  # type: ignore[no-untyped-def]
        fingerprint = stable_hash(
            sorted((item.passage_id, item.metadata.get("_content_hash")) for item in passages)
        )
        pages = self._build_wiki(fingerprint, passages)
        wiki_scores = self._score_pages(task.question, pages)
        raw_slots = min(self.raw_fallback_k, self.top_k) if self.raw_fallback else 0
        wiki_limit = min(self.wiki_top_k, self.top_k - raw_slots)
        selected = _ranked(pages, wiki_scores, wiki_limit)
        telemetry.model_calls += len(pages) if telemetry.cold else 0
        telemetry.counters["wiki_pages"] = float(len(pages))
        telemetry.counters["wiki_synthesis_source_passages"] = float(len(passages))
        if self.raw_fallback:
            raw_scores = self.fused_scores(index, task.question, passages)
            raw = _ranked(passages, raw_scores, self.raw_fallback_k)
            telemetry.embedding_calls += 1 + (len(passages) if telemetry.cold else 0)
            combined: list[RetrievedPassage] = []
            seen: set[str] = set()
            for passage in (*selected, *raw):
                if passage.passage_id not in seen:
                    seen.add(passage.passage_id)
                    combined.append(replace(passage, rank=len(combined) + 1))
            selected = combined[: self.top_k]
        return selected[: self.top_k]


# Names without the ``Baseline`` suffix keep adapters and configuration files
# pleasant while preserving explicit class names in documentation.
VectorRAG = VectorRAGBaseline
HybridRAG = HybridRAGBaseline
RerankedRAG = RerankedRAGBaseline
GraphRAG = GraphRAGBaseline
LongContext = LongContextBaseline
AgenticRAG = AgenticRAGBaseline
SynthesizedWiki = LLMWikiBaseline


def default_baselines(*, top_k: int = 8) -> tuple[OfflineBaseline, ...]:
    """Return controlled variants covering every requested baseline family."""

    return (
        VectorRAGBaseline(top_k=top_k),
        HybridRAGBaseline(top_k=top_k),
        RerankedRAGBaseline(top_k=top_k, candidate_k=max(40, top_k)),
        GraphRAGBaseline(top_k=top_k, mode="local"),
        GraphRAGBaseline(top_k=top_k, mode="global"),
        LongContextBaseline(top_k=top_k),
        AgenticRAGBaseline(top_k=top_k),
        LLMWikiBaseline(top_k=top_k),
        LLMWikiBaseline(top_k=top_k, raw_fallback=True),
    )


BASELINE_REGISTRY: dict[str, type[OfflineBaseline]] = {
    "vector": VectorRAGBaseline,
    "hybrid": HybridRAGBaseline,
    "reranked": RerankedRAGBaseline,
    "graphrag": GraphRAGBaseline,
    "long_context": LongContextBaseline,
    "agentic": AgenticRAGBaseline,
    "wiki": LLMWikiBaseline,
}


def make_baseline(kind: str, **kwargs: Any) -> OfflineBaseline:
    try:
        baseline = BASELINE_REGISTRY[str(kind).casefold()]
    except KeyError as exc:
        choices = ", ".join(sorted(BASELINE_REGISTRY))
        raise ValueError(f"unknown baseline {kind!r}; choose one of: {choices}") from exc
    return baseline(**kwargs)


__all__ = [
    "AgenticRAG",
    "AgenticRAGBaseline",
    "BASELINE_REGISTRY",
    "GraphRAG",
    "GraphRAGBaseline",
    "HybridRAG",
    "HybridRAGBaseline",
    "LLMWikiBaseline",
    "LongContext",
    "LongContextBaseline",
    "OfflineBaseline",
    "OfflineCorpusIndex",
    "RerankedRAG",
    "RerankedRAGBaseline",
    "SynthesizedWiki",
    "VectorRAG",
    "VectorRAGBaseline",
    "default_baselines",
    "make_baseline",
    "token_count",
    "tokenize",
]
