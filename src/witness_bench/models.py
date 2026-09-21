"""Canonical, dependency-free models shared by the benchmark components.

The benchmark intentionally does not depend on a validation framework.  The
``coerce_*`` helpers form the compatibility boundary: callers may provide these
dataclasses, plain mappings, or objects from the prototype execution engine.
Within the benchmark we use the dataclasses so serialization and comparisons
remain deterministic.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass, replace
from hashlib import sha256
import json
from typing import Any, Iterable, Mapping, Sequence


JSONScalar = str | int | float | bool | None


def _tuple_of_strings(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    return tuple(str(item) for item in value)


def _mapping(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return dict(value)
    if is_dataclass(value):
        return dict(asdict(value))
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        converted = to_dict()
        if isinstance(converted, Mapping):
            return dict(converted)
    raise TypeError(f"expected a mapping, got {type(value).__name__}")


def _read(value: Any, *names: str, default: Any = None) -> Any:
    """Read the first present mapping key or object attribute.

    Unlike ``a or b``, this preserves meaningful false-y values such as zero,
    an empty expected answer, or ``False``.
    """

    if isinstance(value, Mapping):
        for name in names:
            if name in value:
                return value[name]
        return default
    for name in names:
        if hasattr(value, name):
            return getattr(value, name)
    return default


def to_primitive(value: Any) -> Any:
    """Convert benchmark values into stable JSON-compatible primitives."""

    if is_dataclass(value):
        return to_primitive(asdict(value))
    if isinstance(value, Mapping):
        return {
            str(key): to_primitive(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (set, frozenset)):
        items = (to_primitive(item) for item in value)
        return sorted(items, key=lambda item: canonical_json(item))
    if isinstance(value, (tuple, list)):
        return [to_primitive(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def canonical_json(value: Any) -> str:
    return json.dumps(
        to_primitive(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def stable_hash(value: Any) -> str:
    return sha256(canonical_json(value).encode("utf-8")).hexdigest()


@dataclass(slots=True)
class EvidenceRef:
    """A ground-truth or retrieved span anchored in a source version."""

    evidence_id: str
    document_id: str
    document_version: str = "1"
    quote: str = ""
    start_offset: int = 0
    end_offset: int = 0
    kind: str = "fact"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.evidence_id = str(self.evidence_id)
        self.document_id = str(self.document_id)
        self.document_version = str(self.document_version)
        self.quote = str(self.quote)
        self.start_offset = int(self.start_offset)
        self.end_offset = int(self.end_offset or (self.start_offset + len(self.quote)))
        self.kind = str(self.kind)
        self.metadata = _mapping(self.metadata)
        if self.start_offset < 0 or self.end_offset < self.start_offset:
            raise ValueError("evidence offsets must satisfy 0 <= start <= end")

    @property
    def source_id(self) -> str:
        return self.document_id

    @property
    def source_version(self) -> str:
        return self.document_version


@dataclass(slots=True)
class SourceDocument:
    """Lossless benchmark corpus document.

    ``content`` is canonical; compatibility properties expose the common
    ``raw_content`` and ``text`` spellings without storing duplicate text.
    """

    document_id: str
    version: str = "1"
    source_type: str = "text"
    content: str = ""
    content_hash: str = ""
    record_time: str | None = None
    valid_from: str | None = None
    valid_to: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    permissions: tuple[str, ...] = ("benchmark",)
    evidence_spans: tuple[EvidenceRef, ...] = ()

    def __post_init__(self) -> None:
        self.document_id = str(self.document_id)
        self.version = str(self.version)
        self.source_type = str(self.source_type)
        self.content = str(self.content)
        actual_hash = sha256(self.content.encode("utf-8")).hexdigest()
        if self.content_hash and not str(self.content_hash).startswith(actual_hash):
            # Fixtures sometimes prefix hashes with ``sha256:``.  Validate only
            # exact hexadecimal hashes; opaque upstream identifiers are kept.
            supplied = str(self.content_hash)
            if len(supplied) == 64 and all(c in "0123456789abcdefABCDEF" for c in supplied):
                if supplied.lower() != actual_hash:
                    raise ValueError(f"content_hash does not match {self.document_id}")
        if not self.content_hash:
            self.content_hash = actual_hash
        else:
            self.content_hash = str(self.content_hash)
        self.metadata = _mapping(self.metadata)
        self.permissions = _tuple_of_strings(self.permissions)
        self.evidence_spans = tuple(
            coerce_evidence_ref(
                span,
                default_document_id=self.document_id,
                default_document_version=self.version,
            )
            for span in self.evidence_spans
        )

    @property
    def source_id(self) -> str:
        return self.document_id

    @property
    def document_version(self) -> str:
        return self.version

    @property
    def raw_content(self) -> str:
        return self.content

    @property
    def text(self) -> str:
        return self.content

    @property
    def timestamps(self) -> dict[str, str | None]:
        return {
            "record_time": self.record_time,
            "valid_from": self.valid_from,
            "valid_to": self.valid_to,
        }


@dataclass(slots=True)
class BenchmarkTask:
    """One benchmark question and its evidence-level gold annotations."""

    task_id: str
    category: str
    question: str
    corpus_versions: tuple[str, ...] = ()
    expected_answer: Any = None
    evidence_ids: tuple[str, ...] = ()
    decisive_evidence_ids: tuple[str, ...] = ()
    counter_evidence_ids: tuple[str, ...] = ()
    as_of: str | None = None
    answer_type: str = "structured"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.task_id = str(self.task_id)
        self.category = str(self.category)
        self.question = str(self.question)
        self.corpus_versions = _tuple_of_strings(self.corpus_versions)
        self.evidence_ids = _tuple_of_strings(self.evidence_ids)
        self.decisive_evidence_ids = _tuple_of_strings(self.decisive_evidence_ids)
        self.counter_evidence_ids = _tuple_of_strings(self.counter_evidence_ids)
        self.answer_type = str(self.answer_type)
        self.metadata = _mapping(self.metadata)

    @property
    def id(self) -> str:
        return self.task_id


@dataclass(slots=True)
class Claim:
    """A rendered claim and the evidence identifiers asserted to support it."""

    claim_id: str
    text: str
    evidence_ids: tuple[str, ...] = ()
    factual: bool = True
    uncertainty: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.claim_id = str(self.claim_id)
        self.text = str(self.text)
        self.evidence_ids = _tuple_of_strings(self.evidence_ids)
        self.factual = bool(self.factual)
        self.metadata = _mapping(self.metadata)


@dataclass(slots=True)
class RunTelemetry:
    """Per-query usage counters in implementation-independent units.

    CPU and latency are milliseconds.  Token counts use the benchmark's
    deterministic lexical token counter, not provider billing tokenization.
    """

    cold: bool = True
    input_tokens: int = 0
    output_tokens: int = 0
    model_calls: int = 0
    embedding_calls: int = 0
    reranker_calls: int = 0
    tool_calls: int = 0
    cpu_time_ms: float = 0.0
    latency_ms: float = 0.0
    storage_bytes: int = 0
    cache_hits: int = 0
    authorized_shards: int = 0
    scanned_shards: int = 0
    soundly_skipped_shards: int = 0
    unresolved_shards: int = 0
    uncertain_shards: int = 0
    counters: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in (
            "input_tokens",
            "output_tokens",
            "model_calls",
            "embedding_calls",
            "reranker_calls",
            "tool_calls",
            "storage_bytes",
            "cache_hits",
            "authorized_shards",
            "scanned_shards",
            "soundly_skipped_shards",
            "unresolved_shards",
            "uncertain_shards",
        ):
            setattr(self, name, int(getattr(self, name)))
        self.cpu_time_ms = float(self.cpu_time_ms)
        self.latency_ms = float(self.latency_ms)
        self.counters = {str(key): float(value) for key, value in _mapping(self.counters).items()}

    @property
    def warm(self) -> bool:
        return not self.cold

    @property
    def conclusively_evaluated_shards(self) -> int:
        return (
            max(0, self.scanned_shards - self.uncertain_shards)
            + self.soundly_skipped_shards
        )

    def with_updates(self, **updates: Any) -> "RunTelemetry":
        return replace(self, **updates)

    def add(self, other: "RunTelemetry") -> "RunTelemetry":
        """Return an aggregate telemetry object without mutating either run."""

        numeric = {
            name: getattr(self, name) + getattr(other, name)
            for name in (
                "input_tokens",
                "output_tokens",
                "model_calls",
                "embedding_calls",
                "reranker_calls",
                "tool_calls",
                "cpu_time_ms",
                "latency_ms",
                "storage_bytes",
                "cache_hits",
                "authorized_shards",
                "scanned_shards",
                "soundly_skipped_shards",
                "unresolved_shards",
                "uncertain_shards",
            )
        }
        keys = set(self.counters) | set(other.counters)
        counters = {key: self.counters.get(key, 0.0) + other.counters.get(key, 0.0) for key in keys}
        return RunTelemetry(cold=self.cold and other.cold, counters=counters, **numeric)


@dataclass(slots=True)
class RetrievedPassage:
    passage_id: str
    document_id: str
    document_version: str
    text: str
    start_offset: int
    end_offset: int
    score: float = 0.0
    rank: int = 0
    evidence: tuple[EvidenceRef, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.passage_id = str(self.passage_id)
        self.document_id = str(self.document_id)
        self.document_version = str(self.document_version)
        self.text = str(self.text)
        self.start_offset = int(self.start_offset)
        self.end_offset = int(self.end_offset)
        self.score = float(self.score)
        self.rank = int(self.rank)
        self.evidence = tuple(
            coerce_evidence_ref(
                item,
                default_document_id=self.document_id,
                default_document_version=self.document_version,
            )
            for item in self.evidence
        )
        self.metadata = _mapping(self.metadata)

    @property
    def evidence_ids(self) -> tuple[str, ...]:
        return tuple(item.evidence_id for item in self.evidence)


@dataclass(slots=True)
class RetrievalResult:
    system_name: str
    task_id: str
    passages: tuple[RetrievedPassage, ...] = ()
    telemetry: RunTelemetry = field(default_factory=RunTelemetry)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.system_name = str(self.system_name)
        self.task_id = str(self.task_id)
        self.passages = tuple(coerce_retrieved_passage(item) for item in self.passages)
        self.telemetry = coerce_telemetry(self.telemetry)
        self.metadata = _mapping(self.metadata)

    @property
    def evidence(self) -> tuple[EvidenceRef, ...]:
        seen: set[str] = set()
        output: list[EvidenceRef] = []
        for passage in self.passages:
            for item in passage.evidence:
                if item.evidence_id not in seen:
                    seen.add(item.evidence_id)
                    output.append(item)
        return tuple(output)

    @property
    def evidence_ids(self) -> tuple[str, ...]:
        return tuple(item.evidence_id for item in self.evidence)


@dataclass(slots=True)
class SystemAnswer:
    system_name: str
    task_id: str
    answer: Any = None
    claims: tuple[Claim, ...] = ()
    evidence: tuple[EvidenceRef, ...] = ()
    retrieved_passages: tuple[RetrievedPassage, ...] = ()
    telemetry: RunTelemetry = field(default_factory=RunTelemetry)
    uncertainty: str | None = None
    coverage: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.system_name = str(self.system_name)
        self.task_id = str(self.task_id)
        self.claims = tuple(coerce_claim(item, index=index) for index, item in enumerate(self.claims))
        self.evidence = tuple(coerce_evidence_ref(item) for item in self.evidence)
        self.retrieved_passages = tuple(
            coerce_retrieved_passage(item) for item in self.retrieved_passages
        )
        self.telemetry = coerce_telemetry(self.telemetry)
        self.coverage = _mapping(self.coverage)
        self.metadata = _mapping(self.metadata)

    @property
    def evidence_ids(self) -> tuple[str, ...]:
        seen: set[str] = set()
        output: list[str] = []
        for item in self.evidence:
            if item.evidence_id not in seen:
                seen.add(item.evidence_id)
                output.append(item.evidence_id)
        for passage in self.retrieved_passages:
            for evidence_id in passage.evidence_ids:
                if evidence_id not in seen:
                    seen.add(evidence_id)
                    output.append(evidence_id)
        return tuple(output)


def coerce_evidence_ref(
    value: Any,
    *,
    default_document_id: str = "",
    default_document_version: str = "1",
) -> EvidenceRef:
    if isinstance(value, EvidenceRef):
        return value
    quote = str(_read(value, "quote", "text", default=""))
    start = int(_read(value, "start_offset", "start", default=0) or 0)
    evidence_id = _read(value, "evidence_id", "witness_id", "id", default=None)
    document_id = str(
        _read(value, "document_id", "source_id", "doc_id", default=default_document_id)
    )
    version = str(
        _read(
            value,
            "document_version",
            "source_version",
            "version",
            default=default_document_version,
        )
    )
    if evidence_id is None:
        evidence_id = f"ev:{stable_hash([document_id, version, start, quote])[:16]}"
    return EvidenceRef(
        evidence_id=str(evidence_id),
        document_id=document_id,
        document_version=version,
        quote=quote,
        start_offset=start,
        end_offset=int(_read(value, "end_offset", "end", default=start + len(quote)) or 0),
        kind=str(_read(value, "kind", "polarity", default="fact")),
        metadata=_mapping(_read(value, "metadata", default={})),
    )


def coerce_source_document(value: Any) -> SourceDocument:
    if isinstance(value, SourceDocument):
        return value
    document_id = str(_read(value, "document_id", "source_id", "doc_id", "id", default=""))
    version = str(_read(value, "version", "document_version", "source_version", default="1"))
    metadata = _mapping(_read(value, "metadata", default={}))
    timestamps = _mapping(_read(value, "timestamps", default={}))
    spans = _read(value, "evidence_spans", "evidence", "witnesses", default=())
    return SourceDocument(
        document_id=document_id,
        version=version,
        source_type=str(_read(value, "source_type", "mime_type", default="text")),
        content=str(_read(value, "content", "raw_content", "text", default="")),
        content_hash=str(_read(value, "content_hash", default="")),
        record_time=_read(value, "record_time", default=timestamps.get("record_time")),
        valid_from=_read(value, "valid_from", "valid_time", default=timestamps.get("valid_from")),
        valid_to=_read(value, "valid_to", default=timestamps.get("valid_to")),
        metadata=metadata,
        permissions=_tuple_of_strings(_read(value, "permissions", default=("benchmark",))),
        evidence_spans=tuple(spans or ()),
    )


def coerce_task(value: Any) -> BenchmarkTask:
    if isinstance(value, BenchmarkTask):
        return value
    if isinstance(value, str):
        return BenchmarkTask(
            task_id=f"query:{stable_hash(value)[:16]}",
            category="unlabeled",
            question=value,
        )
    return BenchmarkTask(
        task_id=str(_read(value, "task_id", "id", default="")),
        category=str(_read(value, "category", "task_type", default="unknown")),
        question=str(_read(value, "question", "query", default="")),
        corpus_versions=_tuple_of_strings(_read(value, "corpus_versions", default=())),
        expected_answer=_read(value, "expected_answer", "gold_answer", "answer", default=None),
        evidence_ids=_tuple_of_strings(_read(value, "evidence_ids", "gold_evidence_ids", default=())),
        decisive_evidence_ids=_tuple_of_strings(
            _read(value, "decisive_evidence_ids", default=())
        ),
        counter_evidence_ids=_tuple_of_strings(
            _read(value, "counter_evidence_ids", default=())
        ),
        as_of=_read(value, "as_of", default=None),
        answer_type=str(_read(value, "answer_type", default="structured")),
        metadata=_mapping(_read(value, "metadata", default={})),
    )


def coerce_claim(value: Any, *, index: int = 0) -> Claim:
    if isinstance(value, Claim):
        return value
    if isinstance(value, str):
        return Claim(claim_id=f"claim-{index + 1}", text=value)
    supporting = _tuple_of_strings(
        _read(value, "evidence_ids", "citations", "supporting_witness_ids", default=())
    )
    counter = _tuple_of_strings(_read(value, "counter_witness_ids", default=()))
    return Claim(
        claim_id=str(_read(value, "claim_id", "id", default=f"claim-{index + 1}")),
        text=str(_read(value, "text", "claim", default="")),
        evidence_ids=tuple(dict.fromkeys((*supporting, *counter))),
        factual=bool(_read(value, "factual", default=True)),
        uncertainty=_read(value, "uncertainty", default=None),
        metadata=_mapping(_read(value, "metadata", default={})),
    )


def coerce_telemetry(value: Any) -> RunTelemetry:
    if isinstance(value, RunTelemetry):
        return value
    aliases = {
        "latency_ms": ("latency_ms", "elapsed_ms"),
        "tool_calls": ("tool_calls", "evaluator_calls"),
    }
    kwargs = {
        name: _read(value, *aliases.get(name, (name,)), default=default)
        for name, default in (
            ("cold", True),
            ("input_tokens", 0),
            ("output_tokens", 0),
            ("model_calls", 0),
            ("embedding_calls", 0),
            ("reranker_calls", 0),
            ("tool_calls", 0),
            ("cpu_time_ms", 0.0),
            ("latency_ms", 0.0),
            ("storage_bytes", 0),
            ("cache_hits", 0),
            ("authorized_shards", 0),
            ("scanned_shards", 0),
            ("soundly_skipped_shards", 0),
            ("unresolved_shards", 0),
            ("uncertain_shards", 0),
            ("counters", {}),
        )
    }
    counters = dict(kwargs["counters"] or {})
    for source_name in (
        "evaluator_calls",
        "cache_misses",
        "verifier_rejections",
        "scan_obligations",
        "dependency_expansions",
    ):
        source_value = _read(value, source_name, default=None)
        if source_value is not None:
            counters[source_name] = float(source_value)
    kwargs["counters"] = counters
    return RunTelemetry(**kwargs)


def coerce_retrieved_passage(value: Any) -> RetrievedPassage:
    if isinstance(value, RetrievedPassage):
        return value
    text = str(_read(value, "text", "content", "quote", default=""))
    start = int(_read(value, "start_offset", "start", default=0) or 0)
    return RetrievedPassage(
        passage_id=str(_read(value, "passage_id", "block_id", "id", default="")),
        document_id=str(_read(value, "document_id", "source_id", default="")),
        document_version=str(_read(value, "document_version", "version", default="1")),
        text=text,
        start_offset=start,
        end_offset=int(_read(value, "end_offset", "end", default=start + len(text)) or 0),
        score=float(_read(value, "score", default=0.0) or 0.0),
        rank=int(_read(value, "rank", default=0) or 0),
        evidence=tuple(_read(value, "evidence", "evidence_spans", default=()) or ()),
        metadata=_mapping(_read(value, "metadata", default={})),
    )


def coerce_retrieval_result(value: Any) -> RetrievalResult:
    if isinstance(value, RetrievalResult):
        return value
    return RetrievalResult(
        system_name=str(_read(value, "system_name", "system", default="unknown")),
        task_id=str(_read(value, "task_id", default="")),
        passages=tuple(_read(value, "passages", "retrieved_passages", "results", default=()) or ()),
        telemetry=coerce_telemetry(_read(value, "telemetry", default={})),
        metadata=_mapping(_read(value, "metadata", default={})),
    )


def coerce_system_answer(value: Any) -> SystemAnswer:
    if isinstance(value, SystemAnswer):
        return value
    retrieval = _read(value, "retrieval", default=None)
    passages = _read(value, "retrieved_passages", "passages", default=None)
    if passages is None and retrieval is not None:
        passages = _read(retrieval, "passages", default=())
    evidence = list(_read(value, "evidence", "witnesses", "rows", default=()) or ())
    evidence.extend(_read(value, "counter_rows", default=()) or ())
    coverage_value = _read(value, "coverage", "coverage_certificate", default={})
    answer_value = _read(
        value,
        "answer",
        "structured_answer",
        "result",
        "final_relation",
        default=None,
    )
    return SystemAnswer(
        system_name=str(_read(value, "system_name", "system", default="unknown")),
        task_id=str(_read(value, "task_id", default="")),
        answer=answer_value,
        claims=tuple(_read(value, "claims", "claim_ledger", default=()) or ()),
        evidence=tuple(evidence),
        retrieved_passages=tuple(passages or ()),
        telemetry=coerce_telemetry(
            _read(value, "telemetry", default=_read(retrieval, "telemetry", default={}))
        ),
        uncertainty=_read(value, "uncertainty", default=None),
        coverage=_mapping(coverage_value),
        metadata=_mapping(_read(value, "metadata", default={})),
    )


def corpus_fingerprint(documents: Iterable[Any]) -> str:
    docs = (coerce_source_document(item) for item in documents)
    values = sorted(
        (doc.document_id, doc.version, doc.content_hash)
        for doc in docs
    )
    return stable_hash(values)


__all__ = [
    "BenchmarkTask",
    "Claim",
    "EvidenceRef",
    "RetrievedPassage",
    "RetrievalResult",
    "RunTelemetry",
    "SourceDocument",
    "SystemAnswer",
    "canonical_json",
    "coerce_claim",
    "coerce_evidence_ref",
    "coerce_retrieval_result",
    "coerce_retrieved_passage",
    "coerce_source_document",
    "coerce_system_answer",
    "coerce_task",
    "coerce_telemetry",
    "corpus_fingerprint",
    "stable_hash",
    "to_primitive",
]
