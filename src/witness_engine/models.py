"""Typed, serializable contracts for the Witness execution engine.

The models deliberately use only JSON-compatible values.  Generated
interpretations are represented separately from the exact source span that
supports them, making it possible to mechanically re-check every row.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json
import math
from typing import Any, Iterable, Mapping


JSONScalar = str | int | float | bool | None
JSONValue = JSONScalar | list["JSONValue"] | dict[str, "JSONValue"]


class ValidationError(ValueError):
    """Raised when compiler output or an evidence object is malformed."""


class EvaluationStatus(str, Enum):
    MATCH = "MATCH"
    NO_MATCH = "NO_MATCH"
    UNCERTAIN = "UNCERTAIN"


class EvidenceRole(str, Enum):
    POSITIVE = "positive"
    COUNTER = "counter"


class ClaimStatus(str, Enum):
    SUPPORTED = "supported"
    REFUTED = "refuted"
    QUALIFIED = "qualified"
    UNRESOLVED = "unresolved"


def canonical_json(value: Any) -> str:
    """Return a stable encoding suitable for hashes and cache keys."""

    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def stable_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def content_hash(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _strings(values: Iterable[Any], name: str) -> tuple[str, ...]:
    if isinstance(values, str):
        raise ValidationError(f"{name} must be a list of strings, not a string")
    result = tuple(values)
    if not all(isinstance(value, str) and value for value in result):
        raise ValidationError(f"{name} must contain non-empty strings")
    return result


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValidationError(f"{name} must be an object")
    return dict(value)


def _reject_unknown(value: Mapping[str, Any], allowed: set[str], name: str) -> None:
    extras = set(value) - allowed
    if extras:
        raise ValidationError(f"unknown {name} fields: {sorted(extras)}")


@dataclass(frozen=True, slots=True)
class Document:
    document_id: str
    document_version: str
    source_type: str
    raw_content: str
    structure: Mapping[str, JSONValue] = field(default_factory=dict)
    valid_from: str | None = None
    valid_to: str | None = None
    record_time: str | None = None
    permissions: tuple[str, ...] = ()
    content_hash: str = ""
    supersedes_version: str | None = None

    def __post_init__(self) -> None:
        for name in ("document_id", "document_version", "source_type"):
            if not getattr(self, name):
                raise ValidationError(f"{name} is required")
        if not isinstance(self.raw_content, str):
            raise ValidationError("raw_content must be a string")
        for name in ("valid_from", "valid_to", "record_time"):
            if getattr(self, name) is not None and not isinstance(getattr(self, name), str):
                raise ValidationError(f"{name} must be an ISO string or null")
        expected = content_hash(self.raw_content)
        if self.content_hash and self.content_hash != expected:
            raise ValidationError("document content_hash does not match raw_content")
        object.__setattr__(self, "content_hash", expected)
        object.__setattr__(
            self, "permissions", tuple(sorted(set(_strings(self.permissions, "permissions"))))
        )
        # Fail early if caller supplied a non-JSON structure.
        structure = dict(self.structure)
        if not all(isinstance(key, str) for key in structure):
            raise ValidationError("structure keys must be strings")
        # Canonical JSON round-tripping detaches the immutable document record
        # from caller-owned nested dictionaries/lists.
        structure = json.loads(canonical_json(structure))
        object.__setattr__(self, "structure", structure)

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "document_id": self.document_id,
            "document_version": self.document_version,
            "source_type": self.source_type,
            "raw_content": self.raw_content,
            "structure": dict(self.structure),
            "valid_from": self.valid_from,
            "valid_to": self.valid_to,
            "record_time": self.record_time,
            "permissions": list(self.permissions),
            "content_hash": self.content_hash,
            "supersedes_version": self.supersedes_version,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "Document":
        return cls(
            document_id=str(value["document_id"]),
            document_version=str(value["document_version"]),
            source_type=str(value.get("source_type", "text")),
            raw_content=str(value["raw_content"]),
            structure=_mapping(value.get("structure"), "structure"),
            valid_from=value.get("valid_from"),
            valid_to=value.get("valid_to"),
            record_time=value.get("record_time"),
            permissions=_strings(value.get("permissions", ()), "permissions"),
            content_hash=str(value.get("content_hash", "")),
            supersedes_version=value.get("supersedes_version"),
        )


@dataclass(frozen=True, slots=True)
class SourceBlock:
    block_id: str
    document_id: str
    document_version: str
    source_type: str
    start_offset: int
    end_offset: int
    content: str
    document_hash: str
    block_hash: str
    valid_from: str | None = None
    valid_to: str | None = None
    record_time: str | None = None
    permissions: tuple[str, ...] = ()
    structure: Mapping[str, JSONValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.start_offset < 0 or self.end_offset < self.start_offset:
            raise ValidationError("invalid source block offsets")
        if self.end_offset - self.start_offset != len(self.content):
            raise ValidationError("source block offsets do not match content length")
        if content_hash(self.content) != self.block_hash:
            raise ValidationError("source block hash does not match content")

    def to_dict(self, include_content: bool = True) -> dict[str, JSONValue]:
        result: dict[str, JSONValue] = {
            "block_id": self.block_id,
            "document_id": self.document_id,
            "document_version": self.document_version,
            "source_type": self.source_type,
            "start_offset": self.start_offset,
            "end_offset": self.end_offset,
            "document_hash": self.document_hash,
            "block_hash": self.block_hash,
            "valid_from": self.valid_from,
            "valid_to": self.valid_to,
            "record_time": self.record_time,
            "permissions": list(self.permissions),
            "structure": dict(self.structure),
        }
        if include_content:
            result["content"] = self.content
        return result


@dataclass(frozen=True, slots=True)
class BindingValue:
    """A typed interpretation plus the exact surface form behind it."""

    value: JSONScalar
    value_type: str
    surface: str
    start_offset: int
    end_offset: int
    resolver: str | None = None

    def __post_init__(self) -> None:
        if self.end_offset < self.start_offset:
            raise ValidationError("invalid binding offsets")
        if not self.value_type:
            raise ValidationError("binding value_type is required")
        if not self.surface:
            raise ValidationError("binding surface cannot be empty")
        if isinstance(self.value, float) and not math.isfinite(self.value):
            raise ValidationError("binding numbers must be finite")

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "value": self.value,
            "value_type": self.value_type,
            "surface": self.surface,
            "start_offset": self.start_offset,
            "end_offset": self.end_offset,
            "resolver": self.resolver,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "BindingValue":
        scalar = value.get("value")
        if not isinstance(scalar, (str, int, float, bool, type(None))):
            raise ValidationError("binding value must be a JSON scalar")
        return cls(
            value=scalar,
            value_type=str(value.get("value_type", "string")),
            surface=str(value.get("surface", scalar if scalar is not None else "")),
            start_offset=int(value["start_offset"]),
            end_offset=int(value["end_offset"]),
            resolver=value.get("resolver"),
        )


@dataclass(frozen=True, slots=True)
class PredicateSpec:
    """A constrained local evidence predicate.

    ``must_terms`` must all be present.  At least one ``should_term`` or
    ``patterns`` match is required when either collection is non-empty.
    ``binding_patterns`` are regular expressions with their entire match used
    as the binding surface.  They are interpretations, not source authority.

    ``closed_world`` means lexical absence is a sound NO_MATCH for this exact
    predicate.  It defaults to false because semantic relevance normally
    cannot be ruled out by token absence.
    """

    predicate_id: str
    description: str
    must_terms: tuple[str, ...] = ()
    should_terms: tuple[str, ...] = ()
    any_term_groups: tuple[tuple[str, ...], ...] = ()
    patterns: tuple[str, ...] = ()
    exclude_terms: tuple[str, ...] = ()
    binding_patterns: Mapping[str, str] = field(default_factory=dict)
    binding_types: Mapping[str, str] = field(default_factory=dict)
    required_bindings: tuple[str, ...] = ()
    case_sensitive: bool = False
    closed_world: bool = False
    quote_window: int = 320
    polarity: str = "supports"
    modality: str = "source_statement"

    def __post_init__(self) -> None:
        if not self.predicate_id or not self.description:
            raise ValidationError("predicate_id and description are required")
        for name in (
            "must_terms",
            "should_terms",
            "patterns",
            "exclude_terms",
            "required_bindings",
        ):
            object.__setattr__(self, name, _strings(getattr(self, name), name))
        groups: list[tuple[str, ...]] = []
        if isinstance(self.any_term_groups, (str, bytes)):
            raise ValidationError("any_term_groups must be a list of string lists")
        for index, group in enumerate(self.any_term_groups):
            groups.append(_strings(group, f"any_term_groups[{index}]"))
        object.__setattr__(self, "any_term_groups", tuple(groups))
        if self.quote_window < 40:
            raise ValidationError("quote_window must be at least 40 characters")
        binding_patterns = _mapping(self.binding_patterns, "binding_patterns")
        binding_types = _mapping(self.binding_types, "binding_types")
        if not all(isinstance(key, str) and isinstance(val, str) for key, val in binding_patterns.items()):
            raise ValidationError("binding_patterns must map strings to regex strings")
        if not all(isinstance(key, str) and isinstance(val, str) for key, val in binding_types.items()):
            raise ValidationError("binding_types must map strings to type names")
        if not set(self.required_bindings).issubset(binding_patterns):
            raise ValidationError("required_bindings must have binding_patterns")
        object.__setattr__(self, "binding_patterns", binding_patterns)
        object.__setattr__(self, "binding_types", binding_types)

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "predicate_id": self.predicate_id,
            "description": self.description,
            "must_terms": list(self.must_terms),
            "should_terms": list(self.should_terms),
            "any_term_groups": [list(group) for group in self.any_term_groups],
            "patterns": list(self.patterns),
            "exclude_terms": list(self.exclude_terms),
            "binding_patterns": dict(self.binding_patterns),
            "binding_types": dict(self.binding_types),
            "required_bindings": list(self.required_bindings),
            "case_sensitive": self.case_sensitive,
            "closed_world": self.closed_world,
            "quote_window": self.quote_window,
            "polarity": self.polarity,
            "modality": self.modality,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "PredicateSpec":
        _reject_unknown(
            value,
            {
                "predicate_id",
                "description",
                "must_terms",
                "should_terms",
                "any_term_groups",
                "patterns",
                "exclude_terms",
                "binding_patterns",
                "binding_types",
                "required_bindings",
                "case_sensitive",
                "closed_world",
                "quote_window",
                "polarity",
                "modality",
            },
            "predicate",
        )
        return cls(
            predicate_id=str(value["predicate_id"]),
            description=str(value["description"]),
            must_terms=_strings(value.get("must_terms", ()), "must_terms"),
            should_terms=_strings(value.get("should_terms", ()), "should_terms"),
            any_term_groups=tuple(
                _strings(group, f"any_term_groups[{index}]")
                for index, group in enumerate(value.get("any_term_groups", ()))
            ),
            patterns=_strings(value.get("patterns", ()), "patterns"),
            exclude_terms=_strings(value.get("exclude_terms", ()), "exclude_terms"),
            binding_patterns=_mapping(value.get("binding_patterns"), "binding_patterns"),
            binding_types=_mapping(value.get("binding_types"), "binding_types"),
            required_bindings=_strings(value.get("required_bindings", ()), "required_bindings"),
            case_sensitive=bool(value.get("case_sensitive", False)),
            closed_world=bool(value.get("closed_world", False)),
            quote_window=int(value.get("quote_window", 320)),
            polarity=str(value.get("polarity", "supports")),
            modality=str(value.get("modality", "source_statement")),
        )

    @property
    def predicate_hash(self) -> str:
        return stable_hash(self.to_dict())


@dataclass(frozen=True, slots=True)
class ScanSpec:
    scan_id: str
    predicate: PredicateSpec
    source_types: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.scan_id:
            raise ValidationError("scan_id is required")
        object.__setattr__(self, "source_types", _strings(self.source_types, "source_types"))

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "scan_id": self.scan_id,
            "predicate": self.predicate.to_dict(),
            "source_types": list(self.source_types),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ScanSpec":
        _reject_unknown(value, {"scan_id", "predicate", "source_types"}, "scan")
        return cls(
            scan_id=str(value["scan_id"]),
            predicate=PredicateSpec.from_dict(_mapping(value["predicate"], "predicate")),
            source_types=_strings(value.get("source_types", ()), "source_types"),
        )


@dataclass(frozen=True, slots=True)
class DependencySpec:
    scan_id: str
    depends_on: str
    # target placeholder -> binding field emitted by depends_on
    bindings: Mapping[str, str]

    def __post_init__(self) -> None:
        if not self.scan_id or not self.depends_on:
            raise ValidationError("dependency scan ids are required")
        values = _mapping(self.bindings, "bindings")
        if not values or not all(isinstance(k, str) and isinstance(v, str) for k, v in values.items()):
            raise ValidationError("dependency bindings must map placeholder to field")
        object.__setattr__(self, "bindings", values)

    def to_dict(self) -> dict[str, JSONValue]:
        return {"scan_id": self.scan_id, "depends_on": self.depends_on, "bindings": dict(self.bindings)}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "DependencySpec":
        _reject_unknown(value, {"scan_id", "depends_on", "bindings"}, "dependency")
        return cls(
            scan_id=str(value["scan_id"]),
            depends_on=str(value["depends_on"]),
            bindings=_mapping(value["bindings"], "bindings"),
        )


ALLOWED_REDUCERS = frozenset(
    {
        "FILTER",
        "JOIN",
        "GROUP_BY",
        "COUNT",
        "SUM",
        "MIN",
        "MAX",
        "ARGMAX",
        "ARGMIN",
        "SORT",
        "DEDUPLICATE",
        "TEMPORAL_RESOLVE",
        "SET_DIFFERENCE",
        "INTERSECTION",
        "UNION",
    }
)


@dataclass(frozen=True, slots=True)
class ReducerSpec:
    reducer_id: str
    operation: str
    inputs: tuple[str, ...]
    output: str
    params: Mapping[str, JSONValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        operation = self.operation.upper()
        if operation not in ALLOWED_REDUCERS:
            raise ValidationError(f"unsupported reducer operation: {operation}")
        if not self.reducer_id or not self.output:
            raise ValidationError("reducer_id and output are required")
        if not self.inputs:
            raise ValidationError("reducers require at least one input")
        object.__setattr__(self, "operation", operation)
        object.__setattr__(self, "inputs", _strings(self.inputs, "inputs"))
        params = _mapping(self.params, "params")
        canonical_json(params)
        object.__setattr__(self, "params", params)

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "reducer_id": self.reducer_id,
            "operation": self.operation,
            "inputs": list(self.inputs),
            "output": self.output,
            "params": dict(self.params),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ReducerSpec":
        _reject_unknown(
            value, {"reducer_id", "operation", "inputs", "output", "params"}, "reducer"
        )
        return cls(
            reducer_id=str(value["reducer_id"]),
            operation=str(value["operation"]),
            inputs=_strings(value.get("inputs", ()), "inputs"),
            output=str(value["output"]),
            params=_mapping(value.get("params"), "params"),
        )


@dataclass(frozen=True, slots=True)
class CounterTestSpec:
    counter_id: str
    target_claim: str
    predicate: PredicateSpec
    rationale: str

    def __post_init__(self) -> None:
        if not self.counter_id or not self.target_claim or not self.rationale:
            raise ValidationError("counter_id, target_claim, and rationale are required")

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "counter_id": self.counter_id,
            "target_claim": self.target_claim,
            "predicate": self.predicate.to_dict(),
            "rationale": self.rationale,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CounterTestSpec":
        _reject_unknown(
            value, {"counter_id", "target_claim", "predicate", "rationale"}, "counter-test"
        )
        return cls(
            counter_id=str(value["counter_id"]),
            target_claim=str(value["target_claim"]),
            predicate=PredicateSpec.from_dict(_mapping(value["predicate"], "predicate")),
            rationale=str(value["rationale"]),
        )


@dataclass(frozen=True, slots=True)
class WitnessProgram:
    question: str
    scans: tuple[ScanSpec, ...]
    dependencies: tuple[DependencySpec, ...] = ()
    reducers: tuple[ReducerSpec, ...] = ()
    counter_tests: tuple[CounterTestSpec, ...] = ()
    inputs: Mapping[str, JSONValue] = field(default_factory=dict)
    answer_contract: Mapping[str, JSONValue] = field(default_factory=dict)
    as_of: str | None = None
    recorded_as_of: str | None = None
    ir_version: str = "wql-0.1.0"

    def __post_init__(self) -> None:
        if not self.question.strip():
            raise ValidationError("question is required")
        if not self.scans:
            raise ValidationError("a Witness program requires at least one scan")
        scan_ids = [scan.scan_id for scan in self.scans]
        if len(scan_ids) != len(set(scan_ids)):
            raise ValidationError("scan ids must be unique")
        predicate_ids = [scan.predicate.predicate_id for scan in self.scans]
        predicate_ids.extend(test.predicate.predicate_id for test in self.counter_tests)
        if len(predicate_ids) != len(set(predicate_ids)):
            raise ValidationError("predicate ids must be unique across scans and counter-tests")
        counter_ids = [test.counter_id for test in self.counter_tests]
        if len(counter_ids) != len(set(counter_ids)):
            raise ValidationError("counter-test ids must be unique")
        reducer_ids = [reducer.reducer_id for reducer in self.reducers]
        if len(reducer_ids) != len(set(reducer_ids)):
            raise ValidationError("reducer ids must be unique")
        if not self.ir_version:
            raise ValidationError("ir_version is required")
        known = set(scan_ids)
        for dependency in self.dependencies:
            if dependency.scan_id not in known or dependency.depends_on not in known:
                raise ValidationError("dependency references an unknown scan")
            if dependency.scan_id == dependency.depends_on:
                raise ValidationError("a scan cannot depend on itself")
        # Ensure dependencies form a DAG before any corpus work occurs.
        edges = {scan_id: set() for scan_id in scan_ids}
        for dependency in self.dependencies:
            edges[dependency.scan_id].add(dependency.depends_on)
        pending = dict(edges)
        while pending:
            ready = [node for node, deps in pending.items() if not deps.intersection(pending)]
            if not ready:
                raise ValidationError("scan dependencies contain a cycle")
            for node in ready:
                pending.pop(node)
        namespace = known.copy()
        for reducer in self.reducers:
            if any(item not in namespace for item in reducer.inputs):
                raise ValidationError(f"reducer {reducer.reducer_id} references an unknown input")
            if reducer.output in namespace:
                raise ValidationError(f"duplicate relation name: {reducer.output}")
            namespace.add(reducer.output)
        inputs = _mapping(self.inputs, "inputs")
        answer_contract = _mapping(self.answer_contract, "answer_contract")
        canonical_json(inputs)
        canonical_json(answer_contract)
        object.__setattr__(self, "inputs", inputs)
        object.__setattr__(self, "answer_contract", answer_contract)

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "question": self.question,
            "scans": [item.to_dict() for item in self.scans],
            "dependencies": [item.to_dict() for item in self.dependencies],
            "reducers": [item.to_dict() for item in self.reducers],
            "counter_tests": [item.to_dict() for item in self.counter_tests],
            "inputs": dict(self.inputs),
            "answer_contract": dict(self.answer_contract),
            "as_of": self.as_of,
            "recorded_as_of": self.recorded_as_of,
            "ir_version": self.ir_version,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "WitnessProgram":
        allowed = {
            "question",
            "scans",
            "dependencies",
            "reducers",
            "counter_tests",
            "inputs",
            "answer_contract",
            "as_of",
            "recorded_as_of",
            "ir_version",
        }
        extras = set(value) - allowed
        if extras:
            raise ValidationError(f"unknown WitnessQL fields: {sorted(extras)}")
        scans = value.get("scans")
        if not isinstance(scans, list):
            raise ValidationError("scans must be a list")
        return cls(
            question=str(value["question"]),
            scans=tuple(ScanSpec.from_dict(_mapping(item, "scan")) for item in scans),
            dependencies=tuple(
                DependencySpec.from_dict(_mapping(item, "dependency"))
                for item in value.get("dependencies", [])
            ),
            reducers=tuple(
                ReducerSpec.from_dict(_mapping(item, "reducer")) for item in value.get("reducers", [])
            ),
            counter_tests=tuple(
                CounterTestSpec.from_dict(_mapping(item, "counter_test"))
                for item in value.get("counter_tests", [])
            ),
            inputs=_mapping(value.get("inputs"), "inputs"),
            answer_contract=_mapping(value.get("answer_contract"), "answer_contract"),
            as_of=value.get("as_of"),
            recorded_as_of=value.get("recorded_as_of"),
            ir_version=str(value.get("ir_version", "wql-0.1.0")),
        )

    @property
    def program_hash(self) -> str:
        return stable_hash(self.to_dict())


@dataclass(frozen=True, slots=True)
class WitnessRow:
    witness_id: str
    predicate_id: str
    source_id: str
    source_version: str
    source_hash: str
    block_id: str
    start_offset: int
    end_offset: int
    quote: str
    bindings: Mapping[str, BindingValue] = field(default_factory=dict)
    polarity: str = "supports"
    modality: str = "source_statement"
    valid_time: str | None = None
    valid_to: str | None = None
    record_time: str | None = None
    evaluator_version: str = ""
    role: EvidenceRole = EvidenceRole.POSITIVE
    uncertainties: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.witness_id or not self.predicate_id or not self.source_id:
            raise ValidationError("witness identity fields are required")
        if self.start_offset < 0 or self.end_offset < self.start_offset:
            raise ValidationError("invalid witness offsets")
        if self.end_offset - self.start_offset != len(self.quote):
            raise ValidationError("witness offsets do not match quote length")
        if not self.quote:
            raise ValidationError("a witness quote cannot be empty")
        bindings: dict[str, BindingValue] = {}
        for name, binding in self.bindings.items():
            bindings[str(name)] = binding if isinstance(binding, BindingValue) else BindingValue.from_dict(binding)
        object.__setattr__(self, "bindings", bindings)
        if not isinstance(self.role, EvidenceRole):
            object.__setattr__(self, "role", EvidenceRole(self.role))
        object.__setattr__(self, "uncertainties", tuple(str(item) for item in self.uncertainties))

    @classmethod
    def create(
        cls,
        *,
        predicate_id: str,
        block: SourceBlock,
        start_offset: int,
        end_offset: int,
        quote: str,
        bindings: Mapping[str, BindingValue] | None = None,
        polarity: str = "supports",
        modality: str = "source_statement",
        evaluator_version: str,
        role: EvidenceRole = EvidenceRole.POSITIVE,
        uncertainties: Iterable[str] = (),
    ) -> "WitnessRow":
        identity = stable_hash(
            {
                "predicate_id": predicate_id,
                "source": block.document_id,
                "version": block.document_version,
                "start": start_offset,
                "end": end_offset,
                "role": role.value,
            }
        )
        return cls(
            witness_id=f"wit-{identity[:24]}",
            predicate_id=predicate_id,
            source_id=block.document_id,
            source_version=block.document_version,
            source_hash=block.document_hash,
            block_id=block.block_id,
            start_offset=start_offset,
            end_offset=end_offset,
            quote=quote,
            bindings=bindings or {},
            polarity=polarity,
            modality=modality,
            valid_time=block.valid_from,
            valid_to=block.valid_to,
            record_time=block.record_time,
            evaluator_version=evaluator_version,
            role=role,
            uncertainties=tuple(uncertainties),
        )

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "witness_id": self.witness_id,
            "predicate_id": self.predicate_id,
            "source_id": self.source_id,
            "source_version": self.source_version,
            "source_hash": self.source_hash,
            "block_id": self.block_id,
            "start_offset": self.start_offset,
            "end_offset": self.end_offset,
            "quote": self.quote,
            "bindings": {name: binding.to_dict() for name, binding in self.bindings.items()},
            "polarity": self.polarity,
            "modality": self.modality,
            "valid_time": self.valid_time,
            "valid_to": self.valid_to,
            "record_time": self.record_time,
            "evaluator_version": self.evaluator_version,
            "role": self.role.value,
            "uncertainties": list(self.uncertainties),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "WitnessRow":
        raw_bindings = _mapping(value.get("bindings"), "bindings")
        return cls(
            witness_id=str(value["witness_id"]),
            predicate_id=str(value["predicate_id"]),
            source_id=str(value["source_id"]),
            source_version=str(value["source_version"]),
            source_hash=str(value["source_hash"]),
            block_id=str(value["block_id"]),
            start_offset=int(value["start_offset"]),
            end_offset=int(value["end_offset"]),
            quote=str(value["quote"]),
            bindings={name: BindingValue.from_dict(item) for name, item in raw_bindings.items()},
            polarity=str(value.get("polarity", "supports")),
            modality=str(value.get("modality", "source_statement")),
            valid_time=value.get("valid_time"),
            valid_to=value.get("valid_to"),
            record_time=value.get("record_time"),
            evaluator_version=str(value.get("evaluator_version", "")),
            role=EvidenceRole(value.get("role", "positive")),
            uncertainties=tuple(str(item) for item in value.get("uncertainties", [])),
        )


@dataclass(frozen=True, slots=True)
class LocalEvaluation:
    status: EvaluationStatus
    witnesses: tuple[WitnessRow, ...] = ()
    reason: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.status, EvaluationStatus):
            object.__setattr__(self, "status", EvaluationStatus(self.status))
        if self.status is EvaluationStatus.MATCH and not self.witnesses:
            raise ValidationError("MATCH evaluations require at least one witness")
        if self.status is EvaluationStatus.NO_MATCH and self.witnesses:
            raise ValidationError("NO_MATCH evaluations cannot contain witnesses")

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "status": self.status.value,
            "witnesses": [row.to_dict() for row in self.witnesses],
            "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "LocalEvaluation":
        return cls(
            status=EvaluationStatus(value["status"]),
            witnesses=tuple(WitnessRow.from_dict(item) for item in value.get("witnesses", [])),
            reason=str(value.get("reason", "")),
        )


@dataclass(frozen=True, slots=True)
class ClaimLedgerEntry:
    claim_id: str
    text: str
    supporting_witness_ids: tuple[str, ...] = ()
    counter_witness_ids: tuple[str, ...] = ()
    status: ClaimStatus = ClaimStatus.UNRESOLVED
    qualifiers: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.claim_id or not self.text:
            raise ValidationError("claim_id and text are required")
        if not isinstance(self.status, ClaimStatus):
            object.__setattr__(self, "status", ClaimStatus(self.status))

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "claim_id": self.claim_id,
            "text": self.text,
            "supporting_witness_ids": list(self.supporting_witness_ids),
            "counter_witness_ids": list(self.counter_witness_ids),
            "status": self.status.value,
            "qualifiers": list(self.qualifiers),
        }


@dataclass(frozen=True, slots=True)
class CoverageCertificate:
    corpus_snapshot: str
    authorized_shards: int
    scanned_shards: int
    soundly_skipped_shards: int
    unresolved_shards: int
    uncertain_shards: int
    positive_witnesses: int
    counter_witnesses: int
    program_version: str
    program_hash: str
    evaluator_version: str
    verifier_version: str
    cache_hits: int = 0
    cache_misses: int = 0
    unresolved_reasons: tuple[str, ...] = ()
    scheduled_obligations: int = 0
    conclusive_obligations: int = 0
    uncertain_obligations: int = 0
    failed_obligations: int = 0

    def __post_init__(self) -> None:
        counts = (
            self.authorized_shards,
            self.scanned_shards,
            self.soundly_skipped_shards,
            self.unresolved_shards,
            self.uncertain_shards,
            self.positive_witnesses,
            self.counter_witnesses,
            self.cache_hits,
            self.cache_misses,
            self.scheduled_obligations,
            self.conclusive_obligations,
            self.uncertain_obligations,
            self.failed_obligations,
        )
        if any(item < 0 for item in counts):
            raise ValidationError("coverage counts cannot be negative")
        if self.scanned_shards + self.soundly_skipped_shards + self.unresolved_shards != self.authorized_shards:
            raise ValidationError("coverage shard accounting does not balance")
        if self.uncertain_shards > self.scanned_shards:
            raise ValidationError("uncertain_shards cannot exceed scanned_shards")
        if (
            self.conclusive_obligations
            + self.uncertain_obligations
            + self.failed_obligations
            != self.scheduled_obligations
        ):
            raise ValidationError("coverage obligation accounting does not balance")

    @property
    def conclusive_fraction(self) -> float:
        if not self.authorized_shards:
            return 1.0
        conclusive = self.scanned_shards - self.uncertain_shards + self.soundly_skipped_shards
        return conclusive / self.authorized_shards

    @property
    def conclusive_obligation_fraction(self) -> float:
        if not self.scheduled_obligations:
            return 1.0
        return self.conclusive_obligations / self.scheduled_obligations

    def negative_claim_language(self, label: str = "matching evidence") -> str:
        conclusive = (
            self.scanned_shards - self.uncertain_shards + self.soundly_skipped_shards
        )
        unresolved = self.unresolved_shards + self.uncertain_shards
        return (
            f"Under the compiled evidence program, no {label} was found in "
            f"{conclusive} conclusively evaluated of "
            f"{self.authorized_shards} authorized shards; {unresolved} shards "
            "were unresolved or uncertain. Plan adequacy for the original "
            "question was not evaluated."
        )

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "metric_name": "program_execution_coverage",
            "plan_adequacy": "not_evaluated",
            "corpus_snapshot": self.corpus_snapshot,
            "authorized_shards": self.authorized_shards,
            "scanned_shards": self.scanned_shards,
            "soundly_skipped_shards": self.soundly_skipped_shards,
            "unresolved_shards": self.unresolved_shards,
            "uncertain_shards": self.uncertain_shards,
            "positive_witnesses": self.positive_witnesses,
            "counter_witnesses": self.counter_witnesses,
            "program_version": self.program_version,
            "program_hash": self.program_hash,
            "evaluator_version": self.evaluator_version,
            "verifier_version": self.verifier_version,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "conclusive_fraction": self.conclusive_fraction,
            "scheduled_obligations": self.scheduled_obligations,
            "conclusive_obligations": self.conclusive_obligations,
            "uncertain_obligations": self.uncertain_obligations,
            "failed_obligations": self.failed_obligations,
            "conclusive_obligation_fraction": self.conclusive_obligation_fraction,
            "unresolved_reasons": list(self.unresolved_reasons),
        }
