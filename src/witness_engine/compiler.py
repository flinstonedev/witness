"""WitnessQL compilation and validation.

An LLM may be used as a planner by supplying a callable that returns a mapping,
but its output is never executed as code.  It must pass the same closed-schema
validation as hand-authored WitnessQL.
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from typing import Any, Callable, Mapping, Protocol

from .models import PredicateSpec, ScanSpec, ValidationError, WitnessProgram


class CompilerBackend(Protocol):
    def __call__(self, question: str, context: Mapping[str, Any]) -> Mapping[str, Any]: ...


_STOP_WORDS = frozenset(
    "a an and are as at be been before by did do does for from had has have how if in into is it its "
    "of on or that the their then there these this those to was were what when where which who whom why "
    "will with would any all about after currently current please tell show give find according".split()
)
_TOKEN = re.compile(r"[\w][\w.'-]*", flags=re.UNICODE)
_QUOTED = re.compile(r"[\"\u201c]([^\"\u201d]{2,160})[\"\u201d]")
_PLACEHOLDER = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)\}")


def fallback_lexical_predicate(
    question: str,
    *,
    max_terms: int = 24,
) -> PredicateSpec:
    """Build the exact predicate used by the no-backend compiler.

    Keeping this small transformation separate from :class:`WitnessProgram`
    construction provides a transparent experimental control: a benchmark can
    map the same predicate and evaluator over every block without also invoking
    Witness's IR execution, verification, counter-search, reduction, or cache.
    The function consumes only the question text and a declared term limit.
    """

    question = str(question).strip()
    if not question:
        raise ValidationError("question cannot be empty")
    if max_terms < 1:
        raise ValueError("max_terms must be positive")
    quoted = [match.group(1).strip() for match in _QUOTED.finditer(question)]
    terms: list[str] = []
    seen: set[str] = set()
    for match in _TOKEN.finditer(question):
        term = match.group(0).strip(".'-")
        normalized = term.casefold()
        if len(normalized) < 3 or normalized in _STOP_WORDS or normalized in seen:
            continue
        seen.add(normalized)
        terms.append(term)
        if len(terms) >= max_terms:
            break
    candidates = tuple(dict.fromkeys([*quoted, *terms]))
    if not candidates:
        candidates = (question,)
    return PredicateSpec(
        predicate_id="question_evidence",
        description=f"Evidence that can materially answer: {question}",
        should_terms=candidates,
        closed_world=False,
        modality="source_statement",
    )


class QuestionCompiler:
    """Compile a question to validated WitnessQL.

    With no backend, compilation is intentionally conservative: it produces a
    broad lexical evidence obligation and marks lexical absence as uncertain.
    This fallback is useful for local tests, not a claim of semantic parity
    with an LLM compiler.
    """

    COMPILER_VERSION = "compiler-0.1.0"

    def __init__(self, backend: CompilerBackend | None = None, *, max_terms: int = 24) -> None:
        if max_terms < 1:
            raise ValueError("max_terms must be positive")
        self.backend = backend
        self.max_terms = max_terms

    def compile(
        self,
        question: str,
        *,
        as_of: str | None = None,
        recorded_as_of: str | None = None,
        inputs: Mapping[str, Any] | None = None,
    ) -> WitnessProgram:
        question = question.strip()
        if not question:
            raise ValidationError("question cannot be empty")
        context = {
            "as_of": as_of,
            "recorded_as_of": recorded_as_of,
            "inputs": dict(inputs or {}),
            "compiler_version": self.COMPILER_VERSION,
        }
        if self.backend is not None:
            raw = self.backend(question, context)
            if not isinstance(raw, Mapping):
                raise ValidationError("compiler backend must return an object")
            payload = dict(raw)
            payload.setdefault("question", question)
            payload.setdefault("as_of", as_of)
            payload.setdefault("recorded_as_of", recorded_as_of)
            payload.setdefault("inputs", dict(inputs or {}))
            program = WitnessProgram.from_dict(payload)
        else:
            program = self._fallback(question, as_of, recorded_as_of, inputs or {})
        self.validate(program)
        return program

    def compile_structured(self, value: Mapping[str, Any] | str) -> WitnessProgram:
        if isinstance(value, str):
            try:
                decoded = json.loads(value)
            except json.JSONDecodeError as exc:
                raise ValidationError(f"invalid WitnessQL JSON: {exc}") from exc
            if not isinstance(decoded, Mapping):
                raise ValidationError("WitnessQL JSON must be an object")
            value = decoded
        program = WitnessProgram.from_dict(value)
        self.validate(program)
        return program

    def _fallback(
        self,
        question: str,
        as_of: str | None,
        recorded_as_of: str | None,
        inputs: Mapping[str, Any],
    ) -> WitnessProgram:
        predicate = fallback_lexical_predicate(question, max_terms=self.max_terms)
        return WitnessProgram(
            question=question,
            scans=(ScanSpec(scan_id="evidence", predicate=predicate),),
            inputs=dict(inputs),
            answer_contract={
                "mode": "evidence_rows",
                "must_cite_witness_ids": True,
                "negative_claims_require_conclusive_coverage": True,
            },
            as_of=as_of,
            recorded_as_of=recorded_as_of,
        )

    @staticmethod
    def validate(program: WitnessProgram) -> None:
        """Perform validation that depends on syntax rather than dataclass shape."""

        predicates = [scan.predicate for scan in program.scans]
        predicates.extend(test.predicate for test in program.counter_tests)
        for predicate in predicates:
            for pattern in (*predicate.patterns, *predicate.binding_patterns.values()):
                if len(pattern) > 1_024:
                    raise ValidationError(
                        f"regex in predicate {predicate.predicate_id} exceeds 1024 characters"
                    )
                try:
                    re.compile(pattern)
                except re.error as exc:
                    raise ValidationError(
                        f"invalid regex in predicate {predicate.predicate_id}: {exc}"
                    ) from exc
        dependencies = {}
        for dependency in program.dependencies:
            dependencies.setdefault(dependency.scan_id, []).append(dependency)
        scans = {scan.scan_id: scan for scan in program.scans}
        for scan_id, scan in scans.items():
            items = dependencies.get(scan_id, [])
            placeholders: set[str] = set()
            predicate = scan.predicate
            templated = (
                predicate.description,
                *predicate.must_terms,
                *predicate.should_terms,
                *(term for group in predicate.any_term_groups for term in group),
                *predicate.patterns,
                *predicate.exclude_terms,
                *predicate.binding_patterns.values(),
            )
            for item in templated:
                placeholders.update(_PLACEHOLDER.findall(item))
            supplied = {name for dependency in items for name in dependency.bindings}
            missing = placeholders - supplied - set(program.inputs)
            if missing:
                raise ValidationError(
                    f"scan {scan_id} has unbound placeholders: {sorted(missing)}"
                )
        for test in program.counter_tests:
            templated = (
                test.predicate.description,
                *test.predicate.must_terms,
                *test.predicate.should_terms,
                *(term for group in test.predicate.any_term_groups for term in group),
                *test.predicate.patterns,
                *test.predicate.exclude_terms,
                *test.predicate.binding_patterns.values(),
            )
            placeholders = {
                name for item in templated for name in _PLACEHOLDER.findall(item)
            }
            missing = placeholders - set(program.inputs)
            if missing:
                raise ValidationError(
                    f"counter-test {test.counter_id} has unbound placeholders: {sorted(missing)}"
                )
        for name, value in (("as_of", program.as_of), ("recorded_as_of", program.recorded_as_of)):
            if value is None:
                continue
            if not isinstance(value, str):
                raise ValidationError(f"{name} must be an ISO-8601 date or timestamp")
            normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
            try:
                datetime.fromisoformat(normalized)
            except (TypeError, ValueError) as exc:
                raise ValidationError(f"{name} must be an ISO-8601 date or timestamp") from exc
