"""End-to-end Witness program execution."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, replace
from itertools import product
import re
import time
from typing import Any, Iterable, Mapping

from .cache import CacheKey, MaterializedWitnessCache
from .compiler import QuestionCompiler
from .corpus import CorpusStore
from .counter import CounterWitnessGenerator
from .evaluator import LexicalEvidenceEvaluator, LocalEvidenceEvaluator
from .models import (
    ClaimLedgerEntry,
    ClaimStatus,
    CounterTestSpec,
    CoverageCertificate,
    DependencySpec,
    EvaluationStatus,
    EvidenceRole,
    JSONValue,
    LocalEvaluation,
    PredicateSpec,
    ScanSpec,
    SourceBlock,
    ValidationError,
    WitnessProgram,
    WitnessRow,
    stable_hash,
)
from .reducer import DeterministicReducer
from .verification import MechanicalVerifier


@dataclass(frozen=True, slots=True)
class ExecutionTelemetry:
    elapsed_ms: float
    evaluator_calls: int
    cache_hits: int
    cache_misses: int
    verifier_rejections: int
    scan_obligations: int
    dependency_expansions: int
    execution_fingerprint: str

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "elapsed_ms": self.elapsed_ms,
            "evaluator_calls": self.evaluator_calls,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "verifier_rejections": self.verifier_rejections,
            "scan_obligations": self.scan_obligations,
            "dependency_expansions": self.dependency_expansions,
            "execution_fingerprint": self.execution_fingerprint,
        }


@dataclass(frozen=True, slots=True)
class ExecutionResult:
    program: WitnessProgram
    rows: tuple[WitnessRow, ...]
    counter_rows: tuple[WitnessRow, ...]
    relations: Mapping[str, list[dict[str, Any]]]
    claims: tuple[ClaimLedgerEntry, ...]
    coverage: CoverageCertificate
    telemetry: ExecutionTelemetry
    counter_phase: str = "awaiting_claims"
    unresolved: tuple[str, ...] = ()

    @property
    def final_relation(self) -> list[dict[str, Any]]:
        if self.program.reducers:
            return self.relations[self.program.reducers[-1].output]
        if self.program.scans:
            return self.relations[self.program.scans[-1].scan_id]
        return []

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "program": self.program.to_dict(),
            "rows": [row.to_dict() for row in self.rows],
            "counter_rows": [row.to_dict() for row in self.counter_rows],
            "relations": dict(self.relations),
            "claims": [claim.to_dict() for claim in self.claims],
            "coverage": self.coverage.to_dict(),
            "telemetry": self.telemetry.to_dict(),
            "counter_phase": self.counter_phase,
            "unresolved": list(self.unresolved),
        }


@dataclass(slots=True)
class _RunState:
    attempted_blocks: set[str] = field(default_factory=set)
    uncertain_blocks: set[str] = field(default_factory=set)
    error_blocks: set[str] = field(default_factory=set)
    unresolved: list[str] = field(default_factory=list)
    cache_hits: int = 0
    cache_misses: int = 0
    evaluator_calls: int = 0
    verifier_rejections: int = 0
    obligations: int = 0
    dependency_expansions: int = 0
    attempted_obligations: set[str] = field(default_factory=set)
    uncertain_obligations: set[str] = field(default_factory=set)
    error_obligations: set[str] = field(default_factory=set)


_PLACEHOLDER = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)\}")


def _coerce_claim(source: ClaimLedgerEntry | Mapping[str, Any] | object) -> ClaimLedgerEntry:
    if isinstance(source, ClaimLedgerEntry):
        return source
    if isinstance(source, Mapping):
        getter = source.get
    else:
        getter = lambda name, default=None: getattr(source, name, default)
    text = getter("text", getter("claim", ""))
    claim_id = getter("claim_id", getter("id", ""))
    if not claim_id and text:
        claim_id = "claim-" + stable_hash(str(text))[:16]
    return ClaimLedgerEntry(
        claim_id=str(claim_id),
        text=str(text),
        supporting_witness_ids=tuple(
            getter("supporting_witness_ids", getter("evidence_ids", ())) or ()
        ),
        counter_witness_ids=tuple(getter("counter_witness_ids", ()) or ()),
        status=ClaimStatus(getter("status", ClaimStatus.UNRESOLVED)),
        qualifiers=tuple(getter("qualifiers", ()) or ()),
    )


class WitnessEngine:
    """Stable facade for lossless ingest, compilation, and execution.

    Example::

        engine = WitnessEngine()
        engine.add_documents([{"id": "policy", "text": "Approval is required."}])
        plan = engine.compile("Is approval required?")
        result = engine.execute(plan)
        assert result.rows[0].quote == "Approval is required."
        assert result.coverage.authorized_shards == 1

    The built-in compiler/evaluator are conservative lexical fallbacks.  Pass a
    constrained semantic compiler and local evaluator for substantive natural-
    language benchmarks.
    """

    ENGINE_VERSION = "witness-engine-0.2.0"

    def __init__(
        self,
        *,
        store: CorpusStore | None = None,
        compiler: QuestionCompiler | None = None,
        evaluator: LocalEvidenceEvaluator | None = None,
        verifier: MechanicalVerifier | None = None,
        reducer: DeterministicReducer | None = None,
        counter_generator: CounterWitnessGenerator | None = None,
        cache: MaterializedWitnessCache | None = None,
        max_workers: int = 4,
        max_binding_expansions: int = 10_000,
    ) -> None:
        if max_workers < 1 or max_binding_expansions < 1:
            raise ValueError("execution limits must be positive")
        self.store = store if store is not None else CorpusStore()
        self.compiler = compiler if compiler is not None else QuestionCompiler()
        self.evaluator = evaluator if evaluator is not None else LexicalEvidenceEvaluator()
        self.verifier = verifier if verifier is not None else MechanicalVerifier()
        self.reducer = reducer if reducer is not None else DeterministicReducer()
        self.counter_generator = (
            counter_generator if counter_generator is not None else CounterWitnessGenerator()
        )
        self.cache = cache if cache is not None else MaterializedWitnessCache()
        self.max_workers = max_workers
        self.max_binding_expansions = max_binding_expansions

    def add_documents(self, sources: Iterable[object]):
        return self.store.add_documents(sources)

    def remove_document(self, document_id: str, document_version: str):
        return self.store.remove_document(document_id, document_version)

    def compile(
        self,
        question: str,
        *,
        as_of: str | None = None,
        recorded_as_of: str | None = None,
        inputs: Mapping[str, Any] | None = None,
    ) -> WitnessProgram:
        return self.compiler.compile(
            question,
            as_of=as_of,
            recorded_as_of=recorded_as_of,
            inputs=inputs,
        )

    def execute(
        self,
        plan: WitnessProgram | Mapping[str, Any] | str,
        *,
        principals: Iterable[str] = (),
        claims: Iterable[ClaimLedgerEntry | Mapping[str, Any] | object] = (),
    ) -> ExecutionResult:
        started = time.perf_counter()
        program = plan if isinstance(plan, WitnessProgram) else self.compiler.compile_structured(plan)
        self.compiler.validate(program)
        principal_tuple = tuple(sorted(set(principals)))
        blocks = self.store.blocks(
            principals=principal_tuple,
            recorded_as_of=program.recorded_as_of,
        )
        block_by_id = {block.block_id: block for block in blocks}
        snapshot = self.store.snapshot_hash(
            principals=principal_tuple,
            recorded_as_of=program.recorded_as_of,
        )
        state = _RunState()
        scan_rows: dict[str, tuple[WitnessRow, ...]] = {}
        scan_statuses: dict[str, list[EvaluationStatus]] = {}
        dependencies = self._dependencies(program.dependencies)

        for scan in self._ordered_scans(program):
            contexts = self._contexts_for_scan(scan, dependencies.get(scan.scan_id, ()), scan_rows, program)
            state.dependency_expansions += len(contexts)
            if len(contexts) > self.max_binding_expansions:
                state.unresolved.append(
                    f"scan {scan.scan_id} produced {len(contexts)} binding expansions, exceeding "
                    f"limit {self.max_binding_expansions}"
                )
                contexts = contexts[: self.max_binding_expansions]
                state.error_blocks.update(block.block_id for block in blocks)
            if not contexts:
                upstream_uncertain = any(
                    status is EvaluationStatus.UNCERTAIN
                    for dependency in dependencies.get(scan.scan_id, ())
                    for status in scan_statuses.get(dependency.depends_on, ())
                )
                if upstream_uncertain:
                    state.unresolved.append(
                        f"scan {scan.scan_id} could not bind variables because upstream evidence was uncertain"
                    )
                scan_rows[scan.scan_id] = ()
                scan_statuses[scan.scan_id] = []
                continue
            predicates = tuple(self._render_predicate(scan.predicate, context) for context in contexts)
            applicable = tuple(
                block for block in blocks if not scan.source_types or block.source_type in scan.source_types
            )
            rows, statuses = self._run_predicates(
                program,
                applicable,
                predicates,
                EvidenceRole.POSITIVE,
                principal_tuple,
                state,
            )
            scan_rows[scan.scan_id] = rows
            scan_statuses[scan.scan_id] = statuses

        claim_entries = tuple(_coerce_claim(claim) for claim in claims)
        if len({claim.claim_id for claim in claim_entries}) != len(claim_entries):
            raise ValidationError("claim ids must be unique")
        counter_tests = self.counter_generator.generate(program, claim_entries)
        counter_phase = "completed" if counter_tests else "awaiting_claims"
        if not counter_tests:
            state.unresolved.append(
                "counter-witness phase awaits provisional claims or explicit counter_tests"
            )
        counter_by_test: dict[str, tuple[WitnessRow, ...]] = {}
        for test in counter_tests:
            predicate = self._render_predicate(test.predicate, program.inputs)
            rows, _ = self._run_predicates(
                program,
                blocks,
                (predicate,),
                EvidenceRole.COUNTER,
                principal_tuple,
                state,
            )
            counter_by_test[test.counter_id] = rows

        positive_rows = self._deduplicate(
            row for rows in scan_rows.values() for row in rows
        )
        counter_rows = self._deduplicate(
            row for rows in counter_by_test.values() for row in rows
        )
        resolved_claims = self.counter_generator.reconcile(
            claim_entries, positive_rows, counter_by_test, counter_tests
        )

        reducer_specs = []
        for spec in program.reducers:
            if spec.operation == "TEMPORAL_RESOLVE":
                params = dict(spec.params)
                params.setdefault("as_of", program.as_of)
                params.setdefault("recorded_as_of", program.recorded_as_of)
                spec = replace(spec, params=params)
            reducer_specs.append(spec)
        relations = self.reducer.execute(scan_rows, reducer_specs)

        # Any block that failed one applicable obligation is unresolved rather
        # than counted as scanned; the categories remain mutually exclusive.
        errors = state.error_blocks.intersection(block_by_id)
        scanned = state.attempted_blocks.difference(errors)
        # The engine has no approximate gate. A never-attempted physical block
        # can only have been excluded by every scan's explicit source_type
        # scope (an exact metadata predicate), so this category is sound.
        skipped = set(block_by_id).difference(scanned, errors)
        uncertain = state.uncertain_blocks.intersection(scanned)
        coverage = CoverageCertificate(
            corpus_snapshot=snapshot,
            authorized_shards=len(blocks),
            scanned_shards=len(scanned),
            soundly_skipped_shards=len(skipped),
            unresolved_shards=len(errors),
            uncertain_shards=len(uncertain),
            positive_witnesses=len(positive_rows),
            counter_witnesses=len(counter_rows),
            program_version=program.ir_version,
            program_hash=program.program_hash,
            evaluator_version=self.evaluator.EVALUATOR_VERSION,
            verifier_version=self.verifier.VERIFIER_VERSION,
            cache_hits=state.cache_hits,
            cache_misses=state.cache_misses,
            unresolved_reasons=tuple(state.unresolved),
            scheduled_obligations=state.obligations,
            conclusive_obligations=len(
                state.attempted_obligations
                - state.uncertain_obligations
                - state.error_obligations
            ),
            uncertain_obligations=len(
                state.uncertain_obligations - state.error_obligations
            ),
            failed_obligations=len(state.error_obligations),
        )
        semantic_coverage = coverage.to_dict()
        semantic_coverage.pop("cache_hits", None)
        semantic_coverage.pop("cache_misses", None)
        fingerprint = stable_hash(
            {
                "engine": self.ENGINE_VERSION,
                "program": program.program_hash,
                "snapshot": snapshot,
                "evaluator": self.evaluator.EVALUATOR_VERSION,
                "verifier": self.verifier.VERIFIER_VERSION,
                "witness_ids": [row.witness_id for row in positive_rows],
                "counter_ids": [row.witness_id for row in counter_rows],
                "claims": [claim.to_dict() for claim in resolved_claims],
                "counter_phase": counter_phase,
                "relations": relations,
                "coverage": semantic_coverage,
            }
        )
        elapsed_ms = (time.perf_counter() - started) * 1000
        telemetry = ExecutionTelemetry(
            elapsed_ms=elapsed_ms,
            evaluator_calls=state.evaluator_calls,
            cache_hits=state.cache_hits,
            cache_misses=state.cache_misses,
            verifier_rejections=state.verifier_rejections,
            scan_obligations=state.obligations,
            dependency_expansions=state.dependency_expansions,
            execution_fingerprint=fingerprint,
        )
        return ExecutionResult(
            program=program,
            rows=positive_rows,
            counter_rows=counter_rows,
            relations=relations,
            claims=resolved_claims,
            coverage=coverage,
            telemetry=telemetry,
            counter_phase=counter_phase,
            unresolved=tuple(state.unresolved),
        )

    @staticmethod
    def _dependencies(items: Iterable[DependencySpec]) -> dict[str, tuple[DependencySpec, ...]]:
        grouped: dict[str, list[DependencySpec]] = {}
        for item in items:
            grouped.setdefault(item.scan_id, []).append(item)
        return {key: tuple(value) for key, value in grouped.items()}

    @staticmethod
    def _ordered_scans(program: WitnessProgram) -> tuple[ScanSpec, ...]:
        scans = {scan.scan_id: scan for scan in program.scans}
        dependencies = {scan_id: set() for scan_id in scans}
        for item in program.dependencies:
            dependencies[item.scan_id].add(item.depends_on)
        output: list[ScanSpec] = []
        done: set[str] = set()
        while len(output) < len(scans):
            ready = sorted(
                scan_id for scan_id, needs in dependencies.items() if scan_id not in done and needs <= done
            )
            if not ready:
                raise ValidationError("scan dependency ordering failed")
            for scan_id in ready:
                output.append(scans[scan_id])
                done.add(scan_id)
        return tuple(output)

    def _contexts_for_scan(
        self,
        scan: ScanSpec,
        dependencies: tuple[DependencySpec, ...],
        rows: Mapping[str, tuple[WitnessRow, ...]],
        program: WitnessProgram,
    ) -> list[dict[str, Any]]:
        base = dict(program.inputs)
        if not dependencies:
            return [base]
        alternatives: list[list[dict[str, Any]]] = []
        for dependency in dependencies:
            options: list[dict[str, Any]] = []
            for row in rows.get(dependency.depends_on, ()):
                candidate: dict[str, Any] = {}
                complete = True
                for target, source in dependency.bindings.items():
                    binding = row.bindings.get(source)
                    if binding is None:
                        complete = False
                        break
                    candidate[target] = binding.value
                if complete:
                    options.append(candidate)
            # Stable deduplication avoids repeating a full corpus pass when the
            # same entity binding has several supporting witnesses.
            unique = {stable_hash(item): item for item in options}
            alternatives.append([unique[key] for key in sorted(unique)])
        if any(not options for options in alternatives):
            return []
        contexts: dict[str, dict[str, Any]] = {}
        for combination in product(*alternatives):
            context = dict(base)
            compatible = True
            for item in combination:
                for key, value in item.items():
                    if key in context and context[key] != value:
                        compatible = False
                    context[key] = value
            if compatible:
                contexts[stable_hash(context)] = context
                if len(contexts) > self.max_binding_expansions:
                    break
        return [contexts[key] for key in sorted(contexts)]

    @staticmethod
    def _render_predicate(predicate: PredicateSpec, context: Mapping[str, Any]) -> PredicateSpec:
        def render(value: str, *, regex_value: bool = False) -> str:
            def substitute(match: re.Match[str]) -> str:
                name = match.group(1)
                if name not in context:
                    raise ValidationError(f"unbound predicate placeholder: {name}")
                replacement = str(context[name])
                return re.escape(replacement) if regex_value else replacement

            return _PLACEHOLDER.sub(substitute, value)

        identity = stable_hash([predicate.predicate_id, dict(context)])[:16]
        return replace(
            predicate,
            predicate_id=f"{predicate.predicate_id}@{identity}",
            description=render(predicate.description),
            must_terms=tuple(render(item) for item in predicate.must_terms),
            should_terms=tuple(render(item) for item in predicate.should_terms),
            any_term_groups=tuple(
                tuple(render(item) for item in group) for group in predicate.any_term_groups
            ),
            patterns=tuple(render(item, regex_value=True) for item in predicate.patterns),
            exclude_terms=tuple(render(item) for item in predicate.exclude_terms),
            binding_patterns={name: render(item, regex_value=True) for name, item in predicate.binding_patterns.items()},
        )

    def _run_predicates(
        self,
        program: WitnessProgram,
        blocks: tuple[SourceBlock, ...],
        predicates: tuple[PredicateSpec, ...],
        role: EvidenceRole,
        principals: tuple[str, ...],
        state: _RunState,
    ) -> tuple[tuple[WitnessRow, ...], list[EvaluationStatus]]:
        jobs = [(block, predicate) for predicate in predicates for block in blocks]
        state.obligations += len(jobs)
        if not jobs:
            return (), []
        results: list[tuple[str, str, LocalEvaluation | None, bool, str | None]] = []

        def work(block: SourceBlock, predicate: PredicateSpec):
            try:
                key = CacheKey.create(
                    program_hash=program.program_hash,
                    predicate_hash=predicate.predicate_hash,
                    block=block,
                    evaluator_version=self.evaluator.EVALUATOR_VERSION,
                    role=role,
                )
                cached = self.cache.get(key)
                if cached is not None:
                    return block.block_id, predicate.predicate_id, cached, True, None
                evaluation = self.evaluator.evaluate(block, predicate, role=role)
                self.cache.put(key, evaluation)
                return block.block_id, predicate.predicate_id, evaluation, False, None
            except Exception as exc:  # local failures are surfaced, never treated as NO_MATCH
                return block.block_id, predicate.predicate_id, None, False, f"{type(exc).__name__}: {exc}"

        with ThreadPoolExecutor(max_workers=self.max_workers, thread_name_prefix="witness") as pool:
            futures = [pool.submit(work, block, predicate) for block, predicate in jobs]
            for future in as_completed(futures):
                results.append(future.result())
        results.sort(key=lambda item: (item[0], item[1]))

        accepted: list[WitnessRow] = []
        statuses: list[EvaluationStatus] = []
        block_lookup = {block.block_id: block for block in blocks}
        for block_id, predicate_id, evaluation, was_cached, error in results:
            obligation_id = stable_hash((block_id, predicate_id, role.value))
            state.attempted_obligations.add(obligation_id)
            state.attempted_blocks.add(block_id)
            if was_cached:
                state.cache_hits += 1
            else:
                state.cache_misses += 1
                state.evaluator_calls += 1
            if error is not None or evaluation is None:
                state.error_obligations.add(obligation_id)
                state.error_blocks.add(block_id)
                state.unresolved.append(f"{block_id}/{predicate_id}: {error or 'no evaluation returned'}")
                continue
            statuses.append(evaluation.status)
            if evaluation.status is EvaluationStatus.UNCERTAIN:
                state.uncertain_obligations.add(obligation_id)
                state.uncertain_blocks.add(block_id)
            block = block_lookup[block_id]
            for row in evaluation.witnesses:
                if row.predicate_id != predicate_id:
                    state.error_obligations.add(obligation_id)
                    state.verifier_rejections += 1
                    state.error_blocks.add(block_id)
                    state.unresolved.append(
                        f"{block_id}/{predicate_id}: evaluator emitted a row for {row.predicate_id}"
                    )
                    continue
                if row.block_id != block_id or row.role is not role:
                    state.error_obligations.add(obligation_id)
                    state.verifier_rejections += 1
                    state.error_blocks.add(block_id)
                    state.unresolved.append(f"{block_id}/{predicate_id}: row block or evidence role mismatch")
                    continue
                verified = self.verifier.verify(row, self.store, principals=principals)
                if not verified.accepted:
                    state.error_obligations.add(obligation_id)
                    state.verifier_rejections += 1
                    state.error_blocks.add(block_id)
                    state.unresolved.append(
                        f"{block_id}/{predicate_id}: " + "; ".join(verified.errors)
                    )
                    continue
                accepted.append(row)
        return self._deduplicate(accepted), statuses

    @staticmethod
    def _deduplicate(rows: Iterable[WitnessRow]) -> tuple[WitnessRow, ...]:
        by_id: dict[str, WitnessRow] = {}
        for row in rows:
            existing = by_id.get(row.witness_id)
            if existing is not None and existing.to_dict() != row.to_dict():
                raise ValidationError(f"witness id collision: {row.witness_id}")
            by_id[row.witness_id] = row
        return tuple(
            sorted(
                by_id.values(),
                key=lambda row: (
                    row.source_id,
                    row.source_version,
                    row.start_offset,
                    row.end_offset,
                    row.predicate_id,
                    row.role.value,
                ),
            )
        )


def create_engine(**kwargs: Any) -> WitnessEngine:
    """Small factory kept stable for benchmark adapters and examples."""

    return WitnessEngine(**kwargs)
