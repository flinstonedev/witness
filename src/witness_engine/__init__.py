"""Witness: proof-carrying information retrieval.

The public workflow is deliberately small::

    from witness_engine import WitnessEngine

    engine = WitnessEngine()
    engine.add_documents([{"id": "policy", "text": "Approval is required."}])
    program = engine.compile("Is approval required?", as_of="2026-01-01")
    result = engine.execute(program)
    print(result.rows, result.coverage)

For real semantic evaluation, inject a constrained ``QuestionCompiler`` backend
and an implementation of ``LocalEvidenceEvaluator``.  The built-ins are honest,
deterministic lexical fallbacks and surface semantic misses as ``UNCERTAIN``.
"""

from .cache import CacheKey, MaterializedWitnessCache
from .compiler import CompilerBackend, QuestionCompiler, fallback_lexical_predicate
from .corpus import CorpusStore, coerce_document
from .counter import CounterWitnessGenerator
from .engine import ExecutionResult, ExecutionTelemetry, WitnessEngine, create_engine
from .evaluator import LexicalEvidenceEvaluator, LocalEvidenceEvaluator
from .models import (
    ALLOWED_REDUCERS,
    BindingValue,
    ClaimLedgerEntry,
    ClaimStatus,
    CounterTestSpec,
    CoverageCertificate,
    DependencySpec,
    Document,
    EvaluationStatus,
    EvidenceRole,
    LocalEvaluation,
    PredicateSpec,
    ReducerSpec,
    ScanSpec,
    SourceBlock,
    ValidationError,
    WitnessProgram,
    WitnessRow,
)
from .reducer import DeterministicReducer, witness_record
from .renderer import LedgerAnswerRenderer, RenderedAnswer
from .verification import MechanicalVerifier, VerificationResult

__all__ = [
    "ALLOWED_REDUCERS",
    "BindingValue",
    "CacheKey",
    "ClaimLedgerEntry",
    "ClaimStatus",
    "CompilerBackend",
    "CorpusStore",
    "CounterTestSpec",
    "CounterWitnessGenerator",
    "CoverageCertificate",
    "DependencySpec",
    "DeterministicReducer",
    "Document",
    "EvaluationStatus",
    "EvidenceRole",
    "ExecutionResult",
    "ExecutionTelemetry",
    "LexicalEvidenceEvaluator",
    "LedgerAnswerRenderer",
    "LocalEvaluation",
    "LocalEvidenceEvaluator",
    "MaterializedWitnessCache",
    "MechanicalVerifier",
    "PredicateSpec",
    "QuestionCompiler",
    "ReducerSpec",
    "RenderedAnswer",
    "ScanSpec",
    "SourceBlock",
    "ValidationError",
    "VerificationResult",
    "WitnessEngine",
    "WitnessProgram",
    "WitnessRow",
    "coerce_document",
    "create_engine",
    "fallback_lexical_predicate",
    "witness_record",
]

__version__ = "0.2.0"
