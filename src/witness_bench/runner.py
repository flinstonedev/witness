"""Reproducible orchestration for the offline Witness benchmark track."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
import platform
import math
import sys
from typing import Any, Iterable, Mapping, Protocol, Sequence

from .baselines import (
    AgenticRAGBaseline,
    GraphRAGBaseline,
    HybridRAGBaseline,
    LLMWikiBaseline,
    LongContextBaseline,
    RerankedRAGBaseline,
    VectorRAGBaseline,
)
from .controls import ExhaustiveLexicalMapControl
from .data import DEFAULT_SEED, BenchmarkDataset, build_benchmark, document_key
from .isolation import answer_key_is_absent, public_documents, public_task
from .metrics import (
    aggregate_metric_reports,
    cost_metrics,
    evaluate_task,
    reproducibility_metrics,
)
from .models import (
    BenchmarkTask,
    RetrievalResult,
    SourceDocument,
    stable_hash,
    to_primitive,
)
from .stats import paired_bootstrap_interval, summarize
from .witness_system import (
    BinaryLexicalEvidenceEvaluator,
    WitnessOfflineSystem,
    WitnessTopKPrefilterSystem,
)


class RetrievalSystem(Protocol):
    name: str
    metadata: Mapping[str, Any]

    def retrieve(
        self, task: BenchmarkTask, documents: Sequence[SourceDocument]
    ) -> RetrievalResult: ...


def _implementation_fingerprint() -> tuple[str, int]:
    """Hash the exact Python implementation used for a benchmark manifest."""

    source_root = Path(__file__).resolve().parents[1]
    paths = sorted(
        path
        for package in (source_root / "witness_bench", source_root / "witness_engine")
        for path in package.rglob("*.py")
    )
    digest = sha256()
    for path in paths:
        relative = path.relative_to(source_root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        content = path.read_bytes()
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return "sha256:" + digest.hexdigest(), len(paths)


@dataclass(frozen=True, slots=True)
class Case:
    case_id: str
    task: BenchmarkTask
    snapshot: str
    documents: tuple[SourceDocument, ...]
    evidence_catalog: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class BenchmarkConfig:
    seed: int = DEFAULT_SEED
    repetitions: int = 2
    profile: str = "full"
    task_ids: tuple[str, ...] = ()
    system_names: tuple[str, ...] = ()
    bootstrap_resamples: int = 2_000

    def __post_init__(self) -> None:
        if self.repetitions < 1:
            raise ValueError("repetitions must be positive")
        if self.profile not in {"smoke", "full"}:
            raise ValueError("profile must be smoke or full")


def _named(system: Any, name: str) -> Any:
    system.name = name
    return system


def build_systems(profile: str = "full") -> tuple[RetrievalSystem, ...]:
    """Construct fresh system instances with unambiguous configuration names."""

    systems: list[RetrievalSystem] = [
        WitnessOfflineSystem(counter_search=True),
        WitnessOfflineSystem(counter_search=False),
        _named(
            VectorRAGBaseline(top_k=10, respect_as_of=False),
            "vector-rag-tfidf-k10",
        ),
        _named(
            HybridRAGBaseline(top_k=10, respect_as_of=False),
            "hybrid-rag-bm25-tfidf-k10",
        ),
        _named(
            RerankedRAGBaseline(
                top_k=10, candidate_k=50, respect_as_of=False
            ),
            "reranked-rag-heuristic-k10",
        ),
        GraphRAGBaseline(top_k=10, mode="local", respect_as_of=False),
        GraphRAGBaseline(top_k=10, mode="global", respect_as_of=False),
        _named(
            LongContextBaseline(
                top_k=10,
                context_token_budget=4_096,
                respect_as_of=False,
            ),
            "long-context-4096",
        ),
        _named(
            AgenticRAGBaseline(
                top_k=10,
                tool_call_budget=6,
                results_per_call=5,
                respect_as_of=False,
            ),
            "agentic-rag-budget6",
        ),
        _named(
            LLMWikiBaseline(top_k=10, respect_as_of=False),
            "llm-wiki-extractive-k10",
        ),
        _named(
            LLMWikiBaseline(
                top_k=10, raw_fallback=True, respect_as_of=False
            ),
            "llm-wiki-extractive-k10-raw-fallback",
        ),
    ]
    if profile == "full":
        systems.extend(
            [
                ExhaustiveLexicalMapControl(),
                _named(
                    WitnessOfflineSystem(
                        counter_search=False,
                        evaluator=BinaryLexicalEvidenceEvaluator(),
                    ),
                    "witness-binary-no-uncertain",
                ),
                _named(
                    WitnessOfflineSystem(
                        counter_search=False,
                        materialized_query_views=False,
                    ),
                    "witness-no-query-view",
                ),
                WitnessTopKPrefilterSystem(
                    prefilter_k=10, counter_search=False
                ),
                WitnessTopKPrefilterSystem(
                    prefilter_k=50, counter_search=False
                ),
                WitnessTopKPrefilterSystem(
                    prefilter_k=100, counter_search=False
                ),
                WitnessTopKPrefilterSystem(
                    prefilter_k=500, counter_search=False
                ),
                _named(
                    VectorRAGBaseline(top_k=50, respect_as_of=False),
                    "vector-rag-tfidf-k50",
                ),
                _named(
                    VectorRAGBaseline(top_k=100, respect_as_of=False),
                    "vector-rag-tfidf-k100",
                ),
                _named(
                    VectorRAGBaseline(top_k=500, respect_as_of=False),
                    "vector-rag-tfidf-k500",
                ),
                _named(
                    HybridRAGBaseline(top_k=50, respect_as_of=False),
                    "hybrid-rag-bm25-tfidf-k50",
                ),
                _named(
                    RerankedRAGBaseline(
                        top_k=50,
                        candidate_k=100,
                        respect_as_of=False,
                    ),
                    "reranked-rag-heuristic-k50",
                ),
                _named(
                    LongContextBaseline(
                        top_k=50,
                        context_token_budget=16_384,
                        respect_as_of=False,
                    ),
                    "long-context-16384",
                ),
                _named(
                    LongContextBaseline(
                        top_k=100,
                        context_token_budget=65_536,
                        respect_as_of=False,
                    ),
                    "long-context-65536",
                ),
            ]
        )
    return tuple(systems)


def _available_ids(dataset: BenchmarkDataset, snapshot: str) -> frozenset[str]:
    keys = set(dataset.corpus.snapshot(snapshot).document_keys)
    return frozenset(
        row.evidence_id
        for row in dataset.corpus.evidence
        if f"{row.document_id}@{row.document_version}" in keys
    )


def _case_task(task: BenchmarkTask, snapshot: str, available: frozenset[str]) -> BenchmarkTask:
    expected = task.expected_answer
    if task.category == "update_handling" and isinstance(expected, Mapping):
        by_version = expected.get("by_corpus_version", {})
        if isinstance(by_version, Mapping) and snapshot in by_version:
            expected = by_version[snapshot]
    return replace(
        task,
        corpus_versions=(snapshot,),
        expected_answer=expected,
        evidence_ids=tuple(item for item in task.evidence_ids if item in available),
        decisive_evidence_ids=tuple(
            item for item in task.decisive_evidence_ids if item in available
        ),
        counter_evidence_ids=tuple(
            item for item in task.counter_evidence_ids if item in available
        ),
    )


def build_cases(
    dataset: BenchmarkDataset,
    *,
    task_ids: Iterable[str] = (),
) -> tuple[Case, ...]:
    selected = set(task_ids)
    evidence_index = dataset.corpus.evidence_index()
    cases: list[Case] = []
    for task in dataset.tasks:
        if selected and task.task_id not in selected:
            continue
        for snapshot in task.corpus_versions:
            available = _available_ids(dataset, snapshot)
            case_task = _case_task(task, snapshot, available)
            relevant = set(
                (
                    *case_task.evidence_ids,
                    *case_task.decisive_evidence_ids,
                    *case_task.counter_evidence_ids,
                )
            )
            catalog = {
                evidence_id: evidence_index[evidence_id]
                for evidence_id in sorted(relevant)
                if evidence_id in evidence_index
            }
            cases.append(
                Case(
                    case_id=f"{task.task_id}@{snapshot}",
                    task=case_task,
                    snapshot=snapshot,
                    documents=public_documents(dataset.corpus.documents_for(snapshot)),
                    evidence_catalog=catalog,
                )
            )
    return tuple(cases)


def _average_reports_by_task(reports: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for report in reports:
        grouped.setdefault(str(report["task_id"]), []).append(report)
    output: list[dict[str, Any]] = []
    for task_id, members in sorted(grouped.items()):
        row: dict[str, Any] = {
            "task_id": task_id,
            "category": members[0]["category"],
        }
        keys = set().union(*(member.keys() for member in members))
        for key in keys:
            values = [
                float(member[key])
                for member in members
                if isinstance(member.get(key), (int, float))
                and not isinstance(member.get(key), bool)
            ]
            if values:
                row[key] = sum(values) / len(values)
        output.append(row)
    return output


def _system_summary(
    system: RetrievalSystem,
    first_reports: Sequence[Mapping[str, Any]],
    all_outputs: Sequence[RetrievalResult],
    repeat_groups: Mapping[str, Sequence[RetrievalResult]],
) -> dict[str, Any]:
    task_averages = _average_reports_by_task(first_reports)
    reproducibility = [
        reproducibility_metrics(outputs)
        for outputs in repeat_groups.values()
        if len(outputs) > 1
    ]
    reproducible_values = [
        float(item["reproducibility"])
        for item in reproducibility
        if item.get("reproducibility") is not None
    ]
    first_latency = [
        output.telemetry.latency_ms
        for outputs in repeat_groups.values()
        for output in outputs[:1]
    ]
    repeat_latency = [
        output.telemetry.latency_ms
        for outputs in repeat_groups.values()
        for output in outputs[1:]
    ]
    def distribution(values: Sequence[float]) -> dict[str, Any]:
        data = summarize(values).to_dict()
        return {
            key: (None if isinstance(value, float) and not math.isfinite(value) else value)
            for key, value in data.items()
        }

    return {
        "metadata": to_primitive(system.metadata),
        "quality_by_case": aggregate_metric_reports(first_reports),
        "quality_macro_by_unique_task": aggregate_metric_reports(task_averages),
        "by_category": aggregate_metric_reports(first_reports, group_key="category"),
        "cost": cost_metrics(all_outputs),
        "first_execution_latency_ms": distribution(first_latency),
        "repeat_execution_latency_ms": distribution(repeat_latency),
        "reproducibility": (
            sum(reproducible_values) / len(reproducible_values)
            if reproducible_values
            else None
        ),
        "reproducibility_groups": len(reproducible_values),
    }


def _paired_comparisons(
    runs: Sequence[Mapping[str, Any]],
    *,
    witness_name: str,
    resamples: int,
) -> dict[str, Any]:
    first = [item for item in runs if item["repetition"] == 0]
    by_system: dict[str, dict[str, Mapping[str, Any]]] = {}
    for item in first:
        by_system.setdefault(str(item["system"]), {})[str(item["case_id"])] = item
    candidate = by_system.get(witness_name, {})
    comparisons: dict[str, Any] = {}
    for system, rows in sorted(by_system.items()):
        if system == witness_name:
            continue
        metrics: dict[str, Any] = {}
        for metric in (
            "decisive_evidence_recall",
            "counter_evidence_recall",
            "evidence_recall",
            "evidence_precision",
        ):
            pairs = [
                (
                    candidate[case_id]["metrics"].get(metric),
                    rows[case_id]["metrics"].get(metric),
                )
                for case_id in sorted(set(candidate) & set(rows))
            ]
            applicable = [
                (float(left), float(right))
                for left, right in pairs
                if isinstance(left, (int, float))
                and isinstance(right, (int, float))
            ]
            if not applicable:
                metrics[metric] = None
                continue
            interval = paired_bootstrap_interval(
                [item[0] for item in applicable],
                [item[1] for item in applicable],
                resamples=resamples,
                seed=17,
            )
            metrics[metric] = interval.to_dict()
        comparisons[system] = metrics
    return comparisons


def run_benchmark(
    config: BenchmarkConfig | None = None,
    *,
    systems: Sequence[RetrievalSystem] | None = None,
) -> dict[str, Any]:
    config = config or BenchmarkConfig()
    dataset = build_benchmark(config.seed)
    cases = build_cases(dataset, task_ids=config.task_ids)
    selected_systems = list(systems or build_systems(config.profile))
    if config.system_names:
        requested = set(config.system_names)
        selected_systems = [item for item in selected_systems if item.name in requested]
        missing = requested - {item.name for item in selected_systems}
        if missing:
            raise KeyError(f"unknown systems: {sorted(missing)}")

    runs: list[dict[str, Any]] = []
    outputs_by_system: dict[str, list[RetrievalResult]] = {
        item.name: [] for item in selected_systems
    }
    repeat_groups: dict[str, dict[str, list[RetrievalResult]]] = {
        item.name: {} for item in selected_systems
    }
    reports_by_system: dict[str, list[dict[str, Any]]] = {
        item.name: [] for item in selected_systems
    }

    for system in selected_systems:
        for case in cases:
            query = public_task(case.task)
            if not answer_key_is_absent(query, case.documents):
                raise RuntimeError(f"answer-key isolation failed for {case.case_id}")
            for repetition in range(config.repetitions):
                output = system.retrieve(query, case.documents)
                report = evaluate_task(
                    case.task,
                    output,
                    evidence_catalog=case.evidence_catalog,
                    corpus=case.documents,
                )
                report["snapshot"] = case.snapshot
                report["case_id"] = case.case_id
                report["repetition"] = repetition
                outputs_by_system[system.name].append(output)
                repeat_groups[system.name].setdefault(case.case_id, []).append(output)
                if repetition == 0:
                    reports_by_system[system.name].append(report)
                runs.append(
                    {
                        "system": system.name,
                        "case_id": case.case_id,
                        "task_id": case.task.task_id,
                        "category": case.task.category,
                        "snapshot": case.snapshot,
                        "repetition": repetition,
                        "metrics": to_primitive(report),
                        "result": {
                            "passages": len(output.passages),
                            "passage_set_hash": stable_hash(
                                [passage.passage_id for passage in output.passages]
                            ),
                            "passage_id_sample": [
                                passage.passage_id for passage in output.passages[:10]
                            ],
                            "coverage": {
                                "authorized_shards": output.telemetry.authorized_shards,
                                "scanned_shards": output.telemetry.scanned_shards,
                                "soundly_skipped_shards": output.telemetry.soundly_skipped_shards,
                                "unresolved_shards": output.telemetry.unresolved_shards,
                            },
                            "telemetry": to_primitive(output.telemetry),
                        },
                    }
                )

    summaries = {
        system.name: _system_summary(
            system,
            reports_by_system[system.name],
            outputs_by_system[system.name],
            repeat_groups[system.name],
        )
        for system in selected_systems
    }
    witness_name = "witness-exhaustive-offline"
    comparisons = (
        _paired_comparisons(
            runs,
            witness_name=witness_name,
            resamples=config.bootstrap_resamples,
        )
        if witness_name in summaries
        else {}
    )
    snapshots = {
        item.snapshot_id: {
            "content_hash": item.content_hash,
            "documents": len(item.document_keys),
            "parent_id": item.parent_id,
            "added": len(item.added_keys),
            "removed": len(item.removed_keys),
            "superseded": len(item.superseded),
        }
        for item in dataset.corpus.snapshots
    }
    implementation_fingerprint, implementation_files = _implementation_fingerprint()
    return {
        "manifest": {
            "benchmark": "witness-offline-proxy",
            "benchmark_version": "0.2.0",
            "implementation_fingerprint": implementation_fingerprint,
            "implementation_python_files": implementation_files,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "seed": config.seed,
            "profile": config.profile,
            "repetitions": config.repetitions,
            "python": sys.version,
            "platform": platform.platform(),
            "documents": len(dataset.corpus.documents),
            "evidence_spans": len(dataset.corpus.evidence),
            "tasks": len({case.task.task_id for case in cases}),
            "cases": len(cases),
            "systems": [item.name for item in selected_systems],
            "snapshots": snapshots,
            "track": "retrieval_execution_only",
            "model_backed": False,
            "production_baselines": False,
            "claim_status": "preliminary_offline_proxy_only",
            "independence_warning": (
                "Snapshot repetitions and nested multi-hop/distributed scales are correlated; "
                "bootstrap intervals are descriptive, not publication-grade inference."
            ),
        },
        "summary": summaries,
        "paired_vs_witness": comparisons,
        "runs": runs,
    }


__all__ = [
    "BenchmarkConfig",
    "Case",
    "build_cases",
    "build_systems",
    "run_benchmark",
]
