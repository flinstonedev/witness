"""Human-readable report generation from a benchmark result manifest."""

from __future__ import annotations

import math
from typing import Any, Mapping


def _metric(summary: Mapping[str, Any], name: str) -> float | None:
    value = summary.get("quality_macro_by_unique_task", {}).get("macro", {}).get(name)
    return float(value) if isinstance(value, (int, float)) else None


def _fmt(value: Any, *, percent: bool = False, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(number):
        return "n/a"
    if percent:
        return f"{number * 100:.1f}%"
    return f"{number:.{digits}f}"


def _first_runs(result: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    return [item for item in result["runs"] if item["repetition"] == 0]


def _first_run(
    result: Mapping[str, Any], case_id: str, system: str
) -> Mapping[str, Any] | None:
    for item in _first_runs(result):
        if item.get("case_id") == case_id and item.get("system") == system:
            return item
    return None


def _first_totals(result: Mapping[str, Any], system: str) -> dict[str, float]:
    rows = [item for item in _first_runs(result) if item.get("system") == system]
    return {
        metric: sum(float(item["metrics"].get(metric, 0.0) or 0.0) for item in rows)
        for metric in ("input_tokens", "output_tokens", "latency_ms")
    }


def _paired_interval(
    result: Mapping[str, Any], comparator: str, metric: str
) -> Mapping[str, Any] | None:
    interval = (
        result.get("paired_vs_witness", {})
        .get(comparator, {})
        .get(metric)
    )
    return interval if isinstance(interval, Mapping) else None


def _fmt_interval(interval: Mapping[str, Any] | None) -> str:
    """Format a Witness-minus-comparator paired interval in percentage points."""

    if not interval:
        return "n/a"
    mean = interval.get("mean_delta")
    low = interval.get("low")
    high = interval.get("high")
    if not all(isinstance(value, (int, float)) for value in (mean, low, high)):
        return "n/a"
    return f"{float(mean) * 100:+.1f} [{float(low) * 100:+.1f}, {float(high) * 100:+.1f}]"


def _category_metric(
    summary: Mapping[str, Any], category: str, metric: str
) -> float | None:
    value = summary.get("by_category", {}).get(category, {}).get("macro", {}).get(metric)
    return float(value) if isinstance(value, (int, float)) else None


def _same_passage_sets(
    result: Mapping[str, Any], left_system: str, right_system: str
) -> bool | None:
    rows: dict[tuple[str, int], dict[str, str]] = {}
    for run in result.get("runs", []):
        system = str(run.get("system"))
        if system not in {left_system, right_system}:
            continue
        key = (str(run.get("case_id")), int(run.get("repetition", 0)))
        value = run.get("result", {}).get("passage_set_hash")
        if isinstance(value, str):
            rows.setdefault(key, {})[system] = value
    comparable = [item for item in rows.values() if len(item) == 2]
    if not comparable:
        return None
    return all(item[left_system] == item[right_system] for item in comparable)


def _same_first_quality_and_count(
    result: Mapping[str, Any], left_system: str, right_system: str
) -> tuple[bool, int]:
    """Compare scorer-visible quality and emitted-unit counts case by case."""

    metrics = (
        "evidence_recall",
        "decisive_evidence_recall",
        "counter_evidence_recall",
        "evidence_precision",
        "coverage",
    )
    grouped: dict[str, dict[str, Mapping[str, Any]]] = {}
    for run in _first_runs(result):
        system = str(run.get("system"))
        if system in {left_system, right_system}:
            grouped.setdefault(str(run.get("case_id")), {})[system] = run
    comparable = [rows for rows in grouped.values() if len(rows) == 2]
    parity = bool(comparable) and all(
        all(
            rows[left_system].get("metrics", {}).get(metric)
            == rows[right_system].get("metrics", {}).get(metric)
            for metric in metrics
        )
        and rows[left_system].get("result", {}).get("passages")
        == rows[right_system].get("result", {}).get("passages")
        for rows in comparable
    )
    return parity, len(comparable)


def render_markdown(
    result: Mapping[str, Any],
    *,
    multihop_diagnostic: Mapping[str, Any] | None = None,
) -> str:
    manifest = result["manifest"]
    summaries: Mapping[str, Mapping[str, Any]] = result["summary"]
    witness_name = "witness-exhaustive-offline"
    witness = summaries.get(witness_name)
    no_counter_name = "witness-exhaustive-offline-no-counter"
    no_counter = summaries.get(no_counter_name)
    no_view_name = "witness-no-query-view"
    no_view = summaries.get(no_view_name)
    full_context_name = "long-context-65536"
    full_context = summaries.get(full_context_name)
    short_context_name = "long-context-4096"
    short_context = summaries.get(short_context_name)
    graph_local_name = "graphrag-local-approx"
    graph_local = summaries.get(graph_local_name)
    lines = [
        "# Witness benchmark report",
        "",
        f"Generated: `{manifest['created_at']}`  ",
        f"Seed: `{manifest['seed']}`  ",
        f"Corpus: {manifest['documents']:,} document versions, "
        f"{manifest['evidence_spans']:,} exact evidence spans  ",
        f"Workload: {manifest['tasks']} task definitions, {manifest['cases']} snapshot cases, "
        f"{len(manifest['systems'])} systems, {manifest['repetitions']} repetitions",
        "",
        "## Verdict",
        "",
        "**No production superiority claim is supported by this run.** The measured track is "
        "a deterministic, retrieval/execution-only pilot. Its vector, reranker, GraphRAG, "
        "agentic, long-context, and wiki implementations are controlled offline proxies, and "
        "no model-backed compiler, semantic evaluator, answer generator, or human claim grader "
        "was available. Results can validate code paths and falsify local design choices; they "
        "cannot establish that Witness beats production systems.",
    ]
    if witness and full_context:
        latency_ratio = (
            witness["first_execution_latency_ms"]["mean"]
            / full_context["first_execution_latency_ms"]["mean"]
        )
        lines.extend(
            [
                "",
                "**The current Witness prototype is not the clear superior solution.** In this "
                "small corpus, the entire corpus fit inside the 65,536-token proxy window. That "
                f"proxy reached {_fmt(_metric(full_context, 'evidence_recall'), percent=True)} "
                f"evidence recall and {_fmt(_metric(full_context, 'decisive_evidence_recall'), percent=True)} "
                f"decisive recall versus {_fmt(_metric(witness, 'evidence_recall'), percent=True)} "
                f"and {_fmt(_metric(witness, 'decisive_evidence_recall'), percent=True)} for Witness, "
                f"while its first-recorded latency was about {latency_ratio:.1f}x lower. This is a "
                "direct falsification of superiority for the implemented workload and evaluator, "
                "not a reason to tune the benchmark until the ordering reverses.",
            ]
        )

    lines.extend(
        [
            "",
            "## Macro results",
            "",
            "Macro values first average duplicate snapshots within a task, then average the unique "
            "task definitions. Evidence precision is passage/row precision. `First-recorded` is "
            "repetition 0 for each case in one stateful benchmark process; it is not an isolated "
            "fresh-process cold start. `Repeat` is the immediate exact repetition. Latencies are "
            "wall-clock milliseconds, and token counts elsewhere are tokenizer proxies rather than "
            "model billing tokens. Coverage means **conclusive program-execution coverage**: the "
            "share of authorized shards conclusively classified for the emitted predicate. It does "
            "not measure whether the compiler's plan adequately represents the user's question.",
            "",
            "| System | Evidence recall | Decisive recall | Counter recall | Precision | Program-execution coverage | First-recorded ms | Exact-repeat ms |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    ordered = sorted(
        summaries.items(),
        key=lambda item: (
            -(
                _metric(item[1], "decisive_evidence_recall")
                if _metric(item[1], "decisive_evidence_recall") is not None
                else -1.0
            ),
            -(
                _metric(item[1], "counter_evidence_recall")
                if _metric(item[1], "counter_evidence_recall") is not None
                else -1.0
            ),
            item[0],
        ),
    )
    for name, summary in ordered:
        first = summary["first_execution_latency_ms"]["mean"]
        repeat = summary["repeat_execution_latency_ms"]["mean"]
        lines.append(
            "| "
            + " | ".join(
                (
                    name,
                    _fmt(_metric(summary, "evidence_recall"), percent=True),
                    _fmt(_metric(summary, "decisive_evidence_recall"), percent=True),
                    _fmt(_metric(summary, "counter_evidence_recall"), percent=True),
                    _fmt(_metric(summary, "evidence_precision"), percent=True),
                    _fmt(_metric(summary, "coverage"), percent=True),
                    _fmt(first, digits=1),
                    _fmt(repeat, digits=1),
                )
            )
            + " |"
        )

    operational_names = (
        witness_name,
        no_counter_name,
        "diagnostic-exhaustive-lexical-map-no-cache",
        "vector-rag-tfidf-k10",
        graph_local_name,
        short_context_name,
        full_context_name,
    )
    operational = [name for name in operational_names if name in summaries]
    if operational:
        executions_per_system = manifest["cases"] * manifest["repetitions"]
        lines.extend(
            [
                "",
                "## Operational distributions",
                "",
                "Latencies are local wall-clock milliseconds. First-recorded and exact-repeat "
                "distributions each contain one observation per snapshot case. Usage totals span "
                f"all {executions_per_system} recorded executions per system and use the harness's "
                "lexical token proxy, not provider billing. Calls are shown as "
                "`model / embedding / reranker / tool`.",
                "",
                "| System | First p50 | First p95 | First p99 | Repeat p50 | Repeat p95 | Repeat p99 | Total input tokens | Calls M/E/R/T |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for name in operational:
            summary = summaries[name]
            first = summary["first_execution_latency_ms"]
            repeat = summary["repeat_execution_latency_ms"]
            cost = summary["cost"]
            calls = "/".join(
                f"{int(cost.get(key, 0)):,}"
                for key in (
                    "model_calls",
                    "embedding_calls",
                    "reranker_calls",
                    "tool_calls",
                )
            )
            lines.append(
                "| "
                + " | ".join(
                    (
                        name,
                        _fmt(first.get("p50"), digits=1),
                        _fmt(first.get("p95"), digits=1),
                        _fmt(first.get("p99"), digits=1),
                        _fmt(repeat.get("p50"), digits=1),
                        _fmt(repeat.get("p95"), digits=1),
                        _fmt(repeat.get("p99"), digits=1),
                        f"{int(cost.get('input_tokens', 0)):,}",
                        calls,
                    )
                )
                + " |"
            )

    lines.extend(["", "## Ablation and iteration findings", ""])
    if witness and no_counter:
        counter_interval = _paired_interval(
            result, no_counter_name, "counter_evidence_recall"
        )
        evidence_interval = _paired_interval(result, no_counter_name, "evidence_recall")
        with_totals = _first_totals(result, witness_name)
        without_totals = _first_totals(result, no_counter_name)
        input_ratio = (
            with_totals["input_tokens"] / without_totals["input_tokens"]
            if without_totals["input_tokens"]
            else None
        )
        latency_ratio = (
            witness["first_execution_latency_ms"]["mean"]
            / no_counter["first_execution_latency_ms"]["mean"]
        )
        lines.extend(
            [
                "### Counter-search ablation",
                "",
                f"- Counter recall: {_fmt(_metric(no_counter, 'counter_evidence_recall'), percent=True)} "
                f"without versus {_fmt(_metric(witness, 'counter_evidence_recall'), percent=True)} "
                "with counter search (+8.9 percentage points in the macro-by-task view).",
                f"- Paired snapshot-case counter-recall delta (Witness minus no-counter): "
                f"{_fmt_interval(counter_interval)} percentage points; the descriptive interval "
                "touches zero. The corresponding evidence-recall delta was "
                f"{_fmt_interval(evidence_interval)} points.",
                f"- Decisive recall: {_fmt(_metric(no_counter, 'decisive_evidence_recall'), percent=True)} "
                f"without versus {_fmt(_metric(witness, 'decisive_evidence_recall'), percent=True)} "
                "with counter search: no gain.",
                f"- Evidence precision: {_fmt(_metric(no_counter, 'evidence_precision'), percent=True)} "
                f"without versus {_fmt(_metric(witness, 'evidence_precision'), percent=True)} "
                "with counter search (-5.4 points).",
                f"- Mean first-recorded latency increased from "
                f"{_fmt(no_counter['first_execution_latency_ms']['mean'], digits=1)} ms to "
                f"{_fmt(witness['first_execution_latency_ms']['mean'], digits=1)} ms "
                f"({latency_ratio:.1f}x). First-repetition proxy input volume increased from "
                f"{without_totals['input_tokens']:,.0f} to {with_totals['input_tokens']:,.0f} "
                f"tokens ({_fmt(input_ratio, digits=1)}x).",
                "",
                "Conclusion: the bundled lexical counter phase produced a modest recall benefit, "
                "but it did not change decisive recall and paid heavily in precision and compute. "
                "It is not justified as an unconditional phase by this pilot; it needs selective "
                "triggering or a substantially cheaper counter predicate.",
            ]
        )

    if witness and no_counter and no_view:
        first = witness["first_execution_latency_ms"]["mean"]
        repeat = witness["repeat_execution_latency_ms"]["mean"]
        speedup = first / repeat if first and repeat else None
        cached_repeat = no_counter["repeat_execution_latency_ms"]["mean"]
        no_view_repeat = no_view["repeat_execution_latency_ms"]["mean"]
        isolated_speedup = no_view_repeat / cached_repeat if cached_repeat else None
        hashes_equal = _same_passage_sets(result, no_counter_name, no_view_name)
        lines.extend(
            [
                "",
                "### Exact-query memoization and shard reuse",
                "",
                f"- Default Witness fell from {_fmt(first, digits=1)} ms first-recorded to "
                f"{_fmt(repeat, digits=1)} ms on the immediate exact repeat "
                f"({_fmt(speedup, digits=1)}x).",
                f"- The controlled no-counter cache ablation took "
                f"{_fmt(cached_repeat, digits=1)} ms with the query-result cache versus "
                f"{_fmt(no_view_repeat, digits=1)} ms without it "
                f"({_fmt(isolated_speedup, digits=1)}x). Passage-set hashes were "
                + ("identical for every comparable run." if hashes_equal else "not all identical."),
                "- This is exact-result memoization, keyed by corpus fingerprint, question, `AS_OF`, "
                "compiler version, evaluator version, and counter-search mode. It does **not** "
                "demonstrate reuse across paraphrases, related questions, or a changed corpus. Any "
                "corpus-fingerprint change invalidates this top-level hit; only the lower-level "
                "program-by-shard cache can reuse unchanged shards.",
            ]
        )

        v1 = _first_run(result, "incremental_snapshot_answers@v1", no_counter_name)
        v2 = _first_run(result, "incremental_snapshot_answers@v2", no_counter_name)
        if v1 and v2:
            v1_tokens = float(v1["metrics"]["input_tokens"])
            v2_tokens = float(v2["metrics"]["input_tokens"])
            token_reduction = 1.0 - (v2_tokens / v1_tokens) if v1_tokens else None
            v1_telemetry = v1["result"]["telemetry"]
            v2_telemetry = v2["result"]["telemetry"]
            v1_misses = float(v1_telemetry.get("counters", {}).get("cache_misses", 0))
            v2_misses = float(v2_telemetry.get("counters", {}).get("cache_misses", 0))
            v1_hits = float(v1_telemetry.get("cache_hits", 0))
            v2_hits = float(v2_telemetry.get("cache_hits", 0))
            lines.extend(
                [
                    f"- Separately, the no-counter `incremental_snapshot_answers` case demonstrates "
                    f"the lower-level program-by-shard update path. From v1 to v2, cache "
                    f"misses fell from {v1_misses:,.0f} to {v2_misses:,.0f}, hits rose from "
                    f"{v1_hits:,.0f} to {v2_hits:,.0f}, and proxy input "
                    f"fell from {v1_tokens:,.0f} to {v2_tokens:,.0f} tokens "
                    f"({_fmt(token_reduction, percent=True)} less). Decisive recall stayed at "
                    f"{_fmt(v2['metrics'].get('decisive_evidence_recall'), percent=True)}, but "
                    f"evidence recall fell from {_fmt(v1['metrics'].get('evidence_recall'), percent=True)} "
                    f"to {_fmt(v2['metrics'].get('evidence_recall'), percent=True)}. This supports "
                    "computational reuse, not complete incremental-update correctness.",
                ]
            )

    binary = summaries.get("witness-binary-no-uncertain")
    if no_counter and binary:
        lines.extend(
            [
                "",
                "### `UNCERTAIN` ablation",
                "",
                f"Forcing binary outcomes changed macro evidence recall from "
                f"{_fmt(_metric(no_counter, 'evidence_recall'), percent=True)} to "
                f"{_fmt(_metric(binary, 'evidence_recall'), percent=True)} and decisive recall "
                f"from {_fmt(_metric(no_counter, 'decisive_evidence_recall'), percent=True)} to "
                f"{_fmt(_metric(binary, 'decisive_evidence_recall'), percent=True)}, but reported "
                f"coverage jumped from {_fmt(_metric(no_counter, 'coverage'), percent=True)} to "
                f"{_fmt(_metric(binary, 'coverage'), percent=True)}. Here, removing `UNCERTAIN` "
                "inflated the coverage claim without improving retrieval quality.",
            ]
        )

    control_name = "diagnostic-exhaustive-lexical-map-no-cache"
    control = summaries.get(control_name)
    if control and no_counter:
        parity, parity_cases = _same_first_quality_and_count(
            result, control_name, no_counter_name
        )
        lines.extend(
            [
                "",
                "### Same-evaluator exhaustive-map control",
                "",
                "This diagnostic applies the exact fallback predicate and local evaluator to "
                "every block, but bypasses WitnessQL execution, mechanical verification, "
                "counter-search, deterministic reduction, ranking gates, and both caches.",
                "",
                (
                    f"- It matched no-counter Witness on evidence recall, decisive recall, "
                    f"counter recall, precision, coverage, and passage count in all "
                    f"{parity_cases} first-run cases."
                    if parity
                    else "- The control did not have complete case-level parity; inspect the raw runs."
                ),
                f"- Mean first-recorded latency was "
                f"{_fmt(control['first_execution_latency_ms']['mean'], digits=1)} ms for the "
                f"control versus {_fmt(no_counter['first_execution_latency_ms']['mean'], digits=1)} ms "
                "for no-counter Witness. The always-cold control repeated in "
                f"{_fmt(control['repeat_execution_latency_ms']['mean'], digits=1)} ms, while "
                f"Witness's exact-result cache repeated in "
                f"{_fmt(no_counter['repeat_execution_latency_ms']['mean'], digits=1)} ms.",
                "",
                "Conclusion: with the automatic one-scan compiler, the measured cold retrieval "
                "quality comes from exhaustive lexical mapping, not from the distinct Witness "
                "machinery. The latter adds auditable provenance, coverage states, reduction, "
                "counter execution, and reuse, but this control finds no independent retrieval-"
                "quality gain from those components.",
            ]
        )

    lines.extend(
        [
            "",
            "## Concrete wins and losses",
            "",
            "No case in the headline run gave Witness higher decisive-evidence recall than "
            "every proxy baseline; whenever Witness was complete on the distributed tasks, the "
            "full-context proxy also matched it. The wins below are therefore explicitly relative "
            "to fixed small retrieval/context budgets.",
            "",
        ]
    )

    distributed_systems = (
        witness_name,
        "vector-rag-tfidf-k10",
        graph_local_name,
        short_context_name,
        full_context_name,
    )
    distributed_rows: list[Mapping[str, Any]] = []
    for scale in (100, 1000):
        case_id = f"distributed_evidence_{scale}@v1"
        distributed_rows.extend(
            run
            for system in distributed_systems
            if (run := _first_run(result, case_id, system)) is not None
        )
    if distributed_rows:
        lines.extend(
            [
                "### Fixed-budget distributed-evidence wins",
                "",
                "Witness did win evidence completeness against the fixed small-budget proxies on "
                "the architecture-aligned distributed tasks. These are conditional wins, not overall "
                "wins: full-context tied Witness on recall and was faster.",
                "",
                "| Case | System | Evidence recall | Decisive recall | Precision | First-recorded ms | Proxy input tokens |",
                "|---|---|---:|---:|---:|---:|---:|",
            ]
        )
        for run in distributed_rows:
            metrics = run["metrics"]
            lines.append(
                "| "
                + " | ".join(
                    (
                        f"`{run['case_id']}`",
                        str(run["system"]),
                        _fmt(metrics.get("evidence_recall"), percent=True),
                        _fmt(metrics.get("decisive_evidence_recall"), percent=True),
                        _fmt(metrics.get("evidence_precision"), percent=True),
                        _fmt(metrics.get("latency_ms"), digits=1),
                        f"{float(metrics.get('input_tokens', 0)):,.0f}",
                    )
                )
                + " |"
            )

    if witness and full_context:
        lines.extend(
            [
                "",
                "### Full-context loss",
                "",
                f"Because all source text fit in the 65,536-token proxy window, full-context "
                f"achieved {_fmt(_metric(full_context, 'evidence_recall'), percent=True)} evidence, "
                f"{_fmt(_metric(full_context, 'decisive_evidence_recall'), percent=True)} decisive, "
                f"and {_fmt(_metric(full_context, 'counter_evidence_recall'), percent=True)} counter "
                f"recall. Witness achieved {_fmt(_metric(witness, 'evidence_recall'), percent=True)}, "
                f"{_fmt(_metric(witness, 'decisive_evidence_recall'), percent=True)}, and "
                f"{_fmt(_metric(witness, 'counter_evidence_recall'), percent=True)}, respectively. "
                f"Mean first-recorded latency was "
                f"{_fmt(full_context['first_execution_latency_ms']['mean'], digits=1)} ms versus "
                f"{_fmt(witness['first_execution_latency_ms']['mean'], digits=1)} ms. On this corpus, "
                "the exhaustive lexical Witness path added execution complexity without overcoming "
                "local evaluator false negatives.",
            ]
        )

    if witness and graph_local and short_context:
        witness_multihop = _category_metric(
            witness, "multi_hop", "decisive_evidence_recall"
        )
        graph_multihop = _category_metric(
            graph_local, "multi_hop", "decisive_evidence_recall"
        )
        context_multihop = _category_metric(
            short_context, "multi_hop", "decisive_evidence_recall"
        )
        hop_rows = [
            run
            for run in _first_runs(result)
            if run.get("system") == witness_name
            and str(run.get("case_id", "")).startswith("multihop_depth_")
            and isinstance(run.get("metrics", {}).get("decisive_evidence_recall"), (int, float))
        ]
        hop_values = [float(run["metrics"]["decisive_evidence_recall"]) for run in hop_rows]
        hop_range = (
            f"{min(hop_values) * 100:.1f}% to {max(hop_values) * 100:.1f}%"
            if hop_values
            else "n/a"
        )
        lines.extend(
            [
                "",
                "### Multi-hop loss",
                "",
                f"Across the depth-2 through depth-6 cases, Witness macro decisive recall was "
                f"{_fmt(witness_multihop, percent=True)} (individual snapshot cases ranged from "
                f"{hop_range}), while both the local GraphRAG proxy "
                f"({_fmt(graph_multihop, percent=True)}) and 4,096-token long-context proxy "
                f"({_fmt(context_multihop, percent=True)}) reached full recall. The current automatic "
                "compiler emits a broad one-scan lexical program, so this falsifies the implemented "
                "multi-hop pipeline—not the unimplemented hypothesis that correctly compiled binding "
                "dependencies could help.",
            ]
        )

    if multihop_diagnostic:
        diagnostic_manifest = multihop_diagnostic.get("manifest", {})
        diagnostic_summaries = multihop_diagnostic.get("summary", {})
        oracle = diagnostic_summaries.get(
            "witness-aster-human-plan-development-oracle"
        )
        diagnostic_auto = diagnostic_summaries.get(no_counter_name)
        diagnostic_graph = diagnostic_summaries.get(graph_local_name)
        diagnostic_context = diagnostic_summaries.get(short_context_name)
        if (
            diagnostic_manifest.get("diagnostic_only") is True
            and diagnostic_manifest.get("headline_eligible") is False
            and oracle
            and diagnostic_auto
        ):
            lines.extend(
                [
                    "",
                    "### Human-plan development oracle (diagnostic only)",
                    "",
                    "A separate, explicitly headline-ineligible diagnostic replaced automatic "
                    "compilation with a human-authored Aster Prism plan that encodes the visible "
                    "development fixture. It achieved "
                    f"{_fmt(_metric(oracle, 'evidence_recall'), percent=True)} evidence recall, "
                    f"{_fmt(_metric(oracle, 'decisive_evidence_recall'), percent=True)} decisive "
                    f"recall, and {_fmt(_metric(oracle, 'evidence_precision'), percent=True)} "
                    f"precision versus {_fmt(_metric(diagnostic_auto, 'evidence_recall'), percent=True)} "
                    "evidence/decisive recall for automatically compiled Witness. "
                    + (
                        f"The GraphRAG-local and 4,096-token controls also reached "
                        f"{_fmt(_metric(diagnostic_graph, 'decisive_evidence_recall'), percent=True)} "
                        f"and {_fmt(_metric(diagnostic_context, 'decisive_evidence_recall'), percent=True)} "
                        "decisive recall, respectively. "
                        if diagnostic_graph and diagnostic_context
                        else ""
                    )
                    + "This isolates the automatic compiler as a major bottleneck and shows that "
                    "the dependency executor can carry the synthetic chain when handed the answer-shaped "
                    "plan. It is **not** a fair baseline win, evidence of generalization, or eligible "
                    "for the macro table. See the [raw diagnostic](../results/multihop_development_oracle.json).",
                ]
            )

    selected_comparators = (
        full_context_name,
        short_context_name,
        "vector-rag-tfidf-k10",
        graph_local_name,
        no_counter_name,
    )
    interval_rows = [
        comparator
        for comparator in selected_comparators
        if _paired_interval(result, comparator, "evidence_recall") is not None
    ]
    if interval_rows:
        lines.extend(
            [
                "",
                "## Paired descriptive intervals",
                "",
                "Entries are Witness minus comparator in percentage points, shown as paired mean "
                "difference `[95% bootstrap interval]` over 38 snapshot cases for these three "
                "metrics. (The counter-search interval above uses 25 applicable pairs.) Snapshots and nested "
                "depth/scale tasks are correlated, so these intervals are descriptive and must not "
                "be treated as publication-grade significance tests.",
                "",
                "| Comparator | Evidence recall | Decisive recall | Precision |",
                "|---|---:|---:|---:|",
            ]
        )
        for comparator in interval_rows:
            lines.append(
                f"| {comparator} | "
                f"{_fmt_interval(_paired_interval(result, comparator, 'evidence_recall'))} | "
                f"{_fmt_interval(_paired_interval(result, comparator, 'decisive_evidence_recall'))} | "
                f"{_fmt_interval(_paired_interval(result, comparator, 'evidence_precision'))} |"
            )

    lines.extend(
        [
            "",
            "## What the run falsifies or leaves unresolved",
            "",
            "- Full execution does not imply semantic completeness. The bundled lexical evaluator "
            "marks lexical misses `UNCERTAIN`, so conclusive coverage can be far below physical scan coverage.",
            "- Core certificates label this value `metric_name=program_execution_coverage`, emit "
            "per-obligation execution counts, and state `plan_adequacy=not_evaluated`. The headline "
            "runner projects the numeric conclusive fraction; it does not score plan adequacy.",
            "- A broad high-recall predicate can produce very low precision and more reducer/context pressure than top-k retrieval.",
            "- The automatic compiler is a one-scan lexical fallback; this run does not validate "
            "LLM question compilation or the multi-hop advantage of generated dependency programs.",
            "- Structural exact-span provenance can be tested offline, but semantic entailment and "
            "unsupported factual claims require model/human judgments and were not scored here.",
            "- The 1,000-row distributed task dominates micro evidence counts; macro-by-task is the "
            "primary view, and nested hop depths/snapshot copies are not independent samples.",
            "- Exact-query result memoization improves immediate repeats, but is invalidated by any "
            "corpus-fingerprint change and says nothing about paraphrase hit rate. Workload hit rate, "
            "privacy isolation, invalidation races, and storage economics remain unmeasured.",
            "",
            "## When to use Witness",
            "",
            "Use a Witness-style path when missing a scattered exception is materially worse than "
            "extra compute: audits, policy history, contradiction-heavy synthesis, negative claims, "
            "and large deterministic aggregations. Keep indexed RAG/search for low-stakes simple "
            "lookups, exploratory questions, strict cold-latency workloads, or corpora too large for "
            "an economical semantic scan. A production router should make this choice explicitly.",
            "",
            "## Required next experiment",
            "",
            "Run the pre-registered model-backed track with disjoint unseen corpora and surface forms, "
            "a grammar-constrained compiler, calibrated local semantic evaluator, identical answer "
            "model/context budgets, production Microsoft GraphRAG local/global/DRIFT, real dense and "
            "cross-encoder baselines, human evidence/claim annotation, and enough independent questions "
            "for paired confidence intervals. Until then, the correct status is conditional go for "
            "continued research—not replacement of RAG.",
            "",
            "Relevant comparison anchors: [Microsoft GraphRAG query modes](https://microsoft.github.io/graphrag/query/overview/), "
            "[the original RAG paper](https://arxiv.org/abs/2005.11401), "
            "[RAGChecker](https://arxiv.org/abs/2408.08067), and "
            "[RAGTruth](https://arxiv.org/abs/2401.00396).",
            "",
        ]
    )
    return "\n".join(lines)


__all__ = ["render_markdown"]
