# Witness

Witness is a research prototype for **proof-carrying information retrieval**:
compile a question into evidence obligations, execute those obligations over the
authorized corpus, verify exact source spans, reduce typed observations, search
for counter-evidence, and render only ledger-backed claims.

This repository also contains a falsifiable benchmark against controlled
offline approximations of vector RAG, hybrid RAG, reranked RAG, GraphRAG,
long-context prompting, agentic RAG, and an LLM-maintained wiki.

> Current status: alpha research scaffold. Offline proxy results are not a
> production superiority claim. See [the latest benchmark report](docs/BENCHMARK_REPORT.md)
> and [the limitations review](docs/CRITIQUE.md).

## What is implemented

- lossless, immutable document versions with stable character offsets, hashes,
  timestamps, source types, and permissions;
- bounded source blocks that remain anchored in original content;
- strict, serializable WitnessQL prototype models;
- conservative question compilation and local `MATCH / NO_MATCH / UNCERTAIN`
  evaluation;
- mechanical quote, offset, type, number, date, hash, version, and ACL checks;
- dependency-bound multi-hop scans;
- deterministic filters, joins, grouping, arithmetic, set operations, sorting,
  deduplication, and bitemporal resolution;
- counter-witness tests and an evidence-linked claim ledger;
- a deterministic renderer that asserts only supported/qualified ledger claims,
  keeps unresolved/refuted entries non-asserted, rejects unknown witness
  citations, carries coverage, and explicitly does not claim semantic entailment;
- coverage certificates that distinguish scanned, soundly skipped, unresolved,
  and uncertain blocks, add per-obligation counts, and separate program
  execution coverage from unmeasured plan adequacy;
- per-predicate/shard cache plus versioned exact-query materialized views;
- a 1,201-document adversarial synthetic corpus with 1,166 scorer-only exact
  spans and 21 task definitions;
- measured evidence recall/precision, decisive/counter recall, coverage,
  reproducibility and local cost/latency proxies; metric helpers for temporal,
  aggregation, negative and unsupported claims remain unmeasured in the
  retrieval-only run;
- top-k, binary-uncertainty, counter-search, and query-view ablations;
- a same-predicate exhaustive-map causal control and a separately labelled,
  headline-ineligible human-plan multi-hop diagnostic.

The executable IR is a strict subset of the target WitnessQL AST described in
[the language specification](docs/WITNESSQL.md). The bundled compiler and
evaluator are lexical fallbacks; production semantic model adapters are not
silently simulated.

## Quick start

The prototype requires Python 3.11+ and has no runtime dependencies. From the
checkout root:

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install .
python -m unittest discover -s tests -v
python examples/policy_evidence.py
witness-bench --profile smoke --seed 20260829 \
  --output artifacts/smoke.json --report artifacts/smoke.md
```

Run the complete offline sweep:

```bash
witness-bench \
  --profile full \
  --seed 20260829 \
  --repetitions 2 \
  --output artifacts/full.json \
  --report artifacts/full.md
python scripts/check_benchmark.py results/benchmark.json artifacts/full.json
```

List exact system configuration names or select subsets:

```bash
witness-bench --list-systems
witness-bench \
  --task distributed_evidence_1000 \
  --system witness-exhaustive-offline \
  --system vector-rag-tfidf-k10 \
  --output artifacts/subset.json --report artifacts/subset.md
```

Use explicit local output paths; the legacy CLI defaults overwrite the tracked
main result/report. [Reproduction instructions](docs/REPRODUCIBILITY.md) explain
independent smoke comparisons, the recorded environment and what is excluded
from timing-independent verification. CI is configured for Python 3.11–3.14.

## Minimal engine example

```python
from witness_engine import WitnessEngine

engine = WitnessEngine()
engine.add_documents(
    [
        {
            "id": "policy-17",
            "version": "3",
            "source_type": "policy",
            "text": "Manager approval is optional for emergency remediation.",
            "permissions": ["auditor"],
        }
    ]
)
program = engine.compile("Is manager approval always required?")
result = engine.execute(program, principals=("auditor",))

for row in result.rows:
    print(row.source_id, row.start_offset, row.end_offset, row.quote)
print(result.coverage.to_dict())
```

For substantive natural-language use, inject a grammar-constrained compiler
backend and a semantic implementation of `LocalEvidenceEvaluator`. Both remain
subject to the same verifier, reducer, cache, and coverage contracts.

Run [the complete example](examples/policy_evidence.py) with
`python examples/policy_evidence.py`. It also checks that a visitor cannot read
the auditor-only documents and that the returned quote exactly matches the
original character offsets. Expected output:

```text
example-policy@1 [0:55]: Manager approval is optional for emergency remediation.
Authorized shards: 2
Uncertain shards: 1
Exact source spans verified; semantic entailment and plan adequacy are not evaluated.
```

The unrelated fictional cafeteria document remains uncertain under the open
lexical predicate. The example returns evidence, not an adjudicated yes/no
answer. The caller is responsible for authenticating the supplied principals.
See [architecture and execution flow](docs/ARCHITECTURE.md).

## Benchmark validity boundary

The benchmark has a scorer-enforced isolation layer: systems receive the raw
documents, public question, AS_OF, answer schema, and public task metadata; expected answers,
relevance IDs, decisive weights, and counter labels remain outside system
inputs. Passage credit requires exact source-version coordinates and surviving
quoted bytes, not a claimed evidence ID.

Task IDs, categories and query features are still exposed to adapters; these
visible-fixture hints are a limitation, not a sealed evaluation. The bundled
lexical systems are inspectable, but an independent model-backed experiment
must remove those hints or justify the shared contract.

The included baselines are deliberately inspectable and deterministic. They are
useful for regression tests and controlled architecture comparisons, but they
are not equivalent to current hosted embeddings, cross-encoders, Microsoft's
GraphRAG, model-driven agents, or generated wikis. The JSON manifest labels
every such run `production_equivalent: false` and
`claim_status: preliminary_offline_proxy_only`.

A production comparison still needs:

- sealed development/validation/test corpora with independent task authors;
- real dense embeddings, BM25/RRF, cross-encoder and LLM rerankers;
- official GraphRAG local/global/DRIFT and competitive configuration tuning;
- actual long-context and agentic models under matched compute/latency budgets;
- a generated and incrementally maintained wiki;
- the same answer model and claim grader across systems;
- human evidence completeness, entailment, and ambiguity judgments;
- clustered statistical inference over independent scenarios;
- billed provider usage and production concurrency measurements.

## Repository map

- `src/witness_engine/`: engine, IR models, store, evaluator, verifier, reducer,
  counter search, caches, and coverage.
- `src/witness_bench/data.py`: deterministic versioned corpus generator.
- `src/witness_bench/tasks.py`: hidden scorer manifest.
- `src/witness_bench/baselines.py`: transparent offline baseline proxies.
- `src/witness_bench/controls.py`: same-evaluator exhaustive-map causal control.
- `src/witness_bench/plans.py`: human-authored development-only multi-hop plan.
- `src/witness_bench/metrics.py`: evidence-first metrics and cost aggregation.
- `src/witness_bench/isolation.py`: answer-key boundary and exact exposure matching.
- `src/witness_bench/runner.py`: cases, repetitions, ablations, summaries, and
  paired descriptive intervals.
- `docs/BENCHMARK_PROTOCOL.md`: predeclared protocol and stopping rule.
- `docs/ARCHITECTURE.md`: trust hierarchy, components, coverage, and security.
- `docs/CRITIQUE.md`: limitations review and production validation matrix.
- `results/benchmark.json`: complete 25-system, 1,900-run offline result manifest.
- `results/multihop_development_oracle.json`: non-headline compiler diagnostic.

## License and data handling

The project uses [Apache-2.0](LICENSE), matching the package metadata; see
[NOTICE](NOTICE) for scope. The fixture and committed benchmark results are
synthetic. Loading your own documents does not relicense them. Persistent stores
and caches can contain plaintext source text and have no automatic retention
or erasure policy. Read [data handling](docs/DATA_HANDLING.md) and
[the security policy](SECURITY.md) before using real data or custom adapters.
[Contributing](CONTRIBUTING.md) describes local checks and research reporting.

## Decision rule

Witness should not replace RAG merely because exhaustive scan raises recall on
a synthetic distributed task. A defensible result must show material decisive-
and counter-evidence gains on unseen target workloads, non-inferior answer
accuracy and unsupported-claim rate, calibrated coverage, and a practical
quality/cost/latency frontier. Simple lookups are expected to remain a place
where ordinary indexed retrieval wins.
