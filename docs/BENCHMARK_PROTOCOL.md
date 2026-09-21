# Witness benchmark protocol

## Implementation status

This document defines both the implemented offline pilot and the stricter
protocol required for a claim-grade result. They are not the same thing.

The repository currently implements only the deterministic
retrieval/execution track on one visible synthetic generator and seed. It has
no sealed test split, model-backed compiler or evaluator, common answer model,
human claim grader, or production GraphRAG/RAG/wiki implementations. The
implemented Witness ablations are counter search, binary versus tri-state
evaluation, approximate top-k gates, and materialized query views. The
no-reducer, no-exact-quote, no-temporal-metadata, and model-backed ablations
below remain requirements for the next experiment, not completed results.

Consequently, output from the current harness is labelled
`preliminary_offline_proxy_only`. It can falsify behavior of this prototype,
but it cannot support a production superiority claim.

## Purpose and claim boundary

This benchmark tests a scoped claim: query-specific exhaustive evidence
execution should improve evidence completeness and grounding on scattered,
multi-hop, contradictory, temporal, negative, and aggregation questions.
It does **not** assume that Witness is universally better than retrieval.

The claim-grade benchmark has two tracks:

1. **Retrieval/execution track.** Every system returns evidence spans. In the
   claim-grade version, a common deterministic task reducer scores what can be
   concluded from those spans. This isolates evidence selection from
   answer-model variance. The current pilot stops at scorer-side evidence
   exposure; it does not run a common answer reducer.
2. **End-to-end track.** Every system uses the same answer model, decoding
   settings, answer contract, and factual-claim scorer. This track is optional
   when no model provider is configured; its absence must be reported.

The included, dependency-free implementation is a controlled offline research
prototype. Its GraphRAG, vector, reranker, agentic, and wiki systems are
transparent algorithmic approximations, not claims to reproduce the quality of
particular hosted products. A production claim requires the model-backed track
and representative production implementations.

## Leakage controls

- Corpus text and public metadata are the only inputs visible to systems.
- Ground-truth evidence IDs, expected answers, counter-evidence labels, and
  decisive-evidence weights are passed only to the scorer.
- A system may not branch on task ID, category, expected answer, or annotation.
- Any deterministic reader used across systems receives only returned source
  text and the compiled public question—not hidden labels.
- Test seeds remain fixed and are evaluated only after an iteration is selected
  on development seeds.
- Prompt-injection strings are inert document bytes, never instructions.

## Corpus and task splits

Datasets contain versioned, lossless source documents and stable span IDs. A
claim-grade generated evaluation must use genuinely disjoint splits (not just
different document order or decoys under the same templates):

- development: architecture debugging and ablation selection;
- validation: stopping and regression checks;
- test: final comparison only.

Each split balances these classes: simple lookup, rare detail, multi-hop,
global aggregation, contradiction/correction, temporal `AS_OF`, negative
claims, qualifier sensitivity, prompt injection, and distributed evidence.
Scale sweeps vary corpus size, relevant-evidence count, hop depth, distractor
ratio, and context budget independently.

## Systems

Required systems and variants are:

- vector RAG at multiple `k` values;
- reciprocal-rank-fused BM25 + vector RAG;
- cross-feature reranked RAG;
- GraphRAG-style local and global retrieval;
- long-context, at multiple budgets and source orderings;
- iterative/agentic retrieval with a fixed call budget;
- synthesized wiki, with and without raw-source fallback;
- Witness, plus every registered ablation.

All systems receive identical authorized corpus snapshots. A baseline may use a
lossy index because that is part of what is under test, but its index-build cost
and storage are recorded. Answer context and tool-call budgets are matched where
the comparison calls for them; compute-unmatched results are reported separately.

## Metrics

Primary evidence metrics:

- evidence recall and precision at the exact-span and source level;
- decisive-evidence recall;
- counter-evidence recall;
- factual unsupported-claim rate;
- authorized-corpus coverage, unresolved coverage, and falsely claimed skips.

Required task metrics for the claim-grade/end-to-end track (implemented metric
functions exist, but retrieval-only outputs leave these values inapplicable):

- structured answer correctness;
- temporal-state accuracy;
- aggregation/count/ranking accuracy;
- contradiction preservation;
- negative-claim calibration (`not found` versus `does not exist`);
- qualifier retention;
- prompt-injection success rate (lower is better).

Operational metrics:

- input/output tokens or deterministic token estimates;
- model, embedding, reranker, and tool calls;
- CPU/GPU time and storage;
- cold, warm, and incremental-update cost;
- p50, p95, and p99 latency;
- evidence, answer, citation, and uncertainty reproducibility.

Every aggregate reports per-category values, macro and micro means, paired
deltas, and bootstrap confidence intervals where the sample size permits. Raw
per-run records remain available so averages cannot hide catastrophic misses.

## Witness ablations

The full protocol requires:

- no counter-witness phase *(implemented)*;
- binary evaluator without `UNCERTAIN` *(implemented)*;
- approximate top-k gates at 10, 50, 100, and 500 *(implemented)*;
- materialized views disabled *(implemented)*;
- sound lexical/metadata pruning only *(not yet isolated)*;
- direct semantic synthesis instead of deterministic reduction *(not yet run)*;
- interpretations without exact quotes *(not yet run)*;
- timestamps removed *(not yet run)*.

An ablation is useful only when all other settings are held constant.

## Iteration and stopping rule

Architecture changes are selected on development data for a stated failure,
not because they raise a single headline score. The change and rationale are
recorded before validation is rerun.

Witness is a "clear superior solution" only for its declared target workload if
all of the following hold on the untouched test split:

1. It has higher macro decisive-evidence and counter-evidence recall than every
   baseline, with paired 95% intervals excluding zero.
2. It has no worse structured-answer accuracy or unsupported-claim rate within a
   predeclared non-inferiority margin.
3. Gains occur in at least four target categories and are not produced solely by
   a larger final answer-context budget.
4. Coverage claims are calibrated: unresolved shards are never counted as
   conclusively searched.
5. Cold, warm, and update costs are all disclosed, including categories where
   indexed RAG is faster or cheaper.

This rule intentionally does not require Witness to beat indexed lookup on
simple questions. If confidence intervals, real-model evaluation, or production
baselines are unavailable, results are labelled preliminary rather than
superior.

## Falsification triggers

The strongest hypothesis is rejected or narrowed if any of these persist after
reasonable engineering fixes:

- local semantic false negatives erase the theoretical exhaustive-scan gain;
- compilation mistakes dominate retrieval mistakes;
- full-scan cost grows faster than useful view reuse can amortize;
- cross-block or entity-resolution failures exceed top-k baseline failures;
- counter-search mostly adds false positives without finding qualifiers;
- cache invalidation changes answers across equivalent snapshots;
- a competitive long-context or agentic baseline matches evidence recall at
  materially lower cost.

## Reproduction

Runs emit a manifest containing the code version, Python/platform details,
dataset seed, corpus snapshot hash, system configuration, evaluator/compiler
versions, timestamps, raw metrics, and latency samples. Repeating that manifest
must reproduce evidence sets and structured answers; timing is compared as a
distribution rather than byte-for-byte.
