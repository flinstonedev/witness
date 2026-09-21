# Witness benchmark report

This historical artifact was independently reproduced locally on 2026-09-21:
all 1,900 deterministic records matched; timings below remain the original
measurements. See [environment, commands and comparison scope](REPRODUCIBILITY.md).

Generated: `2026-08-30T00:30:00.105772+00:00`  
Seed: `20260829`  
Corpus: 1,201 document versions, 1,166 exact evidence spans  
Workload: 21 task definitions, 38 snapshot cases, 25 systems, 2 repetitions

## Verdict

**No production superiority claim is supported by this run.** The measured track is a deterministic, retrieval/execution-only pilot. Its vector, reranker, GraphRAG, agentic, long-context, and wiki implementations are controlled offline proxies, and no model-backed compiler, semantic evaluator, answer generator, or human claim grader was available. Results can validate code paths and falsify local design choices; they cannot establish that Witness beats production systems.

**The current Witness prototype is not the clear superior solution.** In this small corpus, the entire corpus fit inside the 65,536-token proxy window. That proxy reached 100.0% evidence recall and 100.0% decisive recall versus 83.0% and 87.0% for Witness, while its first-recorded latency was about 8.0x lower. This is a direct falsification of superiority for the implemented workload and evaluator, not a reason to tune the benchmark until the ordering reverses.

## Macro results

Macro values first average duplicate snapshots within a task, then average the unique task definitions. Evidence precision is passage/row precision. `First-recorded` is repetition 0 for each case in one stateful benchmark process; it is not an isolated fresh-process cold start. `Repeat` is the immediate exact repetition. Latencies are wall-clock milliseconds, and token counts elsewhere are tokenizer proxies rather than model billing tokens. Coverage means **conclusive program-execution coverage**: the share of authorized shards conclusively classified for the emitted predicate. It does not measure whether the compiler's plan adequately represents the user's question.

| System | Evidence recall | Decisive recall | Counter recall | Precision | Program-execution coverage | First-recorded ms | Exact-repeat ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| long-context-65536 | 100.0% | 100.0% | 100.0% | 4.7% | 100.0% | 48.3 | 45.2 |
| long-context-16384 | 97.0% | 97.0% | 92.9% | 4.6% | 48.0% | 27.9 | 24.1 |
| long-context-4096 | 93.2% | 93.3% | 85.7% | 4.3% | 12.5% | 12.4 | 9.3 |
| witness-exhaustive-offline | 83.0% | 87.0% | 89.3% | 9.5% | 20.4% | 384.8 | 4.3 |
| diagnostic-exhaustive-lexical-map-no-cache | 80.0% | 87.0% | 80.4% | 14.9% | 23.3% | 99.5 | 100.4 |
| witness-binary-no-uncertain | 80.0% | 87.0% | 80.4% | 14.9% | 100.0% | 136.3 | 4.3 |
| witness-exhaustive-offline-no-counter | 80.0% | 87.0% | 80.4% | 14.9% | 23.3% | 128.5 | 4.3 |
| witness-no-query-view | 80.0% | 87.0% | 80.4% | 14.9% | 23.3% | 129.7 | 86.6 |
| vector-rag-tfidf-k500 | 86.8% | 86.6% | 100.0% | 5.4% | 41.7% | 21.8 | 18.7 |
| reranked-rag-heuristic-k50 | 86.3% | 86.0% | 100.0% | 13.4% | 4.2% | 21.7 | 18.5 |
| hybrid-rag-bm25-tfidf-k50 | 85.1% | 84.8% | 100.0% | 8.5% | 4.2% | 17.6 | 14.5 |
| graphrag-global-approx | 82.5% | 83.7% | 91.1% | 35.8% | 0.7% | 243.0 | 240.9 |
| witness-hybrid-prefilter-k500 | 75.3% | 82.0% | 80.4% | 21.9% | 10.0% | 81.9 | 23.6 |
| vector-rag-tfidf-k100 | 80.5% | 80.3% | 100.0% | 8.9% | 8.3% | 15.0 | 11.8 |
| witness-hybrid-prefilter-k100 | 72.0% | 78.7% | 80.4% | 24.0% | 2.2% | 33.4 | 16.0 |
| vector-rag-tfidf-k50 | 78.5% | 78.2% | 100.0% | 10.3% | 4.2% | 14.2 | 11.0 |
| witness-hybrid-prefilter-k50 | 70.5% | 77.1% | 80.4% | 24.3% | 1.4% | 26.9 | 15.0 |
| graphrag-local-approx | 80.5% | 76.5% | 88.1% | 37.7% | 0.7% | 41.5 | 38.4 |
| hybrid-rag-bm25-tfidf-k10 | 71.3% | 70.1% | 95.2% | 25.7% | 0.8% | 17.5 | 14.0 |
| witness-hybrid-prefilter-k10 | 64.3% | 70.1% | 75.6% | 34.3% | 0.5% | 19.3 | 14.3 |
| vector-rag-tfidf-k10 | 71.0% | 69.8% | 95.2% | 26.0% | 0.8% | 15.1 | 11.0 |
| reranked-rag-heuristic-k10 | 70.7% | 69.4% | 95.2% | 25.5% | 0.8% | 19.3 | 16.4 |
| llm-wiki-extractive-k10-raw-fallback | 59.3% | 53.5% | 88.1% | 25.0% | 0.3% | 24.9 | 20.4 |
| agentic-rag-budget6 | 45.7% | 41.6% | 69.0% | 16.0% | 0.8% | 67.2 | 63.6 |
| llm-wiki-extractive-k10 | 8.6% | 0.0% | 41.1% | 5.7% | 0.0% | 15.3 | 11.2 |

## Operational distributions

Latencies are local wall-clock milliseconds. First-recorded and exact-repeat distributions each contain one observation per snapshot case. Usage totals span all 76 recorded executions per system and use the harness's lexical token proxy, not provider billing. Calls are shown as `model / embedding / reranker / tool`.

| System | First p50 | First p95 | First p99 | Repeat p50 | Repeat p95 | Repeat p99 | Total input tokens | Calls M/E/R/T |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| witness-exhaustive-offline | 416.2 | 633.2 | 675.7 | 4.3 | 4.5 | 4.7 | 1,380,658 | 0/0/0/0 |
| witness-exhaustive-offline-no-counter | 121.7 | 240.0 | 272.0 | 4.2 | 4.8 | 5.0 | 656,811 | 0/0/0/0 |
| diagnostic-exhaustive-lexical-map-no-cache | 90.5 | 161.7 | 172.6 | 93.0 | 162.6 | 172.4 | 2,479,798 | 0/0/0/0 |
| vector-rag-tfidf-k10 | 10.6 | 20.9 | 100.9 | 10.6 | 14.2 | 14.3 | 90,830 | 0/2,473/0/0 |
| graphrag-local-approx | 14.0 | 235.7 | 279.0 | 13.9 | 241.4 | 243.9 | 86,926 | 0/2,473/0/0 |
| long-context-4096 | 9.3 | 18.3 | 68.0 | 9.2 | 9.5 | 9.7 | 381,752 | 0/0/0/0 |
| long-context-65536 | 45.1 | 55.2 | 105.7 | 45.0 | 47.3 | 48.2 | 2,685,540 | 0/0/0/0 |

## Ablation and iteration findings

### Counter-search ablation

- Counter recall: 80.4% without versus 89.3% with counter search (+8.9 percentage points in the macro-by-task view).
- Paired snapshot-case counter-recall delta (Witness minus no-counter): +6.0 [+0.0, +16.0] percentage points; the descriptive interval touches zero. The corresponding evidence-recall delta was +2.0 [+0.0, +5.3] points.
- Decisive recall: 87.0% without versus 87.0% with counter search: no gain.
- Evidence precision: 14.9% without versus 9.5% with counter search (-5.4 points).
- Mean first-recorded latency increased from 128.5 ms to 384.8 ms (3.0x). First-repetition proxy input volume increased from 655,992 to 1,379,839 tokens (2.1x).

Conclusion: the bundled lexical counter phase produced a modest recall benefit, but it did not change decisive recall and paid heavily in precision and compute. It is not justified as an unconditional phase by this pilot; it needs selective triggering or a substantially cheaper counter predicate.

### Exact-query memoization and shard reuse

- Default Witness fell from 384.8 ms first-recorded to 4.3 ms on the immediate exact repeat (89.2x).
- The controlled no-counter cache ablation took 4.3 ms with the query-result cache versus 86.6 ms without it (20.2x). Passage-set hashes were identical for every comparable run.
- This is exact-result memoization, keyed by corpus fingerprint, question, `AS_OF`, compiler version, evaluator version, and counter-search mode. It does **not** demonstrate reuse across paraphrases, related questions, or a changed corpus. Any corpus-fingerprint change invalidates this top-level hit; only the lower-level program-by-shard cache can reuse unchanged shards.
- Separately, the no-counter `incremental_snapshot_answers` case demonstrates the lower-level program-by-shard update path. From v1 to v2, cache misses fell from 1,198 to 3, hits rose from 0 to 1,196, and proxy input fell from 34,420 to 111 tokens (99.7% less). Decisive recall stayed at 100.0%, but evidence recall fell from 100.0% to 75.0%. This supports computational reuse, not complete incremental-update correctness.

### `UNCERTAIN` ablation

Forcing binary outcomes changed macro evidence recall from 80.0% to 80.0% and decisive recall from 87.0% to 87.0%, but reported coverage jumped from 23.3% to 100.0%. Here, removing `UNCERTAIN` inflated the coverage claim without improving retrieval quality.

### Same-evaluator exhaustive-map control

This diagnostic applies the exact fallback predicate and local evaluator to every block, but bypasses WitnessQL execution, mechanical verification, counter-search, deterministic reduction, ranking gates, and both caches.

- It matched no-counter Witness on evidence recall, decisive recall, counter recall, precision, coverage, and passage count in all 38 first-run cases.
- Mean first-recorded latency was 99.5 ms for the control versus 128.5 ms for no-counter Witness. The always-cold control repeated in 100.4 ms, while Witness's exact-result cache repeated in 4.3 ms.

Conclusion: with the automatic one-scan compiler, the measured cold retrieval quality comes from exhaustive lexical mapping, not from the distinct Witness machinery. The latter adds auditable provenance, coverage states, reduction, counter execution, and reuse, but this control finds no independent retrieval-quality gain from those components.

## Concrete wins and losses

No case in the headline run gave Witness higher decisive-evidence recall than every proxy baseline; whenever Witness was complete on the distributed tasks, the full-context proxy also matched it. The wins below are therefore explicitly relative to fixed small retrieval/context budgets.

### Fixed-budget distributed-evidence wins

Witness did win evidence completeness against the fixed small-budget proxies on the architecture-aligned distributed tasks. These are conditional wins, not overall wins: full-context tied Witness on recall and was faster.

| Case | System | Evidence recall | Decisive recall | Precision | First-recorded ms | Proxy input tokens |
|---|---|---:|---:|---:|---:|---:|
| `distributed_evidence_100@v1` | witness-exhaustive-offline | 100.0% | 100.0% | 4.5% | 668.7 | 68,817 |
| `distributed_evidence_100@v1` | vector-rag-tfidf-k10 | 2.9% | 0.0% | 30.0% | 10.4 | 211 |
| `distributed_evidence_100@v1` | graphrag-local-approx | 3.9% | 1.0% | 40.0% | 28.1 | 209 |
| `distributed_evidence_100@v1` | long-context-4096 | 57.3% | 59.0% | 39.1% | 9.3 | 4,123 |
| `distributed_evidence_100@v1` | long-context-65536 | 100.0% | 100.0% | 8.6% | 44.7 | 34,422 |
| `distributed_evidence_1000@v1` | witness-exhaustive-offline | 100.0% | 100.0% | 44.0% | 626.9 | 68,817 |
| `distributed_evidence_1000@v1` | vector-rag-tfidf-k10 | 0.3% | 0.0% | 30.0% | 10.8 | 211 |
| `distributed_evidence_1000@v1` | graphrag-local-approx | 0.4% | 0.1% | 40.0% | 120.8 | 209 |
| `distributed_evidence_1000@v1` | long-context-4096 | 0.0% | 0.0% | 0.0% | 9.3 | 4,123 |
| `distributed_evidence_1000@v1` | long-context-65536 | 100.0% | 100.0% | 83.7% | 44.2 | 34,422 |

### Full-context loss

Because all source text fit in the 65,536-token proxy window, full-context achieved 100.0% evidence, 100.0% decisive, and 100.0% counter recall. Witness achieved 83.0%, 87.0%, and 89.3%, respectively. Mean first-recorded latency was 48.3 ms versus 384.8 ms. On this corpus, the exhaustive lexical Witness path added execution complexity without overcoming local evaluator false negatives.

### Multi-hop loss

Across the depth-2 through depth-6 cases, Witness macro decisive recall was 45.3% (individual snapshot cases ranged from 33.3% to 60.0%), while both the local GraphRAG proxy (100.0%) and 4,096-token long-context proxy (100.0%) reached full recall. The current automatic compiler emits a broad one-scan lexical program, so this falsifies the implemented multi-hop pipeline—not the unimplemented hypothesis that correctly compiled binding dependencies could help.

### Human-plan development oracle (diagnostic only)

A separate, explicitly headline-ineligible diagnostic replaced automatic compilation with a human-authored Aster Prism plan that encodes the visible development fixture. It achieved 100.0% evidence recall, 100.0% decisive recall, and 100.0% precision versus 45.3% evidence/decisive recall for automatically compiled Witness. The GraphRAG-local and 4,096-token controls also reached 100.0% and 100.0% decisive recall, respectively. This isolates the automatic compiler as a major bottleneck and shows that the dependency executor can carry the synthetic chain when handed the answer-shaped plan. It is **not** a fair baseline win, evidence of generalization, or eligible for the macro table. See the [raw diagnostic](../results/multihop_development_oracle.json).

## Paired descriptive intervals

Entries are Witness minus comparator in percentage points, shown as paired mean difference `[95% bootstrap interval]` over 38 snapshot cases for these three metrics. (The counter-search interval above uses 25 applicable pairs.) Snapshots and nested depth/scale tasks are correlated, so these intervals are descriptive and must not be treated as publication-grade significance tests.

| Comparator | Evidence recall | Decisive recall | Precision |
|---|---:|---:|---:|
| long-context-65536 | -18.8 [-26.7, -10.9] | -14.4 [-22.5, -7.2] | +4.1 [-0.3, +8.0] |
| long-context-4096 | -11.2 [-23.1, +1.6] | -6.9 [-18.5, +4.8] | +4.8 [+0.2, +9.7] |
| vector-rag-tfidf-k10 | +13.3 [+1.1, +25.9] | +19.0 [+7.6, +30.7] | -16.5 [-22.7, -10.4] |
| graphrag-local-approx | +1.6 [-13.9, +18.5] | +8.9 [-7.4, +25.8] | -29.7 [-38.4, -21.4] |
| witness-exhaustive-offline-no-counter | +2.0 [+0.0, +5.3] | +0.0 [+0.0, +0.0] | -4.3 [-7.2, -2.0] |

## What the run falsifies or leaves unresolved

- Full execution does not imply semantic completeness. The bundled lexical evaluator marks lexical misses `UNCERTAIN`, so conclusive coverage can be far below physical scan coverage.
- Core certificates label this value `metric_name=program_execution_coverage`, emit per-obligation execution counts, and state `plan_adequacy=not_evaluated`. The headline runner projects the numeric conclusive fraction; it does not score plan adequacy.
- A broad high-recall predicate can produce very low precision and more reducer/context pressure than top-k retrieval.
- The automatic compiler is a one-scan lexical fallback; this run does not validate LLM question compilation or the multi-hop advantage of generated dependency programs.
- Structural exact-span provenance can be tested offline, but semantic entailment and unsupported factual claims require model/human judgments and were not scored here.
- The 1,000-row distributed task dominates micro evidence counts; macro-by-task is the primary view, and nested hop depths/snapshot copies are not independent samples.
- Exact-query result memoization improves immediate repeats, but is invalidated by any corpus-fingerprint change and says nothing about paraphrase hit rate. Workload hit rate, privacy isolation, invalidation races, and storage economics remain unmeasured.

## When to use Witness

Use a Witness-style path when missing a scattered exception is materially worse than extra compute: audits, policy history, contradiction-heavy synthesis, negative claims, and large deterministic aggregations. Keep indexed RAG/search for low-stakes simple lookups, exploratory questions, strict cold-latency workloads, or corpora too large for an economical semantic scan. A production router should make this choice explicitly.

## Required next experiment

Run the pre-registered model-backed track with disjoint unseen corpora and surface forms, a grammar-constrained compiler, calibrated local semantic evaluator, identical answer model/context budgets, production Microsoft GraphRAG local/global/DRIFT, real dense and cross-encoder baselines, human evidence/claim annotation, and enough independent questions for paired confidence intervals. Until then, the correct status is conditional go for continued research—not replacement of RAG.

Relevant comparison anchors: [Microsoft GraphRAG query modes](https://microsoft.github.io/graphrag/query/overview/), [the original RAG paper](https://arxiv.org/abs/2005.11401), [RAGChecker](https://arxiv.org/abs/2408.08067), and [RAGTruth](https://arxiv.org/abs/2401.00396).
