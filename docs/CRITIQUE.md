# Critical review of Witness

Review date: 2026-08-30

Implementation reconciliation: 2026-09-21. This repository-maintained review is
not an independently commissioned audit. The counts and resolved engineering
findings below have been reconciled with the current source and tracked
25-system result. The production-research limitations remain open.

## Verdict

**No-go for any claim that Witness is superior to RAG, GraphRAG, long-context
models, agentic RAG, or an LLM-maintained wiki.** The repository is a useful
research scaffold, but it does not yet run the experiment needed to support
that claim. In particular, the bundled systems are deterministic offline
proxies, the task set is small and visible, the corpus is dominated by one
templated aggregation family, and the implemented Witness compiler/evaluator
are lexical fallbacks rather than the proposed semantic system.

**Conditional go as an alpha research prototype.** The lossless-source model,
exact-span checks, tri-state evaluation, deterministic relational reducer, and
explicit coverage object are useful pieces to test. Results from this version
must be labelled **offline proxy results** and **preliminary**. They may validate
mechanics and reveal bugs; they cannot establish comparative product quality or
production economics.

The right success target is a Pareto frontier for a declared workload, not
“iterate until Witness wins.” That instruction creates optional stopping and
benchmark overfitting pressure. Pre-register the target workload, primary
metrics, practical effect threshold, cost ceiling, and stopping rule before
opening a sealed test set. A valid result may be that Witness wins only on a
narrow, high-value workload.

## What is implemented versus what is being claimed

| Label in the proposal | Bundled offline implementation | What is required before a production comparison |
|---|---|---|
| Vector RAG | TF-IDF cosine over fixed chunks | A current dense embedding model, vector index, tuned chunking, and `k` sweep |
| Hybrid RAG | BM25 plus TF-IDF with min-max score fusion | BM25 plus dense retrieval with tuned fusion, including RRF and learned alternatives |
| Reranked RAG | Handwritten lexical/number/qualifier heuristic | A production cross-encoder and an LLM-reranker variant, with candidate-depth tuning |
| GraphRAG | Capitalized-phrase co-occurrence graph and connected components | The official implementation's Standard/Fast indexing and local, global, and DRIFT query modes |
| Long context | Stable corpus-prefix inclusion under a lexical token estimate | Actual provider tokenization, all fitting context sizes, several source orderings, and the same answer model |
| Agentic RAG | Rare-term/entity query expansion heuristic | A real query-decomposing agent with matched tool, token, model, and wall-clock budgets |
| LLM wiki | First-sentence extractive pages grouped by a heuristic topic | A model-synthesized, incrementally maintained wiki; define at least wiki-only and raw-fallback variants |
| Witness | Lexical question terms, regex bindings, and lexical block evaluation | A grammar-constrained semantic compiler, semantic local evaluator, provisional claim builder, counter adjudicator, and ledger-constrained renderer |

The approximations are clearly disclosed in code, which is good. They must not
appear in plots or conclusions under the unqualified family names. For example,
Microsoft's actual GraphRAG index extracts entities and relations and produces
community reports, while its query engine exposes local, global, basic, and
DRIFT modes; global search performs map-reduce over community reports. The
bundled co-occurrence proxy is not an implementation of those methods. See the
[official indexing overview](https://github.com/microsoft/graphrag/blob/main/docs/index/overview.md),
[official query overview](https://microsoft.github.io/graphrag/query/overview/),
and [official configuration surface](https://microsoft.github.io/graphrag/config/yaml/).

The offline track can answer “how do these transparent algorithms behave on
this fixture?” It cannot answer “does Witness beat production GraphRAG/RAG?”

## Observed offline falsification result

The repository's completed full run covers 38 snapshot-expanded cases, 25
systems, and two repetitions. The quality columns below use the report's
headline macro: first average duplicate snapshots within a task, then average
the 21 task definitions. Token totals cover the 38 first executions. These are
local proxy measurements, not production results, but they already falsify a
claim of clear superiority on the repository's own fixture:

| Offline proxy | Macro decisive recall | Macro counter recall | Macro evidence recall | Passage-unit precision | Conclusive coverage | Proxy input tokens | Mean local latency |
|---|---:|---:|---:|---:|---:|---:|---:|
| Witness + counter | 0.870 | 0.893 | 0.830 | 0.095 | 0.204 | 1,379,839 | 385 ms |
| Witness, no counter | 0.870 | 0.804 | 0.800 | 0.149 | 0.233 | 655,992 | 129 ms |
| TF-IDF RAG, `k=10` | 0.698 | 0.952 | 0.710 | 0.260 | 0.008 | 79,824 | 15 ms |
| Long context, `65,536` | **1.000** | **1.000** | **1.000** | 0.047 | **1.000** | 1,377,179 | **48 ms** |

Latency is mean first-recorded execution in one stateful process, not an
isolated cold start. These historical values come from the committed manifest.
The full run reproduced on another OS/Python on 2026-09-21 with all 1,900
deterministic records matching; timing is excluded from that comparison.
See [reproduction details](REPRODUCIBILITY.md).

The full-corpus long-context proxy has perfect decisive, counter, and overall
evidence recall, comparable aggregate input-token proxy, full coverage, and
much lower local latency. Witness has better precision
than retrieve-all long context, but both precision values are poor. The counter
phase more than doubles Witness's input proxy, lowers precision and conclusive
coverage, and adds no decisive-evidence recall; its observed gains are about
0.089 in counter recall and 0.03 in overall recall. The report found no case in
which Witness beat every proxy on decisive recall.

These figures are descriptive: cases are correlated, latency is threaded local
Python rather than model service latency, and the precision unit is not a
minimal span. They should not be assigned inferential confidence. Their value is
that the current implementation did not hide a loss. The correct response is to
analyze the failure and preserve the long-context control—not tune it away.

## Prioritized validity issues

### P0 — issues that invalidate a superiority claim

#### 1. There is no sealed, independent test distribution

The repository currently has 21 tasks over about 11 underlying scenarios. The
task definitions, expected answers, evidence IDs, generator, and tests are all
visible in the same checkout. The five multi-hop tasks are prefixes of one
chain; valid-time and record-time tasks reuse one policy history; before/after
tasks reuse the same incidents and contract; and the three distributed tasks
reuse one generation grammar.

Changing the generator seed changes decoy selection/order while intentionally
leaving the answer and evidence manifest unchanged. That is not a semantic
train/test split. A task-specific compiler, regex, renderer, or routing rule can
therefore “generalize” across seeds without generalizing to a new question.

Gold answers are correctly stripped before system execution, but the public
task still carries semantic task IDs, category, answer schema, and
`query_features`. Values such as `global_scan`, `deduplicate`, `group_by`, and
`source_count`-revealing task names are close to the obligations the Witness
compiler is supposed to infer. The current lexical adapter mostly ignores them,
but a future model-backed adapter could gain an unfair planning oracle. Use
opaque case IDs and keep category/query features scorer-side; expose only the
question, user-supplied temporal scope, and an answer schema that every system
receives identically.

Required correction:

- Create development, validation, and sealed test generators with disjoint
  entity sets, relation graphs, surface templates, document layouts, and task
  authors.
- Keep test questions, seeds, annotations, and generation templates outside the
  development repository, ideally with third-party execution.
- Select prompts, hyperparameters, ablations, and stopping decisions on
  development data; use validation sparingly; open test once.
- Add several independent natural corpora and user-authored queries. Synthetic
  evidence is useful for known completeness but cannot be the only evidence.

#### 2. The dataset is architecture-aligned and statistically tiny

The current v2 snapshot is roughly 34,000 lexical tokens in 1,199 very short
documents. About 1,119 documents are formulaic ledger fragments. One task alone
has 1,000 decisive spans, about 86% of all annotated evidence. Micro evidence
recall will therefore mostly measure whether a system returns the Tessera
fragments. Those fragments repeat the exact batch term from the question, so an
exhaustive lexical filter plus a normal database aggregate can solve the task.
This demonstrates a top-k capacity limit, not the unique value of query-compiled
semantic execution.

The fixture has no genuinely long documents: the largest is under 60 lexical
tokens. Consequently it does not exercise the proposed 2,000–8,000-token blocks,
cross-block coreference, tables split across pages, footnotes, email threads,
OCR, or hierarchical source context. It also lacks the open-ended global
synthesis questions on which GraphRAG is intended to be strongest.

Required correction:

- Use at least tens of independent scenarios per category and run a power
  analysis before setting the final sample size.
- Cluster statistical resampling by underlying corpus/scenario, not by the 21
  correlated task rows.
- Vary corpus size, evidence prevalence, relevant-source count, hop depth,
  document length, block-boundary location, paraphrase distance, ambiguity,
  contradiction density, and context budget independently.
- Add open-ended theme/discovery tasks with blinded human rubrics. Otherwise the
  GraphRAG comparison omits its target use case.
- Add realistic multilingual, tabular, OCR, conversational, and mixed-format
  corpora.

#### 3. The comparison does not isolate the proposed contribution

Witness gets exhaustive map-like execution and deterministic reduction, while
the named baselines are primarily restricted to small top-k answer contexts.
The full profile now includes `diagnostic-exhaustive-lexical-map-no-cache`,
which uses the same fallback lexical predicate without WitnessQL execution or
caching. It matches the no-counter Witness evidence coordinates in regression
tests. This resolves the missing lexical exhaustive-map control, but there is
still no common answer reducer, exhaustive LLM map-reduce baseline, or conventional
analytics/search pipeline that retrieves all exact batch matches before SQL
aggregation. As a result, a win cannot be attributed to compilation,
counter-witnesses, proof rows, or caching; it may simply be “scan all rows beats
top eight rows.”

Add the following controls:

1. Same semantic evaluator, exhaustive map, and same reducer, but a
   human-authored predicate and no Witness compiler.
2. Same evaluator and reducer with generated program but no counter phase.
3. Same total second-pass compute spent on another support scan rather than a
   counter scan.
4. Oracle program plus model evaluator; generated program plus oracle evaluator;
   full model pipeline; and gold evidence plus renderer. These localize compiler,
   evaluator, reducer, and renderer error.
5. Retrieve-all BM25/metadata/inverted-index plus deterministic reduction.
6. An ordinary relational/ETL solution when the data is naturally structured.
7. A long-context model receiving the entire corpus whenever it fits.
8. Closest declarative semantic-data systems, not only RAG systems.

An earlier adapter behavior applied `as_of` as a document valid-time filter to
every baseline, which was wrong for record-time and contradiction tasks. The
current runner correctly disables that prefilter for parity. Preserve that fix
and, in the production track, give every system the same explicit `valid_at`
and `known_at` contract. Score retrieval both before and after any *soundly
specified* temporal filter so no system is penalized for evidence the harness
removed before it ran.

The closest related systems matter to the novelty claim. LOTUS defines natural
language semantic filters, joins, sorts, and aggregates with alternative
physical implementations and accuracy-aware optimizations
([paper](https://arxiv.org/abs/2407.11418)). Palimpzest compiles declarative
AI-data workloads and selects plans by quality/cost trade-offs
([paper](https://www.vldb.org/cidrdb/papers/2025/p12-liu.pdf)). DocETL rewrites
LLM document-processing pipelines for accuracy
([paper](https://www.vldb.org/pvldb/vol18/p3035-shankar.pdf)). SUQL compiles
natural-language questions into queries over structured and unstructured data
([paper](https://aclanthology.org/2024.findings-naacl.283/)).

Question compilation and semantic relational operators are therefore not, by
themselves, novel. Witness's defensible novelty may be the combination of exact
proof spans, tri-state exhaustive obligations, counter-search, bitemporal claim
ledgers, and demand materialization. The benchmark must ablate and compare that
combination directly.

#### 4. Resource parity is not defined

Matching only the final answer-context budget is not sufficient. Witness may
send every shard through a semantic model, potentially multiple times for scans,
bound-variable expansions, and counter-tests, before the renderer sees a small
relation. A top-k baseline may use one embedding query and one answer call. Those
systems have not received equal total inference compute.

Report three comparisons, none of which replaces the others:

- **Quality-unmatched:** best attainable quality for each system.
- **Budget-matched:** equal billed input/output tokens, model class, and tool-call
  budget.
- **Latency-matched:** equal p95 service-level deadline and concurrency quota.

Also plot a quality/cost/latency Pareto frontier. Tune each baseline on the
development split; do not select the best baseline hyperparameter on test.
Indexing, query, rendering, cache fill, and update must be separate cost phases.
Use actual provider tokenizers and billed usage in the production track. The
offline lexical counters, synthetic `model_calls`, and rough storage estimates
are only proxies.

The whole current corpus fits comfortably inside common modern long-context
budgets. The runner's full profile now correctly includes a 65,536-token
retrieve-all control, and it beats Witness on the offline recall metrics above.
The 4,096-token smoke variant must not be used as the headline comparison. The
production track still needs actual provider tokenization, provider maximums,
stable/random/relevance orderings, and lost-in-the-middle behavior with the
actual model.

#### 5. “Iterate until superior” is an invalid stopping rule

Repeatedly changing Witness after seeing benchmark outcomes, while leaving
baselines and the test set fixed, guarantees adaptive overfitting. A confidence
interval computed after that process does not repair it.

Pre-register:

- one or two primary endpoints;
- a minimum practical effect, not merely an interval excluding zero;
- non-inferiority margins for answer accuracy and unsupported claims;
- the hyperparameter search space for every system;
- correction for multiple baselines, metrics, categories, and ablations;
- a maximum number of validation looks; and
- a terminal outcome of **falsified**, **narrowed**, or **supported**.

Use hierarchical or cluster-aware uncertainty over independent corpora and
scenarios, with repeated stochastic runs nested inside each task. Five nested
multi-hop questions are not five independent corpora.

#### 6. The proposed end-to-end Witness pipeline is incomplete

The engine can ingest, execute a supplied plan, verify spans, reduce rows,
reconcile supplied claims, and render an existing ledger deterministically.
`LedgerAnswerRenderer` checks citation existence and copies supported/qualified
claim text; it does not judge semantic entailment or generate new prose.
The engine does not yet construct semantic provisional claims from the
reduced result. Automatic counter-tests
need provisional claims; without externally supplied claims or explicit tests,
the intended answer → falsifier → re-reduction loop cannot occur.

The default compiler and evaluator are intentionally lexical. That is honest,
but it means the core hypothesis—query-specific *semantic* execution—has not
been implemented or measured. An end-to-end result must include compiler model,
prompt, schema-constrained decoding, evaluator model, evaluator prompt, retry
policy, counter adjudicator, answer model, and claim-to-evidence validator.

### P1 — semantic and execution correctness risks

#### 7. Coverage currently certifies execution of a plan, not adequacy to the question

This is the most dangerous conceptual ambiguity. A compiler can mistranslate
“customers who cancelled” as “customers who complained.” The resulting program
may scan every shard perfectly. Its certificate proves exhaustive execution of
the wrong predicate.

The certificate now labels its metric `program_execution_coverage` and sets
`plan_adequacy` to `not_evaluated`. Keep those distinctions and add a separate
plan-adequacy judgment from compiler evaluation. Negative answer language must
state both. For example: “The validated program found no match over 98.6% of its
obligations; plan adequacy for the original question is estimated at 0.91.” Do
not render an unqualified corpus-level negative from execution coverage alone.

The certificate now includes scheduled, conclusive, uncertain and failed
obligation counts in addition to shard coverage, with per-predicate summaries.
That resolves missing obligation accounting. Claim-dependent lineage and a
fully inspectable obligation matrix remain useful extensions, keyed by:

```text
(scan_id, rendered_predicate_hash, binding_context_hash, role, shard_id)
```

Shard-level aggregate coverage alone can hide which predicate was applied to a
block. Preserve the obligation counts and derive each claim's coverage only
from the obligations on which that claim depends.

`source_types` exclusions are presently counted as sound skips because the
metadata comparison is exact. Exact comparison is not the same as sound
question semantics. An arbitrary program that asks about “any source” but scopes
the scan to `email` can call a policy block soundly skipped even when it contains
the answer. Require a compiler proof obligation tying each scope restriction to
the logical predicate; otherwise label the exclusion `plan_scoped`, not sound.

#### 8. Semantic truth is not mechanically verified

Exact quote and offset validation proves that bytes exist at a location. It
does not prove that a quote entails the binding, polarity, modality, temporal
scope, or final claim. A very large exact quote can contain both a claim and its
negation. Mechanical validity must never be reported as semantic support.

Use two explicit layers:

- `mechanically_valid`: source/version/hash/offset/surface/type checks;
- `semantically_adjudicated`: predicate relevance, binding entailment, scope,
  polarity, and claim entailment, with model/human version and confidence.

Require minimal-span or span-IoU evaluation, typed resolver trails, and blinded
human or calibrated NLI judgments for headline unsupported-claim rates. The
current structural metric considers a claim supported when it cites a valid
source span; it does not establish entailment. A system-supplied
`metadata.supported` value must never be accepted as scorer authority.

“Lossless” currently means lossless relative to the Python text string handed
to the store. It does not preserve an original PDF, image, email container,
spreadsheet, OCR alternatives, layout, or parser transformation. Production
provenance needs the original artifact hash, extraction/parser version, byte or
page coordinates, and a reproducible rendering trail in addition to character
offsets in derived text.

#### 9. Temporal and snapshot semantics are not yet bitemporal

The corpus and witness models now carry document-level `valid_from`, `valid_to`,
and one `record_time`, and the reducer honors the half-open valid-time end bound.
That is a sound prototype improvement. It still selects principally by latest
eligible start/record time and does not resolve repeal/supersession, scoped
exceptions, correction status, or separate fact intervals within one document.
A point record time is also not a transaction-time interval.

The corpus store accumulates versions and supports removing an exact document
version and its persisted store file. It does not expose named snapshot
membership or a complete data-erasure lifecycle; old cache files and result
objects are not physically purged by that operation. `supersedes_version` is checked at
ingest but is not a first-class reduction relation. The snapshot hash now
includes the document's semantic metadata, which is appropriate, but there is
still no named membership or coordinated cache/deletion lifecycle in the core store.

Required model:

```text
valid_interval       [valid_from, valid_to)
transaction_interval [known_from, known_to)
assertion_status      proposed | enacted | corrected | superseded | repealed
scope                 jurisdiction, customer, product, amount range, etc.
supersedes            explicit evidence-linked relation
snapshot_membership   named immutable manifest
```

Test late-arriving facts, backdated corrections, overlapping scopes, missing
timestamps, timezone offsets, future-effective rules, retractions, deletions,
and metadata-only changes. A clean four-document policy fixture is insufficient.

#### 10. Counter-witness retrieval lacks contradiction adjudication

The generated lexical counter predicate now requires one term from a broad
claim-anchor group and one cue such as “not,” “except,” or “corrected.” This is
better than one unconstrained OR, but it has no proximity, entity identity,
time, jurisdiction, or proposition-scope check. Any coincidental anchor/cue pair
can qualify a claim; a true implicit counter-witness can contain no cue. The
current reconciliation has little basis to distinguish refutation, exception,
supersession, harmless disagreement, and unrelated text.

Compile typed change conditions from each claim, bind its entities/scope/time,
and run a separate contradiction/qualification adjudicator. Score:

- counter-evidence recall and precision;
- fraction of provisional wrong answers corrected;
- fraction of provisional correct answers degraded;
- net utility at matched extra compute; and
- adjudicator calibration by contradiction type.

An ablation against an equally expensive second support scan is necessary. A
recall gain from simply scanning the corpus twice is not evidence that
counter-witness generation is responsible.

#### 11. The implemented IR is substantially narrower than the documented IR

The executable predicate is primarily lexical terms, regexes, and regex binding
patterns. It does not implement the documented relation/time/qualifier operator
tree, declared emit schemas, or typed evidence obligations. Nested IR objects
now reject unknown fields, with a regression test for executable nested input.
Keep that closed grammar as the language expands. Reducer field/type errors should be caught
before any corpus work.

The previously identified missing/null-key join defect is fixed: missing and
null keys do not match, composite keys require every component, and left joins
retain unmatched rows. Regression tests cover these cases in
`tests/test_witness_core.py`. Add property-based and differential SQL tests for every
reducer, including nulls, mixed types, duplicates, ties, intervals, and
provenance lineage.

Arbitrary model-generated regular expressions are also an execution surface:
catastrophic backtracking can turn crafted corpus text into denial of service.
Use a linear-time regex engine or a restricted pattern language, compile with
resource limits, and validate all nested objects.

The compiler needs explicit ambiguity branches and assumptions. Questions such
as “current,” “reported,” “owns,” and “no customer” often require clarification,
not silent plan generation.

#### 12. Full semantic scanning exchanges retrieval misses for classifier errors

An exhaustive schedule does not imply exhaustive evidence. Every shard still
passes through an approximate local classifier. If the relevant span is judged
`NO_MATCH`, the architecture has reproduced a retrieval false negative at a
different layer. At large scale, even a small per-shard false-positive rate can
produce an overwhelming witness relation; a small false-negative rate can erase
the only decisive exception.

For semantic predicates, a generative model generally cannot *prove* falsehood.
Treat `NO_MATCH` as model-relative, calibrate it on adversarial held-out data,
and report probabilistic rather than “sound” coverage unless the exclusion is a
formally exact metadata/lexical predicate. Measure risk-coverage curves and
answer rate so a system cannot appear safe by returning `UNCERTAIN` everywhere.

The completed ablation demonstrates the inverse gaming risk: coercing
`UNCERTAIN` to binary outcomes raises reported conclusive coverage from 23.3%
to 100% while decisive recall remains 87.0%. The new coverage is not new
knowledge; it is a relabeling of uncertainty. Coverage claims must therefore be
calibrated against audited false negatives, not optimized as a standalone
metric.

A practical design is a cascade: cheap high-recall judge, stronger adjudicator
for uncertain and boundary cases, random audits of skipped shards, and a full
reference evaluator on a sampled subset. Keep the exact exhaustive fallback for
high-risk queries, but do not equate model judgment with logical proof.

#### 13. Cross-block and entity state are not solved

All current benchmark documents fit in one block. Real evidence may require a
heading on one page, a pronoun on the next, a table header far above a cell, or a
definition in an appendix. Independent local evaluation cannot resolve that
without document state. Exact string joins also fail on aliases, renamed
entities, ambiguous names, and many-to-many relationships.

Add hierarchical execution: block scan, neighboring-window expansion,
document-level reconciliation, and evidence-linked entity-resolution rows. Keep
multiple entity hypotheses until a scoped adjudicator resolves them; an early
wrong binding otherwise drives every later hop down the wrong chain. Stress the
Cartesian binding expansion and unknown-depth frontier under realistic fan-out.

#### 14. Demand materialization has an unfavorable long-tail regime

For `N` shards and `Q` mostly unique programs, cold Witness evaluation is
approximately `O(QN)` semantic judgments. Conventional indexing is closer to
`O(N)` ingest plus cheap per-query retrieval. Maintaining `M` active programs
over `ΔN` changed shards costs `O(MΔN)`, and caching negative/uncertain results
can approach `O(MN)` records.

The current cache key includes the full program hash, so semantically equivalent
paraphrases do not naturally share local predicate results. Canonicalize reusable
predicates independently of question wording, while including every semantic
dependency: evaluator weights, prompt, decoding policy, parser/structure
version, source metadata, temporal fields, and authorization scope. Cache
permission-neutral decisions only when this is demonstrably safe; otherwise use
tenant-scoped encryption and deletion.

The reported exact-repeat speedup is primarily an exact full-result cache in the
benchmark adapter: the second identical `(corpus fingerprint, question, AS_OF,
versions, counter mode)` lookup returns a stored `RetrievalResult` without
executing the program. That is useful engineering, but it is generic memoization,
not evidence for Witness's per-predicate demand-compiled view. Every baseline
could use the same response cache. Run four separate variants—no cache,
shard-evaluation cache only, exact-result cache only, and both—and give the exact
result cache to all systems. Report semantic-equivalent paraphrase reuse
separately from byte-identical query replay.

Benchmark Zipfian repeated queries, long-tail unique queries, corpus churn,
active-program count, TTL, policy changes, and hard deletion. Report the measured
break-even repetition count. Demand-compiled views are likely valuable only
above that point.

#### 15. Security is specified as prompting guidance, not an enforced boundary

The evaluator interface is arbitrary in-process Python and therefore can have
filesystem, network, and credential access. “No tools” is not enforced by the
engine. Cached local evaluations contain source quotations in plaintext. The
single uppercase `IGNORE` fixture is not a meaningful prompt-injection red team,
and treating retrieval of the malicious payload as counter-evidence recall can
reward passing it toward the renderer.

Run model evaluators in an isolated process/container with no credentials,
default-deny egress, CPU/memory/time limits, tenant separation, audit logs, and
strict structured IPC. Separate `detected_and_quarantined` from
`included_in_answer_context`. Test indirect instructions, markup and delimiter
breakout, encoded payloads, poisoned metadata/entity names, very long attacks,
cross-document attacks, output-schema attacks, and adaptive attacks against the
actual model-backed pipeline.

### P2 — benchmark and reporting gaps

#### 16. Several promised metrics are only proxies or absent

- Evidence precision uses passage units for retrieval systems and row units for
  evidence-only systems. Those denominators are not directly comparable. Report
  source-, passage-, span-, and byte/token-level precision separately.
- Unsupported-claim rate is structural unless external judgments are supplied.
  Exact citation coordinates do not measure hallucination or entailment.
- Structured answer correctness is all-or-nothing exact equality. Add schema
  validity and per-field accuracy, with entity aliases and human adjudication.
- Temporal and aggregation metrics currently duplicate whole-answer exact match
  rather than isolating temporal selection or arithmetic error.
- The actual negative tasks encode `result`, while the dedicated calibration
  parser expects a status-like field; ensure the promised “not found” versus
  “does not exist” metric is exercised by the real manifest.
- Contradiction preservation, qualifier retention, prompt-injection success,
  claim-level citation completeness/correctness, and hallucination severity need
  first-class metrics.
- Reproducibility must distinguish cold independent runs from warm cache replay.
  Deterministic offline proxies and cache hits trivially score 1.0.
- Report both macro and micro values, but never use micro evidence recall as the
  headline on the current 1,000-span-dominated fixture.

ALCE provides a useful starting point for citation correctness and completeness
evaluation ([paper and code](https://github.com/princeton-nlp/ALCE)); RAGTruth
provides human-annotated hallucination spans
([paper](https://arxiv.org/abs/2401.00396)). Neither should be copied blindly,
but both are stronger than treating a valid citation as proof of entailment.

#### 17. Ground-truth completeness is unvalidated

“All relevant evidence” is a strong annotation claim. Several unannotated decoy
sentences are arguably relevant to disambiguation, while broad annotated
near-misses may be useful as exclusions but not positive evidence. Precision is
therefore partly determined by one author's taxonomy.

Use two independent annotators plus adjudication, blind them to system output,
report agreement by evidence role, and run pooling: merge candidates from all
systems, then judge previously unseen spans. Assign decisive weights before
seeing system results. Keep “support,” “necessary bridge,” “counter,”
“qualifier,” and “distractor” as separate labels.

#### 18. Operational telemetry is not production telemetry

Threaded regex timing on a 34k-token corpus says little about distributed LLM
latency. p95/p99 from a handful of local samples are not service percentiles.
Storage estimates omit or inconsistently include indexes, wiki pages, model
artifacts, cache replication, and raw corpus copies.

In the current adapter, Witness storage is approximately raw corpus bytes plus
returned passage bytes; the in-memory/disk materialized evaluations—especially
the many negative and uncertain program-by-shard entries—are not included.
Conversely, the shared baseline index estimate includes text, postings, and
graph bookkeeping even for systems that do not use every component. The storage
numbers are therefore not comparable and must not support a caching-economics
claim.

Production measurements need fixed hardware/region, provider and model version,
request concurrency, rate limits, retries, batch size, queueing, tokenizer,
cache state, and failure injection. Run enough repetitions for tail estimates.
Report total compute separately from wall latency so parallel fan-out is not
presented as free.

#### 19. Most requested ablations and the end-to-end track are not wired into the runner

The full runner now includes counter-search, binary evaluation without
`UNCERTAIN`, Witness behind top-k gates, and materialized-query-view disabled,
along with baseline `k`/context sweeps. It still does not execute the requested
Witness ablations for sound-only pruning, semantic synthesis without the
deterministic reducer, interpretations without exact quotes, or timestamps
removed. Snapshot cases exercise cache reuse indirectly but do not constitute a
controlled update-cost/stale-answer experiment.

The manifest correctly labels the run `retrieval_execution_only`; consequently
answer correctness, semantic unsupported-claim rate, negative calibration,
qualifier preservation, temporal answer accuracy, and aggregation answer
accuracy remain unmeasured. Do not describe these as implemented benchmark
results merely because metric helper functions exist. Each ablation needs one
registered system configuration, a single-variable intervention check, and a
runner test proving that the intended component—not an incidental budget or
output-format change—is the only difference.

## Where Witness is likely to win and lose

| Workload | Expected outcome | Reason |
|---|---|---|
| High-value exhaustive audit with scattered exceptions | Witness candidate win | Missing one decisive row is costly and full coverage has user value |
| Repeated stable query over slowly changing corpus | Witness candidate win | A maintained demand view can amortize the cold scan |
| Multi-hop chain with explicit, well-resolved bindings | Witness candidate win | Bound follow-up predicates can recover lexically dissimilar later hops |
| Contradiction-heavy policy history with typed scope/time | Witness candidate win after temporal/counter fixes | Evidence ledger can preserve rather than compress disagreement |
| Numerical aggregation over reliably extracted rows | Witness or ordinary ETL win | Deterministic reduction avoids arithmetic synthesis errors; this is not unique to Witness |
| One-document lookup | RAG/BM25 likely win | Indexed retrieval is much cheaper and often equally accurate |
| Corpus already fits the model context | Long context may win | One model call can avoid thousands of local judgments and compiler error |
| Mostly unique queries over a huge corpus | RAG/GraphRAG likely win | Query-independent indexing amortizes; Witness pays `O(N)` semantic work per query |
| High update rate with many active programs | Global index or streaming ETL may win | Witness update work grows with active-program count |
| Open-ended discovery or thematic synthesis | GraphRAG/wiki may win | A narrow predicate is difficult to define; global abstractions are the product |
| Strongly implicit, cross-block, or multimodal evidence | Current Witness likely loses | Independent block predicates lack document/global state |
| Very noisy corpus with weak source authority | Current Witness may lose | Exhaustive evaluation amplifies false positives and low-quality contradictions |
| Strict interactive latency or low-value questions | Indexed RAG likely win | Cold semantic fan-out is hard to justify economically |
| Exact lexical or structured question | Search/SQL should win | An LLM semantic scan adds cost and nondeterminism without value |

Witness should therefore be a risk-aware execution mode inside a hybrid query
planner, not the mandatory path for every question. A fast indexed plan can
answer low-risk lookups; a verifier or policy can escalate exhaustive/counter
execution when the question requires completeness, negative claims, temporal
resolution, or high assurance. An approximate first stage must remain visible
in the certificate and retain an exhaustive fallback.

## Recommended architectural changes

Priority order:

1. **Extend coverage to claim-dependent lineage.** Preserve the implemented
   obligation counts and separation of plan execution from unmeasured adequacy.
2. **Complete the end-to-end loop.** Reduced relation → provisional claims →
   typed counter obligations → adjudication → re-reduction → ledger-constrained
   semantic renderer. A deterministic existing-ledger renderer is implemented.
3. **Add semantic validation.** Keep mechanical verification, but add calibrated
   predicate, binding, and claim entailment judgments with explicit provenance.
4. **Implement true bitemporal snapshots.** Named manifests, intervals,
   supersession/repeal, scope, deletion, and canonical semantic hashes.
5. **Replace lexical WitnessQL with the documented typed operator tree.** Strict
   nested schemas, pre-execution type checking, restricted patterns, and
   ambiguity branches.
6. **Add hierarchical context and evidence-linked entity resolution.** Preserve
   alternate hypotheses through multi-hop execution.
7. **Turn counter-search into typed change-condition evaluation.** Measure net
   corrections, not only candidate recall.
8. **Build a cost-based hybrid planner.** Indexed fast path, semantic cascades,
   audit sampling, and full-scan escalation by risk/SLA.
9. **Canonicalize and secure materialized views.** Predicate-level reuse,
   complete version keys, compressed negative state, tenant isolation,
   retention/deletion, and differential cold-vs-warm tests.
10. **Isolate all untrusted evaluation.** Sandboxed workers and an attack suite
    against the real compiler/evaluator/renderer.

## Production validation matrix

| Stage | Corpus/query cells | Required systems | Primary outputs | Exit condition |
|---|---|---|---|---|
| Engine invariants | Tiny hand-written corpora; mutation, ACL, temporal, cache, and failure cases | Witness only, optimization on/off | Differential answer equality, exact lineage, obligation accounting | Zero invariant violations; every fault is unresolved, never a silent negative |
| Sealed synthetic factorial | Independent hidden generators across all requested categories; vary size, hops, prevalence, block boundaries, ambiguity, contradiction, updates | All offline controls plus model-backed systems | Evidence portfolio recall/precision, compiler accuracy, answer fields, risk-coverage curves | Pre-registered practical effect on target cells; no regression outside margin |
| Public real retrieval/QA | BRIGHT, MultiHop-RAG, CRAG, RGB/NoMIRACL, conflict, citation, and long-context sets | Tuned production RAG families, declarative semantic systems, Witness | Dataset-native metrics plus normalized evidence/claim metrics | Gains repeat across independent datasets, not one generator |
| Internal blinded workload | Several domains, real source formats, historical queries, and user-authored questions | Production candidates under identical access and answer policy | Blinded human correctness, evidence completeness, severity-weighted unsupported claims | Domain owners accept quality and abstention behavior |
| Cost/load | `10^4` to `10^7+` shards; cold/warm Zipf query mixes; concurrency and rate limits | Deployable configurations only | Dollars/query, total tokens, throughput, p50/p95/p99, storage, break-even repetitions | Candidate lies on required quality/cost/latency Pareto frontier |
| Updates | Append, supersede, backdate, retract, delete, ACL/metadata change; varying active-program count | Witness cache, GraphRAG/wiki update, vector incremental index | Stale-answer rate, update lag, recomputed shards, deletion verification | No stale disclosed evidence; update SLA and storage growth met |
| Security/privacy | Adaptive injection, schema attack, poisoned metadata, tenant crossing, worker compromise, erasure | Actual production stack | Attack success, exfiltration, cross-tenant disclosure, residual cache data | Zero critical disclosures; explicit threat-model sign-off |
| Longitudinal shadow | Weeks of mirrored traffic and corpus churn, no user-visible answers | Final candidates | Drift, evaluator calibration, cache hit distribution, incident rate | Stable metrics and measured economic break-even |

Useful independent benchmark anchors include:

- [BRIGHT](https://arxiv.org/abs/2407.12883) for reasoning-intensive retrieval;
- [MultiHop-RAG](https://arxiv.org/abs/2401.15391) for multi-hop questions with supporting evidence;
- [CRAG](https://arxiv.org/abs/2406.04744) for diverse and temporally dynamic factual QA;
- [RGB](https://arxiv.org/abs/2309.01431) for noise, negative rejection, integration, and counterfactual robustness;
- [NoMIRACL](https://arxiv.org/abs/2312.11361) for multilingual rejection of irrelevant retrieval;
- [RAMDocs](https://arxiv.org/abs/2504.13079) for ambiguity and conflicting/misleading evidence;
- [LongBench v2](https://aclanthology.org/2025.acl-long.183/) for realistic long and multi-document reasoning; and
- [StreamingQA](https://proceedings.mlr.press/v162/liska22a.html) for adaptation to new knowledge over time.

No public benchmark alone proves the Witness claim; each covers only part of the
contract. The strongest evidence will be convergence across sealed synthetic
ground truth, independent public corpora, and blinded production traffic.

## Concrete go/no-go gates

Before calling Witness “clearly superior,” require all of the following on a
sealed target-workload test set:

1. A model-backed Witness and production-equivalent baselines, all tuned without
   test access.
2. A predeclared, practically meaningful improvement in macro decisive-evidence
   recall and adjudicated counter-evidence recall over the **best tuned**
   baseline, with cluster-aware intervals.
3. Answer correctness and semantic unsupported-claim rate within predeclared
   non-inferiority margins, including severity-weighted review of every critical
   unsupported claim.
4. Gains on multiple independent scenarios and at least two natural-data
   domains; no conclusion based on the templated 1,000-row task.
5. Correct optimizer-off/optimizer-on differential behavior and calibrated
   obligation-level coverage. Any claimed sound skip violation is a release
   blocker.
6. A measured cost and latency envelope acceptable for the declared workload,
   including cold unique queries, warm repeats, updates, and active-view storage.
7. A clear loss table: simple lookup, long-context-fit, open-ended synthesis,
   and high-churn/long-tail workloads must be reported even if they weaken the
   headline.
8. Security, ACL, deletion, and prompt-injection validation on the actual
   model-backed deployment.

Until these gates are met, the strongest defensible statement is:

> The prototype demonstrates a testable design for proof-carrying,
> query-specific evidence execution. On a small synthetic offline fixture it
> can validate source spans, preserve explicit uncertainty, and deterministically
> reduce extracted rows. Comparative quality and production practicality remain
> unestablished.
