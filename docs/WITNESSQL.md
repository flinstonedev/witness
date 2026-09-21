# WitnessQL 0.1 target specification

> **Implementation status:** this document describes the target typed AST. The
> runnable Python prototype implements a deliberately smaller, closed subset:
> lexical/regex `PredicateSpec` scans, binding dependencies, explicit counter
> tests, and typed `ReducerSpec` operations. It does not yet execute arbitrary
> nested `op/args` predicate ASTs or the full joins/filters envelope shown below.
> `WitnessProgram.to_dict()` is the machine-authoritative schema for the current
> executable subset. Unsupported fields are rejected rather than ignored.

WitnessQL is a constrained JSON intermediate representation for evidence
execution. It is a plan, not an answer. A compiler may use an LLM, rules, or a
human-authored template, but the resulting plan must validate before it sees the
corpus evidence used to answer the question.

## Plan envelope

```json
{
  "version": "wql-0.1",
  "question": "What approval process applied on 2025-03-14?",
  "as_of": "2025-03-14T23:59:59Z",
  "inputs": {},
  "scans": [],
  "joins": [],
  "filters": [],
  "reducers": [],
  "counter_tests": [],
  "answer_contract": {},
  "limits": {}
}
```

Required invariants:

- step IDs are unique and dependencies form a directed acyclic graph;
- every bound variable has a declared type and source step;
- reducers consume only declared fields;
- `AS_OF` is explicit for questions whose answer depends on time;
- counter-tests name the claim or reducer result they could change;
- answer fields name the evidence rows required to render them;
- scans cannot grant tools, alter permissions, or execute source text.

## Scan

```json
{
  "id": "policy_events",
  "scope": {"source_types": ["policy", "amendment"]},
  "predicate": {
    "op": "all",
    "args": [
      {"op": "relation", "name": "governs", "args": ["approval"]},
      {"op": "time_intersects", "value": "$as_of"}
    ]
  },
  "emit": {
    "policy": "entity",
    "requirement": "string",
    "valid_from": "datetime?",
    "valid_to": "datetime?",
    "supersedes": "entity?"
  },
  "uncertainty": "emit_uncertain"
}
```

Predicates use three-valued logic: `TRUE`, `FALSE`, and `UNKNOWN`. An evaluator
may return `NO_MATCH` only when it can justify `FALSE` under its declared
semantics. `UNKNOWN` becomes an unresolved or uncertain evaluation, never a
silent negative. Semantic model confidence is not itself proof of falsehood.

Core predicate operators are:

- boolean: `all`, `any`, `not`;
- typed relations and properties: `relation`, `has_field`, `entity_equal`;
- comparisons: `eq`, `ne`, `lt`, `lte`, `gt`, `gte`, `between`;
- time: `valid_at`, `recorded_by`, `time_intersects`, `supersedes`;
- evidence qualifiers: `polarity_is`, `modality_is`, `scope_is`,
  `contains_qualifier`;
- lexical/metadata operators only when their semantics are explicit.

## Witness row

```json
{
  "witness_id": "sha256:...",
  "scan_id": "policy_events",
  "source_id": "policy-17",
  "source_version": "3",
  "block_id": "policy-17:0",
  "start_offset": 418,
  "end_offset": 476,
  "quote": "Manager approval is optional for emergency remediation.",
  "bindings": {"requirement": "approval optional for emergency remediation"},
  "polarity": "qualifies",
  "modality": "policy_requirement",
  "valid_time": {"from": "2025-02-01", "to": null},
  "record_time": "2025-02-04T11:00:00Z",
  "evaluator_version": "local-0.1",
  "verification": {"exact_quote": true, "offsets": true, "schema": true}
}
```

The quote is evidence. Bindings, polarity, modality, and time interpretations
are defeasible annotations. Stable witness IDs hash the plan predicate, source
version, offsets, and quote.

## Relational steps

Supported deterministic operations are `FILTER`, `JOIN`, `GROUP_BY`, `COUNT`,
`SUM`, `MIN`, `MAX`, `ARGMIN`, `ARGMAX`, `SORT`, `DEDUPLICATE`,
`TEMPORAL_RESOLVE`, `SET_DIFFERENCE`, `INTERSECTION`, and `UNION`.

Semantic equivalence may produce candidate group keys, but it must preserve the
member witness IDs and cannot rewrite source quotes. Arithmetic and temporal
selection are deterministic once their typed inputs have passed verification.

## Counter-tests

A provisional claim is compiled into one or more typed change conditions:

```json
{
  "id": "approval_exceptions",
  "claim_ref": "current_requirement",
  "would_change_if": ["newer_rule", "exception", "repeal", "scope_variant"],
  "scan": {"predicate": {"op": "any", "args": []}},
  "reduction": "TEMPORAL_RESOLVE"
}
```

The engine executes these scans under the same exhaustive/coverage rules as the
primary plan. Counter rows are included in the claim ledger whether they defeat,
qualify, or leave the claim unchanged.

## Mechanical verification

Before a row is visible to reduction, the engine verifies:

1. source ID, version, authorization, and content hash;
2. exact quote equality at the half-open character offsets;
3. typed schema and allowed enum values;
4. numbers and dates against the cited span or an explicit resolver trail;
5. entity mention or a cited, versioned entity-resolution row.

Invalid rows are rejected and the block is retried or counted unresolved.

## Coverage and optimization contract

Every execution records authorized, scanned, soundly skipped, approximately
skipped, and unresolved blocks. These sets must be disjoint and add up to the
authorized snapshot.

A physical optimization is semantics-preserving only if disabling it yields the
same verified witness multiset. Approximate gates are permitted experiments, but
their excluded blocks appear as approximately skipped and reduce conclusive
coverage. The correctness fallback is evaluation of every authorized block.

Materialized views use this logical key:

```text
hash(canonical scan predicate) × block content hash × evaluator version
```

Consequently, an unchanged block can reuse an evaluation, while an update
invalidates only changed block entries. Compiler and reducer caches are versioned
separately so a new interpretation never masquerades as an old result.
