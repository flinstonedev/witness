# Reproducing the offline benchmark

This is a retrieval/execution-only regression fixture. Its visible synthetic
tasks, lexical Witness evaluator, and approximate baseline algorithms cannot
establish production quality or superiority. Reproducing results does not make
the experiment statistically independent or semantically complete.

## Install and run

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install .
python -m unittest discover -s tests -v
python examples/policy_evidence.py
```

The package supports Python 3.11+ and has no runtime dependencies. CI is
configured for 3.11, 3.12, 3.13 and 3.14. The installed CLI and
`PYTHONPATH=src python3 -m witness_bench` are equivalent entry points. Always
choose explicit local output paths: the legacy CLI defaults replace the
tracked main result/report.

Two fresh-process smoke runs exercise all 21 tasks and 38 snapshot cases with
11 systems and two repetitions each (836 records per file):

```bash
witness-bench --profile smoke --seed 20260829 --repetitions 2 \
  --output artifacts/smoke-a.json --report artifacts/smoke-a.md
witness-bench --profile smoke --seed 20260829 --repetitions 2 \
  --output artifacts/smoke-b.json --report artifacts/smoke-b.md
python scripts/check_benchmark.py artifacts/smoke-a.json artifacts/smoke-b.json
```

Reproduce the complete historical run without overwriting it:

```bash
witness-bench --profile full --seed 20260829 --repetitions 2 \
  --bootstrap-resamples 2000 \
  --output artifacts/full.json --report artifacts/full.md
python scripts/check_benchmark.py results/benchmark.json artifacts/full.json
```

Run from the repository root so the historical, separately labelled multi-hop
diagnostic can be included in the generated full report. It is never added to
headline main-run scores. Its original manifest reports benchmark version
0.1.0 and lacks the newer implementation fingerprint; it is a historical
development-only diagnostic, not proof of independent generalization.

## What is compared

`scripts/check_benchmark.py` validates run counts and unique complete
system/case/repetition identities. It compares the seed, corpus snapshot
hashes, implementation fingerprint, system metadata, evidence/passage hashes,
per-run scores, coverage and deterministic counters. Machine/platform strings,
generation timestamps, wall-clock and CPU timing are excluded. It compares
per-run records rather than derived timing summaries or bootstrap tables.

Changing evidence, a score, a corpus hash, a configuration, or the implementation
fingerprint fails verification. The checker deliberately prints no differing
source values. A match is a reproducibility check, not an independent validation
of relevance annotations, the scorer, semantic truth, or a proof that arbitrary
future outputs contain no personal data.

The implementation fingerprint hashes the 26 Python files in the two source
packages. Documentation, examples, tests and build configuration are not part
of that fingerprint. Run `python -m unittest` and the example separately.
Changing source files requires a newly documented result; do not hide a
fingerprint mismatch to preserve an old claim.

## Artifact provenance and local reproduction

The committed `results/benchmark.json` was generated on 2026-08-30 using Python
3.14.4 on Linux x86_64. It contains 1,201 total document versions, 1,166 evidence
spans, 21 tasks, 38 snapshot cases, 25 systems and 1,900 records. Snapshot v1
contains 1,198 active documents and v2 contains 1,199; these are distinct from
the total number of retained versions.

On 2026-09-21, the full run was reproduced locally using Python 3.13.7 on macOS
arm64, seed `20260829`, two repetitions and 2,000 bootstrap resamples. All
1,900 deterministic run records matched the committed artifact, including the
implementation fingerprint and corpus hashes. The original timing measurements
remain unchanged in the tracked result/report. New timings are machine-specific
and are not a speed improvement claim. No external model/service was used.

The benchmark corpus is generated entirely from authored fixture strings and
deterministic templates in `src/witness_bench/data.py`; expected-answer labels
are in `tasks.py`. There is no imported customer corpus. The large JSON stores
scored records and identifiers rather than raw document quotations. Both the
main and diagnostic artifact's snapshot hashes were checked against the
generator. See [data handling](DATA_HANDLING.md) for the privacy review scope.
