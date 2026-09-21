# Contributing

Start with [the architecture](docs/ARCHITECTURE.md),
[benchmark limitations](docs/CRITIQUE.md), and
[reproduction instructions](docs/REPRODUCIBILITY.md). Keep changes focused on a
demonstrable bug or declared experiment. A result that falsifies a hypothesis is
a useful result; do not tune the visible fixture until the headline improves.

## Local checks

Use Python 3.11 or newer. CI covers Python 3.11–3.14. There are no runtime
dependencies; building the wheel requires setuptools as declared in
`pyproject.toml`.

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install .
python -m unittest discover -s tests -v
python examples/policy_evidence.py
witness-bench --profile smoke --seed 20260829 --repetitions 2 \
  --output artifacts/smoke-a.json --report artifacts/smoke-a.md
witness-bench --profile smoke --seed 20260829 --repetitions 2 \
  --output artifacts/smoke-b.json --report artifacts/smoke-b.md
python scripts/check_benchmark.py artifacts/smoke-a.json artifacts/smoke-b.json
```

Install again after changing package source, or use `PYTHONPATH=src` for source
development. Add regression tests for changed engine contracts and protect ACL,
source-coordinate, uncertainty, cache-version, and scorer-isolation behavior.
Use generated or invented data. Never attach real corpus exports or source-bearing
caches to a pull request. Ignored paths do not protect already tracked files.

## Benchmark changes

Record the hypothesis, changed configuration, seed and intended comparison
before running an experiment. Keep offline proxies and the human-plan diagnostic
labelled as such. Do not replace the historical tracked benchmark merely because
timing varies. If behavior changes, retain provenance, explain the change and
publish a fresh manifest/report together after review. The deterministic checker
fails on code-fingerprint changes intentionally; review those changes instead of
silently discarding the fingerprint.

## Rights and review

Contributions are under the repository's Apache-2.0 license. Submit only work you
have the right to contribute; retain existing copyright/license notices and
identify any imported code or datasets and their source/license in the pull
request. The project does not require or imply copyright assignment. The root
license does not license private corpora supplied by users. No project-wide
third-party ownership clearance should be inferred from a passing test suite.
