# Witness repository policy

Witness is a code-only research project for local use. Publish source and
synthetic benchmark artifacts when requested; keep execution in local tools or
test-only CI. Do not add hosted services, deployment workflows, infrastructure
provisioning, package-registry publishing, or automatic releases.

- Preserve the Apache-2.0 LICENSE, NOTICE, and research/source attribution.
  Do not invent ownership claims or relicense documents supplied by users.
- Keep the alpha/offline-proxy limits explicit. Exact source coordinates and
  reproduced scores do not establish semantic truth or production superiority.
- Use synthetic fixtures. Never commit real corpora, plaintext evidence caches,
  personal operator notes, credentials, or private assistant-session links.
  Keep local output under the ignored paths described in `docs/DATA_HANDLING.md`.
- Use the intended public GitHub alias and its noreply address for commit
  identities. Run `scripts/check_repository_hygiene.py` on the final history.
- For Python behavior changes, run the unit suite and worked example documented
  in README. Preserve historical benchmark JSON; reproduce it when generator,
  evaluator, scorer or execution changes affect results. Record changed
  hypotheses/configuration rather than tuning until a preferred system wins.
- CI may build/install the wheel locally, run tests and synthetic benchmarks,
  and scan repository hygiene/secrets. Dependency updates remain reviewed PRs.
