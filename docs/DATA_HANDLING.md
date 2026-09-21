# Corpus, cache and output privacy

The committed benchmark is generated from the repository's synthetic fixture in
`src/witness_bench/data.py` and scorer definitions in `tasks.py`. Names, policies,
contracts and incidents in that fixture are fictional. They are not extracted
from private documents. The offline CLI generates this fixture locally and does
not take a real-data input path or call external services.

The Python engine can be embedded in an application that supplies real data.
Its data handling is different from the synthetic benchmark:

| Location | What it can contain | Lifetime |
|---|---|---|
| `CorpusStore()` | Complete source text, identifiers, ACL labels and metadata in memory | Process/object lifetime |
| `CorpusStore(storage_path=...)` | Complete document versions as plaintext JSON | Until explicitly removed; no TTL |
| `MaterializedWitnessCache(path=...)` | Plaintext evaluations including exact source quotes and bindings | Until files are removed; no TTL |
| `ExecutionResult`, rendered claims, caller logs | Source quotations, derived facts, identities and coverage | Controlled by the integrating application |
| `WitnessOfflineSystem` result cache | Query-specific passages/results in memory | Adapter lifetime |

Hashes in cache filenames are identifiers, not encryption or anonymization.
ACL checks at scheduling and verification protect the engine's result path;
someone with filesystem access to a store/cache can read its JSON directly.
The caller must authenticate users and map identities to principals. Document
permissions are an application contract, not an identity provider.

## Recommended local layout

Prefer in-memory defaults for experiments. If persistence is necessary, choose
an access-controlled directory outside the checkout, restrict it using your OS
permissions, and apply your organization's encryption and retention policy.
Do not place stores/caches under a web server's public directory, a shared
artifact bucket or an automatically synced public folder.

The repository also ignores `.witness/`, `private-data/`, `artifacts/`, and
`results/local/` as a defense against accidental local additions. Use those
names only when a repository-local directory is appropriate for your data.
Ignore rules do not erase tracked files, stop `git add -f`, scrub history, or
govern CI uploads and backups. Review `git diff --cached` before committing.
Keep secrets in your application's external secret store; the bundled engine
does not need model credentials.

## Retention and deletion

`remove_document(id, version)` removes that exact corpus version and its store
file. It makes corresponding cached rows unreachable through ordinary current
execution, but **does not physically erase old cache files, other versions,
existing result objects, logs or backups**. For an erasure request, stop users of
the affected store, remove all relevant versions, purge the application's
associated cache directory and saved results, then recreate engine/cache
instances. Account for backups and external adapters separately. There is no
built-in multi-tenant erasure workflow or scheduled cleanup.

Custom model adapters run in-process and can disclose source text to providers.
Review provider destinations, consent, retention, regional requirements,
logging and least-privilege access before using them. This repository does not
configure or certify those integrations.

## Reviewed public artifacts

During local publication preparation on 2026-09-21, both tracked result JSON
files were parsed and their string fields inspected. Their manifests match the
generated synthetic snapshot hashes. They contain run identifiers, evidence-ID
samples, scores, counters, timing and runtime descriptions; no raw source-text
or quote fields occur in the artifacts. Pattern checks found no email addresses,
user home paths, credential prefixes or private-key blocks. All 1,900 main-run
records were independently reproduced with matching deterministic content.
See [reproduction details](REPRODUCIBILITY.md).

These checks support the provenance of these particular artifacts. They are not
a general PII detector, ownership review, or clearance of future outputs. Review
new artifacts independently, especially when adapting the harness to real data.
