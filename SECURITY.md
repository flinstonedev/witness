# Security policy

Witness is an alpha, local Python research prototype. There is no production
support window, hosted service, authentication gateway, or security SLA. Fixes
are developed on the main branch; published versions should not be assumed to
receive security backports.

## Reporting

If GitHub private vulnerability reporting is enabled, use the repository's
**Security → Report a vulnerability** form. Do not include private source
documents, cache contents, credentials or personal data in a public issue.
If that form is unavailable, open a public issue containing only a request for
a private reporting channel and a broad component name, without exploit details
or sensitive material. A dedicated monitored security mailbox is not currently
published. Reproduce with a small synthetic fixture whenever possible.

Include the version/commit, Python version, affected component, expected and
actual behavior, and a minimal redacted reproduction. Do not test against other
people's accounts or data.

## Trust boundary

- The application integrating Witness must authenticate users and supply
  trustworthy principal identities. Passing a string in `principals` is not
  authentication. Empty document permissions mean the document is public to
  callers of that store.
- Compiler/evaluator adapters are arbitrary in-process Python. They are trusted
  application code, **not sandboxed plugins**. Do not execute untrusted adapters
  or accept unrestricted regex/program input in a public service. Regex CPU
  exhaustion and large corpora/binding fan-out need external resource limits.
- The bundled lexical engine does not call model services. A custom adapter
  can send data externally; review its permissions and data handling separately.
- Exact-span verification establishes source coordinates and mechanical
  validity, not semantic truth, complete plan adequacy, or protection from all
  prompt injection. The synthetic injection fixture is not a production red team.
- Disk stores and caches are plaintext and are not a tenant isolation boundary.
  See [data handling](docs/DATA_HANDLING.md) before loading real documents.

CI exercises synthetic data and engine invariants. It does not certify a
production deployment or the safety of arbitrary third-party adapters.
