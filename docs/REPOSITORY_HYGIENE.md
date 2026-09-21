# Repository privacy and secret checks

The repository includes a standard-library Python history guard, regression tests,
read-only security CI and weekly dependency-update configuration. These checks do
not rewrite history or publish anything. Required legal attribution is preserved.

## Local commands

```bash
python3 -m unittest discover -s tests -p 'test_repository_hygiene.py' -v
python3 scripts/check_repository_hygiene.py
python3 scripts/check_repository_hygiene.py --repo /path/to/repository --json
```

Exit codes: `0` means no findings under the stated rules; `1` means findings;
`2` means inspection failed or history is incomplete. Shallow history is refused.
Uncommitted changes and ignored files are not scanned by this history command.
Review/stage intended files, commit using a public identity, and scan the exact
resulting history before any authorized push. A nonzero result never mutates
the input repository.

## Narrow rules and exceptions

- Scan all commits, tags, blobs and trees reachable from locally available refs,
  with Git replacement-object substitution disabled. Each tree entry is checked
  independently, so a private filename cannot hide behind identical bytes in a
  safe template. No historical deletion, test directory or legal file is skipped.
- Author/committer/tagger emails must be GitHub user noreply addresses, with a
  display name matching the public alias after a leading numeric-ID-plus prefix
  is removed. GitHub service commits may use display name `GitHub` and
  `noreply@github.com`. GitHub bot aliases are accepted. A real full name paired
  with a noreply address is still flagged by this publication policy.
- Private ChatGPT/legacy ChatGPT/Claude conversation or coding-session URLs and
  specifically named assistant-session trailers are flagged in messages and
  blob text. Public product/docs/share links are not treated as private sessions.
- The literal text of absolute user-home paths on macOS, Linux and Windows is
  flagged, including JSON-escaped Windows separators and ASCII embedded in a
  binary blob. Use portable relative paths or environment-variable notation in
  documentation. The guard itself and its tests construct synthetic paths from
  segments, so no blanket fixture exclusion is needed.
- Block real `.env`/`.env.*` and `.dev.vars`/`.dev.vars.*` names, `.private` at
  any tree depth, `*.private.json`/`*.private.jsonc` overrides, and the exact
  local override names `wrangler.local.json` and `wrangler.local.jsonc`.
  Only `.env.example`, `.env.local.example`, `.dev.vars.example` and
  `.dev.vars.local.example` are exempt templates. A template filename does not
  exempt its contents from path/session checks or the separate secret scan.
- Ordinary source/license/research attribution is preserved, including email
  text in notices and public `Co-Authored-By: Claude <noreply@anthropic.com>`
  credit. Co-author text is not automatically rewritten. Private values in such
  text still need contextual human review; this is not broad name/email removal.

Findings print only a category, Git object ID and line/identity field or a
hashed tree-entry locator. Names, email addresses, matches and filenames are
never included. Inspect the object privately when resolving a finding. A hash
is a locator, not a promise of anonymization against guessing.

There is no suppression file or broad allowlist. If a legitimate fixture needs
an apparent prohibited literal, prefer constructing it from parts during the
test. Any policy change should be explicit, narrowly justified and reviewed
with a regression case. Preserve required legal attribution instead of treating
it as a privacy mistake.

## CI and secret scanner

The workflow has read-only contents permission, checks out full fetched history
with credential persistence disabled, runs the regression suite and guard, then
runs the free Gitleaks CLI. It contains no dependency installs, deploy,
publication or auto-merge steps. The scanner still runs if the hygiene check
fails, unless the workflow is cancelled.

Checkout is pinned to verified official v4 commit
`11d5960a326750d5838078e36cf38b85af677262`. Gitleaks is pinned to v8.30.1's Linux
x64 archive and checksum
`551f6fc83ea457d62a0d98237cbad105af8d557003051f41f3e7ca7b3f2470eb`, independently
verified through the official GitHub release API on 2026-09-21. The downloaded
archive is checksum-checked before extracting the single executable.

Gitleaks receives `--redact=100`, `--log-opts=--all`, no size limit, a temporary
config that extends only built-in rules, an empty external ignore directory and
`--ignore-gitleaks-allow`. Repository configs, ignore files and inline allow
comments therefore cannot silently suppress findings. Raw diagnostics and the
redacted report remain temporary and are not uploaded. The console prints only
rule IDs, commit hashes and line numbers, withholding filenames and values.

Protect workflow/checker changes with required review and make these checks
required through repository settings when authorized. A workflow in a pull
request can be changed by that pull request; it is not its own tamper-proof
policy boundary. Gitleaks updates require reviewing the version and new official
checksum together; Dependabot's Actions entry does not update this download.

## What a pass does not establish

This does not detect arbitrary PII, decide whether fictional people represent
real people, establish license ownership, validate deployments, scan GitHub
issues/PR bodies/Actions artifacts, erase caches/forks, or inspect unavailable
refs. The blob checks do not unpack archives or fully decode arbitrary encodings;
Gitleaks has its own distinct detection limits. External LFS content and
submodule repositories require separate review. A full-history clone covers
only refs actually fetched. Keep a contextual privacy/security audit as part of
the release process.

Sources: [Gitleaks release](https://github.com/gitleaks/gitleaks/releases/tag/v8.30.1),
[Dependabot options](https://docs.github.com/en/code-security/reference/supply-chain-security/dependabot-options-reference),
[supported ecosystems](https://docs.github.com/en/code-security/reference/supply-chain-security/supported-ecosystems-and-repositories).
