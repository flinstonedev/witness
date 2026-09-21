#!/usr/bin/env python3
"""Read-only checks for narrowly defined publication mistakes in Git history.

This is not an arbitrary-PII detector or secret scanner. Findings contain only
Git object IDs, categories and line/field/hashed-entry locations, never values.
Run a separate secret scanner and review ownership/data provenance as well.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import subprocess
from typing import Iterable


ENV_EXAMPLES = frozenset({
    b".env.example", b".env.local.example",
    b".dev.vars.example", b".dev.vars.local.example",
})
PRIVATE_OVERRIDES = frozenset({
    b"wrangler.local.json", b"wrangler.local.jsonc",
    b"wrangler.private.json", b"wrangler.private.jsonc",
})

# Grouped regex source avoids embedding an actual user-home path in this file.
# Both ordinary and JSON-escaped Windows separators are recognized.
HOME_PATH = re.compile(
    r"(?<![\w])(?:/(?:Users|home)/[^/\s\"'<>]+"
    r"|[A-Za-z]:[\\/]+(?:Users|Documents and Settings)[\\/]+[^\\/\s\"'<>]+)",
    re.IGNORECASE,
)
SESSION_URL = re.compile(
    r"https?://(?:www\.)?(?:"
    r"(?:chatgpt\.com|chat\.openai\.com)/(?:c/|g/[^/\s]+/c/|codex/(?:tasks|sessions)/)"
    r"|claude\.ai/(?:chat/|code/(?:session[_/-]|sessions/))"
    r")[^\s<>\"')]+",
    re.IGNORECASE,
)
SESSION_TRAILER = re.compile(
    r"^\s*(?:Claude[-_ ](?:Code[-_ ])?Session(?:[-_ ]Id)?"
    r"|(?:OpenAI[-_ ])?Codex[-_ ](?:Session|Thread|Task)(?:[-_ ]Id)?"
    r"|ChatGPT[-_ ](?:Conversation|Session)(?:[-_ ]Id)?):\s*\S",
    re.IGNORECASE,
)
GITHUB_ALIAS = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,37}[A-Za-z0-9])?(?:\[bot\])?")
IDENTITY = re.compile(rb"^(author|committer|tagger) (.*?) <([^>\r\n]*)> [0-9]+ [+-][0-9]{4}$")


@dataclass(frozen=True, order=True)
class Finding:
    category: str
    object_id: str
    location: str


class InspectionError(Exception):
    """An intentionally non-sensitive inspection failure."""


def git(repo: Path, *args: str, data: bytes | None = None) -> bytes:
    env = dict(os.environ, GIT_NO_REPLACE_OBJECTS="1", GIT_OPTIONAL_LOCKS="0")
    result = subprocess.run(
        ["git", "-C", str(repo), *args], input=data,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env, check=False,
    )
    if result.returncode:
        # Git diagnostics may contain names, addresses or local paths.
        raise InspectionError("Git inspection failed; inspect locally without publishing its diagnostics")
    return result.stdout


def identity_findings(object_id: str, header: bytes) -> Iterable[Finding]:
    for line in header.splitlines():
        if not line.startswith((b"author ", b"committer ", b"tagger ")):
            continue
        match = IDENTITY.fullmatch(line)
        if not match:
            yield Finding("invalid_identity_metadata", object_id, "identity")
            continue
        field, raw_name, raw_email = match.groups()
        name = raw_name.decode("utf-8", errors="replace")
        email = raw_email.decode("utf-8", errors="replace").casefold()
        location = field.decode("ascii")
        alias = None
        if email == "noreply@github.com":
            alias = "GitHub"
        elif email.count("@") == 1 and email.endswith("@users.noreply.github.com"):
            candidate = re.sub(r"^\d+\+", "", email.split("@", 1)[0])
            if GITHUB_ALIAS.fullmatch(candidate):
                alias = candidate
        if alias is None:
            yield Finding("nonpublic_identity_email", object_id, location)
        elif name.casefold() != alias.casefold():
            yield Finding("nonpublic_identity_name", object_id, location)


def message_findings(object_id: str, message: bytes) -> Iterable[Finding]:
    for number, line in enumerate(message.decode("utf-8", errors="replace").splitlines(), 1):
        if SESSION_URL.search(line):
            yield Finding("private_assistant_session_url", object_id, f"message-line:{number}")
        if SESSION_TRAILER.search(line):
            yield Finding("private_assistant_session_trailer", object_id, f"message-line:{number}")
        if HOME_PATH.search(line):
            yield Finding("absolute_user_home_path", object_id, f"message-line:{number}")
    # Co-authored-by and research/source attribution are deliberately not erased.
    # A public Anthropic noreply credit is not an assistant session identifier.


def blob_findings(object_id: str, content: bytes) -> Iterable[Finding]:
    # Decode with replacement so ASCII paths embedded in binary blobs are not
    # silently skipped. This does not promise UTF-16/archive/encoded-data scans.
    for number, line in enumerate(content.decode("utf-8", errors="replace").splitlines(), 1):
        if HOME_PATH.search(line):
            yield Finding("absolute_user_home_path", object_id, f"line:{number}")
        if SESSION_URL.search(line):
            yield Finding("private_assistant_session_url", object_id, f"line:{number}")
        if SESSION_TRAILER.search(line):
            yield Finding("private_assistant_session_trailer", object_id, f"line:{number}")


def private_entry(name: bytes) -> bool:
    lowered = name.lower()
    if lowered in ENV_EXAMPLES:
        return False
    return (
        lowered == b".private"
        or lowered in PRIVATE_OVERRIDES
        or lowered.endswith((b".private.json", b".private.jsonc"))
        or lowered == b".env" or lowered.startswith(b".env.")
        or lowered == b".dev.vars" or lowered.startswith(b".dev.vars.")
    )


def inspect_repository(repo: Path) -> tuple[list[Finding], dict[str, int]]:
    if git(repo, "rev-parse", "--is-shallow-repository").strip() != b"false":
        raise InspectionError("Full history is required; fetch with depth zero before checking")
    # --no-object-names prevents paths from being interpreted as object IDs.
    object_ids = git(repo, "rev-list", "--objects", "--all", "--no-object-names").splitlines()
    counts = {"commit": 0, "tag": 0, "tree": 0, "blob": 0}
    findings: set[Finding] = set()
    if not object_ids:
        return [], counts
    inventory = git(repo, "cat-file", "--batch-check=%(objectname) %(objecttype)", data=b"\n".join(object_ids) + b"\n")
    for record in inventory.splitlines():
        parts = record.decode("ascii").split()
        if len(parts) != 2 or parts[1] not in counts:
            raise InspectionError("Reachable object could not be inspected")
        object_id, kind = parts
        counts[kind] += 1
        if kind == "tree":
            # Inspect every tree's entries, not rev-list's one representative
            # path per blob. An identical blob can have safe and private names.
            for entry in git(repo, "ls-tree", "-z", object_id).split(b"\0"):
                if not entry:
                    continue
                _, name = entry.split(b"\t", 1)
                if private_entry(name):
                    path_id = sha256(name).hexdigest()[:16]
                    findings.add(Finding("private_configuration_path", object_id, f"entry-sha256:{path_id}"))
        else:
            content = git(repo, "cat-file", kind, object_id)
            if kind == "blob":
                findings.update(blob_findings(object_id, content))
            else:
                header, _, message = content.partition(b"\n\n")
                findings.update(identity_findings(object_id, header))
                findings.update(message_findings(object_id, message))
    return sorted(findings), counts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--json", action="store_true", help="Emit only sanitized structured findings")
    args = parser.parse_args(argv)
    try:
        findings, counts = inspect_repository(args.repo)
    except (InspectionError, OSError, UnicodeError, ValueError):
        print("repository hygiene: inspection failed or history is incomplete; no input values disclosed")
        return 2
    if args.json:
        print(json.dumps({"objects_checked": counts, "findings": [asdict(finding) for finding in findings]}, indent=2))
    else:
        for finding in findings:
            print(f"{finding.category} object:{finding.object_id} {finding.location}")
        print(f"Repository hygiene: {len(findings)} finding(s); checked {counts['commit']} commits, {counts['tag']} tags, {counts['blob']} blobs, {counts['tree']} trees.")
        print("Scope: commit identity, private session metadata, user-home paths and private config names; not arbitrary PII clearance.")
    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
