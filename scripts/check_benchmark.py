"""Compare deterministic benchmark outputs without comparing machine timing.

This checks independent executions, not only the runner's warm-query replay.
It deliberately does not print differing values: results from user corpora may
contain sensitive identifiers. See docs/REPRODUCIBILITY.md for its scope.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


MANIFEST_FIELDS = (
    "benchmark", "benchmark_version", "implementation_fingerprint",
    "implementation_python_files", "seed", "profile", "repetitions",
    "documents", "evidence_spans", "tasks", "cases", "systems", "snapshots",
    "track", "model_backed", "production_baselines", "claim_status",
)


def without_timing(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: without_timing(item)
            for key, item in value.items()
            if key not in {"latency_ms", "cpu_time_ms"}
        }
    if isinstance(value, list):
        return [without_timing(item) for item in value]
    return value


def projection(document: dict[str, Any]) -> dict[str, Any]:
    """Validate run identity/completeness and select deterministic fields."""
    manifest = document["manifest"]
    selected_manifest = {key: manifest[key] for key in MANIFEST_FIELDS}
    if manifest["claim_status"] != "preliminary_offline_proxy_only":
        raise ValueError("expected an explicitly preliminary offline result")
    if manifest["model_backed"] is not False or manifest["production_baselines"] is not False:
        raise ValueError("this checker is for deterministic offline proxies only")
    systems = manifest["systems"]
    if not systems or len(systems) != len(set(systems)):
        raise ValueError("system identities must be nonempty and unique")
    repetitions = manifest["repetitions"]
    cases = manifest["cases"]
    if not isinstance(repetitions, int) or repetitions < 1 or not isinstance(cases, int) or cases < 1:
        raise ValueError("case and repetition counts must be positive integers")
    runs = document["runs"]
    if len(runs) != len(systems) * cases * repetitions:
        raise ValueError("run count does not match the manifest")
    indexed: dict[tuple[str, str, int], Any] = {}
    case_ids: set[str] = set()
    task_ids: set[str] = set()
    for run in runs:
        key = (run["system"], run["case_id"], run["repetition"])
        if key in indexed:
            raise ValueError("duplicate run identity")
        if key[0] not in systems or key[2] not in range(repetitions):
            raise ValueError("run identity is outside the manifest")
        case_ids.add(key[1])
        task_ids.add(run["task_id"])
        indexed[key] = without_timing(run)
    if len(case_ids) != cases or len(task_ids) != manifest["tasks"]:
        raise ValueError("case/task identities do not match the manifest")
    expected = {(system, case, repeat) for system in systems for case in case_ids for repeat in range(repetitions)}
    if set(indexed) != expected:
        raise ValueError("a system/case/repetition is missing")
    return {
        "manifest": selected_manifest,
        "system_metadata": {name: document["summary"][name]["metadata"] for name in systems},
        "runs": [indexed[key] for key in sorted(indexed)],
    }


def compare(left: dict[str, Any], right: dict[str, Any]) -> None:
    first, second = projection(left), projection(right)
    for section in ("manifest", "system_metadata", "runs"):
        if first[section] != second[section]:
            raise ValueError(f"deterministic {section} differ; inspect locally (values withheld)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("reproduced", type=Path)
    args = parser.parse_args()
    try:
        first = json.loads(args.reference.read_text(encoding="utf-8"))
        second = json.loads(args.reproduced.read_text(encoding="utf-8"))
        compare(first, second)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        # Do not echo JSON parse context, input paths, keys, or document values.
        message = str(exc) if type(exc) is ValueError else type(exc).__name__
        parser.exit(1, f"benchmark verification failed: {message}\n")
    print(f"Verified {len(first['runs'])} run records: evidence hashes, metrics and counters agree; timing excluded.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
