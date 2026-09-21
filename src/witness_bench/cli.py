"""Command-line interface for the Witness benchmark."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile
from typing import Sequence

from .report import render_markdown
from .runner import BenchmarkConfig, build_systems, run_benchmark


def _atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="witness-bench",
        description="Run the falsifiable offline Witness retrieval benchmark.",
    )
    parser.add_argument("--profile", choices=("smoke", "full"), default="full")
    parser.add_argument("--seed", type=int, default=20260829)
    parser.add_argument("--repetitions", type=int, default=2)
    parser.add_argument(
        "--task",
        action="append",
        default=[],
        help="Task ID to include; repeat the flag to select several.",
    )
    parser.add_argument(
        "--system",
        action="append",
        default=[],
        help="Exact system name to include; repeat the flag to select several.",
    )
    parser.add_argument("--bootstrap-resamples", type=int, default=2_000)
    parser.add_argument("--output", type=Path, default=Path("results/benchmark.json"))
    parser.add_argument("--report", type=Path, default=Path("docs/BENCHMARK_REPORT.md"))
    parser.add_argument(
        "--multihop-diagnostic",
        type=Path,
        default=Path("results/multihop_development_oracle.json"),
        help=(
            "Optional headline-ineligible diagnostic JSON to include in the report "
            "when the file exists."
        ),
    )
    parser.add_argument("--list-systems", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.list_systems:
        for system in build_systems(args.profile):
            print(system.name)
        return 0
    result = run_benchmark(
        BenchmarkConfig(
            seed=args.seed,
            repetitions=args.repetitions,
            profile=args.profile,
            task_ids=tuple(args.task),
            system_names=tuple(args.system),
            bootstrap_resamples=args.bootstrap_resamples,
        )
    )
    encoded = json.dumps(
        result,
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
        allow_nan=False,
    ) + "\n"
    diagnostic = None
    include_diagnostic = (
        args.profile == "full" and not args.task and not args.system
    )
    if include_diagnostic and args.multihop_diagnostic.is_file():
        try:
            decoded = json.loads(args.multihop_diagnostic.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise SystemExit(
                f"cannot read multihop diagnostic {args.multihop_diagnostic}: {exc}"
            ) from exc
        if not isinstance(decoded, dict):
            raise SystemExit("multihop diagnostic JSON must be an object")
        diagnostic_seed = decoded.get("manifest", {}).get("seed")
        if diagnostic_seed != args.seed:
            raise SystemExit(
                "multihop diagnostic seed does not match the main benchmark seed"
            )
        diagnostic = decoded
    report = render_markdown(result, multihop_diagnostic=diagnostic)
    _atomic_text(args.output.resolve(), encoded)
    _atomic_text(args.report.resolve(), report)
    print(
        f"wrote {args.output.resolve()} and {args.report.resolve()} "
        f"({result['manifest']['cases']} cases, {len(result['manifest']['systems'])} systems)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
