"""Small, dependency-free statistical helpers for benchmark reporting."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import random
from statistics import fmean
from typing import Iterable, Sequence


@dataclass(frozen=True)
class DistributionSummary:
    count: int
    mean: float
    minimum: float
    maximum: float
    p50: float
    p95: float
    p99: float

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


@dataclass(frozen=True)
class PairedInterval:
    """Bootstrap interval for `candidate - reference`."""

    pairs: int
    mean_delta: float
    low: float
    high: float
    confidence: float
    resamples: int

    @property
    def excludes_zero(self) -> bool:
        return self.low > 0.0 or self.high < 0.0

    def to_dict(self) -> dict[str, int | float | bool]:
        result = asdict(self)
        result["excludes_zero"] = self.excludes_zero
        return result


def percentile(values: Sequence[float], quantile: float) -> float:
    """Return a linearly interpolated percentile for `0 <= quantile <= 1`."""

    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must be between zero and one")
    if not values:
        return math.nan
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = quantile * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def summarize(values: Iterable[float]) -> DistributionSummary:
    samples = [float(value) for value in values]
    if not samples:
        nan = math.nan
        return DistributionSummary(0, nan, nan, nan, nan, nan, nan)
    return DistributionSummary(
        count=len(samples),
        mean=fmean(samples),
        minimum=min(samples),
        maximum=max(samples),
        p50=percentile(samples, 0.50),
        p95=percentile(samples, 0.95),
        p99=percentile(samples, 0.99),
    )


def paired_bootstrap_interval(
    candidate: Sequence[float],
    reference: Sequence[float],
    *,
    confidence: float = 0.95,
    resamples: int = 2_000,
    seed: int = 0,
) -> PairedInterval:
    """Estimate a percentile interval over paired mean differences.

    Pairing by benchmark case is important: easy/hard task mix must not be
    allowed to masquerade as a system difference.
    """

    if len(candidate) != len(reference):
        raise ValueError("paired samples must have equal length")
    if not candidate:
        raise ValueError("at least one paired sample is required")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between zero and one")
    if resamples < 1:
        raise ValueError("resamples must be positive")

    deltas = [float(left) - float(right) for left, right in zip(candidate, reference)]
    rng = random.Random(seed)
    boot_means: list[float] = []
    for _ in range(resamples):
        boot_means.append(fmean(deltas[rng.randrange(len(deltas))] for _ in deltas))
    alpha = (1.0 - confidence) / 2.0
    return PairedInterval(
        pairs=len(deltas),
        mean_delta=fmean(deltas),
        low=percentile(boot_means, alpha),
        high=percentile(boot_means, 1.0 - alpha),
        confidence=confidence,
        resamples=resamples,
    )

