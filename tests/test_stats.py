from __future__ import annotations

import math
import unittest

from witness_bench.stats import paired_bootstrap_interval, percentile, summarize


class StatsTests(unittest.TestCase):
    def test_percentile_interpolates(self) -> None:
        self.assertEqual(percentile([0, 10], 0.5), 5.0)
        self.assertEqual(percentile([4], 0.99), 4.0)
        self.assertTrue(math.isnan(percentile([], 0.5)))

    def test_summary_has_latency_percentiles(self) -> None:
        summary = summarize(range(1, 101))
        self.assertEqual(summary.count, 100)
        self.assertEqual(summary.p50, 50.5)
        self.assertGreater(summary.p99, summary.p95)

    def test_paired_bootstrap_is_deterministic(self) -> None:
        first = paired_bootstrap_interval([1, 1, 1], [0, 0, 0], seed=17)
        second = paired_bootstrap_interval([1, 1, 1], [0, 0, 0], seed=17)
        self.assertEqual(first, second)
        self.assertEqual(first.low, 1.0)
        self.assertTrue(first.excludes_zero)

    def test_pair_count_must_match(self) -> None:
        with self.assertRaises(ValueError):
            paired_bootstrap_interval([1], [1, 2])


if __name__ == "__main__":
    unittest.main()

