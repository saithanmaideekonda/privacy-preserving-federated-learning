"""Tests for PSI scoring, edge cases, and deterministic drift scenarios."""
import unittest
import numpy as np
from src.full_dataset.drift import (
    apply_synthetic_shift, fit_feature_reference, psi_from_counts,
    score_feature, score_window,
)
from src.full_dataset.federated import partition_clients


class DriftDetectionTests(unittest.TestCase):
    def test_unchanged_distribution_has_zero_psi(self):
        rng = np.random.default_rng(77)
        reference_values = rng.normal(0, 1, 20_000).astype(np.float32)
        ref = fit_feature_reference(reference_values, n_bins=10)
        score, finite_rows, current_counts = score_feature(ref, reference_values)
        self.assertAlmostEqual(score, 0.0, places=10)
        self.assertEqual(finite_rows, len(reference_values))
        self.assertEqual(int(current_counts.sum()), len(reference_values))

    def test_large_distribution_shift_exceeds_operational_threshold(self):
        rng = np.random.default_rng(101)
        reference_values = rng.normal(0, 1, 20_000).astype(np.float32)
        shifted_values = (rng.normal(0, 1, 5_000) + 5.0).astype(np.float32)
        ref = fit_feature_reference(reference_values, n_bins=10)
        score, _, _ = score_feature(ref, shifted_values)
        self.assertGreaterEqual(score, 0.25)

    def test_empty_bins_and_constant_feature_are_numerically_safe(self):
        score = psi_from_counts([100, 0, 0], [0, 100, 0], epsilon=1e-6)
        self.assertTrue(np.isfinite(score))
        self.assertGreater(score, 0.25)
        constant = fit_feature_reference(np.full(100, 3.5, dtype=np.float32), n_bins=10)
        self.assertEqual(constant["effective_bins"], 1)
        unchanged, _, _ = score_feature(constant, np.full(20, 3.5))
        self.assertEqual(unchanged, 0.0)

    def test_nonfinite_values_are_ignored_but_empty_distributions_rejected(self):
        ref = fit_feature_reference(np.asarray([0, 0, 1, 1, np.nan, np.inf]), n_bins=4)
        score, count, bins = score_feature(ref, np.asarray([0, 1, np.nan, -np.inf]))
        self.assertTrue(np.isfinite(score))
        self.assertEqual(count, 2)
        self.assertEqual(int(bins.sum()), 2)
        with self.assertRaises(ValueError):
            fit_feature_reference(np.asarray([np.nan, np.inf]))
        with self.assertRaises(ValueError):
            score_feature(ref, np.asarray([np.nan, np.inf]))
        with self.assertRaises(ValueError):
            psi_from_counts([0, 0], [1, 0])

    def test_window_alert_uses_max_feature_psi_and_is_reproducible(self):
        rng = np.random.default_rng(1234)
        reference = rng.normal(size=(3000, 2)).astype(np.float32)
        names = ["stable", "shifted"]
        refs = {name: fit_feature_reference(reference[:, i]) for i, name in enumerate(names)}
        window = rng.normal(size=(1000, 2)).astype(np.float32)
        window[:, 1] += 5
        first = score_window(refs, window, names, 0.25)
        second = score_window(refs, window.copy(), names, 0.25)
        self.assertEqual(first, second)
        self.assertTrue(first["alert"])
        self.assertEqual(first["top_feature"], "shifted")

    def test_client_partition_and_synthetic_injection_are_reproducible(self):
        labels = np.repeat(np.arange(3, dtype=np.int32), [100, 150, 250])
        a = partition_clients(labels, 3, alpha=1.0, seed=20261026)
        b = partition_clients(labels, 3, alpha=1.0, seed=20261026)
        for left, right in zip(a, b):
            np.testing.assert_array_equal(left, right)
        window = np.zeros((10, 4), dtype=np.float32)
        one, changed_one = apply_synthetic_shift(window, 2, 4, 1, affected_client=2, first_shift_window=4, shift=5.0)
        two, changed_two = apply_synthetic_shift(window, 2, 4, 1, affected_client=2, first_shift_window=4, shift=5.0)
        np.testing.assert_array_equal(one, two)
        self.assertTrue(changed_one and changed_two)
        np.testing.assert_array_equal(one[:, 1], np.full(10, 5.0))
        control, changed = apply_synthetic_shift(window, 1, 8, 1, affected_client=2, first_shift_window=4, shift=5.0)
        self.assertFalse(changed)
        np.testing.assert_array_equal(control, window)
        before_change, changed = apply_synthetic_shift(window, 2, 3, 1, affected_client=2, first_shift_window=4, shift=5.0)
        self.assertFalse(changed)
        np.testing.assert_array_equal(before_change, window)


if __name__ == "__main__":
    unittest.main()
