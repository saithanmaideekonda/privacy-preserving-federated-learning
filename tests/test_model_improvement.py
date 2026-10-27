"""Tests for validation-only rare-class weighting experiments."""
import inspect
import json
import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np
from sklearn.neural_network import MLPClassifier
from threadpoolctl import threadpool_limits

from src.full_dataset.centralized import (
    class_weights_from_counts, load_model_artifact, metrics_from_confusion,
)
from src.full_dataset.model_improvement import (
    CheckpointSelector, capped_inverse_weights, class_weights_for_strategy,
    effective_number_weights, load_train_validation_arrays, run_weighting_experiments,
    normalize_weighted_mean,
)


class ModelImprovementTests(unittest.TestCase):
    def test_reproduction_weights_match_phase2_formula(self):
        counts = np.asarray([1000, 120, 15], dtype=np.int64)
        actual = class_weights_for_strategy(counts, "reproduction_control")
        np.testing.assert_allclose(actual, class_weights_from_counts(counts), rtol=0, atol=1e-14)
        self.assertAlmostEqual(float(np.average(actual, weights=counts)), 1.0)

    def test_weight_normalization_and_invalid_inputs(self):
        counts = np.asarray([10, 3, 1], dtype=np.int64)
        values = normalize_weighted_mean(np.asarray([1., 2., 8.]), counts)
        self.assertAlmostEqual(float(np.average(values, weights=counts)), 1.)
        with self.assertRaises(ValueError):
            normalize_weighted_mean(np.asarray([1., 0.]), np.asarray([2, 1]))

    def test_stronger_inverse_frequency_increases_rare_weight(self):
        counts = np.asarray([10_000, 100, 5], dtype=np.int64)
        sqrt_weights = class_weights_for_strategy(counts, "reproduction_control")
        stronger = class_weights_for_strategy(counts, "inverse_power_075")
        self.assertGreater(stronger[-1] / stronger[0], sqrt_weights[-1] / sqrt_weights[0])
        self.assertAlmostEqual(float(np.average(stronger, weights=counts)), 1.)

    def test_capped_inverse_has_weighted_mean_one_and_cap(self):
        weights = capped_inverse_weights(np.asarray([100_000, 100, 2]), power=1., cap=10.)
        self.assertTrue(np.isfinite(weights).all())
        self.assertLessEqual(float(weights.max()), 10. + 1e-12)
        self.assertAlmostEqual(float(np.average(weights, weights=[100_000, 100, 2])), 1., places=12)
        self.assertEqual(float(weights.min()), 1.0 if np.allclose(weights, 1) else float(weights.min()))

    def test_effective_number_weights_are_finite_and_normalized(self):
        counts = np.asarray([100_000, 100, 2], dtype=np.int64)
        weights = effective_number_weights(counts, beta=.9999)
        self.assertTrue(np.isfinite(weights).all())
        self.assertGreater(weights[-1], weights[0])
        self.assertAlmostEqual(float(np.average(weights, weights=counts)), 1.)
        with self.assertRaises(ValueError):
            effective_number_weights(counts, beta=1.)

    def test_checkpoint_selection_uses_validation_score_and_keeps_earlier_tie(self):
        selector = CheckpointSelector()
        self.assertTrue(selector.consider(1, .42))
        self.assertFalse(selector.consider(2, .42))
        self.assertFalse(selector.consider(3, float("nan")))
        self.assertTrue(selector.consider(4, .43))
        self.assertEqual(selector.best_epoch, 4)
        self.assertAlmostEqual(selector.best_score, .43)

    def test_metrics_match_confusion_matrix_counts(self):
        cm = np.asarray([[8, 2, 0], [1, 2, 0], [0, 1, 1]], dtype=np.int64)
        result = metrics_from_confusion(cm, ["common", "rare_a", "rare_b"])
        self.assertEqual(sum(row["support"] for row in result["per_class"]), int(cm.sum()))
        self.assertAlmostEqual(result["accuracy"], float(np.trace(cm) / cm.sum()))
        self.assertAlmostEqual(result["macro_f1"], np.mean([row["f1"] for row in result["per_class"]]))

    def test_reproducible_weighted_training_initialization_and_order(self):
        x = np.asarray([[-2., -1.], [-1., -2.], [-2.2, -1.1], [2., 1.], [1., 2.],
                        [2.2, 1.1], [0., 3.], [.2, 2.8], [-.2, 3.1]], dtype=np.float32)
        y = np.asarray([0, 0, 0, 1, 1, 1, 2, 2, 2], dtype=np.int32)
        weights = class_weights_for_strategy(np.bincount(y), "inverse_power_075")
        fitted = []
        with threadpool_limits(limits=1):
            for _ in range(2):
                model = MLPClassifier(hidden_layer_sizes=(5, 3), batch_size=3, max_iter=1,
                                      shuffle=False, random_state=91, tol=0.0)
                order = np.random.default_rng(2026).permutation(len(y))
                for batch_no, start in enumerate(range(0, len(order), 3)):
                    idx = order[start:start + 3]
                    model.partial_fit(x[idx], y[idx], classes=np.arange(3) if batch_no == 0 else None,
                                      sample_weight=weights[y[idx]])
                fitted.append(model)
        for a, b in zip(fitted[0].coefs_, fitted[1].coefs_):
            np.testing.assert_array_equal(a, b)
        for a, b in zip(fitted[0].intercepts_, fitted[1].intercepts_):
            np.testing.assert_array_equal(a, b)

    def test_training_validation_loader_does_not_require_or_reference_test_arrays(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd() / "work") as temp:
            root = Path(temp)
            (root / "arrays").mkdir()
            mapping = {"class_a": 0, "class_b": 1}
            prep = {"class_to_id": mapping, "feature_columns_retained": ["f0", "f1"]}
            phase1 = {"derived_rows": {"train": 4, "validation": 2, "test": 99}}
            phase1["derived_class_counts"] = {"validation": {"class_a": 1, "class_b": 1}}
            (root / "preprocessing_metadata.json").write_text(json.dumps(prep), encoding="utf-8")
            (root / "phase1_metadata.json").write_text(json.dumps(phase1), encoding="utf-8")
            np.save(root / "arrays" / "X_train.npy", np.ones((4, 2), dtype=np.float32))
            np.save(root / "arrays" / "y_train.npy", np.asarray([0, 0, 1, 1], dtype=np.int32))
            np.save(root / "arrays" / "X_validation.npy", np.ones((2, 2), dtype=np.float32))
            np.save(root / "arrays" / "y_validation.npy", np.asarray([0, 1], dtype=np.int32))
            loaded = load_train_validation_arrays(root)
            self.assertEqual(loaded[0].shape, (4, 2))
            self.assertEqual(loaded[2].shape, (2, 2))
            self.assertFalse((root / "arrays" / "X_test.npy").exists())
            self.assertFalse((root / "arrays" / "y_test.npy").exists())
            for array in loaded[:4]:
                array._mmap.close()
            del loaded
        source = inspect.getsource(load_train_validation_arrays)
        self.assertNotIn("X_test.npy", source)
        self.assertNotIn("y_test.npy", source)
        runner_source = inspect.getsource(run_weighting_experiments)
        self.assertNotIn("X_test.npy", runner_source)
        self.assertNotIn("y_test.npy", runner_source)

    def test_candidate_checkpoint_loads_with_label_mapping(self):
        x = np.asarray([[-2., -1.], [-1., -2.], [2., 1.], [1., 2.]], dtype=np.float32)
        y = np.asarray([0, 0, 1, 1], dtype=np.int32)
        model = MLPClassifier(hidden_layer_sizes=(3,), solver="lbfgs", max_iter=20, random_state=4).fit(x, y)
        with tempfile.TemporaryDirectory(dir=Path.cwd() / "work") as temp:
            path = Path(temp) / "candidate.joblib"
            joblib.dump(model, path)
            loaded = load_model_artifact(path, {"a": 0, "b": 1})
            np.testing.assert_array_equal(loaded.predict(x), model.predict(x))


if __name__ == "__main__":
    unittest.main()





