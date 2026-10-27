"""Tests for Phase 2 metrics, label mappings, and saved checkpoint loading."""

import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np
from sklearn.neural_network import MLPClassifier

from src.full_dataset.centralized import (
    class_weights_from_counts,
    load_model_artifact,
    metrics_from_confusion,
    validate_label_mapping,
)


class CentralizedBaselineTests(unittest.TestCase):
    def test_confusion_matrix_metrics_are_consistent(self):
        cm = np.asarray([[5, 1, 0], [2, 2, 0], [0, 0, 0]], dtype=np.int64)
        result = metrics_from_confusion(cm, ["A", "B", "C"])
        self.assertAlmostEqual(result["accuracy"], 0.7)
        self.assertAlmostEqual(result["macro_precision"], 29 / 42)
        self.assertAlmostEqual(result["macro_recall"], 2 / 3)
        self.assertAlmostEqual(result["macro_f1"], 61 / 91)
        self.assertAlmostEqual(result["weighted_f1"], 314 / 455)
        self.assertEqual(result["supported_classes"], 2)
        self.assertEqual(result["per_class"][2]["support"], 0)
        self.assertIsNone(result["per_class"][2]["f1"])

    def test_label_mapping_requires_contiguous_training_ids(self):
        self.assertEqual(validate_label_mapping({"B": 1, "A": 0}), ["A", "B"])
        with self.assertRaises(ValueError):
            validate_label_mapping({"A": 0, "B": 2})

    def test_inverse_sqrt_class_weights_are_finite_and_mean_one(self):
        weights = class_weights_from_counts(np.asarray([100, 25, 4]))
        self.assertTrue(np.isfinite(weights).all())
        self.assertAlmostEqual(float(np.average(weights, weights=[100, 25, 4])), 1.0)
        self.assertGreater(weights[2], weights[1])
        self.assertGreater(weights[1], weights[0])

    def test_saved_checkpoint_loads_with_matching_label_mapping(self):
        x = np.asarray([
            [-2.0, -1.0], [-1.0, -2.0], [-1.5, -1.2],
            [2.0, 1.0], [1.0, 2.0], [1.5, 1.2],
        ], dtype=np.float32)
        y = np.asarray([0, 0, 0, 1, 1, 1], dtype=np.int32)
        model = MLPClassifier(hidden_layer_sizes=(4,), solver="lbfgs", max_iter=30, random_state=17)
        model.fit(x, y)
        with tempfile.TemporaryDirectory(dir=Path.cwd() / "work") as temp:
            checkpoint = Path(temp) / "best_model.joblib"
            joblib.dump(model, checkpoint)
            loaded = load_model_artifact(checkpoint, {"class_a": 0, "class_b": 1})
            np.testing.assert_array_equal(loaded.predict(x), model.predict(x))
            with self.assertRaises(ValueError):
                load_model_artifact(checkpoint, {"class_a": 0, "class_b": 2})


if __name__ == "__main__":
    unittest.main()


