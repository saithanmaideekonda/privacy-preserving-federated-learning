"""Unit tests for the Stage 2 centralized parameter study."""
import inspect
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from sklearn.neural_network import MLPClassifier
from threadpoolctl import threadpool_limits

from src.full_dataset.centralized import _fit_epoch, metrics_from_confusion
from src.full_dataset.model_improvement import load_train_validation_arrays
from src.full_dataset.training_parameter_study import (
    ValidationEarlyStopper, build_candidate_configs, validate_candidate_config,
    run_candidate,
)


class TrainingParameterStudyTests(unittest.TestCase):
    def test_configuration_grid_is_anchored_and_one_factor_at_a_time(self):
        phase2 = {"learning_rate_init": 1e-3, "max_epochs": 12, "early_stopping_patience": 3}
        configs = build_candidate_configs(phase2, seed=20261026)
        self.assertEqual(len(configs), 4)
        self.assertEqual([c["learning_rate_init"] for c in configs], [0.001, 0.0005, 0.002, 0.001])
        self.assertEqual([c["max_epochs"] for c in configs], [12, 12, 12, 18])
        self.assertEqual({c["patience"] for c in configs}, {3})
        self.assertEqual({c["seed"] for c in configs}, {20261026})
        self.assertEqual(configs[0]["factor_changed"], "none; Phase 2 reproduction control")
        self.assertEqual(configs[1]["max_epochs"], configs[0]["max_epochs"])
        self.assertEqual(configs[2]["max_epochs"], configs[0]["max_epochs"])
        self.assertEqual(configs[3]["learning_rate_init"], configs[0]["learning_rate_init"])

    def test_configuration_validation_rejects_invalid_values(self):
        valid = {"name": "candidate", "learning_rate_init": .001,
                 "max_epochs": 12, "patience": 3, "seed": 7}
        self.assertEqual(validate_candidate_config(valid)["learning_rate_init"], .001)
        for patch in ({"learning_rate_init": 0}, {"learning_rate_init": float("nan")},
                      {"max_epochs": 0}, {"patience": -1}, {"seed": -1}, {"name": "../bad"}):
            with self.subTest(patch=patch), self.assertRaises((ValueError, TypeError)):
                validate_candidate_config({**valid, **patch})

    def test_early_stopping_tracks_best_score_and_patience(self):
        stopper = ValidationEarlyStopper(patience=2)
        self.assertEqual(stopper.update(1, .5), (True, False))
        self.assertEqual(stopper.update(2, .49), (False, False))
        self.assertEqual(stopper.update(3, .5), (False, True))
        self.assertEqual(stopper.best_epoch, 1)
        self.assertEqual(stopper.bad_epochs, 2)
        with self.assertRaises(ValueError):
            ValidationEarlyStopper(0)

    def test_validation_metrics_are_consistent_with_confusion_counts(self):
        cm = np.asarray([[8, 1, 0], [2, 3, 0], [0, 1, 2]], dtype=np.int64)
        result = metrics_from_confusion(cm, ["common", "rare_a", "rare_b"])
        self.assertAlmostEqual(result["accuracy"], np.trace(cm) / cm.sum())
        self.assertEqual(sum(row["support"] for row in result["per_class"]), int(cm.sum()))
        self.assertAlmostEqual(result["macro_f1"], np.mean([row["f1"] for row in result["per_class"]]))

    def test_same_seed_and_configuration_reproduce_local_training_parameters(self):
        x = np.asarray([[-2., -1.], [-1., -2.], [-2.2, -1.1], [2., 1.], [1., 2.],
                        [2.2, 1.1], [0., 3.], [.2, 2.8], [-.2, 3.1]], dtype=np.float32)
        y = np.asarray([0, 0, 0, 1, 1, 1, 2, 2, 2], dtype=np.int32)
        weights = np.asarray([1., 1., 1.])
        trained = []
        with threadpool_limits(limits=1):
            for _ in range(2):
                model = MLPClassifier(hidden_layer_sizes=(5, 3), batch_size=3, max_iter=1,
                                      shuffle=False, random_state=19, tol=0.0,
                                      learning_rate_init=.001)
                rng = np.random.default_rng(20261026)
                first, losses = True, []
                for _epoch in range(2):
                    loss, first = _fit_epoch(model, x, y, weights, rng, np.arange(3), first)
                    losses.append(loss)
                trained.append((model, losses))
        self.assertEqual(trained[0][1], trained[1][1])
        for a, b in zip(trained[0][0].coefs_, trained[1][0].coefs_):
            np.testing.assert_array_equal(a, b)

    def test_training_updates_use_only_train_arrays_and_never_reference_test(self):
        source = inspect.getsource(run_candidate)
        self.assertIn("_fit_epoch(model, x_train, y_train", source)
        self.assertNotIn("_fit_epoch(model, x_validation", source)
        self.assertNotIn("X_test", source)
        self.assertNotIn("y_test", source)
        self.assertIn("_evaluate_validation(model, x_validation, y_validation", source)

    def test_prepared_loader_works_without_test_arrays(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd() / "work") as temp:
            root = Path(temp)
            (root / "arrays").mkdir()
            prep = {"class_to_id": {"a": 0, "b": 1}, "feature_columns_retained": ["f0"]}
            phase1 = {"derived_rows": {"train": 4, "validation": 2, "test": 99},
                      "derived_class_counts": {"validation": {"a": 1, "b": 1}}}
            (root / "preprocessing_metadata.json").write_text(json.dumps(prep), encoding="utf-8")
            (root / "phase1_metadata.json").write_text(json.dumps(phase1), encoding="utf-8")
            np.save(root / "arrays" / "X_train.npy", np.ones((4, 1), dtype=np.float32))
            np.save(root / "arrays" / "y_train.npy", np.asarray([0, 0, 1, 1], dtype=np.int32))
            np.save(root / "arrays" / "X_validation.npy", np.ones((2, 1), dtype=np.float32))
            np.save(root / "arrays" / "y_validation.npy", np.asarray([0, 1], dtype=np.int32))
            arrays = load_train_validation_arrays(root)
            self.assertEqual(arrays[0].shape, (4, 1))
            self.assertEqual(arrays[2].shape, (2, 1))
            self.assertFalse((root / "arrays" / "X_test.npy").exists())
            self.assertFalse((root / "arrays" / "y_test.npy").exists())
            for arr in arrays[:4]:
                arr._mmap.close()
            del arrays


if __name__ == "__main__":
    unittest.main()
