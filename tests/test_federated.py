"""Reproducibility and mathematical checks for simulated Phase 3 FedAvg."""
import copy
import tempfile
import unittest
from pathlib import Path
import joblib
import numpy as np
from sklearn.neural_network import MLPClassifier
from src.full_dataset.centralized import metrics_from_confusion
from src.full_dataset.federated import (
    fedavg_aggregate, initialize_global_model, load_model_artifact, model_signature, partition_clients,
)


class FederatedTests(unittest.TestCase):
    def test_partition_is_reproducible_disjoint_and_exhaustive(self):
        y = np.repeat(np.arange(4, dtype=np.int32), [41, 83, 127, 209])
        a = partition_clients(y, n_clients=3, alpha=1.0, seed=91)
        b = partition_clients(y, n_clients=3, alpha=1.0, seed=91)
        for left, right in zip(a, b):
            np.testing.assert_array_equal(left, right)
        merged = np.concatenate(a)
        self.assertEqual(len(merged), len(y))
        self.assertEqual(len(np.unique(merged)), len(y))
        np.testing.assert_array_equal(np.sort(merged), np.arange(len(y)))
        for indices in a:
            self.assertGreater(len(indices), 0)

    def test_three_client_sizes_sum_and_cover_source_classes(self):
        y = np.repeat(np.arange(3, dtype=np.int32), [100, 200, 300])
        clients = partition_clients(y, 3, 1.0, 2026)
        self.assertEqual(sum(map(len, clients)), len(y))
        recovered = np.concatenate([y[i] for i in clients])
        np.testing.assert_array_equal(np.bincount(recovered), [100, 200, 300])

    def test_fedavg_is_sample_weighted_and_preserves_compatible_signature(self):
        base = initialize_global_model(4, 3, seed=7)
        models = [copy.deepcopy(base), copy.deepcopy(base)]
        models[0].coefs_[0].fill(2.0); models[0].intercepts_[0].fill(2.0)
        models[1].coefs_[0].fill(8.0); models[1].intercepts_[0].fill(8.0)
        models[0].coefs_[1].fill(2.0); models[0].intercepts_[1].fill(2.0)
        models[1].coefs_[1].fill(8.0); models[1].intercepts_[1].fill(8.0)
        result = fedavg_aggregate(models, [1, 3])
        self.assertEqual(model_signature(result), model_signature(base))
        for parameter in [*result.coefs_[:2], *result.intercepts_[:2]]:
            np.testing.assert_allclose(parameter, 6.5, rtol=0, atol=1e-6)
        with self.assertRaises(ValueError):
            fedavg_aggregate(models, [0, 1])

    def test_client_local_partial_fit_updates_finite_parameters(self):
        from src.full_dataset.centralized import class_weights_from_counts
        from src.full_dataset.federated import _fit_local_epoch
        x = np.asarray([[-1.0, 0.0], [-0.9, 0.1], [-1.1, -0.1],
                        [1.0, 0.0], [0.9, 0.1], [1.1, -0.1],
                        [0.0, 1.0], [0.1, 0.9], [-0.1, 1.1]], dtype=np.float32)
        y = np.asarray([0, 0, 0, 1, 1, 1, 2, 2, 2], dtype=np.int32)
        model = initialize_global_model(2, 3, seed=7)
        before = [p.copy() for p in [*model.coefs_, *model.intercepts_]]
        loss, seconds = _fit_local_epoch(model, x, y, np.arange(len(y), dtype=np.int32),
                                         class_weights_from_counts(np.asarray([3, 3, 3])),
                                         np.arange(3), np.random.default_rng(5))
        self.assertTrue(np.isfinite(loss))
        self.assertGreaterEqual(seconds, 0)
        self.assertTrue(any(not np.array_equal(a, b) for a, b in zip(before, [*model.coefs_, *model.intercepts_])))
        self.assertTrue(all(np.isfinite(p).all() for p in [*model.coefs_, *model.intercepts_]))
        optimizer = model._optimizer
        _fit_local_epoch(model, x, y, np.arange(len(y), dtype=np.int32),
                         class_weights_from_counts(np.asarray([3, 3, 3])),
                         np.arange(3), np.random.default_rng(6))
        self.assertIs(model._optimizer, optimizer, "Adam state should continue between local epochs")
    def test_incompatible_models_are_rejected(self):
        a = initialize_global_model(4, 3, seed=7)
        b = initialize_global_model(5, 3, seed=7)
        with self.assertRaises(ValueError):
            fedavg_aggregate([a, b], [1, 1])

    def test_fresh_model_is_compatible_with_phase2_checkpoint_and_loads(self):
        prepared = Path(__file__).resolve().parents[1] / "experiments" / "full_dataset" / "prepared"
        checkpoint = prepared.parent / "centralized" / "checkpoints" / "best_model.joblib"
        model = joblib.load(checkpoint)
        fresh = initialize_global_model(model.n_features_in_, len(model.classes_), seed=20261026)
        self.assertEqual(model_signature(model), model_signature(fresh))
        with tempfile.TemporaryDirectory(dir=Path.cwd() / "work") as temp:
            path = Path(temp) / "checkpoint.joblib"
            joblib.dump(fresh, path)
            loaded = load_model_artifact(path, {f"class_{i}": i for i in range(len(model.classes_))})
            self.assertEqual(model_signature(loaded), model_signature(fresh))
            self.assertTrue(np.isfinite(loaded.coefs_[0]).all())

    def test_metric_calculation_is_consistent(self):
        cm = np.asarray([[7, 2, 0], [1, 4, 0], [0, 1, 0]], dtype=np.int64)
        result = metrics_from_confusion(cm, ["A", "B", "C"])
        self.assertEqual(result["evaluated_rows"], int(cm.sum()))
        self.assertAlmostEqual(result["accuracy"], 11 / 15)
        self.assertEqual(result["supported_classes"], 3)
        self.assertEqual(result["confusion_matrix"], cm.tolist())
        self.assertTrue(all(0 <= result[key] <= 1 for key in
                            ("accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1")))


if __name__ == "__main__":
    unittest.main()






