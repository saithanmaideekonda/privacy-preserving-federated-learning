"""Tests for validation-only Stage 3 selection and one-shot test gating."""
import hashlib
import inspect
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from sklearn.neural_network import MLPClassifier
from threadpoolctl import threadpool_limits

from src.full_dataset.centralized import _fit_epoch, metrics_from_confusion
from src.full_dataset.robustness_evaluation import (
    ADDITIONAL_SEEDS, FOCUS_CLASSES, _check_frozen, aggregate_seed_metrics,
    evaluate_frozen_candidate_once, evaluate_selection_rule, sha256_file, write_json,
    run_validation_stage,
)

class RobustnessEvaluationTests(unittest.TestCase):
    def _run(self, seed, score=.61, uploading_recall=0.0):
        per_class = {name: {"precision": .5, "recall": .5, "f1": .5, "support": 20}
                     for name in FOCUS_CLASSES}
        per_class["Uploading_Attack"]["recall"] = uploading_recall
        per_class["Uploading_Attack"]["f1"] = uploading_recall
        return {"seed": seed, "accuracy": .96, "macro_precision": .60,
            "macro_recall": .61, "macro_f1": score, "weighted_f1": .96,
            "per_class": per_class}

    def _baseline(self):
        per_class = {name: {"precision": .5, "recall": .5, "f1": .5, "support": 20}
                     for name in FOCUS_CLASSES}
        per_class["Uploading_Attack"]["recall"] = 0.0
        per_class["Uploading_Attack"]["f1"] = 0.0
        return {"accuracy": .95, "macro_precision": .59, "macro_recall": .60,
            "macro_f1": .60, "weighted_f1": .95, "per_class": per_class}

    def test_seed_plan_is_fixed_and_reproducible(self):
        self.assertEqual(ADDITIONAL_SEEDS, (20261028, 20261029))
        self.assertEqual(tuple(sorted(set(ADDITIONAL_SEEDS))), ADDITIONAL_SEEDS)

    def test_seed_aggregation_mean_sample_std_and_range(self):
        rows = [self._run(seed, score=value) for seed, value in
                zip((1, 2, 3, 4), (.60, .62, .64, .66))]
        a = aggregate_seed_metrics(rows)
        b = aggregate_seed_metrics(rows)
        self.assertEqual(a, b)
        self.assertAlmostEqual(a["metrics"]["macro_f1"]["mean"], .63)
        self.assertAlmostEqual(a["metrics"]["macro_f1"]["sample_std"], np.std([.60,.62,.64,.66], ddof=1))
        self.assertAlmostEqual(a["metrics"]["macro_f1"]["range"], .06)

    def test_selection_gate_passes_only_when_every_condition_holds(self):
        runs = [self._run(10+i, uploading_recall=.04 if i < 2 else 0.0) for i in range(4)]
        decision = evaluate_selection_rule(self._baseline(), runs)
        self.assertTrue(decision["passed"], decision)
        runs[1]["per_class"]["Uploading_Attack"]["recall"] = 0.0
        self.assertFalse(evaluate_selection_rule(self._baseline(), runs)["passed"])

    def test_checkpoint_hash_is_verified(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd() / "work") as temp:
            root = Path(temp); checkpoint = root / "frozen_candidate.joblib"
            checkpoint.write_bytes(b"frozen weights")
            write_json(root / "selection_frozen.json", {
                "checkpoint_path": checkpoint.name, "checkpoint_sha256": sha256_file(checkpoint),
                "test_evaluation_performed": False})
            frozen, found = _check_frozen(root)
            self.assertEqual(found, checkpoint)
            checkpoint.write_bytes(b"modified")
            with self.assertRaisesRegex(RuntimeError, "SHA-256"):
                _check_frozen(root)

    def test_test_arrays_cannot_be_loaded_before_freeze(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd() / "work") as temp, patch("src.full_dataset.robustness_evaluation.np.load") as loader:
            with self.assertRaisesRegex(RuntimeError, "not frozen"):
                evaluate_frozen_candidate_once(Path(temp))
            loader.assert_not_called()

    def test_training_stage_source_does_not_reference_test_arrays(self):
        source = inspect.getsource(run_validation_stage)
        self.assertNotIn("X_test", source)
        self.assertNotIn("y_test", source)

    def test_small_training_run_is_reproducible(self):
        x = np.asarray([[-2., -1.], [-1., -2.], [-2.2, -1.1], [2., 1.], [1., 2.],
                        [2.2, 1.1], [0., 3.], [.2, 2.8], [-.2, 3.1]], dtype=np.float32)
        y = np.asarray([0, 0, 0, 1, 1, 1, 2, 2, 2], dtype=np.int32)
        outputs = []
        with threadpool_limits(limits=1):
            for _ in range(2):
                model = MLPClassifier(hidden_layer_sizes=(5, 3), batch_size=3, max_iter=1,
                    shuffle=False, random_state=29, tol=0.0, learning_rate_init=.002)
                rng = np.random.default_rng(20261028); first = True; losses = []
                for _epoch in range(2):
                    loss, first = _fit_epoch(model, x, y, np.ones(3), rng, np.arange(3), first)
                    losses.append(loss)
                outputs.append((model, losses))
        self.assertEqual(outputs[0][1], outputs[1][1])
        for left, right in zip(outputs[0][0].coefs_, outputs[1][0].coefs_):
            np.testing.assert_array_equal(left, right)

    def test_metric_summary_matches_confusion_matrix(self):
        cm = np.asarray([[8, 1], [2, 3]], dtype=np.int64)
        metrics = metrics_from_confusion(cm, ["common", "rare"])
        self.assertEqual(metrics["evaluated_rows"], int(cm.sum()))
        self.assertAlmostEqual(metrics["accuracy"], np.trace(cm) / cm.sum())
        self.assertEqual(sum(row["support"] for row in metrics["per_class"]), int(cm.sum()))

if __name__ == "__main__":
    unittest.main()


