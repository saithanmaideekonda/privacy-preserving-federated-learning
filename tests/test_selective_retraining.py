"""Tests for per-client selective-retraining decisions and split safeguards."""
import tempfile, unittest
from pathlib import Path
import numpy as np
from src.full_dataset.federated import fedavg_aggregate, initialize_global_model
from src.full_dataset.selective_retraining import (
    DEFAULT_POLICY, candidate_acceptance, load_test_arrays_after_selection,
    load_training_validation_arrays, make_policy_decisions,
)


def fixture_rows():
    rows, events = [], []
    for scenario in ("no_drift", "synthetic_shift"):
        for client in (1, 2, 3):
            for window in range(1, 9):
                alert = scenario == "synthetic_shift" and client == 2 and window >= 4
                row = {"scenario": scenario, "client_id": str(client), "window_index": str(window),
                       "max_feature_psi": "0.5" if alert else "0.01", "top_feature": "Rate",
                       "threshold": "0.25", "alert": str(alert),
                       "synthetic_shift_applied": str(alert)}
                rows.append(row)
                if alert:
                    events.append((scenario, client, window))
    return rows, events


class SelectiveRetrainingTests(unittest.TestCase):
    def test_no_drift_produces_no_retraining_or_suppressed_alerts(self):
        rows, events = fixture_rows()
        decisions, _ = make_policy_decisions(rows, events, 0.25, DEFAULT_POLICY)
        no_drift = [d for d in decisions if d["scenario"] == "no_drift"]
        self.assertEqual(len(no_drift), 24)
        self.assertTrue(all(d["decision"] == "no_alert" for d in no_drift))
        self.assertFalse(any(d["retraining_triggered"] for d in no_drift))

    def test_two_consecutive_alerts_trigger_then_cooldown_suppresses(self):
        rows, events = fixture_rows()
        decisions, _ = make_policy_decisions(rows, events, 0.25, DEFAULT_POLICY)
        client2 = [d for d in decisions if d["scenario"] == "synthetic_shift" and d["client_id"] == 2]
        self.assertEqual([d["decision"] for d in client2[3:8]], [
            "pending_consecutive_alerts", "retraining_triggered", "suppressed_cooldown",
            "suppressed_cooldown", "suppressed_cooldown"])
        self.assertEqual(client2[4]["cooldown_until_window"], 8)

    def test_maximum_frequency_applies_after_cooldown(self):
        rows, events = [], []
        for window in range(1, 11):
            alert = window >= 2
            row = {"scenario": "s", "client_id": 1, "window_index": window,
                   "max_feature_psi": 0.5 if alert else 0.0, "top_feature": "x",
                   "threshold": 0.25, "alert": alert}
            rows.append(row)
            if alert:
                events.append(("s", 1, window))
        decisions, _ = make_policy_decisions(rows, events, 0.25, DEFAULT_POLICY)
        self.assertEqual(decisions[2]["decision"], "retraining_triggered")
        self.assertEqual(decisions[3]["decision"], "suppressed_cooldown")
        self.assertEqual(decisions[6]["decision"], "suppressed_max_frequency")

    def test_client_state_is_isolated_and_decisions_reproducible(self):
        rows, events = fixture_rows()
        first, _ = make_policy_decisions(rows, events, 0.25, DEFAULT_POLICY)
        second, _ = make_policy_decisions(rows, events, 0.25, DEFAULT_POLICY)
        self.assertEqual(first, second)
        triggers = [(d["scenario"], d["client_id"], d["window_index"])
                    for d in first if d["retraining_triggered"]]
        self.assertEqual(triggers, [("synthetic_shift", 2, 5)])
        self.assertTrue(all(d["decision"] == "no_alert" for d in first
                            if d["scenario"] == "synthetic_shift" and d["client_id"] in (1, 3)))

    def test_unchanged_clients_remain_unchanged_in_selective_aggregate(self):
        import copy
        base = initialize_global_model(4, 3, seed=8)
        base_before = [p.copy() for p in [*base.coefs_, *base.intercepts_]]
        affected = copy.deepcopy(base)
        for parameter in [*affected.coefs_, *affected.intercepts_]:
            parameter += np.float32(0.6)
        candidate = fedavg_aggregate([base, affected, base], [2, 3, 5])
        for actual, before in zip([*base.coefs_, *base.intercepts_], base_before):
            np.testing.assert_array_equal(actual, before)
        expected_delta = np.float32(0.6 * 3 / 10)
        for result, before in zip([*candidate.coefs_, *candidate.intercepts_], base_before):
            np.testing.assert_allclose(result - before, expected_delta, rtol=1e-5, atol=1e-6)
    def test_score_threshold_and_event_log_must_agree(self):
        rows, events = fixture_rows()
        with self.assertRaises(ValueError):
            make_policy_decisions(rows, events[:-1], 0.25, DEFAULT_POLICY)
        rows[0] = {**rows[0], "threshold": "0.2"}
        with self.assertRaises(ValueError):
            make_policy_decisions(rows, events, 0.25, DEFAULT_POLICY)

    def test_train_validation_loader_never_requires_test_arrays(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd() / "work") as folder:
            arrays = Path(folder) / "arrays"
            arrays.mkdir()
            np.save(arrays / "X_train.npy", np.zeros((5, 3), dtype=np.float32))
            np.save(arrays / "y_train.npy", np.zeros(5, dtype=np.int32))
            np.save(arrays / "X_validation.npy", np.ones((4, 3), dtype=np.float32))
            np.save(arrays / "y_validation.npy", np.ones(4, dtype=np.int32))
            loaded = load_training_validation_arrays(folder)
            self.assertEqual([a.shape[0] for a in loaded], [5, 5, 4, 4])
            self.assertFalse((arrays / "X_test.npy").exists())
            with self.assertRaises(PermissionError):
                load_test_arrays_after_selection(folder, candidate_selected=False)
            del loaded
            import gc; gc.collect()

    def test_candidate_acceptance_uses_validation_guardrails(self):
        def metrics(f1):
            return {"macro_f1": f1}
        accepted = candidate_acceptance(metrics(.50), metrics(.495), metrics(.40), metrics(.41), DEFAULT_POLICY)
        self.assertTrue(accepted["accepted"])
        rejected = candidate_acceptance(metrics(.50), metrics(.48), metrics(.40), metrics(.41), DEFAULT_POLICY)
        self.assertFalse(rejected["accepted"])


if __name__ == "__main__":
    unittest.main()



