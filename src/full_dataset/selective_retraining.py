"""Phase 5: PSI-triggered, client-selective retraining policy."""
from __future__ import annotations
import argparse, copy, csv, json, os, platform, time
from datetime import datetime, timezone
from pathlib import Path
import joblib, numpy as np, sklearn
from sklearn.metrics import confusion_matrix
from threadpoolctl import threadpool_limits
from src.full_dataset.centralized import BATCH_SIZE, CPU_THREADS, class_weights_from_counts, metrics_from_confusion, validate_label_mapping
from src.full_dataset.federated import DEFAULT_SEED, _fit_local_epoch, fedavg_aggregate, load_model_artifact, model_signature, partition_clients

ROOT = Path(__file__).resolve().parents[2] / "experiments" / "full_dataset"
DEFAULT_PREPARED, DEFAULT_FEDERATED = ROOT / "prepared", ROOT / "federated" / "experiment"
DEFAULT_DRIFT, DEFAULT_OUTPUT = ROOT / "drift", ROOT / "selective_retraining"
DEFAULT_POLICY = {"required_consecutive_alerts": 2, "cooldown_windows": 3,
                  "max_retrainings_per_client_per_scenario": 1, "candidate_local_epochs": 1,
                  "minimum_target_validation_macro_f1_improvement": 0.0,
                  "maximum_clean_validation_macro_f1_drop": 0.01}


def _bool(value):
    return value if isinstance(value, bool) else str(value).strip().lower() in {"true", "1", "yes"}


def event_key(row):
    return str(row["scenario"]), int(row["client_id"]), int(row["window_index"])


def make_policy_decisions(score_rows, event_keys, threshold, policy=DEFAULT_POLICY):
    """Apply consecutive-alert and cooldown rules independently per scenario/client."""
    required = int(policy["required_consecutive_alerts"])
    cooldown_len = int(policy["cooldown_windows"])
    max_count = int(policy["max_retrainings_per_client_per_scenario"])
    if required < 1 or cooldown_len < 0 or max_count < 1:
        raise ValueError("invalid policy parameters")
    by_client, scores = {}, {}
    for raw in score_rows:
        row = dict(raw)
        key = event_key(row)
        if key in scores:
            raise ValueError(f"duplicate drift score row: {key}")
        row.update(max_feature_psi=float(row["max_feature_psi"]), threshold=float(row["threshold"]),
                   alert=_bool(row["alert"]), synthetic_shift_applied=_bool(row.get("synthetic_shift_applied", False)))
        if abs(row["threshold"] - threshold) > 1e-12:
            raise ValueError(f"Phase 4 threshold mismatch in {key}")
        if row["alert"] != (row["max_feature_psi"] >= threshold):
            raise ValueError(f"score/alert mismatch in {key}")
        scores[key] = row
        by_client.setdefault((key[0], key[1]), []).append(row)
    alert_keys = {key for key, row in scores.items() if row["alert"]}
    if alert_keys != set(event_keys):
        raise ValueError("Phase 4 score rows and event log disagree")
    decisions = []
    for (scenario, client), rows in sorted(by_client.items()):
        rows.sort(key=lambda r: int(r["window_index"]))
        streak = 0
        previous_window = None
        previous_alert = False
        cooldown_until = 0
        triggered = 0
        for row in rows:
            wi, alert = int(row["window_index"]), row["alert"]
            if not alert:
                streak, decision, reason = 0, "no_alert", "PSI score below threshold"
            else:
                streak = streak + 1 if previous_alert and previous_window == wi - 1 else 1
                if wi <= cooldown_until:
                    decision, reason = "suppressed_cooldown", f"cooldown active through window {cooldown_until}"
                elif triggered >= max_count:
                    decision, reason = "suppressed_max_frequency", f"maximum {max_count} action(s) used for this client/scenario"
                elif streak < required:
                    decision, reason = "pending_consecutive_alerts", f"streak {streak}; requires {required}"
                else:
                    decision, reason = "retraining_triggered", f"{streak} consecutive alerts reached policy requirement"
                    triggered += 1
                    cooldown_until = wi + cooldown_len
            decisions.append({"scenario": scenario, "client_id": client, "window_index": wi,
                "max_feature_psi": row["max_feature_psi"], "top_feature": row["top_feature"],
                "threshold": float(threshold), "psi_alert": bool(alert),
                "synthetic_shift_applied": bool(row["synthetic_shift_applied"]),
                "consecutive_alerts": int(streak), "cooldown_until_window": int(cooldown_until),
                "decision": decision, "reason": reason,
                "retraining_triggered": decision == "retraining_triggered", "actual_training_run": False,
                "candidate_accepted": None})
            previous_window, previous_alert = wi, alert
    return decisions, scores


def load_training_validation_arrays(prepared_dir):
    """Read only prepared train/validation arrays as read-only memory maps."""
    d = Path(prepared_dir)
    xt = np.load(d / "arrays" / "X_train.npy", mmap_mode="r")
    yt = np.load(d / "arrays" / "y_train.npy", mmap_mode="r")
    xv = np.load(d / "arrays" / "X_validation.npy", mmap_mode="r")
    yv = np.load(d / "arrays" / "y_validation.npy", mmap_mode="r")
    if xt.dtype != np.float32 or xv.dtype != np.float32 or yt.dtype != np.int32 or yv.dtype != np.int32:
        raise TypeError("prepared arrays must use float32 features and int32 labels")
    if xt.ndim != 2 or xv.ndim != 2 or xt.shape[1] != xv.shape[1] or yt.shape != (len(xt),) or yv.shape != (len(xv),):
        raise ValueError("train/validation array shapes are incompatible")
    return xt, yt, xv, yv


def load_test_arrays_after_selection(prepared_dir, candidate_selected):
    """The only test loader; fails before touching test paths unless validation selected a candidate."""
    if not candidate_selected:
        raise PermissionError("test arrays are only accessible after candidate selection")
    d = Path(prepared_dir)
    return np.load(d / "arrays" / "X_test.npy", mmap_mode="r"), np.load(d / "arrays" / "y_test.npy", mmap_mode="r")


def replay_phase4_stream_indices(y_train, fed_config, drift_config):
    """Recreate Phase 4's deterministic client-local reference and stream slices."""
    clients = partition_clients(y_train, 3, float(fed_config["dirichlet_alpha"]), int(fed_config["partition_seed"]))
    rng = np.random.default_rng(int(drift_config["seed"]))
    ref_n, win_n, n_windows = (int(drift_config[k]) for k in
                                ("reference_rows_per_client", "window_rows", "windows_per_client"))
    streams = {}
    for cid, indices in enumerate(clients, 1):
        local = indices.copy()
        rng.shuffle(local)
        local = local[ref_n:ref_n + win_n * n_windows]
        streams[cid] = [local[i * win_n:(i + 1) * win_n] for i in range(n_windows)]
        if any(len(w) != win_n for w in streams[cid]):
            raise ValueError(f"insufficient training stream rows for client {cid}")
    return clients, streams


def evaluate_validation(model, x, y, class_names, shifted_indices=None, feature_idx=None, shift=0.0):
    """Chunked validation evaluation with optional synthetic shift on one client shard."""
    mask = None
    if shifted_indices is not None:
        mask = np.zeros(len(y), dtype=bool)
        mask[np.asarray(shifted_indices, dtype=np.int64)] = True
    cm = np.zeros((len(class_names), len(class_names)), dtype=np.int64)
    for start in range(0, len(y), BATCH_SIZE):
        end = min(start + BATCH_SIZE, len(y))
        xb = np.asarray(x[start:end], dtype=np.float32).copy()
        yb = np.asarray(y[start:end], dtype=np.int64)
        if mask is not None and mask[start:end].any():
            local_mask = mask[start:end]
            xb[local_mask, feature_idx] += np.float32(shift)
        pred = model.predict(xb).astype(np.int64, copy=False)
        cm += confusion_matrix(yb, pred, labels=np.arange(len(class_names), dtype=np.int64))
    return metrics_from_confusion(cm, class_names)


def _write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def _write_csv(path, rows, columns):
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader(); writer.writerows(rows)


def _metric_bundle(metrics):
    return {key: metrics[key] for key in ("accuracy", "macro_precision", "macro_recall", "macro_f1",
           "weighted_f1", "evaluated_rows", "supported_classes", "per_class", "confusion_matrix")}


def _save_validation_bundle(folder, name, metrics):
    _write_json(folder / f"{name}_metrics.json", _metric_bundle(metrics))
    np.save(folder / f"{name}_confusion.npy", np.asarray(metrics["confusion_matrix"], dtype=np.int64))
    _write_csv(folder / f"{name}_per_class.csv", metrics["per_class"],
               ["label", "precision", "recall", "f1", "support", "note"])


def candidate_acceptance(base_clean, candidate_clean, base_target, candidate_target, policy):
    target_gain = candidate_target["macro_f1"] - base_target["macro_f1"]
    clean_drop = base_clean["macro_f1"] - candidate_clean["macro_f1"]
    accepted = (target_gain > float(policy["minimum_target_validation_macro_f1_improvement"])
                and clean_drop <= float(policy["maximum_clean_validation_macro_f1_drop"]))
    return {"accepted": bool(accepted), "target_validation_macro_f1_gain": float(target_gain),
            "clean_validation_macro_f1_drop": float(clean_drop)}

def _report(meta, decisions, actions, candidate_results):
    lines = ["# Phase 5 — PSI-triggered selective retraining", "",
        f"**Created:** {meta['created_utc']}  ", f"**Runtime:** {meta['runtime_seconds']:.2f} seconds.  ",
        "**Scope:** Phase 4 no-drift and synthetic-shift scenarios only; no natural temporal drift was tested.", "",
        "## Policy", "",
        f"- PSI threshold **{meta['policy']['psi_threshold']:.2f}**, read unchanged from Phase 4.",
        f"- Trigger after **{meta['policy']['required_consecutive_alerts']} consecutive alert windows** for one client.",
        f"- Cooldown: **{meta['policy']['cooldown_windows']} windows after a trigger**.",
        f"- Maximum **{meta['policy']['max_retrainings_per_client_per_scenario']} action(s) per client/scenario**.",
        "- State is isolated per client and per scenario. Trigger decisions are logged separately from training and candidate selection.", "",
        "## Scenario decisions", "",
        "| Scenario | Alerts | Pending | Triggers | Cooldown-suppressed | Max-frequency-suppressed | Actual training runs | Accepted candidates |",
        "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for s in meta["scenario_summaries"]:
        lines.append(f"| {s['scenario']} | {s['alert_windows']} | {s['pending_alerts']} | {s['trigger_decisions']} | {s['cooldown_suppressed']} | {s['max_frequency_suppressed']} | {s['actual_training_runs']} | {s['accepted_candidates']} |")
    lines += ["", "### Alert-level decisions", "",
        "| Scenario | Client | Window | PSI | Decision | Reason | Training run |", "|---|---:|---:|---:|---|---|---|"]
    for row in decisions:
        if row["psi_alert"]:
            lines.append(f"| {row['scenario']} | {row['client_id']} | {row['window_index']} | {row['max_feature_psi']:.6f} | {row['decision']} | {row['reason']} | {'yes' if row['actual_training_run'] else 'no'} |")
    if not candidate_results:
        lines += ["", "No policy trigger occurred; no candidate training was run."]
    for cand in candidate_results:
        lines += ["", f"## Candidate: Client {cand['client_id']} — {cand['scenario']}", "",
            f"A real local training run was performed after the trigger at window {cand['trigger_window']}. It fitted on {cand['training_rows']:,} rows from the affected client's training partition, taken from the two consecutive alert windows. No validation or test rows were used for fitting.",
            f"Synthetic shift applied to fitting rows: **{'yes' if cand['synthetic_shift_training'] else 'no'}** ({cand['shift_feature'] or 'none'}, +{cand['shift_scaled_units']:.1f} standardized units). Only Client {cand['client_id']} was retrained; clients {', '.join(map(str,cand['skipped_clients']))} were skipped. Their parameters remained at the Phase 3 global checkpoint and were used unchanged in the weighted candidate aggregate.",
            f"Validation target macro-F1: **{cand['baseline_target_validation_macro_f1']:.6f} → {cand['candidate_target_validation_macro_f1']:.6f}** ({cand['target_validation_macro_f1_gain']:+.6f}). Clean validation macro-F1: **{cand['baseline_clean_validation_macro_f1']:.6f} → {cand['candidate_clean_validation_macro_f1']:.6f}** (drop {cand['clean_validation_macro_f1_drop']:.6f}).",
            f"Candidate: **{'accepted for final evaluation' if cand['accepted'] else 'rejected'}** under the configured validation rules. It was not deployed. Selected checkpoint remains {'the separate candidate checkpoint' if cand['accepted'] else 'the original Phase 3 FedAvg checkpoint'}."]
        if cand.get("test_evaluation") is None:
            lines.append("No Phase 5 test evaluation was performed because the candidate was rejected. Existing Phase 3 test results remain unchanged.")
        else:
            lines.append("The validation-selected candidate was evaluated once on the held-out test split after selection; see its candidate test metrics artifact.")
    lines += ["", "## Data use and limitations", "",
        "- Decisions consume Phase 4 `drift_scores.csv` and `event_log.json`; the PSI threshold and detector are unchanged.",
        "- Candidate fitting uses only affected-client training rows from the alert windows. Validation selection uses prepared validation, with a client-2 synthetic shifted view when evaluating the synthetic-shift scenario.",
        "- The test loader is guarded and is called only after validation accepts a candidate.",
        "- Dataset records have no reliable device/session/time identifiers. All streams and drift events here are synthetic; natural temporal drift is not validated.",
        "- Drift indicates a feature-distribution change and does not confirm an intrusion. Simulated clients on one computer do not provide formal privacy or network isolation.", ""]
    return "\n".join(lines)


def run_selective_retraining(prepared_dir=DEFAULT_PREPARED, federated_dir=DEFAULT_FEDERATED,
                             drift_dir=DEFAULT_DRIFT, output_dir=DEFAULT_OUTPUT,
                             policy_overrides=None, seed=DEFAULT_SEED):
    started = time.perf_counter()
    prepared_dir, federated_dir, drift_dir, output_dir = [Path(p).resolve() for p in
        (prepared_dir, federated_dir, drift_dir, output_dir)]
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite Phase 5 artifacts: {output_dir}")
    build_dir = output_dir.with_name(f".{output_dir.name}_building")
    if build_dir.exists():
        raise FileExistsError(f"Refusing to overwrite incomplete Phase 5 build: {build_dir}")
    required_paths = [prepared_dir / "phase1_metadata.json", prepared_dir / "preprocessing_metadata.json",
        federated_dir / "configuration.json", federated_dir / "client_partitions.json",
        federated_dir / "checkpoints" / "best_model.joblib", drift_dir / "configuration.json",
        drift_dir / "drift_scores.csv", drift_dir / "event_log.json"]
    if any(not p.is_file() for p in required_paths):
        raise FileNotFoundError("A required Phase 1-4 artifact is missing")
    fed_config = json.loads((federated_dir / "configuration.json").read_text(encoding="utf-8"))
    part_meta = json.loads((federated_dir / "client_partitions.json").read_text(encoding="utf-8"))
    drift_config = json.loads((drift_dir / "configuration.json").read_text(encoding="utf-8"))
    threshold = float(drift_config["threshold"])
    if threshold != 0.25:
        raise ValueError("Phase 4 PSI threshold changed; expected existing threshold 0.25")
    policy = dict(DEFAULT_POLICY)
    if policy_overrides:
        policy.update(policy_overrides)
    policy_doc = {**policy, "psi_threshold": threshold,
                  "threshold_source": "Phase 4 configuration.json; reused unchanged"}
    with (drift_dir / "drift_scores.csv").open(newline="", encoding="utf-8") as f:
        score_rows = list(csv.DictReader(f))
    event_doc = json.loads((drift_dir / "event_log.json").read_text(encoding="utf-8"))
    decisions, normalized_scores = make_policy_decisions(score_rows,
        [event_key(e) for e in event_doc["events"]], threshold, policy)

    x_train, y_train, x_val, y_val = load_training_validation_arrays(prepared_dir)
    phase1 = json.loads((prepared_dir / "phase1_metadata.json").read_text(encoding="utf-8"))
    preprocessing = json.loads((prepared_dir / "preprocessing_metadata.json").read_text(encoding="utf-8"))
    class_names = validate_label_mapping(preprocessing["class_to_id"])
    feature_names = preprocessing["feature_columns_retained"]
    if x_train.shape[1] != len(feature_names) or x_val.shape[1] != len(feature_names):
        raise ValueError("array features do not match fitted preprocessing metadata")
    if len(y_train) != int(phase1["derived_rows"]["train"]) or len(y_val) != int(phase1["derived_rows"]["validation"]):
        raise ValueError("train/validation row counts do not match Phase 1 metadata")
    train_counts = np.bincount(np.asarray(y_train, dtype=np.int64), minlength=len(class_names))
    class_weights = class_weights_from_counts(train_counts)
    client_indices, stream_indices = replay_phase4_stream_indices(y_train, fed_config, drift_config)
    if [len(a) for a in client_indices] != [int(c["sample_count"]) for c in part_meta["client_distributions"]]:
        raise ValueError("recreated client partitions disagree with Phase 3 counts")
    build_dir.mkdir(parents=True)
    candidates_dir = build_dir / "candidates"
    candidates_dir.mkdir()
    triggers = [row for row in decisions if row["retraining_triggered"]]
    actions, candidate_results = [], []

    val_clients = partition_clients(y_val, 3, float(fed_config["dirichlet_alpha"]),
                                    int(fed_config["partition_seed"]) + 1)
    drift_feature = drift_config["synthetic_shift"]["feature"]
    feature_idx = feature_names.index(drift_feature)
    for trigger in triggers:
        scenario, cid, trigger_window = trigger["scenario"], trigger["client_id"], trigger["window_index"]
        required = int(policy["required_consecutive_alerts"])
        alert_windows = list(range(trigger_window - required + 1, trigger_window + 1))
        prior = [d for d in decisions if d["scenario"] == scenario and d["client_id"] == cid
                 and d["psi_alert"] and d["window_index"] <= trigger_window]
        if [d["window_index"] for d in prior[-required:]] != alert_windows:
            raise AssertionError("trigger lacks required consecutive client alerts")
        source_indices = np.concatenate([stream_indices[cid][wi - 1] for wi in alert_windows]).astype(np.int32)
        local_x = np.asarray(x_train[source_indices], dtype=np.float32).copy()
        local_y = np.asarray(y_train[source_indices], dtype=np.int32)
        shifted = scenario == "synthetic_shift" and all(
            normalized_scores[(scenario, cid, wi)]["synthetic_shift_applied"] for wi in alert_windows)
        shift_amount = float(drift_config["synthetic_shift"]["shift_in_standardized_feature_units"])
        if shifted:
            local_x[:, feature_idx] += np.float32(shift_amount)

        base_model = load_model_artifact(federated_dir / "checkpoints" / "best_model.joblib", preprocessing["class_to_id"])
        local_model = copy.deepcopy(base_model)
        if hasattr(local_model, "_optimizer"):
            del local_model._optimizer
        local_history, rng = [], np.random.default_rng(seed + cid + trigger_window)
        with threadpool_limits(limits=CPU_THREADS):
            for epoch in range(1, int(policy["candidate_local_epochs"]) + 1):
                loss, seconds = _fit_local_epoch(local_model, local_x, local_y,
                    np.arange(len(local_y), dtype=np.int32), class_weights,
                    np.arange(len(class_names), dtype=np.int64), rng)
                local_history.append({"local_epoch": epoch, "rows": len(local_y),
                                      "weighted_loss": float(loss), "training_seconds": float(seconds)})
        local_file = f"{scenario}_client_{cid}_local_candidate.joblib"
        joblib.dump(local_model, candidates_dir / local_file, compress=3)
        client_counts = [int(c["sample_count"]) for c in part_meta["client_distributions"]]
        candidate_models = [local_model if client == cid else base_model for client in (1, 2, 3)]
        global_candidate = fedavg_aggregate(candidate_models, client_counts)
        if model_signature(global_candidate) != model_signature(base_model):
            raise ValueError("candidate global model is incompatible with Phase 3")
        global_file = f"{scenario}_client_{cid}_global_candidate.joblib"
        joblib.dump(global_candidate, candidates_dir / global_file, compress=3)

        baseline_clean = evaluate_validation(base_model, x_val, y_val, class_names)
        candidate_clean = evaluate_validation(global_candidate, x_val, y_val, class_names)
        target_indices = val_clients[cid - 1] if shifted else None
        baseline_target = evaluate_validation(base_model, x_val, y_val, class_names, target_indices,
                                               feature_idx if shifted else None, shift_amount if shifted else 0.0)
        candidate_target = evaluate_validation(global_candidate, x_val, y_val, class_names, target_indices,
                                                feature_idx if shifted else None, shift_amount if shifted else 0.0)
        target_gain = candidate_target["macro_f1"] - baseline_target["macro_f1"]
        clean_drop = baseline_clean["macro_f1"] - candidate_clean["macro_f1"]
        accepted = (target_gain > float(policy["minimum_target_validation_macro_f1_improvement"])
                    and clean_drop <= float(policy["maximum_clean_validation_macro_f1_drop"]))
        tag = f"{scenario}_client_{cid}"
        for name, metric in ((f"{tag}_baseline_clean_validation", baseline_clean),
                             (f"{tag}_candidate_clean_validation", candidate_clean),
                             (f"{tag}_baseline_target_validation", baseline_target),
                             (f"{tag}_candidate_target_validation", candidate_target)):
            _save_validation_bundle(candidates_dir, name, metric)
        test_metrics = None
        if accepted:
            x_test, y_test = load_test_arrays_after_selection(prepared_dir, True)
            test_cm = np.zeros((len(class_names), len(class_names)), dtype=np.int64)
            for start in range(0, len(y_test), BATCH_SIZE):
                end = min(start + BATCH_SIZE, len(y_test))
                xb = np.asarray(x_test[start:end], dtype=np.float32)
                yb = np.asarray(y_test[start:end], dtype=np.int64)
                pred = global_candidate.predict(xb).astype(np.int64, copy=False)
                test_cm += confusion_matrix(yb, pred, labels=np.arange(len(class_names)))
            test_metrics = metrics_from_confusion(test_cm, class_names)
            np.save(candidates_dir / f"{tag}_candidate_test_confusion.npy", test_cm)
            _write_json(candidates_dir / f"{tag}_candidate_test_metrics.json", _metric_bundle(test_metrics))
            del x_test, y_test
        result = {"scenario": scenario, "client_id": cid, "trigger_window": trigger_window,
            "trigger_alert_windows": alert_windows, "training_rows": int(len(local_y)),
            "training_source_split": "training", "synthetic_shift_training": bool(shifted),
            "shift_feature": drift_feature if shifted else None, "shift_scaled_units": shift_amount if shifted else 0.0,
            "local_epochs": int(policy["candidate_local_epochs"]), "local_training_history": local_history,
            "local_candidate_checkpoint": f"candidates/{local_file}",
            "global_candidate_checkpoint": f"candidates/{global_file}",
            "skipped_clients": [c for c in (1, 2, 3) if c != cid],
            "unaffected_client_parameters_changed_during_local_training": False,
            "baseline_clean_validation_macro_f1": float(baseline_clean["macro_f1"]),
            "candidate_clean_validation_macro_f1": float(candidate_clean["macro_f1"]),
            "baseline_target_validation_macro_f1": float(baseline_target["macro_f1"]),
            "candidate_target_validation_macro_f1": float(candidate_target["macro_f1"]),
            "target_validation_macro_f1_gain": float(target_gain),
            "clean_validation_macro_f1_drop": float(clean_drop),
            "minimum_target_validation_macro_f1_improvement": float(policy["minimum_target_validation_macro_f1_improvement"]),
            "maximum_clean_validation_macro_f1_drop": float(policy["maximum_clean_validation_macro_f1_drop"]),
            "accepted": bool(accepted), "deployment_performed": False,
            "test_evaluation": _metric_bundle(test_metrics) if test_metrics is not None else None}
        _write_json(candidates_dir / f"{tag}_candidate_summary.json", result)
        candidate_results.append(result)
        action = {"scenario": scenario, "client_id": cid, "trigger_window": trigger_window,
            "reason": trigger["reason"], "trigger_alert_windows": ",".join(map(str, alert_windows)),
            "affected_client_retrained": cid,
            "unaffected_clients_skipped": ",".join(map(str, result["skipped_clients"])),
            "actual_training_run": True, "training_rows": len(local_y),
            "accepted_by_validation": bool(accepted),
            "selected_checkpoint": f"candidates/{global_file}" if accepted else "Phase 3 FedAvg best checkpoint",
            "deployment_performed": False}
        actions.append(action)
        for decision in decisions:
            if (decision["scenario"], decision["client_id"], decision["window_index"]) == (scenario, cid, trigger_window):
                decision["actual_training_run"] = True
                decision["candidate_accepted"] = bool(accepted)

    summaries = []
    for scenario in sorted({row["scenario"] for row in decisions}):
        subset = [row for row in decisions if row["scenario"] == scenario]
        acted = [a for a in actions if a["scenario"] == scenario]
        summaries.append({"scenario": scenario, "windows_checked": len(subset),
            "alert_windows": sum(r["psi_alert"] for r in subset),
            "pending_alerts": sum(r["decision"] == "pending_consecutive_alerts" for r in subset),
            "trigger_decisions": sum(r["retraining_triggered"] for r in subset),
            "cooldown_suppressed": sum(r["decision"] == "suppressed_cooldown" for r in subset),
            "max_frequency_suppressed": sum(r["decision"] == "suppressed_max_frequency" for r in subset),
            "actual_training_runs": len(acted),
            "accepted_candidates": sum(a["accepted_by_validation"] for a in acted)})

    created = datetime.now(timezone.utc).isoformat()
    policy_doc = {**policy, "psi_threshold": threshold,
                  "threshold_source": "Phase 4 configuration.json; reused unchanged"}
    config_record = {"created_utc": created, "policy": policy_doc,
        "drift_score_source": str(drift_dir / "drift_scores.csv"),
        "drift_event_source": str(drift_dir / "event_log.json"),
        "phase4_detector_changed": False,
        "candidate_training": {"initial_checkpoint": str(federated_dir / "checkpoints" / "best_model.joblib"),
            "fitting_data": "affected client's source training rows from consecutive trigger windows; recorded synthetic shift applied to copied rows only",
            "candidate_selection": "clean and scenario-target validation macro-F1; no test use",
            "test_access": "test arrays opened only after candidate acceptance"}}
    _write_json(build_dir / "configuration.json", config_record)
    _write_csv(build_dir / "policy_decisions.csv", decisions,
        ["scenario", "client_id", "window_index", "max_feature_psi", "top_feature", "threshold", "psi_alert",
         "synthetic_shift_applied", "consecutive_alerts", "cooldown_until_window", "decision", "reason",
         "retraining_triggered", "actual_training_run", "candidate_accepted"])
    _write_json(build_dir / "decision_log.json", {"created_utc": created,
        "alert_decisions": [r for r in decisions if r["psi_alert"]], "retraining_actions": actions})
    _write_csv(build_dir / "retraining_actions.csv", actions,
        ["scenario", "client_id", "trigger_window", "reason", "trigger_alert_windows",
         "affected_client_retrained", "unaffected_clients_skipped", "actual_training_run", "training_rows",
         "accepted_by_validation", "selected_checkpoint", "deployment_performed"])
    _write_csv(build_dir / "scenario_summary.csv", summaries,
        ["scenario", "windows_checked", "alert_windows", "pending_alerts", "trigger_decisions",
         "cooldown_suppressed", "max_frequency_suppressed", "actual_training_runs", "accepted_candidates"])
    metadata = {"created_utc": created, "status": "completed",
        "software": {"python": platform.python_version(), "numpy": np.__version__,
                     "scikit_learn": sklearn.__version__, "joblib": joblib.__version__},
        "data_splits": {"training_rows": int(len(y_train)), "validation_rows": int(len(y_val)),
            "test_arrays_loaded": any(c["test_evaluation"] is not None for c in candidate_results),
            "test_used_for_trigger_or_training": False},
        "policy": policy_doc, "scenario_summaries": summaries,
        "retraining_actions": actions, "candidate_results": candidate_results,
        "runtime_seconds": time.perf_counter() - started,
        "limitations": ["All Phase 4 windows and distribution changes are synthetic; no natural temporal drift is validated.",
            "PSI indicates distribution change, not an attack.",
            "Clients are single-process simulations without OS-level isolation or formal privacy guarantees.",
            "Candidate acceptance is based only on the documented validation rule; deployment is not implemented."]}
    _write_json(build_dir / "run_metadata.json", metadata)
    (build_dir / "report.md").write_text(_report(metadata, decisions, actions, candidate_results), encoding="utf-8")
    os.replace(build_dir, output_dir)
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, default=DEFAULT_PREPARED)
    parser.add_argument("--federated", type=Path, default=DEFAULT_FEDERATED)
    parser.add_argument("--drift", type=Path, default=DEFAULT_DRIFT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--required-consecutive-alerts", type=int, default=2)
    parser.add_argument("--cooldown-windows", type=int, default=3)
    parser.add_argument("--max-retrainings-per-client", type=int, default=1)
    parser.add_argument("--candidate-local-epochs", type=int, default=1)
    args = parser.parse_args()
    overrides = {"required_consecutive_alerts": args.required_consecutive_alerts,
        "cooldown_windows": args.cooldown_windows,
        "max_retrainings_per_client_per_scenario": args.max_retrainings_per_client,
        "candidate_local_epochs": args.candidate_local_epochs}
    result = run_selective_retraining(args.prepared, args.federated, args.drift,
                                     args.output, overrides, args.seed)
    print(json.dumps({"status": result["status"], "runtime_seconds": result["runtime_seconds"],
        "scenario_summaries": result["scenario_summaries"], "actions": result["retraining_actions"]}, indent=2))

if __name__ == "__main__":
    main()
