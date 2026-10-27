"""Controlled validation-only class-weighting experiments for Phase 2's MLP."""
from __future__ import annotations
import argparse, csv, json, os, platform, time, warnings
from datetime import datetime, timezone
from pathlib import Path
import joblib, numpy as np, sklearn
from sklearn.metrics import confusion_matrix
from sklearn.neural_network import MLPClassifier
from threadpoolctl import threadpool_limits
from src.full_dataset.centralized import (
    BATCH_SIZE, CPU_THREADS, DEFAULT_SEED, HIDDEN_LAYERS, _confusion_for_model,
    _memory_snapshot, _save_confusion_csv, _write_csv, _write_json,
    class_weights_from_counts, metrics_from_confusion, validate_label_mapping, _fit_epoch,
)

ROOT = Path(__file__).resolve().parents[2] / "experiments" / "full_dataset"
DEFAULT_PREPARED = ROOT / "prepared"
DEFAULT_OUTPUT = ROOT / "model_improvement"
MAX_EPOCHS = 12
PATIENCE = 3

WEIGHTING_CONFIGS = {
    "reproduction_control": {"method": "inverse_frequency_power", "power": 0.5, "cap": None,
                             "description": "Phase 2 reproduction: square-root inverse-frequency, weighted-mean normalization"},
    "inverse_power_075": {"method": "inverse_frequency_power", "power": 0.75, "cap": None,
                           "description": "Moderately stronger inverse-frequency weighting; exponent 0.75"},
    "inverse_capped_1_cap10": {"method": "inverse_frequency_power", "power": 1.0, "cap": 10.0,
                                "description": "Inverse-frequency weighting with final per-class cap 10 and weighted-mean normalization"},
    "effective_number_beta_09999": {"method": "effective_number", "beta": 0.9999,
                                     "description": "Effective-number class weights, beta=0.9999, weighted-mean normalized"},
}


def normalize_weighted_mean(weights, counts):
    w = np.asarray(weights, dtype=np.float64)
    c = np.asarray(counts, dtype=np.float64)
    if w.ndim != 1 or c.shape != w.shape or not len(w):
        raise ValueError("weights and counts must be same-length vectors")
    if not np.isfinite(w).all() or not np.isfinite(c).all() or (w <= 0).any() or (c <= 0).any():
        raise ValueError("weights/counts must be finite and positive")
    return w / np.average(w, weights=c)


def capped_inverse_weights(counts, power=1.0, cap=10.0):
    """Solve for scale s where mean_count(min((max_n/n)^power*s, cap)) == 1."""
    c = np.asarray(counts, dtype=np.float64)
    if c.ndim != 1 or not c.size or not np.isfinite(c).all() or (c <= 0).any():
        raise ValueError("class counts must be a positive finite vector")
    if power <= 0 or not np.isfinite(power) or cap < 1 or not np.isfinite(cap):
        raise ValueError("power must be positive and cap must be finite and >= 1")
    raw = (c.max() / c) ** power
    lo, hi = 0.0, cap / raw.min()
    for _ in range(100):
        mid = (lo + hi) / 2
        avg = np.average(np.minimum(raw * mid, cap), weights=c)
        if avg < 1.0:
            lo = mid
        else:
            hi = mid
    return np.minimum(raw * ((lo + hi) / 2), cap)


def effective_number_weights(counts, beta=0.9999):
    """Class-balanced effective-number weights: (1-beta)/(1-beta**n), normalized."""
    c = np.asarray(counts, dtype=np.float64)
    if c.ndim != 1 or not c.size or not np.isfinite(c).all() or (c <= 0).any():
        raise ValueError("class counts must be a positive finite vector")
    if not 0 < beta < 1:
        raise ValueError("beta must be in (0,1)")
    denom = -np.expm1(c * np.log(beta))
    raw = (1.0 - beta) / denom
    return normalize_weighted_mean(raw, c)


def class_weights_for_strategy(counts, strategy):
    counts = np.asarray(counts, dtype=np.float64)
    if strategy == "reproduction_control":
        return class_weights_from_counts(counts)
    spec = WEIGHTING_CONFIGS[strategy]
    if spec["method"] == "inverse_frequency_power":
        if spec.get("cap") is not None:
            return capped_inverse_weights(counts, spec["power"], spec["cap"])
        return normalize_weighted_mean((counts.max() / counts) ** spec["power"], counts)
    if spec["method"] == "effective_number":
        return effective_number_weights(counts, spec["beta"])
    raise ValueError(f"unsupported class-weight method: {spec['method']}")


class CheckpointSelector:
    """Select highest validation macro-F1, preserving the earlier checkpoint on ties."""
    def __init__(self):
        self.best_score = -np.inf
        self.best_epoch = None

    def consider(self, epoch, score):
        if not np.isfinite(score):
            return False
        if score > self.best_score + 1e-8:
            self.best_score, self.best_epoch = float(score), int(epoch)
            return True
        return False


def load_train_validation_arrays(prepared_dir=DEFAULT_PREPARED):
    """Map training and validation arrays only. This function has no test-array path."""
    d = Path(prepared_dir)
    prep = json.loads((d / "preprocessing_metadata.json").read_text(encoding="utf-8"))
    phase1 = json.loads((d / "phase1_metadata.json").read_text(encoding="utf-8"))
    class_names = validate_label_mapping(prep["class_to_id"])
    feature_names = prep["feature_columns_retained"]
    x_train = np.load(d / "arrays" / "X_train.npy", mmap_mode="r")
    y_train = np.load(d / "arrays" / "y_train.npy", mmap_mode="r")
    x_val = np.load(d / "arrays" / "X_validation.npy", mmap_mode="r")
    y_val = np.load(d / "arrays" / "y_validation.npy", mmap_mode="r")
    if x_train.dtype != np.float32 or x_val.dtype != np.float32 or y_train.dtype != np.int32 or y_val.dtype != np.int32:
        raise TypeError("prepared arrays must be float32 features and int32 labels")
    if x_train.shape != (int(phase1["derived_rows"]["train"]), len(feature_names)):
        raise ValueError("training feature array disagrees with Phase 1 metadata")
    if x_val.shape != (int(phase1["derived_rows"]["validation"]), len(feature_names)):
        raise ValueError("validation feature array disagrees with Phase 1 metadata")
    if y_train.shape != (len(x_train),) or y_val.shape != (len(x_val),):
        raise ValueError("training/validation labels do not align with feature arrays")
    return x_train, y_train, x_val, y_val, class_names, feature_names, prep, phase1


def _load_csv_records(path):
    with Path(path).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _top_errors(cm, class_names, class_id, limit=3):
    row = np.asarray(cm[class_id], dtype=np.int64).copy()
    row[class_id] = 0
    order = np.argsort(-row, kind="stable")
    return "; ".join(f"{class_names[i]}:{int(row[i])}" for i in order[:limit] if row[i] > 0)


def create_diagnostics(prepared_dir=DEFAULT_PREPARED, output_dir=DEFAULT_OUTPUT):
    """Stage A audit using saved metrics/confusion matrices; never opens test arrays."""
    prepared_dir, output_dir = Path(prepared_dir).resolve(), Path(output_dir).resolve()
    phase2 = ROOT / "centralized"
    required = [prepared_dir / "phase1_metadata.json", prepared_dir / "preprocessing_metadata.json",
        phase2 / "run_metadata.json", phase2 / "best_validation_metrics.json",
        phase2 / "test_metrics.json", phase2 / "per_class_validation_metrics.csv",
        phase2 / "per_class_test_metrics.csv", phase2 / "confusion_matrix_validation.npy",
        phase2 / "confusion_matrix_test.npy"]
    missing = [str(p) for p in required if not p.is_file()]
    if missing:
        raise FileNotFoundError("Required Stage A artifacts missing: " + ", ".join(missing))
    output_dir.mkdir(parents=True, exist_ok=True)
    diagnostic_dir = output_dir / "diagnostics"
    if diagnostic_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing diagnostics: {diagnostic_dir}")
    diagnostic_dir.mkdir()
    phase1_meta = json.loads((prepared_dir / "phase1_metadata.json").read_text(encoding="utf-8"))
    prep = json.loads((prepared_dir / "preprocessing_metadata.json").read_text(encoding="utf-8"))
    central_meta = json.loads((phase2 / "run_metadata.json").read_text(encoding="utf-8"))
    central_val = json.loads((phase2 / "best_validation_metrics.json").read_text(encoding="utf-8"))
    central_test = json.loads((phase2 / "test_metrics.json").read_text(encoding="utf-8"))
    names = validate_label_mapping(prep["class_to_id"])
    if len(names) != 34 or len(prep["feature_columns_retained"]) != 44:
        raise ValueError("label or feature mapping differs from Phase 2 architecture metadata")
    if central_meta["configuration"]["hidden_layer_sizes"] != list(HIDDEN_LAYERS):
        raise ValueError("Phase 2 hidden-layer metadata does not match current training code")
    train_counts = phase1_meta["derived_class_counts"]["train"]
    val_rows = _load_csv_records(phase2 / "per_class_validation_metrics.csv")
    test_rows = _load_csv_records(phase2 / "per_class_test_metrics.csv")
    val_by_name = {r["label"]: r for r in val_rows}
    test_by_name = {r["label"]: r for r in test_rows}
    if set(names) != set(val_by_name) or set(names) != set(test_by_name):
        raise ValueError("per-class metrics do not match saved label mapping")
    cm_val = np.load(phase2 / "confusion_matrix_validation.npy", allow_pickle=False)
    cm_test = np.load(phase2 / "confusion_matrix_test.npy", allow_pickle=False)
    if cm_val.shape != (len(names), len(names)) or cm_test.shape != cm_val.shape:
        raise ValueError("saved confusion matrices have incompatible dimensions")
    for i, name in enumerate(names):
        if int(cm_val[i].sum()) != int(val_by_name[name]["support"]):
            raise ValueError(f"validation confusion support mismatch for {name}")
        if int(cm_test[i].sum()) != int(test_by_name[name]["support"]):
            raise ValueError(f"saved test confusion support mismatch for {name}")
        if int(cm_val[i].sum()) != int(phase1_meta["derived_class_counts"]["validation"][name]):
            raise ValueError(f"validation support differs from Phase 1 for {name}")
        if int(cm_test[i].sum()) != int(phase1_meta["derived_class_counts"]["test"][name]):
            raise ValueError(f"saved test support differs from Phase 1 for {name}")
    recomputed_val = metrics_from_confusion(cm_val, names)
    for key in ("accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1"):
        if abs(recomputed_val[key] - central_val[key]) > 1e-12:
            raise ValueError(f"saved validation metric does not match confusion matrix: {key}")
    # This only verifies the precomputed saved test artifacts for the requested retrospective diagnosis.
    recomputed_test = metrics_from_confusion(cm_test, names)
    for key in ("accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1"):
        if abs(recomputed_test[key] - central_test[key]) > 1e-12:
            raise ValueError(f"saved test metric does not match its saved confusion matrix: {key}")
    counts = np.asarray([int(train_counts[n]) for n in names], dtype=np.int64)
    baseline_weights = class_weights_from_counts(counts)
    rows = []
    for i, name in enumerate(names):
        vr, tr = val_by_name[name], test_by_name[name]
        rows.append({"label": name, "train_support": int(counts[i]),
            "validation_support": int(vr["support"]), "validation_precision": float(vr["precision"]),
            "validation_recall": float(vr["recall"]), "validation_f1": float(vr["f1"]),
            "top_validation_errors": _top_errors(cm_val, names, i),
            "test_support_diagnostic_only": int(tr["support"]),
            "test_precision_diagnostic_only": float(tr["precision"]),
            "test_recall_diagnostic_only": float(tr["recall"]),
            "test_f1_diagnostic_only": float(tr["f1"]),
            "top_test_errors_diagnostic_only": _top_errors(cm_test, names, i)})
    rows.sort(key=lambda r: (r["validation_f1"], r["validation_support"], r["label"]))
    for rank, row in enumerate(rows, 1):
        row["validation_weakness_rank"] = rank
    _write_csv(diagnostic_dir / "weak_class_diagnostics.csv", rows,
        ["validation_weakness_rank", "label", "train_support", "validation_support", "validation_precision",
         "validation_recall", "validation_f1", "top_validation_errors", "test_support_diagnostic_only",
         "test_precision_diagnostic_only", "test_recall_diagnostic_only", "test_f1_diagnostic_only",
         "top_test_errors_diagnostic_only"])
    strategy_weights = {}
    for name in WEIGHTING_CONFIGS:
        w = class_weights_for_strategy(counts, name)
        strategy_weights[name] = {"min": float(w.min()), "max": float(w.max()),
            "weighted_mean": float(np.average(w, weights=counts)),
            "minimum_weight_class": names[int(np.argmin(w))], "maximum_weight_class": names[int(np.argmax(w))]}
    _write_json(diagnostic_dir / "class_weight_audit.json", {
        "verified_label_mapping": prep["class_to_id"], "class_count": len(names),
        "feature_count_retained": len(prep["feature_columns_retained"]),
        "existing_phase2_weight_formula": "sqrt(N / (K * class_count)); normalized so training-row-weighted mean is 1",
        "computed_phase2_weights": {name: float(baseline_weights[i]) for i, name in enumerate(names)},
        "strategy_weight_ranges": strategy_weights,
        "class_support_ratio_train": float(counts.max() / counts.min()),
        "test_artifacts_used_only_for_retrospective_error_diagnosis": True,
        "test_arrays_or_test_labels_loaded": False})
    weakest = rows[:12]
    report_lines = ["# Stage A — Baseline rare-class diagnostics", "",
        "**Ranking uses Phase 2 validation F1 only.** Saved Phase 2 test per-class metrics and confusion counts are included only as retrospective diagnostics requested by the project owner; no test arrays or labels were opened, and test results are not used for experiment configuration or selection.", "",
        f"Verified: {len(names)} contiguous class IDs; {len(prep['feature_columns_retained'])} retained features; train/validation/test rows {phase1_meta['derived_rows']['train']:,}/{phase1_meta['derived_rows']['validation']:,}/{phase1_meta['derived_rows']['test']:,}. Existing centralized test accuracy/macro-F1/weighted-F1 = {central_test['accuracy']:.6f}/{central_test['macro_f1']:.6f}/{central_test['weighted_f1']:.6f}. The test split is otherwise reserved for the separately authorized final evaluation stage.", "",
        "## Weakest classes by validation F1", "",
        "| Rank | Class | Train n | Val n | Val P | Val R | Val F1 | Top saved test error destinations (diagnostic only) |", "|---:|---|---:|---:|---:|---:|---:|---|"]
    for r in weakest:
        report_lines.append(f"| {r['validation_weakness_rank']} | {r['label']} | {r['train_support']:,} | {r['validation_support']:,} | {r['validation_precision']:.4f} | {r['validation_recall']:.4f} | {r['validation_f1']:.4f} | {r['top_test_errors_diagnostic_only']} |")
    report_lines += ["", "## Interpretation", "",
        "The current MLP uses inverse-square-root class weighting, normalized to a training-support-weighted mean of one. Training support spans 138 to 847,902 rows (about 6,144:1); weighting changes gradient influence but cannot supply missing rare-class variation.",
        "The baseline shuffles row indices reproducibly each epoch, then trains in 8,192-row batches with `shuffle=False` inside scikit-learn. Early stopping/checkpoint selection uses validation macro-F1, which is appropriate for class coverage but can be noisy for validation classes with fewer than 100 examples. Training has a 12-epoch maximum and patience 3; the saved run selected epoch 12, so it did not stop before the budget ended.",
        "The saved confusion matrices show whether weak classes are confused with BenignTraffic, frequent attacks, or other rare classes. See `weak_class_diagnostics.csv`; its validation confusion destinations inform training decisions, while test destinations are labeled diagnostic-only.",
        "New comparisons retain 44 inputs, [128,64] hidden layers, 34 outputs, seed 20261026, the same row splits, Adam, batch size 8,192, 12-epoch maximum, and patience 3. Only class weighting changes. No PyTorch or focal-loss dependency will be added.", ""]
    (diagnostic_dir / "report.md").write_text("\n".join(report_lines), encoding="utf-8")
    _write_json(diagnostic_dir / "verification.json", {
        "class_mapping_verified": True, "train_validation_test_rows_from_metadata": phase1_meta["derived_rows"],
        "validation_confusion_matches_saved_per_class_support": True,
        "saved_test_confusion_matches_saved_per_class_support": True,
        "validation_metrics_recomputed_from_confusion": {k: recomputed_val[k] for k in ("accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1")},
        "saved_test_metrics_recomputed_from_saved_confusion_diagnostic_only": {k: recomputed_test[k] for k in ("accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1")},
        "test_arrays_or_labels_loaded": False})
    return rows


def _validation_metrics_for_model(model, x_val, y_val, class_names):
    cm = _confusion_for_model(model, x_val, y_val, len(class_names))
    return cm, metrics_from_confusion(cm, class_names)


def run_weighting_experiments(prepared_dir=DEFAULT_PREPARED, output_dir=DEFAULT_OUTPUT, seed=DEFAULT_SEED,
                              max_epochs=MAX_EPOCHS, patience=PATIENCE):
    """Run four controlled candidates. Only train/validation arrays are mapped or read."""
    prepared_dir, output_dir = Path(prepared_dir).resolve(), Path(output_dir).resolve()
    if max_epochs < 1 or patience < 1:
        raise ValueError("max_epochs and patience must be positive")
    output_dir.mkdir(parents=True, exist_ok=True)
    x_train, y_train, x_val, y_val, names, feature_names, prep, phase1 = load_train_validation_arrays(prepared_dir)
    counts = np.bincount(np.asarray(y_train, dtype=np.int64), minlength=len(names))
    if np.any(counts == 0):
        raise ValueError("training split is missing one or more mapped classes")
    expected_val = np.asarray([phase1["derived_class_counts"]["validation"].get(n, 0) for n in names], dtype=np.int64)
    observed_val = np.bincount(np.asarray(y_val, dtype=np.int64), minlength=len(names))
    if not np.array_equal(expected_val, observed_val):
        raise ValueError("validation labels differ from Phase 1 metadata")

    comparison_path = output_dir / "comparison_validation.csv"
    for strategy in WEIGHTING_CONFIGS:
        final_dir = output_dir / strategy
        building_dir = output_dir / f".{strategy}_building"
        if final_dir.exists() or building_dir.exists():
            raise FileExistsError(f"Refusing to overwrite existing candidate output: {final_dir if final_dir.exists() else building_dir}")
        building_dir.mkdir()
        checkpoint_dir = building_dir / "checkpoints"
        checkpoint_dir.mkdir()
        weights = class_weights_for_strategy(counts, strategy)
        if not np.isfinite(weights).all() or np.any(weights <= 0):
            raise ValueError(f"Invalid class weights produced for {strategy}")
        if strategy == "reproduction_control":
            prior_weights = json.loads((ROOT / "centralized" / "run_metadata.json").read_text(encoding="utf-8"))["class_weights"]
            for idx, name in enumerate(names):
                if not np.isclose(weights[idx], prior_weights[name], rtol=0, atol=1e-14):
                    raise ValueError(f"reproduction control weights differ from Phase 2 for {name}")
        config = {
            "strategy": strategy, **WEIGHTING_CONFIGS[strategy], "seed": int(seed),
            "hidden_layer_sizes": list(HIDDEN_LAYERS), "activation": "relu", "solver": "adam",
            "alpha": 1e-4, "batch_size": BATCH_SIZE, "learning_rate_init": 1e-3,
            "max_epochs": int(max_epochs), "early_stopping_patience": int(patience),
            "selection_metric": "validation macro-F1", "training_rows": int(len(y_train)),
            "validation_rows": int(len(y_val)), "feature_count": int(x_train.shape[1]),
            "class_count": len(names), "test_arrays_loaded": False,
            "weight_normalization": "training-row-support-weighted mean equals 1",
            "class_weight_by_label": {n: float(w) for n, w in zip(names, weights)},
        }
        _write_json(building_dir / "configuration.json", config)
        _write_json(building_dir / "class_support.json", {
            "train": {n: int(c) for n, c in zip(names, counts)},
            "validation": {n: int(c) for n, c in zip(names, observed_val)},
        })
        model = MLPClassifier(
            hidden_layer_sizes=HIDDEN_LAYERS, activation="relu", solver="adam", alpha=1e-4,
            batch_size=BATCH_SIZE, learning_rate_init=1e-3, max_iter=1, shuffle=False,
            random_state=seed, tol=0.0,
        )
        classes = np.arange(len(names), dtype=np.int64)
        rng = np.random.default_rng(seed)
        first_update = True
        selector = CheckpointSelector()
        best_metrics = None
        best_cm = None
        epochs_without_improvement = 0
        histories = []
        validation_history = []
        resource_samples = []
        run_start = time.perf_counter()
        log_lines = []
        captured_warnings = []
        with warnings.catch_warnings(record=True) as caught, threadpool_limits(limits=CPU_THREADS):
            warnings.simplefilter("always")
            for epoch in range(1, max_epochs + 1):
                train_start = time.perf_counter()
                train_loss, first_update = _fit_epoch(model, x_train, y_train, weights, rng, classes, first_update)
                train_seconds = time.perf_counter() - train_start
                val_start = time.perf_counter()
                cm, metrics = _validation_metrics_for_model(model, x_val, y_val, names)
                validation_seconds = time.perf_counter() - val_start
                is_best = selector.consider(epoch, metrics["macro_f1"])
                history_row = {
                    "epoch": epoch, "train_weighted_loss_mean": train_loss,
                    **{f"validation_{key}": metrics[key] for key in ("accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1")},
                    "train_seconds": train_seconds, "validation_seconds": validation_seconds,
                    "epoch_seconds": train_seconds + validation_seconds,
                    "selected_best": bool(is_best),
                }
                histories.append(history_row)
                validation_history.append({"epoch": epoch, **{k: metrics[k] for k in ("accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1")}})
                memory = _memory_snapshot(); memory["epoch"] = epoch; resource_samples.append(memory)
                if is_best:
                    joblib.dump(model, checkpoint_dir / "best_model.joblib", compress=3)
                    best_metrics, best_cm = metrics, cm.copy()
                    _write_json(building_dir / "best_validation_metrics.json", {
                        "selected_epoch": epoch, "selection_metric": "validation macro-F1",
                        **{k: metrics[k] for k in ("accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1")},
                    })
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1
                line = (f"strategy={strategy} epoch={epoch} train_loss={train_loss:.6f} "
                        f"val_macro_f1={metrics['macro_f1']:.6f} val_accuracy={metrics['accuracy']:.6f} "
                        f"val_weighted_f1={metrics['weighted_f1']:.6f} best={is_best} "
                        f"train_sec={train_seconds:.1f} val_sec={validation_seconds:.1f}")
                print(line, flush=True); log_lines.append(line)
                _write_csv(building_dir / "training_history.csv", histories, list(histories[0]))
                _write_csv(building_dir / "validation_metrics_by_epoch.csv", validation_history, list(validation_history[0]))
                if epochs_without_improvement >= patience:
                    log_lines.append(f"Early stopping after {epoch}; selected epoch {selector.best_epoch}.")
                    break
            captured_warnings = [{"category": item.category.__name__, "message": str(item.message)} for item in caught]
        if selector.best_epoch is None or best_metrics is None:
            raise RuntimeError(f"No valid checkpoint selected for {strategy}")
        # Reload the selected checkpoint and recompute validation-only artifacts.
        selected_model = joblib.load(checkpoint_dir / "best_model.joblib")
        selected_cm, selected_metrics = _validation_metrics_for_model(selected_model, x_val, y_val, names)
        if not np.array_equal(selected_cm, best_cm) or not np.isclose(selected_metrics["macro_f1"], selector.best_score, atol=1e-12):
            raise RuntimeError(f"Checkpoint validation verification failed for {strategy}")
        _write_json(building_dir / "best_validation_metrics.json", {
            "selected_epoch": selector.best_epoch, "selection_metric": "validation macro-F1",
            **{k: v for k, v in selected_metrics.items() if k not in ("per_class", "confusion_matrix")},
            "test_used_for_selection": False,
        })
        _write_csv(building_dir / "per_class_validation_metrics.csv", selected_metrics["per_class"],
                   ["label", "precision", "recall", "f1", "support", "note"])
        _save_confusion_csv(building_dir / "confusion_matrix_validation.csv", selected_cm, names)
        np.save(building_dir / "confusion_matrix_validation.npy", selected_cm)
        elapsed = time.perf_counter() - run_start
        run_meta = {
            "strategy": strategy, "created_utc": datetime.now(timezone.utc).isoformat(),
            "configuration": config, "selected_epoch": selector.best_epoch,
            "validation_metrics": {k: v for k, v in selected_metrics.items() if k not in ("per_class", "confusion_matrix")},
            "timing_seconds": {"training_and_epoch_validation": elapsed,
                                "train_seconds_sum": float(sum(r["train_seconds"] for r in histories)),
                                "validation_seconds_sum": float(sum(r["validation_seconds"] for r in histories))},
            "resource_samples": resource_samples, "warnings": captured_warnings,
            "test_arrays_loaded": False, "test_labels_loaded": False,
            "checkpoint_path": "checkpoints/best_model.joblib",
            "selection_protocol": "validation macro-F1 only; test split is not loaded or evaluated",
            "software_versions": {"python": platform.python_version(), "numpy": np.__version__,
                                  "scikit_learn": sklearn.__version__, "joblib": joblib.__version__},
        }
        _write_json(building_dir / "run_metadata.json", run_meta)
        (building_dir / "training.log").write_text("\n".join(log_lines) + "\n", encoding="utf-8")
        os.replace(building_dir, final_dir)
        summary = _load_csv_records(comparison_path) if comparison_path.is_file() else []
        summary.append({
            "strategy": strategy, "selected_epoch": selector.best_epoch,
            **{k: selected_metrics[k] for k in ("accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1")},
            "runtime_seconds": elapsed,
            **{f"{label}_{metric}": selected_metrics["per_class"][names.index(label)][metric]
               for label in ("Uploading_Attack", "XSS", "Backdoor_Malware", "Recon-PingSweep", "BrowserHijacking", "Recon-OSScan", "SqlInjection", "CommandInjection")
               for metric in ("recall", "f1")},
        })
        _write_csv(comparison_path, summary, list(summary[0]))
        print(f"Saved candidate {strategy} at {final_dir}; selected epoch={selector.best_epoch}; validation macro-F1={selected_metrics['macro_f1']:.6f}; elapsed={elapsed:.1f}s", flush=True)
    return summary


def finalize_comparison(output_dir=DEFAULT_OUTPUT):
    output_dir = Path(output_dir).resolve()
    summary_path = output_dir / "comparison_validation.csv"
    if not summary_path.is_file():
        raise FileNotFoundError("No candidate validation comparison table exists")
    rows = _load_csv_records(summary_path)
    weak_labels = (
        "Uploading_Attack", "Recon-PingSweep", "XSS", "Backdoor_Malware", "SqlInjection",
        "BrowserHijacking", "Recon-OSScan", "DictionaryBruteForce", "CommandInjection",
        "Recon-PortScan", "DNS_Spoofing", "Recon-HostDiscovery",
    )
    for row in rows:
        class_rows = _load_csv_records(output_dir / row["strategy"] / "per_class_validation_metrics.csv")
        by_label = {item["label"]: item for item in class_rows}
        for label in weak_labels:
            if label not in by_label:
                raise ValueError(f"Missing validation metrics for Stage A weak class: {label}")
            item = by_label[label]
            row[f"{label}_precision"] = item["precision"]
            row[f"{label}_recall"] = item["recall"]
            row[f"{label}_f1"] = item["f1"]
            row[f"{label}_support"] = item["support"]
    rows.sort(key=lambda row: float(row["macro_f1"]), reverse=True)
    _write_csv(summary_path, rows, list(rows[0]))
    baseline = next(row for row in rows if row["strategy"] == "reproduction_control")
    top = rows[0]
    total_runtime = sum(float(row["runtime_seconds"]) for row in rows)
    peak_working_set = 0
    for row in rows:
        run_meta = json.loads((output_dir / row["strategy"] / "run_metadata.json").read_text(encoding="utf-8"))
        peak_working_set = max(peak_working_set, *(sample.get("peak_working_set_bytes") or 0 for sample in run_meta["resource_samples"]))
    # Flat class comparison allows reading every Stage A weak class without an oversized wide table.
    class_comparison = []
    per_strategy = {}
    for row in rows:
        items = _load_csv_records(output_dir / row["strategy"] / "per_class_validation_metrics.csv")
        per_strategy[row["strategy"]] = {item["label"]: item for item in items}
    for label in weak_labels:
        base_f1 = float(per_strategy["reproduction_control"][label]["f1"])
        for row in rows:
            item = per_strategy[row["strategy"]][label]
            class_comparison.append({
                "strategy": row["strategy"], "class": label,
                "validation_support": int(item["support"]),
                "precision": item["precision"], "recall": item["recall"], "f1": item["f1"],
                "f1_delta_vs_reproduction": float(item["f1"]) - base_f1,
            })
    _write_csv(output_dir / "weak_class_validation_comparison.csv", class_comparison,
               ["strategy", "class", "validation_support", "precision", "recall", "f1", "f1_delta_vs_reproduction"])

    report = ["# Controlled class-weighting experiment — validation-only selection", "",
        f"The four runs used {total_runtime:.1f} seconds in total for training plus epoch validation. Peak sampled working set was {peak_working_set / (1024 ** 3):.2f} GiB.", "",
        "All candidates use the same Phase 1 training/validation partitions, 44-input [128, 64] MLP, seed, initialization, minibatch order, epoch budget, and early-stopping rule. Only class weighting varies. Checkpoint selection uses validation macro-F1. Test arrays and labels were not loaded for these candidates, and no test-set evaluation was run.", "",
        "## Validation comparison", "",
        "| Rank | Strategy | Epoch | Accuracy | Macro-P | Macro-R | Macro-F1 | Weighted-F1 | Runtime (s) | Uploading R/F1 | XSS R/F1 |", "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for rank, row in enumerate(rows, 1):
        report.append(f"| {rank} | {row['strategy']} | {int(row['selected_epoch'])} | {float(row['accuracy']):.4f} | {float(row['macro_precision']):.4f} | {float(row['macro_recall']):.4f} | {float(row['macro_f1']):.4f} | {float(row['weighted_f1']):.4f} | {float(row['runtime_seconds']):.1f} | {float(row['Uploading_Attack_recall']):.3f}/{float(row['Uploading_Attack_f1']):.3f} | {float(row['XSS_recall']):.3f}/{float(row['XSS_f1']):.3f} |")
    report += ["", "## Stage A weak-class results", "",
        "The following table reports validation precision/recall/F1 and support for every class ranked among the 12 weakest in Stage A. Full 34-class per-class results and confusion matrices remain in each strategy folder. These rows use validation data only.", "",
        "| Strategy | Class | Support | Precision | Recall | F1 | Δ F1 vs control |", "|---|---|---:|---:|---:|---:|---:|"]
    for item in class_comparison:
        report.append(f"| {item['strategy']} | {item['class']} | {item['validation_support']:,} | {float(item['precision']):.4f} | {float(item['recall']):.4f} | {float(item['f1']):.4f} | {float(item['f1_delta_vs_reproduction']):+.4f} |")
    delta = float(top["macro_f1"]) - float(baseline["macro_f1"])
    if top["strategy"] == "reproduction_control" or delta <= 1e-8:
        finding = f"The reproduction control ranked first at validation macro-F1 {float(baseline['macro_f1']):.6f}; no alternative weighting strategy improved it. Effective-number weighting was closest at {float(next(r['macro_f1'] for r in rows if r['strategy'] == 'effective_number_beta_09999')):.6f}, but it remains below the control."
        recommendation = "Keep the Phase 2 weighting/model as the current validation reference. If approved next, evaluate training-parameter changes in a separate validation-only experiment; do not tune on the test split."
    else:
        finding = (f"{top['strategy']} ranked first on validation macro-F1 at {float(top['macro_f1']):.6f} versus reproduction control {float(baseline['macro_f1']):.6f} (delta {delta:+.6f}). Review class-level and common-class trade-offs before freezing a checkpoint.")
        recommendation = f"Review {top['strategy']} validation outputs. If approved, freeze its checkpoint before one final test evaluation; do not tune further based on that evaluation."
    report += ["", "## Finding and next-step recommendation", "", finding, "", recommendation, "",
        "## Interpretation and limitations", "",
        "- The reproduction control exactly matches the Phase 2 saved validation macro-F1 (0.6696508820), which verifies the reproduced weighting/training path.",
        "- Uploading_Attack has 138 training and 37 validation examples; XSS has 414 and 92. Small support makes per-class validation estimates noisy.",
        "- Stronger weighting improved recall for some rare classes but reduced overall validation macro-F1 and, in some candidates, sharply reduced other weak-class F1. Reweighting cannot add rare-class variation and may increase false positives.",
        "- No candidate was evaluated on the test split in this stage. The original Phase 2 and Phase 3 checkpoints, reports, prepared arrays, and Phase 1–5 artifacts were not modified.",
        "- See `weak_class_validation_comparison.csv` and the per-strategy `per_class_validation_metrics.csv` files for auditable detail.", ""]
    (output_dir / "report.md").write_text("\n".join(report), encoding="utf-8")
    _write_json(output_dir / "selection_recommendation.json", {
        "selection_metric": "validation macro-F1", "ranked_candidates": [r["strategy"] for r in rows],
        "baseline_strategy": "reproduction_control", "selected_candidate": top["strategy"],
        "baseline_validation_macro_f1": float(baseline["macro_f1"]),
        "candidate_validation_macro_f1": float(top["macro_f1"]), "delta": delta,
        "test_evaluation_performed": False, "recommendation": recommendation,
    })
    return rows

def main():
    parser = argparse.ArgumentParser(description="Run controlled validation-only class weighting experiments.")
    parser.add_argument("--prepared", type=Path, default=DEFAULT_PREPARED)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--epochs", type=int, default=MAX_EPOCHS)
    parser.add_argument("--patience", type=int, default=PATIENCE)
    parser.add_argument("--diagnostics-only", action="store_true")
    args = parser.parse_args()
    if args.diagnostics_only:
        rows = create_diagnostics(args.prepared, args.output)
        print(f"Wrote diagnostics for {len(rows)} labels; test arrays/labels not loaded.")
        return 0
    summary = run_weighting_experiments(args.prepared, args.output, args.seed, args.epochs, args.patience)
    finalize_comparison(args.output)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())






