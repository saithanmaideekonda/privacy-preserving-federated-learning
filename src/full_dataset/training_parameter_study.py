"""Validation-only centralized MLP training-parameter study (Stage 2)."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import sklearn
from sklearn.neural_network import MLPClassifier
from threadpoolctl import threadpool_limits

from src.full_dataset.centralized import (
    BATCH_SIZE, CPU_THREADS, DEFAULT_SEED, HIDDEN_LAYERS, _confusion_for_model,
    _fit_epoch, _memory_snapshot, _save_confusion_csv, _write_csv, _write_json,
    class_weights_from_counts, load_model_artifact, metrics_from_confusion,
)
from src.full_dataset.model_improvement import load_train_validation_arrays

ROOT = Path(__file__).resolve().parents[2] / "experiments" / "full_dataset"
DEFAULT_OUTPUT = ROOT / "model_improvement" / "training_parameters"
FOCUS_CLASSES = (
    "Uploading_Attack", "Recon-PingSweep", "XSS", "Backdoor_Malware", "SqlInjection",
    "BrowserHijacking", "Recon-OSScan", "DictionaryBruteForce", "CommandInjection",
    "Recon-PortScan", "DNS_Spoofing", "Recon-HostDiscovery",
)


def validate_candidate_config(config: dict) -> dict:
    """Validate a candidate's tunable fields; all non-tunable fields come from Phase 2."""
    if not isinstance(config, dict):
        raise TypeError("candidate configuration must be a dictionary")
    required = {"name", "learning_rate_init", "max_epochs", "patience", "seed"}
    if not required.issubset(config):
        raise ValueError(f"candidate config is missing fields: {sorted(required - set(config))}")
    name = config["name"]
    if not isinstance(name, str) or not name or any(ch in name for ch in "\\/:*?\"<>|"):
        raise ValueError("candidate name must be a non-empty safe directory name")
    learning_rate = float(config["learning_rate_init"])
    if not np.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate_init must be finite and positive")
    for field in ("max_epochs", "patience", "seed"):
        value = config[field]
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{field} must be an integer")
    if config["max_epochs"] < 1 or config["patience"] < 1 or config["seed"] < 0:
        raise ValueError("max_epochs/patience must be positive and seed non-negative")
    return {**config, "learning_rate_init": learning_rate}


def build_candidate_configs(phase2_configuration: dict, seed: int = DEFAULT_SEED) -> list[dict]:
    """Create a small one-factor-at-a-time grid anchored to the saved Phase 2 settings."""
    baseline_lr = float(phase2_configuration["learning_rate_init"])
    baseline_epochs = int(phase2_configuration["max_epochs"])
    patience = int(phase2_configuration["early_stopping_patience"])
    configs = [
        {"name": "phase2_reproduction_lr_0p001_epochs_12", "learning_rate_init": baseline_lr,
         "max_epochs": baseline_epochs, "patience": patience, "seed": seed,
         "factor_changed": "none; Phase 2 reproduction control"},
        {"name": "learning_rate_half_epochs_12", "learning_rate_init": baseline_lr * 0.5,
         "max_epochs": baseline_epochs, "patience": patience, "seed": seed,
         "factor_changed": "learning_rate_init only (0.5x Phase 2)"},
        {"name": "learning_rate_double_epochs_12", "learning_rate_init": baseline_lr * 2.0,
         "max_epochs": baseline_epochs, "patience": patience, "seed": seed,
         "factor_changed": "learning_rate_init only (2x Phase 2)"},
        {"name": "budget_18_lr_baseline", "learning_rate_init": baseline_lr,
         "max_epochs": int(np.ceil(baseline_epochs * 1.5)), "patience": patience, "seed": seed,
         "factor_changed": "max_epochs only (1.5x Phase 2)"},
    ]
    return [validate_candidate_config(config) for config in configs]


class ValidationEarlyStopper:
    """Track best validation macro-F1; ties keep the earlier checkpoint."""
    def __init__(self, patience: int):
        if isinstance(patience, bool) or not isinstance(patience, int) or patience < 1:
            raise ValueError("patience must be a positive integer")
        self.patience = patience
        self.best_score = -np.inf
        self.best_epoch = None
        self.bad_epochs = 0

    def update(self, epoch: int, score: float) -> tuple[bool, bool]:
        if not np.isfinite(score):
            self.bad_epochs += 1
            return False, self.bad_epochs >= self.patience
        if score > self.best_score + 1e-8:
            self.best_score, self.best_epoch, self.bad_epochs = float(score), int(epoch), 0
            return True, False
        self.bad_epochs += 1
        return False, self.bad_epochs >= self.patience


def _count_labels(y: np.ndarray, n_classes: int, chunk_size: int = 250_000) -> np.ndarray:
    counts = np.zeros(n_classes, dtype=np.int64)
    for start in range(0, len(y), chunk_size):
        chunk = np.asarray(y[start:start + chunk_size], dtype=np.int64)
        if chunk.size and (chunk.min() < 0 or chunk.max() >= n_classes):
            raise ValueError("observed label ID is outside the saved mapping")
        counts += np.bincount(chunk, minlength=n_classes)
    return counts


def _evaluate_validation(model, x_validation, y_validation, class_names):
    """Inference only: validation features never enter partial_fit or gradient updates."""
    cm = _confusion_for_model(model, x_validation, y_validation, len(class_names))
    return cm, metrics_from_confusion(cm, class_names)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _phase2_context():
    phase2_dir = ROOT / "centralized"
    required = [phase2_dir / "run_metadata.json", phase2_dir / "best_validation_metrics.json",
                phase2_dir / "checkpoints" / "best_model.joblib"]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Phase 2 artifacts missing: " + ", ".join(missing))
    metadata = json.loads(required[0].read_text(encoding="utf-8"))
    validation = json.loads(required[1].read_text(encoding="utf-8"))
    config = metadata["configuration"]
    if tuple(config["hidden_layer_sizes"]) != HIDDEN_LAYERS or int(config["batch_size"]) != BATCH_SIZE:
        raise ValueError("Phase 2 architecture or batch size differs from current implementation")
    if config["class_weighting"] != "normalized inverse square-root of training class frequency":
        raise ValueError("Phase 2 class-weighting method differs from requested fixed strategy")
    return metadata, validation, required[2]


def run_candidate(config: dict, prepared_dir=ROOT / "prepared", output_dir=DEFAULT_OUTPUT) -> dict:
    """Run a single candidate; only train arrays update parameters and only validation selects."""
    config = validate_candidate_config(config)
    output_dir = Path(output_dir).resolve()
    final_dir = output_dir / config["name"]
    build_dir = output_dir / f".{config['name']}_building"
    if final_dir.exists() or build_dir.exists():
        raise FileExistsError(f"Refusing to overwrite candidate output: {final_dir if final_dir.exists() else build_dir}")

    x_train, y_train, x_validation, y_validation, class_names, feature_names, prep, phase1 = load_train_validation_arrays(prepared_dir)
    if x_train.shape[1] != 44 or len(class_names) != 34:
        raise ValueError("prepared feature/class dimensions differ from the Phase 2 MLP")
    train_counts = _count_labels(y_train, len(class_names))
    val_counts = _count_labels(y_validation, len(class_names))
    for split, observed in (("train", train_counts), ("validation", val_counts)):
        expected = np.asarray([phase1["derived_class_counts"][split].get(name, 0) for name in class_names], dtype=np.int64)
        if not np.array_equal(observed, expected):
            raise ValueError(f"{split} label counts differ from Phase 1 metadata")
    weights = class_weights_from_counts(train_counts)
    phase2_meta, phase2_validation, _ = _phase2_context()
    saved_weights = phase2_meta["class_weights"]
    for class_id, name in enumerate(class_names):
        if not np.isclose(weights[class_id], saved_weights[name], rtol=0, atol=1e-14):
            raise ValueError(f"training class weight differs from Phase 2 for {name}")

    runtime_config = {
        **config,
        "model": "sklearn.neural_network.MLPClassifier",
        "hidden_layer_sizes": list(HIDDEN_LAYERS), "activation": "relu", "solver": "adam",
        "alpha": float(phase2_meta["configuration"]["alpha"]), "batch_size": BATCH_SIZE,
        "cpu_threads": CPU_THREADS, "selection_metric": "validation macro-F1",
        "class_weighting": "Phase 2 normalized inverse square-root class weights; fixed across candidates",
        "class_weight_by_label": {name: float(weight) for name, weight in zip(class_names, weights)},
        "training_rows": int(len(y_train)), "validation_rows": int(len(y_validation)),
        "feature_count": int(x_train.shape[1]), "class_count": len(class_names),
        "test_arrays_loaded": False, "test_labels_loaded": False,
    }
    build_dir.mkdir(parents=True)
    (build_dir / "checkpoints").mkdir()
    _write_json(build_dir / "configuration.json", runtime_config)
    _write_json(build_dir / "class_support.json", {
        "train": {name: int(count) for name, count in zip(class_names, train_counts)},
        "validation": {name: int(count) for name, count in zip(class_names, val_counts)},
    })
    model = MLPClassifier(
        hidden_layer_sizes=HIDDEN_LAYERS, activation="relu", solver="adam",
        alpha=runtime_config["alpha"], batch_size=BATCH_SIZE,
        learning_rate_init=config["learning_rate_init"], max_iter=1, shuffle=False,
        random_state=config["seed"], tol=0.0,
    )
    stopper = ValidationEarlyStopper(config["patience"])
    rng = np.random.default_rng(config["seed"])
    classes = np.arange(len(class_names), dtype=np.int64)
    first_update = True
    history, validation_history, resource_samples, log_lines = [], [], [], []
    best_metrics = best_cm = None
    run_start = time.perf_counter()
    caught_warnings = []
    with warnings.catch_warnings(record=True) as caught, threadpool_limits(limits=CPU_THREADS):
        warnings.simplefilter("always")
        for epoch in range(1, config["max_epochs"] + 1):
            train_start = time.perf_counter()
            # Only the training split is passed into the parameter-update routine.
            train_loss, first_update = _fit_epoch(model, x_train, y_train, weights, rng, classes, first_update)
            train_seconds = time.perf_counter() - train_start
            validation_start = time.perf_counter()
            cm, metrics = _evaluate_validation(model, x_validation, y_validation, class_names)
            validation_seconds = time.perf_counter() - validation_start
            improved, should_stop = stopper.update(epoch, metrics["macro_f1"])
            history_row = {
                "epoch": epoch, "train_weighted_loss_mean": train_loss,
                **{f"validation_{key}": metrics[key] for key in ("accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1")},
                "train_seconds": train_seconds, "validation_seconds": validation_seconds,
                "epoch_seconds": train_seconds + validation_seconds, "selected_best": bool(improved),
                "early_stopping_bad_epochs": stopper.bad_epochs,
            }
            history.append(history_row)
            validation_history.append({"epoch": epoch, **{key: metrics[key] for key in ("accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1")}})
            resource = _memory_snapshot(); resource["epoch"] = epoch; resource_samples.append(resource)
            if improved:
                joblib.dump(model, build_dir / "checkpoints" / "best_model.joblib", compress=3)
                best_metrics, best_cm = metrics, cm.copy()
                _write_json(build_dir / "best_validation_metrics.json", {
                    "selected_epoch": epoch, "selection_metric": "validation macro-F1",
                    **{key: value for key, value in metrics.items() if key not in ("per_class", "confusion_matrix")},
                    "test_used_for_selection": False,
                })
            line = (f"candidate={config['name']} epoch={epoch} train_loss={train_loss:.6f} "
                    f"val_macro_f1={metrics['macro_f1']:.6f} val_accuracy={metrics['accuracy']:.6f} "
                    f"val_weighted_f1={metrics['weighted_f1']:.6f} best={improved} "
                    f"train_sec={train_seconds:.1f} val_sec={validation_seconds:.1f}")
            print(line, flush=True); log_lines.append(line)
            _write_csv(build_dir / "training_history.csv", history, list(history[0]))
            _write_csv(build_dir / "validation_metrics_by_epoch.csv", validation_history, list(validation_history[0]))
            if should_stop:
                log_lines.append(f"Early stopping after epoch {epoch}; selected epoch {stopper.best_epoch}.")
                break
        caught_warnings = [{"category": item.category.__name__, "message": str(item.message)} for item in caught]
    if stopper.best_epoch is None or best_metrics is None:
        raise RuntimeError("No validation checkpoint was selected")

    selected_model = load_model_artifact(build_dir / "checkpoints" / "best_model.joblib", prep["class_to_id"])
    selected_cm, selected_metrics = _evaluate_validation(selected_model, x_validation, y_validation, class_names)
    if not np.array_equal(selected_cm, best_cm) or not np.isclose(selected_metrics["macro_f1"], stopper.best_score, rtol=0, atol=1e-12):
        raise RuntimeError("saved validation-selected checkpoint failed its validation recheck")
    _write_json(build_dir / "best_validation_metrics.json", {
        "selected_epoch": stopper.best_epoch, "selection_metric": "validation macro-F1",
        **{key: value for key, value in selected_metrics.items() if key not in ("per_class", "confusion_matrix")},
        "test_used_for_selection": False,
    })
    _write_csv(build_dir / "per_class_validation_metrics.csv", selected_metrics["per_class"],
               ["label", "precision", "recall", "f1", "support", "note"])
    _save_confusion_csv(build_dir / "confusion_matrix_validation.csv", selected_cm, class_names)
    np.save(build_dir / "confusion_matrix_validation.npy", selected_cm)
    elapsed = time.perf_counter() - run_start
    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(), "configuration": runtime_config,
        "validation_metrics": {key: value for key, value in selected_metrics.items() if key not in ("per_class", "confusion_matrix")},
        "selected_epoch": stopper.best_epoch, "epochs_completed": len(history),
        "timing_seconds": {"training_and_epoch_validation": elapsed,
                            "train_seconds_sum": sum(row["train_seconds"] for row in history),
                            "validation_seconds_sum": sum(row["validation_seconds"] for row in history)},
        "resource_samples": resource_samples, "warnings": caught_warnings,
        "software_versions": {"python": platform.python_version(), "numpy": np.__version__,
                               "scikit_learn": sklearn.__version__, "joblib": joblib.__version__},
        "checkpoint_path": "checkpoints/best_model.joblib",
        "checkpoint_sha256": _sha256(build_dir / "checkpoints" / "best_model.joblib"),
        "gradient_update_split": "training only", "model_selection_split": "validation only",
        "test_arrays_loaded": False, "test_labels_loaded": False,
    }
    _write_json(build_dir / "run_metadata.json", metadata)
    (build_dir / "training.log").write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    os.replace(build_dir, final_dir)
    print(f"Saved {config['name']}: selected epoch {stopper.best_epoch}, validation macro-F1 {stopper.best_score:.6f}, elapsed {elapsed:.1f}s", flush=True)
    return {"name": config["name"], "selected_epoch": stopper.best_epoch,
            **{key: selected_metrics[key] for key in ("accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1")},
            "runtime_seconds": elapsed}


def _read_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def finalize_study(output_dir=DEFAULT_OUTPUT) -> list[dict]:
    output_dir = Path(output_dir).resolve()
    rows = []
    for config_path in output_dir.glob("*/configuration.json"):
        candidate_dir = config_path.parent
        metrics = json.loads((candidate_dir / "best_validation_metrics.json").read_text(encoding="utf-8"))
        metadata = json.loads((candidate_dir / "run_metadata.json").read_text(encoding="utf-8"))
        per_class = {item["label"]: item for item in _read_rows(candidate_dir / "per_class_validation_metrics.csv")}
        row = {"candidate": candidate_dir.name, "selected_epoch": metrics["selected_epoch"],
               "learning_rate_init": metadata["configuration"]["learning_rate_init"],
               "max_epochs": metadata["configuration"]["max_epochs"], "epochs_completed": metadata["epochs_completed"],
               **{key: metrics[key] for key in ("accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1")},
               "runtime_seconds": metadata["timing_seconds"]["training_and_epoch_validation"],
               "peak_working_set_bytes": max((sample.get("peak_working_set_bytes") or 0) for sample in metadata["resource_samples"]),
               "warnings_count": len(metadata["warnings"])}
        for label in FOCUS_CLASSES:
            item = per_class[label]
            for metric in ("precision", "recall", "f1", "support"):
                row[f"{label}_{metric}"] = item[metric]
        rows.append(row)
    if not rows:
        raise FileNotFoundError("No completed parameter-study candidates found")
    rows.sort(key=lambda row: float(row["macro_f1"]), reverse=True)
    comparison_path = output_dir / "comparison_validation.csv"
    _write_csv(comparison_path, rows, list(rows[0]))

    phase2_meta, phase2_val, phase2_checkpoint = _phase2_context()
    weighting_reproduction = _read_rows(ROOT / "model_improvement" / "comparison_validation.csv")
    weighting_control = next((row for row in weighting_reproduction if row["strategy"] == "reproduction_control"), None)
    if weighting_control is None:
        raise ValueError("completed weighting-study reproduction control is missing")
    baseline_row = next((row for row in rows if row["candidate"] == "phase2_reproduction_lr_0p001_epochs_12"), None)
    if baseline_row is None:
        raise ValueError("Phase 2 reproduction candidate is missing")
    if not np.isclose(float(baseline_row["macro_f1"]), float(phase2_val["macro_f1"]), rtol=0, atol=1e-12):
        raise ValueError("new Phase 2 reproduction candidate does not match saved Phase 2 validation macro-F1")
    if not np.isclose(float(baseline_row["macro_f1"]), float(weighting_control["macro_f1"]), rtol=0, atol=1e-12):
        raise ValueError("new reproduction differs from the completed weighting-study control")
    for row in rows:
        row["macro_f1_delta_vs_phase2"] = float(row["macro_f1"]) - float(phase2_val["macro_f1"])
    _write_csv(comparison_path, rows, list(rows[0]))
    best = rows[0]
    previous_weak = _read_rows(ROOT / "model_improvement" / "diagnostics" / "weak_class_diagnostics.csv")
    phase1 = json.loads((ROOT / "prepared" / "phase1_metadata.json").read_text(encoding="utf-8"))
    phase2_class_metrics = {row["label"]: row for row in _read_rows(ROOT / "centralized" / "per_class_validation_metrics.csv")}
    weak_rows = []
    for old in sorted(previous_weak, key=lambda item: int(item["validation_weakness_rank"])):
        label = old["label"]
        for row in rows:
            metrics = {item["label"]: item for item in _read_rows(output_dir / row["candidate"] / "per_class_validation_metrics.csv")}
            item = metrics[label]
            weak_rows.append({"candidate": row["candidate"], "class": label,
                              "train_support": int(phase1["derived_class_counts"]["train"][label]),
                              "validation_support": int(item["support"]),
                              "precision": item["precision"], "recall": item["recall"], "f1": item["f1"],
                              "f1_delta_vs_phase2": float(item["f1"]) - float(phase2_class_metrics[label]["f1"])})
    _write_csv(output_dir / "weak_class_validation_comparison.csv", weak_rows,
               ["candidate", "class", "train_support", "validation_support", "precision", "recall", "f1", "f1_delta_vs_phase2"])

    aggregate_runtime = sum(float(row["runtime_seconds"]) for row in rows)
    peak_bytes = max(int(row["peak_working_set_bytes"]) for row in rows)
    report = ["# Stage 2 — Controlled centralized training-parameter study", "",
              f"Four runs used {aggregate_runtime:.1f} seconds in total for training plus validation; peak sampled working set was {peak_bytes / (1024 ** 3):.2f} GiB.", "",
              "All runs use the prepared Phase 1 train/validation arrays, identical preprocessing/feature order/label mapping, the Phase 2 MLP (44 inputs, hidden layers 128/64, 34 outputs), the original inverse-square-root class weights, 8,192-row minibatches, Adam, alpha 1e-4, and seed 20261026. Only the named learning-rate or epoch-budget factor changes. Validation macro-F1 selects the checkpoint. Test arrays and labels were not loaded.", "",
              "## Reproduction controls", "",
              "| Reference | Accuracy | Macro precision | Macro recall | Macro-F1 | Weighted-F1 |", "|---|---:|---:|---:|---:|---:|",
              f"| Saved Phase 2 validation | {float(phase2_val['accuracy']):.6f} | {float(phase2_val['macro_precision']):.6f} | {float(phase2_val['macro_recall']):.6f} | {float(phase2_val['macro_f1']):.6f} | {float(phase2_val['weighted_f1']):.6f} |",
              f"| Weighting-study reproduction | {float(weighting_control['accuracy']):.6f} | {float(weighting_control['macro_precision']):.6f} | {float(weighting_control['macro_recall']):.6f} | {float(weighting_control['macro_f1']):.6f} | {float(weighting_control['weighted_f1']):.6f} |",
              f"| Stage 2 reproduction run | {float(baseline_row['accuracy']):.6f} | {float(baseline_row['macro_precision']):.6f} | {float(baseline_row['macro_recall']):.6f} | {float(baseline_row['macro_f1']):.6f} | {float(baseline_row['weighted_f1']):.6f} |", "",
              "## Validation comparison", "",
              "| Rank | Candidate | Learning rate | Epoch budget | Epochs completed | Selected epoch | Accuracy | Macro-P | Macro-R | Macro-F1 | Weighted-F1 | Δ Macro-F1 | Runtime (s) |", "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for rank, row in enumerate(rows, 1):
        report.append(f"| {rank} | {row['candidate']} | {float(row['learning_rate_init']):.4g} | {row['max_epochs']} | {row['epochs_completed']} | {row['selected_epoch']} | {float(row['accuracy']):.4f} | {float(row['macro_precision']):.4f} | {float(row['macro_recall']):.4f} | {float(row['macro_f1']):.4f} | {float(row['weighted_f1']):.4f} | {float(row['macro_f1_delta_vs_phase2']):+.4f} | {float(row['runtime_seconds']):.1f} |")
    report += ["", "All primary runs completed their configured epoch caps; the patience-3 rule was monitored but did not terminate a run early. The 18-epoch candidate selected epoch 17 because epoch 18 had lower validation macro-F1.", "",
               "## Weak-class validation results", "",
               "The following table includes all 12 classes ranked weakest in Stage A. Support is shown for both training and validation; candidate scores are validation-only.", "",
               "| Candidate | Class | Train n | Val n | Precision | Recall | F1 | Δ F1 vs Phase 2 |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for item in weak_rows:
        report.append(f"| {item['candidate']} | {item['class']} | {item['train_support']:,} | {item['validation_support']:,} | {float(item['precision']):.4f} | {float(item['recall']):.4f} | {float(item['f1']):.4f} | {float(item['f1_delta_vs_phase2']):+.4f} |")
    base = float(baseline_row["macro_f1"])
    if best["candidate"] == baseline_row["candidate"] or float(best["macro_f1"]) <= base + 1e-8:
        finding = f"The Phase 2 reproduction remains the top validation candidate at macro-F1 {base:.6f}; no training-parameter candidate improved it."
        recommendation = "Retain the Phase 2 model/configuration as the current selected reference. The evidence does not justify freezing a new parameter candidate for test evaluation."
        selected_row = baseline_row
    else:
        finding = f"{best['candidate']} ranked first at validation macro-F1 {float(best['macro_f1']):.6f} versus Phase 2 reproduction {base:.6f}; inspect rare-class and overall trade-offs before deciding whether to freeze it."
        recommendation = f"Freeze {best['candidate']} only if the per-class trade-offs are acceptable; its checkpoint is recorded in selected_checkpoint.json. Test evaluation was not run."
        selected_row = best
    recommendation += " Do not tune FedAvg or use test results to revise this selection."
    selected_checkpoint = output_dir / selected_row["candidate"] / "checkpoints" / "best_model.joblib"
    selected_metadata = json.loads((output_dir / selected_row["candidate"] / "run_metadata.json").read_text(encoding="utf-8"))
    frozen = {"selected_candidate": selected_row["candidate"],
              "selection_metric": "validation macro-F1", "validation_macro_f1": float(selected_row["macro_f1"]),
              "checkpoint_path": str(selected_checkpoint.relative_to(output_dir)),
              "checkpoint_sha256": _sha256(selected_checkpoint),
              "configuration": selected_metadata["configuration"],
              "test_evaluation_performed": False,
              "frozen_utc": datetime.now(timezone.utc).isoformat()}
    _write_json(output_dir / "selected_checkpoint.json", frozen)
    report += ["", "## Selection and recommendation", "", finding, "", recommendation, "",
               "## Controls and limitations", "",
               "- The Phase 2 reproduction and completed weighting-study reproduction are both used as validation controls; their selected macro-F1 values match exactly.",
               "- The longer-budget candidate uses the same learning rate and patience; a longer maximum does not guarantee better generalization.",
               "- Rare-class metrics are noisy where validation support is small. Improvements for individual classes must be weighed against macro-F1, common-class performance, and the complete per-class table.",
               "- No candidate received test evaluation. The selected checkpoint metadata is frozen for a separately approved final evaluation.", "",
               "## Artifacts", "",
               "Each candidate directory contains its checkpoint, configuration, class support, training history, validation metrics by epoch, per-class validation metrics, validation confusion matrix, run metadata, and training log.", ""]
    (output_dir / "report.md").write_text("\n".join(report), encoding="utf-8")
    _write_json(output_dir / "selection_recommendation.json", {
        "ranked_candidates": [row["candidate"] for row in rows],
        "phase2_validation_macro_f1": float(phase2_val["macro_f1"]),
        "weighting_study_reproduction_validation_macro_f1": float(weighting_control["macro_f1"]),
        "selected_candidate": selected_row["candidate"],
        "selected_validation_macro_f1": float(selected_row["macro_f1"]),
        "test_evaluation_performed": False, "recommendation": recommendation,
    })
    protected_paths = {
        "phase1_inspection_report": ROOT / "inspection_report.md",
        "phase1_preparation_report": ROOT / "prepared" / "preparation_report.md",
        "phase1_metadata": ROOT / "prepared" / "phase1_metadata.json",
        "phase1_preprocessing_metadata": ROOT / "prepared" / "preprocessing_metadata.json",
        "phase2_checkpoint": ROOT / "centralized" / "checkpoints" / "best_model.joblib",
        "phase2_validation_metrics": ROOT / "centralized" / "best_validation_metrics.json",
        "phase2_report": ROOT / "centralized" / "report.md",
        "phase3_checkpoint": ROOT / "federated" / "experiment" / "checkpoints" / "best_model.joblib",
        "phase3_report": ROOT / "federated" / "experiment" / "report.md",
        "phase4_report": ROOT / "drift" / "report.md",
        "phase5_report": ROOT / "selective_retraining" / "report.md",
        "weighting_diagnostics_report": ROOT / "model_improvement" / "diagnostics" / "report.md",
        "weighting_study_report": ROOT / "model_improvement" / "report.md",
        "weighting_study_selection": ROOT / "model_improvement" / "selection_recommendation.json",
    }
    preserved_hashes = {}
    for name, path in protected_paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"Expected prior artifact missing during preservation audit: {path}")
        preserved_hashes[name] = {"path": str(path.resolve()), "size_bytes": path.stat().st_size,
                                  "sha256_after_stage": _sha256(path),
                                  "last_write_time_ns": path.stat().st_mtime_ns}
    _write_json(output_dir / "preservation_audit.json", {
        "note": "This stage writes only below experiments/full_dataset/model_improvement/training_parameters/. These hashes and timestamps record the prior artifacts' state after Stage 2; the Stage 2 code uses them read-only.",
        "prior_artifacts": preserved_hashes,
        "all_candidates_test_arrays_loaded": False,
        "all_candidates_test_labels_loaded": False,
    })
    return rows


def finalize_seed_repeat(output_dir=DEFAULT_OUTPUT):
    """Summarize the second-seed repeat without changing the primary fixed-seed ranking."""
    output_dir = Path(output_dir).resolve()
    primary_dir = output_dir / "learning_rate_double_epochs_12"
    repeat_dir = output_dir / "seed_repeat_20261027" / "learning_rate_double_seed_20261027"
    if not (primary_dir / "run_metadata.json").is_file() or not (repeat_dir / "run_metadata.json").is_file():
        raise FileNotFoundError("Primary 0.002 run or second-seed repeat is incomplete")
    runs = []
    for name, path in (("seed_20261026", primary_dir), ("seed_20261027", repeat_dir)):
        metadata = json.loads((path / "run_metadata.json").read_text(encoding="utf-8"))
        metrics = json.loads((path / "best_validation_metrics.json").read_text(encoding="utf-8"))
        config = metadata["configuration"]
        if float(config["learning_rate_init"]) != 0.002 or int(config["max_epochs"]) != 12:
            raise ValueError("seed repeat does not match the primary promising configuration")
        runs.append({"run": name, "seed": config["seed"], "epochs_completed": metadata["epochs_completed"], "selected_epoch": metrics["selected_epoch"],
                     **{key: metrics[key] for key in ("accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1")},
                     "runtime_seconds": metadata["timing_seconds"]["training_and_epoch_validation"],
                     "test_evaluation_performed": False})
    score_a, score_b = float(runs[0]["macro_f1"]), float(runs[1]["macro_f1"])
    baseline = json.loads((ROOT / "centralized" / "best_validation_metrics.json").read_text(encoding="utf-8"))["macro_f1"]
    runs[0]["macro_f1_delta_vs_phase2"] = score_a - baseline
    runs[0]["macro_f1_delta_vs_primary_seed"] = 0.0
    runs[1]["macro_f1_delta_vs_phase2"] = score_b - baseline
    runs[1]["macro_f1_delta_vs_primary_seed"] = score_b - score_a
    _write_csv(output_dir / "seed_repeat_validation_metrics.csv", runs, list(runs[0]))

    weak_rows = []
    for run, path in ((runs[0], primary_dir), (runs[1], repeat_dir)):
        per_class = {item["label"]: item for item in _read_rows(path / "per_class_validation_metrics.csv")}
        for label in FOCUS_CLASSES:
            item = per_class[label]
            weak_rows.append({"run": run["run"], "seed": run["seed"], "class": label,
                              "validation_support": item["support"], "precision": item["precision"],
                              "recall": item["recall"], "f1": item["f1"]})
    _write_csv(output_dir / "seed_repeat_weak_class_validation.csv", weak_rows,
               ["run", "seed", "class", "validation_support", "precision", "recall", "f1"])
    focus_class_summary = {}
    weak_by_key = {(row["run"], row["class"]): row for row in weak_rows}
    for label in FOCUS_CLASSES:
        focus_class_summary[label] = {
            run["run"]: {"validation_support": int(weak_by_key[(run["run"], label)]["validation_support"]),
                         "recall": float(weak_by_key[(run["run"], label)]["recall"]),
                         "f1": float(weak_by_key[(run["run"], label)]["f1"])}
            for run in runs
        }
    repeat_runtime = float(runs[1]["runtime_seconds"])
    main_runtime = sum(float(row["runtime_seconds"]) for row in _read_rows(output_dir / "comparison_validation.csv"))
    repeat_summary = {
        "configuration": {"learning_rate_init": 0.002, "max_epochs": 12, "patience": 3,
                          "seeds": [runs[0]["seed"], runs[1]["seed"]]},
        "validation_macro_f1_mean_two_seeds": (score_a + score_b) / 2,
        "validation_macro_f1_min": min(score_a, score_b),
        "validation_macro_f1_max": max(score_a, score_b),
        "absolute_seed_spread": abs(score_a - score_b),
        "each_seed_delta_vs_phase2": [runs[0]["macro_f1_delta_vs_phase2"], runs[1]["macro_f1_delta_vs_phase2"]],
        "focus_class_validation": focus_class_summary,
        "repeat_runtime_seconds": repeat_runtime,
        "all_runs_runtime_seconds": main_runtime + repeat_runtime,
        "test_evaluation_performed": False,
        "primary_checkpoint_selection_unchanged": True,
    }
    _write_json(output_dir / "seed_repeat_summary.json", repeat_summary)
    study_config_path = output_dir / "study_configuration.json"
    study_config = json.loads(study_config_path.read_text(encoding="utf-8"))
    study_config["secondary_seed_repeat"] = {"name": "learning_rate_double_seed_20261027", "learning_rate_init": 0.002,
        "max_epochs": 12, "patience": 3, "seed": 20261027,
        "role": "robustness repeat only; excluded from primary fixed-seed ranking"}
    _write_json(study_config_path, study_config)
    report_path = output_dir / "report.md"
    text = report_path.read_text(encoding="utf-8")
    marker = "\n## Second-seed robustness check\n"
    if marker in text:
        text = text.split(marker, 1)[0]
    appendix = ["## Second-seed robustness check", "",
        "Because the fixed-seed 0.002 learning-rate run exceeded the Phase 2 validation macro-F1, that configuration alone was repeated with seed 20261027. This repeat is a robustness check and does not alter the primary four-run ranking or the frozen primary checkpoint.", "",
        "| Seed | Epochs completed | Selected epoch | Accuracy | Macro-P | Macro-R | Macro-F1 | Weighted-F1 | Δ Macro-F1 vs Phase 2 | Runtime (s) |", "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in runs:
        appendix.append(f"| {row['seed']} | {row['epochs_completed']} | {row['selected_epoch']} | {float(row['accuracy']):.4f} | {float(row['macro_precision']):.4f} | {float(row['macro_recall']):.4f} | {float(row['macro_f1']):.4f} | {float(row['weighted_f1']):.4f} | {float(row['macro_f1_delta_vs_phase2']):+.4f} | {float(row['runtime_seconds']):.1f} |")
    appendix += ["", f"Across these two seeds, validation macro-F1 mean was {(score_a + score_b) / 2:.6f}, range {min(score_a, score_b):.6f}–{max(score_a, score_b):.6f}, and absolute spread {abs(score_a - score_b):.6f}. Both runs are above the reproduced Phase 2 validation macro-F1, but the observed spread is material and this is only a two-seed check. Total recorded training-plus-validation time including the repeat was {main_runtime + repeat_runtime:.1f} seconds.", "",
        "The rare-class results also vary by seed (notably Uploading_Attack); this does not establish stable rare-attack detection. Full metrics and support for all Stage A weak classes are saved in `seed_repeat_weak_class_validation.csv`.", "",
        "| Class | Val support | Seed 20261026 recall/F1 | Seed 20261027 recall/F1 |", "|---|---:|---:|---:|"]
    for label in ("Uploading_Attack", "XSS", "Backdoor_Malware", "SqlInjection", "BrowserHijacking", "Recon-PingSweep"):
        a, b = focus_class_summary[label]["seed_20261026"], focus_class_summary[label]["seed_20261027"]
        appendix.append(f"| {label} | {a['validation_support']} | {a['recall']:.4f}/{a['f1']:.4f} | {b['recall']:.4f}/{b['f1']:.4f} |")
    appendix += ["", "No test data was used in this repeat.", ""]
    report_path.write_text(text.rstrip() + "\n\n" + "\n".join(appendix), encoding="utf-8")
    selection_path = output_dir / "selection_recommendation.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    selection["learning_rate_0p002_seed_repeat"] = repeat_summary
    _write_json(selection_path, selection)
    return repeat_summary

def run_study(prepared_dir=ROOT / "prepared", output_dir=DEFAULT_OUTPUT, seed=DEFAULT_SEED):
    output_dir = Path(output_dir).resolve()
    if output_dir.exists():
        if any(output_dir.iterdir()):
            raise FileExistsError(f"Refusing to overwrite non-empty stage output: {output_dir}")
    else:
        output_dir.mkdir(parents=True)
    phase2_meta, _, _ = _phase2_context()
    configs = build_candidate_configs(phase2_meta["configuration"], seed)
    _write_json(output_dir / "study_configuration.json", {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "base_phase2_configuration": phase2_meta["configuration"],
        "candidate_configs": configs,
        "selection_metric": "validation macro-F1",
        "test_data_access": "prohibited in this stage",
        "gradient_update_data": "training split only",
    })
    results = []
    for config in configs:
        results.append(run_candidate(config, prepared_dir=prepared_dir, output_dir=output_dir))
        # Save incremental status after each completed candidate for recoverability.
        _write_csv(output_dir / "completed_candidates.csv", results, list(results[0]))
    ranked = finalize_study(output_dir)
    return ranked


def main():
    parser = argparse.ArgumentParser(description="Run the Stage 2 centralized training-parameter study.")
    parser.add_argument("--prepared", type=Path, default=ROOT / "prepared")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()
    rows = run_study(args.prepared, args.output, args.seed)
    print(json.dumps(rows, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())











