from __future__ import annotations

import argparse
import csv
import ctypes
import json
import os
import platform
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import sklearn
from sklearn.metrics import confusion_matrix
from sklearn.neural_network import MLPClassifier
from threadpoolctl import threadpool_limits


DEFAULT_PREPARED = Path(__file__).resolve().parents[2] / "experiments" / "full_dataset" / "prepared"
DEFAULT_OUTPUT = Path(__file__).resolve().parents[2] / "experiments" / "full_dataset" / "centralized"
DEFAULT_SEED = 20261026
BATCH_SIZE = 8192
HIDDEN_LAYERS = (128, 64)
MAX_EPOCHS = 12
PATIENCE = 3
CPU_THREADS = 4


def validate_label_mapping(mapping: dict[str, int]) -> list[str]:
    if not isinstance(mapping, dict) or not mapping:
        raise ValueError("class_to_id must be a non-empty mapping")
    values = list(mapping.values())
    if any(not isinstance(value, int) for value in values):
        raise ValueError("class IDs must be integers")
    if sorted(values) != list(range(len(values))):
        raise ValueError("class IDs must be contiguous from zero")
    return [label for label, _ in sorted(mapping.items(), key=lambda item: item[1])]


def metrics_from_confusion(cm: np.ndarray, class_names: list[str]) -> dict[str, Any]:
    matrix = np.asarray(cm, dtype=np.int64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.shape[0] != len(class_names):
        raise ValueError("confusion matrix must be square and align with class_names")
    if (matrix < 0).any():
        raise ValueError("confusion matrix counts cannot be negative")
    total = int(matrix.sum())
    if total == 0:
        raise ValueError("cannot calculate metrics from an empty confusion matrix")
    tp = np.diag(matrix).astype(np.float64)
    support = matrix.sum(axis=1).astype(np.int64)
    predicted = matrix.sum(axis=0).astype(np.int64)
    precision = np.divide(tp, predicted, out=np.zeros_like(tp), where=predicted > 0)
    recall = np.divide(tp, support, out=np.zeros_like(tp), where=support > 0)
    f1 = np.divide(2 * precision * recall, precision + recall,
                   out=np.zeros_like(tp), where=(precision + recall) > 0)
    supported = support > 0
    class_rows = []
    for i, name in enumerate(class_names):
        if support[i] == 0:
            class_rows.append({
                "label": name, "precision": None, "recall": None, "f1": None, "support": 0,
                "note": "No true examples in this evaluation split; class score is undefined.",
            })
        else:
            class_rows.append({
                "label": name, "precision": float(precision[i]), "recall": float(recall[i]),
                "f1": float(f1[i]), "support": int(support[i]), "note": "",
            })
    return {
        "accuracy": float(tp.sum() / total),
        "macro_precision": float(precision[supported].mean()),
        "macro_recall": float(recall[supported].mean()),
        "macro_f1": float(f1[supported].mean()),
        "weighted_f1": float(np.average(f1[supported], weights=support[supported])),
        "evaluated_rows": total,
        "supported_classes": int(supported.sum()),
        "per_class": class_rows,
        "confusion_matrix": matrix.tolist(),
    }


def load_model_artifact(checkpoint_path: Path, class_to_id: dict[str, int]) -> MLPClassifier:
    class_names = validate_label_mapping(class_to_id)
    model = joblib.load(checkpoint_path)
    if not isinstance(model, MLPClassifier):
        raise TypeError(f"Expected sklearn MLPClassifier checkpoint, got {type(model).__name__}")
    expected = np.arange(len(class_names), dtype=np.int64)
    if not np.array_equal(np.asarray(model.classes_), expected):
        raise ValueError("Checkpoint class IDs do not match the saved label mapping")
    return model


def _count_labels(y: np.ndarray, n_classes: int, chunk_size: int = 250_000) -> np.ndarray:
    counts = np.zeros(n_classes, dtype=np.int64)
    for start in range(0, len(y), chunk_size):
        batch = np.asarray(y[start:min(start + chunk_size, len(y))], dtype=np.int64)
        if batch.size and (batch.min() < 0 or batch.max() >= n_classes):
            raise ValueError("Label array contains IDs outside the training label mapping")
        counts += np.bincount(batch, minlength=n_classes)
    return counts


def inspect_prepared(prepared_dir: Path) -> dict:
    metadata_path = prepared_dir / "phase1_metadata.json"
    preprocessing_path = prepared_dir / "preprocessing_metadata.json"
    if not metadata_path.is_file() or not preprocessing_path.is_file():
        raise FileNotFoundError("Phase 1 metadata or preprocessing metadata is missing")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    preprocessing = json.loads(preprocessing_path.read_text(encoding="utf-8"))
    fitted_path = prepared_dir / "arrays" / "preprocessing.joblib"
    if not fitted_path.is_file():
        raise FileNotFoundError("Saved Phase 1 fitted preprocessing object is missing")
    fitted = joblib.load(fitted_path)
    if fitted.get("class_to_id") != preprocessing.get("class_to_id"):
        raise ValueError("Saved fitted object and preprocessing JSON have different label mappings")
    if fitted.get("feature_columns_retained") != preprocessing.get("feature_columns_retained"):
        raise ValueError("Saved fitted object and preprocessing JSON have different retained features")
    class_to_id = preprocessing["class_to_id"]
    class_names = validate_label_mapping(class_to_id)
    if class_to_id != metadata["class_to_id"]:
        raise ValueError("Phase 1 class mappings disagree")
    feature_names = preprocessing["feature_columns_retained"]
    if not feature_names:
        raise ValueError("No retained feature names in preprocessing metadata")

    arrays = {}
    split_counts = {}
    for split in ("train", "validation", "test"):
        x_path = prepared_dir / "arrays" / f"X_{split}.npy"
        y_path = prepared_dir / "arrays" / f"y_{split}.npy"
        if not x_path.is_file() or not y_path.is_file():
            raise FileNotFoundError(f"Prepared {split} arrays are incomplete")
        x = np.load(x_path, mmap_mode="r")
        y = np.load(y_path, mmap_mode="r")
        if x.dtype != np.float32 or y.dtype != np.int32:
            raise TypeError(f"{split} arrays must be float32 features and int32 labels")
        if x.ndim != 2 or x.shape[1] != len(feature_names) or y.shape != (x.shape[0],):
            raise ValueError(f"{split} array shapes do not match preprocessing metadata")
        if x.shape[0] != int(metadata["derived_rows"][split]):
            raise ValueError(f"{split} row count differs from Phase 1 metadata")
        arrays[f"X_{split}"] = x
        arrays[f"y_{split}"] = y

    # Verify training and validation labels and values before fitting. Test values are not scanned here.
    for split in ("train", "validation"):
        x, y = arrays[f"X_{split}"], arrays[f"y_{split}"]
        observed_counts = _count_labels(y, len(class_names))
        expected = np.asarray([
            metadata["derived_class_counts"][split].get(name, 0) for name in class_names
        ], dtype=np.int64)
        if not np.array_equal(observed_counts, expected):
            raise ValueError(f"{split} label support differs from Phase 1 class counts")
        for start in range(0, x.shape[0], 250_000):
            if not np.isfinite(x[start:min(start + 250_000, x.shape[0])]).all():
                raise ValueError(f"{split} feature array contains non-finite values")
        split_counts[split] = observed_counts.tolist()
    # Test class counts come from the Phase 1 audit; test arrays are only shape/dtype inspected here.
    split_counts["test_from_phase1_metadata"] = [
        int(metadata["derived_class_counts"]["test"].get(name, 0)) for name in class_names
    ]
    return {
        "metadata": metadata,
        "preprocessing": preprocessing,
        "class_to_id": class_to_id,
        "class_names": class_names,
        "arrays": arrays,
        "class_counts": split_counts,
    }


def class_weights_from_counts(counts: np.ndarray) -> np.ndarray:
    counts = np.asarray(counts, dtype=np.float64)
    if counts.ndim != 1 or not counts.size or (counts <= 0).any():
        raise ValueError("Every training class must have positive support")
    weights = np.sqrt(counts.sum() / (len(counts) * counts))
    weights /= np.average(weights, weights=counts)
    return weights.astype(np.float64)


def _confusion_for_model(model: MLPClassifier, x: np.ndarray, y: np.ndarray, n_classes: int) -> np.ndarray:
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    ids = np.arange(n_classes)
    for start in range(0, len(y), BATCH_SIZE):
        end = min(start + BATCH_SIZE, len(y))
        xb = np.asarray(x[start:end], dtype=np.float32)
        yb = np.asarray(y[start:end], dtype=np.int64)
        pred = model.predict(xb).astype(np.int64, copy=False)
        cm += confusion_matrix(yb, pred, labels=ids)
    return cm


def _memory_snapshot() -> dict[str, int | None]:
    if os.name == "nt":
        try:
            class Counters(ctypes.Structure):
                _fields_ = [
                    ("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong),
                    ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t),
                    ("PrivateUsage", ctypes.c_size_t),
                ]
            counters = Counters()
            counters.cb = ctypes.sizeof(Counters)
            psapi = ctypes.WinDLL("psapi", use_last_error=True)
            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            get_info = psapi.GetProcessMemoryInfo
            get_info.argtypes = [ctypes.c_void_p, ctypes.POINTER(Counters), ctypes.c_ulong]
            get_info.restype = ctypes.c_int
            kernel32.GetCurrentProcess.restype = ctypes.c_void_p
            if get_info(kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
                return {
                    "working_set_bytes": int(counters.WorkingSetSize),
                    "peak_working_set_bytes": int(counters.PeakWorkingSetSize),
                    "private_bytes": int(counters.PrivateUsage),
                }
        except Exception:
            pass
    return {"working_set_bytes": None, "peak_working_set_bytes": None, "private_bytes": None}


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _save_confusion_csv(path: Path, cm: np.ndarray, class_names: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["actual_label / predicted_label", *class_names])
        for label, row in zip(class_names, cm):
            writer.writerow([label, *(int(value) for value in row)])


def _fit_epoch(
    model: MLPClassifier, x_train: np.ndarray, y_train: np.ndarray,
    weights: np.ndarray, rng: np.random.Generator, classes: np.ndarray,
    first_update: bool,
) -> tuple[float, bool]:
    order = rng.permutation(len(y_train))
    loss_sum = 0.0
    seen = 0
    for start in range(0, len(order), BATCH_SIZE):
        indices = order[start:start + BATCH_SIZE]
        xb = np.asarray(x_train[indices], dtype=np.float32)
        yb = np.asarray(y_train[indices], dtype=np.int32)
        sample_weights = weights[yb]
        kwargs = {"classes": classes} if first_update else {}
        configured_batch_size = model.batch_size
        if len(indices) < configured_batch_size:
            model.batch_size = len(indices)
        try:
            model.partial_fit(xb, yb, sample_weight=sample_weights, **kwargs)
        finally:
            model.batch_size = configured_batch_size
        first_update = False
        loss = float(model.loss_)
        if np.isfinite(loss):
            loss_sum += loss * len(indices)
        seen += len(indices)
    return loss_sum / max(seen, 1), first_update


def run_centralized(
    prepared_dir: Path = DEFAULT_PREPARED,
    output_dir: Path = DEFAULT_OUTPUT,
    seed: int = DEFAULT_SEED,
    max_epochs: int = MAX_EPOCHS,
    patience: int = PATIENCE,
) -> dict:
    prepared_dir = Path(prepared_dir).resolve()
    output_dir = Path(output_dir).resolve()
    build_dir = output_dir.with_name(f".{output_dir.name}_building")
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing experiment output: {output_dir}")
    if build_dir.exists():
        raise FileExistsError(f"Refusing to overwrite incomplete build directory: {build_dir}")
    if max_epochs < 1 or patience < 1:
        raise ValueError("max_epochs and patience must be positive")

    inspected = inspect_prepared(prepared_dir)
    arrays = inspected["arrays"]
    x_train, y_train = arrays["X_train"], arrays["y_train"]
    x_val, y_val = arrays["X_validation"], arrays["y_validation"]
    classes = np.arange(len(inspected["class_names"]), dtype=np.int64)
    counts = np.asarray(inspected["class_counts"]["train"], dtype=np.int64)
    class_weights = class_weights_from_counts(counts)

    model_config = {
        "model": "sklearn.neural_network.MLPClassifier",
        "hidden_layer_sizes": list(HIDDEN_LAYERS),
        "activation": "relu",
        "solver": "adam",
        "alpha": 1e-4,
        "batch_size": BATCH_SIZE,
        "learning_rate_init": 1e-3,
        "max_epochs": max_epochs,
        "early_stopping_patience": patience,
        "selection_metric": "validation macro-F1",
        "class_weighting": "normalized inverse square-root of training class frequency",
        "seed": seed,
        "cpu_threads": CPU_THREADS,
    }
    param_count = (
        x_train.shape[1] * HIDDEN_LAYERS[0] + HIDDEN_LAYERS[0]
        + HIDDEN_LAYERS[0] * HIDDEN_LAYERS[1] + HIDDEN_LAYERS[1]
        + HIDDEN_LAYERS[1] * len(classes) + len(classes)
    )
    disk = shutil.disk_usage(output_dir.parent)
    estimated = {
        "model_parameter_count": int(param_count),
        "float64_parameter_bytes": int(param_count * 8),
        "train_feature_array_bytes_mapped": int((prepared_dir / "arrays" / "X_train.npy").stat().st_size),
        "validation_feature_array_bytes_mapped": int((prepared_dir / "arrays" / "X_validation.npy").stat().st_size),
        "batch_input_bytes_float32": int(BATCH_SIZE * x_train.shape[1] * 4),
        "per_epoch_shuffle_index_bytes": int(len(y_train) * 8),
        "free_disk_bytes_before_run": int(disk.free),
        "disk_space_needed_for_checkpoint_and_results": "under 100 MB; prediction vector is about 4.5 MiB",
    }

    build_dir.mkdir(parents=True, exist_ok=False)
    checkpoints = build_dir / "checkpoints"
    checkpoints.mkdir()
    log_path = build_dir / "training.log"
    history_rows = []
    validation_rows = []
    resource_samples = []
    rng = np.random.default_rng(seed)
    model = MLPClassifier(
        hidden_layer_sizes=HIDDEN_LAYERS,
        activation="relu",
        solver="adam",
        alpha=1e-4,
        batch_size=BATCH_SIZE,
        learning_rate_init=1e-3,
        max_iter=1,
        shuffle=False,
        random_state=seed,
        tol=0.0,
    )
    started_all = time.perf_counter()
    first_update = True
    best_macro = -1.0
    best_epoch = 0
    epochs_without_improvement = 0
    log_lines = []

    with threadpool_limits(limits=CPU_THREADS):
        for epoch in range(1, max_epochs + 1):
            epoch_start = time.perf_counter()
            train_loss, first_update = _fit_epoch(
                model, x_train, y_train, class_weights, rng, classes, first_update
            )
            train_seconds = time.perf_counter() - epoch_start
            val_start = time.perf_counter()
            val_cm = _confusion_for_model(model, x_val, y_val, len(classes))
            val_metrics = metrics_from_confusion(val_cm, inspected["class_names"])
            validation_seconds = time.perf_counter() - val_start
            epoch_seconds = train_seconds + validation_seconds
            row = {
                "epoch": epoch,
                "train_weighted_loss_mean": train_loss,
                "validation_accuracy": val_metrics["accuracy"],
                "validation_macro_precision": val_metrics["macro_precision"],
                "validation_macro_recall": val_metrics["macro_recall"],
                "validation_macro_f1": val_metrics["macro_f1"],
                "validation_weighted_f1": val_metrics["weighted_f1"],
                "train_seconds": train_seconds,
                "validation_seconds": validation_seconds,
                "epoch_seconds": epoch_seconds,
            }
            history_rows.append(row)
            validation_rows.append({
                "epoch": epoch,
                **{k: val_metrics[k] for k in ("accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1")},
            })
            memory = _memory_snapshot()
            memory["epoch"] = epoch
            resource_samples.append(memory)

            improved = val_metrics["macro_f1"] > best_macro + 1e-8
            if improved:
                best_macro = val_metrics["macro_f1"]
                best_epoch = epoch
                epochs_without_improvement = 0
                joblib.dump(model, checkpoints / "best_model.joblib", compress=3)
                _write_json(build_dir / "best_validation_metrics.json", {
                    "selected_epoch": best_epoch,
                    "selection_metric": "validation macro-F1",
                    **{k: val_metrics[k] for k in ("accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1")},
                })
            else:
                epochs_without_improvement += 1
            line = (
                f"epoch={epoch} train_loss={train_loss:.6f} val_accuracy={val_metrics['accuracy']:.6f} "
                f"val_macro_f1={val_metrics['macro_f1']:.6f} val_weighted_f1={val_metrics['weighted_f1']:.6f} "
                f"train_sec={train_seconds:.2f} val_sec={validation_seconds:.2f} best={improved}"
            )
            print(line, flush=True)
            log_lines.append(line)
            _write_csv(build_dir / "training_history.csv", history_rows, list(history_rows[0].keys()))
            _write_csv(build_dir / "validation_metrics_by_epoch.csv", validation_rows, list(validation_rows[0].keys()))
            if epochs_without_improvement >= patience:
                print(f"Early stopping after {epoch} epochs; best epoch {best_epoch}.", flush=True)
                break

    fit_seconds = time.perf_counter() - started_all
    if best_epoch == 0:
        raise RuntimeError("No valid validation checkpoint was selected")

    best_model = load_model_artifact(checkpoints / "best_model.joblib", inspected["class_to_id"])
    selected_validation_start = time.perf_counter()
    best_val_cm = _confusion_for_model(best_model, x_val, y_val, len(classes))
    best_val_metrics = metrics_from_confusion(best_val_cm, inspected["class_names"])
    _write_json(build_dir / "best_validation_metrics.json", {
        "selected_epoch": best_epoch,
        "selection_metric": "validation macro-F1",
        **best_val_metrics,
    })
    _write_csv(build_dir / "per_class_validation_metrics.csv", best_val_metrics["per_class"],
               ["label", "precision", "recall", "f1", "support", "note"])
    _save_confusion_csv(build_dir / "confusion_matrix_validation.csv", best_val_cm, inspected["class_names"])
    np.save(build_dir / "confusion_matrix_validation.npy", best_val_cm)
    selected_validation_recheck_seconds = time.perf_counter() - selected_validation_start
    # Test values are read only after checkpoint selection; earlier inspection only mapped array headers, shapes, and dtypes.
    x_test = arrays["X_test"]
    y_test = arrays["y_test"]
    n_test = len(y_test)
    predictions_path = build_dir / "test_predictions.npy"
    predictions = np.lib.format.open_memmap(
        predictions_path, mode="w+", dtype=np.int32, shape=(n_test,)
    )
    test_cm = np.zeros((len(classes), len(classes)), dtype=np.int64)
    test_evaluation_start = time.perf_counter()
    for start in range(0, n_test, BATCH_SIZE):
        end = min(start + BATCH_SIZE, n_test)
        xb = np.asarray(x_test[start:end], dtype=np.float32)
        pred = best_model.predict(xb).astype(np.int32, copy=False)
        true = np.asarray(y_test[start:end], dtype=np.int64)
        predictions[start:end] = pred
        test_cm += confusion_matrix(true, pred, labels=classes)
    predictions.flush()
    del predictions
    test_metrics = metrics_from_confusion(test_cm, inspected["class_names"])
    test_evaluation_seconds = time.perf_counter() - test_evaluation_start
    _write_json(build_dir / "test_metrics.json", {
        "split": "test",
        "use": "final evaluation after validation-based checkpoint selection; no tuning",
        **{k: v for k, v in test_metrics.items() if k not in ("confusion_matrix", "per_class")},
    })
    _write_csv(
        build_dir / "per_class_test_metrics.csv",
        test_metrics["per_class"],
        ["label", "precision", "recall", "f1", "support", "note"],
    )
    _save_confusion_csv(build_dir / "confusion_matrix_test.csv", test_cm, inspected["class_names"])
    np.save(build_dir / "confusion_matrix_test.npy", test_cm)

    elapsed = time.perf_counter() - started_all
    versions = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scikit_learn": sklearn.__version__,
        "joblib": joblib.__version__,
        "torch": None,
    }
    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "configuration": model_config,
        "software_versions": versions,
        "platform": platform.platform(),
        "cpu": "Intel Core i5-13420H (user-provided; CPU training)",
        "ram_gib_user_specified": 16,
        "phase1_prepared_directory": str(prepared_dir),
        "phase1_split_rows": {s: int(arrays[f"y_{s}"].shape[0]) for s in ("train", "validation", "test")},
        "feature_count": int(x_train.shape[1]),
        "class_to_id": inspected["class_to_id"],
        "class_counts": {
            "train": {name: int(value) for name, value in zip(inspected["class_names"], inspected["class_counts"]["train"])},
            "validation": {name: int(value) for name, value in zip(inspected["class_names"], inspected["class_counts"]["validation"])},
            "test_from_phase1_metadata": {
                name: int(value) for name, value in zip(inspected["class_names"], inspected["class_counts"]["test_from_phase1_metadata"])
            },
        },
        "class_weights": {
            name: float(value) for name, value in zip(inspected["class_names"], class_weights)
        },
        "validation_selection": {
            "metric": "macro-F1",
            "selected_epoch": best_epoch,
            "best_score": best_macro,
            "patience": patience,
            "test_used_for_selection": False,
        },
        "timing_seconds": {
            "fit_and_epoch_validation": fit_seconds,
            "selected_validation_recheck": selected_validation_recheck_seconds,
            "test_prediction_and_metrics": test_evaluation_seconds,
            "total": elapsed,
        },
        "runtime_notes": [
            "The training run used the configured full 8,192-row minibatches and processed the final smaller minibatch each epoch; scikit-learn clipped its batch size without dropping rows.",
            "The actual run emitted an informational scikit-learn clipping warning for that final smaller minibatch; this code now sets the final call batch_size explicitly to avoid the warning with equivalent processing.",
        ],
        "resource_estimates": estimated,
        "resource_samples": resource_samples,
        "resource_measurement_note": "Windows process working-set/private bytes sampled after validation epochs when the OS API was available; None means unavailable.",
        "test_evaluation": test_metrics,
    }
    _write_json(build_dir / "run_metadata.json", metadata)
    (build_dir / "training.log").write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    report = render_report(metadata)
    (build_dir / "report.md").write_text(report, encoding="utf-8")
    os.replace(build_dir, output_dir)
    print(f"Phase 2 artifacts saved under {output_dir}", flush=True)
    return metadata


def render_report(metadata: dict) -> str:
    tm = metadata["test_evaluation"]
    val = metadata["validation_selection"]
    per_class = tm["per_class"]
    difficult = sorted(
        [row for row in per_class if row["support"] > 0],
        key=lambda row: (row["f1"] if row["f1"] is not None else -1, row["support"]),
    )[:8]
    difficult_table = ["| Class | Precision | Recall | F1 | Support |", "| --- | ---: | ---: | ---: | ---: |"]
    for row in difficult:
        difficult_table.append(
            f"| {row['label']} | {row['precision']:.4f} | {row['recall']:.4f} | {row['f1']:.4f} | {row['support']:,} |"
        )
    timing = metadata["timing_seconds"]
    validation_recheck = timing.get("selected_validation_recheck")
    test_eval = timing.get("test_prediction_and_metrics")
    post_selection = timing.get("post_selection_artifact_prep")
    validation_recheck_text = f"{validation_recheck:.1f} seconds" if validation_recheck is not None else "not separately timed"
    test_eval_text = f"{test_eval:.1f} seconds" if test_eval is not None else "not separately timed"
    post_selection_text = f"{post_selection:.1f} seconds" if post_selection is not None else "not separately isolated"
    metric_table = [
        "| Metric | Test result |",
        "| --- | ---: |",
        f"| Accuracy | {tm['accuracy']:.4f} |",
        f"| Macro precision | {tm['macro_precision']:.4f} |",
        f"| Macro recall | {tm['macro_recall']:.4f} |",
        f"| Macro-F1 | {tm['macro_f1']:.4f} |",
        f"| Weighted-F1 | {tm['weighted_f1']:.4f} |",
    ]
    return f"""# Phase 2 — Centralized baseline report

**Model:** CPU scikit-learn MLPClassifier, hidden layers {metadata['configuration']['hidden_layer_sizes']}.  
**Created:** {metadata['created_utc']}  
**Training rows:** {metadata['phase1_split_rows']['train']:,}; validation rows: {metadata['phase1_split_rows']['validation']:,}; test rows: {metadata['phase1_split_rows']['test']:,}.  
**Seed:** {metadata['seed']}

## Method

The classifier uses ReLU hidden layers and Adam with 8,192-row minibatches. It consumes the Phase 1 float32 memmaps without materializing the complete feature matrix in RAM. Class imbalance is handled with normalized inverse-square-root training-frequency sample weights. This moderates the majority-class dominance without duplicating rare records; it does not make the rare classes adequately represented, and the weighting can affect calibration and false positives.

The best checkpoint is selected by validation macro-F1. Training stops after {metadata['configuration']['early_stopping_patience']} epochs without improvement, up to {metadata['configuration']['max_epochs']} epochs. The test split was not used for training, checkpoint selection, or tuning.

## Validation checkpoint selection

Selected epoch: **{val['selected_epoch']}**; validation macro-F1: **{val['best_score']:.4f}**. Test used for selection: **No**.

## Final test evaluation

{chr(10).join(metric_table)}

All 34 classes were present in the final test split according to the Phase 1 support audit. The per-class table and confusion matrix are saved as separate artifacts. Do not interpret the high-support weighted score as uniform performance across classes.

### Lowest per-class F1 among supported test classes

{chr(10).join(difficult_table)}

## Runtime and resources

- Training plus per-epoch validation: {timing['fit_and_epoch_validation']:.1f} seconds.
- Selected-checkpoint validation recheck: {validation_recheck_text}.
- Final test prediction and metric calculation: {test_eval_text}.
- Post-selection validation, test, and artifact finalization: {post_selection_text}.
- Total elapsed through evaluation artifacts: {timing['total']:.1f} seconds.
- Peak measured working set: {max((row['peak_working_set_bytes'] or 0) for row in metadata['resource_samples']) / (1024 ** 3):.2f} GiB where available.
- Scikit-learn warned on the smaller final minibatch and clipped its configured size; the minibatch was processed and no rows were skipped. The preparation code now sets that final size explicitly to suppress the warning with equivalent processing.
- Parameter count: {metadata['resource_estimates']['model_parameter_count']:,}; estimated float64 parameter storage: {metadata['resource_estimates']['float64_parameter_bytes']:,} bytes.
- Training feature memmap: {metadata['resource_estimates']['train_feature_array_bytes_mapped']:,} bytes; each minibatch input: {metadata['resource_estimates']['batch_input_bytes_float32']:,} bytes.
- Peak working-set samples are recorded in run_metadata.json when available.

## Limitations

1. The class distribution is highly imbalanced. The rarest training class has only 138 examples, so its metrics are uncertain. Weighting cannot replace missing diversity.
2. The split policy groups exact serialized feature-token fingerprints; it is not device-, session-, or time-aware.
3. The baseline is a single-machine CPU experiment, not a federated or privacy-preserving deployment.
4. This run evaluates the supplied Phase 1 derived split, not the original sampled experiment.

## Artifacts

See checkpoints/best_model.joblib, training_history.csv, validation_metrics_by_epoch.csv, best_validation_metrics.json, per_class_validation_metrics.csv, confusion_matrix_validation.csv, test_metrics.json, per_class_test_metrics.csv, confusion_matrix_test.csv, confusion_matrix_test.npy, test_predictions.npy, run_metadata.json, and training.log.
"""


def main() -> int:
    parser = argparse.ArgumentParser(description="Train and evaluate the full-data centralized MLP baseline.")
    parser.add_argument("--prepared", type=Path, default=DEFAULT_PREPARED)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--epochs", type=int, default=MAX_EPOCHS)
    parser.add_argument("--patience", type=int, default=PATIENCE)
    args = parser.parse_args()
    result = run_centralized(args.prepared, args.output, args.seed, args.epochs, args.patience)
    print(json.dumps({
        "selected_epoch": result["validation_selection"]["selected_epoch"],
        "test_metrics": {key: result["test_evaluation"][key] for key in
                         ("accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1")},
        "timing_seconds": result["timing_seconds"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())





