"""Phase 3: CPU-friendly, single-process FedAvg simulation."""
from __future__ import annotations
import argparse, copy, csv, json, os, platform, shutil, time
from datetime import datetime, timezone
from pathlib import Path
import joblib, numpy as np, sklearn
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils import check_random_state
from threadpoolctl import threadpool_limits
from src.full_dataset.centralized import (
    BATCH_SIZE, CPU_THREADS, DEFAULT_PREPARED, DEFAULT_SEED, HIDDEN_LAYERS,
    _confusion_for_model, _memory_snapshot, _save_confusion_csv, _write_csv,
    _write_json, class_weights_from_counts, inspect_prepared, load_model_artifact,
    metrics_from_confusion,
)

DEFAULT_OUTPUT = Path(__file__).resolve().parents[2] / "experiments" / "full_dataset" / "federated" / "experiment"
DIRICHLET_ALPHA = 1.0
PARTITION_SEED = 20261026
MAX_ROUNDS = 12
LOCAL_EPOCHS = 1
PATIENCE = 3
PILOT_ROUND_LIMIT_SECONDS = 300.0
TOTAL_RUNTIME_LIMIT_SECONDS = 2700.0
WORKING_SET_LIMIT_BYTES = 8 * 1024**3


def partition_clients(y, n_clients=3, alpha=DIRICHLET_ALPHA, seed=PARTITION_SEED):
    """Reproducible class-wise Dirichlet split; each source row is assigned once."""
    labels = np.asarray(y)
    if labels.ndim != 1 or labels.size == 0 or not np.issubdtype(labels.dtype, np.integer) or labels.min() < 0:
        raise ValueError("y must be a nonempty vector of nonnegative integer class IDs")
    if n_clients < 2 or alpha <= 0:
        raise ValueError("n_clients must be >= 2 and alpha positive")
    rng = np.random.default_rng(seed)
    pieces = [[] for _ in range(n_clients)]
    for class_id in np.unique(labels):
        rows = np.flatnonzero(labels == class_id)
        rng.shuffle(rows)
        counts = rng.multinomial(len(rows), rng.dirichlet(np.full(n_clients, alpha)))
        start = 0
        for client_id, count in enumerate(counts):
            if count:
                pieces[client_id].append(rows[start:start + count])
            start += int(count)
        if start != len(rows):
            raise AssertionError("partition failed to assign all rows")
    clients = []
    for client_pieces in pieces:
        if not client_pieces:
            raise ValueError("empty client partition")
        indices = np.concatenate(client_pieces).astype(np.int32, copy=False)
        rng.shuffle(indices)
        clients.append(indices)
    return clients


def client_distribution(y, client_indices, class_names):
    y = np.asarray(y)
    rows = []
    for cid, indices in enumerate(client_indices, 1):
        counts = np.bincount(y[indices].astype(np.int64), minlength=len(class_names))
        rows.append({"client_id": cid, "sample_count": int(len(indices)),
                     "represented_classes": int(np.count_nonzero(counts)),
                     "class_counts": {name: int(counts[i]) for i, name in enumerate(class_names)}})
    return rows


def initialize_global_model(n_features, n_classes, seed=DEFAULT_SEED):
    """Fresh Phase-2-shaped sklearn MLP; sklearn has no public untrained init API."""
    if n_features < 1 or n_classes < 2:
        raise ValueError("invalid model dimensions")
    model = MLPClassifier(hidden_layer_sizes=HIDDEN_LAYERS, activation="relu", solver="adam",
                          alpha=1e-4, batch_size=BATCH_SIZE, learning_rate_init=1e-3,
                          max_iter=1, shuffle=False, random_state=seed, tol=0.0)
    # Version-sensitive private sklearn initialization; this call is covered by tests.
    model._random_state = check_random_state(seed)
    model._label_binarizer = LabelBinarizer()
    model._label_binarizer.fit(np.arange(n_classes, dtype=np.int64))
    model.classes_ = np.arange(n_classes, dtype=np.int64)
    model._initialize(np.zeros((1, n_classes), dtype=np.float32),
                      [n_features, *HIDDEN_LAYERS, n_classes], np.float32)
    model.n_features_in_ = n_features
    if hasattr(model, "_optimizer"):
        del model._optimizer
    return model


def model_signature(model):
    return (tuple(tuple(a.shape) for a in model.coefs_),
            tuple(tuple(a.shape) for a in model.intercepts_), str(model.coefs_[0].dtype))


def fedavg_aggregate(models, sample_counts):
    """Average compatible MLP parameters by client row count (optimizer excluded)."""
    if not models or len(models) != len(sample_counts):
        raise ValueError("models and sample_counts must have equal nonzero lengths")
    if any(not isinstance(n, (int, np.integer)) or int(n) <= 0 for n in sample_counts):
        raise ValueError("sample counts must be positive integers")
    signature = model_signature(models[0])
    for model in models:
        if model_signature(model) != signature:
            raise ValueError("incompatible client parameter shapes or dtypes")
        if not all(np.isfinite(a).all() for a in [*model.coefs_, *model.intercepts_]):
            raise ValueError("client model has nonfinite parameters")
    weights = np.asarray(sample_counts, dtype=np.float64)
    weights /= weights.sum()
    result = copy.deepcopy(models[0])
    for layer in range(len(result.coefs_)):
        result.coefs_[layer] = sum((np.asarray(m.coefs_[layer], dtype=np.float64) * w
                                    for m, w in zip(models, weights)),
                                   np.zeros_like(result.coefs_[layer], dtype=np.float64)).astype(result.coefs_[layer].dtype)
        result.intercepts_[layer] = sum((np.asarray(m.intercepts_[layer], dtype=np.float64) * w
                                         for m, w in zip(models, weights)),
                                        np.zeros_like(result.intercepts_[layer], dtype=np.float64)).astype(result.intercepts_[layer].dtype)
    if hasattr(result, "_optimizer"):
        del result._optimizer
    if model_signature(result) != signature:
        raise AssertionError("aggregation changed the model signature")
    return result


def _fit_local_epoch(model, x, y, indices, class_weights, classes, rng):
    order = indices.copy()
    rng.shuffle(order)
    loss_sum = seen = 0
    first_batch = True
    started = time.perf_counter()
    for offset in range(0, len(order), BATCH_SIZE):
        ids = order[offset:offset + BATCH_SIZE]
        xb = np.asarray(x[ids], dtype=np.float32)
        yb = np.asarray(y[ids], dtype=np.int32)
        old_batch_size = model.batch_size
        if len(ids) < old_batch_size:
            model.batch_size = len(ids)
        try:
            model.partial_fit(xb, yb, classes=classes if first_batch else None,
                              sample_weight=class_weights[yb])
        finally:
            model.batch_size = old_batch_size
        first_batch = False
        if np.isfinite(model.loss_):
            loss_sum += float(model.loss_) * len(ids)
        seen += len(ids)
    return loss_sum / max(seen, 1), time.perf_counter() - started


def _metric_summary(metrics):
    return {key: metrics[key] for key in ("accuracy", "macro_precision", "macro_recall", "macro_f1",
                                          "weighted_f1", "evaluated_rows", "supported_classes")}


def _report(metadata, clients, val_metrics, test_metrics, central, history):
    lines = ["# Phase 3 — Full-dataset FedAvg report", "",
             f"**Status:** {metadata['status']}  ", f"**Created:** {metadata['created_utc']}  ",
             f"**Rows:** {metadata['dataset']['train_rows']:,} train; {metadata['dataset']['validation_rows']:,} validation; {metadata['dataset']['test_rows']:,} test.  ",
             f"**Model:** same [128, 64] sklearn MLP shape as Phase 2, {metadata['configuration']['local_epochs']} local epoch(s), {len(history)} rounds.  ",
             f"**Selection:** round {metadata['selection']['selected_round']} by validation macro-F1; test was not used for selection.", "",
             "## Method and compatibility", "",
             "The Phase 2 checkpoint was verified to have compatible layer shapes, but its learned parameters were not used because they were trained on all training rows. FedAvg starts from one fresh initialization with the same architecture/seed; all clients begin each round from the same global parameters. The server averages only weight and bias arrays using each client’s fraction of training rows. Client Adam state is reset each round and is not averaged.", "",
             "Clients are simulated in one process using disjoint row-index partitions and minibatches from the prepared training memmap. This does not provide machine-level isolation, secure aggregation, or a formal privacy guarantee.", "",
             "## Client distribution", "",
             "| Client | Rows | Classes | Uploading_Attack | XSS | Recon-PingSweep |", "|---|---:|---:|---:|---:|---:|"]
    for c in clients:
        dist = c["class_counts"]
        lines.append(f"| {c['client_id']} | {c['sample_count']:,} | {c['represented_classes']} | {dist.get('Uploading_Attack', 0)} | {dist.get('XSS', 0)} | {dist.get('Recon-PingSweep', 0)} |")
    lines += ["", "Full client class counts are in `client_partitions.json`.", "",
              "## Validation checkpoint", "", "| Metric | Validation |", "|---|---:|"]
    for key in ("accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1"):
        lines.append(f"| {key.replace('_', ' ').title()} | {val_metrics[key]:.6f} |")
    central_val = metadata.get("centralized_validation_metrics")
    if central_val:
        lines += ["", "### Validation comparison (checkpoint selection split)", "",
                  "Both approaches use the same prepared validation rows; these values guide checkpoint selection.", "",
                  "| Metric | FedAvg selected round | Centralized selected epoch | FedAvg - centralized |", "|---|---:|---:|---:|"]
        for key in ("accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1"):
            lines.append(f"| {key.replace('_', ' ').title()} | {val_metrics[key]:.6f} | {central_val[key]:.6f} | {val_metrics[key]-central_val[key]:+.6f} |")
    lines += ["", "## Final test evaluation", ""]
    if test_metrics is None:
        lines.append("Validation-only resource pilot: no test arrays or labels were evaluated.")
    else:
        lines += ["The validation-selected checkpoint was evaluated on the prepared test split after the protocol was fixed.", "",
                  "| Metric | FedAvg | Centralized | Difference |", "|---|---:|---:|---:|"]
        for key in ("accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1"):
            c = central[key]
            lines.append(f"| {key.replace('_', ' ').title()} | {test_metrics[key]:.6f} | {c:.6f} | {test_metrics[key]-c:+.6f} |")
        lines += ["", "All 34 classes occur in the test split by Phase 1 audit. Rare-class scores remain uncertain; Uploading_Attack has 33 test rows and XSS 102. See per-class metrics and confusion matrix artifacts."]
    runtime = metadata["runtime"]
    comm = metadata["communication_estimate"]
    lines += ["", "## Runtime and communication", "",
              f"Training and validation took {runtime['training_and_validation_seconds']:.1f} seconds across {runtime['rounds_completed']} rounds. Stop reason: `{runtime['stop_reason']}`.",
              f"Estimated parameter upload/download traffic: {comm['upload_download_parameter_bytes_estimate']:,} bytes; transport/security overhead is excluded.", "",
              "## Limitations", "",
              "- Class-wise Dirichlet partitioning is a reproducible simulation, not evidence of real organizational data separation.",
              "- The training distribution is highly imbalanced. Global inverse-square-root weights cannot create rare-class diversity.",
              "- Phase 1 grouping uses exact feature-token fingerprints; no authentic device, session, or time IDs exist.",
              "- sklearn’s private `_initialize` API is version-sensitive; the run records versions and tests this initialization.",
              "- One fixed-seed run does not establish that FedAvg outperforms the centralized baseline.", ""]
    return "\n".join(lines)


def run_fedavg(prepared_dir=DEFAULT_PREPARED, output_dir=DEFAULT_OUTPUT, rounds=MAX_ROUNDS,
               local_epochs=LOCAL_EPOCHS, seed=DEFAULT_SEED, partition_seed=PARTITION_SEED,
               alpha=DIRICHLET_ALPHA, patience=PATIENCE, pilot_only=False):
    prepared_dir, output_dir = Path(prepared_dir).resolve(), Path(output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output_dir}")
    if rounds < 1 or local_epochs < 1 or patience < 1:
        raise ValueError("rounds, local_epochs and patience must be positive")
    inspected = inspect_prepared(prepared_dir)
    arrays = inspected["arrays"]
    x_train, y_train = arrays["X_train"], arrays["y_train"]
    x_val, y_val = arrays["X_validation"], arrays["y_validation"]
    names = inspected["class_names"]
    classes = np.arange(len(names), dtype=np.int64)
    counts = np.asarray(inspected["class_counts"]["train"], dtype=np.int64)
    class_weights = class_weights_from_counts(counts)
    client_indices = partition_clients(y_train, 3, alpha, partition_seed)
    clients = client_distribution(y_train, client_indices, names)
    all_indices = np.concatenate(client_indices)
    if (sum(len(i) for i in client_indices) != len(y_train) or
        len(np.unique(all_indices)) != len(y_train) or all_indices.min() != 0 or all_indices.max() != len(y_train)-1):
        raise AssertionError("client partitions are not disjoint and exhaustive")

    global_model = initialize_global_model(x_train.shape[1], len(names), seed)
    central_dir = prepared_dir.parent / "centralized"
    central_checkpoint = central_dir / "checkpoints" / "best_model.joblib"
    central_metrics_path = central_dir / "test_metrics.json"
    central_model = load_model_artifact(central_checkpoint, inspected["class_to_id"])
    if model_signature(global_model) != model_signature(central_model):
        raise ValueError("fresh model is not parameter-compatible with Phase 2")
    del central_model

    effective_rounds = 1 if pilot_only else rounds
    build_dir = output_dir.with_name(f".{output_dir.name}_building")
    if build_dir.exists():
        raise FileExistsError(f"Refusing to overwrite incomplete build directory: {build_dir}")
    build_dir.mkdir(parents=True)
    checkpoints = build_dir / "checkpoints"
    checkpoints.mkdir()
    config = {"model": "sklearn MLPClassifier, Phase 2 architecture", "hidden_layer_sizes": list(HIDDEN_LAYERS),
              "activation": "relu", "solver": "Adam", "batch_size": BATCH_SIZE, "learning_rate_init": 0.001,
              "alpha": 1e-4, "rounds_requested": rounds, "rounds_run_limit": effective_rounds,
              "local_epochs": local_epochs, "client_count": 3, "partition": "class-wise Dirichlet",
              "dirichlet_alpha": alpha, "partition_seed": partition_seed, "seed": seed,
              "aggregation": "sample-count-weighted average of coefs_ and intercepts_",
              "optimizer_state": "fresh local Adam optimizer each client/round; never aggregated",
              "class_weighting": "Phase 2 inverse-square-root weights fitted on training counts",
              "selection_metric": "validation macro-F1", "patience": patience,
              "test_used_for_model_selection": False, "centralized_weights_used_as_init": False,
              "cpu_threads": CPU_THREADS}
    _write_json(build_dir / "configuration.json", config)
    _write_json(build_dir / "client_partitions.json", {"method": config["partition"], "alpha": alpha,
                "seed": partition_seed, "client_distributions": clients})
    _write_csv(build_dir / "client_partition_summary.csv",
               [{k: c[k] for k in ("client_id", "sample_count", "represented_classes")} for c in clients],
               ["client_id", "sample_count", "represented_classes"])

    model_bytes = int(sum(p.nbytes for p in [*global_model.coefs_, *global_model.intercepts_]))
    history, local_log, memory_samples, log_lines = [], [], [], []
    best_macro, best_round, no_improve = -1.0, 0, 0
    rng = np.random.default_rng(seed + 1)
    start_all = time.perf_counter()
    stop_reason = "configured_round_limit"
    with threadpool_limits(limits=CPU_THREADS):
        for round_id in range(1, effective_rounds + 1):
            round_start = time.perf_counter()
            local_models, local_losses, client_sizes = [], [], []
            for cid, indices in enumerate(client_indices, 1):
                local_model = copy.deepcopy(global_model)
                if hasattr(local_model, "_optimizer"):
                    del local_model._optimizer
                losses, client_seconds = [], 0.0
                for local_epoch in range(1, local_epochs + 1):
                    loss, secs = _fit_local_epoch(local_model, x_train, y_train, indices,
                                                  class_weights, classes, rng)
                    losses.append(loss)
                    client_seconds += secs
                    local_log.append({"round": round_id, "client_id": cid, "local_epoch": local_epoch,
                                      "rows": int(len(indices)), "weighted_loss": float(loss),
                                      "training_seconds": float(secs)})
                local_models.append(local_model)
                local_losses.append(float(np.mean(losses)))
                client_sizes.append(len(indices))
                print(f"round={round_id} client={cid} rows={len(indices)} local_seconds={client_seconds:.1f}", flush=True)
            global_model = fedavg_aggregate(local_models, client_sizes)
            del local_models
            val_start = time.perf_counter()
            val_cm = _confusion_for_model(global_model, x_val, y_val, len(names))
            val_metrics = metrics_from_confusion(val_cm, names)
            val_seconds = time.perf_counter() - val_start
            round_seconds = time.perf_counter() - round_start
            improved = val_metrics["macro_f1"] > best_macro + 1e-8
            if improved:
                best_macro, best_round, no_improve = val_metrics["macro_f1"], round_id, 0
                joblib.dump(global_model, checkpoints / "best_model.joblib", compress=3)
                np.save(checkpoints / "best_validation_confusion.npy", val_cm)
                _write_json(build_dir / "best_validation_metrics.json",
                            {"selected_round": best_round, "selection_metric": "validation macro-F1",
                             **_metric_summary(val_metrics)})
            else:
                no_improve += 1
            history.append({"round": round_id, "validation_accuracy": val_metrics["accuracy"],
                            "validation_macro_precision": val_metrics["macro_precision"],
                            "validation_macro_recall": val_metrics["macro_recall"],
                            "validation_macro_f1": val_metrics["macro_f1"],
                            "validation_weighted_f1": val_metrics["weighted_f1"],
                            "mean_client_weighted_loss": float(np.average(local_losses, weights=client_sizes)),
                            "validation_seconds": val_seconds, "round_seconds": round_seconds,
                            "round_communication_bytes_estimate": int(2 * 3 * model_bytes),
                            "best_checkpoint_selected": bool(improved)})
            mem = _memory_snapshot(); mem["round"] = round_id; memory_samples.append(mem)
            _write_csv(build_dir / "round_history.csv", history, list(history[0]))
            _write_csv(build_dir / "local_training_history.csv", local_log, list(local_log[0]))
            line = (f"round={round_id} val_accuracy={val_metrics['accuracy']:.6f} "
                    f"val_macro_f1={val_metrics['macro_f1']:.6f} val_weighted_f1={val_metrics['weighted_f1']:.6f} "
                    f"seconds={round_seconds:.2f} best={improved}")
            print(line, flush=True); log_lines.append(line)
            if pilot_only and round_seconds > PILOT_ROUND_LIMIT_SECONDS:
                stop_reason = "pilot_round_exceeded_300_seconds"; break
            working_set = mem.get("working_set_bytes")
            if working_set is not None and working_set > WORKING_SET_LIMIT_BYTES:
                stop_reason = "working_set_exceeded_8_gib"; break
            if time.perf_counter() - start_all > TOTAL_RUNTIME_LIMIT_SECONDS:
                stop_reason = "total_runtime_exceeded_45_minutes"; break
            if not pilot_only and no_improve >= patience:
                stop_reason = "validation_macro_f1_early_stopping"; break
    fit_seconds = time.perf_counter() - start_all
    if not best_round:
        raise RuntimeError("No validation checkpoint was selected")
    best_model = load_model_artifact(checkpoints / "best_model.joblib", inspected["class_to_id"])
    val_cm = np.load(checkpoints / "best_validation_confusion.npy")
    val_metrics = metrics_from_confusion(val_cm, names)
    _save_confusion_csv(build_dir / "confusion_matrix_validation.csv", val_cm, names)
    _write_csv(build_dir / "per_class_validation_metrics.csv", val_metrics["per_class"],
               ["label", "precision", "recall", "f1", "support", "note"])
    _write_json(build_dir / "validation_metrics.json", {"split": "validation", "selected_round": best_round,
                "selection_role": "checkpoint selection and early stopping", **_metric_summary(val_metrics)})

    test_metrics = None
    if not pilot_only:
        # First test-value access occurs strictly after validation checkpoint selection is complete.
        x_test, y_test = arrays["X_test"], arrays["y_test"]
        test_cm = _confusion_for_model(best_model, x_test, y_test, len(names))
        test_metrics = metrics_from_confusion(test_cm, names)
        np.save(build_dir / "confusion_matrix_test.npy", test_cm)
        _save_confusion_csv(build_dir / "confusion_matrix_test.csv", test_cm, names)
        _write_csv(build_dir / "per_class_test_metrics.csv", test_metrics["per_class"],
                   ["label", "precision", "recall", "f1", "support", "note"])
        _write_json(build_dir / "test_metrics.json", {"split": "test", "use": "final evaluation after validation selection",
                    **_metric_summary(test_metrics)})
        del x_test, y_test

    central = json.loads(central_metrics_path.read_text(encoding="utf-8"))
    central_validation_path = central_dir / "best_validation_metrics.json"
    central_validation = json.loads(central_validation_path.read_text(encoding="utf-8"))
    validation_comparison = [{"metric": key, "fedavg": val_metrics[key],
                              "centralized": central_validation[key],
                              "fedavg_minus_centralized": val_metrics[key] - central_validation[key]}
                             for key in ("accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1")]
    _write_csv(build_dir / "centralized_validation_comparison.csv", validation_comparison,
               ["metric", "fedavg", "centralized", "fedavg_minus_centralized"])
    if test_metrics is not None:
        comparison = [{"metric": key, "centralized": central[key], "fedavg": test_metrics[key],
                       "fedavg_minus_centralized": test_metrics[key] - central[key]}
                      for key in ("accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1")]
        _write_csv(build_dir / "centralized_test_comparison.csv", comparison,
                   ["metric", "centralized", "fedavg", "fedavg_minus_centralized"])
    peak = max((m.get("peak_working_set_bytes") or 0 for m in memory_samples), default=0) or None
    metadata = {"created_utc": datetime.now(timezone.utc).isoformat(),
                "status": "pilot_only_no_test_evaluation" if pilot_only else "completed",
                "configuration": config,
                "software": {"python": platform.python_version(), "numpy": np.__version__,
                             "scikit_learn": sklearn.__version__, "joblib": joblib.__version__},
                "dataset": {"train_rows": int(len(y_train)), "validation_rows": int(len(y_val)),
                            "test_rows": int(inspected["metadata"]["derived_rows"]["test"]),
                            "features": int(x_train.shape[1]), "classes": len(names),
                            "class_to_id": inspected["class_to_id"]},
                "client_distributions": clients,
                "selection": {"selected_round": best_round, "validation_macro_f1": best_macro,
                              "test_used_for_selection": False},
                "runtime": {"training_and_validation_seconds": fit_seconds, "rounds_completed": len(history),
                            "max_round_seconds": max(r["round_seconds"] for r in history),
                            "stop_reason": stop_reason, "peak_working_set_bytes": peak,
                            "final_memory": _memory_snapshot()},
                "communication_estimate": {"parameter_count": int(sum(p.size for p in [*best_model.coefs_, *best_model.intercepts_])),
                    "model_parameter_bytes_float32": model_bytes,
                    "upload_download_parameter_bytes_estimate": int(sum(r["round_communication_bytes_estimate"] for r in history)),
                    "assumption": "one full model download and upload per client per round; excludes protocol/security overhead"},
                "validation_evaluation": _metric_summary(val_metrics),
                "test_evaluation": _metric_summary(test_metrics) if test_metrics is not None else None,
                "centralized_test_metrics": central,
                "centralized_validation_metrics": _metric_summary(central_validation),
                "limitations": ["Single-computer simulation; no machine isolation or formal privacy guarantee.",
                                "sklearn private _initialize API is version-sensitive.",
                                "FedAvg averages parameters only; Adam optimizer state is local and reset each round."]}
    _write_json(build_dir / "run_metadata.json", metadata)
    (build_dir / "training.log").write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    (build_dir / "report.md").write_text(_report(metadata, clients, val_metrics, test_metrics, central, history), encoding="utf-8")
    os.replace(build_dir, output_dir)
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, default=DEFAULT_PREPARED)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--rounds", type=int, default=MAX_ROUNDS)
    parser.add_argument("--local-epochs", type=int, default=LOCAL_EPOCHS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--partition-seed", type=int, default=PARTITION_SEED)
    parser.add_argument("--alpha", type=float, default=DIRICHLET_ALPHA)
    parser.add_argument("--patience", type=int, default=PATIENCE)
    parser.add_argument("--pilot-only", action="store_true", help="one validation-only feasibility round; no test access")
    args = parser.parse_args()
    result = run_fedavg(args.prepared, args.output, args.rounds, args.local_epochs, args.seed,
                        args.partition_seed, args.alpha, args.patience, args.pilot_only)
    print(json.dumps({"status": result["status"], "rounds_completed": result["runtime"]["rounds_completed"],
                      "selected_round": result["selection"]["selected_round"],
                      "test_evaluation": result["test_evaluation"]}, indent=2))

if __name__ == "__main__":
    main()



