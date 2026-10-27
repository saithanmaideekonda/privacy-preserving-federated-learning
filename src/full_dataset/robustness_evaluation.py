"""Stage 3 validation robustness and one-shot frozen-candidate evaluation."""
from __future__ import annotations
import argparse, csv, hashlib, json, os, shutil, time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import numpy as np
from src.full_dataset.centralized import (BATCH_SIZE, DEFAULT_SEED, _save_confusion_csv,
    _write_csv, _write_json, load_model_artifact, metrics_from_confusion, validate_label_mapping)
from src.full_dataset.training_parameter_study import run_candidate

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXP = PROJECT_ROOT / "experiments" / "full_dataset"
MI = EXP / "model_improvement"
STAGE2 = MI / "training_parameters"
PREPARED = EXP / "prepared"
DEFAULT_OUTPUT = MI / "robustness_evaluation"
ADDITIONAL_SEEDS = (20261028, 20261029)
FOCUS_CLASSES = ("Uploading_Attack", "Recon-PingSweep", "XSS", "Backdoor_Malware",
    "SqlInjection", "BrowserHijacking", "Recon-OSScan", "DictionaryBruteForce",
    "CommandInjection", "Recon-PortScan", "DNS_Spoofing", "Recon-HostDiscovery")
METRICS = ("accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1")
SELECTION = "selection_frozen.json"
TEST_STARTED = "test_evaluation_started.json"

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def read_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))

def read_csv(path: Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))

def write_json(path: Path, value: dict[str, Any]) -> None:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    _write_json(tmp, value); os.replace(tmp, path)

def _run_dir(seed: int, output: Path = DEFAULT_OUTPUT) -> Path:
    if seed == 20261026:
        return STAGE2 / "learning_rate_double_epochs_12"
    if seed == 20261027:
        return STAGE2 / "seed_repeat_20261027" / "learning_rate_double_seed_20261027"
    return output / f"lr_0p002_seed_{seed}"

def _protected_paths() -> dict[str, Path]:
    return {
        "phase1_inspection": EXP / "inspection_report.md",
        "phase1_preparation_report": PREPARED / "preparation_report.md",
        "phase1_metadata": PREPARED / "phase1_metadata.json",
        "phase1_preprocessing": PREPARED / "preprocessing_metadata.json",
        "phase2_checkpoint": EXP / "centralized/checkpoints/best_model.joblib",
        "phase2_validation_metrics": EXP / "centralized/best_validation_metrics.json",
        "phase2_report": EXP / "centralized/report.md",
        "phase3_checkpoint": EXP / "federated/experiment/checkpoints/best_model.joblib",
        "phase3_report": EXP / "federated/experiment/report.md",
        "phase4_report": EXP / "drift/report.md",
        "phase5_report": EXP / "selective_retraining/report.md",
        "weighting_report": MI / "report.md",
        "weighting_selection": MI / "selection_recommendation.json",
        "diagnostics_report": MI / "diagnostics/report.md",
        "stage2_report": STAGE2 / "report.md",
        "stage2_config": STAGE2 / "study_configuration.json",
        "stage2_selection": STAGE2 / "selection_recommendation.json",
        "stage2_checkpoint_record": STAGE2 / "selected_checkpoint.json",
        "stage2_checkpoint": STAGE2 / "learning_rate_double_epochs_12/checkpoints/best_model.joblib",
        "stage2_seed27_checkpoint": STAGE2 / "seed_repeat_20261027/learning_rate_double_seed_20261027/checkpoints/best_model.joblib",
    }

def snapshot_protected() -> dict[str, dict[str, Any]]:
    snapshot, missing = {}, []
    for name, path in _protected_paths().items():
        if not path.is_file():
            missing.append(str(path)); continue
        snapshot[name] = {"path": str(path.resolve()), "size_bytes": path.stat().st_size,
            "mtime_ns": path.stat().st_mtime_ns, "sha256": sha256_file(path)}
    if missing: raise FileNotFoundError("Protected artifacts missing: " + "; ".join(missing))
    return snapshot

def verify_stage2_audit() -> dict[str, Any]:
    audit = read_json(STAGE2 / "preservation_audit.json"); verified = 0; failures = []
    for name, item in audit.get("prior_artifacts", {}).items():
        path = Path(item["path"])
        if not path.is_file() or sha256_file(path) != item.get("sha256_after_stage"):
            failures.append(name)
        else: verified += 1
    if failures: raise RuntimeError("Stage 2 preservation audit mismatch: " + ", ".join(failures))
    return {"verified_entries": verified, "mismatches": failures}

def verify_prior_results() -> dict[str, Any]:
    phase2 = read_json(EXP / "centralized/best_validation_metrics.json")
    seed26 = read_json(_run_dir(20261026) / "best_validation_metrics.json")
    seed27 = read_json(_run_dir(20261027) / "best_validation_metrics.json")
    actual = {"phase2_reproduction": float(phase2["macro_f1"]),
        "lr002_seed_20261026": float(seed26["macro_f1"]),
        "lr002_seed_20261027": float(seed27["macro_f1"])}
    expected = {"phase2_reproduction": .6697, "lr002_seed_20261026": .6917, "lr002_seed_20261027": .6849}
    if any(abs(actual[k] - expected[k]) > .00006 for k in expected):
        raise ValueError(f"Saved validation values differ from prompt: {actual}")
    selected = read_json(STAGE2 / "selected_checkpoint.json")
    checkpoint = STAGE2 / selected["checkpoint_path"]
    if sha256_file(checkpoint) != selected["checkpoint_sha256"]:
        raise RuntimeError("Stage 2 selected checkpoint hash mismatch")
    return {"verified_values": actual, "stage2_selected_checkpoint_sha256": selected["checkpoint_sha256"]}

def selection_rule() -> dict[str, Any]:
    return {
        "candidate": "learning rate 0.002; seeds 20261026, 20261027, 20261028, 20261029",
        "baseline": "Phase 2 reproduction validation metrics",
        "pass_all": [
            "four-seed mean validation macro-F1 exceeds Phase 2 baseline",
            "at least three of four macro-F1 values exceed Phase 2 baseline",
            "four-seed mean macro-recall is at least Phase 2 baseline",
            "mean accuracy and weighted-F1 are each no more than 0.01 below baseline",
            "mean F1 over the 12 Stage A weak classes is no more than 0.01 below baseline",
            "Uploading_Attack validation recall is nonzero in at least two of four runs"],
        "if_pass": "freeze preselected Stage 2 seed-20261026 checkpoint; do not pick the best of four seeds",
        "if_fail": "freeze original Phase 2 centralized checkpoint",
        "rare_class_caveat": "Uploading_Attack has 37 validation rows; this is exploratory, not reliable rare-attack evidence",
        "test_data_used": False}

def predeclare(output: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    output = Path(output).resolve(); plan_path = output / "predeclared_plan.json"
    if plan_path.exists():
        saved = read_json(plan_path)
        if saved.get("additional_seeds") != list(ADDITIONAL_SEEDS):
            raise ValueError("A different seed plan is already declared")
        return saved
    audit = verify_stage2_audit(); prior = verify_prior_results(); before = snapshot_protected()
    selected = read_json(STAGE2 / "selected_checkpoint.json")["configuration"]
    if (float(selected["learning_rate_init"]) != .002 or int(selected["max_epochs"]) != 12
            or int(selected["patience"]) != 3):
        raise ValueError("Stage 2 configuration does not match requested robustness study")
    output.mkdir(parents=True, exist_ok=True)
    plan = {"created_utc": datetime.now(timezone.utc).isoformat(),
        "additional_seeds": list(ADDITIONAL_SEEDS), "existing_seeds": [20261026, 20261027],
        "configuration": {"learning_rate_init": .002, "max_epochs": 12, "patience": 3,
            "architecture": [44, 128, 64, 34], "batch_size": 8192,
            "class_weighting": "Phase 2 normalized inverse square-root weights",
            "implementation": "unchanged training_parameter_study.run_candidate",
            "updates": "training split only", "selection": "validation macro-F1"},
        "selection_rule": selection_rule(), "verified_prior_results": prior,
        "stage2_preservation_audit": audit, "protected_artifacts_before": before,
        "test_access_before_freeze": "prohibited"}
    write_json(plan_path, plan); return plan

def _per_class(folder: Path) -> dict[str, dict[str, Any]]:
    result = {}
    for row in read_csv(folder / "per_class_validation_metrics.csv"):
        result[row["label"]] = {key: (None if row[key] == "" else float(row[key]))
            for key in ("precision", "recall", "f1")}
        result[row["label"]]["support"] = int(row["support"])
    return result

def _load_run(seed: int, output: Path) -> dict[str, Any]:
    folder = _run_dir(seed, output)
    meta = read_json(folder / "run_metadata.json")
    metrics = read_json(folder / "best_validation_metrics.json")
    config = meta["configuration"]
    if (int(config["seed"]) != seed or float(config["learning_rate_init"]) != .002
            or int(config["max_epochs"]) != 12 or int(config["patience"]) != 3):
        raise ValueError(f"Unexpected run configuration for seed {seed}")
    cm = np.load(folder / "confusion_matrix_validation.npy", allow_pickle=False)
    if int(cm.sum()) != int(metrics["evaluated_rows"]): raise ValueError(f"Validation CM mismatch: {seed}")
    return {"run": f"seed_{seed}", "seed": seed, "selected_epoch": int(metrics["selected_epoch"]),
        **{key: float(metrics[key]) for key in METRICS},
        "runtime_seconds": float(meta["timing_seconds"]["training_and_epoch_validation"]),
        "epochs_completed": int(meta["epochs_completed"]), "warnings": meta.get("warnings", []),
        "per_class": _per_class(folder), "cm": cm}

def _load_baseline() -> dict[str, Any]:
    folder = EXP / "centralized"; metrics = read_json(folder / "best_validation_metrics.json")
    cm = np.load(folder / "confusion_matrix_validation.npy", allow_pickle=False)
    if int(cm.sum()) != int(metrics["evaluated_rows"]): raise ValueError("Phase 2 validation CM mismatch")
    return {"run": "phase2_reproduction", **{key: float(metrics[key]) for key in METRICS},
        "per_class": _per_class(folder), "cm": cm}


def aggregate_seed_metrics(runs: list[dict[str, Any]]) -> dict[str, Any]:
    if not runs: raise ValueError("No seed metrics supplied")
    summary = {"seed_count": len(runs), "metrics": {}}
    for key in METRICS:
        values = np.asarray([r[key] for r in runs], dtype=np.float64)
        summary["metrics"][key] = {"mean": float(values.mean()),
            "sample_std": float(values.std(ddof=1)) if len(values) > 1 else None,
            "min": float(values.min()), "max": float(values.max()),
            "range": float(values.max() - values.min())}
    summary["warning"] = "Four seeds are exploratory; this small set is not formal evidence of reliability."
    return summary

def evaluate_selection_rule(baseline: dict[str, Any], runs: list[dict[str, Any]]) -> dict[str, Any]:
    if len(runs) != 4 or len({r["seed"] for r in runs}) != 4:
        raise ValueError("Selection requires exactly four distinct seeds")
    base_f1 = float(baseline["macro_f1"])
    mean = {key: float(np.mean([r[key] for r in runs])) for key in METRICS}
    base_weak = float(np.mean([baseline["per_class"][c]["f1"] for c in FOCUS_CLASSES]))
    cand_weak = float(np.mean([r["per_class"][c]["f1"] for r in runs for c in FOCUS_CLASSES]))
    uploading = [float(r["per_class"]["Uploading_Attack"]["recall"]) for r in runs]
    checks = {
        "mean_macro_f1_exceeds_baseline": mean["macro_f1"] > base_f1,
        "three_of_four_macro_f1_exceed_baseline": sum(r["macro_f1"] > base_f1 for r in runs) >= 3,
        "mean_macro_recall_not_lower": mean["macro_recall"] >= baseline["macro_recall"],
        "accuracy_within_1pp": mean["accuracy"] >= baseline["accuracy"] - .01,
        "weighted_f1_within_1pp": mean["weighted_f1"] >= baseline["weighted_f1"] - .01,
        "12_weak_class_mean_f1_within_1pp": cand_weak >= base_weak - .01,
        "uploading_recall_positive_in_two_runs": sum(x > 0 for x in uploading) >= 2}
    return {"checks": checks, "passed": all(checks.values()), "candidate_mean_metrics": mean,
        "baseline_macro_f1": base_f1, "baseline_12_weak_class_mean_f1": base_weak,
        "candidate_12_weak_class_mean_f1": cand_weak,
        "uploading_attack_validation_recall_by_seed": uploading,
        "uploading_attack_positive_seed_count": sum(x > 0 for x in uploading),
        "test_data_used": False}

def _write_validation_summary(output: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    runs = [_load_run(seed, output) for seed in (20261026, 20261027, *ADDITIONAL_SEEDS)]
    baseline = _load_baseline(); class_rows, weak_rows = [], []
    for run in runs:
        names = list(run["per_class"])
        for i, label in enumerate(names):
            item = run["per_class"][label]
            class_rows.append({"run": run["run"], "seed": run["seed"], "class": label,
                "support": item["support"], "precision": item["precision"], "recall": item["recall"],
                "f1": item["f1"], "predicted_count": int(run["cm"][:, i].sum())})
        for label in FOCUS_CLASSES:
            i = names.index(label); item = run["per_class"][label]
            errors = run["cm"][i].copy(); errors[i] = 0
            top = np.argsort(errors)[::-1][:3]
            weak_rows.append({"run": run["run"], "seed": run["seed"], "class": label,
                "validation_support": item["support"], "precision": item["precision"],
                "recall": item["recall"], "f1": item["f1"],
                "predicted_count": int(run["cm"][:, i].sum()),
                "top_confusion_destinations": "; ".join(
                    f"{names[j]}:{int(errors[j])}" for j in top if errors[j] > 0)})
    rows = [{key: run[key] for key in
        ("run", "seed", "selected_epoch", *METRICS, "runtime_seconds", "epochs_completed")} for run in runs]
    _write_csv(output / "validation_seed_metrics.csv", rows, list(rows[0]))
    _write_csv(output / "validation_per_class_by_seed.csv", class_rows, list(class_rows[0]))
    _write_csv(output / "validation_weak_classes_by_seed.csv", weak_rows, list(weak_rows[0]))
    result = {"aggregate": aggregate_seed_metrics(runs), "selection": evaluate_selection_rule(baseline, runs)}
    write_json(output / "seed_aggregation.json", {**result, "runs": rows,
        "phase2_validation_baseline": {key: baseline[key] for key in METRICS},
        "test_evaluation_performed": False})
    return runs, {**result, "baseline": baseline}

def run_validation_stage(output: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    output = Path(output).resolve(); output.mkdir(parents=True, exist_ok=True)
    predeclare(output)
    if (output / SELECTION).exists() or (output / TEST_STARTED).exists():
        raise RuntimeError("Selection or test evaluation already started")
    statuses = []
    for seed in ADDITIONAL_SEEDS:
        config = {"name": f"lr_0p002_seed_{seed}", "learning_rate_init": .002,
            "max_epochs": 12, "patience": 3, "seed": seed,
            "factor_changed": "seed only; exact Stage 2 implementation"}
        folder = _run_dir(seed, output)
        if folder.exists():
            if not (folder / "run_metadata.json").is_file():
                raise FileExistsError(f"Incomplete run exists; refusing overwrite: {folder}")
            saved = read_json(folder / "run_metadata.json")["configuration"]
            if (int(saved["seed"]) != seed or float(saved["learning_rate_init"]) != .002
                    or int(saved["max_epochs"]) != 12 or int(saved["patience"]) != 3):
                raise ValueError(f"Existing run has incompatible configuration: {folder}")
            statuses.append({"seed": seed, "status": "reused_completed_run"})
        else:
            run_candidate(config, prepared_dir=PREPARED, output_dir=output)
            statuses.append({"seed": seed, "status": "trained"})
    _, result = _write_validation_summary(output)
    write_json(output / "validation_stage_status.json", {
        "completed_utc": datetime.now(timezone.utc).isoformat(), "runs": statuses,
        "selection_rule_result": result["selection"], "test_arrays_loaded": False,
        "test_labels_loaded": False})
    return {"runs": statuses, "selection": result["selection"], "aggregate": result["aggregate"]}

def freeze_selection(output: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    output = Path(output).resolve()
    if not (output / "validation_stage_status.json").is_file() or not (output / "predeclared_plan.json").is_file():
        raise RuntimeError("Predeclared plan and completed validation results are required")
    if (output / SELECTION).exists() or (output / TEST_STARTED).exists():
        raise FileExistsError("A freeze or test marker already exists")
    plan = read_json(output / "predeclared_plan.json")
    _, result = _write_validation_summary(output); decision = result["selection"]
    if decision["passed"]:
        record = read_json(STAGE2 / "selected_checkpoint.json")
        source = STAGE2 / record["checkpoint_path"]; source_dir = source.parent.parent
        config = read_json(source_dir / "run_metadata.json")["configuration"]
        val_metrics = read_json(source_dir / "best_validation_metrics.json")
        source_description = "Stage 2 preselected learning-rate 0.002 checkpoint, seed 20261026"
    else:
        source = EXP / "centralized/checkpoints/best_model.joblib"
        source_dir = EXP / "centralized"
        config = read_json(source_dir / "run_metadata.json")["configuration"]
        val_metrics = read_json(source_dir / "best_validation_metrics.json")
        source_description = "Original Phase 2 centralized checkpoint; validation gates did not all pass"
    if val_metrics.get("test_used_for_selection", False):
        raise RuntimeError("Source checkpoint indicates test-based selection")
    checkpoint = output / "frozen_candidate.joblib"
    if checkpoint.exists(): raise FileExistsError(f"Refusing to overwrite {checkpoint}")
    src_hash = sha256_file(source); shutil.copyfile(source, checkpoint)
    frozen_hash = sha256_file(checkpoint)
    if src_hash != frozen_hash: raise IOError("Frozen checkpoint copy hash mismatch")
    frozen_dir = output / "frozen_artifacts"; frozen_dir.mkdir(exist_ok=False)
    artifacts = {}
    for name in ("phase1_metadata.json", "preprocessing_metadata.json"):
        src, dst = PREPARED / name, frozen_dir / name
        shutil.copyfile(src, dst)
        artifacts[name] = {"source": str(src.resolve()), "sha256": sha256_file(dst)}
    prep_src = PREPARED / "arrays/preprocessing.joblib"
    prep_dst = frozen_dir / "preprocessing.joblib"; shutil.copyfile(prep_src, prep_dst)
    artifacts["preprocessing.joblib"] = {"source": str(prep_src.resolve()), "sha256": sha256_file(prep_dst)}
    prep = read_json(PREPARED / "preprocessing_metadata.json")
    names = validate_label_mapping(prep["class_to_id"])
    label_map = frozen_dir / "label_mapping.json"
    write_json(label_map, {"class_to_id": prep["class_to_id"], "class_names_in_id_order": names})
    artifacts["label_mapping.json"] = {
        "source": str((PREPARED / "preprocessing_metadata.json").resolve()),
        "sha256": sha256_file(label_map)}
    config_dst = frozen_dir / "configuration.json"; write_json(config_dst, config)
    artifacts["configuration.json"] = {"source": str((source_dir / "run_metadata.json").resolve()),
        "sha256": sha256_file(config_dst)}
    write_json(output / "frozen_artifact_manifest.json", {
        "checkpoint": {"source": str(source.resolve()), "source_sha256": src_hash,
            "frozen_path": str(checkpoint.resolve()), "frozen_sha256": frozen_hash},
        "artifacts": artifacts})
    frozen = {"frozen_utc": datetime.now(timezone.utc).isoformat(),
        "selected_source": source_description, "candidate_selected": bool(decision["passed"]),
        "selection_rule": plan["selection_rule"], "selection_rule_result": decision,
        "checkpoint_path": checkpoint.name, "checkpoint_sha256": frozen_hash,
        "configuration": config, "validation_metrics": {key: float(val_metrics[key]) for key in METRICS},
        "class_mapping_sha256": sha256_file(label_map),
        "preprocessing_metadata_sha256": artifacts["preprocessing_metadata.json"]["sha256"],
        "test_arrays_loaded": False, "test_labels_loaded": False, "test_evaluation_performed": False}
    write_json(output / SELECTION, frozen); write_json(output / "selection_decision.json", frozen)
    return frozen


def _check_frozen(output: Path) -> tuple[dict[str, Any], Path]:
    path = output / SELECTION
    if not path.is_file(): raise RuntimeError("Final evaluation blocked: validation selection is not frozen")
    frozen = read_json(path); checkpoint = output / frozen["checkpoint_path"]
    if not checkpoint.is_file() or sha256_file(checkpoint) != frozen.get("checkpoint_sha256"):
        raise RuntimeError("Frozen checkpoint is missing or its SHA-256 does not match")
    if frozen.get("test_evaluation_performed"): raise RuntimeError("Final test evaluation already recorded")
    return frozen, checkpoint

def evaluate_frozen_candidate_once(output: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    output = Path(output).resolve(); frozen, checkpoint = _check_frozen(output)
    marker = output / TEST_STARTED
    try:
        fd = os.open(marker, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise RuntimeError("Final test evaluation is one-shot and has already started") from exc
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        json.dump({"started_utc": datetime.now(timezone.utc).isoformat(),
            "frozen_checkpoint_sha256": frozen["checkpoint_sha256"], "completed": False}, stream, indent=2)

    prep = read_json(output / "frozen_artifacts/preprocessing_metadata.json")
    class_names = validate_label_mapping(prep["class_to_id"])
    # No test-array path is opened until the frozen-selection gate above succeeds.
    x_test = np.load(PREPARED / "arrays/X_test.npy", mmap_mode="r", allow_pickle=False)
    y_test = np.load(PREPARED / "arrays/y_test.npy", mmap_mode="r", allow_pickle=False)
    if (x_test.dtype != np.float32 or y_test.dtype != np.int32 or x_test.ndim != 2
            or x_test.shape[1] != len(prep["feature_columns_retained"]) or y_test.shape != (len(x_test),)):
        raise ValueError("Prepared test arrays do not match frozen preprocessing metadata")
    if len(y_test) and (int(y_test.min()) < 0 or int(y_test.max()) >= len(class_names)):
        raise ValueError("Test labels fall outside frozen label mapping")
    model = load_model_artifact(checkpoint, prep["class_to_id"])
    matrix = np.zeros((len(class_names), len(class_names)), dtype=np.int64)
    started = time.perf_counter()
    for start in range(0, len(y_test), BATCH_SIZE):
        end = min(start + BATCH_SIZE, len(y_test))
        truth = np.asarray(y_test[start:end], dtype=np.int64)
        predicted = model.predict(np.asarray(x_test[start:end], dtype=np.float32)).astype(np.int64, copy=False)
        np.add.at(matrix, (truth, predicted), 1)
    elapsed = time.perf_counter() - started
    metrics = metrics_from_confusion(matrix, class_names)
    metrics.update({"test_evaluation_seconds": elapsed, "checkpoint_sha256": frozen["checkpoint_sha256"],
        "test_evaluation_performed": True})
    base_dir = EXP / "centralized"
    baseline = read_json(base_dir / "test_metrics.json")
    baseline_matrix = np.load(base_dir / "confusion_matrix_test.npy", allow_pickle=False)
    baseline_rows = read_csv(base_dir / "per_class_test_metrics.csv")
    if int(baseline["evaluated_rows"]) != len(y_test) or int(baseline_matrix.sum()) != len(y_test):
        raise ValueError("Saved Phase 2 test results do not cover the same prepared test row count")
    if baseline_matrix.shape != matrix.shape or {r["label"] for r in baseline_rows} != set(class_names):
        raise ValueError("Saved Phase 2 test evaluation uses a different class mapping")
    baseline_recomputed = metrics_from_confusion(baseline_matrix, class_names)
    for key in METRICS:
        if not np.isclose(float(baseline[key]), baseline_recomputed[key], rtol=0, atol=1e-10):
            raise ValueError(f"Phase 2 saved test metric is inconsistent with its confusion matrix: {key}")

    _write_json(output / "final_candidate_test_metrics.json", metrics)
    _write_csv(output / "final_candidate_per_class_test_metrics.csv", metrics["per_class"],
        ["label", "precision", "recall", "f1", "support", "note"])
    np.save(output / "final_candidate_confusion_matrix.npy", matrix)
    _save_confusion_csv(output / "final_candidate_confusion_matrix.csv", matrix, class_names)
    rare = {}
    for label in ("Uploading_Attack", "XSS", "Backdoor_Malware", "SqlInjection", "BrowserHijacking"):
        i = class_names.index(label); row = metrics["per_class"][i]
        errors = matrix[i].copy(); errors[i] = 0; top = np.argsort(errors)[::-1][:5]
        rare[label] = {"true_support": int(matrix[i].sum()), "predicted_count": int(matrix[:, i].sum()),
            "true_positive": int(matrix[i, i]), "recall": row["recall"], "f1": row["f1"],
            "top_confusion_destinations": [{"label": class_names[j], "count": int(errors[j])}
                for j in top if errors[j] > 0]}
    comparison = {
        "test_rows": int(len(y_test)),
        "frozen_candidate": {key: metrics[key] for key in METRICS},
        "phase2_baseline_saved_evaluation": {key: float(baseline[key]) for key in METRICS},
        "delta_candidate_minus_phase2": {key: float(metrics[key] - float(baseline[key])) for key in METRICS},
        "same_prepared_test_row_count": True, "baseline_evaluation_reused": True,
        "candidate_checkpoint_sha256": frozen["checkpoint_sha256"],
        "phase2_checkpoint_sha256": sha256_file(base_dir / "checkpoints/best_model.joblib"),
        "rare_class_predictions": rare, "candidate_confusion_matrix": matrix.tolist(),
        "phase2_confusion_matrix": baseline_matrix.tolist(),
        "test_data_used_for_selection": False, "frozen_candidate_evaluated_once": True}
    write_json(output / "final_test_comparison.json", comparison)
    table = [{"approach": name, **{key: values[key] for key in METRICS}}
        for name, values in (("Frozen candidate", metrics), ("Phase 2 baseline", baseline))]
    _write_csv(output / "final_test_comparison.csv", table, ["approach", *METRICS])
    write_json(marker, {"started_utc": read_json(marker)["started_utc"],
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "frozen_checkpoint_sha256": frozen["checkpoint_sha256"], "completed": True,
        "rows_evaluated": int(len(y_test))})
    return comparison

def verify_stage2_selected_metadata() -> dict[str, Any]:
    record = read_json(STAGE2 / "selected_checkpoint.json")
    checkpoint = STAGE2 / record["checkpoint_path"]
    directory = checkpoint.parent.parent
    run_meta = read_json(directory / "run_metadata.json")
    config = read_json(directory / "configuration.json")
    metrics = read_json(directory / "best_validation_metrics.json")
    checks = {
        "checkpoint_matches_selected_record": sha256_file(checkpoint) == record["checkpoint_sha256"],
        "run_metadata_matches_selected_configuration": run_meta["configuration"] == record["configuration"],
        "configuration_file_matches_run_metadata": config == run_meta["configuration"],
        "validation_macro_f1_matches_selected_record": bool(np.isclose(
            float(metrics["macro_f1"]), float(record["validation_macro_f1"]), rtol=0, atol=1e-12)),
        "stage2_candidate_has_no_test_selection": not bool(run_meta.get("test_arrays_loaded")) and
            not bool(run_meta.get("test_labels_loaded")),
    }
    if not all(checks.values()): raise RuntimeError(f"Stage 2 selected metadata integrity check failed: {checks}")
    return {"checks": checks, "files": {
        "checkpoint": {"path": str(checkpoint.resolve()), "sha256": sha256_file(checkpoint)},
        "run_metadata": {"path": str((directory / "run_metadata.json").resolve()),
                         "sha256": sha256_file(directory / "run_metadata.json")},
        "configuration": {"path": str((directory / "configuration.json").resolve()),
                          "sha256": sha256_file(directory / "configuration.json")},
        "validation_metrics": {"path": str((directory / "best_validation_metrics.json").resolve()),
                               "sha256": sha256_file(directory / "best_validation_metrics.json")},
    }}

def finalize_preservation_audit(output: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    output = Path(output).resolve(); plan = read_json(output / "predeclared_plan.json")
    before = plan["protected_artifacts_before"]; after = snapshot_protected()
    changed = [key for key, old in before.items()
        if key not in after or old["sha256"] != after[key]["sha256"]
        or old["size_bytes"] != after[key]["size_bytes"]]
    audit = {"completed_utc": datetime.now(timezone.utc).isoformat(),
        "protected_artifact_count": len(before), "unchanged": not changed, "changed_artifacts": changed,
        "before": before, "after": after, "stage2_prior_audit": verify_stage2_audit(), "stage2_selected_metadata": verify_stage2_selected_metadata(),
        "sampled_project_modified": False, "phase2_phase3_checkpoints_modified": False}
    write_json(output / "preservation_audit.json", audit)
    if changed: raise RuntimeError("Protected artifacts changed: " + ", ".join(changed))
    return audit

def write_report(output: Path = DEFAULT_OUTPUT) -> Path:
    output = Path(output).resolve(); agg_doc = read_json(output / "seed_aggregation.json")
    frozen = read_json(output / SELECTION); final = read_json(output / "final_test_comparison.json")
    rows = read_csv(output / "validation_seed_metrics.csv")
    lines = ["# Stage 3 — Validation robustness and final candidate evaluation", "",
        "## Protocol", "",
        "Two additional predeclared seeds (20261028, 20261029) used the unchanged Stage 2 run_candidate implementation: learning rate 0.002, 12-epoch maximum, patience 3, Phase 2 class weights, 8,192-row batches, [128, 64] MLP, and identical prepared train/validation splits. Training updates used training only; validation selected checkpoints. Test arrays were not opened before selection was frozen.",
        "Four seeds are a small exploratory check, not formal evidence of robustness.", "",
        "## Validation results", "",
        "| Seed | Epoch | Accuracy | Macro-P | Macro-R | Macro-F1 | Weighted-F1 | Runtime (s) |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(f"| {r['seed']} | {r['selected_epoch']} | {float(r['accuracy']):.6f} | {float(r['macro_precision']):.6f} | {float(r['macro_recall']):.6f} | {float(r['macro_f1']):.6f} | {float(r['weighted_f1']):.6f} | {float(r['runtime_seconds']):.1f} |")
    lines += ["", "| Metric | Mean | Sample SD | Min | Max | Range |", "|---|---:|---:|---:|---:|---:|"]
    for key, value in agg_doc["aggregate"]["metrics"].items():
        lines.append(f"| {key} | {value['mean']:.6f} | {value['sample_std']:.6f} | {value['min']:.6f} | {value['max']:.6f} | {value['range']:.6f} |")
    lines += ["", "Phase 2 reproduction validation macro-F1 was 0.669651; Stage 2 learning-rate 0.002 seeds scored 0.691748 and 0.684908. Additional seeds are reported above.", "",
        "## Validation-only freeze", "",
        f"Selection rule: {'passed' if frozen['selection_rule_result']['passed'] else 'failed'}. Frozen source: {frozen['selected_source']}. Validation macro-F1: {frozen['validation_metrics']['macro_f1']:.6f}. Frozen checkpoint SHA-256: {frozen['checkpoint_sha256']}.", "",
        "| Predeclared gate | Result |", "|---|:---:|"]
    for key, passed in frozen["selection_rule_result"]["checks"].items():
        lines.append(f"| {key.replace('_', ' ')} | {'Pass' if passed else 'Fail'} |")
    lines += ["", "Uploading_Attack has 37 validation examples. Per-seed support, recall, F1, predicted counts, and leading confusion destinations are in validation_weak_classes_by_seed.csv. These small supports do not establish stable detection. Full per-class metrics are in validation_per_class_by_seed.csv.", "",
        "## Final test evaluation after freeze", "",
        f"The frozen checkpoint was evaluated once on {final['test_rows']:,} prepared test rows. The Phase 2 baseline evaluation was reused from its saved metrics and confusion matrix, after checking that it covers the same row count and class mapping.", "",
        "| Approach | Accuracy | Macro-P | Macro-R | Macro-F1 | Weighted-F1 |", "|---|---:|---:|---:|---:|---:|"]
    for key, label in (("frozen_candidate", "Frozen candidate"), ("phase2_baseline_saved_evaluation", "Phase 2 centralized baseline")):
        m = final[key]
        lines.append(f"| {label} | {m['accuracy']:.6f} | {m['macro_precision']:.6f} | {m['macro_recall']:.6f} | {m['macro_f1']:.6f} | {m['weighted_f1']:.6f} |")
    lines += ["", "Per-class test metrics, confusion counts, rare-class prediction counts, and deltas are stored in adjacent machine-readable files.", "",
        "## Limitations", "", "- Four seeds are too few for formal reliability claims; rare-class validation support is especially small.",
        "- Accuracy and weighted-F1 are dominated by common classes; macro metrics provide a complementary view.",
        "- A single test evaluation does not establish dependable deployment detection.",
        "- Previous Phase 1–5, Phase 2/3, and Stage 2 artifacts were checked with SHA-256 audits.", ""]
    report = output / "report.md"; report.write_text("\n".join(lines), encoding="utf-8"); return report

def write_per_class_test_comparison(output: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    """Build a comparison from saved confusion matrices; never reload test arrays."""
    output = Path(output).resolve()
    comparison = read_json(output / "final_test_comparison.json")
    prep = read_json(output / "frozen_artifacts/preprocessing_metadata.json")
    names = validate_label_mapping(prep["class_to_id"])
    candidate_cm = np.load(output / "final_candidate_confusion_matrix.npy", allow_pickle=False)
    baseline_cm = np.asarray(comparison["phase2_confusion_matrix"], dtype=np.int64)
    if candidate_cm.shape != baseline_cm.shape or candidate_cm.shape != (len(names), len(names)):
        raise ValueError("Confusion matrices do not match frozen class mapping")
    if int(candidate_cm.sum()) != int(baseline_cm.sum()):
        raise ValueError("Candidate and baseline confusion matrices cover different rows")
    candidate = metrics_from_confusion(candidate_cm, names)["per_class"]
    baseline = metrics_from_confusion(baseline_cm, names)["per_class"]
    rows = []
    for i, label in enumerate(names):
        a, b = candidate[i], baseline[i]
        if a["support"] != b["support"]: raise ValueError(f"Test support mismatch for {label}")
        rows.append({"label": label, "test_support": int(a["support"]),
            "candidate_predicted_count": int(candidate_cm[:, i].sum()),
            "phase2_predicted_count": int(baseline_cm[:, i].sum()),
            "candidate_precision": a["precision"], "phase2_precision": b["precision"],
            "delta_precision": None if a["precision"] is None or b["precision"] is None else a["precision"] - b["precision"],
            "candidate_recall": a["recall"], "phase2_recall": b["recall"],
            "delta_recall": None if a["recall"] is None or b["recall"] is None else a["recall"] - b["recall"],
            "candidate_f1": a["f1"], "phase2_f1": b["f1"],
            "delta_f1": None if a["f1"] is None or b["f1"] is None else a["f1"] - b["f1"],
            "note": a["note"] or b["note"]})
    _write_csv(output / "per_class_test_comparison.csv", rows, list(rows[0]))
    supported = [r for r in rows if r["delta_f1"] is not None]
    gains = sorted(supported, key=lambda r: r["delta_f1"], reverse=True)
    losses = sorted(supported, key=lambda r: r["delta_f1"])
    absent = [r["label"] for r in rows if r["test_support"] == 0]
    summary = {"class_count": len(rows), "supported_classes": len(supported),
        "classes_absent_from_test": absent, "top_f1_gains": gains[:5],
        "top_f1_regressions": losses[:5], "all_classes_have_matching_test_support": True,
        "source": "saved candidate and Phase 2 confusion matrices; test arrays not reloaded"}
    write_json(output / "per_class_test_comparison_summary.json", summary)
    report = output / "report.md"
    if report.exists():
        focus = [r for r in rows if r["label"] in ("Uploading_Attack", "XSS", "Backdoor_Malware", "SqlInjection", "BrowserHijacking")]
        section = ["", "## Per-class comparison", "",
            "Supports match because both confusion matrices summarize the same test rows. Undefined metrics remain blank; they do not mean zero performance.", "",
            "| Class | Support | Candidate P/R/F1 | Phase 2 P/R/F1 | Delta F1 |",
            "|---|---:|---:|---:|---:|"]
        def fmt(v): return "undefined" if v is None else f"{v:.4f}"
        for r in focus:
            section.append(f"| {r['label']} | {r['test_support']} | {fmt(r['candidate_precision'])} / {fmt(r['candidate_recall'])} / {fmt(r['candidate_f1'])} | {fmt(r['phase2_precision'])} / {fmt(r['phase2_recall'])} / {fmt(r['phase2_f1'])} | {fmt(r['delta_f1'])} |")
        section += ["", "### Largest supported-class F1 changes", "",
            "| Class | Support | Candidate F1 | Phase 2 F1 | Delta F1 |",
            "|---|---:|---:|---:|---:|"]
        for r in gains[:5]:
            section.append(f"| {r['label']} | {r['test_support']} | {fmt(r['candidate_f1'])} | {fmt(r['phase2_f1'])} | +{fmt(r['delta_f1'])} |")
        for r in losses[:5]:
            section.append(f"| {r['label']} | {r['test_support']} | {fmt(r['candidate_f1'])} | {fmt(r['phase2_f1'])} | {fmt(r['delta_f1'])} |")
        if absent: section += ["", "Classes absent from test: " + ", ".join(absent) + "."]
        with report.open("a", encoding="utf-8") as stream: stream.write("\n".join(section) + "\n")
    return summary

def write_runtime_summary(output: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    output = Path(output).resolve()
    records = []
    for seed in ADDITIONAL_SEEDS:
        run = read_json(_run_dir(seed, output) / "run_metadata.json")
        samples = run.get("resource_samples", [])
        peak = max((int(item["peak_working_set_bytes"]) for item in samples
                    if item.get("peak_working_set_bytes") is not None), default=None)
        records.append({"seed": seed,
            "training_and_validation_seconds": float(run["timing_seconds"]["training_and_epoch_validation"]),
            "peak_working_set_bytes": peak, "warnings": run.get("warnings", [])})
    final_metrics = read_json(output / "final_candidate_test_metrics.json")
    summary = {"additional_seed_runs": records,
        "total_additional_seed_runtime_seconds": sum(r["training_and_validation_seconds"] for r in records),
        "peak_working_set_bytes": max((r["peak_working_set_bytes"] for r in records if r["peak_working_set_bytes"] is not None), default=None),
        "final_test_inference_seconds": float(final_metrics["test_evaluation_seconds"]),
        "warnings": {str(r["seed"]): r["warnings"] for r in records}}
    write_json(output / "runtime_resource_summary.json", summary)
    report = output / "report.md"
    if report.exists():
        contents = report.read_text(encoding="utf-8")
        heading = "## Runtime, memory and warnings"
        if heading not in contents:
            total = summary["total_additional_seed_runtime_seconds"]
            peak = summary["peak_working_set_bytes"]
            warnings_text = "No training warnings were recorded." if not any(summary["warnings"].values()) else "See runtime_resource_summary.json for warnings."
            section = ["", heading, "",
                f"The two additional training runs took {total:.1f} seconds combined; peak sampled working set was {peak / (1024 ** 3):.2f} GiB.",
                f"The final test inference pass took {summary['final_test_inference_seconds']:.2f} seconds.",
                warnings_text, ""]
            with report.open("a", encoding="utf-8") as stream: stream.write("\n".join(section))
    return summary

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("validation", "freeze", "final-test"), required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    if args.stage == "validation":
        result = run_validation_stage(args.output_dir)
    elif args.stage == "freeze":
        result = freeze_selection(args.output_dir); finalize_preservation_audit(args.output_dir)
    else:
        result = evaluate_frozen_candidate_once(args.output_dir)
        finalize_preservation_audit(args.output_dir); write_report(args.output_dir); write_per_class_test_comparison(args.output_dir); write_runtime_summary(args.output_dir)
    print(json.dumps(result, indent=2)); return 0

if __name__ == "__main__":
    raise SystemExit(main())





