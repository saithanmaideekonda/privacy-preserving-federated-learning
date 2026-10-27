"""Read-only loaders for saved full-dataset experiment artifacts.

Evaluation loads saved metrics/confusion matrices only, never prepared test arrays.
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
EXPERIMENTS = ROOT / "experiments" / "full_dataset"
ROBUSTNESS = EXPERIMENTS / "model_improvement" / "robustness_evaluation"
PREPARED = EXPERIMENTS / "prepared"
MODEL_PATHS = {
    "Original centralized baseline": (
        EXPERIMENTS / "centralized" / "test_metrics.json",
        EXPERIMENTS / "centralized" / "per_class_test_metrics.csv",
        EXPERIMENTS / "centralized" / "confusion_matrix_test.npy"),
    "Frozen improved centralized": (
        ROBUSTNESS / "final_candidate_test_metrics.json",
        ROBUSTNESS / "final_candidate_per_class_test_metrics.csv",
        ROBUSTNESS / "final_candidate_confusion_matrix.npy"),
    "FedAvg": (
        EXPERIMENTS / "federated" / "experiment" / "test_metrics.json",
        EXPERIMENTS / "federated" / "experiment" / "per_class_test_metrics.csv",
        EXPERIMENTS / "federated" / "experiment" / "confusion_matrix_test.npy"),
}

class ArtifactError(RuntimeError):
    """A required saved artifact is missing or malformed."""

def read_json(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise ArtifactError(f"Required artifact is missing: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactError(f"Could not read JSON artifact {path.name}: {exc}") from exc
    if not isinstance(data, dict):
        raise ArtifactError(f"Expected a JSON object in {path.name}.")
    return data

def _load_saved_labels() -> list[str]:
    """Read class names from the frozen saved Phase 1 mapping."""
    path = ROBUSTNESS / "frozen_artifacts" / "label_mapping.json"
    try:
        mapping = read_json(path)
        names = mapping.get("class_names_in_id_order")
        class_to_id = mapping.get("class_to_id", {})
        if (not isinstance(names, list) or not names or
                len(names) != len(class_to_id) or
                sorted(int(v) for v in class_to_id.values()) != list(range(len(names)))):
            return []
        return [str(name) for name in names]
    except (ArtifactError, TypeError, ValueError):
        return []


LABELS = _load_saved_labels()


def read_csv(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.is_file():
        raise ArtifactError(f"Required artifact is missing: {path}")
    try:
        return pd.read_csv(path)
    except (OSError, pd.errors.ParserError, UnicodeError) as exc:
        raise ArtifactError(f"Could not read CSV artifact {path.name}: {exc}") from exc

def read_confusion(path: str | Path) -> np.ndarray:
    path = Path(path)
    if not path.is_file():
        raise ArtifactError(f"Required confusion-matrix artifact is missing: {path}")
    try:
        matrix = np.load(path, allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise ArtifactError(f"Could not read {path.name}: {exc}") from exc
    if matrix.shape != (len(LABELS), len(LABELS)) or not np.issubdtype(matrix.dtype, np.integer):
        raise ArtifactError(f"Unexpected confusion matrix shape or dtype in {path.name}.")
    return matrix

def load_dataset_summary() -> dict[str, Any]:
    metadata = read_json(PREPARED / "phase1_metadata.json")
    sizes = read_json(PREPARED / "split_sizes.json")
    counts = read_csv(PREPARED / "class_counts.csv")
    if not {"label", "train", "validation", "test"}.issubset(counts.columns):
        raise ArtifactError("class_counts.csv is missing expected split columns.")
    for col in ("train", "validation", "test"):
        counts[col] = pd.to_numeric(counts[col], errors="raise").astype("int64")
    return {"metadata": metadata, "sizes": sizes, "class_counts": counts,
            "input_features": metadata.get("feature_columns_input", []),
            "retained_features": metadata.get("feature_columns_retained", []),
            "classes": metadata.get("labels", LABELS)}

def load_dataset_audit() -> dict[str, Any]:
    """Load only the saved audit metadata used on the Dataset page."""
    paths = {
        "preprocessing": PREPARED / "preprocessing_metadata.json",
        "split_methodology": PREPARED / "split_methodology.json",
        "duplicate_overlap_audit": PREPARED / "duplicate_overlap_audit.json",
    }
    result: dict[str, Any] = {"errors": {}}
    for name, path in paths.items():
        try:
            result[name] = read_json(path)
        except ArtifactError as exc:
            result[name] = None
            result["errors"][name] = str(exc)
    return result


def dataset_integrity_items(summary: dict[str, Any], audit: dict[str, Any]) -> list[dict[str, str]]:
    """Create brief, evidence-backed Dataset page status items."""
    counts = summary.get("class_counts")
    class_names = summary.get("classes", [])
    class_total = len(class_names)
    split_coverage: dict[str, int] = {}
    if isinstance(counts, pd.DataFrame):
        for split in ("train", "validation", "test"):
            if split in counts:
                split_coverage[split] = int((pd.to_numeric(counts[split], errors="coerce").fillna(0) > 0).sum())
    all_classes = class_total > 0 and all(split_coverage.get(split, -1) == class_total
                                           for split in ("train", "validation", "test"))
    if all_classes:
        class_status, class_detail = "Verified", f"{class_total} of {class_total} classes appear in every split."
    else:
        coverage = ", ".join(f"{s}: {split_coverage.get(s, 0)}/{class_total}" for s in ("train", "validation", "test"))
        class_status, class_detail = "Review", f"Class coverage is incomplete or unavailable ({coverage})."

    overlap = (audit.get("duplicate_overlap_audit") or {})
    exact = overlap.get("derived_cross_split_exact_row_overlap_groups")
    feature = overlap.get("derived_cross_split_feature_vector_overlap_groups")
    if isinstance(exact, int) and isinstance(feature, int):
        if exact == 0 and feature == 0:
            overlap_status = "Verified"
            overlap_detail = "No shared feature fingerprints across derived splits; device/session independence is not guaranteed."
        else:
            overlap_status = "Review"
            overlap_detail = f"The audit reports {exact:,} exact-row and {feature:,} feature-vector overlap groups across derived splits."
    else:
        overlap_status, overlap_detail = "Unavailable", "The derived-split fingerprint overlap result is missing from saved audit metadata."

    preprocessing = audit.get("preprocessing") or {}
    fit_split = preprocessing.get("fit_split")
    if isinstance(fit_split, str) and fit_split.casefold() == "derived train only":
        prep_status, prep_detail = "Verified", "Imputation and scaling fitted on derived training records only."
    elif fit_split:
        prep_status, prep_detail = "Review", f"Saved preprocessing fit split: {fit_split}."
    else:
        prep_status, prep_detail = "Unavailable", "Preprocessing fit-split metadata is unavailable."

    methodology = audit.get("split_methodology") or {}
    duplicate_policy = str(methodology.get("duplicates", ""))
    if "retain every source row" in duplicate_policy.casefold():
        duplicate_status, duplicate_detail = "Verified", "Repeated records retained; no automatic deduplication."
    elif duplicate_policy:
        duplicate_status, duplicate_detail = "Review", duplicate_policy
    else:
        duplicate_status, duplicate_detail = "Unavailable", "Duplicate-retention policy is unavailable in saved metadata."

    grouping = str(methodology.get("grouping_limitation", ""))
    if grouping:
        independence_status = "Limitation"
        independence_detail = "Stable device, session, and timestamp identifiers are unavailable."
    else:
        independence_status = "Unavailable"
        independence_detail = "The saved methodology does not describe whether device/session identifiers are available."

    return [
        {"status": class_status, "title": "Class coverage", "detail": class_detail},
        {"status": overlap_status, "title": "Derived split overlap", "detail": overlap_detail},
        {"status": prep_status, "title": "Preprocessing", "detail": prep_detail},
        {"status": duplicate_status, "title": "Duplicate handling", "detail": duplicate_detail},
        {"status": independence_status, "title": "Independence limitation", "detail": independence_detail},
    ]


def flatten_audit(value: Any, prefix: str = "") -> list[dict[str, str]]:
    """Flatten nested saved metadata into a readable, lossless key/value table."""
    rows: list[dict[str, str]] = []
    if isinstance(value, dict):
        if not value:
            return [{"Field": prefix or "(root)", "Saved value": "{}"}]
        for key, child in value.items():
            rows.extend(flatten_audit(child, f"{prefix}.{key}" if prefix else str(key)))
    elif isinstance(value, (list, tuple)):
        if not value:
            return [{"Field": prefix, "Saved value": "[]"}]
        for index, child in enumerate(value):
            rows.extend(flatten_audit(child, f"{prefix}[{index}]"))
    else:
        rows.append({"Field": prefix or "(root)", "Saved value": "null" if value is None else str(value)})
    return rows

def load_model_evaluation(model_name: str) -> dict[str, Any]:
    if model_name not in MODEL_PATHS:
        raise ArtifactError(f"Unknown model evaluation: {model_name}")
    metrics_path, per_class_path, confusion_path = MODEL_PATHS[model_name]
    metrics = read_json(metrics_path)
    per_class = read_csv(per_class_path)
    confusion = read_confusion(confusion_path)
    expected_rows = int(metrics.get("evaluated_rows", -1))
    if int(confusion.sum()) != expected_rows:
        raise ArtifactError(
            f"Saved confusion matrix for {model_name} sums to {confusion.sum():,}, "
            f"but metrics report {expected_rows:,} rows.")
    required = {"label", "precision", "recall", "f1", "support"}
    if not required.issubset(per_class.columns):
        raise ArtifactError(f"Per-class metrics for {model_name} are incomplete.")
    support = pd.to_numeric(per_class["support"], errors="coerce").fillna(0).astype(int)
    if int(support.sum()) != expected_rows:
        raise ArtifactError(f"Per-class support for {model_name} does not match evaluated rows.")
    if per_class["label"].astype(str).tolist() != LABELS:
        per_class = per_class.set_index("label").reindex(LABELS).reset_index()
    return {"metrics": metrics, "per_class": per_class, "confusion": confusion}

def load_comparison_metrics() -> pd.DataFrame:
    rows = []
    for name in MODEL_PATHS:
        metrics = read_json(MODEL_PATHS[name][0])
        rows.append({"Model": name, **{k: metrics[k] for k in
            ("accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1")}})
    return pd.DataFrame(rows)

def load_federated_summary() -> dict[str, Any]:
    base = EXPERIMENTS / "federated" / "experiment"
    partition = read_json(base / "client_partitions.json")
    config = read_json(base / "configuration.json")
    validation = read_json(base / "validation_metrics.json")
    rounds = read_csv(base / "round_history.csv")
    test_metrics = read_json(base / "test_metrics.json")
    return {"clients": partition["client_distributions"], "config": config,
            "validation": validation, "test_metrics": test_metrics, "rounds": rounds}

def prepare_federated_chart_data(rounds: pd.DataFrame, selected_round: int) -> pd.DataFrame:
    """Return sorted, validated validation Macro-F1 points for the FedAvg chart."""
    required = {"round", "validation_macro_f1"}
    if not required.issubset(rounds.columns):
        raise ArtifactError("FedAvg round history is missing chart columns.")
    result = rounds[["round", "validation_macro_f1"]].copy()
    for column in result.columns:
        result[column] = pd.to_numeric(result[column], errors="raise")
    result = result.sort_values("round", kind="stable").reset_index(drop=True)
    result["selected_checkpoint"] = result["round"].eq(int(selected_round))
    if result.empty or result["round"].duplicated().any():
        raise ArtifactError("FedAvg round history is empty or contains duplicate rounds.")
    return result


def prepare_drift_chart_data(scores: pd.DataFrame, scenario: str) -> pd.DataFrame:
    """Return unrounded per-client PSI series; presentation rounding is table-only."""
    required = {"scenario", "client_id", "window_index", "max_feature_psi", "threshold"}
    if not required.issubset(scores.columns):
        raise ArtifactError("Drift scores are missing chart columns.")
    result = scores.loc[scores["scenario"].astype(str).eq(str(scenario)),
                        ["scenario", "client_id", "window_index", "max_feature_psi", "threshold"]].copy()
    for column in ("client_id", "window_index", "max_feature_psi", "threshold"):
        result[column] = pd.to_numeric(result[column], errors="raise")
    return result.sort_values(["client_id", "window_index"], kind="stable").reset_index(drop=True)


def prepare_drift_event_table(scores: pd.DataFrame) -> pd.DataFrame:
    """Create a compact, display-formatted window event table without mutating scores."""
    required = {"scenario", "client_id", "window_index", "max_feature_psi", "top_feature", "alert"}
    if not required.issubset(scores.columns):
        raise ArtifactError("Drift scores are missing event-table columns.")
    result = scores[["scenario", "client_id", "window_index", "max_feature_psi", "top_feature", "alert"]].copy()
    scenarios = {"no_drift": "No-drift control", "synthetic_shift": "Synthetic shift"}
    result["scenario"] = result["scenario"].astype(str).map(lambda value: scenarios.get(value, value))
    result["client_id"] = pd.to_numeric(result["client_id"], errors="raise").map(lambda value: f"Client {int(value)}")
    result["window_index"] = pd.to_numeric(result["window_index"], errors="raise").astype(int)
    psi = pd.to_numeric(result["max_feature_psi"], errors="raise")
    result["max_feature_psi"] = psi.map(lambda value: f"{value:.3f}")
    result["alert"] = result["alert"].map(
        lambda value: "Alert" if (value if isinstance(value, bool) else str(value).strip().casefold() in {"true", "1", "yes"}) else "Clear")
    result = result.rename(columns={"scenario": "Scenario", "client_id": "Client", "window_index": "Window",
                                    "max_feature_psi": "Maximum PSI", "top_feature": "Top feature", "alert": "Alert status"})
    return result.sort_values(["Scenario", "Client", "Window"], kind="stable").reset_index(drop=True)


def format_metadata_table(metadata: dict[str, Any]) -> pd.DataFrame:
    """Format saved nested configuration as a two-column key/value table."""
    return pd.DataFrame(flatten_audit(metadata)).rename(columns={"Field": "Setting", "Saved value": "Value"})


def load_drift_summary() -> dict[str, Any]:
    base = EXPERIMENTS / "drift"
    return {"config": read_json(base / "configuration.json"),
            "scores": read_csv(base / "drift_scores.csv"),
            "summary": read_csv(base / "scenario_summary.csv"),
            "events": read_json(base / "event_log.json").get("events", []),
            "feature_scores": read_csv(base / "feature_psi_scores.csv")}

def load_retraining_summary() -> dict[str, Any]:
    base = EXPERIMENTS / "selective_retraining"
    return {"config": read_json(base / "configuration.json"),
            "decisions": read_csv(base / "policy_decisions.csv"),
            "actions": read_csv(base / "retraining_actions.csv"),
            "summary": read_csv(base / "scenario_summary.csv")}
