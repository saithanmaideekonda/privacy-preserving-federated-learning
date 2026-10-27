"""Verified inference for user-supplied IoT flow feature records.

Raw inputs use the saved 46-feature schema. Training-fitted metadata produces
the model's 44 standardized float32 inputs. Missing/non-finite supplied values
are rejected; the UI never silently imputes user records.
"""
from __future__ import annotations
import hashlib
import io
from functools import lru_cache
from pathlib import Path
from typing import Any
import joblib
import numpy as np
import pandas as pd
from dashboard_data import EXPERIMENTS, read_json

ROBUSTNESS = EXPERIMENTS / "model_improvement" / "robustness_evaluation"
CHECKPOINT = ROBUSTNESS / "frozen_candidate.joblib"
FROZEN_ARTIFACTS = ROBUSTNESS / "frozen_artifacts"
MANIFEST_PATH = ROBUSTNESS / "frozen_artifact_manifest.json"
SELECTION_PATH = ROBUSTNESS / "selection_frozen.json"
TRAIN_FEATURES = EXPERIMENTS / "prepared" / "arrays" / "X_train.npy"

class InferenceError(ValueError):
    """Invalid user record or unavailable/incompatible saved inference artifacts."""

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def verify_frozen_artifacts() -> dict[str, Any]:
    manifest = read_json(MANIFEST_PATH)
    selection = read_json(SELECTION_PATH)
    ckpt_hash = sha256_file(CHECKPOINT)
    expected = manifest.get("checkpoint", {}).get("frozen_sha256")
    if not expected or ckpt_hash != expected or ckpt_hash != selection.get("checkpoint_sha256"):
        raise InferenceError("Frozen candidate checkpoint hash does not match saved metadata.")
    hashes = {"checkpoint": ckpt_hash}
    for name, entry in manifest.get("artifacts", {}).items():
        path = FROZEN_ARTIFACTS / name
        if not path.is_file() or sha256_file(path) != entry.get("sha256"):
            raise InferenceError(f"Frozen preprocessing artifact failed SHA-256 verification: {name}")
        hashes[name] = entry["sha256"]
    return {"manifest": manifest, "selection": selection, "hashes": hashes}

@lru_cache(maxsize=1)
def load_predictor() -> "TrafficPredictor":
    """Load the trusted local model and preprocessing once per process."""
    verified = verify_frozen_artifacts()
    model = joblib.load(CHECKPOINT)
    preprocessing = joblib.load(FROZEN_ARTIFACTS / "preprocessing.joblib")
    preprocessing_metadata = read_json(FROZEN_ARTIFACTS / "preprocessing_metadata.json")
    preprocessing["imputer_medians"] = preprocessing_metadata["imputer_medians"]
    if not hasattr(model, "predict_proba") or not hasattr(model, "classes_"):
        raise InferenceError("Frozen checkpoint does not expose class probabilities.")
    predictor = TrafficPredictor(model, preprocessing, verified["hashes"])
    if len(predictor.input_features) != 46:
        raise InferenceError("Frozen preprocessing metadata does not define 46 input features.")
    if len(predictor.retained_features) != int(model.n_features_in_):
        raise InferenceError("Model input width does not match frozen retained feature order.")
    return predictor

class TrafficPredictor:
    def __init__(self, model: Any, preprocessing: dict[str, Any], hashes: dict[str, str]):
        self.model = model
        self.preprocessing = preprocessing
        self.hashes = dict(hashes)
        self.input_features = list(preprocessing["feature_columns_input"])
        self.retained_features = list(preprocessing["feature_columns_retained"])
        self.retained_indices = np.asarray(preprocessing["retained_feature_indices"], dtype=np.int64)
        self.mean = np.asarray(preprocessing["scaler_mean"], dtype=np.float64)
        self.scale = np.asarray(preprocessing["scaler_scale"], dtype=np.float64)
        self.class_to_id = dict(preprocessing["class_to_id"])
        self.id_to_class = {int(k): str(v) for k, v in preprocessing["id_to_class"].items()}
        if len(self.input_features) != 46 or len(self.retained_indices) != 44:
            raise InferenceError("Saved feature schema is incompatible with the 46-to-44 transform.")
        if self.mean.shape != (44,) or self.scale.shape != (44,) or np.any(self.scale == 0):
            raise InferenceError("Saved scaler parameters have invalid shapes or values.")
        if len(self.class_to_id) != 34 or len(self.id_to_class) != 34:
            raise InferenceError("Saved class mapping does not contain the expected 34 labels.")
        expected_inverse = {int(v): str(k) for k, v in self.class_to_id.items()}
        if expected_inverse != self.id_to_class:
            raise InferenceError("Saved forward and inverse class mappings are inconsistent.")
        if preprocessing.get("fit_split") not in (None, "train", "training", "derived train only"):
            raise InferenceError("Preprocessing metadata was not fitted on training data.")

    def validate_frame(self, frame: pd.DataFrame, *, max_rows: int = 2000) -> pd.DataFrame:
        if not isinstance(frame, pd.DataFrame):
            raise InferenceError("Input must be a table with numeric feature columns.")
        if frame.empty:
            raise InferenceError("Provide at least one traffic feature row.")
        if len(frame) > max_rows:
            raise InferenceError(f"Batch has {len(frame):,} rows; the dashboard limit is {max_rows:,}.")
        columns = list(frame.columns)
        if "label" in columns:
            raise InferenceError("Do not include a label column; the dashboard predicts labels.")
        missing = [name for name in self.input_features if name not in columns]
        extra = [name for name in columns if name not in self.input_features]
        if missing:
            preview = ", ".join(missing[:5]) + ("..." if len(missing) > 5 else "")
            raise InferenceError(f"Missing {len(missing)} required feature column(s): {preview}")
        if extra:
            preview = ", ".join(map(str, extra[:5])) + ("..." if len(extra) > 5 else "")
            raise InferenceError(f"Unexpected column(s): {preview}. Remove extras; exact schema is required.")
        if len(columns) != len(set(columns)):
            raise InferenceError("Duplicate feature column names are not allowed.")
        ordered = frame.loc[:, self.input_features]
        try:
            numeric = ordered.apply(pd.to_numeric, errors="raise")
            values = numeric.to_numpy(dtype=np.float64, na_value=np.nan, copy=True)
        except (ValueError, TypeError) as exc:
            raise InferenceError(f"All 46 features must be numeric: {exc}") from exc
        if np.isnan(values).any():
            bad = [self.input_features[i] for i in np.flatnonzero(np.isnan(values).any(axis=0))]
            raise InferenceError("Missing values found in: " + ", ".join(bad[:6]))
        if not np.isfinite(values).all():
            bad = [self.input_features[i] for i in np.flatnonzero(~np.isfinite(values).all(axis=0))]
            raise InferenceError("Infinite or invalid values found in: " + ", ".join(bad[:6]))
        return pd.DataFrame(values, columns=self.input_features)

    def transform(self, frame: pd.DataFrame, *, max_rows: int = 2000) -> np.ndarray:
        ordered = self.validate_frame(frame, max_rows=max_rows)
        values = ordered.to_numpy(dtype=np.float64, copy=False)
        scaled = (values[:, self.retained_indices] - self.mean) / self.scale
        if not np.isfinite(scaled).all():
            raise InferenceError("Preprocessing generated a non-finite model input.")
        return np.asarray(scaled, dtype=np.float32, order="C")

    def predict(self, frame: pd.DataFrame, *, max_rows: int = 2000) -> dict[str, Any]:
        x = self.transform(frame, max_rows=max_rows)
        probabilities = np.asarray(self.model.predict_proba(x), dtype=np.float64)
        ids = [int(i) for i in self.model.classes_]
        if sorted(ids) != list(range(34)):
            raise InferenceError("Frozen model output classes do not match the 34 saved class IDs.")
        if probabilities.shape != (len(frame), len(ids)) or not np.isfinite(probabilities).all():
            raise InferenceError("Model returned an invalid probability array.")
        if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-4):
            raise InferenceError("Model probability rows do not sum to one.")
        labels = [self.id_to_class.get(i) for i in ids]
        if any(label is None for label in labels):
            raise InferenceError("Model class IDs do not match the saved label mapping.")
        best = probabilities.argmax(axis=1)
        predicted = [labels[i] for i in best]
        top3 = []
        for row in probabilities:
            ix = np.argsort(row)[::-1][:3]
            top3.append([{"label": labels[i], "score": float(row[i])} for i in ix])
        return {"labels": predicted, "status": [traffic_status(x) for x in predicted],
                "probabilities": probabilities, "class_labels": labels, "top3": top3,
                "features": x}

    def demo_records(self, *, seed: int = 20261026, count: int = 3) -> list[pd.Series]:
        """Reconstruct illustrative inputs from X_train only; never opens labels."""
        if not TRAIN_FEATURES.is_file():
            raise InferenceError("Training feature array for illustrative examples is missing.")
        try:
            training = np.load(TRAIN_FEATURES, mmap_mode="r", allow_pickle=False)
        except (OSError, ValueError) as exc:
            raise InferenceError(f"Could not open training-only examples: {exc}") from exc
        if training.ndim != 2 or training.shape[1] != 44 or training.shape[0] < count:
            raise InferenceError("Prepared training feature array has an unexpected shape.")
        rng = np.random.default_rng(seed)
        indices = np.sort(rng.choice(training.shape[0], size=count, replace=False))
        medians = np.asarray(self.preprocessing["imputer_medians"], dtype=np.float64)
        if medians.shape != (46,):
            raise InferenceError("Saved training imputation metadata is incompatible.")
        rows = []
        for index in indices:
            raw = medians.copy()
            scaled = np.asarray(training[int(index)], dtype=np.float64)
            raw[self.retained_indices] = scaled * self.scale + self.mean
            rows.append(pd.Series(raw, index=self.input_features, dtype="float64"))
        return rows

def parse_csv_bytes(payload: bytes, predictor: TrafficPredictor, *, max_bytes: int = 5 * 1024 * 1024, max_rows: int = 2000) -> pd.DataFrame:
    """Parse and validate a user CSV without retaining it or changing the model."""
    if not isinstance(payload, (bytes, bytearray)):
        raise InferenceError("CSV input must be uploaded as bytes.")
    if len(payload) > max_bytes:
        raise InferenceError("CSV exceeds the 5 MB dashboard limit.")
    try:
        frame = pd.read_csv(io.BytesIO(payload))
    except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeError, ValueError) as exc:
        raise InferenceError(f"CSV could not be parsed: {exc}") from exc
    predictor.validate_frame(frame, max_rows=max_rows)
    return frame.loc[:, predictor.input_features]


def traffic_status(label: str) -> str:
    return "NORMAL TRAFFIC" if label == "BenignTraffic" else "MALICIOUS / ATTACK-LIKE TRAFFIC"
