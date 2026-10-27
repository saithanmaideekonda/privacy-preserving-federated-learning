"""Phase 4 PSI drift detector and training-only reproducible stream experiment."""
from __future__ import annotations
import argparse, csv, json, os, platform, time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

from src.full_dataset.federated import partition_clients

DEFAULT_PREPARED = Path(__file__).resolve().parents[2] / "experiments" / "full_dataset" / "prepared"
DEFAULT_FEDERATED = Path(__file__).resolve().parents[2] / "experiments" / "full_dataset" / "federated" / "experiment"
DEFAULT_OUTPUT = Path(__file__).resolve().parents[2] / "experiments" / "full_dataset" / "drift"
PSI_THRESHOLD = 0.25
PSI_EPSILON = 1e-6
QUANTILE_BINS = 10
REFERENCE_ROWS_PER_CLIENT = 50_000
WINDOW_ROWS = 4_096
WINDOWS_PER_CLIENT = 8
FIRST_SHIFT_WINDOW = 4
SYNTHETIC_SHIFT_SCALED_UNITS = 5.0
SYNTHETIC_AFFECTED_CLIENT = 2
SYNTHETIC_FEATURE = "Rate"
CHUNK_ROWS = 50_000
SEED = 20261026


def psi_from_counts(reference_counts, current_counts, epsilon=PSI_EPSILON):
    """Calculate PSI using proportions floored at epsilon and renormalized."""
    ref = np.asarray(reference_counts, dtype=np.float64)
    cur = np.asarray(current_counts, dtype=np.float64)
    if ref.ndim != 1 or cur.ndim != 1 or ref.shape != cur.shape or ref.size == 0:
        raise ValueError("reference and current counts must be nonempty vectors of equal length")
    if not np.isfinite(ref).all() or not np.isfinite(cur).all() or (ref < 0).any() or (cur < 0).any():
        raise ValueError("counts must be finite and nonnegative")
    if ref.sum() <= 0 or cur.sum() <= 0:
        raise ValueError("both distributions must have positive support")
    if not 0 < epsilon < 1:
        raise ValueError("epsilon must be in (0, 1)")
    expected = np.maximum(ref / ref.sum(), epsilon)
    actual = np.maximum(cur / cur.sum(), epsilon)
    expected /= expected.sum()
    actual /= actual.sum()
    return float(np.sum((actual - expected) * np.log(actual / expected)))


def fit_feature_reference(values, n_bins=QUANTILE_BINS):
    """Fit one feature's empirical quantile bin edges and counts on reference rows."""
    x = np.asarray(values, dtype=np.float64).reshape(-1)
    x = x[np.isfinite(x)]
    if x.size == 0:
        raise ValueError("reference feature has no finite observations")
    if n_bins < 2:
        raise ValueError("n_bins must be at least two")
    quantiles = np.arange(1, n_bins, dtype=np.float64) / n_bins
    cuts = np.unique(np.quantile(x, quantiles, method="linear"))
    cuts = cuts[(cuts > x.min()) & (cuts < x.max())]
    # Duplicate and boundary quantiles collapse; a constant feature has one bin.
    counts = np.bincount(np.searchsorted(cuts, x, side="right"), minlength=len(cuts) + 1)
    return {"edges": cuts.astype(np.float64), "counts": counts.astype(np.int64),
            "finite_reference_rows": int(x.size), "effective_bins": int(len(cuts) + 1)}


def score_feature(reference, values, epsilon=PSI_EPSILON):
    """Score one current feature window using fixed training-reference edges."""
    x = np.asarray(values, dtype=np.float64).reshape(-1)
    x = x[np.isfinite(x)]
    if x.size == 0:
        raise ValueError("current feature window has no finite observations")
    edges = np.asarray(reference["edges"], dtype=np.float64)
    counts = np.bincount(np.searchsorted(edges, x, side="right"), minlength=len(edges) + 1)
    return psi_from_counts(reference["counts"], counts, epsilon), int(x.size), counts


def fit_reference_matrix(reference_matrix, feature_names, n_bins=QUANTILE_BINS):
    matrix = np.asarray(reference_matrix, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[1] != len(feature_names) or matrix.shape[0] == 0:
        raise ValueError("reference matrix shape does not match feature names")
    refs = {}
    for idx, name in enumerate(feature_names):
        refs[name] = fit_feature_reference(matrix[:, idx], n_bins)
    return refs


def score_window(reference_distributions, window, feature_names, threshold=PSI_THRESHOLD):
    """Return feature PSI rows and a window alert based on maximum feature PSI."""
    data = np.asarray(window, dtype=np.float32)
    if data.ndim != 2 or data.shape[1] != len(feature_names) or data.shape[0] == 0:
        raise ValueError("window shape does not match feature names")
    scores = []
    for idx, name in enumerate(feature_names):
        psi, finite_rows, current_counts = score_feature(reference_distributions[name], data[:, idx])
        scores.append({"feature": name, "psi": psi, "threshold": float(threshold),
                       "alert": bool(psi >= threshold), "finite_rows": finite_rows,
                       "current_bin_counts": current_counts.tolist()})
    top = max(scores, key=lambda row: row["psi"])
    return {"window_psi": float(top["psi"]), "top_feature": top["feature"],
            "threshold": float(threshold), "alert": bool(top["psi"] >= threshold),
            "feature_scores": scores}


def apply_synthetic_shift(window, client_id, window_index, feature_index,
                          affected_client=SYNTHETIC_AFFECTED_CLIENT,
                          first_shift_window=FIRST_SHIFT_WINDOW,
                          shift=SYNTHETIC_SHIFT_SCALED_UNITS):
    """Return a copied window with a clearly synthetic additive shift when eligible."""
    result = np.asarray(window, dtype=np.float32).copy()
    shifted = client_id == affected_client and window_index >= first_shift_window
    if shifted:
        result[:, feature_index] += np.float32(shift)
    return result, bool(shifted)


def _load_training_only(prepared_dir):
    prepared_dir = Path(prepared_dir).resolve()
    metadata = json.loads((prepared_dir / "phase1_metadata.json").read_text(encoding="utf-8"))
    prep = json.loads((prepared_dir / "preprocessing_metadata.json").read_text(encoding="utf-8"))
    features = prep["feature_columns_retained"]
    x = np.load(prepared_dir / "arrays" / "X_train.npy", mmap_mode="r")
    y = np.load(prepared_dir / "arrays" / "y_train.npy", mmap_mode="r")
    if x.dtype != np.float32 or y.dtype != np.int32 or x.ndim != 2 or y.shape != (x.shape[0],):
        raise ValueError("training arrays have unexpected dtype or shape")
    if x.shape[1] != len(features) or x.shape[0] != int(metadata["derived_rows"]["train"]):
        raise ValueError("training arrays disagree with Phase 1 metadata")
    return prepared_dir, metadata, prep, x, y


def _write_csv(path, rows, columns):
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def _sample_rows_streaming(x, indices, chunk_rows=CHUNK_ROWS):
    """Gather sorted selected indices through sequential memmap chunks."""
    chosen = np.sort(np.asarray(indices, dtype=np.int64))
    result = np.empty((len(chosen), x.shape[1]), dtype=np.float32)
    out_cursor = 0
    for start in range(0, x.shape[0], chunk_rows):
        end = min(start + chunk_rows, x.shape[0])
        left = np.searchsorted(chosen, start, side="left")
        right = np.searchsorted(chosen, end, side="left")
        if right > left:
            result[out_cursor:out_cursor + (right - left)] = x[chosen[left:right]]
            out_cursor += right - left
    if out_cursor != len(chosen):
        raise IndexError("sample index outside training array")
    return result


def run_drift(prepared_dir=DEFAULT_PREPARED, federated_dir=DEFAULT_FEDERATED,
              output_dir=DEFAULT_OUTPUT, seed=SEED, n_bins=QUANTILE_BINS,
              threshold=PSI_THRESHOLD, reference_rows=REFERENCE_ROWS_PER_CLIENT,
              window_rows=WINDOW_ROWS, n_windows=WINDOWS_PER_CLIENT,
              chunk_rows=CHUNK_ROWS):
    prepared_dir, federated_dir, output_dir = map(lambda p: Path(p).resolve(),
                                                   (prepared_dir, federated_dir, output_dir))
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing drift artifacts: {output_dir}")
    config_path = federated_dir / "configuration.json"
    partition_path = federated_dir / "client_partitions.json"
    if not config_path.is_file() or not partition_path.is_file():
        raise FileNotFoundError("Phase 3 configuration or client partitions are missing")
    federated_config = json.loads(config_path.read_text(encoding="utf-8"))
    saved_partition = json.loads(partition_path.read_text(encoding="utf-8"))
    prepared_dir, phase1, prep, x_train, y_train = _load_training_only(prepared_dir)
    feature_names = prep["feature_columns_retained"]
    if SYNTHETIC_FEATURE not in feature_names:
        raise ValueError(f"Synthetic feature {SYNTHETIC_FEATURE!r} is not in the retained feature mapping")
    synthetic_feature_index = feature_names.index(SYNTHETIC_FEATURE)
    client_indices = partition_clients(y_train, n_clients=3,
                                       alpha=float(federated_config["dirichlet_alpha"]),
                                       seed=int(federated_config["partition_seed"]))
    saved_clients = saved_partition["client_distributions"]
    if [len(a) for a in client_indices] != [int(c["sample_count"]) for c in saved_clients]:
        raise ValueError("regenerated partitions do not match Phase 3 saved client counts")
    required = reference_rows + window_rows * n_windows
    if any(len(indices) < required for indices in client_indices):
        raise ValueError("client does not have enough local training rows for configured reference and stream")
    started = time.perf_counter()
    build_dir = output_dir.with_name(f".{output_dir.name}_building")
    if build_dir.exists():
        raise FileExistsError(f"Refusing to overwrite incomplete build directory: {build_dir}")
    build_dir.mkdir(parents=True)

    rng = np.random.default_rng(seed)
    references = {}
    streams = {}
    for client_id, indices in enumerate(client_indices, start=1):
        # Client index arrays are reproducibly shuffled by Phase 3 partitioning.
        # Reference and monitored rows are disjoint slices of that local partition.
        local = indices.copy()
        rng.shuffle(local)
        ref_idx = local[:reference_rows]
        stream_idx = local[reference_rows:required]
        ref_matrix = _sample_rows_streaming(x_train, ref_idx, chunk_rows)
        references[client_id] = fit_reference_matrix(ref_matrix, feature_names, n_bins)
        del ref_matrix
        windows = []
        for wi in range(n_windows):
            start = wi * window_rows
            windows.append(_sample_rows_streaming(x_train, stream_idx[start:start + window_rows], chunk_rows))
        streams[client_id] = windows

    rows, feature_rows = [], []
    expected_shift_windows = []
    for scenario in ("no_drift", "synthetic_shift"):
        for client_id in range(1, 4):
            for window_index, source_window in enumerate(streams[client_id], start=1):
                if scenario == "synthetic_shift":
                    window, shifted = apply_synthetic_shift(source_window, client_id, window_index,
                                                            synthetic_feature_index)
                else:
                    window, shifted = source_window, False
                if scenario == "synthetic_shift" and shifted:
                    expected_shift_windows.append((client_id, window_index))
                score = score_window(references[client_id], window, feature_names, threshold)
                summary = {"scenario": scenario, "client_id": client_id,
                           "window_index": window_index, "rows": int(window.shape[0]),
                           "max_feature_psi": score["window_psi"], "top_feature": score["top_feature"],
                           "threshold": float(threshold), "alert": score["alert"],
                           "synthetic_shift_applied": shifted,
                           "shifted_feature": SYNTHETIC_FEATURE if shifted else "",
                           "injected_shift_scaled_units": SYNTHETIC_SHIFT_SCALED_UNITS if shifted else 0.0,
                           "source_split": "training"}
                rows.append(summary)
                for feature_score in score["feature_scores"]:
                    feature_rows.append({"scenario": scenario, "client_id": client_id,
                                         "window_index": window_index, "feature": feature_score["feature"],
                                         "psi": feature_score["psi"], "threshold": threshold,
                                         "alert": feature_score["alert"],
                                         "synthetic_shift_applied": shifted,
                                         "current_bin_counts": json.dumps(feature_score["current_bin_counts"])})

    no_drift_rows = [r for r in rows if r["scenario"] == "no_drift"]
    synth_rows = [r for r in rows if r["scenario"] == "synthetic_shift"]
    shift_rows = [r for r in synth_rows if r["synthetic_shift_applied"]]
    synth_control_rows = [r for r in synth_rows if not r["synthetic_shift_applied"]]
    first_detected = next((r["window_index"] for r in shift_rows if r["alert"]), None)
    delay = None if first_detected is None else int(first_detected - FIRST_SHIFT_WINDOW)
    summaries = [
        {"scenario": "no_drift", "windows_checked": len(no_drift_rows),
         "alerts": sum(r["alert"] for r in no_drift_rows),
         "false_alarm_windows": sum(r["alert"] for r in no_drift_rows),
         "shift_windows": 0, "detected_shift_windows": 0, "missed_shift_windows": 0,
         "false_alarm_rate": float(np.mean([r["alert"] for r in no_drift_rows])),
         "detection_delay_windows": None},
        {"scenario": "synthetic_shift", "windows_checked": len(synth_rows),
         "alerts": sum(r["alert"] for r in synth_rows),
         "false_alarm_windows": sum(r["alert"] for r in synth_control_rows),
         "shift_windows": len(shift_rows),
         "detected_shift_windows": sum(r["alert"] for r in shift_rows),
         "missed_shift_windows": sum(not r["alert"] for r in shift_rows),
         "false_alarm_rate": float(np.mean([r["alert"] for r in synth_control_rows])) if synth_control_rows else 0.0,
         "detection_delay_windows": delay},
    ]
    alerts = [{**r, "event_type": "distribution_drift_alert"} for r in rows if r["alert"]]
    created = datetime.now(timezone.utc).isoformat()
    config = {"method": "feature-wise Population Stability Index", "reference_source": "Phase 1 prepared training rows only",
              "client_baseline": "separate reference histogram for each Phase 3 simulated client",
              "binning": "10 empirical quantile bins per client/feature; duplicate quantile cut-points collapse; effective bin count may be lower",
              "psi_formula": "sum((actual_prop-reference_prop)*ln(actual_prop/reference_prop))",
              "empty_bins": f"floor each bin proportion at epsilon={PSI_EPSILON:g}, then renormalize before logarithm",
              "threshold": float(threshold),
              "threshold_rationale": "Predeclared PSI 0.25 operational heuristic for a material distribution shift; not fitted or tuned on validation/test, and not a statistical guarantee.",
              "alert_rule": "window alert if any feature PSI is greater than or equal to threshold; window score is maximum feature PSI",
              "quantile_bins_requested": n_bins, "reference_rows_per_client": reference_rows,
              "window_rows": window_rows, "windows_per_client": n_windows,
              "no_drift_stream": "independent rows from the same simulated client's training partition, disjoint from its reference rows",
              "synthetic_shift": {"affected_client": SYNTHETIC_AFFECTED_CLIENT, "feature": SYNTHETIC_FEATURE,
                                  "first_window_index": FIRST_SHIFT_WINDOW, "shift_in_standardized_feature_units": SYNTHETIC_SHIFT_SCALED_UNITS,
                                  "source": "training rows copied and modified in memory; not natural traffic drift"},
              "seed": seed, "chunk_rows": chunk_rows, "no_test_data_or_labels_used": True}
    references_json = {}
    for client_id, features in references.items():
        references_json[str(client_id)] = {name: {"edges": value["edges"].tolist(),
            "reference_counts": value["counts"].tolist(), "finite_reference_rows": value["finite_reference_rows"],
            "effective_bins": value["effective_bins"]} for name, value in features.items()}
    _write_json(build_dir / "configuration.json", {"created_utc": created, **config})
    _write_json(build_dir / "reference_distributions.json", references_json)
    _write_json(build_dir / "event_log.json", {"created_utc": created, "events": alerts})
    _write_csv(build_dir / "drift_scores.csv", rows,
               ["scenario", "client_id", "window_index", "rows", "max_feature_psi", "top_feature",
                "threshold", "alert", "synthetic_shift_applied", "shifted_feature",
                "injected_shift_scaled_units", "source_split"])
    _write_csv(build_dir / "feature_psi_scores.csv", feature_rows,
               ["scenario", "client_id", "window_index", "feature", "psi", "threshold", "alert",
                "synthetic_shift_applied", "current_bin_counts"])
    _write_csv(build_dir / "scenario_summary.csv", summaries,
               ["scenario", "windows_checked", "alerts", "false_alarm_windows", "shift_windows",
                "detected_shift_windows", "missed_shift_windows", "false_alarm_rate", "detection_delay_windows"])
    metadata = {"created_utc": created, "status": "completed", "configuration": config,
                "software": {"python": platform.python_version(), "numpy": np.__version__},
                "dataset": {"prepared_training_rows": int(x_train.shape[0]),
                            "features": int(x_train.shape[1]), "validation_used": False, "test_used": False},
                "client_partitions": [{"client_id": c["client_id"], "sample_count": c["sample_count"]}
                                      for c in saved_clients],
                "scenario_summaries": summaries,
                "runtime_seconds": time.perf_counter() - started,
                "runtime_scope": "after prepared arrays are mapped and Phase 3 client partitions are regenerated/verified; includes reference fitting, stream scoring, and artifact writes, but excludes process startup and input/partition loading",
                "memory_strategy": "training arrays memory-mapped; reference/windows gathered in bounded row batches; one client reference and one current window held per scoring operation",
                "interpretation": "A PSI alert indicates feature-distribution change only; it is not evidence or confirmation of a cyberattack."}
    _write_json(build_dir / "run_metadata.json", metadata)
    report = _make_report(metadata, summaries, rows, references)
    (build_dir / "report.md").write_text(report, encoding="utf-8")
    os.replace(build_dir, output_dir)
    return metadata


def _make_report(metadata, summaries, scores, references):
    lines = ["# Phase 4 — PSI-based drift detection", "",
             f"**Created:** {metadata['created_utc']}  ",
             f"**Input:** {metadata['dataset']['prepared_training_rows']:,} prepared training rows; {metadata['dataset']['features']} features. Validation and test were not used.  ",
             f"**Runtime:** {metadata['runtime_seconds']:.2f} seconds after input mapping and partition verification (the end-to-end command also includes startup/input loading).", "",
             "## Detector", "",
             "Each of the three Phase 3 simulated clients has its own reference distributions, fitted from 50,000 rows in its training partition. The 10 requested quantile bins are fitted feature by feature; repeated quantile cut-points collapse, so tied/discrete features can have fewer bins. Current windows use those fixed training-derived edges. Empty-bin proportions are floored at 1e-6 and renormalized before the PSI logarithm. A window alerts when its maximum feature PSI is at least 0.25.", "",
             "The 0.25 cutoff is a predeclared operational heuristic, not a threshold tuned on held-out data or a statistical guarantee. No validation or test data was used to establish reference distributions or thresholds.", "",
             "## Scenarios and results", "",
             "No-drift windows contain unmodified training rows disjoint from each client’s reference sample. The synthetic scenario replays the same streams and adds +5.0 standardized units to `Rate` for Client 2 starting at window 4. Clients 1 and 3 are unshifted controls. This is a deliberately injected synthetic distribution change, not observed temporal drift.", "",
             "| Scenario | Windows | Alerts | False-alarm windows | Shift windows | Detected shift windows | Missed shifts | Delay (windows) | False-alarm rate |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for s in summaries:
        delay = "—" if s["detection_delay_windows"] is None else str(s["detection_delay_windows"])
        lines.append(f"| {s['scenario']} | {s['windows_checked']} | {s['alerts']} | {s['false_alarm_windows']} | {s['shift_windows']} | {s['detected_shift_windows']} | {s['missed_shift_windows']} | {delay} | {s['false_alarm_rate']:.3f} |")
    lines += ["", "Detection delay is measured from the injected shift’s first window (window 4); a value of zero means the first shifted window alerted. False alarms are alerts in windows without an injected shift. These rates characterize this finite synthetic replay only.", "",
              "## Client-specific alert windows", "",
              "| Scenario | Client | Window | Max PSI | Threshold | Top feature | Injected shift | Alert |", "|---|---:|---:|---:|---:|---|---|---|"]
    for r in scores:
        if r["alert"] or r["synthetic_shift_applied"]:
            lines.append(f"| {r['scenario']} | {r['client_id']} | {r['window_index']} | {r['max_feature_psi']:.6f} | {r['threshold']:.2f} | {r['top_feature']} | {'yes' if r['synthetic_shift_applied'] else 'no'} | {'yes' if r['alert'] else 'no'} |")
    lines += ["", "Full window scores are in `drift_scores.csv`; all per-feature scores and bin counts are in `feature_psi_scores.csv`. Reference edges and histograms are in `reference_distributions.json`; triggered events are in `event_log.json`.", "",
              "## Interpretation and limitations", "",
              "**A detected distribution change is not proof of a cyberattack.** PSI detects changes in feature frequencies; benign traffic changes, sensor changes, or preprocessing differences can also trigger it.",
              "- This is a synthetic train-derived replay. The dataset has no device/session/time IDs to establish a natural temporal stream, so the experiment does not validate real-world drift.",
              "- The PSI threshold is a heuristic. The maximum across 44 feature tests can create false alarms; observed false-alarm windows are reported rather than hidden.",
              "- Per-client references account for client-specific baseline distributions but can be weak for rare classes and small windows.",
              "- The clients remain a single-computer simulation; this phase does not implement retraining or provide a privacy guarantee.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, default=DEFAULT_PREPARED)
    parser.add_argument("--federated", type=Path, default=DEFAULT_FEDERATED)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--bins", type=int, default=QUANTILE_BINS)
    parser.add_argument("--threshold", type=float, default=PSI_THRESHOLD)
    parser.add_argument("--reference-rows", type=int, default=REFERENCE_ROWS_PER_CLIENT)
    parser.add_argument("--window-rows", type=int, default=WINDOW_ROWS)
    parser.add_argument("--windows", type=int, default=WINDOWS_PER_CLIENT)
    parser.add_argument("--chunk-rows", type=int, default=CHUNK_ROWS)
    args = parser.parse_args()
    result = run_drift(args.prepared, args.federated, args.output, args.seed, args.bins,
                       args.threshold, args.reference_rows, args.window_rows, args.windows, args.chunk_rows)
    print(json.dumps({"status": result["status"], "runtime_seconds": result["runtime_seconds"],
                      "scenario_summaries": result["scenario_summaries"]}, indent=2))

if __name__ == "__main__":
    main()

