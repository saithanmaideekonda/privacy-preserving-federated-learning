from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import sqlite3
import time
from collections import Counter
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler

SPLIT_NAMES = ("train", "validation", "test")
SPLIT_TO_ID = {name: i for i, name in enumerate(SPLIT_NAMES)}
ID_TO_SPLIT = {i: name for name, i in SPLIT_TO_ID.items()}
DEFAULT_DATASET_ROOT = Path(r"C:\Users\saith\Downloads\archive (1)\CICIoT23")
EXPECTED_FEATURES = 46
CHUNK_ROWS = 20_000
WRITE_BUFFER_ROWS = 10_000


def _stable_tie_index(digest: bytes, options: list[int], seed: int) -> int:
    payload = b"ciciot23-phase1-feature-group-v1:" + str(seed).encode("ascii") + b":" + digest
    number = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    return options[number % len(options)]


def choose_group_split(counts: tuple[int, int, int] | list[int], digest: bytes, seed: int = 20261026) -> int:
    """Assign one feature group by majority original split; resolve ties reproducibly."""
    if len(counts) != 3 or any(int(value) < 0 for value in counts):
        raise ValueError("counts must contain three non-negative split counts")
    largest = max(counts)
    if largest == 0:
        raise ValueError("a feature group must occur in a source split")
    winners = [i for i, value in enumerate(counts) if value == largest]
    return winners[0] if len(winners) == 1 else _stable_tie_index(digest, winners, seed)


def load_group_assignment(audit_db: Path, seed: int = 20261026) -> tuple[dict[bytes, int], set[bytes], dict]:
    """Load shared-group destinations and repeated/shared candidates for label auditing."""
    if not audit_db.is_file():
        raise FileNotFoundError(f"Fingerprint audit database not found: {audit_db}")
    uri = f"file:{audit_db.resolve().as_posix()}?mode=ro"
    connection = sqlite3.connect(uri, uri=True)
    assignments: dict[bytes, int] = {}
    candidates: set[bytes] = set()
    ties = 0
    moved_rows = Counter()
    try:
        candidates = {
            row[0] for row in connection.execute(
                """SELECT digest FROM fingerprints
                   WHERE kind=1 AND (train_count>1 OR validation_count>1 OR test_count>1
                   OR (train_count>0)+(validation_count>0)+(test_count>0)>1)"""
            )
        }
        for digest, train_n, validation_n, test_n in connection.execute(
            """SELECT digest,train_count,validation_count,test_count FROM fingerprints
               WHERE kind=1 AND ((train_count>0)+(validation_count>0)+(test_count>0)>1)"""
        ):
            counts = (train_n, validation_n, test_n)
            target = choose_group_split(counts, digest, seed)
            assignments[digest] = target
            if sum(count == max(counts) for count in counts) > 1:
                ties += 1
            for source_id, count in enumerate(counts):
                if source_id != target:
                    moved_rows[SPLIT_NAMES[source_id]] += count
    finally:
        connection.close()
    return assignments, candidates, {
        "shared_feature_groups": len(assignments),
        "repeated_or_shared_groups_for_label_audit": len(candidates),
        "tie_groups": ties,
        "occurrences_reassigned_from_source": dict(moved_rows),
        "seed": seed,
        "assignment_rule": "Assign each 46-feature SHA-256 group to the original split with the most occurrences; ties use a stable seeded SHA-256 choice.",
    }


def _read_header(path: Path) -> list[str]:
    with path.open("rb") as file:
        raw = file.readline()
    if raw.startswith(b"\xef\xbb\xbf"):
        raw = raw[3:]
    return [item.strip().decode("utf-8-sig") for item in raw.rstrip(b"\r\n").split(b",")]


def _parse_record(row: bytes, expected_fields: int) -> tuple[bytes, str, np.ndarray, int, int, int]:
    fields = row.split(b",")
    if len(fields) != expected_fields:
        raise ValueError(f"Expected {expected_fields} fields, found {len(fields)}")
    feature_bytes = row.rsplit(b",", 1)[0]
    try:
        label = fields[-1].decode("utf-8").strip()
        feature_text = feature_bytes.decode("ascii")
    except UnicodeDecodeError as error:
        raise ValueError("Feature or label text has invalid encoding") from error
    if not label:
        raise ValueError("Empty label")
    values = np.fromstring(feature_text, sep=",", dtype=np.float32)
    invalid = 0
    if values.size != expected_fields - 1:
        values = np.empty(expected_fields - 1, dtype=np.float32)
        for index, token in enumerate(fields[:-1]):
            if token.strip() == b"":
                values[index] = np.nan
            else:
                try:
                    values[index] = float(token)
                except (ValueError, OverflowError):
                    values[index] = np.nan
                    invalid += 1
    missing = int(np.isnan(values).sum())
    infinite = int(np.isinf(values).sum())
    if infinite:
        values[~np.isfinite(values)] = np.nan
    return hashlib.sha256(feature_bytes).digest(), label, values, missing, infinite, invalid


def inspect_rows(
    source_paths: dict[str, Path], assignments: dict[bytes, int], candidates: set[bytes]
) -> tuple[list[str], dict]:
    source_rows = Counter()
    derived_rows = Counter()
    source_to_derived = {split: Counter() for split in SPLIT_NAMES}
    class_counts = {split: Counter() for split in SPLIT_NAMES}
    source_class_counts = {split: Counter() for split in SPLIT_NAMES}
    quality = {split: {
        "missing_feature_values": 0, "infinite_feature_values": 0,
        "non_numeric_feature_tokens": 0, "malformed_rows": 0, "empty_rows": 0,
    } for split in SPLIT_NAMES}
    first_labels: dict[bytes, str] = {}
    conflict_groups: set[bytes] = set()
    expected_header = None
    started = time.perf_counter()

    for source_split in SPLIT_NAMES:
        path = Path(source_paths[source_split])
        if not path.is_file():
            raise FileNotFoundError(f"Required source split not found: {path}")
        header = _read_header(path)
        if expected_header is None:
            expected_header = header
        elif header != expected_header:
            raise ValueError(f"Column names/order differ in {path}")
        if len(header) != EXPECTED_FEATURES + 1 or header[-1] != "label":
            raise ValueError(f"Expected {EXPECTED_FEATURES} features and final label in {path}")

        with path.open("rb") as file:
            file.readline()
            for line_number, line in enumerate(file, start=2):
                row = line.rstrip(b"\r\n")
                if not row:
                    quality[source_split]["empty_rows"] += 1
                    continue
                try:
                    digest, label, _, missing, infinite, invalid = _parse_record(row, len(header))
                except (ValueError, UnicodeDecodeError) as error:
                    quality[source_split]["malformed_rows"] += 1
                    raise ValueError(f"Invalid row {line_number} in {path}: {error}") from error
                target = ID_TO_SPLIT[assignments.get(digest, SPLIT_TO_ID[source_split])]
                source_rows[source_split] += 1
                derived_rows[target] += 1
                source_to_derived[source_split][target] += 1
                source_class_counts[source_split][label] += 1
                class_counts[target][label] += 1
                quality[source_split]["missing_feature_values"] += missing
                quality[source_split]["infinite_feature_values"] += infinite
                quality[source_split]["non_numeric_feature_tokens"] += invalid
                if digest in candidates:
                    previous = first_labels.setdefault(digest, label)
                    if previous != label:
                        conflict_groups.add(digest)
        print(f"Checked {source_split}: {source_rows[source_split]:,} rows", flush=True)

    if expected_header is None:
        raise ValueError("No source files were inspected")
    return expected_header, {
        "source_rows": dict(source_rows),
        "derived_rows": dict(derived_rows),
        "source_to_derived_rows": {k: dict(v) for k, v in source_to_derived.items()},
        "source_class_counts": {k: dict(sorted(v.items())) for k, v in source_class_counts.items()},
        "class_counts": {k: dict(sorted(v.items())) for k, v in class_counts.items()},
        "quality": quality,
        "candidate_feature_groups_checked": len(first_labels),
        "conflicting_label_feature_groups": len(conflict_groups),
        "elapsed_seconds": round(time.perf_counter() - started, 2),
    }


def _write_raw_arrays(
    source_paths: dict[str, Path], assignments: dict[bytes, int], class_to_id: dict[str, int],
    row_counts: dict[str, int], staging: Path, expected_header: list[str],
) -> dict:
    staging.mkdir(parents=True, exist_ok=False)
    x_maps, y_maps = {}, {}
    for split in SPLIT_NAMES:
        x_maps[split] = np.lib.format.open_memmap(
            staging / f"X_{split}_raw.npy", mode="w+", dtype=np.float32,
            shape=(int(row_counts[split]), EXPECTED_FEATURES),
        )
        y_maps[split] = np.lib.format.open_memmap(
            staging / f"y_{split}.npy", mode="w+", dtype=np.int32, shape=(int(row_counts[split]),)
        )
    x_buffers = {split: np.empty((WRITE_BUFFER_ROWS, EXPECTED_FEATURES), dtype=np.float32) for split in SPLIT_NAMES}
    y_buffers = {split: np.empty(WRITE_BUFFER_ROWS, dtype=np.int32) for split in SPLIT_NAMES}
    pending, written, unknown = Counter(), Counter(), Counter()

    def flush(split: str) -> None:
        count = pending[split]
        if count == 0:
            return
        start = written[split]
        end = start + count
        x_maps[split][start:end] = x_buffers[split][:count]
        y_maps[split][start:end] = y_buffers[split][:count]
        written[split] = end
        pending[split] = 0

    for source_split in SPLIT_NAMES:
        path = Path(source_paths[source_split])
        with path.open("rb") as file:
            header = [part.strip().decode("utf-8-sig") for part in file.readline().rstrip(b"\r\n").split(b",")]
            if header != expected_header:
                raise ValueError(f"Input schema changed between passes: {path}")
            for line_number, line in enumerate(file, start=2):
                row = line.rstrip(b"\r\n")
                if not row:
                    continue
                digest, label, values, _, _, _ = _parse_record(row, len(expected_header))
                target = ID_TO_SPLIT[assignments.get(digest, SPLIT_TO_ID[source_split])]
                class_id = class_to_id.get(label, -1)
                if class_id < 0:
                    unknown[label] += 1
                slot = pending[target]
                x_buffers[target][slot] = values
                y_buffers[target][slot] = class_id
                pending[target] += 1
                if pending[target] == WRITE_BUFFER_ROWS:
                    flush(target)
                if line_number % 100_000 == 0:
                    print(f"Writing arrays: {source_split} row {line_number:,}", flush=True)
    for split in SPLIT_NAMES:
        flush(split)
        x_maps[split].flush()
        y_maps[split].flush()
        if written[split] != row_counts[split]:
            raise AssertionError(f"{split}: expected {row_counts[split]}, wrote {written[split]}")
        del x_maps[split], y_maps[split]
    return {"written_rows": dict(written), "labels_without_training_mapping": dict(unknown)}


def _fit_train_preprocessor(staging: Path, output_arrays: Path, labels: dict, batch_rows: int) -> dict:
    x_train = np.load(staging / "X_train_raw.npy", mmap_mode="r+")
    medians = np.zeros(EXPECTED_FEATURES, dtype=np.float64)
    entirely_missing = []
    for column_id in range(EXPECTED_FEATURES):
        column = np.asarray(x_train[:, column_id])
        finite = column[np.isfinite(column)]
        if finite.size:
            medians[column_id] = float(np.median(finite))
        else:
            entirely_missing.append(column_id)

    scaler = StandardScaler(copy=False)
    for start in range(0, x_train.shape[0], batch_rows):
        end = min(start + batch_rows, x_train.shape[0])
        chunk = np.array(x_train[start:end], dtype=np.float32, copy=True)
        bad = ~np.isfinite(chunk)
        if bad.any():
            columns = np.nonzero(bad)[1]
            chunk[bad] = medians[columns].astype(np.float32, copy=False)
        scaler.partial_fit(chunk)

    original_indices = np.arange(EXPECTED_FEATURES, dtype=np.int32)
    keep_mask = np.isfinite(scaler.var_) & (scaler.var_ > 0.0)
    if entirely_missing:
        keep_mask[np.asarray(entirely_missing, dtype=int)] = False
    keep_indices = original_indices[keep_mask]
    if keep_indices.size == 0:
        raise ValueError("All training features are constant or entirely missing")
    means = scaler.mean_[keep_indices].astype(np.float64)
    scales = scaler.scale_[keep_indices].astype(np.float64)
    scales[~np.isfinite(scales) | (scales == 0)] = 1.0

    output_arrays.mkdir(parents=True, exist_ok=False)
    for split in SPLIT_NAMES:
        raw_x = np.load(staging / f"X_{split}_raw.npy", mmap_mode="r")
        raw_y = np.load(staging / f"y_{split}.npy", mmap_mode="r")
        final_x = np.lib.format.open_memmap(
            output_arrays / f"X_{split}.npy", mode="w+", dtype=np.float32,
            shape=(raw_x.shape[0], int(keep_indices.size)),
        )
        final_y = np.lib.format.open_memmap(
            output_arrays / f"y_{split}.npy", mode="w+", dtype=np.int32, shape=(raw_y.shape[0],)
        )
        for start in range(0, raw_x.shape[0], batch_rows):
            end = min(start + batch_rows, raw_x.shape[0])
            chunk = np.array(raw_x[start:end], dtype=np.float32, copy=True)
            bad = ~np.isfinite(chunk)
            if bad.any():
                columns = np.nonzero(bad)[1]
                chunk[bad] = medians[columns].astype(np.float32, copy=False)
            selected = chunk[:, keep_indices]
            final_x[start:end] = ((selected.astype(np.float64) - means) / scales).astype(np.float32)
            final_y[start:end] = raw_y[start:end]
        final_x.flush()
        final_y.flush()
        del final_x, final_y, raw_x, raw_y

    removed = []
    for index in original_indices:
        if not keep_mask[index]:
            why = "entirely missing in derived training data" if index in entirely_missing else "constant in derived training data"
            removed.append({"feature": labels["feature_columns"][int(index)], "reason": why})
    preprocessor = {
        "feature_columns_input": labels["feature_columns"],
        "feature_columns_retained": [labels["feature_columns"][int(i)] for i in keep_indices],
        "retained_feature_indices": keep_indices,
        "removed_features": removed,
        "imputer_strategy": "median per feature, fitted only on derived training rows; non-finite values treated as missing",
        "imputer_medians": medians.astype(np.float32),
        "scaler": "sklearn StandardScaler.partial_fit on imputed derived training rows only",
        "scaler_mean": means,
        "scaler_scale": scales,
        "class_to_id": labels["class_to_id"],
        "id_to_class": labels["id_to_class"],
        "fit_split": "derived train only",
    }
    joblib.dump(preprocessor, output_arrays / "preprocessing.joblib", compress=3)
    compact = {
        "feature_columns_input": preprocessor["feature_columns_input"],
        "feature_columns_retained": preprocessor["feature_columns_retained"],
        "removed_features": removed,
        "imputer_strategy": preprocessor["imputer_strategy"],
        "imputer_medians": [float(x) for x in medians],
        "scaler": preprocessor["scaler"],
        "scaler_mean": [float(x) for x in means],
        "scaler_scale": [float(x) for x in scales],
        "fit_split": "derived train only",
        "class_to_id": labels["class_to_id"],
        "id_to_class": labels["id_to_class"],
        "feature_dtype": "float32",
        "label_dtype": "int32; -1 means label absent from derived train",
    }
    return compact


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def _write_class_counts(path: Path, counts: dict[str, dict[str, int]], labels: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["label", *SPLIT_NAMES])
        for label in labels:
            writer.writerow([label, *(counts[split].get(label, 0) for split in SPLIT_NAMES)])


def _tables(data: dict) -> tuple[str, str, str, str, str]:
    old_audit = data.get("source_duplicate_overlap_audit", {})
    within = old_audit.get("within_split", {})
    overlap = old_audit.get("cross_split_unique_fingerprint_overlap", {})
    dup = ["| Source split | Repeated exact-row fingerprints | Extra exact-row occurrences | Repeated feature vectors |", "| --- | ---: | ---: | ---: |"]
    for split in SPLIT_NAMES:
        row = within.get(split, {})
        dup.append(f"| {split} | {row.get('duplicate_exact_row_fingerprints', 'n/a')} | {row.get('duplicate_row_occurrences_beyond_first', 'n/a')} | {row.get('duplicate_feature_vector_fingerprints', 'n/a')} |")
    shared = ["| Pair | Shared exact-row fingerprints | Shared feature-vector fingerprints |", "| --- | ---: | ---: |"]
    for pair, values in overlap.items():
        shared.append(f"| {pair.replace('__', ' ↔ ')} | {values.get('exact_row', 0):,} | {values.get('feature_vector', 0):,} |")
    quality = ["| Source | Missing feature values | Infinite values | Invalid numeric tokens | Rows |", "| --- | ---: | ---: | ---: | ---: |"]
    for split in SPLIT_NAMES:
        q = data["source_quality"][split]
        quality.append(f"| {split} | {q['missing_feature_values']:,} | {q['infinite_feature_values']:,} | {q['non_numeric_feature_tokens']:,} | {data['source_rows'][split]:,} |")
    coverage = ["| Split | Rows | Class coverage | Missing classes |", "| --- | ---: | ---: | --- |"]
    for split in SPLIT_NAMES:
        absent = [label for label in data["labels"] if not data["derived_class_counts"][split].get(label, 0)]
        present = len(data["labels"]) - len(absent)
        coverage.append(f"| {split} | {data['derived_rows'][split]:,} | {present}/{len(data['labels'])} | {', '.join(absent) if absent else 'None'} |")
    movement = ["| Original file | To train | To validation | To test |", "| --- | ---: | ---: | ---: |"]
    for split in SPLIT_NAMES:
        counts = data["source_to_derived_rows"][split]
        movement.append(f"| {split} | {counts['train']:,} | {counts['validation']:,} | {counts['test']:,} |")
    return "\n".join(dup), "\n".join(shared), "\n".join(quality), "\n".join(coverage), "\n".join(movement)


def render_report(data: dict) -> str:
    dup, shared, quality, coverage, movement = _tables(data)
    class_table = ["| Class | Train | Validation | Test |", "| --- | ---: | ---: | ---: |"]
    for label in data["labels"]:
        counts = data["derived_class_counts"]
        class_table.append("| " + label + " | " + " | ".join(f"{counts[s].get(label, 0):,}" for s in SPLIT_NAMES) + " |")
    removed = data["preprocessing"]["removed_features"]
    removed_text = "None; all retained features vary in derived training." if not removed else "; ".join(f"{x['feature']} ({x['reason']})" for x in removed)
    train_support = data["derived_class_counts"]["train"]
    smallest_label = min(train_support, key=train_support.get)
    largest_label = max(train_support, key=train_support.get)
    imbalance_text = f"Training support ranges from {smallest_label} ({train_support[smallest_label]:,}) to {largest_label} ({train_support[largest_label]:,}), about {train_support[largest_label] / train_support[smallest_label]:,.0f}:1."
    total = sum(data["source_rows"].values())
    return f"""# Phase 1 — Full-dataset preparation report

**Preparation date:** {time.strftime('%Y-%m-%d')}  
**Status:** split construction and preprocessing only; no training was run.  
**Seed:** {data['seed']}

## Summary

Created derived arrays from all {total:,} rows in the supplied source CSVs. The original files and sampled experiments were read only. Every well-formed, non-empty row was retained once. Derived sizes: **{data['derived_rows']['train']:,} train**, **{data['derived_rows']['validation']:,} validation**, and **{data['derived_rows']['test']:,} test**.

The schema has no stable device, capture, session, timestamp, or flow ID. This is an **exact-feature-group-aware split**, not a claim of device-, capture-, session-, or time-level independence. Related but non-identical flows may still cross splits.

## Duplicate and overlap audit

The earlier SQLite fingerprint audit stores one row per distinct fingerprint with occurrence counters. Its overlap values are counts of **distinct shared fingerprints**, not all matching row occurrences. Pair counts are not additive because a fingerprint present in all three files appears in more than one pair.

{dup}

{shared}

The streamed Phase 1 check examined {data['candidate_feature_groups_checked']:,} repeated or shared feature groups for label consistency and found **{data['conflicting_label_feature_groups']:,} groups with conflicting labels**. Repeated vectors are not assumed erroneous: none were deleted, and all repeats follow their group's assignment.

## Split methodology

For each SHA-256 fingerprint of the original 46 feature tokens (label excluded), assign the entire group to the source split with the largest occurrence count. Resolve ties with a deterministic SHA-256 choice and seed {data['seed']}. Groups unique to one source split remain there. No records are duplicated or oversampled.

{movement}

All occurrences of one serialized-token feature fingerprint have one destination, so overlap for that exact fingerprint definition is **0 by construction**. The digest is computed from the original numeric text tokens, not canonicalized float values; alternate text encodings of equal numeric values were not separately merged. The procedure retains all rows and conservatively reconciles the supplied splits; it is not a new random split from a known population.

## Derived class support

{coverage}

{imbalance_text}

{chr(10).join(class_table)}

No examples were added to force class coverage. Low-support classes will still have uncertain later per-class metrics.

## Data quality

{quality}

The three files were streamed and checked for schema width, numeric parseability, missing feature values, and infinities. Non-finite or non-numeric feature values are treated as missing for the train-fitted imputer. Malformed rows cause a clear failure instead of being silently discarded.

## Preprocessing

- Separate the label from the 46 numeric feature columns.
- Fit label mapping using derived training labels only. A label not seen in derived train is recorded with ID -1 in validation/test.
- Fit per-feature medians on derived training rows only; apply fixed medians to validation/test.
- Fit StandardScaler incrementally on imputed derived training rows only; apply the saved transform unchanged to validation/test.
- Identify constant or entirely missing columns using derived training only. Removed: {removed_text}
- Save X arrays as float32 memory-mappable NumPy files and y arrays as int32.

Test fingerprints are used only to reconcile identical-vector overlap. Test labels do not fit preprocessing or select models. Output metadata records the fitted mapping and numeric transforms.

## Limitations

1. Exact-feature grouping cannot replace missing device/capture/session/time identifiers; similar traffic with different values may still cross splits.
2. Majority-source assignment retains the provided split structure where possible, but does not prove the original assignments were random or independent.
3. Class imbalance is severe and later per-class metrics must be interpreted with support.
4. No classifier, FedAvg, drift detector, selective retraining, or dashboard was run in this phase.

## Next step

Given the available schema, exact-token feature grouping is the strongest defensible local leakage mitigation; do not invent device or event IDs. For stronger group-aware generalization, obtain authentic capture/device/session identifiers and split by those groups. Review and approve this policy before Phase 2. Keep the derived test arrays reserved for final evaluation.
"""


def run_preparation(
    source_paths: dict[str, Path], audit_db: Path, output_dir: Path,
    seed: int = 20261026, chunk_rows: int = CHUNK_ROWS,
) -> dict:
    output_dir = Path(output_dir).resolve()
    build_dir = output_dir.with_name(f".{output_dir.name}_building")
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite prepared output: {output_dir}")
    if build_dir.exists():
        raise FileExistsError(f"Refusing to overwrite build directory: {build_dir}")
    for split in SPLIT_NAMES:
        if split not in source_paths or not Path(source_paths[split]).is_file():
            raise FileNotFoundError(f"Missing required {split} source file: {source_paths.get(split)}")

    started = time.perf_counter()
    assignments, candidates, assignment_summary = load_group_assignment(Path(audit_db), seed)
    print(f"Loaded {len(assignments):,} shared group assignments and {len(candidates):,} audit candidates", flush=True)
    header, inspected = inspect_rows(source_paths, assignments, candidates)
    if sum(inspected["derived_rows"].values()) != sum(inspected["source_rows"].values()):
        raise AssertionError("Row conservation failed")
    if any(q["malformed_rows"] for q in inspected["quality"].values()):
        raise ValueError("Malformed input records were found")

    train_classes = sorted(inspected["class_counts"]["train"])
    if not train_classes:
        raise ValueError("No classes assigned to derived training")
    class_to_id = {label: idx for idx, label in enumerate(train_classes)}
    labels_info = {
        "feature_columns": header[:-1],
        "class_to_id": class_to_id,
        "id_to_class": {str(idx): label for label, idx in class_to_id.items()},
    }
    build_dir.mkdir(parents=True, exist_ok=False)
    staging = build_dir / "_staging"
    arrays_dir = build_dir / "arrays"
    write_summary = _write_raw_arrays(
        source_paths, assignments, class_to_id, inspected["derived_rows"], staging, header
    )
    preprocessing = _fit_train_preprocessor(staging, arrays_dir, labels_info, chunk_rows)
    shutil.rmtree(staging)

    labels = sorted(set().union(*(set(inspected["class_counts"][split]) for split in SPLIT_NAMES)))
    result = {
        "source_paths": {name: str(Path(source_paths[name]).resolve()) for name in SPLIT_NAMES},
        "source_rows": inspected["source_rows"],
        "derived_rows": inspected["derived_rows"],
        "source_class_counts": inspected["source_class_counts"],
        "derived_class_counts": inspected["class_counts"],
        "source_to_derived_rows": {
            source: {target: int(inspected["source_to_derived_rows"][source].get(target, 0)) for target in SPLIT_NAMES}
            for source in SPLIT_NAMES
        },
        "source_quality": inspected["quality"],
        "feature_columns_input": header[:-1],
        "feature_columns_retained": preprocessing["feature_columns_retained"],
        "labels": labels,
        "class_to_id": class_to_id,
        "assignment": assignment_summary,
        "write": write_summary,
        "candidate_feature_groups_checked": inspected["candidate_feature_groups_checked"],
        "conflicting_label_feature_groups": inspected["conflicting_label_feature_groups"],
        "preprocessing": preprocessing,
        "seed": seed,
        "elapsed_seconds": round(time.perf_counter() - started, 2),
    }
    audit_json = Path(audit_db).with_name("duplicate_overlap_audit.json")
    source_audit = json.loads(audit_json.read_text(encoding="utf-8")) if audit_json.is_file() else {}
    if source_audit:
        result["source_duplicate_overlap_audit"] = source_audit
    duplicate_overlap = {
        "source_fingerprint_audit": source_audit,
        "repeated_or_shared_feature_groups_checked_for_label_conflicts": result["candidate_feature_groups_checked"],
        "conflicting_label_feature_groups": result["conflicting_label_feature_groups"],
        "derived_cross_split_feature_vector_overlap_groups": 0,
        "derived_cross_split_exact_row_overlap_groups": 0,
        "post_assignment_claim": "zero by construction: every feature fingerprint is assigned exactly one destination; tests exercise disjointness on fixtures",
        "assignment": assignment_summary,
    }
    _write_json(build_dir / "duplicate_overlap_audit.json", duplicate_overlap)
    _write_json(build_dir / "split_sizes.json", {"source_rows": result["source_rows"], "derived_rows": result["derived_rows"], "row_total_preserved": sum(result["source_rows"].values()) == sum(result["derived_rows"].values())})
    _write_class_counts(build_dir / "class_counts.csv", inspected["class_counts"], labels)
    _write_json(build_dir / "phase1_metadata.json", result)
    _write_json(build_dir / "preprocessing_metadata.json", preprocessing)
    _write_json(build_dir / "split_methodology.json", {
        "policy": assignment_summary["assignment_rule"],
        "fingerprint": "SHA-256 over the original serialized 46 feature tokens; label and line ending excluded",
        "label_conflict_policy": "keep all identical feature vectors in one split; report conflicts, never silently resolve them",
        "duplicates": "retain every source row exactly once; no automatic deduplication or oversampling",
        "grouping_limitation": "the available columns include no stable device, capture, session, timestamp, or event identifier",
        "test_use": "test fingerprints participate only in exact-group overlap reconciliation; test labels do not fit preprocessing or model selection",
        "seed": seed,
    })
    (build_dir / "preparation_report.md").write_text(render_report(result), encoding="utf-8")
    os.replace(build_dir, output_dir)
    print(f"Prepared arrays and report at {output_dir}", flush=True)
    return result


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prepare group-aware float32 CICIoT2023 arrays without training.")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--audit-db", type=Path, default=Path(__file__).resolve().parents[2] / "work" / "row_fingerprints.sqlite")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[2] / "experiments" / "full_dataset" / "prepared")
    parser.add_argument("--seed", type=int, default=20261026)
    args = parser.parse_args(argv)
    paths = {split: args.dataset_root / split / f"{split}.csv" for split in SPLIT_NAMES}
    result = run_preparation(paths, args.audit_db, args.output, args.seed)
    print(json.dumps({
        "derived_rows": result["derived_rows"],
        "class_count": len(result["labels"]),
        "elapsed_seconds": result["elapsed_seconds"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())





