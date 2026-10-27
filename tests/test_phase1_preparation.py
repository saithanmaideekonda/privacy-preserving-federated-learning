"""Phase 1 preparation pipeline tests using tiny temporary CSV fixtures."""

import hashlib
import sqlite3
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.full_dataset.prepare import (
    SPLIT_NAMES,
    choose_group_split,
    inspect_rows,
    load_group_assignment,
    run_preparation,
)


def _features(a: int, b: int) -> list[str]:
    return [str(a), str(b), *("0" for _ in range(44))]


def _write_splits(root: Path, rows_by_split: dict[str, list[tuple[list[str], str]]]) -> dict[str, Path]:
    paths = {}
    header = [f"feature_{i}" for i in range(46)] + ["label"]
    for split in SPLIT_NAMES:
        directory = root / split
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{split}.csv"
        with path.open("w", encoding="utf-8", newline="") as f:
            f.write(",".join(header) + "\n")
            for features, label in rows_by_split[split]:
                f.write(",".join(features + [label]) + "\n")
        paths[split] = path
    return paths


def _make_audit(paths: dict[str, Path], db_path: Path) -> None:
    con = sqlite3.connect(db_path)
    con.execute(
        """CREATE TABLE fingerprints (
            kind INTEGER NOT NULL, digest BLOB NOT NULL,
            train_count INTEGER NOT NULL DEFAULT 0,
            validation_count INTEGER NOT NULL DEFAULT 0,
            test_count INTEGER NOT NULL DEFAULT 0,
            PRIMARY KEY(kind,digest)) WITHOUT ROWID"""
    )
    split_index = {name: i for i, name in enumerate(SPLIT_NAMES)}
    for split, path in paths.items():
        index = split_index[split]
        with path.open("rb") as f:
            f.readline()
            for line in f:
                row = line.rstrip(b"\r\n")
                if not row:
                    continue
                feature = row.rsplit(b",", 1)[0]
                for kind, value in ((0, row), (1, feature)):
                    digest = hashlib.sha256(value).digest()
                    counts = [0, 0, 0]
                    counts[index] = 1
                    con.execute(
                        """INSERT INTO fingerprints VALUES(?,?,?,?,?)
                           ON CONFLICT(kind,digest) DO UPDATE SET
                           train_count=train_count+excluded.train_count,
                           validation_count=validation_count+excluded.validation_count,
                           test_count=test_count+excluded.test_count""",
                        (kind, digest, *counts),
                    )
    con.commit()
    con.close()


class Phase1PreparationTests(unittest.TestCase):
    def _fixture(self, root: Path):
        rows = {
            "train": [
                (_features(1, 1), "A"),
                (_features(1, 1), "A"),
                (_features(1, 2), "A"),
                (_features(2, 1), "B"),
                (_features(3, 1), "C"),
            ],
            "validation": [
                (_features(1, 1), "A"),
                (_features(1, 3), "A"),
                (_features(2, 2), "B"),
                (_features(3, 2), "C"),
            ],
            "test": [
                (_features(1, 4), "A"),
                (_features(2, 3), "B"),
                (_features(3, 3), "C"),
            ],
        }
        paths = _write_splits(root / "csv", rows)
        db = root / "fingerprints.sqlite"
        _make_audit(paths, db)
        return paths, db, rows

    def test_group_assignment_is_reproducible_and_majority_based(self):
        digest = hashlib.sha256(b"same features").digest()
        self.assertEqual(choose_group_split((4, 1, 0), digest, 7), 0)
        self.assertEqual(choose_group_split((1, 1, 0), digest, 7), choose_group_split((1, 1, 0), digest, 7))
        with self.assertRaises(ValueError):
            choose_group_split((0, 0, 0), digest, 7)

    def test_prepared_partitions_are_disjoint_complete_and_class_covered(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd() / "work") as temp:
            root = Path(temp)
            paths, db, source_rows = self._fixture(root)
            output = root / "prepared"
            result = run_preparation(paths, db, output, seed=123, chunk_rows=2)

            self.assertEqual(sum(result["source_rows"].values()), sum(result["derived_rows"].values()))
            self.assertEqual(sum(result["derived_rows"].values()), sum(len(v) for v in source_rows.values()))
            self.assertEqual(len(result["labels"]), 3)
            for split in SPLIT_NAMES:
                self.assertEqual(set(result["derived_class_counts"][split]), {"A", "B", "C"})
                x = np.load(output / "arrays" / f"X_{split}.npy")
                y = np.load(output / "arrays" / f"y_{split}.npy")
                self.assertEqual(x.dtype, np.float32)
                self.assertEqual(y.dtype, np.int32)
                self.assertEqual(x.shape[0], result["derived_rows"][split])
                self.assertEqual(y.shape, (x.shape[0],))
                self.assertEqual(x.shape[1], len(result["feature_columns_retained"]))
                self.assertTrue(np.isfinite(x).all())
            assignments, _, _ = load_group_assignment(db, seed=123)
            self.assertEqual(len(assignments), 1)
            destination_groups = {split: set() for split in SPLIT_NAMES}
            for source_id, split in enumerate(SPLIT_NAMES):
                with paths[split].open("rb") as stream:
                    stream.readline()
                    for line in stream:
                        row = line.rstrip(b"\r\n")
                        digest = hashlib.sha256(row.rsplit(b",", 1)[0]).digest()
                        target_id = assignments.get(digest, source_id)
                        destination_groups[SPLIT_NAMES[target_id]].add(digest)
            for i, left in enumerate(SPLIT_NAMES):
                for right in SPLIT_NAMES[i + 1:]:
                    self.assertFalse(destination_groups[left] & destination_groups[right])
            for split in SPLIT_NAMES:
                expected_ids = {result["class_to_id"][label] for label in result["derived_class_counts"][split]}
                self.assertEqual(set(np.load(output / "arrays" / f"y_{split}.npy").tolist()), expected_ids)

    def test_conflicting_labels_are_reported_not_rewritten(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd() / "work") as temp:
            root = Path(temp)
            conflict_vector = _features(9, 9)
            rows = {
                "train": [(conflict_vector, "A"), (_features(1, 0), "A"), (_features(2, 0), "B")],
                "validation": [(conflict_vector, "B"), (_features(1, 1), "A"), (_features(2, 1), "B")],
                "test": [(_features(1, 2), "A"), (_features(2, 2), "B")],
            }
            paths = _write_splits(root / "csv", rows)
            db = root / "fingerprints.sqlite"
            _make_audit(paths, db)
            assignments, candidates, _ = load_group_assignment(db, seed=11)
            _, inspected = inspect_rows(paths, assignments, candidates)
            self.assertEqual(inspected["conflicting_label_feature_groups"], 1)
            total_labeled = sum(sum(counts.values()) for counts in inspected["class_counts"].values())
            self.assertEqual(total_labeled, sum(inspected["source_rows"].values()))

    def test_same_seed_produces_identical_arrays_and_mappings(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd() / "work") as temp:
            root = Path(temp)
            paths, db, _ = self._fixture(root)
            first = run_preparation(paths, db, root / "first", seed=2026, chunk_rows=2)
            second = run_preparation(paths, db, root / "second", seed=2026, chunk_rows=2)
            self.assertEqual(first["derived_rows"], second["derived_rows"])
            self.assertEqual(first["class_to_id"], second["class_to_id"])
            for split in SPLIT_NAMES:
                x1 = np.load(root / "first" / "arrays" / f"X_{split}.npy")
                x2 = np.load(root / "second" / "arrays" / f"X_{split}.npy")
                y1 = np.load(root / "first" / "arrays" / f"y_{split}.npy")
                y2 = np.load(root / "second" / "arrays" / f"y_{split}.npy")
                np.testing.assert_array_equal(x1, x2)
                np.testing.assert_array_equal(y1, y2)


if __name__ == "__main__":
    unittest.main()






