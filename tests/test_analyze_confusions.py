"""Ranking rules for the confusion-matrix diagnostic (#12 / Task 1).

The script must stay purely descriptive and derive everything from the
ablation matrix: diagonal cells excluded, direction preserved (``M -> N`` vs
``N -> M``), descending by error count, zero-count pairs dropped.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from scripts.analyze_confusions import (
    load_confusion_csv,
    off_diagonal_total,
    rank_confusions,
    symmetric_pairs,
)

CLASSES = ["M", "N", "V", "X"]

def _write_csv(tmp_path, rows) -> "np.ndarray":
    """Build a 4x4 CSV from a list of rows and return the matrix it encodes."""
    matrix = np.array(rows, dtype=int)
    lines = ["," + ",".join(CLASSES)]
    for label, row in zip(CLASSES, matrix):
        lines.append(label + "," + ",".join(str(int(v)) for v in row))
    (tmp_path / "confusion.csv").write_text("\n".join(lines) + "\n")
    return matrix


def test_diagonal_cells_are_excluded(tmp_path):
    # Huge diagonal values must not leak into the ranking.
    matrix = _write_csv(tmp_path, [
        [999, 2, 0, 0],
        [0, 888, 0, 0],
        [0, 0, 777, 0],
        [0, 0, 0, 666],
    ])
    classes, loaded = load_confusion_csv(tmp_path / "confusion.csv")
    assert np.array_equal(loaded, matrix)
    rows = rank_confusions(loaded, classes)
    assert all(true != pred for true, pred, _ in rows)
    assert all(count <= 2 for _, _, count in rows)


def test_directional_pairs_stay_separate(tmp_path):
    matrix = _write_csv(tmp_path, [
        [0, 15, 0, 0],
        [36, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ])
    classes, loaded = load_confusion_csv(tmp_path / "confusion.csv")
    rows = rank_confusions(loaded, classes)
    assert ("M", "N", 15) in rows
    assert ("N", "M", 36) in rows
    assert len(rows) == 2


def test_sorted_descending_by_error_count(tmp_path):
    matrix = _write_csv(tmp_path, [
        [0, 5, 9, 1],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ])
    classes, loaded = load_confusion_csv(tmp_path / "confusion.csv")
    rows = rank_confusions(loaded, classes)
    counts = [c for _, _, c in rows]
    assert counts == sorted(counts, reverse=True)
    assert rows[0] == ("M", "V", 9)


def test_zero_count_pairs_are_excluded(tmp_path):
    matrix = _write_csv(tmp_path, [
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 4],
        [0, 0, 0, 0],
    ])
    classes, loaded = load_confusion_csv(tmp_path / "confusion.csv")
    rows = rank_confusions(loaded, classes)
    assert rows == [("V", "X", 4)]


def test_off_diagonal_total(tmp_path):
    matrix = _write_csv(tmp_path, [
        [600, 1, 5, 2],
        [0, 600, 0, 0],
        [0, 0, 600, 0],
        [0, 0, 0, 600],
    ])
    assert off_diagonal_total(matrix) == 8


def test_symmetric_pairs_combine_directions(tmp_path):
    matrix = _write_csv(tmp_path, [
        [0, 15, 0, 0],
        [36, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ])
    classes, loaded = load_confusion_csv(tmp_path / "confusion.csv")
    pairs = symmetric_pairs(loaded, classes)
    assert ("M", "N", 51) in pairs


def test_ablation_artifact_headline_numbers():
    """Integration on the real ablation matrix (docs/figures)."""
    classes, matrix = load_confusion_csv(Path(
        "docs/figures/confusion_matrix_ablate.csv"
    ))
    assert len(classes) == 29
    rows = rank_confusions(matrix, classes, top=20)
    by_pair = {(true, pred): count for true, pred, count in rows}
    assert by_pair[("V", "K")] == 160
    assert by_pair[("X", "R")] == 60
    assert by_pair[("S", "E")] == 23
    assert off_diagonal_total(matrix) == 378
