"""Tests for the MediaPipe vs RGB analysis utilities.

Uses synthetic DataFrames for unit tests; integrated tests read the real CSV.
"""

from __future__ import annotations

import string

import numpy as np
import pandas as pd
import pytest

from scripts.analyze_mediapipe_vs_rgb import (
    AZ_CLASSES,
    compute_overall,
    compute_per_class,
    filter_az,
    load_records,
    validate_az,
)

ROOT = __import__("pathlib").Path(__file__).resolve().parent.parent
REAL_CSV = ROOT / "mediapipe_vs_rgb" / "mediapipe_vs_rgb_sample_records.csv"
HAS_REAL = REAL_CSV.exists()

# ---------------------------------------------------------------------------
# Synthetic fixtures
# ---------------------------------------------------------------------------

def _make_df(n_per_class: int = 10, seed: int = 0) -> pd.DataFrame:
    """Build a small synthetic DataFrame with A-Z + del/nothing/space."""
    rng = np.random.RandomState(seed)
    rows = []
    all_classes = list(AZ_CLASSES) + ["del", "nothing", "space"]
    for cls in all_classes:
        for i in range(n_per_class):
            detected = bool(rng.rand() > 0.3)
            correct = bool(rng.rand() > 0.1) if detected else bool(rng.rand() > 0.2)
            rows.append({
                "class": cls,
                "path": f"data/{cls}/{cls}{i}.jpg",
                "found_hand": detected,
                "prediction": cls if correct else "X",
                "confidence": round(float(rng.uniform(0.5, 1.0)), 4),
                "correct": correct,
            })
    return pd.DataFrame(rows)


@pytest.fixture
def synthetic_df():
    return _make_df()


# ---------------------------------------------------------------------------
# Unit tests: filter_az
# ---------------------------------------------------------------------------

def test_filter_az_excludes_special_classes(synthetic_df):
    filtered = filter_az(synthetic_df)
    assert set(filtered["class"].unique()).isdisjoint({"del", "nothing", "space"})


def test_filter_az_preserves_all_letters(synthetic_df):
    filtered = filter_az(synthetic_df)
    assert set(filtered["class"].unique()) == set(AZ_CLASSES)


def test_filter_az_row_count(synthetic_df):
    filtered = filter_az(synthetic_df)
    assert len(filtered) == 26 * 10


def test_validate_az_passes(synthetic_df):
    validate_az(filter_az(synthetic_df), expected_per_class=10)


def test_validate_az_rejects_wrong_count(synthetic_df):
    bad = filter_az(synthetic_df).iloc[:50]
    with pytest.raises(ValueError, match="2600"):
        validate_az(bad)


# ---------------------------------------------------------------------------
# Unit tests: 2x2 counts
# ---------------------------------------------------------------------------

def test_2x2_sums_to_total(synthetic_df):
    az = filter_az(synthetic_df)
    overall = compute_overall(az)
    assert (
        overall["correct_detected"]
        + overall["wrong_detected"]
        + overall["correct_missed"]
        + overall["wrong_missed"]
        == overall["n"]
    )


def test_detected_plus_missed_equals_n(synthetic_df):
    az = filter_az(synthetic_df)
    overall = compute_overall(az)
    assert overall["detected"] + overall["missed"] == overall["n"]


def test_correct_detected_plus_wrong_detected_equals_detected(synthetic_df):
    az = filter_az(synthetic_df)
    overall = compute_overall(az)
    assert overall["correct_detected"] + overall["wrong_detected"] == overall["detected"]


def test_correct_missed_plus_wrong_missed_equals_missed(synthetic_df):
    az = filter_az(synthetic_df)
    overall = compute_overall(az)
    assert overall["correct_missed"] + overall["wrong_missed"] == overall["missed"]


# ---------------------------------------------------------------------------
# Unit tests: discarded correct
# ---------------------------------------------------------------------------

def test_discarded_correct_is_missed_and_correct(synthetic_df):
    az = filter_az(synthetic_df)
    overall = compute_overall(az)
    missed = az[~az["found_hand"]]
    assert overall["discarded_correct"] == int(missed["correct"].sum())


def test_discarded_correct_percentage(synthetic_df):
    az = filter_az(synthetic_df)
    overall = compute_overall(az)
    expected_pct = round(100.0 * overall["correct_missed"] / overall["missed"], 1) if overall["missed"] else None
    assert overall["discarded_correct_pct_of_missed"] == expected_pct


# ---------------------------------------------------------------------------
# Unit tests: per-class
# ---------------------------------------------------------------------------

def test_per_class_count_is_26(synthetic_df):
    az = filter_az(synthetic_df)
    per_class = compute_per_class(az)
    assert len(per_class) == 26


def test_per_class_total_equals_df_total(synthetic_df):
    az = filter_az(synthetic_df)
    per_class = compute_per_class(az)
    assert sum(pc["n"] for pc in per_class) == len(az)


def test_per_class_accuracy_manual():
    """Hand-computed tiny example."""
    rows = [
        {"class": "A", "path": "a1", "found_hand": True, "prediction": "A", "confidence": 0.9, "correct": True},
        {"class": "A", "path": "a2", "found_hand": False, "prediction": "A", "confidence": 0.8, "correct": True},
        {"class": "A", "path": "a3", "found_hand": True, "prediction": "B", "confidence": 0.7, "correct": False},
        {"class": "B", "path": "b1", "found_hand": True, "prediction": "B", "confidence": 0.9, "correct": True},
        {"class": "B", "path": "b2", "found_hand": False, "prediction": "X", "confidence": 0.6, "correct": False},
    ]
    df = pd.DataFrame(rows)
    az = filter_az(df)
    overall = compute_overall(az)
    assert overall["n"] == 5
    assert overall["detected"] == 3
    assert overall["missed"] == 2
    assert overall["rgb_correct"] == 3
    assert overall["discarded_correct"] == 1  # a2: missed + correct
    per_class = compute_per_class(az)
    a_cls = next(pc for pc in per_class if pc["class"] == "A")
    assert a_cls["detected"] == 2
    assert a_cls["missed"] == 1
    assert a_cls["correct_missed"] == 1


# ---------------------------------------------------------------------------
# Unit tests: accuracy math
# ---------------------------------------------------------------------------

def test_rgb_acc_pct(synthetic_df):
    az = filter_az(synthetic_df)
    overall = compute_overall(az)
    assert overall["rgb_acc_pct"] == round(100.0 * overall["rgb_correct"] / overall["n"], 2)


def test_rgb_acc_detected_pct(synthetic_df):
    az = filter_az(synthetic_df)
    overall = compute_overall(az)
    det = az[az["found_hand"]]
    expected = round(100.0 * int(det["correct"].sum()) / len(det), 2)
    assert overall["rgb_acc_detected_pct"] == expected


# ---------------------------------------------------------------------------
# No duplicates
# ---------------------------------------------------------------------------

def test_no_duplicate_paths_in_real_csv():
    if not HAS_REAL:
        pytest.skip("real CSV not present")
    df = load_records(REAL_CSV)
    assert not df["path"].duplicated().any()


# ---------------------------------------------------------------------------
# Real CSV integrated tests
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not HAS_REAL, reason="real CSV not present")
class TestRealCSV:
    def test_total_rows(self):
        df = load_records(REAL_CSV)
        assert len(df) == 2900

    def test_29_classes(self):
        df = load_records(REAL_CSV)
        assert df["class"].nunique() == 29

    def test_100_per_class(self):
        df = load_records(REAL_CSV)
        counts = df["class"].value_counts()
        assert (counts == 100).all()

    def test_az_total_2600(self):
        df = load_records(REAL_CSV)
        az = filter_az(df)
        assert len(az) == 2600

    def test_az_26_classes(self):
        df = load_records(REAL_CSV)
        az = filter_az(df)
        assert az["class"].nunique() == 26

    def test_mp_coverage(self):
        df = load_records(REAL_CSV)
        az = filter_az(df)
        overall = compute_overall(az)
        assert overall["detected"] == 2023
        assert overall["missed"] == 577

    def test_rgb_acc(self):
        df = load_records(REAL_CSV)
        az = filter_az(df)
        overall = compute_overall(az)
        assert overall["rgb_acc_pct"] == 99.27

    def test_discarded_correct(self):
        df = load_records(REAL_CSV)
        az = filter_az(df)
        overall = compute_overall(az)
        assert overall["discarded_correct"] == 565

    def test_m_coverage(self):
        df = load_records(REAL_CSV)
        az = filter_az(df)
        per_class = compute_per_class(az)
        m = next(pc for pc in per_class if pc["class"] == "M")
        assert m["mp_coverage_pct"] == 46.0

    def test_n_coverage(self):
        df = load_records(REAL_CSV)
        az = filter_az(df)
        per_class = compute_per_class(az)
        n = next(pc for pc in per_class if pc["class"] == "N")
        assert n["mp_coverage_pct"] == 48.0
