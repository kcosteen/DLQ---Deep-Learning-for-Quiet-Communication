"""Lightweight tests for the S->E analysis helpers (#13 / Task 5).

Selection (``top_k``, ``boundary_k``, ``random_k``), frame-id parsing
(``frame_id``) and frame clustering (``contiguous_runs``, ``run_stats``) must be
deterministic and edge-case-safe, because the error galleries and the
clustering statistics in ``s_to_e_error_analysis_targeted.json`` are built from
them.
"""

from __future__ import annotations

import pytest

from scripts.analyze_s_to_e import (
    boundary_k,
    contiguous_runs,
    frame_id,
    random_k,
    run_stats,
    top_k,
)


# --- frame_id ----------------------------------------------------------------
def test_frame_id_extracts_numeric_suffix():
    assert frame_id("S2451.jpg") == 2451
    assert frame_id("data/raw/asl_alphabet_train/S/S0001.jpg") == 1


def test_frame_id_requires_digits():
    with pytest.raises(ValueError):
        frame_id("S.jpg")


# --- top_k -------------------------------------------------------------------
def test_top_k_returns_largest_descending():
    recs = [{"id": i} for i in range(10)]
    got = top_k(recs, 3, key=lambda r: r["id"])
    assert [r["id"] for r in got] == [9, 8, 7]


def test_top_k_k_larger_than_list_returns_all():
    recs = [{"id": 1}, {"id": 2}]
    assert len(top_k(recs, 5, key=lambda r: r["id"])) == 2


def test_top_k_rejects_k_lt_1():
    with pytest.raises(ValueError):
        top_k([{"id": 1}], 0, key=lambda r: r["id"])


# --- boundary_k --------------------------------------------------------------
def test_boundary_k_smallest_positive_first():
    recs = [
        {"id": "b", "margin": 0.05},
        {"id": "d", "margin": -0.10},   # negative -> excluded
        {"id": "a", "margin": 0.01},
        {"id": "c", "margin": 0.90},
    ]
    got = boundary_k(recs, 2, key=lambda r: r["margin"])
    assert [r["id"] for r in got] == ["a", "b"]


def test_boundary_k_excludes_zero():
    recs = [{"id": "a", "margin": 0.0}, {"id": "b", "margin": 0.2}]
    got = boundary_k(recs, 5, key=lambda r: r["margin"])
    assert [r["id"] for r in got] == ["b"]


# --- random_k ----------------------------------------------------------------
def test_random_k_is_deterministic_given_seed():
    recs = [{"id": i} for i in range(50)]
    a = random_k(recs, 25, seed=42)
    b = random_k(recs, 25, seed=42)
    assert [r["id"] for r in a] == [r["id"] for r in b]


def test_random_k_different_seeds_differ():
    recs = [{"id": i} for i in range(50)]
    a = random_k(recs, 25, seed=42)
    b = random_k(recs, 25, seed=1)
    assert [r["id"] for r in a] != [r["id"] for r in b]


def test_random_k_size_and_no_repeats():
    recs = [{"id": i} for i in range(30)]
    ids = [r["id"] for r in random_k(recs, 25, seed=7)]
    assert len(ids) == 25
    assert len(ids) == len(set(ids))


def test_random_k_k_larger_than_list_returns_all():
    recs = [{"id": i} for i in range(10)]
    assert len(random_k(recs, 25, seed=0)) == 10


# --- contiguous_runs ---------------------------------------------------------
def test_contiguous_runs_empty():
    assert contiguous_runs([]) == []


def test_contiguous_runs_single():
    assert contiguous_runs([5]) == [(5, 5)]


def test_contiguous_runs_gaps_and_runs():
    assert contiguous_runs([1, 2, 3, 7, 8, 15]) == [(1, 3), (7, 8), (15, 15)]


def test_contiguous_runs_unsorted_and_duplicate_input():
    assert contiguous_runs([3, 1, 3, 2, 7]) == [(1, 3), (7, 7)]


# --- run_stats ---------------------------------------------------------------
def test_run_stats_longest_and_pcts():
    # lengths 3, 2, 1 across 6 total errors
    st = run_stats([(1, 3), (7, 8), (15, 15)], total=6)
    assert st["n_runs"] == 3
    assert st["longest_run"] == 3
    assert st["longest_run_pct"] == pytest.approx(50.0)
    assert st["top3_runs_pct"] == pytest.approx(100.0)


def test_run_stats_empty():
    st = run_stats([], total=0)
    assert st["n_runs"] == 0
    assert st["longest_run"] == 0
    assert st["longest_run_pct"] == 0.0
