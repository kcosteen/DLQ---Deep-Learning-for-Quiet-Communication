"""Tests for the transition-run selection helpers (#13 / Task 5 follow-up).

``even_sample``, ``correct_before``, ``correct_after``, ``segment_frames`` and
``probe_frames`` build the tile lists for ``E_transition_runs.png`` and the
boundary tables in ``s_to_e_transition_visual_notes.md``. They must be
deterministic, chronological, and edge-case-safe.
"""

from __future__ import annotations

import pytest

from scripts.inspect_s_to_e_runs import (
    correct_after,
    correct_before,
    even_sample,
    probe_frames,
    segment_frames,
)


def _rec(fid: int, pred: str = "S") -> dict:
    return {"frame_id": fid, "pred": pred, "p_s": 0.5, "p_e": 0.4}


# --- even_sample -------------------------------------------------------------
def test_even_sample_shorter_than_n_returns_all():
    seq = [_rec(i) for i in range(3)]
    assert [r["frame_id"] for r in even_sample(seq, 10)] == [0, 1, 2]


def test_even_sample_keeps_first_and_last():
    seq = [_rec(i) for i in range(40)]
    got = [r["frame_id"] for r in even_sample(seq, 6)]
    assert got[0] == 0
    assert got[-1] == 39


def test_even_sample_is_deterministic():
    seq = [_rec(i) for i in range(45)]
    a = [r["frame_id"] for r in even_sample(seq, 14)]
    b = [r["frame_id"] for r in even_sample(seq, 14)]
    assert a == b
    assert len(a) == 14


def test_even_sample_preserves_order():
    seq = [_rec(i) for i in range(30)]
    got = [r["frame_id"] for r in even_sample(seq, 9)]
    assert got == sorted(got)


def test_even_sample_rejects_n_lt_1():
    with pytest.raises(ValueError):
        even_sample([_rec(1)], 0)


# --- correct_before / correct_after -----------------------------------------
def test_correct_before_returns_last_k_ascending():
    recs = [_rec(i) for i in range(1, 11)]
    got = correct_before(recs, 8, k=3)
    assert [r["frame_id"] for r in got] == [5, 6, 7]


def test_correct_before_excludes_non_s_and_errors():
    recs = [_rec(2), _rec(3, pred="E"), _rec(4, pred="X"), _rec(5)]
    got = correct_before(recs, 6, k=5)
    assert [r["frame_id"] for r in got] == [2, 5]


def test_correct_after_returns_first_k_ascending():
    recs = [_rec(i) for i in range(1, 11)]
    got = correct_after(recs, 7, k=3)
    assert [r["frame_id"] for r in got] == [8, 9, 10]


def test_correct_after_handles_no_candidates():
    recs = [_rec(i) for i in range(1, 6)]
    assert correct_after(recs, 5, k=3) == []
    assert correct_before(recs, 1, k=3) == []


# --- segment_frames ----------------------------------------------------------
def test_segment_frames_is_chronological_and_bounded():
    # S2550-S2621 correct, S2575-S2619 "run", S2620+ correct
    recs = [_rec(i) for i in range(2550, 2622)]
    recs += [_rec(i, pred="E") for i in range(2575, 2620)]
    frames = segment_frames(recs, 2575, 2619, mid_k=8, before_k=2, after_k=2)
    ids = [r["frame_id"] for r in frames]
    assert ids == sorted(ids)
    assert ids[0] == 2573  # correct frame before the run
    assert ids[1] == 2574
    assert ids[-2] == 2620  # first correct after the run
    assert ids[-1] == 2621
    assert ids[0] < 2575 and ids[-1] > 2619


def test_segment_frames_no_overlap_before_after():
    recs = [_rec(i) for i in range(2800, 2816)]
    recs += [_rec(i, pred="E") for i in range(2801, 2815)]
    frames = segment_frames(recs, 2801, 2814, mid_k=10, before_k=1, after_k=1)
    ids = [r["frame_id"] for r in frames]
    assert ids == sorted(set(ids))  # no duplicated frame across groups


# --- probe_frames ------------------------------------------------------------
def test_probe_frames_first_max_last():
    # S2829-S2879 correct, S2831-S2877 the run
    recs = [_rec(i) for i in range(2829, 2880)]
    recs += [_rec(i, pred="E") for i in range(2831, 2878)]
    probes = probe_frames(recs, 2831, 2877, before_k=2, after_k=2)
    ids = [r["frame_id"] for r in probes]
    assert ids[:2] == [2829, 2830]
    assert ids[2] == 2831            # first error
    assert ids[-2:] == [2878, 2879]  # correct after the run
    assert ids[-3] == 2877           # last error
    # max-P(E) probe is in the list somewhere (may coincide with first/last)
    assert ids[2] in ids and ids[-3] in ids


def test_probe_frames_empty_when_no_errors():
    recs = [_rec(i) for i in range(3000, 3010)]
    assert probe_frames(recs, 3001, 3005, before_k=1, after_k=1) == []
