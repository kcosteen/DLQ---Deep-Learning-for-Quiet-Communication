"""Tests for the human-inspection gallery builders (#13 / Task 5 follow-up).

``transition_strip`` chooses the exact boundary-focused frames for each dominant
run (4 correct-S before, first 8 errors, blips, 2-3 middle errors, final 4
errors, 4 correct-S after). The integration test pins the exact expected frame
lists against the real cached inference records so a regression would break the
quality-check guarantee that "every annotation matches its stored predictions".
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.build_human_galleries import (
    SEGMENTS,
    annotate,
    _fmt_p,
    pose_groups,
    transition_strip,
)

CACHE = Path("data/s_to_e_all_s_records.json")


def _rec(fid: int, pred: str = "S", p_s: float = 0.5, p_e: float = 0.4) -> dict:
    return {"frame_id": fid, "pred": pred, "p_s": p_s, "p_e": p_e,
            "margin": p_e - p_s,
            "path": f"data/raw/asl_alphabet_train/S/S{fid:04d}.jpg"}


# --- annotation format -------------------------------------------------------
def test_fmt_p_drops_leading_zero():
    assert _fmt_p(0.20) == ".20"
    assert _fmt_p(0.54) == ".54"
    assert _fmt_p(1.0) == "1.00"


def test_annotate_matches_requested_format():
    rec = {"path": "data/raw/asl_alphabet_train/S/S2575.jpg", "pred": "E",
           "p_s": 0.20, "p_e": 0.54}
    assert annotate(rec) == "S2575 | pred=E | S=.20 | E=.54"


# --- transition_strip (synthetic) ---------------------------------------------
def _run_records(seg_start: int, seg_end: int, n_before: int = 10,
                 n_after: int = 10) -> list:
    """Synthetic records: correct S around a solid error run [start, end]."""
    recs = [_rec(i) for i in range(seg_start - n_before, seg_start)]
    recs += [_rec(i, pred="E", p_e=0.6, p_s=0.2)
             for i in range(seg_start, seg_end + 1)]
    recs += [_rec(i) for i in range(seg_end + 1, seg_end + 1 + n_after)]
    return recs


def test_transition_strip_group_counts():
    tiles = transition_strip(_run_records(500, 519), 500, 519)
    phases = [t["phase"] for t in tiles]
    assert phases.count("before") == 4
    assert phases.count("first") == 8
    assert phases.count("middle") == 3
    assert phases.count("final") == 4
    assert phases.count("after") == 4


def test_transition_strip_is_chronological_and_dedup():
    tiles = transition_strip(_run_records(500, 519), 500, 519)
    ids = [t["rec"]["frame_id"] for t in tiles]
    assert ids == sorted(ids)
    assert len(ids) == len(set(ids))


def test_transition_strip_before_after_are_correct_s_only():
    tiles = transition_strip(_run_records(500, 519), 500, 519)
    for t in tiles:
        if t["phase"] in ("before", "after"):
            assert t["rec"]["pred"] == "S"
        else:
            assert t["rec"]["pred"] == "E"


def test_transition_strip_first_error_is_seg_start():
    tiles = transition_strip(_run_records(500, 519), 500, 519)
    first_phase = next(t for t in tiles if t["phase"] == "first")
    assert first_phase["rec"]["frame_id"] == 500
    assert first_phase["rec"]["pred"] == "E"


def test_transition_strip_middle_force_includes_max_pe():
    recs = _run_records(500, 519)
    recs.append(_rec(510, pred="E", p_e=0.99, p_s=0.01))
    tiles = transition_strip(recs, 500, 519)
    middle = [t["rec"]["frame_id"] for t in tiles if t["phase"] == "middle"]
    assert 510 in middle


def test_transition_strip_includes_internal_blips():
    recs = _run_records(500, 519)
    recs.append(_rec(505, pred="S"))  # correct blip inside the run
    tiles = transition_strip(recs, 500, 519)
    blips = [t["rec"]["frame_id"] for t in tiles if t["phase"] == "blip"]
    assert 505 in blips


def test_transition_strip_raises_when_no_errors():
    recs = [_rec(i) for i in range(100, 110)]
    with pytest.raises(ValueError):
        transition_strip(recs, 100, 105)


def test_transition_strip_short_run_still_covers_run():
    recs = _run_records(500, 506)  # 7-error run (shorter than first_k+final_k)
    tiles = transition_strip(recs, 500, 506)
    ids = [t["rec"]["frame_id"] for t in tiles]
    for i in range(500, 507):
        assert i in ids


# --- pose_groups --------------------------------------------------------------
def test_pose_groups_include_first_frame_of_each_run():
    recs = []
    for _, s, e in SEGMENTS:
        recs.append(_rec(s, pred="E", p_e=0.6, p_s=0.2))
        recs.append(_rec(s + 1, pred="E", p_e=0.6, p_s=0.2))
        recs.append(_rec(s - 1))
        recs.append(_rec(e))
    errors = [r for r in recs if r["pred"] == "E"]
    correct = [r for r in recs if r["pred"] == "S"]
    groups = dict(pose_groups(errors, correct))
    first_frames = [r["frame_id"] for r in groups["First-frame S->E"]]
    for _, s, _ in SEGMENTS:
        assert s in first_frames


# --- integration against real cached inference ---------------------------------
@pytest.mark.skipif(not CACHE.is_file(),
                    reason="cached inference records not present")
def test_transition_strip_exact_frames_from_cache():
    records = json.loads(CACHE.read_text())
    expected = {
        "Run 1": {
            "ids": [2570, 2572, 2573, 2574, 2575, 2576, 2577, 2578, 2579,
                    2580, 2581, 2582, 2583, 2584, 2601, 2615, 2616, 2617,
                    2618, 2619, 2620, 2621, 2622, 2623],
            "phases": ["before"] * 4 + ["first"] * 8 + ["blip", "middle",
                     "middle", "middle"] + ["final"] * 4 + ["after"] * 4,
        },
        "Run 2": {
            "ids": [2795, 2797, 2798, 2800, 2801, 2802, 2803, 2804, 2805,
                    2806, 2807, 2808, 2809, 2810, 2811, 2812, 2813, 2814,
                    2815, 2816, 2817, 2818],
            "phases": ["before"] * 4 + ["first"] * 8 + ["middle", "final",
                     "blip", "final", "final", "final"] + ["after"] * 4,
        },
        "Run 3": {
            "ids": [2827, 2828, 2829, 2830, 2831, 2832, 2833, 2834, 2835,
                    2836, 2837, 2838, 2839, 2854, 2864, 2871, 2872, 2873,
                    2874, 2875, 2876, 2877, 2878, 2879, 2880, 2881],
            "phases": ["before"] * 4 + ["first"] * 8 + ["middle"] * 3 +
                     ["final", "blip", "blip", "blip", "final", "final",
                      "final"] + ["after"] * 4,
        },
    }
    for label, s, e in SEGMENTS:
        tiles = transition_strip(records, s, e)
        got_ids = [t["rec"]["frame_id"] for t in tiles]
        got_phases = [t["phase"] for t in tiles]
        exp = expected[label]
        assert got_ids == exp["ids"], f"{label} frame ids mismatch"
        assert got_phases == exp["phases"], f"{label} phases mismatch"


@pytest.mark.skipif(not CACHE.is_file(),
                    reason="cached inference records not present")
def test_annotations_match_cached_probabilities():
    records = json.loads(CACHE.read_text())
    by_id = {r["frame_id"]: r for r in records}
    for label, s, e in SEGMENTS:
        for t in transition_strip(records, s, e):
            rec = by_id[t["rec"]["frame_id"]]
            assert annotate(rec) == (
                f"S{rec['frame_id']} | pred={rec['pred']} | "
                f"S={_fmt_p(rec['p_s'])} | E={_fmt_p(rec['p_e'])}"
            )
