"""Tests for the S->E embedding-neighbor diagnosis.

Locks the numerically-verifiable contract of ``analyze_s_to_e_embeddings``:
L2 normalization, cosine-based top-k ranking with self-exclusion and tie
determinism, the nearest-correct-S vs nearest-actual-E comparison, and the
integrated pipeline invariants (reports/galleries written, S/E pools populated,
queries present and actually S->E). All of it is diagnosis only — no fix is
proposed anywhere in the script.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts.analyze_s_to_e_embeddings import (
    DEFAULT_GALLERY_DIR,
    DEFAULT_NPZ,
    DEFAULT_OUT_JSON,
    DEFAULT_OUT_MD,
    QUERY_FRAMES,
    build_pools,
    compare_nearest,
    cosine_top_k,
    l2_normalize,
    query_records,
    summarize_nearest,
)

CACHE = Path("data/s_to_e_all_s_records.json")
OUT_JSON = Path(DEFAULT_OUT_JSON)
OUT_MD = Path(DEFAULT_OUT_MD)
NPZ = Path(DEFAULT_NPZ)


def _emb(*vals: float) -> np.ndarray:
    return np.asarray(vals, dtype=np.float32)


# --- l2_normalize --------------------------------------------------------------
def test_l2_normalize_rows_have_unit_norm():
    mat = np.array([[3.0, 4.0], [0.0, 0.0], [1.0, 2.0]], dtype=np.float32)
    out = l2_normalize(mat)
    norms = np.linalg.norm(out, axis=1)
    assert np.allclose(norms, [1.0, 0.0, 1.0])  # zero row stays zero


def test_l2_normalize_rejects_1d():
    with pytest.raises(ValueError):
        l2_normalize(_emb(1.0, 2.0, 3.0))


def test_l2_normalize_preserves_unit_rows_unchanged():
    row = _emb(1.0, 0.0)
    out = l2_normalize(row[np.newaxis, :])
    assert np.allclose(out[0], row)


# --- cosine_top_k -------------------------------------------------------------
def _pool() -> dict:
    return {10: _emb(1, 0), 20: _emb(0, 1), 30: _emb(0.8, 0.6), 40: _emb(-1, 0)}


def test_cosine_top_k_ranks_by_cosine_desc():
    q = _emb(1.0, 0.1)
    top = cosine_top_k(l2_normalize(q[np.newaxis, :])[0],
                       {k: l2_normalize(v[np.newaxis, :])[0] for k, v in _pool().items()},
                       k=2)
    assert [fid for fid, _ in top] == [10, 30]
    assert top[0][1] > top[1][1]


def test_cosine_top_k_self_exclusion():
    q = _emb(1.0, 0.0)
    pool = {k: l2_normalize(v[np.newaxis, :])[0] for k, v in _pool().items()}
    pool[10] = l2_normalize(q[np.newaxis, :])[0]  # exact self at id 10
    top = cosine_top_k(l2_normalize(q[np.newaxis, :])[0], pool, k=3, exclude_id=10)
    assert 10 not in [fid for fid, _ in top]
    assert len(top) == 3


def test_cosine_top_k_ties_resolve_to_lower_frame_id():
    # two pool vectors both at cosine 0.6 vs q
    q = l2_normalize(_emb(1.0, 1.0)[np.newaxis, :])[0]
    pool = {50: l2_normalize(_emb(1.0, 1.0)[np.newaxis, :])[0],
            30: l2_normalize(_emb(1.0, 1.0)[np.newaxis, :])[0]}
    top = cosine_top_k(q, pool, k=2)
    assert [fid for fid, _ in top] == [30, 50]


def test_cosine_top_k_k_invalid():
    q = l2_normalize(_emb(1.0, 0.0)[np.newaxis, :])[0]
    with pytest.raises(ValueError):
        cosine_top_k(q, {10: q}, k=0)


def test_cosine_top_k_empty_pool():
    q = l2_normalize(_emb(1.0, 0.0)[np.newaxis, :])[0]
    assert cosine_top_k(q, {}, k=3) == []


# --- compare_nearest ----------------------------------------------------------
def _normalized_dict(mapping: dict) -> dict:
    return {k: l2_normalize(v[np.newaxis, :])[0] for k, v in mapping.items()}


def test_compare_nearest_closer_labels():
    pool_s = _normalized_dict({10: _emb(1.0, 0.0), 11: _emb(0.95, 0.31)})
    pool_e = _normalized_dict({20: _emb(0.0, 1.0)})
    q_e = l2_normalize(_emb(0.5, 0.866)[np.newaxis, :])[0]
    q_s = l2_normalize(_emb(0.99, 0.02)[np.newaxis, :])[0]
    rows = compare_nearest({101: q_e, 102: q_s}, pool_s, pool_e)
    by_id = {r["frame_id"]: r for r in rows}
    assert by_id[101]["closer"] == "E"
    assert by_id[102]["closer"] == "S"
    assert by_id[101]["nearest_e_minus_s"] > 0
    assert by_id[102]["nearest_e_minus_s"] < 0


def test_compare_nearest_excludes_self():
    pool_s = _normalized_dict({5: _emb(1.0, 0.0), 7: _emb(0.98, 0.2)})
    pool_e = _normalized_dict({6: _emb(0.0, 1.0)})
    q = l2_normalize(_emb(1.0, 0.0)[np.newaxis, :])[0]
    rows = compare_nearest({5: q}, pool_s, pool_e)
    assert len(rows) == 1
    assert rows[0]["frame_id"] == 5
    assert rows[0]["nearest_s_sim"] < 1.0  # self excluded


# --- summarize_nearest ---------------------------------------------------------
def test_summarize_nearest_counts_and_stats():
    rows = [
        {"frame_id": 1, "nearest_s_sim": 0.5, "nearest_e_sim": 0.8,
         "nearest_e_minus_s": 0.3, "closer": "E"},
        {"frame_id": 2, "nearest_s_sim": 0.6, "nearest_e_sim": 0.5,
         "nearest_e_minus_s": -0.1, "closer": "S"},
        {"frame_id": 3, "nearest_s_sim": 0.4, "nearest_e_sim": 0.4,
         "nearest_e_minus_s": 0.0, "closer": "tie"},
    ]
    s = summarize_nearest(rows)
    assert s["n"] == 3
    assert s["closer_to_e"] == 1
    assert s["closer_to_s"] == 1
    assert s["ties"] == 1
    assert s["nearest_e_minus_nearest_s"]["mean"] == pytest.approx(0.2 / 3)


def test_summarize_nearest_empty():
    s = summarize_nearest([])
    assert s["n"] == 0
    assert np.isnan(s["nearest_e_minus_nearest_s"]["mean"])


# --- integration against real cached inference --------------------------------
@pytest.mark.skipif(not CACHE.is_file(),
                    reason="cached inference records not present")
def test_query_frames_are_s_to_e_in_cache():
    records = json.loads(CACHE.read_text())
    by_id = {r["frame_id"]: r for r in records}
    for fid in QUERY_FRAMES:
        assert fid in by_id, f"S{fid} missing from cache"
        assert by_id[fid]["pred"] == "E", f"S{fid} is not an S->E error"


@pytest.mark.skipif(not CACHE.is_file(),
                    reason="cached inference records not present")
def test_query_records_returns_query_order():
    records = json.loads(CACHE.read_text())
    by_id = {r["frame_id"]: r for r in records}
    recs = query_records(by_id)
    assert [r["frame_id"] for r in recs] == QUERY_FRAMES
    assert all(r["pred"] == "E" for r in recs)


@pytest.mark.skipif(not OUT_JSON.is_file(),
                    reason="embedding report not generated")
def test_report_schema_and_pools():
    report = json.loads(OUT_JSON.read_text())
    assert report["embedding"]["dim"] == 1280
    assert "model[0]" in report["embedding"]["layer"]
    assert report["pools"]["n_correct_s"] > 0
    assert report["pools"]["n_actual_e"] > 0
    assert report["s_to_e_138"]["n"] == 138
    assert report["correct_s_control"]["n"] == 462
    for fid in QUERY_FRAMES:
        q = report["representative_queries"][str(fid)]
        assert q["pred"] == "E"
        assert len(q["s_neighbors"]) == 5
        assert len(q["e_neighbors"]) == 5
        assert all(nb["pred"] == "S" for nb in q["s_neighbors"])


@pytest.mark.skipif(not OUT_JSON.is_file(),
                    reason="embedding report not generated")
def test_report_counts_tie_to_138():
    report = json.loads(OUT_JSON.read_text())
    s = report["s_to_e_138"]
    assert s["closer_to_e"] + s["closer_to_s"] + s["ties"] == s["n"] == 138


@pytest.mark.skipif(not NPZ.is_file(),
                    reason="embedding cache not generated")
def test_npz_embedding_shapes():
    data = np.load(NPZ)
    assert data["s_embs"].shape == (len(data["s_ids"]), 1280)
    assert data["e_embs"].shape == (len(data["e_ids"]), 1280)
    assert len(data["s_ids"]) == len(set(data["s_ids"].tolist()))
    assert len(data["e_ids"]) == len(set(data["e_ids"].tolist()))


@pytest.mark.skipif(not (Path(DEFAULT_GALLERY_DIR) / "S2575_neighbors.png").is_file(),
                    reason="neighbor gallery not generated")
def test_gallery_images_exist():
    d = Path(DEFAULT_GALLERY_DIR)
    for fid in QUERY_FRAMES:
        assert (d / f"S{fid}_neighbors.png").is_file()
