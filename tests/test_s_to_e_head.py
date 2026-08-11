"""Tests for the S->E classifier-head diagnosis (``analyze_s_to_e_head.py``).

Locks the numerically-verifiable contract: Linear-head extraction (structure,
shapes, eval-dropout identity, ``model(x) == head(backbone(x))``), the
``margin_SE = logit_S - logit_E`` calculation, the feature+bias decomposition
``(w_S-w_E)^T x + (b_S-b_E)``, the signed geometric boundary distance
``margin / ||w_S-w_E||``, weight cosine, the embedding-neighbor join, and the
integrated report invariants (138/462/600 groups, all 138 margins negative,
six representative records matching both the full records and the cached
inference). Diagnosis only — no fix is tested or proposed.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from scripts.analyze_s_to_e_head import (
    DEFAULT_GALLERY_DIR,
    DEFAULT_NPZ,
    DEFAULT_OUT_JSON,
    DEFAULT_S_CACHE,
    QUERY_FRAMES,
    decompose_margin,
    extract_head,
    group_summary,
    load_embeddings,
    margin_se,
    neighbor_join,
    pearson,
    signed_boundary_distance,
    spearman,
    weight_cosine,
)

CHECKPOINT = Path("checkpoints/efficientnet_b0_targeted.pt")
CACHE = Path(DEFAULT_S_CACHE)
NPZ = Path(DEFAULT_NPZ)
OUT_JSON = Path(DEFAULT_OUT_JSON)


def _model() -> torch.nn.Module:
    return torch.nn.Sequential(
        torch.nn.Identity(),
        torch.nn.Sequential(torch.nn.Dropout(0.3),
                            torch.nn.Linear(1280, 29)),
    )


# --- extract_head -------------------------------------------------------------
def test_extract_head_returns_w_b_info():
    model = _model()
    model.eval()
    W, b, info = extract_head(model)
    assert W.shape == (29, 1280)
    assert b.shape == (29,)
    assert info["layers"] == ["Dropout(p=0.3)", "Linear(1280, 29)"]
    assert info["weight_shape"] == [29, 1280]
    assert info["dropout_eval_is_identity"] is True


def test_extract_head_rejects_wrong_container():
    with pytest.raises(ValueError):
        extract_head(torch.nn.Sequential(torch.nn.Linear(4, 4),
                                         torch.nn.Linear(4, 29)))


def test_extract_head_rejects_wrong_head_dropout():
    model = torch.nn.Sequential(
        torch.nn.Identity(),
        torch.nn.Sequential(torch.nn.Dropout(0.5),
                            torch.nn.Linear(1280, 29)),
    )
    with pytest.raises(ValueError):
        extract_head(model)


def test_extract_head_rejects_wrong_linear_dims():
    model = torch.nn.Sequential(
        torch.nn.Identity(),
        torch.nn.Sequential(torch.nn.Dropout(0.3),
                            torch.nn.Linear(1280, 10)),
    )
    with pytest.raises(ValueError):
        extract_head(model)


def test_dropout_identity_in_eval_forward():
    """model(x) must equal head(backbone(x)): dropout is identity in eval."""
    model = _model()
    model.eval()
    x = torch.randn(4, 1280)
    backbone, head = model[0], model[1]
    assert torch.allclose(model(x), head(backbone(x)), atol=1e-6)
    # the head's Linear, not the dropout module, defines the logits
    W, b, _ = extract_head(model)
    manual = x.detach().numpy() @ W.T + b
    assert np.allclose(manual, head(x).detach().numpy(), atol=1e-5)


@pytest.mark.skipif(not CHECKPOINT.is_file(),
                    reason="targeted checkpoint not present")
def test_extract_head_real_checkpoint():
    from src.evaluate import load_checkpoint
    model = load_checkpoint(str(CHECKPOINT), "cpu")
    W, b, info = extract_head(model)
    assert W.shape == (29, 1280)
    assert b.shape == (29,)
    assert info["layers"] == ["Dropout(p=0.3)", "Linear(1280, 29)"]
    model.eval()
    x = torch.randn(2, 3, 224, 224)
    assert torch.allclose(model(x), model[1](model[0](x)), atol=1e-5)


# --- margin / decomposition / distance / cosine --------------------------------
def test_margin_se():
    assert margin_se(3.0, 5.0) == -2.0
    assert margin_se(5.0, 3.0) == 2.0


def test_decompose_margin_equals_margin():
    w_s = np.array([1.0, 0.0])
    w_e = np.array([0.0, 1.0])
    b_s, b_e = 0.5, 0.1
    x = np.array([2.0, 3.0])
    feature, bias = decompose_margin(w_s, w_e, b_s, b_e, x)
    assert feature == pytest.approx(-1.0)  # (1-0)*2 + (0-1)*3
    assert bias == pytest.approx(0.4)
    logit_s = w_s @ x + b_s
    logit_e = w_e @ x + b_e
    assert feature + bias == pytest.approx(margin_se(logit_s, logit_e))


def test_signed_boundary_distance():
    w_s = np.array([1.0, 0.0])
    w_e = np.array([0.0, 1.0])
    assert signed_boundary_distance(-2.0, w_s, w_e) == pytest.approx(
        -2.0 / math.sqrt(2))
    assert signed_boundary_distance(2.0, w_s, w_e) == pytest.approx(
        2.0 / math.sqrt(2))


def test_signed_boundary_distance_zero_weight_diff():
    w = np.array([1.0, 0.0])
    assert np.isnan(signed_boundary_distance(1.0, w, w))


def test_weight_cosine():
    assert weight_cosine([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)
    assert weight_cosine([1.0, 1.0], [1.0, 1.0]) == pytest.approx(1.0)
    assert weight_cosine([1.0, 1.0], [-1.0, 1.0]) == pytest.approx(0.0)
    assert np.isnan(weight_cosine([0.0, 0.0], [1.0, 0.0]))


# --- correlations -------------------------------------------------------------
def test_pearson_and_spearman_perfect_linear():
    x = list(range(20))
    y = [2 * v + 1 for v in x]
    assert pearson(x, y)["r"] == pytest.approx(1.0)
    assert spearman(x, y)["r"] == pytest.approx(1.0)


def test_correlation_constant_input_is_none():
    assert pearson([1.0] * 5, [0.0] * 5) is None
    assert spearman([1.0] * 5, [0.0] * 5) is None


# --- group_summary ------------------------------------------------------------
def test_group_summary_keys():
    recs = [
        {"logit_S": 1.0, "logit_E": 2.0, "margin_SE": -1.0, "P(S)": 0.2,
         "P(E)": 0.6, "feature_term": -1.2, "bias_term": 0.2,
         "distance_SE": -0.3},
        {"logit_S": 3.0, "logit_E": 2.0, "margin_SE": 1.0, "P(S)": 0.7,
         "P(E)": 0.1, "feature_term": 0.8, "bias_term": 0.2,
         "distance_SE": 0.3},
    ]
    s = group_summary(recs)
    assert s["n"] == 2
    assert s["margin_SE"]["mean"] == pytest.approx(0.0)
    assert s["margin_SE"]["median"] == pytest.approx(0.0)
    assert s["bias_term"] == pytest.approx(0.2)
    assert s["margin_SE_quantiles"]["50%"] == pytest.approx(0.0)
    assert s["distance_SE_quantiles"]["50%"] == pytest.approx(0.0)


# --- neighbor_join (synthetic) -------------------------------------------------
def test_neighbor_join_synthetic(tmp_path):
    cache = [{"frame_id": 10, "pred": "S"}, {"frame_id": 11, "pred": "E"},
             {"frame_id": 12, "pred": "S"}]
    cp = tmp_path / "cache.json"
    cp.write_text(json.dumps(cache))
    s_ids = np.array([10, 11, 12])
    s_embs = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                      dtype=np.float32)
    e_ids = np.array([50, 51])
    e_embs = np.array([[1.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    errors = [{"frame_id": 11}]
    rows = neighbor_join(errors, s_embs, s_ids, e_embs, e_ids, str(cp))
    assert rows[0]["frame_id"] == 11
    # correct-S pool {10,12} both at cos 0 vs [0,1,0] -> tie -> frame 10
    assert rows[0]["nearest_s_cosine"] == pytest.approx(0.0)
    # nearest E [1,1,0]/sqrt2 vs [0,1,0] -> 1/sqrt2
    assert rows[0]["nearest_e_cosine"] == pytest.approx(1.0 / math.sqrt(2))
    assert rows[0]["nearest_s_minus_nearest_e"] == pytest.approx(
        -1.0 / math.sqrt(2))


# --- integration against generated report --------------------------------------
@pytest.mark.skipif(not CACHE.is_file(),
                    reason="cached inference records not present")
def test_representative_matches_cached_probabilities():
    report = json.loads(OUT_JSON.read_text())
    cache = {r["frame_id"]: r for r in json.loads(CACHE.read_text())}
    for fid in QUERY_FRAMES:
        q = report["representative"]["samples"][str(fid)]
        c = cache[fid]
        assert c["pred"] == "E"
        assert q["pred"] == "E"
        assert q["P(S)"] == pytest.approx(c["p_s"], abs=1e-5)
        assert q["P(E)"] == pytest.approx(c["p_e"], abs=1e-5)


@pytest.mark.skipif(not OUT_JSON.is_file(),
                    reason="head report not generated")
def test_report_groups_and_counts():
    report = json.loads(OUT_JSON.read_text())
    g = report["groups"]
    assert g["s_to_e_errors"]["n"] == 138
    assert g["correct_s"]["n"] == 462
    assert g["true_e"]["n"] == 600
    assert report["classifier_head"]["layers"] == [
        "Dropout(p=0.3)", "Linear(1280, 29)"]
    assert report["classifier_head"]["weight_shape"] == [29, 1280]
    assert report["classifier_head"]["dropout_eval_is_identity"] is True
    assert report["classifier_head"]["verified_against_cache"][
        "pred_mismatches_vs_cache"] == 0


@pytest.mark.skipif(not OUT_JSON.is_file(),
                    reason="head report not generated")
def test_all_138_records_negative_margin():
    report = json.loads(OUT_JSON.read_text())
    recs = report["full_s_to_e_records"]
    assert len(recs) == 138
    assert all(r["margin_SE"] < 0 for r in recs)
    assert all(r["pred"] == "E" for r in recs)
    assert all(r["true"] == "S" for r in recs)


@pytest.mark.skipif(not OUT_JSON.is_file(),
                    reason="head report not generated")
def test_decomposition_reconstruction_in_report():
    report = json.loads(OUT_JSON.read_text())
    for r in report["full_s_to_e_records"]:
        assert r["feature_term"] + r["bias_term"] == pytest.approx(
            r["margin_SE"], abs=1e-5)


@pytest.mark.skipif(not OUT_JSON.is_file(),
                    reason="head report not generated")
def test_embedding_neighbor_join_in_report():
    report = json.loads(OUT_JSON.read_text())
    emb = report["embedding_vs_head"]
    assert emb["n_errors"] == 138
    assert emb["all_errors_nearest_s_gt_nearest_e"] is True
    counts = emb["count_strongly_s_like_on_e_side"]
    assert counts["nearest_S - nearest_E > 0.0 AND margin_SE < 0"] == 138
    r = emb["correlations"]["nearest_S_minus_nearest_E_vs_margin_SE"]["pearson"]
    assert 0.0 < r["r"] < 1.0
    assert r["n"] == 138


@pytest.mark.skipif(not OUT_JSON.is_file(),
                    reason="head report not generated")
def test_representative_all_s_like_but_negative_margin():
    report = json.loads(OUT_JSON.read_text())
    samples = report["representative"]["samples"]
    for fid in QUERY_FRAMES:
        q = samples[str(fid)]
        assert q["margin_SE"] < 0
        assert q["s_like_but_margin_negative"] is True


@pytest.mark.skipif(not (Path(DEFAULT_GALLERY_DIR) / "margin_distribution.png").is_file(),
                    reason="head gallery not generated")
def test_gallery_files_exist():
    d = Path(DEFAULT_GALLERY_DIR)
    for name in ("margin_distribution.png", "boundary_distance_distribution.png",
                 "embedding_vs_margin.png", "class_head_parameters.png"):
        assert (d / name).is_file()


@pytest.mark.skipif(not NPZ.is_file(),
                    reason="embedding cache not generated")
def test_embeddings_cache_consistent_with_report():
    report = json.loads(OUT_JSON.read_text())
    emb = load_embeddings(str(NPZ))
    assert len(emb["s_ids"]) == 600
    assert len(emb["e_ids"]) == 600
    assert emb["s_embs"].shape == (600, 1280)
    assert emb["e_embs"].shape == (600, 1280)
    by_id = {r["frame_id"]: r for r in report["full_s_to_e_records"]}
    assert all(fid in by_id for fid in QUERY_FRAMES)
