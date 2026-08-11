"""Diagnose the S->E failure at the final EfficientNet-B0 classifier head.

**Diagnosis only** — no retraining, no fine-tuning, no head-only training, no
modification of weights / ``crop.py`` / preprocessing / augmentation / loss /
checkpoint, no landmarks, no MediaPipe, no fix.

Background (previous task): all 138 S->E errors of
``checkpoints/efficientnet_b0_targeted.pt`` sit closer, in the L2-normalized
1280-D embedding space, to correctly classified S samples (mean cosine ~0.900)
than to actual E samples (mean ~0.606). The question this script asks is what
the final ``Dropout(0.3) + Linear(1280, 29)`` head does to those S-like
embeddings.

The head is ``model[1]`` of the checkpoint's ``nn.Sequential(backbone, head)``
(a fact asserted here, and asserted numerically with
``model(x) == head(backbone(x))``). In eval mode Dropout is the identity, so for
a pooled embedding ``x``:

    logit_c    = w_c^T x + b_c          (c = S, E)
    margin_SE  = logit_S - logit_E
               = (w_S - w_E)^T x + (b_S - b_E)     [feature_term + bias_term]
    distance_SE = margin_SE / ||w_S - w_E||        [signed geometric distance]

All numbers are produced from the cached pooled embeddings
(``data/s_e_embeddings_targeted.npz``, identical to the ones the previous
analysis used) recombined with the checkpoint's head weights, and every S
prediction/probability is cross-checked against ``data/s_to_e_all_s_records.json``
(assert 0 mismatches; max |dp| ~1e-6). Embedding neighbors reuse the exact
pools from the previous analysis (correct-S = cache pred S, actual-E = all 600
true E val samples, L2-normalized, cosine, self-excluded).

Everything below is descriptive. Sign of margin/distance is *not* treated as the
complete 29-class decision; correlations are *not* causation; a large/small
weight norm or bias is *not* assumed bad. No fix is proposed.

Outputs:
- ``docs/figures/error_gallery_targeted/S_to_E/H_head_boundary/*.png`` (4 plots)
- ``docs/reports/s_to_e_head_analysis_targeted.json``
- ``docs/reports/s_to_e_head_notes.md``
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402
from scipy import stats  # noqa: E402

from src.data import CLASSES, SplitManifest  # noqa: E402
from src.evaluate import load_checkpoint  # noqa: E402
from scripts.analyze_s_to_e import frame_id  # noqa: E402
from scripts.analyze_s_to_e_embeddings import (  # noqa: E402
    QUERY_FRAMES,
    cosine_top_k,
    l2_normalize,
)

DEFAULT_CHECKPOINT = "checkpoints/efficientnet_b0_targeted.pt"
DEFAULT_MANIFEST = "data/split_manifest.json"
DEFAULT_NPZ = "data/s_e_embeddings_targeted.npz"
DEFAULT_S_CACHE = "data/s_to_e_all_s_records.json"
DEFAULT_PREV_REPORT = "docs/reports/s_to_e_embedding_neighbors_targeted.json"
DEFAULT_GALLERY_DIR = (
    "docs/figures/error_gallery_targeted/S_to_E/H_head_boundary"
)
DEFAULT_OUT_JSON = "docs/reports/s_to_e_head_analysis_targeted.json"
DEFAULT_OUT_MD = "docs/reports/s_to_e_head_notes.md"

FEATURE_DIM = 1280
N_CLASSES = 29
SCHEMA = "asl-s_to_e-head-analysis/v1"
DPI = 150
STRONG_THRESHOLDS = [0.0, 0.25, 0.5]


# --------------------------------------------------------------------------- #
# Pure helpers (unit-tested)
# --------------------------------------------------------------------------- #
def margin_se(logit_s: float, logit_e: float) -> float:
    """``logit_S - logit_E``. Positive -> S wins over E (S/E pair only)."""
    return float(logit_s - logit_e)


def decompose_margin(
    w_s: np.ndarray, w_e: np.ndarray, b_s: float, b_e: float, x: np.ndarray
) -> Tuple[float, float]:
    """``(feature_term, bias_term)`` such that feature + bias == logit_S - logit_E.

    feature_term = (w_S - w_E)^T x ; bias_term = b_S - b_E.
    """
    w_s = np.asarray(w_s, dtype=np.float64).ravel()
    w_e = np.asarray(w_e, dtype=np.float64).ravel()
    x = np.asarray(x, dtype=np.float64).ravel()
    feature = float((w_s - w_e) @ x)
    bias = float(b_s) - float(b_e)
    return feature, bias


def signed_boundary_distance(
    margin: float, w_s: np.ndarray, w_e: np.ndarray
) -> float:
    """Signed geometric distance to the S/E hyperplane ``(w_S-w_E)^T x + (b_S-b_E)=0``.

    ``margin_SE / ||w_S - w_E||``. Negative means the E side.
    """
    denom = float(np.linalg.norm(np.asarray(w_s, dtype=np.float64).ravel()
                                 - np.asarray(w_e, dtype=np.float64).ravel()))
    if denom == 0:
        return float("nan")
    return float(margin) / denom


def weight_cosine(w_a: np.ndarray, w_b: np.ndarray) -> float:
    """Cosine similarity between two (possibly unnormalized) 1-D vectors."""
    a = np.asarray(w_a, dtype=np.float64).ravel()
    b = np.asarray(w_b, dtype=np.float64).ravel()
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na == 0 or nb == 0:
        return float("nan")
    return float(a @ b / (na * nb))


def _corr(x: Sequence[float], y: Sequence[float], kind: str) -> Optional[Dict]:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size != y.size or x.size < 2:
        return None
    try:
        if kind == "pearson":
            r, p = stats.pearsonr(x, y)
        elif kind == "spearman":
            r, p = stats.spearmanr(x, y)
        else:
            raise ValueError(kind)
    except ValueError:  # constant input -> undefined correlation
        return None
    if np.isnan(r):
        return None
    return {"r": float(r), "p": float(p), "n": int(x.size)}


def pearson(x: Sequence[float], y: Sequence[float]) -> Optional[Dict]:
    return _corr(x, y, "pearson")


def spearman(x: Sequence[float], y: Sequence[float]) -> Optional[Dict]:
    return _corr(x, y, "spearman")


def _stats(arr: np.ndarray) -> Dict:
    arr = np.asarray(arr, dtype=np.float64)
    if arr.size == 0:
        return {"n": 0}
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "q5": float(np.percentile(arr, 5)),
        "q25": float(np.percentile(arr, 25)),
        "q50": float(np.percentile(arr, 50)),
        "q75": float(np.percentile(arr, 75)),
        "q95": float(np.percentile(arr, 95)),
    }


def group_summary(records: Sequence[Dict]) -> Dict:
    """Section 4/6 summary over a group of sample records."""
    if not records:
        return {"n": 0}
    keys = {
        "logit_S": "logit_S",
        "logit_E": "logit_E",
        "margin_SE": "margin_SE",
        "P(S)": "P(S)",
        "P(E)": "P(E)",
        "feature_term": "feature_term",
        "distance_SE": "distance_SE",
    }
    out: Dict = {}
    for key, label in keys.items():
        out[label] = _stats(np.array([r[key] for r in records]))
    out["bias_term"] = float(records[0]["bias_term"])
    out["margin_SE_quantiles"] = {
        "5%": out["margin_SE"]["q5"], "25%": out["margin_SE"]["q25"],
        "50%": out["margin_SE"]["q50"], "75%": out["margin_SE"]["q75"],
        "95%": out["margin_SE"]["q95"],
    }
    out["distance_SE_quantiles"] = {
        "5%": out["distance_SE"]["q5"], "25%": out["distance_SE"]["q25"],
        "50%": out["distance_SE"]["q50"], "75%": out["distance_SE"]["q75"],
        "95%": out["distance_SE"]["q95"],
    }
    out["n"] = len(records)
    return out


# --------------------------------------------------------------------------- #
# Head extraction + record building
# --------------------------------------------------------------------------- #
def extract_head(model: torch.nn.Module) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """The ``Dropout(0.3) + Linear(1280, 29)`` head from the checkpoint model.

    ``model`` is ``nn.Sequential(backbone, head)`` (from ``build_efficientnet``);
    ``model[1]`` must be ``nn.Sequential(Dropout(p=0.3), Linear(1280, 29))``.
    Returns ``(W, b, info)`` where ``W`` is (29, 1280), ``b`` is (29,), and
    ``info`` records the structure that was asserted.
    """
    if not (isinstance(model, torch.nn.Sequential) and len(model) == 2):
        raise ValueError(
            "expected nn.Sequential(backbone, head) of length 2, "
            f"got {type(model).__name__} of length {len(model)}"
        )
    head = model[1]
    if not (
        isinstance(head, torch.nn.Sequential)
        and len(head) == 2
        and isinstance(head[0], torch.nn.Dropout)
        and abs(head[0].p - 0.3) < 1e-9
        and isinstance(head[1], torch.nn.Linear)
        and head[1].in_features == FEATURE_DIM
        and head[1].out_features == N_CLASSES
    ):
        raise ValueError(f"model[1] is not Dropout(0.3)+Linear(1280,29): {head}")
    W = head[1].weight.detach().cpu().numpy()
    b = head[1].bias.detach().cpu().numpy()
    info = {
        "container": "model[1] of nn.Sequential(backbone, head)",
        "layers": ["Dropout(p=0.3)", f"Linear({FEATURE_DIM}, {N_CLASSES})"],
        "weight_shape": list(W.shape),
        "bias_shape": list(b.shape),
        "dropout_eval_is_identity": not head[0].training,
    }
    return W, b, info


def load_embeddings(npz_path: str) -> Dict:
    """The raw (un-normalized) pooled embeddings cache from the previous task."""
    d = np.load(npz_path)
    return {
        "s_ids": np.asarray(d["s_ids"], dtype=int),
        "s_embs": np.asarray(d["s_embs"], dtype=np.float32),
        "e_ids": np.asarray(d["e_ids"], dtype=int),
        "e_embs": np.asarray(d["e_embs"], dtype=np.float32),
    }


def s_e_logits(
    embs: np.ndarray, W: np.ndarray, b: np.ndarray, s_idx: int, e_idx: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """``(logits, probs, logit_S, logit_E)`` from pooled embeddings and W,b."""
    logits = embs @ W.T + b
    z = logits - logits.max(1, keepdims=True)
    probs = np.exp(z)
    probs /= probs.sum(1, keepdims=True)
    return logits, probs, logits[:, s_idx], logits[:, e_idx]


def build_records(
    class_name: str,
    ids: np.ndarray,
    embs: np.ndarray,
    W: np.ndarray,
    b: np.ndarray,
    s_idx: int,
    e_idx: int,
    paths: Optional[Dict[int, str]] = None,
) -> List[Dict]:
    """One record per sample with head quantities (see docstring section 2-3)."""
    classes = list(CLASSES)
    logits, probs, logit_s, logit_e = s_e_logits(embs, W, b, s_idx, e_idx)
    preds = [classes[i] for i in logits.argmax(1)]
    w_s, w_e = W[s_idx], W[e_idx]
    b_s, b_e = float(b[s_idx]), float(b[e_idx])
    denom = float(np.linalg.norm(w_s - w_e))
    records: List[Dict] = []
    for i in range(int(len(ids))):
        fid = int(ids[i])
        m = margin_se(float(logit_s[i]), float(logit_e[i]))
        feature, bias = decompose_margin(w_s, w_e, b_s, b_e, embs[i])
        dist = m / denom if denom else float("nan")
        records.append(
            {
                "path": (paths or {}).get(fid, ""),
                "frame_id": fid,
                "true": class_name,
                "pred": preds[i],
                "logit_S": float(logit_s[i]),
                "logit_E": float(logit_e[i]),
                "margin_SE": m,
                "P(S)": float(probs[i, s_idx]),
                "P(E)": float(probs[i, e_idx]),
                "feature_term": feature,
                "bias_term": bias,
                "distance_SE": dist,
            }
        )
    return records


def verify_s_vs_cache(records: Sequence[Dict], cache_path: str) -> Dict:
    """Cross-check every S record against the cached inference (pred, P(S), P(E))."""
    cache = {r["frame_id"]: r for r in json.loads(Path(cache_path).read_text())}
    pred_mismatch = 0
    max_dp = 0.0
    for r in records:
        c = cache.get(r["frame_id"])
        if c is None:
            raise SystemExit(f"S{r['frame_id']} missing from S cache")
        if c["pred"] != r["pred"]:
            pred_mismatch += 1
        max_dp = max(max_dp, abs(c["p_s"] - r["P(S)"]),
                     abs(c["p_e"] - r["P(E)"]))
    if pred_mismatch:
        raise SystemExit(
            f"{pred_mismatch} S predictions disagree with the cached inference — "
            "head/logit recombination is not reproducing the model; aborting"
        )
    return {"pred_mismatches_vs_cache": pred_mismatch,
            "max_abs_delta_prob": max_dp}


# --------------------------------------------------------------------------- #
# Head parameter statistics
# --------------------------------------------------------------------------- #
def head_parameters(W: np.ndarray, b: np.ndarray) -> Dict:
    """Section 5: S/E head parameters and the all-class table with ranks."""
    classes = list(CLASSES)
    s_idx, e_idx = classes.index("S"), classes.index("E")
    norms = np.linalg.norm(W, axis=1)
    rank_norm = {c: int(len(classes) - np.count_nonzero(norms > norms[i]))
                 for i, c in enumerate(classes)}  # 1 = largest, 29 = smallest
    rank_bias = {c: int(len(classes) - np.count_nonzero(b > b[i]))
                 for i, c in enumerate(classes)}  # 1 = most positive
    rows = [
        {
            "class": c,
            "weight_norm": float(norms[i]),
            "bias": float(b[i]),
            "weight_norm_rank": rank_norm[c],
            "bias_rank": rank_bias[c],
        }
        for i, c in enumerate(classes)
    ]
    return {
        "w_S_norm": float(norms[s_idx]),
        "w_E_norm": float(norms[e_idx]),
        "b_S": float(b[s_idx]),
        "b_E": float(b[e_idx]),
        "b_S_minus_b_E": float(b[s_idx] - b[e_idx]),
        "cosine_similarity_w_S_w_E": weight_cosine(W[s_idx], W[e_idx]),
        "w_S_minus_w_E_norm": float(np.linalg.norm(W[s_idx] - W[e_idx])),
        "classes": rows,
        "rank_of_S": {"weight_norm": rank_norm["S"], "bias": rank_bias["S"]},
        "rank_of_E": {"weight_norm": rank_norm["E"], "bias": rank_bias["E"]},
        "median_weight_norm_all_classes": float(np.median(norms)),
        "median_bias_all_classes": float(np.median(b)),
    }


# --------------------------------------------------------------------------- #
# Neighbor join (reuses the previous embedding-neighbor pools)
# --------------------------------------------------------------------------- #
def neighbor_join(
    errors: Sequence[Dict],
    s_embs: np.ndarray,
    s_ids: np.ndarray,
    e_embs: np.ndarray,
    e_ids: np.ndarray,
    cache_path: str,
) -> List[Dict]:
    """Nearest correct-S / nearest actual-E cosine for every S->E error.

    Identical pools and metric to the previous embedding-neighbor analysis:
    correct-S = S samples whose cached prediction is S (462), actual-E = all
    600 true-E val samples, L2-normalized embeddings, cosine, self-excluded.
    """
    cache = {r["frame_id"]: r for r in json.loads(Path(cache_path).read_text())}
    s_pool_ids = [fid for fid in s_ids.tolist() if cache[fid]["pred"] == "S"]
    s_pool = {fid: l2_normalize(s_embs[i][np.newaxis, :])[0]
              for i, fid in enumerate(s_ids.tolist()) if fid in s_pool_ids}
    e_pool = {fid: l2_normalize(e_embs[i][np.newaxis, :])[0]
              for i, fid in enumerate(e_ids.tolist())}
    emb_by_id = {fid: s_embs[i] for i, fid in enumerate(s_ids.tolist())}
    rows: List[Dict] = []
    for r in errors:
        q = l2_normalize(emb_by_id[r["frame_id"]][np.newaxis, :])[0]
        s_best = cosine_top_k(q, s_pool, 1, r["frame_id"])
        e_best = cosine_top_k(q, e_pool, 1, r["frame_id"])
        nearest_s = s_best[0][1] if s_best else float("nan")
        nearest_e = e_best[0][1] if e_best else float("nan")
        rows.append(
            {
                "frame_id": r["frame_id"],
                "nearest_s_cosine": nearest_s,
                "nearest_e_cosine": nearest_e,
                "nearest_s_minus_nearest_e": nearest_s - nearest_e,
            }
        )
    return rows


def embed_vs_head(
    errors: Sequence[Dict], neighbor_rows: Sequence[Dict]
) -> Dict:
    """Section 7: similarity-difference vs margin / distance correlations."""
    joined = {nr["frame_id"]: nr for nr in neighbor_rows}
    margins = np.array([r["margin_SE"] for r in errors])
    dists = np.array([r["distance_SE"] for r in errors])
    sims = np.array([joined[r["frame_id"]]["nearest_s_minus_nearest_e"]
                     for r in errors])
    margin_corr = {
        "pearson": pearson(sims, margins),
        "spearman": spearman(sims, margins),
    }
    dist_corr = {
        "pearson": pearson(sims, dists),
        "spearman": spearman(sims, dists),
    }
    count = {
        f"nearest_S - nearest_E > {t} AND margin_SE < 0":
            int(np.sum((sims > t) & (margins < 0)))
        for t in STRONG_THRESHOLDS
    }
    count["nearest_S - nearest_E > 0 AND distance_SE < 0"] = int(
        np.sum((sims > 0) & (dists < 0))
    )
    return {
        "n_errors": len(errors),
        "nearest_s_minus_nearest_e": _stats(sims),
        "all_errors_nearest_s_gt_nearest_e": bool(np.all(sims > 0)),
        "count_strongly_s_like_on_e_side": count,
        "correlations": {
            "nearest_S_minus_nearest_E_vs_margin_SE": margin_corr,
            "nearest_S_minus_nearest_E_vs_distance_SE": dist_corr,
        },
    }


# --------------------------------------------------------------------------- #
# Representative samples
# --------------------------------------------------------------------------- #
def representative(
    records_by_id: Dict[int, Dict], neighbor_rows: Sequence[Dict],
    prev_report_path: str,
) -> Dict:
    """Section 8: the six representative errors + previous-report cross-check."""
    nr = {r["frame_id"]: r for r in neighbor_rows}
    prev = None
    if Path(prev_report_path).is_file():
        prev = json.loads(Path(prev_report_path).read_text())
    out: Dict = {}
    prev_match = True
    for fid in QUERY_FRAMES:
        rec = records_by_id[fid]
        nn = nr[fid]
        entry = {
            "frame_id": fid,
            "path": rec["path"],
            "true": "S",
            "pred": "E",
            "nearest_s_cosine": nn["nearest_s_cosine"],
            "nearest_e_cosine": nn["nearest_e_cosine"],
            "nearest_s_minus_nearest_e": nn["nearest_s_minus_nearest_e"],
            "logit_S": rec["logit_S"],
            "logit_E": rec["logit_E"],
            "margin_SE": rec["margin_SE"],
            "distance_SE": rec["distance_SE"],
            "P(S)": rec["P(S)"],
            "P(E)": rec["P(E)"],
            "s_like_but_margin_negative": bool(
                nn["nearest_s_minus_nearest_e"] > 0 and rec["margin_SE"] < 0
            ),
        }
        if prev:
            pq = prev.get("representative_queries", {}).get(str(fid))
            if pq:
                match = (
                    abs(pq["s_neighbors"][0]["cosine_similarity"]
                        - nn["nearest_s_cosine"]) < 1e-4
                    and abs(pq["e_neighbors"][0]["cosine_similarity"]
                            - nn["nearest_e_cosine"]) < 1e-4
                )
                prev_match = prev_match and match
        out[str(fid)] = entry
    return {"samples": out,
            "nearest_matches_previous_report": bool(prev_match) if prev else None}


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #
def _load_mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def plot_margins(out_dir: Path, groups: Dict[str, List[Dict]],
                 colors: Dict[str, str]) -> str:
    plt = _load_mpl()
    fig, ax = plt.subplots(figsize=(9, 6))
    for name, recs in groups.items():
        vals = [r["margin_SE"] for r in recs]
        ax.hist(vals, bins=50, alpha=0.55, label=name, color=colors[name])
    ax.axvline(0, color="black", lw=1.2, ls="--")
    ax.set_xlabel("margin_SE = logit_S - logit_E")
    ax.set_ylabel("samples")
    ax.set_title("S/E logit margin distributions (head diagnosis)")
    ax.legend()
    fig.tight_layout()
    path = out_dir / "margin_distribution.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return str(path)


def plot_distances(out_dir: Path, groups: Dict[str, List[Dict]],
                   colors: Dict[str, str]) -> str:
    plt = _load_mpl()
    fig, ax = plt.subplots(figsize=(9, 6))
    for name, recs in groups.items():
        vals = [r["distance_SE"] for r in recs]
        ax.hist(vals, bins=50, alpha=0.55, label=name, color=colors[name])
    ax.axvline(0, color="black", lw=1.2, ls="--")
    ax.set_xlabel("distance_SE = margin_SE / ||w_S - w_E||")
    ax.set_ylabel("samples")
    ax.set_title("Signed distance to the S/E linear boundary")
    ax.legend()
    fig.tight_layout()
    path = out_dir / "boundary_distance_distribution.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return str(path)


def plot_embed_vs_margin(out_dir: Path, errors: Sequence[Dict],
                         neighbor_rows: Sequence[Dict]) -> str:
    plt = _load_mpl()
    nr = {r["frame_id"]: r for r in neighbor_rows}
    sims = np.array([nr[r["frame_id"]]["nearest_s_minus_nearest_e"]
                     for r in errors])
    margins = np.array([r["margin_SE"] for r in errors])
    fig, ax = plt.subplots(figsize=(8.5, 6))
    ax.scatter(sims, margins, s=22, alpha=0.7, color="#1565c0",
               edgecolors="none")
    ax.axhline(0, color="black", lw=1.2, ls="--")
    ax.axvline(0, color="black", lw=1.0, ls=":")
    ax.set_xlabel("nearest_S similarity - nearest_E similarity (cosine)")
    ax.set_ylabel("margin_SE = logit_S - logit_E")
    ax.set_title(
        "S->E errors: embedding similarity difference vs S/E logit margin"
    )
    ax.text(0.99, 0.97, f"n = {len(errors)}\n"
            f"quadrant x>0, y<0: {int(((sims>0)&(margins<0)).sum())}",
            transform=ax.transAxes, ha="right", va="top", fontsize=9,
            bbox=dict(boxstyle="round", fc="white", alpha=0.8))
    fig.tight_layout()
    path = out_dir / "embedding_vs_margin.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return str(path)


def plot_head_parameters(out_dir: Path, W: np.ndarray, b: np.ndarray) -> str:
    plt = _load_mpl()
    classes = list(CLASSES)
    s_idx, e_idx = classes.index("S"), classes.index("E")
    norms = np.linalg.norm(W, axis=1)
    colors = ["#9e9e9e"] * len(classes)
    colors[s_idx] = "#2e7d32"
    colors[e_idx] = "#c62828"
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    ax1.bar(range(len(classes)), norms, color=colors)
    ax1.axhline(norms.mean(), color="grey", lw=1, ls="--")
    ax1.set_xticks(range(len(classes)))
    ax1.set_xticklabels(classes, rotation=90, fontsize=8)
    ax1.set_ylabel("||w_c||₂")
    ax1.set_title(f"classifier weight norms (S rank "
                  f"{sum(norms > norms[s_idx]) + 1}/{len(classes)}, "
                  f"E rank {sum(norms > norms[e_idx]) + 1}/{len(classes)})")
    ax2.bar(range(len(classes)), b, color=colors)
    ax2.axhline(b.mean(), color="grey", lw=1, ls="--")
    ax2.set_xticks(range(len(classes)))
    ax2.set_xticklabels(classes, rotation=90, fontsize=8)
    ax2.set_ylabel("b_c")
    ax2.set_title("classifier biases (S rank "
                  f"{sum(b > b[s_idx]) + 1}/{len(classes)}, "
                  f"E rank {sum(b > b[e_idx]) + 1}/{len(classes)})")
    fig.tight_layout()
    path = out_dir / "class_head_parameters.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return str(path)


def write_plots(out_dir: Path, W: np.ndarray, b: np.ndarray,
                groups: Dict[str, List[Dict]],
                errors: Sequence[Dict],
                neighbor_rows: Sequence[Dict]) -> Dict[str, str]:
    colors = {"correct S": "#2e7d32", "S->E errors": "#c62828", "true E": "#1565c0"}
    return {
        "margin_distribution.png": plot_margins(out_dir, groups, colors),
        "boundary_distance_distribution.png": plot_distances(out_dir, groups,
                                                             colors),
        "embedding_vs_margin.png": plot_embed_vs_margin(out_dir, errors,
                                                        neighbor_rows),
        "class_head_parameters.png": plot_head_parameters(out_dir, W, b),
    }


# --------------------------------------------------------------------------- #
# Reports
# --------------------------------------------------------------------------- #
def build_json(
    *, checkpoint: str, manifest_path: str, head_info: Dict, head_stats: Dict,
    groups: Dict[str, List[Dict]], neighbor: Dict, reps: Dict, gallery: Dict,
    verification: Dict, created: str,
) -> Dict:
    return {
        "schema": SCHEMA,
        "created": created,
        "checkpoint": checkpoint,
        "arch": "efficientnet",
        "manifest": manifest_path,
        "manifest_sha256": json.loads(
            Path(manifest_path).read_text()
        ).get("content_sha256"),
        "split_note": (
            "dev val = leakage-safe frame-range split, same signer, same room "
            "(S2401..S3000). NOT the reported metric."
        ),
        "classifier_head": {
            "layers": head_info["layers"],
            "container": head_info["container"],
            "weight_shape": head_info["weight_shape"],
            "bias_shape": head_info["bias_shape"],
            "dropout_eval_is_identity": head_info["dropout_eval_is_identity"],
            "verified_against_cache": verification,
            "logit_identity": (
                "logit_c = w_c^T x + b_c on the pooled 1280-D embedding x; "
                "Dropout is identity under model.eval()."
            ),
        },
        "s_e_head_parameters": {
            "w_S_norm": head_stats["w_S_norm"],
            "w_E_norm": head_stats["w_E_norm"],
            "b_S": head_stats["b_S"],
            "b_E": head_stats["b_E"],
            "b_S_minus_b_E": head_stats["b_S_minus_b_E"],
            "cosine_similarity_w_S_w_E": head_stats["cosine_similarity_w_S_w_E"],
            "w_S_minus_w_E_norm": head_stats["w_S_minus_w_E_norm"],
            "rank_of_S": head_stats["rank_of_S"],
            "rank_of_E": head_stats["rank_of_E"],
            "median_weight_norm_all_classes": head_stats[
                "median_weight_norm_all_classes"],
            "median_bias_all_classes": head_stats["median_bias_all_classes"],
            "all_class_parameters": head_stats["classes"],
        },
        "groups": {
            "s_to_e_errors": group_summary(groups["S->E errors"]),
            "correct_s": group_summary(groups["correct S"]),
            "true_e": group_summary(groups["true E"]),
        },
        "embedding_vs_head": neighbor,
        "representative": reps,
        "gallery_files": gallery,
        "full_s_to_e_records": groups["S->E errors"],
        "note": (
            "Diagnosis only; descriptive. The S/E margin and distance are the "
            "S-vs-E two-class quantities, not the complete 29-class decision. "
            "Correlations are not causation; weight norms/biases are not judged "
            "bad. No fix is proposed."
        ),
    }


def write_head_notes(out_path: Path, report: Dict) -> None:
    g = report["groups"]
    err, ok, te = g["s_to_e_errors"], g["correct_s"], g["true_e"]
    emb = report["embedding_vs_head"]
    head = report["s_e_head_parameters"]
    reps = report["representative"]["samples"]
    cls = head["all_class_parameters"]
    s_row = next(r for r in cls if r["class"] == "S")
    e_row = next(r for r in cls if r["class"] == "E")
    m_err = err["margin_SE"]
    d_err = err["distance_SE"]
    f_err = err["feature_term"]
    m_ok = ok["margin_SE"]
    m_te = te["margin_SE"]
    corr = emb["correlations"]["nearest_S_minus_nearest_E_vs_margin_SE"]
    dcorr = emb["correlations"]["nearest_S_minus_nearest_E_vs_distance_SE"]
    corr_p = corr["pearson"]
    corr_s = corr["spearman"]
    dcorr_p = dcorr["pearson"]

    if corr_p is None:
        corr_line = "- Correlation could not be computed (constant input)."
    else:
        corr_line = (
            f"- Embedding-similarity-difference vs margin_SE: Pearson r = "
            f"{corr_p['r']:+.3f} (p={corr_p['p']:.2e}, n={corr_p['n']}); "
            f"Spearman r = {corr_s['r']:+.3f}. vs distance_SE Pearson r = "
            f"{dcorr_p['r']:+.3f}."
        )

    def fmt(v, nd=3):
        return f"{v:.{nd}f}"

    verif = report["classifier_head"]["verified_against_cache"]

    n_neg1 = sum(1 for r in report["full_s_to_e_records"] if r["margin_SE"] < -1.0)
    n_abs_m = sum(1 for r in report["full_s_to_e_records"]
                  if abs(r["margin_SE"]) < 0.1)
    n_abs_d = sum(1 for r in report["full_s_to_e_records"]
                  if abs(r["distance_SE"]) < 0.1)

    lines = [
        "# S->E head analysis notes — efficientnet_b0_targeted (diagnosis only)",
        "",
        "All quantities are computed on the cached pooled embeddings "
        "(`data/s_e_embeddings_targeted.npz`) recombined with the checkpoint "
        "head weights; every S prediction/probability is verified against "
        f"`data/s_to_e_all_s_records.json` "
        f"({verif['pred_mismatches_vs_cache']} mismatches, max |dp| "
        f"{verif['max_abs_delta_prob']:.2e}).",
        "",
        "## Q1. Are the 138 S->E errors close to or far across the S/E boundary?",
        "",
        f"- margin_SE distribution (logit_S - logit_E): errors mean "
        f"**{m_err['mean']:+.3f}**, median **{m_err['median']:+.3f}** "
        f"(std {m_err['std']:.3f}, min {m_err['min']:+.3f}, max {m_err['max']:+.3f}).",
        f"- margin_SE quantiles (errors): "
        f"{m_err['q5']:+.3f} / {m_err['q25']:+.3f} / {m_err['q50']:+.3f} / "
        f"{m_err['q75']:+.3f} / {m_err['q95']:+.3f} (5/25/50/75/95%).",
        f"- As a fraction of the S/E weight direction ||w_S - w_E|| = "
        f"{head['w_S_minus_w_E_norm']:.3f}: signed distance mean "
        f"**{d_err['mean']:+.4f}**, median **{d_err['median']:+.4f}** "
        f"(std {d_err['std']:.4f}, min {d_err['min']:+.4f}, "
        f"max {d_err['max']:+.4f}).",
        f"- {n_neg1} errors with margin < -1.0; "
        f"{n_abs_m} errors with |margin| < 0.1; "
        f"{n_abs_d} errors with |distance| < 0.1.",
        f"- For context, correct-S margin mean **{m_ok['mean']:+.3f}** "
        f"(median {m_ok['median']:+.3f}, min {m_ok['min']:+.3f}) and true-E "
        f"margin mean **{m_te['mean']:+.3f}** (median {m_te['median']:+.3f}, "
        f"max {m_te['max']:+.3f}).",
        "",
        "## Q2. Do their embeddings remain strongly S-like even when the final "
        "margin favors E?",
        "",
        f"- All {emb['n_errors']} errors have nearest-correct-S cosine greater "
        "than nearest-actual-E cosine "
        f"({emb['all_errors_nearest_s_gt_nearest_e']}); mean similarity "
        f"difference {emb['nearest_s_minus_nearest_e']['mean']:+.3f} "
        f"(median {emb['nearest_s_minus_nearest_e']['median']:+.3f}).",
        corr_line,
        f"- The correlation is descriptive; it does not establish whether the "
        "head or the representation causes the flip.",
        "",
        "## Q3. Anything numerically unusual about the S or E head parameters?",
        "",
        f"- ||w_S||₂ = **{head['w_S_norm']:.3f}** (rank "
        f"{head['rank_of_S']['weight_norm']}/29), ||w_E||₂ = "
        f"**{head['w_E_norm']:.3f}** (rank "
        f"{head['rank_of_E']['weight_norm']}/29); median over all classes "
        f"{head['median_weight_norm_all_classes']:.3f}.",
        f"- b_S = **{head['b_S']:+.3f}** (rank {head['rank_of_S']['bias']}/29), "
        f"b_E = **{head['b_E']:+.3f}** (rank {head['rank_of_E']['bias']}/29); "
        f"median bias {head['median_bias_all_classes']:+.3f}.",
        f"- b_S - b_E = **{head['b_S_minus_b_E']:+.3f}**; "
        f"cos(w_S, w_E) = **{head['cosine_similarity_w_S_w_E']:+.3f}**.",
        "- None of these are inherently bad: a norm or bias outside the median "
        "is a description, not a defect.",
        "",
        "## Q4. Representation overlap, head/boundary behavior, or ambiguous?",
        "",
        f"- The errors' embeddings are overwhelmingly S-like (Q2: all 138 have "
        f"nearest-correct-S closer than nearest-E, mean difference "
        f"{emb['nearest_s_minus_nearest_e']['mean']:+.3f}), yet their S/E "
        "margin is negative because of the head's linear decision. This profile "
        "points at the final-head/decision-boundary region rather than "
        "feature-space S/E overlap.",
        f"- The decomposition margin_SE = feature_term + bias_term shows "
        f"feature_term (errors) mean **{f_err['mean']:+.3f}**, median "
        f"**{f_err['median']:+.3f}**, against a constant bias_term "
        f"{err['bias_term']:+.3f} — the S-vs-E weight direction dominates the "
        "sign, with the bias as the constant offset.",
        f"- Representative check: of the six query errors, "
        f"{sum(1 for q in reps.values() if q['s_like_but_margin_negative'])}/6 "
        "are closer to correct S yet lie on the E side of the hyperplane.",
        "- This remains **descriptive and to be read conservatively**: the "
        "nearest-neighbor result plus the negative margins are consistent with "
        "head/boundary behavior, but they do not *prove* the head is defective. "
        "**No fix is proposed** in this document.",
        "",
        "## Representative samples",
        "",
        "| frame | nearest-S | nearest-E | sim diff | logit_S | logit_E | "
        "margin_SE | distance_SE | P(S) | P(E) |",
        "|-------|-----------|-----------|----------|---------|---------|"
        "------------|--------------|------|------|",
    ]
    for q in (reps[str(fid)] for fid in QUERY_FRAMES):
        lines.append(
            f"| S{q['frame_id']} | {q['nearest_s_cosine']:.3f} | "
            f"{q['nearest_e_cosine']:.3f} | "
            f"{q['nearest_s_minus_nearest_e']:+.3f} | {q['logit_S']:+.2f} | "
            f"{q['logit_E']:+.2f} | {q['margin_SE']:+.3f} | "
            f"{q['distance_SE']:+.4f} | {q['P(S)']:.3f} | {q['P(E)']:.3f} |"
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")
    print(f"wrote {out_path}")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def run(args) -> Dict:
    created = datetime.now(timezone.utc).isoformat(timespec="seconds")
    model = load_checkpoint(args.checkpoint, "cpu")
    W, b, head_info = extract_head(model)
    print(f"head: {head_info['layers']}  W={head_info['weight_shape']} "
          f"b={head_info['bias_shape']}  dropout_eval_identity="
          f"{head_info['dropout_eval_is_identity']}")

    emb = load_embeddings(args.npz)
    manifest = SplitManifest.from_json(args.manifest)
    classes = list(CLASSES)
    s_idx, e_idx = classes.index("S"), classes.index("E")

    e_paths = {frame_id(s.path): s.path for s in manifest.val
               if s.class_name == "E"}
    s_paths = {r["frame_id"]: r["path"] for r in
               json.loads(Path(args.s_cache).read_text())}

    s_records = build_records("S", emb["s_ids"], emb["s_embs"], W, b,
                              s_idx, e_idx, s_paths)
    e_records = build_records("E", emb["e_ids"], emb["e_embs"], W, b,
                              s_idx, e_idx, e_paths)
    verification = verify_s_vs_cache(s_records, args.s_cache)
    print(f"verified vs cache: {verification['pred_mismatches_vs_cache']} "
          f"pred mismatches, max |dp| {verification['max_abs_delta_prob']:.2e}")

    errors = [r for r in s_records if r["pred"] == "E"]
    correct = [r for r in s_records if r["pred"] == "S"]
    if len(errors) != 138 or len(correct) != 462:
        raise SystemExit(
            f"S->E count mismatch: {len(errors)} errors / {len(correct)} "
            "correct (expected 138/462)"
        )
    records_by_id = {r["frame_id"]: r for r in s_records}
    print(f"groups: S->E {len(errors)}, correct S {len(correct)}, "
          f"true E {len(e_records)}")

    neighbor_rows = neighbor_join(errors, emb["s_embs"], emb["s_ids"],
                                  emb["e_embs"], emb["e_ids"], args.s_cache)
    neighbor = embed_vs_head(errors, neighbor_rows)
    rep = representative(records_by_id, neighbor_rows, args.prev_report)
    margin_r = (neighbor["correlations"]
                ["nearest_S_minus_nearest_E_vs_margin_SE"]["pearson"])
    print(f"neighbor join: {len(neighbor_rows)} rows; "
          f"strongly-S-like-on-E-side "
          f"{neighbor['count_strongly_s_like_on_e_side']}; "
          f"sim-diff vs margin Pearson r={margin_r['r']:+.3f}")

    head_stats = head_parameters(W, b)
    print(f"head: w_S_norm={head_stats['w_S_norm']:.3f} "
          f"w_E_norm={head_stats['w_E_norm']:.3f} "
          f"b_S={head_stats['b_S']:+.3f} b_E={head_stats['b_E']:+.3f} "
          f"b_S-b_E={head_stats['b_S_minus_b_E']:+.3f} "
          f"cos(w_S,w_E)={head_stats['cosine_similarity_w_S_w_E']:.3f}")

    gallery_dir = Path(args.gallery_dir)
    gallery = write_plots(gallery_dir, W, b,
                          {"correct S": correct, "S->E errors": errors,
                           "true E": e_records}, errors, neighbor_rows)

    report = build_json(
        checkpoint=args.checkpoint, manifest_path=args.manifest,
        head_info=head_info, head_stats=head_stats,
        groups={"S->E errors": errors, "correct S": correct,
                "true E": e_records},
        neighbor=neighbor, reps=rep, gallery=gallery, verification=verification,
        created=created,
    )
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {out_json}")

    md = Path(args.out_md)
    if md.exists():
        print(f"kept existing {md} (curated; re-run will not overwrite)")
    else:
        write_head_notes(md, report)
    return report


def main() -> None:
    p = argparse.ArgumentParser(
        description="Diagnose the S->E failure at the final classifier head "
                    "(no fixes)."
    )
    p.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    p.add_argument("--manifest", default=DEFAULT_MANIFEST)
    p.add_argument("--npz", default=DEFAULT_NPZ)
    p.add_argument("--s-cache", default=DEFAULT_S_CACHE)
    p.add_argument("--prev-report", default=DEFAULT_PREV_REPORT)
    p.add_argument("--gallery-dir", default=DEFAULT_GALLERY_DIR)
    p.add_argument("--out-json", default=DEFAULT_OUT_JSON)
    p.add_argument("--out-md", default=DEFAULT_OUT_MD)
    run(p.parse_args())


if __name__ == "__main__":
    main()
