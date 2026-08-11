"""S->E embedding-neighbor diagnosis — diagnosis only, no fixes.

Question: are the S->E failures genuinely close to real E in the model's
learned feature space, or is the linear head collapsing visually distinct S
poses toward E?

Embedding: the penultimate EfficientNet-B0 feature vector — the output of the
timm backbone (``num_classes=0, global_pool='avg'``), which is exactly what
sits before the ``Dropout(0.3) + Linear(1280, 29)`` head. ``build_efficientnet``
returns ``nn.Sequential(backbone, head)``, so ``model[0](x)`` is the pooled
1280-D vector and ``model[1]`` is the head; the script asserts both the head's
structure and that ``model(x) == head(backbone(x))``. Embeddings are
L2-normalized before any cosine similarity.

Workflow:

1. Extract pooled 1280-D embeddings for every S and E validation sample using
   the exact val preprocessing (``ASLImageDataset(train=False)`` -> the frozen
   ``preprocess_bgr`` contract) and ``checkpoints/efficientnet_b0_targeted.pt``.
   Logits are also computed through the head so each E sample gets its predicted
   class; S predictions are cross-checked against the cached S->E records.
2. For each representative S->E query (S2575, S2601, S2805, S2864 + boundary
   S2832, S2876): top-5 nearest correctly classified S and top-5 nearest actual
   E, by cosine similarity, self-excluded.
3. For all 138 S->E errors (and the 462 correct-S control): nearest correct-S
   vs nearest actual-E cosine similarity, plus the distribution of
   ``nearest_E - nearest_S``.
4. Writes neighbor galleries to
   ``docs/figures/error_gallery_targeted/S_to_E/G_embedding_neighbors_human/``,
   a JSON report, and a descriptive notes file.

The nearest-* statistics are descriptive. They do not prove causality: nearest
to E is a *possible* representation/data-overlap signal; nearest to correct S
while predicting E is a *possible* head/boundary signal. No fix is proposed.
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
from torch.utils.data import DataLoader  # noqa: E402

from src.data import ASLImageDataset, CLASSES, SplitManifest  # noqa: E402
from src.evaluate import load_checkpoint  # noqa: E402
from scripts.analyze_s_to_e import frame_id  # noqa: E402
from scripts.inspect_s_to_e_runs import _resolve_device, render_model_input  # noqa: E402

DEFAULT_CHECKPOINT = "checkpoints/efficientnet_b0_targeted.pt"
DEFAULT_MANIFEST = "data/split_manifest.json"
DEFAULT_S_CACHE = "data/s_to_e_all_s_records.json"
DEFAULT_NPZ = "data/s_e_embeddings_targeted.npz"
DEFAULT_GALLERY_DIR = (
    "docs/figures/error_gallery_targeted/S_to_E/G_embedding_neighbors_human"
)
DEFAULT_OUT_JSON = "docs/reports/s_to_e_embedding_neighbors_targeted.json"
DEFAULT_OUT_MD = "docs/reports/s_to_e_embedding_notes.md"

FEATURE_DIM = 1280
TOP_K = 5
QUERY_FRAMES = [2575, 2601, 2805, 2864, 2832, 2876]
TILE_IN = 2.4
DPI = 150
SCHEMA = "asl-s_to_e-embedding-neighbors/v1"


# --------------------------------------------------------------------------- #
# Pure helpers (unit-tested)
# --------------------------------------------------------------------------- #
def l2_normalize(mat: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalization. Zero rows stay zero (norm 1 guard)."""
    arr = np.asarray(mat, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"expected 2D array, got {arr.ndim}D")
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return arr / norms


def cosine_top_k(
    query: np.ndarray,
    pool: Dict[int, np.ndarray],
    k: int,
    exclude_id: Optional[int] = None,
) -> List[Tuple[int, float]]:
    """Top ``k`` ``(frame_id, cosine)`` in the pool, descending, self-excluded.

    ``query`` and every pool vector must already be L2-normalized, so cosine is
    a plain dot product.
    """
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    sims = [
        (fid, float(query @ emb))
        for fid, emb in pool.items()
        if fid != exclude_id
    ]
    sims.sort(key=lambda t: (-t[1], t[0]))
    return sims[:k]


def nearest_similarity(
    query: np.ndarray,
    pool: Dict[int, np.ndarray],
    exclude_id: Optional[int] = None,
) -> Optional[float]:
    """Cosine to the single nearest pool member (self excluded), or None."""
    best = cosine_top_k(query, pool, 1, exclude_id)
    return best[0][1] if best else None


def compare_nearest(
    queries: Dict[int, np.ndarray],
    pool_s: Dict[int, np.ndarray],
    pool_e: Dict[int, np.ndarray],
) -> List[Dict]:
    """Per-query nearest correct-S vs nearest actual-E cosine.

    Queries are L2-normalized embedding vectors keyed by frame id; pools are
    ``{frame_id: normalized emb}``. Every query is self-excluded from its own
    pool (the query is an S sample, so it can only be inside ``pool_s``).
    """
    rows: List[Dict] = []
    for fid, emb in queries.items():
        sim_s = nearest_similarity(emb, pool_s, fid)
        sim_e = nearest_similarity(emb, pool_e, fid)
        if sim_s is None or sim_e is None:
            continue
        diff = sim_e - sim_s
        closer = "E" if diff > 1e-9 else ("S" if diff < -1e-9 else "tie")
        rows.append(
            {
                "frame_id": fid,
                "nearest_s_sim": float(sim_s),
                "nearest_e_sim": float(sim_e),
                "nearest_e_minus_s": float(diff),
                "closer": closer,
            }
        )
    return rows


def _pct(arr: np.ndarray, p: float) -> float:
    return float(np.percentile(arr, p)) if arr.size else float("nan")


def summarize_nearest(rows: Sequence[Dict]) -> Dict:
    """Counts/means/distribution over ``compare_nearest`` rows."""
    d = np.array([r["nearest_e_minus_s"] for r in rows], dtype=np.float64)
    ss = np.array([r["nearest_s_sim"] for r in rows], dtype=np.float64)
    ee = np.array([r["nearest_e_sim"] for r in rows], dtype=np.float64)
    closer = [r["closer"] for r in rows]
    return {
        "n": len(rows),
        "closer_to_e": closer.count("E"),
        "closer_to_s": closer.count("S"),
        "ties": closer.count("tie"),
        "mean_nearest_s_sim": float(ss.mean()) if ss.size else float("nan"),
        "mean_nearest_e_sim": float(ee.mean()) if ee.size else float("nan"),
        "nearest_e_minus_nearest_s": {
            "mean": float(d.mean()) if d.size else float("nan"),
            "std": float(d.std()) if d.size else float("nan"),
            "min": float(d.min()) if d.size else float("nan"),
            "p25": _pct(d, 25),
            "p50": _pct(d, 50),
            "p75": _pct(d, 75),
            "max": float(d.max()) if d.size else float("nan"),
        },
    }


# --------------------------------------------------------------------------- #
# Embedding extraction (device side)
# --------------------------------------------------------------------------- #
def extract_s_e_embeddings(
    checkpoint: str, manifest_path: str, device: str, batch_size: int
) -> List[Dict]:
    """Pooled 1280-D features + head predictions for every S and E val sample."""
    manifest = SplitManifest.from_json(manifest_path)
    samples = [s for s in manifest.val if s.class_name in ("S", "E")]
    if not samples:
        raise SystemExit("no S/E samples in the val split")
    ds = ASLImageDataset(samples, train=False)
    model = load_checkpoint(checkpoint, device)
    model.eval()

    backbone = model[0]
    head = model[1]
    if not (
        isinstance(head, torch.nn.Sequential)
        and len(head) == 2
        and isinstance(head[0], torch.nn.Dropout)
        and isinstance(head[1], torch.nn.Linear)
        and head[1].in_features == FEATURE_DIM
        and head[1].out_features == len(CLASSES)
    ):
        raise SystemExit(
            f"model[1] is not the Dropout(0.3)+Linear({FEATURE_DIM},29) head: {head}"
        )

    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    classes = list(CLASSES)
    rows: List[Dict] = []
    with torch.no_grad():
        for bi, (x, _) in enumerate(loader):
            x = x.to(device)
            emb = backbone(x)  # pooled EfficientNet-B0 features, (B, 1280)
            if tuple(emb.shape)[1:] != (FEATURE_DIM,):
                raise SystemExit(
                    f"backbone output shape {tuple(emb.shape)} != (B, {FEATURE_DIM})"
                )
            # Verify the split point: full forward must equal head(backbone).
            full = model(x)
            via_head = head(emb)
            if not torch.allclose(full, via_head, atol=1e-5):
                raise SystemExit("head(backbone(x)) != model(x): wrong embedding layer")
            preds = via_head.argmax(1).cpu().numpy()
            emb_np = emb.cpu().numpy()
            for i in range(int(x.shape[0])):
                s = ds.samples[bi * batch_size + i]
                rows.append(
                    {
                        "path": s.path,
                        "frame_id": frame_id(s.path),
                        "class": s.class_name,
                        "pred": classes[int(preds[i])],
                        "emb": emb_np[i],
                    }
                )
    print(f"extracted {len(rows)} embeddings "
          f"({sum(1 for r in rows if r['class']=='S')} S, "
          f"{sum(1 for r in rows if r['class']=='E')} E) on {device}")
    return rows


# --------------------------------------------------------------------------- #
# Pools + comparison
# --------------------------------------------------------------------------- #
def build_pools(
    rows: Sequence[Dict], s_cache_path: str
) -> Tuple[Dict[int, Dict], Dict[int, Dict], Dict[int, Dict]]:
    """Return ``(s_by_id, e_by_id, cache_by_id)`` with S predictions cross-checked."""
    s_by_id = {r["frame_id"]: r for r in rows if r["class"] == "S"}
    e_by_id = {r["frame_id"]: r for r in rows if r["class"] == "E"}
    cache = json.loads(Path(s_cache_path).read_text())
    cache_by = {r["frame_id"]: r for r in cache}
    missing = [fid for fid in s_by_id if fid not in cache_by]
    if missing:
        raise SystemExit(f"S frames missing from cache: {missing[:10]}...")
    for fid, r in s_by_id.items():
        if cache_by[fid]["pred"] != r["pred"]:
            raise SystemExit(
                f"S{fid} prediction mismatch: extraction says {r['pred']}, "
                f"cache says {cache_by[fid]['pred']}"
            )
    return s_by_id, e_by_id, cache_by


def query_records(cache_by: Dict[int, Dict]) -> List[Dict]:
    """The representative S->E queries as cached records, in QUERY_FRAMES order."""
    missing = [fid for fid in QUERY_FRAMES if fid not in cache_by]
    if missing:
        raise SystemExit(f"query frames missing from S cache: {missing}")
    recs = [cache_by[fid] for fid in QUERY_FRAMES]
    bad = [r["frame_id"] for r in recs if r["pred"] != "E"]
    if bad:
        raise SystemExit(f"query frames are not S->E errors: {bad}")
    return recs


# --------------------------------------------------------------------------- #
# Galleries
# --------------------------------------------------------------------------- #
def _fmt(v: float) -> str:
    s = f"{v:.2f}"
    return s[1:] if s.startswith("0") else s


def _neighbor_title(rec: Dict, sim: float) -> str:
    stem = Path(rec["path"]).stem
    return f"{stem} | {rec['class']}->{rec['pred']} | cos=.{_fmt(sim)}"


def _query_title(rec: Dict) -> str:
    stem = Path(rec["path"]).stem
    return f"{stem} | pred=E | P(S)=.{_fmt(rec['p_s'])} | P(E)=.{_fmt(rec['p_e'])}"


def write_neighbor_gallery(
    out_path: Path,
    query: Dict,
    s_neighbors: Sequence[Tuple[int, float]],
    e_neighbors: Sequence[Tuple[int, float]],
    s_by_id: Dict[int, Dict],
    e_by_id: Dict[int, Dict],
) -> Tuple[int, int]:
    """One horizontal strip: LEFT nearest correct S, CENTER query, RIGHT actual E."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = 1 + len(s_neighbors) + len(e_neighbors)
    fig, axes = plt.subplots(1, n, figsize=(TILE_IN * n, TILE_IN * 1.4))
    left = axes[: len(s_neighbors)]
    right = axes[len(s_neighbors) + 1 :]

    for ax, (fid, sim) in zip(left, s_neighbors):
        rec = s_by_id[fid]
        ax.imshow(render_model_input(rec["path"]))
        ax.axis("off")
        ax.set_title(_neighbor_title(rec, sim), fontsize=8)
        for sp in ax.spines.values():
            sp.set_edgecolor("#2e7d32")
            sp.set_linewidth(2)

    qax = axes[len(s_neighbors)]
    qax.imshow(render_model_input(query["path"]))
    qax.axis("off")
    qax.set_title(_query_title(query), fontsize=9)
    for sp in qax.spines.values():
        sp.set_edgecolor("#1565c0")
        sp.set_linewidth(3.5)

    for ax, (fid, sim) in zip(right, e_neighbors):
        rec = e_by_id[fid]
        ax.imshow(render_model_input(rec["path"]))
        ax.axis("off")
        ax.set_title(_neighbor_title(rec, sim), fontsize=8)
        for sp in ax.spines.values():
            sp.set_edgecolor("#c62828")
            sp.set_linewidth(2)

    fig.text(0.5, 0.94, f"S{query['frame_id']} — S->E  (embedding neighbors)",
             ha="center", fontsize=13)
    fig.text(0.25, 0.02, "nearest correct S", ha="center", fontsize=11)
    fig.text(0.5, 0.02, "query (pred=E)", ha="center", fontsize=11)
    fig.text(0.75, 0.02, "nearest actual E", ha="center", fontsize=11)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)
    return int(TILE_IN * n * DPI), int(TILE_IN * 1.4 * DPI)


# --------------------------------------------------------------------------- #
# Reports
# --------------------------------------------------------------------------- #
def build_json(
    rows: Sequence[Dict],
    s_by_id: Dict[int, Dict],
    e_by_id: Dict[int, Dict],
    cache_by: Dict[int, Dict],
    checkpoint: str,
    manifest_path: str,
    npz_path: str,
    gallery_files: Dict[str, str],
) -> Dict:
    pool_s_ids = sorted(fid for fid in s_by_id if cache_by[fid]["pred"] == "S")
    pool_e_ids = sorted(e_by_id)
    pool_s = {fid: l2_normalize(s_by_id[fid]["emb"][np.newaxis, :])[0]
              for fid in pool_s_ids}
    pool_e = {fid: l2_normalize(e_by_id[fid]["emb"][np.newaxis, :])[0]
              for fid in pool_e_ids}

    queries = query_records(cache_by)
    query_embs = {r["frame_id"]: l2_normalize(s_by_id[r["frame_id"]]["emb"][np.newaxis, :])[0]
                  for r in queries}

    q_out = {}
    for r in queries:
        s_n = cosine_top_k(query_embs[r["frame_id"]], pool_s, TOP_K, r["frame_id"])
        e_n = cosine_top_k(query_embs[r["frame_id"]], pool_e, TOP_K, r["frame_id"])
        q_out[str(r["frame_id"])] = {
            "path": r["path"],
            "true": "S",
            "pred": "E",
            "p_s": r["p_s"],
            "p_e": r["p_e"],
            "s_neighbors": [
                {"frame_id": fid, "true": "S", "pred": "S",
                 "cosine_similarity": sim}
                for fid, sim in s_n
            ],
            "e_neighbors": [
                {"frame_id": fid, "true": "E",
                 "pred": e_by_id[fid]["pred"],
                 "cosine_similarity": sim}
                for fid, sim in e_n
            ],
        }

    err_embs = {fid: l2_normalize(s_by_id[fid]["emb"][np.newaxis, :])[0]
                for fid in cache_by if cache_by[fid]["pred"] == "E"}
    ctl_embs = {fid: pool_s[fid] for fid in pool_s_ids}  # already normalized
    s_to_e_rows = compare_nearest(err_embs, pool_s, pool_e)
    control_rows = compare_nearest(ctl_embs, pool_s, pool_e)

    return {
        "schema": SCHEMA,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "checkpoint": checkpoint,
        "arch": "efficientnet",
        "manifest": manifest_path,
        "embedding": {
            "source": (
                "penultimate EfficientNet-B0 pooled feature vector: timm "
                "efficientnet_b0 (num_classes=0, global_pool='avg') = model[0] "
                "of nn.Sequential(backbone, head)"
            ),
            "layer": "model[0], immediately before head[0]=Dropout(0.3) and "
                     "head[1]=Linear(1280, 29)",
            "dim": FEATURE_DIM,
            "verification": (
                "asserted head structure Dropout+Linear(1280,29) and "
                "torch.allclose(model(x), head(backbone(x)))"
            ),
            "normalization": "row-wise L2-normalization before any cosine product",
            "similarity": "cosine on L2-normalized embeddings (plain dot product)",
        },
        "embeddings_cache": npz_path,
        "pools": {"n_correct_s": len(pool_s_ids), "n_actual_e": len(pool_e_ids)},
        "representative_queries": q_out,
        "s_to_e_138": summarize_nearest(s_to_e_rows),
        "correct_s_control": summarize_nearest(control_rows),
        "gallery_files": gallery_files,
        "note": (
            "Descriptive only. nearest-to-E is a possible representation/"
            "data-overlap signal; nearest-to-correct-S-while-predicting-E is a "
            "possible head/decision-boundary signal. Neither is treated as a "
            "conclusion, and no fix is proposed."
        ),
    }


def write_embedding_notes(out_path: Path, report: Dict) -> None:
    s = report["s_to_e_138"]
    c = report["correct_s_control"]
    closer_e = s["closer_to_e"]
    closer_s = s["closer_to_s"]
    d = s["nearest_e_minus_nearest_s"]
    queries = report["representative_queries"]

    def nearer(fid: int) -> str:
        q = queries[str(fid)]
        return q["s_neighbors"][0]["cosine_similarity"], q["e_neighbors"][0][
            "cosine_similarity"]

    lines = [
        "# S->E embedding-neighbor notes — efficientnet_b0_targeted (diagnosis only)",
        "",
        "## Method",
        "",
        f"- Embedding: penultimate EfficientNet-B0 pooled feature vector "
        f"(`model[0]`, {report['embedding']['dim']}-D), the input to "
        "`Dropout(0.3) + Linear(1280, 29)`. Location verified against the head "
        "structure and `model(x) == head(backbone(x))`.",
        f"- Normalization: row-wise L2; similarity is cosine (dot product).",
        f"- Pools: {report['pools']['n_correct_s']} correctly classified S, "
        f"{report['pools']['n_actual_e']} actual E validation samples; queries "
        "self-excluded.",
        "- This is descriptive. Nearest-to-E vs nearest-to-correct-S splits a "
        "*possible* representation/data-overlap problem from a *possible* "
        "head/boundary problem, but neither is established as a cause here.",
        "",
        "## Q1: Are S->E errors generally embedded closer to E than correct S?",
        "",
        f"- **{s['closer_to_e']}** of {s['n']} S->E errors have a nearest actual-E "
        f"closer than their nearest correct S; **{s['closer_to_s']}** have nearest "
        f"correct S closer ({s['ties']} ties).",
        f"- Mean nearest-E similarity: **{s['mean_nearest_e_sim']:.3f}**; "
        f"mean nearest-correct-S similarity: **{s['mean_nearest_s_sim']:.3f}**.",
        f"- Distribution of nearest-E minus nearest-S: mean **{d['mean']:+.3f}**, "
        f"std {d['std']:.3f}, min {d['min']:+.3f}, median {d['p50']:+.3f}, "
        f"max {d['max']:+.3f}.",
        "",
        "## Q2: Are the representative errors visually surrounded by real E?",
        "",
        "- The six query galleries ("
        + ", ".join(f"`S{fid}_neighbors.png`" for fid in QUERY_FRAMES)
        + ") show top-5 nearest correct S (left), the query (center), top-5 "
        "nearest actual E (right). Per-query nearest similarities:",
        "",
        "| query | nearest S | nearest E | | query | nearest S | nearest E |",
        "|-------|-----------|-----------|-|-------|-----------|-----------|",
    ]
    half = len(QUERY_FRAMES) // 2
    for a, b in zip(QUERY_FRAMES[:half], QUERY_FRAMES[half:]):
        sa, ea = nearer(a)
        sb, eb = nearer(b)
        lines.append(
            f"| S{a} | {sa:.3f} | {ea:.3f} | | S{b} | {sb:.3f} | {eb:.3f} |"
        )
    lines += [
        "",
        "Whether the neighboring E tiles *look* like the query is for a human "
        "reviewer to decide from the galleries; the similarities above are "
        "measured, the visual judgment is not.",
        "",
        "## Q3: Are some failures embedded close to correct S but misclassified?",
        "",
        f"- **{s['closer_to_s']}** of {s['n']} errors sit closer to correct S than "
        "to actual E in embedding space yet still predict E — the profile of a "
        f"head/decision-boundary problem. **{s['closer_to_e']}** sit closer to E — "
        "the profile of a representation/data-overlap problem.",
        "- Control (correctly classified S): "
        f"{c['closer_to_e']} of {c['n']} have nearest-E closer than nearest-other-S "
        f"(mean nearest-other-S {c['mean_nearest_s_sim']:.3f}, nearest-E "
        f"{c['mean_nearest_e_sim']:.3f}).",
        "",
        "## Bottom line (descriptive)",
        "",
        f"The 138 S->E errors split {s['closer_to_e']} (nearer to E) vs "
        f"{s['closer_to_s']} (nearer to correct S). No fix is proposed in this "
        "document; the split is reported for the human reviewer to weigh.",
        "",
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")
    print(f"wrote {out_path}")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def run(args) -> Dict:
    device = _resolve_device(args.device)
    rows = extract_s_e_embeddings(args.checkpoint, args.manifest, device,
                                  args.batch_size)
    s_by_id, e_by_id, cache_by = build_pools(rows, args.s_cache)

    npz = Path(args.npz)
    np.savez_compressed(
        npz,
        s_ids=np.array([fid for fid in sorted(s_by_id)]),
        s_embs=np.stack([s_by_id[fid]["emb"] for fid in sorted(s_by_id)]),
        e_ids=np.array([fid for fid in sorted(e_by_id)]),
        e_embs=np.stack([e_by_id[fid]["emb"] for fid in sorted(e_by_id)]),
    )
    print(f"wrote {npz}")

    gallery_dir = Path(args.gallery_dir)
    pool_s_ids = sorted(fid for fid in s_by_id if cache_by[fid]["pred"] == "S")
    pool_e_ids = sorted(e_by_id)
    pool_s = {fid: l2_normalize(s_by_id[fid]["emb"][np.newaxis, :])[0]
              for fid in pool_s_ids}
    pool_e = {fid: l2_normalize(e_by_id[fid]["emb"][np.newaxis, :])[0]
              for fid in pool_e_ids}
    queries = query_records(cache_by)

    gallery_files: Dict[str, str] = {}
    for r in queries:
        qemb = l2_normalize(s_by_id[r["frame_id"]]["emb"][np.newaxis, :])[0]
        s_n = cosine_top_k(qemb, pool_s, TOP_K, r["frame_id"])
        e_n = cosine_top_k(qemb, pool_e, TOP_K, r["frame_id"])
        name = f"S{r['frame_id']}_neighbors.png"
        write_neighbor_gallery(
            gallery_dir / name, r, s_n, e_n, s_by_id, e_by_id
        )
        gallery_files[name] = str(gallery_dir / name)

    report = build_json(rows, s_by_id, e_by_id, cache_by, args.checkpoint,
                        args.manifest, str(npz), gallery_files)
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {out_json}")

    s = report["s_to_e_138"]
    print(f"S->E 138: closer_to_E={s['closer_to_e']} "
          f"closer_to_S={s['closer_to_s']} ties={s['ties']}")
    print(f"  mean nearest-S={s['mean_nearest_s_sim']:.3f} "
          f"mean nearest-E={s['mean_nearest_e_sim']:.3f}")
    c = report["correct_s_control"]
    print(f"control 462: closer_to_E={c['closer_to_e']} "
          f"closer_to_S={c['closer_to_s']} "
          f"mean nearest-S={c['mean_nearest_s_sim']:.3f} "
          f"mean nearest-E={c['mean_nearest_e_sim']:.3f}")

    md = Path(args.out_md)
    if md.exists():
        print(f"kept existing {md} (curated; re-run will not overwrite)")
    else:
        write_embedding_notes(md, report)
    return report


def main() -> None:
    p = argparse.ArgumentParser(
        description="S->E embedding-neighbor diagnosis (no fixes)."
    )
    p.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    p.add_argument("--manifest", default=DEFAULT_MANIFEST)
    p.add_argument("--s-cache", default=DEFAULT_S_CACHE)
    p.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"],
                   default="auto")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--npz", default=DEFAULT_NPZ)
    p.add_argument("--gallery-dir", default=DEFAULT_GALLERY_DIR)
    p.add_argument("--out-json", default=DEFAULT_OUT_JSON)
    p.add_argument("--out-md", default=DEFAULT_OUT_MD)
    run(p.parse_args())


if __name__ == "__main__":
    main()
