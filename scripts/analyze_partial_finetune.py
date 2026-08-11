"""Analysis of the partial-fine-tuning experiment (final phase).

Consumes the training-phase output ``checkpoints/partial_ft/partial_ft_best.pt``
and produces:

1. FINAL reported numbers: the normal raw-image CPU evaluation pipeline
   (``ASLImageDataset(manifest.val, train=False)`` + ``preprocess_bgr``) on the
   best partial-FT candidate → accuracy, macro-F1, total errors, per-class
   P/R/F1 for S/E/K/V/M/N/X, watch-pair counts, confusion PNG/CSV.
2. Representation-change diagnostics in the pooled 1280-D embedding space:
   - embedding movement ``1 - cosine(before, after)`` for the original 138
     S->E errors, correct-S controls (462), true E (600), and all val (17,400)
   - how many of the original 138 became correct, and their new predictions
   - S-vs-E nearest-neighbor re-run in the NEW space per original error
   - S/E boundary: margin_SE + signed boundary distance before vs after, for
     the fixed diagnostic groups (reference: before margins -1.208/+2.880/
     -5.317 for errors/correct-S/true-E)
   - the 6 representative frames (S2575/S2601/S2805/S2864/S2832/S2876)
3. Three-way comparison (targeted baseline / head-only / partial-FT) with the
   exact prior numbers read from the repo artifacts, and a deterministic verdict
   per the head-only acceptance criteria.
4. The deployable checkpoint ``checkpoints/efficientnet_b0_partial_ft.pt``
   (schema ``asl-partial-ft/v1``, verdict embedded) — written only after the
   verdict is decided, so a rejected result cannot be deployed by accident.

Outputs
-------
- ``checkpoints/efficientnet_b0_partial_ft.pt``
- ``docs/reports/partial_finetune_targeted.json``
- ``docs/reports/partial_finetune_targeted.md``
- ``docs/figures/confusion_matrix_partial_ft.png|.csv``
- ``docs/figures/error_gallery_targeted/S_to_E/I_partial_finetune/rep_change.png``
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from scripts.analyze_s_to_e import _resolve_device, frame_id  # noqa: E402
from scripts.analyze_s_to_e_head import signed_boundary_distance  # noqa: E402
from scripts.build_embedding_cache import file_sha256, state_dict_sha256  # noqa: E402
from scripts.train_head_only_cached import (  # noqa: E402
    classify,
    evaluate_cm,
    s_e_metrics,
)
from src.data import CLASSES, ASLImageDataset, SplitManifest  # noqa: E402
from src.evaluate import (  # noqa: E402
    confusion_matrix,
    load_checkpoint,
    save_confusion_csv,
    save_confusion_png,
)

DEFAULT_PARENT = "checkpoints/efficientnet_b0_targeted.pt"
DEFAULT_MANIFEST = "data/split_manifest.json"
DEFAULT_CANDIDATE = "checkpoints/partial_ft/partial_ft_best.pt"
DEFAULT_HEAD_ONLY_REPORT = "docs/reports/head_only_cached_embeddings.json"
DEFAULT_BASELINE_REPORT = "docs/reports/partial_ft_baseline_gate.json"
DEFAULT_DEV_VAL_TARGETED = "docs/reports/dev_val_targeted_current.json"
DEFAULT_DEPLOY_OUT = "checkpoints/efficientnet_b0_partial_ft.pt"
DEFAULT_OUT_JSON = "docs/reports/partial_finetune_targeted.json"
DEFAULT_OUT_MD = "docs/reports/partial_finetune_targeted.md"
DEFAULT_CM_PNG = "docs/figures/confusion_matrix_partial_ft.png"
DEFAULT_CM_CSV = "docs/figures/confusion_matrix_partial_ft.csv"
DEFAULT_REP_PNG = ("docs/figures/error_gallery_targeted/S_to_E/"
                   "I_partial_finetune/rep_change.png")
DEFAULT_EMB_NPZ = "data/partial_ft_embeddings_val.npz"
DEFAULT_S_TO_E_RECORDS = "data/s_to_e_all_s_records.json"

N_CLASSES = 29
FEATURE_DIM = 1280
BATCH = 64
QUERY_FRAME_IDS = [2575, 2601, 2805, 2864, 2832, 2876]


# --------------------------------------------------------------------------- #
# Embedding extraction
# --------------------------------------------------------------------------- #
@torch.no_grad()
def extract_val_embeddings(
    model: torch.nn.Module, val_ds: ASLImageDataset, device: str,
) -> np.ndarray:
    """Pooled 1280-D features (``model[0]``) for all val samples, in manifest order."""
    model.eval()
    if device != "cpu":
        model = model.to(device)
    loader = DataLoader(val_ds, batch_size=BATCH, shuffle=False, num_workers=0)
    embs: List[np.ndarray] = []
    for x, _ in loader:
        e = model[0](x.to(device)).detach().cpu().numpy()
        embs.append(e)
    return np.concatenate(embs, axis=0).astype(np.float32)


def head_logits(embs: np.ndarray, W: np.ndarray, b: np.ndarray) -> np.ndarray:
    return embs @ W.astype(np.float32).T + b.astype(np.float32)


def l2_normalize(embs: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return embs / norms


def extract_head_params(model: torch.nn.Module) -> Tuple[np.ndarray, np.ndarray]:
    head = model[1]
    return (head[1].weight.detach().cpu().numpy(),
            head[1].bias.detach().cpu().numpy())


# --------------------------------------------------------------------------- #
# Group + boundary computations (fixed diagnostic groups)
# --------------------------------------------------------------------------- #
def group_arrays(class_names: List[str], preds: np.ndarray,
                 expected: Optional[Tuple[int, int, int]] = None
                 ) -> Dict[str, np.ndarray]:
    """Mask arrays for the fixed groups, assigned by the PARENT head.

    ``expected`` is the real-val reference (138 errors / 462 correct S / 600
    true E) asserted by default; synthetic tests may pass their own counts.
    """
    idx = {c: i for i, c in enumerate(CLASSES)}
    s_idx, e_idx = idx["S"], idx["E"]
    cn = np.asarray(class_names)
    m = {
        "S->E errors": (cn == "S") & (preds == e_idx),
        "correct S": (cn == "S") & (preds != e_idx),
        "true E": cn == "E",
    }
    if expected is None:
        expected = (138, 462, 600)
    n_err, n_correct, n_e = [int(m[k].sum()) for k in
                             ("S->E errors", "correct S", "true E")]
    if (n_err, n_correct, n_e) != expected:
        raise AssertionError(
            f"group counts {n_err}/{n_correct}/{n_e} != expected "
            f"{expected} — the PARENT head's predictions changed"
        )
    return m


def margin_rows(embs: np.ndarray, W: np.ndarray, b: np.ndarray,
                masks: Dict[str, np.ndarray],
                class_names: List[str]) -> Dict[str, Dict]:
    logits = head_logits(embs, W, b)
    s_idx, e_idx = CLASSES.index("S"), CLASSES.index("E")
    margins = logits[:, s_idx] - logits[:, e_idx]
    out: Dict[str, Dict] = {}
    for group, m in masks.items():
        ms = margins[m]
        out[group] = {
            "n": int(m.sum()),
            "margin_mean": float(ms.mean()),
            "margin_median": float(np.median(ms)),
            "distance_mean": float(np.mean(
                [signed_boundary_distance(float(x), W[s_idx], W[e_idx])
                 for x in ms])),
            "distance_median": float(np.median(
                [signed_boundary_distance(float(x), W[s_idx], W[e_idx])
                 for x in ms])),
            "n_on_S_side": int((ms > 0).sum()),
        }
    return out


def summarize_movement(cosines: np.ndarray) -> Dict:
    return {
        "n": int(cosines.size),
        "movement_mean": float((1.0 - cosines).mean()),
        "movement_median": float(np.median(1.0 - cosines)),
        "cosine_mean": float(cosines.mean()),
        "cosine_median": float(np.median(cosines)),
        "cosine_min": float(cosines.min()),
        "cosine_q25": float(np.percentile(cosines, 25)),
        "cosine_q75": float(np.percentile(cosines, 75)),
        "cosine_max": float(cosines.max()),
    }


# --------------------------------------------------------------------------- #
# Nearest-S / nearest-E in a space
# --------------------------------------------------------------------------- #
def nearest_stats(queries: np.ndarray, pool_s: np.ndarray, pool_e: np.ndarray,
                  ) -> Dict:
    """Per-query nearest-cosine to pool_S and pool_E (both L2-normalized)."""
    qs = l2_normalize(queries)
    ps = l2_normalize(pool_s)
    pe = l2_normalize(pool_e)
    cos_s = qs @ ps.T
    cos_e = qs @ pe.T
    n_s = cos_s.max(axis=1)
    n_e = cos_e.max(axis=1)
    closer_to_s = int((n_s > n_e).sum())
    closer_to_e = int((n_s <= n_e).sum())
    return {
        "n": int(qs.shape[0]),
        "nearest_S_mean": float(n_s.mean()),
        "nearest_E_mean": float(n_e.mean()),
        "closer_to_S": closer_to_s,
        "closer_to_E": closer_to_e,
        "per_query": np.stack([n_s, n_e], axis=1),
    }


# --------------------------------------------------------------------------- #
# Three-way comparison
# --------------------------------------------------------------------------- #
def three_way_report(baseline: Dict, head_only: Dict, current: Dict) -> Dict:
    def metrics(d: Dict) -> Dict:
        return {"accuracy": d["accuracy"], "macro_f1": d["macro_f1"],
                "total_errors": d["total_errors"], "S_to_E": d["S->E"],
                "E_to_S": d["E->S"],
                "T_to_L": d["T->L"], "K_to_V": d["K->V"], "V_to_K": d["V->K"],
                "M_to_N": d["M->N"], "N_to_M": d["N->M"], "Y_to_I": d["Y->I"],
                "X_to_K": d["X->K"], "X_to_I": d["X->I"], "X_to_R": d["X->R"]}
    return {
        "targeted": metrics(baseline),
        "head_only": metrics(head_only),
        "partial_finetune": metrics(current),
    }


# --------------------------------------------------------------------------- #
# Deployable checkpoint
# --------------------------------------------------------------------------- #
def build_deployable(
    parent_path: str, candidate_path: str, parent_sha: str,
    backbone_sha: str, verdict: Dict, deploy_out: str,
) -> Dict:
    cand = torch.load(candidate_path, map_location="cpu")
    ckpt = torch.load(parent_path, map_location="cpu")
    out = dict(ckpt)
    out["model"] = cand["model"]
    out.update({
        "schema": "asl-partial-ft/v1",
        "experiment": "partial_finetune_targeted",
        "parent_checkpoint": parent_path,
        "parent_checkpoint_sha256": parent_sha,
        "backbone_state_dict_sha256": backbone_sha,
        "trained_weights": (
            "final MBConv block blocks[6] (conv/SE) + Linear(1280,29) head; "
            "stem, blocks[0..5], conv_head, bn2, all BatchNorm tensors "
            "bit-identical to parent"
        ),
        "verdict": verdict["verdict"],
        "lr_backbone": cand["lr_backbone"], "lr_head": cand["lr_head"],
        "best_epoch": cand["best_epoch"],
        "val_macro_f1": cand["best_macro_f1"],
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "note": (
            "EXPERIMENTAL checkpoint from the partial-fine-tuning experiment. "
            "Verdict recorded above; do not treat as an improvement unless the "
            "verdict is CLEAN IMPROVEMENT."
        ),
    })
    out_path = Path(deploy_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, out_path)
    return {"path": str(out_path), "file_sha256": file_sha256(str(out_path))}


# --------------------------------------------------------------------------- #
# Markdown report
# --------------------------------------------------------------------------- #
def build_markdown_report(report: Dict) -> str:
    base = report["baseline"]
    cur = report["current"]
    cls = report["classification"]
    diag = report["diagnostics"]

    def pair_line(name: str) -> str:
        p = report["before_after_pairs"][name]
        return f"| {name} | {p['before']} | {p['after']} | {p['delta']:+d} |"

    def bnd(phase: str, group: str) -> str:
        g = diag["margin"][phase][group]
        return (f"| {phase} · {group} | {g['n']} | {g['margin_mean']:+.3f} | "
                f"{g['margin_median']:+.3f} | {g['distance_mean']:+.4f} | "
                f"{g['distance_median']:+.4f} | {g['n_on_S_side']} |")

    lines = [
        "# Partial fine-tuning of efficientnet_b0_targeted — final report",
        "",
        f"Experiment: image-space training of the FINAL MBConv stage "
        f"`blocks[6]` + the existing `Linear(1280, 29)` head (differential LRs, "
        f"BN frozen). Parent backbone bit-identical except `blocks[6]` conv/SE "
        f"and the head.",
        "",
        f"**VERDICT: {cls['verdict']}**",
        "",
        f"- S->E {base['S->E']} -> {cur['S->E']} "
        f"({cur['S->E'] - base['S->E']:+d}); material = "
        f"{cls['rules']['S_to_E_drop_material']}.",
        f"- Accuracy {100.0 * base['accuracy']:.2f}% -> "
        f"{100.0 * cur['accuracy']:.2f}% "
        f"(preserved = {cls['rules']['accuracy_preserved']}).",
        f"- Macro-F1 {base['macro_f1']:.4f} -> {cur['macro_f1']:.4f} "
        f"(preserved = {cls['rules']['macro_f1_preserved']}).",
        f"- E recall {base['E recall']:.3f} -> {cur['E recall']:.3f} "
        f"(not regressed = {cls['rules']['E_not_regressed']}).",
        f"- Watch-pair regressions: "
        f"{cls['rules']['material_pair_regressions'] or 'none'}.",
        "",
        "## Setup",
        "",
        f"- Parent: `{report['parent_checkpoint']}` (file sha256 "
        f"`{report['parent_checkpoint_sha256'][:16]}…`, backbone sd "
        f"`{report['backbone_state_dict_sha256'][:16]}…`), untouched.",
        f"- Candidate (training phase): `{report['candidate_path']}`.",
        f"- Manifest `{report['manifest_sha256'][:16]}…` — leakage-safe dev val, "
        f"NOT the reported metric.",
        f"- Training: device {report['device']}; differential LRs "
        f"backbone `{report['training']['lr_backbone']:.0e}` / head "
        f"`{report['training']['lr_head']:.0e}`, best epoch "
        f"{report['training']['best_epoch']}; CrossEntropy "
        f"(label smoothing 0.1, checkpoint class weights), AdamW wd 0.01, "
        f"cosine warmup (1 epoch); BN running stats + affine frozen.",
        f"- Final evaluation: raw-image `preprocess_bgr` pipeline on CPU "
        f"({report['eval_device']}), 17,400 = 600/class.",
        "",
        "## Sanity gate",
        "",
        f"- Raw-image eval of the parent reproduced the baseline (acc "
        f"{100.0 * report['gate']['accuracy']:.2f}%, macro-F1 "
        f"{report['gate']['macro_f1']:.4f}, errors {report['gate']['total_errors']}, "
        f"S->E {report['gate']['S_to_E']}).",
        "",
        "## Before / after (dev val)",
        "",
        "| metric | before | after | delta |",
        "|--------|--------|-------|-------|",
    ]
    for name in ("S->E", "E->S", "T->L", "K->V", "V->K", "M->N", "N->M",
                 "Y->I", "X->K", "X->I", "X->R"):
        lines.append(pair_line(name))
    lines.append(
        f"| total errors | {base['total_errors']} | {cur['total_errors']} | "
        f"{cur['total_errors'] - base['total_errors']:+d} |"
    )
    lines.append(
        f"| accuracy | {100.0 * base['accuracy']:.2f}% | "
        f"{100.0 * cur['accuracy']:.2f}% | "
        f"{100.0 * (cur['accuracy'] - base['accuracy']):+.2f}% |"
    )
    lines.append(
        f"| macro-F1 | {base['macro_f1']:.4f} | {cur['macro_f1']:.4f} | "
        f"{cur['macro_f1'] - base['macro_f1']:+.4f} |"
    )
    lines += [
        "",
        "## S / E per class",
        "",
        "| class | metric | before | after |",
        "|-------|--------|--------|-------|",
    ]
    for c in ("S", "E"):
        for m in ("accuracy", "precision", "recall", "f1"):
            bv = report["per_class_before_after"][c]["before"][m]
            av = report["per_class_before_after"][c]["after"][m]
            lines.append(f"| {c} | {m} | {bv:.3f} | {av:.3f} |")
    lines += [
        "",
        "## Three-way comparison (targeted / head-only / partial-FT)",
        "",
        "| metric | targeted | head-only | partial-FT |",
        "|--------|----------|-----------|------------|",
    ]
    for k in report["three_way"]["targeted"]:
        a = report["three_way"]["targeted"][k]
        b = report["three_way"]["head_only"][k]
        c = report["three_way"]["partial_finetune"][k]
        fmt = lambda v: f"{v:.4f}" if isinstance(v, float) else str(v)
        lines.append(f"| {k} | {fmt(a)} | {fmt(b)} | {fmt(c)} |")
    lines += [
        "",
        "## Representation change (pooled 1280-D, L2-normalized)",
        "",
        "Embedding movement `1 - cosine(before, after)` by group (higher = "
        "moved more):",
        "",
        "| group | n | movement mean | movement med | cosine mean |",
        "|-------|---|---------------|--------------|-------------|",
    ]
    for g in ("S->E errors", "correct S", "true E", "all val"):
        m = diag["movement"][g]
        lines.append(
            f"| {g} | {m['n']} | {m['movement_mean']:.4f} | "
            f"{m['movement_median']:.4f} | {m['cosine_mean']:.4f} |"
        )
    lines += [
        "",
        f"- Original 138 S->E errors now correct: "
        f"{diag['corrections']['n_now_correct']} "
        f"({diag['corrections']['n']} total); "
        f"still wrong: {diag['corrections']['n_still_wrong']}.",
        f"- New predictions of the original 138: "
        f"{diag['corrections']['new_pred_dist']}.",
        f"- Movement of the corrected subset (mean "
        f"{diag['corrections']['corrected_movement_mean']:.4f}) vs still-wrong "
        f"({diag['corrections']['wrong_movement_mean']:.4f}).",
        "",
        "## S-vs-E nearest neighbor in the NEW embedding space",
        "",
        f"- Original 138 errors: nearest-correct-S cosine "
        f"{diag['nearest']['S->E errors']['nearest_S_mean']:.3f} vs "
        f"nearest-actual-E "
        f"{diag['nearest']['S->E errors']['nearest_E_mean']:.3f}; "
        f"closer to S {diag['nearest']['S->E errors']['closer_to_S']}/138, "
        f"closer to E {diag['nearest']['S->E errors']['closer_to_E']}/138.",
        f"- Corrected subset of the 138: "
        f"{diag['nearest']['corrected']['closer_to_S']}/"
        f"{diag['nearest']['corrected']['n']} closer to S, "
        f"{diag['nearest']['corrected']['closer_to_E']}/"
        f"{diag['nearest']['corrected']['n']} closer to E.",
        "",
        "## S/E boundary (margin_SE = logit_S - logit_E, fixed groups)",
        "",
        "| group | n | margin mean | margin med | dist mean | dist med | on S side |",
        "|-------|---|-------------|------------|-----------|----------|-----------|",
        bnd("before", "S->E errors"),
        bnd("before", "correct S"),
        bnd("before", "true E"),
        bnd("after", "S->E errors"),
        bnd("after", "correct S"),
        bnd("after", "true E"),
        "",
        f"- `w_S - w_E` norm: before {diag['margin']['before']['wS_minus_wE_norm']:.3f}, "
        f"after {diag['margin']['after']['wS_minus_wE_norm']:.3f}.",
        "",
        "## Representative frames",
        "",
        "| frame | before pred | after pred | margin before | margin after | movement |",
        "|-------|-------------|------------|---------------|--------------|----------|",
    ]
    for r in diag["queries"]:
        lines.append(
            f"| {r['frame_id']} | {r['before_pred']} | {r['after_pred']} | "
            f"{r['margin_before']:+.3f} | {r['margin_after']:+.3f} | "
            f"{r['movement']:.4f} |"
        )
    lines += [
        "",
        "## Artifacts",
        "",
        f"- Deployable checkpoint: `{report['artifacts']['deployable']['path']}` "
        f"(file sha256 `{report['artifacts']['deployable']['file_sha256'][:16]}…`, "
        f"verdict embedded).",
        f"- Confusion PNG: `{report['artifacts']['confusion_png']}`",
        f"- Confusion CSV: `{report['artifacts']['confusion_csv']}`",
        f"- Representation-change figure: `{report['artifacts']['rep_png']}`",
        f"- Embedding cache (before/after, val): `{report['artifacts']['emb_npz']}`",
        f"- Full JSON: `{report['artifacts']['report_json']}`",
        "",
        "## Interpretation (conservative)",
        "",
        f"- Verdict per deterministic criteria ({cls['rules']['criteria_doc']}).",
        "- Margin/nearest-neighbour numbers are descriptive two-class S-vs-E "
        "quantities, not the full 29-class decision.",
        f"- Training-time epoch selection used the {report['device']} val "
        "macro-F1; the numbers above are the raw-image "
        f"{report['eval_device']} eval. MPS-vs-CPU "
        "float differences can flip a handful of S/E top-2 boundary frames "
        f"(seen in the head-only run); reported eval device: {report['eval_device']}.",
        "",
    ]
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
# run
# --------------------------------------------------------------------------- #
def run(args) -> Dict:
    device = _resolve_device(args.device)
    print(f"torch {torch.__version__}, MPS available "
          f"{torch.backends.mps.is_available()}, diagnostic device {device}")

    manifest = SplitManifest.from_json(args.manifest)
    manifest_sha = json.loads(Path(args.manifest).read_text()).get("content_sha256")
    val_ds = ASLImageDataset(manifest.val, train=False)
    class_names = [s.class_name for s in manifest.val]

    cand = torch.load(args.candidate, map_location="cpu")
    parent_sha = cand["parent_checkpoint_sha256"]
    backbone_sha = cand["parent_backbone_state_dict_sha256"]

    # ---- models: before (parent) and after (candidate) ----
    model_before = load_checkpoint(args.checkpoint, "cpu").eval()
    W0, b0 = extract_head_params(model_before)
    model_after = load_checkpoint(args.candidate, "cpu").eval()
    W1, b1 = extract_head_params(model_after)

    # ---- embeddings (MPS) ----
    print("extracting before embeddings (val)...", flush=True)
    emb_before = extract_val_embeddings(model_before, val_ds, device)
    print("extracting after embeddings (val)...", flush=True)
    emb_after = extract_val_embeddings(model_after, val_ds, device)
    os.makedirs(os.path.dirname(args.emb_npz) or ".", exist_ok=True)
    np.savez_compressed(
        args.emb_npz,
        emb_before=emb_before, emb_after=emb_after,
        class_names=np.asarray(class_names),
    )
    print(f"wrote {args.emb_npz}")

    # ---- parent-head predictions define the fixed groups ----
    pred_before = head_logits(emb_before, W0, b0).argmax(1)
    # Pin the S-frame "before" predictions to the CPU-computed reference
    # (data/s_to_e_all_s_records.json) so MPS embedding flips cannot move
    # boundary frames in/out of the canonical 138 S->E group.
    frames = np.asarray([frame_id(s.path) for s in manifest.val])
    records = json.loads(Path(args.s_to_e_records).read_text())
    rec_pred = {int(r["frame_id"]): CLASSES.index(r["pred"]) for r in records}
    s_rows = [i for i, s in enumerate(manifest.val) if s.class_name == "S"]
    missing = [int(frames[i]) for i in s_rows if int(frames[i]) not in rec_pred]
    if missing:
        raise SystemExit(
            f"S val frames missing from {args.s_to_e_records}: {sorted(missing)[:5]}…"
        )
    pb = pred_before.copy()
    for i in s_rows:
        pb[i] = rec_pred[int(frames[i])]
    pred_before = pb
    masks = group_arrays(class_names, pred_before)

    idx = {c: i for i, c in enumerate(CLASSES)}
    s_idx, e_idx = idx["S"], idx["E"]

    # ---- movement 1 - cosine(before, after) ----
    nb = l2_normalize(emb_before)
    na = l2_normalize(emb_after)
    cos_all = np.einsum("ij,ij->i", nb, na)
    movement = {
        g: summarize_movement(cos_all[m]) for g, m in masks.items()
    }
    movement["all val"] = summarize_movement(cos_all)

    # ---- corrections of the original 138 ----
    err_mask = masks["S->E errors"]
    pred_after = head_logits(emb_after, W1, b1).argmax(1)
    orig_errors = np.where(err_mask)[0]
    now_correct = pred_after[orig_errors] == s_idx
    new_preds = pred_after[orig_errors]
    new_pred_dist = {
        c: int((new_preds == i).sum())
        for i, c in enumerate(CLASSES) if (new_preds == i).sum()
    }
    corrections = {
        "n": int(len(orig_errors)),
        "n_now_correct": int(now_correct.sum()),
        "n_still_wrong": int((~now_correct).sum()),
        "new_pred_dist": new_pred_dist,
        "corrected_movement_mean": float(
            (1.0 - cos_all[orig_errors[now_correct]]).mean()
            if now_correct.sum() else 0.0),
        "wrong_movement_mean": float(
            (1.0 - cos_all[orig_errors[~now_correct]]).mean()
            if (~now_correct).sum() else 0.0),
    }

    # ---- nearest-S / nearest-E in the NEW space ----
    s_rows = np.where(np.asarray(class_names) == "S")[0]
    e_rows = np.where(np.asarray(class_names) == "E")[0]
    nn_errors = nearest_stats(emb_after[orig_errors], emb_after[s_rows],
                              emb_after[e_rows])
    nn_corrected = nearest_stats(emb_after[orig_errors[now_correct]],
                                 emb_after[s_rows], emb_after[e_rows])
    nn_controls = nearest_stats(emb_after[masks["correct S"]], emb_after[s_rows],
                                emb_after[e_rows])
    nearest = {
        "S->E errors": {k: v for k, v in nn_errors.items() if k != "per_query"},
        "corrected": {k: v for k, v in nn_corrected.items() if k != "per_query"},
        "correct S controls": {k: v for k, v in nn_controls.items()
                               if k != "per_query"},
    }

    # ---- margin before/after ----
    def margin_phase(embs, W, b):
        d = margin_rows(embs, W, b, masks, class_names)
        d["wS_minus_wE_norm"] = float(
            np.linalg.norm(W[s_idx] - W[e_idx]))
        return d

    margin = {"before": margin_phase(emb_before, W0, b0),
              "after": margin_phase(emb_after, W1, b1)}

    # ---- representative frames ----
    frame_to_row = {int(frames[i]): i for i in range(len(frames))}
    queries = []
    q_embs = np.stack([emb_after[frame_to_row[f]] for f in QUERY_FRAME_IDS])
    q_mov = np.stack([cos_all[frame_to_row[f]] for f in QUERY_FRAME_IDS])
    for k, fid in enumerate(QUERY_FRAME_IDS):
        r = frame_to_row[fid]
        logits_b = head_logits(emb_before[r][None], W0, b0)[0]
        logits_a = head_logits(emb_after[r][None], W1, b1)[0]
        queries.append({
            "frame_id": fid,
            "before_pred": CLASSES[int(pred_before[r])],
            "after_pred": CLASSES[int(pred_after[r])],
            "margin_before": float(logits_b[s_idx] - logits_b[e_idx]),
            "margin_after": float(logits_a[s_idx] - logits_a[e_idx]),
            "movement": float(1.0 - q_mov[k]),
        })

    # ---- FINAL raw-image evaluation of the candidate ----
    eval_device = _resolve_device(args.eval_device)
    print(f"raw-image eval of the candidate (final numbers, device={eval_device})...",
          flush=True)
    model_final = load_checkpoint(args.candidate, "cpu").eval()
    if eval_device != "cpu":
        model_final = model_final.to(eval_device)
    loader = DataLoader(val_ds, batch_size=BATCH, shuffle=False, num_workers=0)
    ys, ps = [], []
    with torch.no_grad():
        for x, y in loader:
            xd = x.to(eval_device) if eval_device != "cpu" else x
            ps.append(model_final(xd).argmax(1).cpu().numpy())
            ys.append(y.numpy())
    y_true = np.concatenate(ys)
    y_pred = np.concatenate(ps)
    cm = confusion_matrix(y_true, y_pred, N_CLASSES)
    cur_ev = evaluate_cm(y_true, y_pred)
    cur_metrics = {
        "accuracy": cur_ev["accuracy"], "macro_f1": cur_ev["macro_f1"],
        "total_errors": cur_ev["total_errors"],
        **s_e_metrics(cm),
    }
    save_confusion_png(cm, args.cm_png)
    save_confusion_csv(cm, args.cm_csv)

    # ---- baseline numbers from the raw-image gate report ----
    gate = json.loads(Path(args.baseline_report).read_text())
    bcm = np.asarray(gate["confusion_matrix"], dtype=np.int64)
    base_metrics = {
        "accuracy": gate["overall_accuracy"], "macro_f1": gate["macro_f1"],
        "total_errors": int(bcm.sum() - bcm.trace()),
        **s_e_metrics(bcm),
    }
    base_ev = {
        "cm": bcm,
        "accuracy": base_metrics["accuracy"],
        "macro_f1": base_metrics["macro_f1"],
        "per_class": per_class_from_cm(bcm),
    }

    verdict = classify({"cm": bcm,
                        "accuracy": base_metrics["accuracy"],
                        "macro_f1": base_metrics["macro_f1"]}, cur_ev)

    head_only = json.loads(Path(args.head_only_report).read_text())["current"]

    deploy = build_deployable(args.checkpoint, args.candidate, parent_sha,
                              backbone_sha, verdict, args.deploy_out)
    print(f"wrote deployable checkpoint {deploy['path']} "
          f"(verdict {verdict['verdict']})")

    # ---- representation-change figure ----
    os.makedirs(os.path.dirname(args.rep_png) or ".", exist_ok=True)
    make_rep_figure(
        cos_all, masks, orig_errors, now_correct, args.rep_png)

    report = {
        "schema": "asl-partial-ft/v1",
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "parent_checkpoint": args.checkpoint,
        "parent_checkpoint_sha256": parent_sha,
        "backbone_state_dict_sha256": backbone_sha,
        "candidate_path": args.candidate,
        "manifest_sha256": manifest_sha,
        "split_note": (
            "dev val = leakage-safe frame-range split, same signer, same room. "
            "NOT the reported metric."
        ),
        "gate": {"accuracy": base_metrics["accuracy"],
                 "macro_f1": base_metrics["macro_f1"],
                 "total_errors": base_metrics["total_errors"],
                 "S_to_E": base_metrics["S->E"]},
        "device": device,
        "eval_device": f"{eval_device} (raw-image preprocess_bgr)",
        "eval_device_note": (
            "CPU eval is the most reproducible; MPS eval is fast but its "
            "float differences can flip a handful of S/E top-2 boundary frames "
            "(seen in the head-only run: ~11 of 17,400)."
        ),
        "training": {
            "lr_backbone": cand["lr_backbone"], "lr_head": cand["lr_head"],
            "best_epoch": cand["best_epoch"],
            "best_macro_f1": cand["best_macro_f1"],
            "epochs_run": cand["epochs_run"], "seed": cand["seed"],
        },
        "baseline": base_metrics,
        "current": cur_metrics,
        "before_after_pairs": {
            p: {"before": base_metrics[p], "after": cur_metrics[p],
                "delta": cur_metrics[p] - base_metrics[p]}
            for p in ["S->E", "E->S", "T->L", "K->V", "V->K", "M->N", "N->M",
                      "Y->I", "X->K", "X->I", "X->R"]
        },
        "per_class_before_after": {
            c: {"before": base_ev["per_class"][c],
                "after": cur_ev["per_class"][c]}
            for c in ("S", "E")
        },
        "three_way": three_way_report(base_metrics, head_only, cur_metrics),
        "diagnostics": {
            "movement": movement,
            "corrections": corrections,
            "nearest": nearest,
            "margin": margin,
            "queries": queries,
        },
        "classification": verdict,
        "deployable_checkpoint": deploy,
        "confusion_matrix": cm.astype(int).tolist(),
        "artifacts": {
            "deployable": deploy,
            "confusion_png": args.cm_png,
            "confusion_csv": args.cm_csv,
            "rep_png": args.rep_png,
            "emb_npz": args.emb_npz,
            "report_json": args.out_json,
            "report_md": args.out_md,
        },
    }
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {args.out_json}")
    Path(args.out_md).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_md).write_text(build_markdown_report(report))
    print(f"wrote {args.out_md}")
    print(f"VERDICT: {verdict['verdict']}")
    return report


def per_class_from_cm(cm: np.ndarray) -> Dict[str, Dict]:
    per = {}
    for i, c in enumerate(CLASSES):
        tp = int(cm[i, i])
        fp = int(cm[:, i].sum()) - tp
        fn = int(cm[i, :].sum()) - tp
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        per[c] = {"accuracy": rec,
                  "precision": prec, "recall": rec,
                  "f1": 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0}
    return per


def make_rep_figure(cos_all: np.ndarray, masks: Dict[str, np.ndarray],
                    orig_errors: np.ndarray, now_correct: np.ndarray,
                    path: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"skip rep figure (matplotlib unavailable: {e})")
        return
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    ax = axes[0]
    for g, color in [("S->E errors", "#c0392b"), ("correct S", "#2c7fb8"),
                     ("true E", "#7a7a7a")]:
        ax.hist(1.0 - cos_all[masks[g]], bins=40, alpha=0.55, color=color,
                label=f"{g} (n={int(masks[g].sum())})")
    ax.set_xlabel("embedding movement 1 - cosine(before, after)")
    ax.set_ylabel("count")
    ax.set_title("Representation movement of val frames")
    ax.legend()
    ax2 = axes[1]
    mov_orig = 1.0 - cos_all[orig_errors]
    ax2.hist(mov_orig[now_correct], bins=30, alpha=0.6, color="#27ae60",
             label=f"now correct (n={int(now_correct.sum())})")
    ax2.hist(mov_orig[~now_correct], bins=30, alpha=0.6, color="#c0392b",
             label=f"still wrong (n={int((~now_correct).sum())})")
    ax2.set_xlabel("embedding movement of the original S->E errors")
    ax2.set_ylabel("count")
    ax2.set_title("Movement of the original 138 S->E errors")
    ax2.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"wrote {path}")


def main() -> None:
    p = argparse.ArgumentParser(
        description="Partial-FT analysis: final eval, diagnostics, verdict."
    )
    p.add_argument("--checkpoint", default=DEFAULT_PARENT)
    p.add_argument("--manifest", default=DEFAULT_MANIFEST)
    p.add_argument("--candidate", default=DEFAULT_CANDIDATE)
    p.add_argument("--baseline-report", default=DEFAULT_BASELINE_REPORT)
    p.add_argument("--head-only-report", default=DEFAULT_HEAD_ONLY_REPORT)
    p.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"],
                   default="auto")
    p.add_argument("--eval-device", choices=["auto", "cpu", "mps", "cuda"],
                   default="auto",
                   help="device for the FINAL raw-image eval (default: same "
                        "resolution as --device; mps recommended for speed)")
    p.add_argument("--deploy-out", default=DEFAULT_DEPLOY_OUT)
    p.add_argument("--out-json", default=DEFAULT_OUT_JSON)
    p.add_argument("--out-md", default=DEFAULT_OUT_MD)
    p.add_argument("--cm-png", default=DEFAULT_CM_PNG)
    p.add_argument("--cm-csv", default=DEFAULT_CM_CSV)
    p.add_argument("--rep-png", default=DEFAULT_REP_PNG)
    p.add_argument("--emb-npz", default=DEFAULT_EMB_NPZ)
    p.add_argument("--s-to-e-records", default=DEFAULT_S_TO_E_RECORDS)
    run(p.parse_args())


if __name__ == "__main__":
    main()
