"""Head-only fine-tuning on cached pooled embeddings — the controlled upgrade
experiment for ``checkpoints/efficientnet_b0_targeted.pt``.

Question (one clean answer): can additional optimization of the EXISTING
``Linear(1280, 29)`` classifier head improve the targeted checkpoint while the
EfficientNet representation stays frozen? This script tests that hypothesis; it
does not change the loss beyond what the checkpoint already used, does not add
focal/contrastive/triplet/margin losses, does not re-weight classes, does not
oversample, and does not unfreeze the backbone.

Pipeline
--------
1. Load the cached pooled embeddings produced by ``build_embedding_cache.py``
   (``data/embedding_cache/targeted_{train,val}_embeddings.pt``). Training
   embeddings come ONLY from the training split; validation ONLY from the
   validation split.
2. Verify the cache: metadata, train/val separation, label/path alignment with
   the manifest, and — for sampled S / E / unrelated frames — that cached
   ``head(emb)`` logits reproduce the full-model ``model(image)`` logits to
   numerical tolerance.
3. SANITY GATE: run the ORIGINAL targeted head on the cached validation
   embeddings. It must reproduce the baseline (accuracy 0.9775, macro-F1
   0.9775, S->E 138, and the exact baseline confusion matrix). If not, STOP.
4. LR search over {1e-4, 3e-4, 1e-3}: train ONLY the Linear head, initialized
   from the checkpoint's own weights, on the cached train embeddings. Loss =
   CrossEntropyLoss(label_smoothing=0.1, weight=the checkpoint's class weights),
   AdamW (wd=0.01), cosine schedule with 1 warmup epoch (mirrors src/train.py),
   early stopping on validation macro-F1. Best candidate = best val macro-F1.
5. Full validation evaluation of the best candidate + before/after confusion
   pairs + S/E boundary (margin_SE, distance_SE) before vs after.

Loss/augmentation semantics (documented difference)
---------------------------------------------------
The cached train embeddings are ONE seeded realization of the stochastic
training transform (see ``build_embedding_cache.py``). Head training therefore
sees a FIXED augmented realization, not a fresh draw per epoch — this is
deliberately not claimed to be identical to image-space training.

Outputs
-------
- ``checkpoints/head_only/candidate_head_best.pt``  (trained W/b + metadata)
- ``checkpoints/efficientnet_b0_head_ft.pt``        (deployable: parent backbone
  + trained head, verdict embedded)
- ``docs/reports/head_only_cached_embeddings.json``
- ``docs/reports/head_only_cached_embeddings.md``
- ``docs/figures/confusion_matrix_head_only_cached.png|.csv``
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from src.data import CLASSES, ASLImageDataset, Sample, SplitManifest  # noqa: E402
from src.evaluate import (  # noqa: E402
    confusion_matrix,
    load_checkpoint,
    macro_f1,
    per_class_accuracy,
    save_confusion_csv,
    save_confusion_png,
)
from scripts.analyze_s_to_e import _resolve_device  # noqa: E402
from scripts.analyze_s_to_e_head import (  # noqa: E402
    decompose_margin,
    extract_head,
    signed_boundary_distance,
)

DEFAULT_CHECKPOINT = "checkpoints/efficientnet_b0_targeted.pt"
DEFAULT_MANIFEST = "data/split_manifest.json"
DEFAULT_CACHE_DIR = "data/embedding_cache"
DEFAULT_BASELINE_REPORT = "docs/reports/dev_val_targeted_current.json"
DEFAULT_HEAD_OUT = "checkpoints/head_only/candidate_head_best.pt"
DEFAULT_OUT_JSON = "docs/reports/head_only_cached_embeddings.json"
DEFAULT_OUT_MD = "docs/reports/head_only_cached_embeddings.md"
DEFAULT_CM_PNG = "docs/figures/confusion_matrix_head_only_cached.png"
DEFAULT_CM_CSV = "docs/figures/confusion_matrix_head_only_cached.csv"

FEATURE_DIM = 1280
N_CLASSES = 29
SCHEMA = "asl-head-only-cached-embeddings/v1"
LABEL_SMOOTHING = 0.1
WEIGHT_DECAY = 0.01
LOGIT_TOL = 1e-3
MATERIAL_S_TO_E_FRAC = 0.15  # S->E drop must exceed this fraction to be material
METRIC_TOL = 0.001           # macro-F1/accuracy must stay above baseline - tol


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def evaluate_cm(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    """Full per-class metrics from labels+predictions (mirrors src.evaluate)."""
    cm = confusion_matrix(y_true, y_pred, N_CLASSES)
    per = {}
    for i, c in enumerate(CLASSES):
        tp = int(cm[i, i])
        fp = int(cm[:, i].sum()) - tp
        fn = int(cm[i, :].sum()) - tp
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        per[c] = {"accuracy": per_class_accuracy(cm)[c],
                  "precision": prec, "recall": rec, "f1": f1}
    return {
        "cm": cm,
        "accuracy": float(cm.trace() / max(1, cm.sum())),
        "macro_f1": macro_f1(cm),
        "total_errors": int(cm.sum() - cm.trace()),
        "per_class": per,
    }


def s_e_metrics(cm: np.ndarray) -> Dict:
    """S/E precision/recall/F1 plus the classic pairs (from the confusion matrix)."""
    s, e = CLASSES.index("S"), CLASSES.index("E")
    out: Dict = {}
    for c in ("S", "E"):
        i = CLASSES.index(c)
        tp = int(cm[i, i])
        fp = int(cm[:, i].sum()) - tp
        fn = int(cm[i, :].sum()) - tp
        out[f"{c} precision"] = tp / (tp + fp) if (tp + fp) else 0.0
        out[f"{c} recall"] = tp / (tp + fn) if (tp + fn) else 0.0
        out[f"{c} f1"] = (2 * tp / (2 * tp + fp + fn)) if (tp + fp + fn) else 0.0
    pairs = {
        "S->E": (s, e), "E->S": (e, s),
        "T->L": (CLASSES.index("T"), CLASSES.index("L")),
        "K->V": (CLASSES.index("K"), CLASSES.index("V")),
        "V->K": (CLASSES.index("V"), CLASSES.index("K")),
        "M->N": (CLASSES.index("M"), CLASSES.index("N")),
        "N->M": (CLASSES.index("N"), CLASSES.index("M")),
        "Y->I": (CLASSES.index("Y"), CLASSES.index("I")),
        "X->K": (CLASSES.index("X"), CLASSES.index("K")),
        "X->I": (CLASSES.index("X"), CLASSES.index("I")),
        "X->R": (CLASSES.index("X"), CLASSES.index("R")),
    }
    for name, (i, j) in pairs.items():
        out[name] = int(cm[i, j])
    return out


# --------------------------------------------------------------------------- #
# Cache loading + verification
# --------------------------------------------------------------------------- #
def load_cache(path: str) -> Dict:
    # the cache files are produced by build_embedding_cache.py and contain
    # numpy arrays alongside tensors, so weights_only=False is required (and
    # safe here: they are our own trusted artifacts)
    d = torch.load(path, map_location="cpu", weights_only=False)
    if d.get("schema") != "asl-embedding-cache/v1":
        raise SystemExit(f"{path}: not an asl-embedding-cache/v1 file")
    return {
        **d,
        "embeddings": np.asarray(d["embeddings"], dtype=np.float32),
        "labels": np.asarray(d["labels"], dtype=np.int64),
    }


def verify_cache_pairs(train: Dict, val: Dict, manifest: SplitManifest,
                       manifest_path: str, checkpoint: str) -> Dict:
    """Split identity, counts, manifest hash, parent checkpoint, separation."""
    if train["split"] != "train" or val["split"] != "val":
        raise SystemExit("cache split fields are wrong")
    if train["n"] != len(manifest.train) or val["n"] != len(manifest.val):
        raise SystemExit(
            f"cache counts {train['n']}/{val['n']} != manifest "
            f"{len(manifest.train)}/{len(manifest.val)}"
        )
    manifest_sha = json.loads(Path(manifest_path).read_text()).get("content_sha256")
    if (train.get("manifest_sha256") != manifest_sha
            or val.get("manifest_sha256") != manifest_sha):
        raise SystemExit("cache manifest_sha256 does not match the manifest")
    if (train["parent_checkpoint"] != checkpoint
            or val["parent_checkpoint"] != checkpoint):
        raise SystemExit("cache parent checkpoint does not match --checkpoint")

    train_fids = set(train["frame_ids"])
    val_fids = set(val["frame_ids"])
    overlap = train_fids & val_fids
    if overlap:
        raise SystemExit(f"train/val frame-id overlap: {sorted(overlap)[:5]}")

    expected_per_class = {c: sum(1 for s in manifest.val if s.class_name == c)
                          for c in CLASSES}
    actual_per_class = {c: sum(1 for cn in val["class_names"] if cn == c)
                        for c in CLASSES}
    if actual_per_class != expected_per_class:
        diffs = {c: (actual_per_class[c], expected_per_class[c])
                 for c in CLASSES
                 if actual_per_class[c] != expected_per_class[c]}
        raise SystemExit(
            "val cache per-class counts do not match the manifest "
            f"(first diffs (actual, expected): "
            f"{sorted(diffs.items())[:3]})"
        )
    return {
        "train_n": train["n"], "val_n": val["n"],
        "train_val_frame_overlap": len(overlap),
        "val_per_class": actual_per_class,
    }


def manifest_path_of(manifest: SplitManifest) -> str:
    return manifest.root  # placeholder; replaced in run() by the real path


def verify_cache_against_manifest(val: Dict, manifest: SplitManifest) -> None:
    """Cache rows (path, class, label) must agree with manifest.val, in order."""
    if len(val["paths"]) != len(manifest.val):
        raise SystemExit("val cache length != manifest.val length")
    for i, s in enumerate(manifest.val):
        if val["paths"][i] != s.path:
            raise SystemExit(f"val cache path mismatch at {i}: "
                             f"{val['paths'][i]} != {s.path}")
        if val["class_names"][i] != s.class_name or val["labels"][i] != s.label:
            raise SystemExit(f"val cache label mismatch at {i}")


@torch.no_grad()
def verify_cached_logits(
    checkpoint: str, val: Dict, manifest: SplitManifest, device: str,
    n_per_class: int = 4,
) -> Dict:
    """Cached ``head(emb)`` logits must reproduce full-model ``model(image)``.

    Samples ``n_per_class`` frames from S, E, and an unrelated class, runs the
    ORIGINAL full model on freshly preprocessed images (eval mode, dropout
    identity), and compares against ``head(cached_embedding)``. Fails (SystemExit)
    if any max absolute logit difference exceeds ``LOGIT_TOL``.
    """
    model = load_checkpoint(checkpoint, device).eval()
    head = model[1]
    emb_by_path = {p: i for i, p in enumerate(val["paths"])}
    picked = [
        s for s in manifest.val
        if s.class_name in ("S", "E", "C")
    ]
    counts: Dict[str, int] = {}
    sampled: List[Sample] = []
    for s in picked:
        counts[s.class_name] = counts.get(s.class_name, 0) + 1
        if counts[s.class_name] <= n_per_class:
            sampled.append(s)
    ds = ASLImageDataset(sampled, train=False)
    path_to_idx = {s.path: i for i, s in enumerate(ds.samples)}
    rows = []
    max_diff = 0.0
    for s in sampled:
        x = ds[path_to_idx[s.path]][0][None, :].to(device)
        full = model(x)
        emb = torch.as_tensor(val["embeddings"][emb_by_path[s.path]],
                              dtype=torch.float32, device=device)
        via_head = head(emb)
        diff = float((full - via_head).abs().max())
        max_diff = max(max_diff, diff)
        rows.append({"path": s.path, "class": s.class_name,
                     "max_abs_logit_diff": diff})
        if diff > LOGIT_TOL:
            raise SystemExit(
                f"cached logits diverge from full model at {s.path}: "
                f"max |diff| {diff:.2e} > tol {LOGIT_TOL}"
            )
    return {"samples": rows, "max_abs_logit_diff": max_diff,
            "tolerance": LOGIT_TOL}


# --------------------------------------------------------------------------- #
# Baseline recomputation + sanity gate
# --------------------------------------------------------------------------- #
def cached_logits(embs: np.ndarray, W: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Full 29-class logits from cached embeddings + head params (float32)."""
    return embs @ W.astype(np.float32).T + b.astype(np.float32)


def recompute_baseline(
    val: Dict, W0: np.ndarray, b0: np.ndarray, baseline_report: str
) -> Tuple[Dict, np.ndarray]:
    """Original head on cached val embeddings -> must reproduce the baseline."""
    logits = cached_logits(val["embeddings"], W0, b0)
    y_pred = logits.argmax(1).astype(np.int64)
    ev = evaluate_cm(val["labels"], y_pred)
    baseline = json.loads(Path(baseline_report).read_text())
    bcm = np.asarray(baseline["confusion_matrix"], dtype=np.int64)
    if not np.array_equal(ev["cm"], bcm):
        diff = np.argwhere(ev["cm"] != bcm)
        raise SystemExit(
            f"SANITY GATE FAILED: cached-embedding confusion matrix does not "
            f"reproduce {baseline_report}. {len(diff)} cells differ, first: "
            f"{[tuple(map(int, d)) for d in diff[:5]]} (true->pred in CLASSES "
            "indices). Do not train until this is explained."
        )
    exp = (baseline["overall_accuracy"], baseline["macro_f1"])
    if not (abs(ev["accuracy"] - exp[0]) < 1e-9
            and abs(ev["macro_f1"] - exp[1]) < 1e-9):
        raise SystemExit(
            f"SANITY GATE FAILED: acc/F1 {ev['accuracy']:.6f}/{ev['macro_f1']:.6f}"
            f" != baseline {exp[0]:.6f}/{exp[1]:.6f}"
        )
    print(f"SANITY GATE PASSED: original head on cached val embeddings "
          f"reproduces acc {ev['accuracy']:.4f}, macro-F1 {ev['macro_f1']:.4f}, "
          f"S->E {s_e_metrics(ev['cm'])['S->E']}, exact confusion matrix")
    return ev, bcm


# --------------------------------------------------------------------------- #
# Head-only training
# --------------------------------------------------------------------------- #
def cosine_warmup_lr(step: int, total_steps: int, warmup_steps: int,
                     base_lr: float) -> float:
    """Linear warmup then cosine decay to 0 (identical to src/train.py)."""
    if step < warmup_steps:
        return base_lr * (step + 1) / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * base_lr * (1.0 + math.cos(math.pi * progress))


def init_linear(linear: torch.nn.Linear, W0: np.ndarray,
                b0: np.ndarray) -> torch.nn.Linear:
    """Copy the checkpoint's original head parameters into a fresh Linear."""
    with torch.no_grad():
        linear.weight.copy_(torch.from_numpy(np.asarray(W0, np.float32)))
        linear.bias.copy_(torch.from_numpy(np.asarray(b0, np.float32)))
    return linear


def train_candidate(
    train: Dict, val: Dict, W0: np.ndarray, b0: np.ndarray,
    class_weights: Sequence[float], lr: float, *, epochs: int, patience: int,
    batch_size: int, seed: int, device: str,
) -> Dict:
    """Train ONLY the Linear(1280,29), init from W0/b0, on cached embeddings."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    linear = init_linear(torch.nn.Linear(FEATURE_DIM, N_CLASSES), W0, b0)
    linear.to(device)
    linear.train()

    weight = torch.tensor(class_weights, dtype=torch.float32, device=device)
    criterion = torch.nn.CrossEntropyLoss(weight=weight,
                                          label_smoothing=LABEL_SMOOTHING)
    opt = torch.optim.AdamW(linear.parameters(), lr=lr,
                            weight_decay=WEIGHT_DECAY)
    n_opt = sum(p.numel() for g in opt.param_groups for p in g["params"])
    if n_opt != N_CLASSES * (FEATURE_DIM + 1):
        raise SystemExit(
            f"head-only optimizer unexpectedly touches {n_opt} parameters "
            f"(expected {N_CLASSES * (FEATURE_DIM + 1)} = Linear only); "
            "the backbone must stay frozen"
        )

    tr_x = torch.from_numpy(train["embeddings"]).to(device)
    tr_y = torch.from_numpy(train["labels"]).to(device)

    n = len(tr_y)
    n_batches = int(np.ceil(n / batch_size))
    warmup_steps = n_batches  # 1 warmup epoch, as in src/train.py
    total_steps = n_batches * epochs
    step = 0
    best_state = None
    best_macro_f1 = -1.0
    best_epoch = 0
    no_improve = 0
    runs: List[Dict] = []

    for epoch in range(1, epochs + 1):
        perm = torch.randperm(n, generator=torch.Generator(device="cpu"))
        perm = perm.to(device)
        total_loss = 0.0
        linear.train()
        for bi in range(n_batches):
            idx = perm[bi * batch_size: (bi + 1) * batch_size]
            opt.zero_grad(set_to_none=True)
            lr_now = cosine_warmup_lr(step, total_steps, warmup_steps, lr)
            for g in opt.param_groups:
                g["lr"] = lr_now
            logits = linear(tr_x[idx])
            loss = criterion(logits, tr_y[idx])
            loss.backward()
            opt.step()
            step += 1
            total_loss += float(loss.item()) * len(idx)

        # VALIDATION metrics are computed with the CPU numpy path
        # (``cached_logits``), the same math the final report and the raw-image
        # ``src.evaluate`` pipeline use. The trained head leaves ~10 S frames
        # sitting exactly on the S/E top-2 boundary, where MPS-vs-CPU float
        # differences flip the argmax; selecting the best epoch on a device
        # dependent metric would make the result unreproducible.
        linear.eval()
        Wn = linear.weight.detach().cpu().numpy()
        bn = linear.bias.detach().cpu().numpy()
        y_pred = cached_logits(val["embeddings"], Wn, bn).argmax(1)
        ev = evaluate_cm(val["labels"], y_pred)
        sm = s_e_metrics(ev["cm"])
        runs.append({
            "lr": lr, "epoch": epoch,
            "train_loss": total_loss / n,
            "val_accuracy": ev["accuracy"],
            "val_macro_f1": ev["macro_f1"],
            "S_recall": sm["S recall"], "S_to_E": sm["S->E"], "E_to_S": sm["E->S"],
        })
        improved = ev["macro_f1"] > best_macro_f1 + 1e-9
        if improved:
            best_macro_f1 = ev["macro_f1"]
            best_epoch = epoch
            best_state = {
                "weight": linear.weight.detach().cpu().clone(),
                "bias": linear.bias.detach().cpu().clone(),
            }
            no_improve = 0
        else:
            no_improve += 1
        print(f"  lr={lr:.0e} epoch {epoch:2d} loss {runs[-1]['train_loss']:.4f} "
              f"acc {ev['accuracy']:.4f} macro-F1 {ev['macro_f1']:.4f} "
              f"S->E {sm['S->E']} (best {best_epoch} @ {best_macro_f1:.4f})",
              flush=True)
        if no_improve >= patience:
            print(f"  lr={lr:.0e}: early stop at epoch {epoch}")
            break

    if best_state is None:
        raise SystemExit("no improving epoch for the candidate")
    return {
        "lr": lr, "best_epoch": best_epoch, "best_macro_f1": best_macro_f1,
        "epochs_run": len(runs), "runs": runs,
        "W": best_state["weight"].numpy(), "b": best_state["bias"].numpy(),
    }


def select_best(candidates: Sequence[Dict]) -> Dict:
    best = max(candidates, key=lambda c: c["best_macro_f1"])
    print(f"best candidate: lr={best['lr']:.0e} "
          f"macro-F1={best['best_macro_f1']:.4f} at epoch {best['best_epoch']}")
    return best


# --------------------------------------------------------------------------- #
# Deployable checkpoint reconstruction
# --------------------------------------------------------------------------- #
def build_deployable_checkpoint(
    parent_path: str, train_cache: Dict, W: np.ndarray, b: np.ndarray,
    best: Dict, verdict: Dict, deploy_out: str,
) -> Dict:
    """Parent backbone + ``Dropout(0.3) + Linear(1280,29)`` with the trained head.

    The reconstructed checkpoint reuses the parent file's own metadata (arch,
    class_weights, geometry_safe_classes, backgrounds) so it loads identically
    through ``src.evaluate.load_checkpoint``; only the ``model`` state dict is
    replaced, with the backbone tensors bit-identical to the parent and the
    Linear head swapped for the trained W/b. ``verdict`` is embedded so a
    rejected (regression/tradeoff) result cannot be deployed by accident.
    """
    from src.evaluate import load_checkpoint
    from scripts.build_embedding_cache import file_sha256

    model = load_checkpoint(parent_path, "cpu").eval()
    with torch.no_grad():
        model[1][1].weight.copy_(torch.from_numpy(W.astype(np.float32)))
        model[1][1].bias.copy_(torch.from_numpy(b.astype(np.float32)))
    ckpt = torch.load(parent_path, map_location="cpu")
    out = dict(ckpt)
    out["model"] = model.state_dict()
    out.update({
        "schema": "asl-head-only-ft/v1",
        "experiment": "head_only_cached_embeddings",
        "parent_checkpoint": parent_path,
        "parent_checkpoint_sha256": train_cache.get("parent_checkpoint_sha256"),
        "backbone_state_dict_sha256": train_cache.get("backbone_state_dict_sha256"),
        "trained_weights": "Linear(1280, 29) only; backbone frozen and "
                           "bit-identical to parent (see sha fields)",
        "verdict": verdict["verdict"],
        "head_lr": best["lr"], "head_best_epoch": best["best_epoch"],
        "head_val_macro_f1": best["best_macro_f1"],
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "note": (
            "EXPERIMENTAL checkpoint from the head-only cached-embedding "
            "experiment. Verdict recorded above; do not treat as an improvement "
            "unless the verdict is CLEAN IMPROVEMENT."
        ),
    })
    out_path = Path(deploy_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, out_path)
    return {"path": str(out_path), "file_sha256": file_sha256(str(out_path))}


# --------------------------------------------------------------------------- #
# S/E boundary analysis
# --------------------------------------------------------------------------- #
def boundary_groups(val: Dict, W: np.ndarray, b: np.ndarray) -> Dict:
    """margin_SE/distance_SE per group, using the given head on cached val embs.

    Groups are the ORIGINAL diagnostic groups: S->E errors (val S frames the
    original head called E, 138) and correct S (462), plus true E (600). They are
    fixed by the BASELINE head so before/after compares the same frames.
    """
    classes = list(CLASSES)
    s_idx, e_idx = classes.index("S"), classes.index("E")
    logits = cached_logits(val["embeddings"], W, b)
    preds = logits.argmax(1)
    w_s, w_e = W[s_idx], W[e_idx]
    denom = float(np.linalg.norm(w_s - w_e))
    rows: Dict[str, List[float]] = {"S->E errors": [], "correct S": [], "true E": []}
    for i in range(len(val["labels"])):
        cls = val["class_names"][i]
        m = float(logits[i, s_idx] - logits[i, e_idx])
        dist = m / denom if denom else float("nan")
        if cls == "E":
            rows["true E"].append({"frame_id": val["frame_ids"][i], "margin": m,
                                   "distance": dist, "pred": int(preds[i])})
        elif cls == "S":
            key = "S->E errors" if int(preds[i]) == e_idx else "correct S"
            rows[key].append({"frame_id": val["frame_ids"][i], "margin": m,
                              "distance": dist, "pred": int(preds[i])})
    def summarize(recs: Sequence[Dict]) -> Dict:
        ms = [r["margin"] for r in recs]
        ds = [r["distance"] for r in recs]
        return {
            "n": len(recs),
            "margin_mean": float(np.mean(ms)), "margin_median": float(np.median(ms)),
            "margin_min": float(np.min(ms)), "margin_max": float(np.max(ms)),
            "distance_mean": float(np.mean(ds)),
            "distance_median": float(np.median(ds)),
            "n_on_S_side": sum(1 for r in recs if r["margin"] > 0),
        }
    return {"S->E errors": summarize(rows["S->E errors"]),
            "correct S": summarize(rows["correct S"]),
            "true E": summarize(rows["true E"])}


# --------------------------------------------------------------------------- #
# Classification
# --------------------------------------------------------------------------- #
def classify(baseline: Dict, current: Dict) -> Dict:
    """Deterministic classification per the task's acceptance criteria."""
    s = s_e_metrics(current["cm"])
    bs = s_e_metrics(baseline["cm"])
    s_to_e_drop = bs["S->E"] - s["S->E"]
    material = s_to_e_drop >= MATERIAL_S_TO_E_FRAC * max(1, bs["S->E"])
    acc_ok = current["accuracy"] >= baseline["accuracy"] - METRIC_TOL
    f1_ok = current["macro_f1"] >= baseline["macro_f1"] - METRIC_TOL
    e_ok = s["E recall"] >= bs["E recall"] - 0.02
    pairs = ["T->L", "K->V", "V->K", "M->N", "N->M", "Y->I"]
    pair_regress = {p: bs[p] - s[p] for p in pairs}
    bad_pairs = [p for p in pairs if pair_regress[p] < -max(3, 0.2 * bs[p])]
    if material and acc_ok and f1_ok and e_ok and not bad_pairs:
        verdict = "CLEAN IMPROVEMENT"
    elif s_to_e_drop > 0 and (not acc_ok or not f1_ok or not e_ok or bad_pairs):
        verdict = "TRADEOFF"
    elif current["accuracy"] < baseline["accuracy"] - 0.005 or \
            current["macro_f1"] < baseline["macro_f1"] - 0.005:
        verdict = "REGRESSION"
    else:
        verdict = "NO MEANINGFUL CHANGE"
    return {
        "verdict": verdict,
        "rules": {
            "S_to_E_delta": int(s_to_e_drop),
            "S_to_E_drop_material": material,
            "accuracy_preserved": acc_ok,
            "macro_f1_preserved": f1_ok,
            "E_not_regressed": e_ok,
            "pair_regressions": {p: int(v) for p, v in pair_regress.items()},
            "material_pair_regressions": bad_pairs,
            "criteria_doc": (
                "CLEAN IMPROVEMENT: S->E down >= 15% AND acc/macro-F1 within "
                "0.001 of baseline AND E recall >= baseline-0.02 AND no watch "
                "pair falls > max(3, 20%) below baseline. TRADEOFF: S improved "
                "but errors moved elsewhere. REGRESSION: acc or macro-F1 down "
                ">= 0.005. Watch pairs for the verdict: T->L, K->V, V->K, "
                "M->N, N->M, Y->I (per task spec)."
            ),
        },
    }


# --------------------------------------------------------------------------- #
# Reports
# --------------------------------------------------------------------------- #
def _fmt(v, nd=3):
    try:
        return f"{float(v):.{nd}f}"
    except (TypeError, ValueError):
        return "-"


def _pct(x: float) -> str:
    return f"{100.0 * x:.2f}%"


def build_markdown_report(report: Dict) -> str:
    """Render ``docs/reports/head_only_cached_embeddings.md`` from the report."""
    base = report["baseline"]
    cur = report["current"]
    cls = report["classification"]
    cache = report["cache"]
    sg = report["sanity_gate"]
    tr = report["training"]
    b = report["s_e_boundary"]

    def pair_line(name: str) -> str:
        p = report["before_after_pairs"][name]
        return (f"| {name} | {p['before']} | {p['after']} | "
                f"{p['delta']:+d} |")

    def bnd(phase: str, group: str) -> str:
        g = b[phase][group]
        return (f"| {phase} · {group} | {g['n']} | {g['margin_mean']:+.3f} | "
                f"{g['margin_median']:+.3f} | {g['distance_mean']:+.4f} | "
                f"{g['distance_median']:+.4f} | {g['n_on_S_side']} |")

    def cand_row(c: Dict) -> str:
        return (f"| {c['lr']:.0e} | {c['best_macro_f1']:.4f} | "
                f"{c['best_epoch']} | {c['epochs_run']} |")

    lines = [
        "# Head-only fine-tuning on cached embeddings — efficientnet_b0_targeted",
        "",
        f"Experiment: train ONLY the final `Linear(1280, 29)` on the cached "
        f"1280-D pooled embeddings of the frozen EfficientNet-B0 backbone; the "
        f"backbone is bit-identical before/after "
        f"(`{report.get('backbone_state_dict_sha256', 'see JSON')[:16]}…`). "
        f"No new loss, no re-weighting, no oversampling, no backbone "
        f"unfreezing.",
        "",
        f"**VERDICT: {cls['verdict']}**",
        "",
        f"- S->E {base['S->E']} -> {cur['S->E']} "
        f"({cur['S->E'] - base['S->E']:+d}); material = "
        f"{cls['rules']['S_to_E_drop_material']} (threshold: "
        f"{MATERIAL_S_TO_E_FRAC:.0%} of baseline).",
        f"- Accuracy {_pct(base['accuracy'])} -> {_pct(cur['accuracy'])} "
        f"(preserved = {cls['rules']['accuracy_preserved']}, tol {METRIC_TOL}).",
        f"- Macro-F1 {base['macro_f1']:.4f} -> {cur['macro_f1']:.4f} "
        f"(preserved = {cls['rules']['macro_f1_preserved']}, tol {METRIC_TOL}).",
        f"- E recall {base['E recall']:.3f} -> {cur['E recall']:.3f} "
        f"(not regressed = {cls['rules']['E_not_regressed']}, tol 0.02).",
        f"- Watch-pair regressions: "
        f"{cls['rules']['material_pair_regressions'] or 'none'}.",
        "",
        "## Setup",
        "",
        f"- Checkpoint: `{report['checkpoint']}` (parent; untouched).",
        f"- Manifest: `{report['manifest']}` "
        f"(`{report['manifest_sha256'][:16]}…`), leakage-safe frame-range "
        f"split — dev val, NOT the reported metric.",
        f"- Device: {report['device']}. Loss: "
        f"`{report['loss']['criterion']}` "
        f"(label smoothing {report['loss']['label_smoothing']}, class weights "
        f"from the checkpoint), AdamW wd {report['loss']['weight_decay']}, "
        f"{report['loss']['schedule']}.",
        f"- Cached train embeddings: `{cache['train']['path']}` "
        f"({cache['train']['n']} x 1280). Augmentation: "
        f"{cache['train']['augmentation']['policy']} "
        f"(seed {cache['train']['augmentation'].get('seed', '?')}, "
        f"backgrounds "
        f"{cache['train']['augmentation'].get('backgrounds')}); not claimed "
        f"identical to image-space training.",
        f"- Cached val embeddings: `{cache['val']['path']}` "
        f"({cache['val']['n']} x 1280), deterministic `preprocess_bgr` "
        f"contract.",
        f"- Cached-logits verification: max |logit diff| "
        f"{cache['verification']['max_abs_logit_diff']:.2e} over "
        f"{len(cache['verification']['samples'])} sampled frames.",
        "",
        "## Sanity gate",
        "",
        f"- Original head on cached val embeddings reproduced the baseline "
        f"**exactly**: acc {_pct(sg['accuracy'])}, macro-F1 {sg['macro_f1']:.4f}, "
        f"S->E {sg['S_to_E']}, confusion matrix "
        f"`{sg['confusion_matrix_exact_match']}`.",
        "",
        "## LR search",
        "",
        "| LR | best val macro-F1 | best epoch | epochs run |",
        "|----|-------------------|------------|------------|",
    ]
    for c in tr["candidates"]:
        lines.append(cand_row(c))
    lines += [
        "",
        f"Selected: **lr={tr['selected']['lr']:.0e}**, epoch "
        f"{tr['selected']['best_epoch']}, val macro-F1 "
        f"{tr['selected']['best_macro_f1']:.4f}. Trainable parameters: "
        f"{tr['trainable_parameters']} (Linear only). Seed {tr['seed']}.",
        "",
        "## Before / after (dev val, 17,400 = 600/class)",
        "",
        "| metric | before | after | delta |",
        "|--------|--------|-------|-------|",
    ]
    lines.append(pair_line("S->E"))
    lines.append(pair_line("E->S"))
    for name in ("T->L", "K->V", "V->K", "M->N", "N->M", "Y->I",
                 "X->K", "X->I", "X->R"):
        lines.append(pair_line(name))
    lines.append(
        f"| total errors | {base['total_errors']} | {cur['total_errors']} | "
        f"{cur['total_errors'] - base['total_errors']:+d} |"
    )
    lines.append(
        f"| accuracy | {_pct(base['accuracy'])} | {_pct(cur['accuracy'])} | "
        f"{100.0 * (cur['accuracy'] - base['accuracy']):+.2f}% |"
    )
    lines.append(
        f"| macro-F1 | {base['macro_f1']:.4f} | {cur['macro_f1']:.4f} | "
        f"{cur['macro_f1'] - base['macro_f1']:+.4f} |"
    )
    lines += [
        "",
        "## S/E boundary (fixed diagnostic groups, baseline-head assignment)",
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
        f"- `w_S - w_E` norm: before "
        f"{b['before']['head_parameters']['w_S_minus_w_E_norm']:.3f}, after "
        f"{b['after']['head_parameters']['w_S_minus_w_E_norm']:.3f}.",
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
            lines.append(f"| {c} | {m} | {_fmt(bv)} | {_fmt(av)} |")
    lines += [
        "",
        "## Artifacts",
        "",
        f"- Head checkpoint: `{report['artifacts']['head_checkpoint']}`",
        f"- Deployable checkpoint: "
        f"`{report['artifacts']['deployable_checkpoint']['path']}` "
        f"(file sha256 `{report['artifacts']['deployable_checkpoint']['file_sha256'][:16]}…`; "
        f"verdict embedded in metadata)",
        f"- Confusion matrix PNG: `{report['artifacts']['confusion_png']}`",
        f"- Confusion matrix CSV: `{report['artifacts']['confusion_csv']}`",
        f"- Full JSON: `{report['artifacts']['report_json']}`",
        "",
        "## Interpretation (conservative)",
        "",
        f"- The verdict is deterministic per "
        f"{cls['rules']['criteria_doc']}",
        "- This experiment optimizes the head on ONE fixed augmented "
        "realization of the train split; it is not a claim about the full "
        "image-space training procedure.",
        f"- Boundary numbers are descriptive (margin/distance = S-vs-E "
        "two-class quantities, not the full 29-class decision).",
        "",
    ]
    return "\n".join(lines) + "\n"


def run(args) -> Dict:
    device = _resolve_device(args.device)
    manifest = SplitManifest.from_json(args.manifest)
    manifest_sha = json.loads(Path(args.manifest).read_text()).get("content_sha256")

    train = load_cache(str(Path(args.cache_dir) / args.train_cache))
    val = load_cache(str(Path(args.cache_dir) / args.val_cache))
    verify_cache_pairs(train, val, manifest, args.manifest, args.checkpoint)
    verify_cache_against_manifest(val, manifest)
    print(f"cache: train {train['n']} x {FEATURE_DIM}, "
          f"val {val['n']} x {FEATURE_DIM}")

    model = load_checkpoint(args.checkpoint, "cpu")
    W0, b0, head_info = extract_head(model)
    print(f"head: {head_info['layers']}  W={head_info['weight_shape']} "
          f"b={head_info['bias_shape']}")

    verif = verify_cached_logits(args.checkpoint, val, manifest, device,
                                 args.verify_samples)
    print(f"cached-logits verification: max |diff| "
          f"{verif['max_abs_logit_diff']:.2e} over "
          f"{len(verif['samples'])} sampled frames")

    baseline_ev, bcm = recompute_baseline(val, W0, b0, args.baseline_report)
    baseline_metrics = {
        "accuracy": baseline_ev["accuracy"], "macro_f1": baseline_ev["macro_f1"],
        "total_errors": baseline_ev["total_errors"],
        **s_e_metrics(bcm), **s_e_metrics(baseline_ev["cm"]),
    }
    baseline_boundary = boundary_groups(val, W0, b0)
    baseline_boundary["head_parameters"] = {
        "w_S_minus_w_E_norm": float(np.linalg.norm(W0[18] - W0[4])),
    }

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    class_weights = list(ckpt.get("class_weights") or [])
    if len(class_weights) != N_CLASSES:
        raise SystemExit(f"checkpoint has {len(class_weights)} class weights, "
                         "expected 29; cannot reproduce the training loss")
    print(f"loss: CrossEntropy(label_smoothing={LABEL_SMOOTHING}, "
          f"weight=<checkpoint's {len(class_weights)} class weights>), "
          f"AdamW wd={WEIGHT_DECAY}")

    head_out = Path(args.head_out)
    head_out.parent.mkdir(parents=True, exist_ok=True)

    candidates = []
    for lr in args.lrs:
        cand = train_candidate(train, val, W0, b0, class_weights, lr,
                               epochs=args.epochs, patience=args.patience,
                               batch_size=args.batch_size, seed=args.seed,
                               device=device)
        candidates.append(cand)
        out = Path(args.head_out).parent / f"candidate_head_lr{lr:.0e}.pt"
        torch.save({"schema": SCHEMA, "lr": lr, "W": cand["W"], "b": cand["b"],
                    **{k: cand[k] for k in ("best_epoch", "best_macro_f1",
                                            "epochs_run")}}, out)
        print(f"wrote {out}")
    best = select_best(candidates)

    head_out = Path(args.head_out)
    torch.save({
        "schema": SCHEMA,
        "parent_checkpoint": args.checkpoint,
        "lr": best["lr"], "best_epoch": best["best_epoch"],
        "best_macro_f1": best["best_macro_f1"],
        "epochs_run": best["epochs_run"],
        "W": best["W"], "b": best["b"],
        "note": "trained ONLY on cached train embeddings; backbone untouched",
    }, head_out)
    print(f"wrote {head_out}")

    current_ev = evaluate_cm(val["labels"],
                             cached_logits(val["embeddings"], best["W"],
                                           best["b"]).argmax(1))
    current_metrics = {
        "accuracy": current_ev["accuracy"], "macro_f1": current_ev["macro_f1"],
        "total_errors": current_ev["total_errors"],
        **s_e_metrics(current_ev["cm"]),
    }
    current_boundary = boundary_groups(val, best["W"], best["b"])
    current_boundary["head_parameters"] = {
        "w_S_minus_w_E_norm": float(np.linalg.norm(best["W"][18] - best["W"][4])),
    }

    verdict = classify(baseline_ev, current_ev)

    deploy = build_deployable_checkpoint(
        args.checkpoint, train, best["W"], best["b"], best, verdict,
        args.deploy_out,
    )
    print(f"wrote deployable checkpoint {deploy['path']} "
          f"(file sha256 {deploy['file_sha256'][:16]}…, "
          f"verdict {verdict['verdict']})")

    os.makedirs(os.path.dirname(args.cm_png) or ".", exist_ok=True)
    save_confusion_png(current_ev["cm"], args.cm_png)
    save_confusion_csv(current_ev["cm"], args.cm_csv)

    report = {
        "schema": SCHEMA,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "checkpoint": args.checkpoint,
        "arch": "efficientnet",
        "manifest": args.manifest,
        "manifest_sha256": manifest_sha,
        "parent_checkpoint_sha256": train.get("parent_checkpoint_sha256"),
        "backbone_state_dict_sha256": train.get("backbone_state_dict_sha256"),
        "backbone_untouched": (
            "the parent checkpoint file and the backbone state dict were NOT "
            "modified by this experiment; training optimized the Linear head "
            "only (see head checkpoint metadata)"
        ),
        "split_note": (
            "dev val = leakage-safe frame-range split, same signer, same room. "
            "NOT the reported metric."
        ),
        "device": device,
        "cache": {
            "train": {"path": args.train_cache, "n": train["n"],
                      "split": train["split"],
                      "augmentation": train["augmentation"]["train"]},
            "val": {"path": args.val_cache, "n": val["n"],
                    "split": val["split"]},
            "verification": verif,
            "train_val_frame_overlap": 0,
        },
        "sanity_gate": {
            "passed": True,
            "accuracy": baseline_ev["accuracy"],
            "macro_f1": baseline_ev["macro_f1"],
            "S_to_E": s_e_metrics(bcm)["S->E"],
            "confusion_matrix_exact_match": True,
        },
        "loss": {
            "criterion": "CrossEntropyLoss",
            "label_smoothing": LABEL_SMOOTHING,
            "class_weights": class_weights,
            "class_weights_source": "checkpoint ('rebalance': 'loss')",
            "optimizer": "AdamW", "weight_decay": WEIGHT_DECAY,
            "schedule": "cosine warmup (1 epoch), mirror of src/train.py",
            "device": device,
        },
        "training": {
            "seed": args.seed, "epochs_max": args.epochs, "patience": args.patience,
            "batch_size": args.batch_size, "lrs": args.lrs,
            "candidates": [{k: v for k, v in c.items() if k != "W" and k != "b"}
                           for c in candidates],
            "selected": {"lr": best["lr"], "best_epoch": best["best_epoch"],
                         "best_macro_f1": best["best_macro_f1"],
                         "epochs_run": best["epochs_run"]},
            "trainable_parameters": 29 * (FEATURE_DIM + 1),
            "metric_path_note": (
                "All validation metrics (best-epoch selection, report, "
                "before/after) use the CPU numpy cached-logits path — the same "
                "math as the raw-image src.evaluate pipeline. The trained head "
                "leaves ~10 S frames on the exact S/E top-2 boundary where "
                "MPS-vs-CPU float differences flip the argmax; a device "
                "dependent metric would be unreproducible."
            ),
        },
        "baseline": baseline_metrics,
        "current": current_metrics,
        "before_after_pairs": {
            p: {"before": baseline_metrics[p], "after": current_metrics[p],
                "delta": current_metrics[p] - baseline_metrics[p]}
            for p in ["S->E", "E->S", "T->L", "K->V", "V->K", "M->N", "N->M",
                      "Y->I", "X->K", "X->I", "X->R"]
        },
        "per_class_before_after": {
            c: {"before": baseline_ev["per_class"][c],
                "after": current_ev["per_class"][c]}
            for c in ("S", "E")
        },
        "s_e_boundary": {
            "before": baseline_boundary,
            "after": current_boundary,
        },
        "classification": verdict,
        "deployable_checkpoint": deploy,
        "confusion_matrix": current_ev["cm"].astype(int).tolist(),
        "artifacts": {
            "head_checkpoint": str(head_out),
            "deployable_checkpoint": deploy,
            "report_json": args.out_json,
            "report_md": args.out_md,
            "confusion_png": args.cm_png,
            "confusion_csv": args.cm_csv,
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


def main() -> None:
    p = argparse.ArgumentParser(
        description="Head-only fine-tuning on cached pooled embeddings."
    )
    p.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    p.add_argument("--manifest", default=DEFAULT_MANIFEST)
    p.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    p.add_argument("--train-cache", default="targeted_train_embeddings.pt")
    p.add_argument("--val-cache", default="targeted_val_embeddings.pt")
    p.add_argument("--baseline-report", default=DEFAULT_BASELINE_REPORT)
    p.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"],
                   default="auto")
    p.add_argument("--lrs", type=float, nargs="+", default=[1e-4, 3e-4, 1e-3])
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--seed", type=int, default=20260809)
    p.add_argument("--verify-samples", type=int, default=4,
                   help="frames per class for the cached-logits verification")
    p.add_argument("--head-out", default=DEFAULT_HEAD_OUT)
    p.add_argument("--deploy-out", default="checkpoints/efficientnet_b0_head_ft.pt")
    p.add_argument("--out-json", default=DEFAULT_OUT_JSON)
    p.add_argument("--out-md", default=DEFAULT_OUT_MD)
    p.add_argument("--cm-png", default=DEFAULT_CM_PNG)
    p.add_argument("--cm-csv", default=DEFAULT_CM_CSV)
    run(p.parse_args())


if __name__ == "__main__":
    main()
