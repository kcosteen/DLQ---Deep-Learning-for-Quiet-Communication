"""Evaluation harness: metrics, confusion matrix, and the webcam-set eval.

Two evaluation modes:

1. ``--split``  : accuracy on the leakage-safe frame-range val split. Useful for
   development, but NOT the number we report.
2. ``--webcam`` : accuracy on ``data/webcam_testset`` — folder-per-class crops the
   team recorded. THIS is the real, reported benchmark.

Emits: overall accuracy, macro-F1, per-class accuracy, and a confusion matrix
(PNG + CSV). Flags the classic confusions to watch: M/N/S/T, A/E, K/V.

Targets: leakage-safe val > 95%; webcam test >= 90%; no class below 85%.
"""

from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import CLASSES, ASLImageDataset, Sample, SplitManifest, frame_range_split
from .model import build_model

WATCH_CONFUSIONS = [("M", "N", "S", "T"), ("A", "E"), ("K", "V")]


def load_checkpoint(path: str, device: str) -> torch.nn.Module:
    model = build_model("efficientnet", pretrained=False).to(device)
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
    model.eval()
    return model


def _webcam_samples(root: str) -> List[Sample]:
    from .data import CLASS_TO_IDX, list_class_files

    samples: List[Sample] = []
    for c in CLASSES:
        for f in list_class_files(root, c):
            samples.append(Sample(f, CLASS_TO_IDX[c], c))
    return samples


@torch.no_grad()
def predict_all(model, loader, device) -> Tuple[np.ndarray, np.ndarray]:
    ys, ps = [], []
    for x, y in loader:
        x = x.to(device)
        ps.append(model(x).argmax(1).cpu().numpy())
        ys.append(y.numpy())
    return np.concatenate(ys), np.concatenate(ps)


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n: int) -> np.ndarray:
    cm = np.zeros((n, n), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def per_class_accuracy(cm: np.ndarray) -> Dict[str, float]:
    accs = {}
    for i, c in enumerate(CLASSES):
        total = cm[i].sum()
        accs[c] = float(cm[i, i] / total) if total else float("nan")
    return accs


def macro_f1(cm: np.ndarray) -> float:
    f1s = []
    for i in range(len(CLASSES)):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        f1s.append(f1)
    return float(np.mean(f1s))


def save_confusion_png(cm: np.ndarray, path: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"skip PNG (matplotlib unavailable: {e})")
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig, ax = plt.subplots(figsize=(11, 10))
    im = ax.imshow(cm, cmap="viridis")
    ax.set_xticks(range(len(CLASSES)))
    ax.set_yticks(range(len(CLASSES)))
    ax.set_xticklabels(CLASSES, rotation=90, fontsize=7)
    ax.set_yticklabels(CLASSES, fontsize=7)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title("ASL fingerspelling confusion matrix")
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"wrote {path}")


def save_confusion_csv(cm: np.ndarray, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([""] + list(CLASSES))
        for i, c in enumerate(CLASSES):
            w.writerow([c] + cm[i].tolist())
    print(f"wrote {path}")


def report(cm: np.ndarray, target_per_class: float = 0.85) -> None:
    overall = cm.trace() / max(1, cm.sum())
    print(f"\noverall accuracy: {overall:.4f}")
    print(f"macro-F1:         {macro_f1(cm):.4f}")
    accs = per_class_accuracy(cm)
    print("\nper-class accuracy:")
    below = []
    for c in CLASSES:
        flag = "" if (np.isnan(accs[c]) or accs[c] >= target_per_class) else "  <-- BELOW 85%"
        if flag:
            below.append(c)
        print(f"  {c:>8}: {accs[c]:.3f}{flag}")
    print("\nwatch these known-confusable groups:")
    for group in WATCH_CONFUSIONS:
        idxs = [CLASSES.index(g) for g in group if g in CLASSES]
        for i in idxs:
            confused = {CLASSES[j]: int(cm[i, j]) for j in idxs if j != i and cm[i, j]}
            if confused:
                print(f"  {CLASSES[i]} -> {confused}")
    if below:
        print(f"\nFAIL: classes below 85%: {below}")
    else:
        print("\nall classes meet the 85% per-class bar.")


def run(args: argparse.Namespace) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_checkpoint(args.checkpoint, device)

    if args.webcam:
        samples = _webcam_samples(args.webcam)
        if not samples:
            raise SystemExit(f"no images under {args.webcam}")
        ds = ASLImageDataset(samples, train=False)
        print(f"WEBCAM TEST SET (the reported benchmark): {len(ds)} images")
    else:
        if args.manifest and os.path.exists(args.manifest):
            manifest = SplitManifest.from_json(args.manifest)
        else:
            manifest = frame_range_split(args.split, args.train_frac)
        ds = ASLImageDataset(manifest.val, train=False)
        print(f"LEAKAGE-SAFE VAL SPLIT (dev only, not reported): {len(ds)} images")

    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers)
    y_true, y_pred = predict_all(model, loader, device)
    cm = confusion_matrix(y_true, y_pred, len(CLASSES))
    report(cm)
    if args.figures:
        save_confusion_png(cm, os.path.join(args.figures, "confusion_matrix.png"))
        save_confusion_csv(cm, os.path.join(args.figures, "confusion_matrix.csv"))


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate a checkpoint.")
    p.add_argument("--checkpoint", required=True)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--split", help="asl_alphabet_train/ dir -> dev val split")
    g.add_argument("--webcam", help="data/webcam_testset dir -> reported benchmark")
    p.add_argument("--manifest", default=None)
    p.add_argument("--train-frac", type=float, default=0.8)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--figures", default="docs/figures",
                   help="dir for confusion PNG/CSV (omit to skip)")
    run(p.parse_args())


if __name__ == "__main__":
    main()
