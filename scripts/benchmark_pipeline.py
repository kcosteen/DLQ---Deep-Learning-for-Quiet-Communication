"""Phase 5: headless per-stage latency benchmark for the webcam bridge (#17).

Measures, on a representative sample of training frames (so it runs without a
webcam and is reproducible), the three pipeline stages the live app tracks:

    MediaPipe localization   -> src.crop.HandCropper.tight_bbox
    crop + preprocess        -> crop_square_with_margin + preprocess_bgr
    EfficientNet forward     -> targeted checkpoint probs

Each stage is timed with ``time.perf_counter``; means/medians are reported in
ms plus the implied pipeline FPS. Cropping does not shrink the network's work —
its input is always 224x224 — so the total is exactly the sum of the three
stages (plus a few tens of microseconds of overhead).

Usage::

    python scripts/benchmark_pipeline.py \\
        --checkpoint checkpoints/efficientnet_b0_targeted.pt \\
        --root data/raw/asl_alphabet_train --frames 100 --crop-margin 0.62
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path
from typing import Dict, List

import cv2

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.crop import HandCropper, crop_square_with_margin, preprocess_bgr  # noqa: E402
from src.data import CLASSES, list_class_files  # noqa: E402


def _sample_frames(root: str, frames: int) -> List[str]:
    """``frames`` paths spread across classes (deterministic stride per class)."""
    out = []
    for cls in CLASSES:
        files = list_class_files(root, cls)
        if not files:
            continue
        take = max(1, frames // len(CLASSES))
        stride = len(files) / take
        out += [files[int(i * stride)] for i in range(min(take, len(files)))]
    return out[:frames]


def _stats(vals: List[float]) -> Dict[str, float]:
    if not vals:
        return {"n": 0, "mean_ms": 0.0, "median_ms": 0.0}
    ms = [v * 1000.0 for v in vals]
    return {"n": len(ms), "mean_ms": round(sum(ms) / len(ms), 2),
            "median_ms": round(statistics.median(ms), 2),
            "p95_ms": round(sorted(ms)[int(0.95 * (len(ms) - 1))], 2)}


def run(checkpoint: str, root: str, frames: int, crop_margin: float,
        device: str) -> None:
    import torch

    from app.webcam_speller import Classifier

    clf = Classifier(checkpoint, device=device)
    paths = _sample_frames(root, frames)

    stages: Dict[str, List[float]] = {
        "localization": [], "crop_preprocess": [], "inference": [], "total": [],
    }
    n_hand = 0
    n_no_hand = 0
    with HandCropper() as cropper:
        for p in paths:
            img = cv2.imread(p, cv2.IMREAD_COLOR)
            if img is None:
                continue
            t0 = time.perf_counter()
            tb = cropper.tight_bbox(img)
            t1 = time.perf_counter()
            stages["localization"].append(t1 - t0)
            if tb is None:
                n_no_hand += 1
                stages["crop_preprocess"].append(0.0)
                stages["inference"].append(0.0)
                stages["total"].append(t1 - t0)
                continue
            n_hand += 1
            crop = crop_square_with_margin(img, tb, crop_margin)
            x = clf.tensor_from_bgr(crop)   # resize (no-op on 224) + normalize
            t2 = time.perf_counter()
            stages["crop_preprocess"].append(t2 - t1)
            with torch.no_grad():
                clf.model(x)
            t3 = time.perf_counter()
            stages["inference"].append(t3 - t2)
            stages["total"].append(t3 - t0)

    print(f"benchmark: {len(paths)} frames, crop-margin {crop_margin}, "
          f"hand detected on {n_hand}, no hand on {n_no_hand}, device {device}")
    print(f"{'stage':>18} {'n':>5} {'mean ms':>9} {'median ms':>11} {'p95 ms':>8}")
    for name in ("localization", "crop_preprocess", "inference", "total"):
        s = _stats(stages[name])
        print(f"{name:>18} {s['n']:>5} {s['mean_ms']:>9.2f} {s['median_ms']:>11.2f} "
              f"{s['p95_ms']:>8.2f}")
    total = _stats(stages["total"])
    if total["n"] and total["mean_ms"]:
        print(f"\nimplied pipeline FPS (1 / mean total): "
              f"{1000.0 / total['mean_ms']:.1f}")
        print(f"implied pipeline FPS (1 / median total): "
              f"{1000.0 / total['median_ms']:.1f}")
        # FPS over hand frames only is the honest live number.
        hand_total = [t for t in stages["total"] if t > 0]
        ht = _stats(hand_total)
        if ht["n"]:
            print(f"hand-frame FPS (1 / mean total over {ht['n']} hand frames): "
                  f"{1000.0 / ht['mean_ms']:.1f}")


def main() -> None:
    p = argparse.ArgumentParser(description="Headless per-stage latency benchmark.")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--root", default="data/raw/asl_alphabet_train")
    p.add_argument("--frames", type=int, default=100)
    p.add_argument("--crop-margin", type=float, default=0.62)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()
    run(args.checkpoint, args.root, args.frames, args.crop_margin, args.device)


if __name__ == "__main__":
    main()
