"""Step 6: Framing reconstruction error — compare hand centre/scale distributions.

For val images where MediaPipe detects a hand, compare the hand framing in the
final 224x224 letterboxed crop across four methods:

  original     : raw training image -> preprocess_bgr
  symmetric    : crop_square_with_margin(0.62) -> preprocess_bgr
  forearm      : crop_square_with_forearm_context -> preprocess_bgr
  calibrated   : crop_square_with_training_framing(TRAIN_L/R/T/B) -> preprocess_bgr

For each method, run MediaPipe on the *letterboxed 224x224 crop* to measure:
  hand centre x (frac of 224)
  hand centre y (frac of 224)
  hand width  / 224
  hand height / 224

Compare distributions against the original training images to determine which
crop method best recreates the training input distribution.

Usage::

    python scripts/measure_framing_reconstruction.py \\
        --checkpoint checkpoints/efficientnet_b0_targeted.pt --device mps
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.crop import (  # noqa: E402
    DEFAULT_ASYMMETRIC_MARGINS,
    HandCropper,
    TrainingFramingMargins,
    asymmetric_context_bbox,
    crop_square_with_forearm_context,
    crop_square_with_margin,
    crop_square_with_training_framing,
    preprocess_bgr,
    resize_square,
    training_calibrated_context_bbox,
)
from src.data import CLASSES, SplitManifest  # noqa: E402


# --------------------------------------------------------------------------- #
# Measure hand framing in a 224x224 letterboxed crop
# --------------------------------------------------------------------------- #

def measure_crop_framing(
    crop_224: np.ndarray,
    cropper: HandCropper,
) -> Optional[Dict[str, float]]:
    """Run MediaPipe on a 224x224 crop and return hand centre/scale metrics."""
    tb = cropper.tight_bbox(crop_224)
    if tb is None:
        return None
    x0, y0, x1, y1 = tb
    bw, bh = x1 - x0, y1 - y0
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    return {
        "hand_cx": round(cx / 224.0, 4),
        "hand_cy": round(cy / 224.0, 4),
        "hand_w_frac": round(bw / 224.0, 4),
        "hand_h_frac": round(bh / 224.0, 4),
    }


# --------------------------------------------------------------------------- #
# Aggregate
# --------------------------------------------------------------------------- #

def _agg(values: List[float]) -> Dict:
    if not values:
        return {"n": 0, "mean": None, "median": None,
                "p10": None, "p25": None, "p75": None, "p90": None}
    a = np.array(values)
    return {
        "n": len(values),
        "mean": round(float(a.mean()), 4),
        "median": round(float(np.median(a)), 4),
        "p10": round(float(np.percentile(a, 10)), 4),
        "p25": round(float(np.percentile(a, 25)), 4),
        "p75": round(float(np.percentile(a, 75)), 4),
        "p90": round(float(np.percentile(a, 90)), 4),
    }


def aggregate_framing(records: List[Dict]) -> Dict:
    metrics = ["hand_cx", "hand_cy", "hand_w_frac", "hand_h_frac"]
    by_method: Dict[str, Dict[str, List[float]]] = {
        m: {k: [] for k in metrics} for m in
        ("original", "symmetric", "forearm", "calibrated")
    }
    for rec in records:
        for method in by_method:
            fm = rec.get(f"framing_{method}")
            if fm is not None:
                for k in metrics:
                    by_method[method][k].append(fm[k])
    return {
        method: {k: _agg(v) for k, v in metrics_dict.items()}
        for method, metrics_dict in by_method.items()
    }


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def run(args: argparse.Namespace) -> None:
    # ---- margins ---- #
    if args.calibrated_json:
        d = json.loads(Path(args.calibrated_json).read_text())
        margins = TrainingFramingMargins(
            left=d["TRAIN_L"], right=d["TRAIN_R"],
            top=d["TRAIN_T"], bottom=d["TRAIN_B"],
        )
        print(f"calibrated margins: L={margins.left:.4f} R={margins.right:.4f} "
              f"T={margins.top:.4f} B={margins.bottom:.4f}")
    else:
        margins = TrainingFramingMargins(left=0.5, right=0.5, top=0.5, bottom=0.5)

    # ---- load data ---- #
    manifest = SplitManifest.from_json(args.manifest)
    samples = manifest.val
    if args.limit:
        keep: List = []
        per: Counter = Counter()
        for s in samples:
            if per[s.class_name] >= args.limit:
                continue
            per[s.class_name] += 1
            keep.append(s)
        samples = keep
    print(f"val split: {len(samples)} images"
          + (f" (limited to {args.limit}/class)" if args.limit else ""))

    det_cache: Dict[str, Optional[List[int]]] = {}
    if args.detection_cache and Path(args.detection_cache).is_file():
        det_cache = json.loads(Path(args.detection_cache).read_text())
        kept = sum(1 for v in det_cache.values() if v is not None)
        print(f"detection cache: {len(det_cache)} entries ({kept} with hand)")

    detected = [(s, tuple(det_cache[s.path]))
                for s in samples if det_cache.get(s.path) is not None]
    print(f"detected: {len(detected)}/{len(samples)}")

    # ---- process each detected image ---- #
    records: List[Dict] = []
    t0 = time.time()
    with HandCropper(min_detection_confidence=0.5) as crop_mp:
        for i, (s, tb) in enumerate(detected):
            img = cv2.imread(s.path, cv2.IMREAD_COLOR)
            if img is None:
                continue

            rec: Dict = {
                "path": s.path,
                "class": s.class_name,
                "bbox": list(tb),
            }

            # original: raw val image letterboxed to 224
            orig_224 = resize_square(img)
            rec["framing_original"] = measure_crop_framing(orig_224, crop_mp)

            # symmetric
            sym_crop = crop_square_with_margin(img, tb, args.crop_margin)
            rec["framing_symmetric"] = measure_crop_framing(sym_crop, crop_mp)

            # forearm
            fore_crop = crop_square_with_forearm_context(
                img, tb, DEFAULT_ASYMMETRIC_MARGINS)
            rec["framing_forearm"] = measure_crop_framing(fore_crop, crop_mp)

            # calibrated
            cal_crop = crop_square_with_training_framing(img, tb, margins)
            rec["framing_calibrated"] = measure_crop_framing(cal_crop, crop_mp)

            records.append(rec)
            if (i + 1) % 500 == 0:
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed
                eta = (len(detected) - i - 1) / rate
                print(f"  {i + 1}/{len(detected)} "
                      f"({rate:.1f} img/s, ETA {eta / 60:.1f} min)")

    print(f"processed {len(records)} images ({time.time() - t0:.1f}s)")

    # ---- aggregate ---- #
    summary = aggregate_framing(records)
    summary["n_images"] = len(records)

    print("\n=== FRAMING RECONSTRUCTION (224x224 crop, MediaPipe-measured) ===")
    print(f"{'method':>14} {'n':>5}  {'mean_cx':>8} {'med_cx':>8}  "
          f"{'mean_cy':>8} {'med_cy':>8}  {'mean_w':>8} {'med_w':>8}  "
          f"{'mean_h':>8} {'med_h':>8}")
    for method in ("original", "symmetric", "forearm", "calibrated"):
        m = summary[method]
        cx, cy = m["hand_cx"], m["hand_cy"]
        hw, hh = m["hand_w_frac"], m["hand_h_frac"]
        print(f"{method:>14} {cx['n']:>5}  "
              f"{cx['mean']:>8.4f} {cx['median']:>8.4f}  "
              f"{cy['mean']:>8.4f} {cy['median']:>8.4f}  "
              f"{hw['mean']:>8.4f} {hw['median']:>8.4f}  "
              f"{hh['mean']:>8.4f} {hh['median']:>8.4f}")

    # ---- save ---- #
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "summary": summary,
        "n_images": len(records),
        "records": records[:200],  # cap for portability
    }
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"\nwrote {out}")


def main() -> None:
    p = argparse.ArgumentParser(
        description="Framing reconstruction error across crop methods.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--manifest", default="data/split_manifest.json")
    p.add_argument("--crop-margin", type=float, default=0.62)
    p.add_argument("--limit", type=int, default=0,
                   help="max per class (0 = full)")
    p.add_argument("--detection-cache",
                   default="data/forearm_crop_detection_records.json")
    p.add_argument("--calibrated-json",
                   default="data/training_framing_derived.json")
    p.add_argument("--out",
                   default="docs/reports/framing_reconstruction_error.json")
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
