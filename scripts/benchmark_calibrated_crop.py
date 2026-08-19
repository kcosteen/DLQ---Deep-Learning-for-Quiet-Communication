"""Val-split benchmark: original vs symmetric vs forearm vs training-calibrated.

For every val image where MediaPipe detects a hand, score four paths:

  original     : raw val image -> preprocess_bgr -> model
  symmetric    : crop_square_with_margin(margin=0.62) -> preprocess_bgr -> model
  forearm      : crop_square_with_forearm_context -> preprocess_bgr -> model
  calibrated   : crop_square_with_training_framing(TRAIN_L/R/T/B) -> preprocess_bgr -> model

Uses the cached detection bboxes from the forearm-crop benchmark to avoid
re-running MediaPipe. Reports overall accuracy, macro-F1, per-class recall
for the focus classes (S/E/M/N/K/V/X), and the full confusion matrices.

Usage::

    python scripts/benchmark_calibrated_crop.py \\
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
    TrainingFramingMargins,
    asymmetric_context_bbox,
    crop_square_with_forearm_context,
    crop_square_with_margin,
    crop_square_with_training_framing,
    preprocess_bgr,
    training_calibrated_context_bbox,
)
from src.data import CLASSES, SplitManifest  # noqa: E402
from src.evaluate import (  # noqa: E402
    confusion_matrix,
    load_checkpoint,
    macro_f1,
    per_class_accuracy,
    save_confusion_csv,
    save_confusion_png,
)

FOCUS_CLASSES = ("S", "E", "M", "N", "K", "V", "X")
REPORT_SCHEMA = "asl-calibrated-crop-benchmark/v1"


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _to_tensor(img_bgr: np.ndarray) -> np.ndarray:
    return preprocess_bgr(img_bgr)


def batch_predict(model, device, chunk_tensors, batch_size: int) -> np.ndarray:
    import torch
    preds: List[int] = []
    model.eval()
    n = len(chunk_tensors)
    last_pct = -1
    with torch.no_grad():
        for i in range(0, n, batch_size):
            x = torch.from_numpy(chunk_tensors[i : i + batch_size]).to(device)
            preds.append(model(x).argmax(1).cpu().numpy())
            pct = 100 * (i + batch_size) // n
            if pct != last_pct and pct % 10 == 0:
                last_pct = pct
                print(f"  forward {min(i + batch_size, n)}/{n} ({pct}%)", flush=True)
    return np.concatenate(preds)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def run(args: argparse.Namespace) -> None:
    import torch

    if args.device == "auto":
        if torch.cuda.is_available():
            args.device = "cuda"
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            args.device = "mps"
        else:
            args.device = "cpu"
    print(f"device: {args.device}")

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
        print(f"using default calibrated margins (L/R/T/B = 0.5)")

    # ---- load manifest + detection cache ---- #
    manifest = SplitManifest.from_json(args.manifest)
    samples = manifest.val
    if args.limit_per_class:
        keep: List = []
        per: Counter = Counter()
        for s in samples:
            if per[s.class_name] >= args.limit_per_class:
                continue
            per[s.class_name] += 1
            keep.append(s)
        samples = keep
    print(f"val split: {len(samples)} images"
          + (f" (limited to {args.limit_per_class}/class)"
             if args.limit_per_class else ""))

    paths = [s.path for s in samples]
    det_cache: Dict[str, Optional[List[int]]] = {}
    if args.detection_cache and Path(args.detection_cache).is_file():
        det_cache = json.loads(Path(args.detection_cache).read_text())
        kept = sum(1 for v in det_cache.values() if v is not None)
        print(f"detection cache: {len(det_cache)} entries ({kept} with hand)")
    else:
        raise SystemExit(f"detection cache not found: {args.detection_cache}")

    detected = [(s, tuple(det_cache[s.path]))
                for s in samples if det_cache.get(s.path) is not None]
    undetected = [s for s in samples if det_cache.get(s.path) is None]
    print(f"MediaPipe detection: {len(detected)}/{len(samples)} "
          f"({100 * len(detected) / len(samples):.1f}%)")

    model = load_checkpoint(args.checkpoint, args.device)

    # ---- build tensors ---- #
    orig_tensors: List[np.ndarray] = []
    sym_tensors: List[np.ndarray] = []
    fore_tensors: List[np.ndarray] = []
    cal_tensors: List[np.ndarray] = []
    sym_bboxes: List[Tuple[int, int, int, int]] = []
    fore_bboxes: List[Tuple[int, int, int, int]] = []
    cal_bboxes: List[Tuple[int, int, int, int]] = []

    t0 = time.time()
    # original: every frame
    for s in samples:
        img = cv2.imread(s.path, cv2.IMREAD_COLOR)
        if img is None:
            raise SystemExit(f"unreadable image: {s.path}")
        orig_tensors.append(_to_tensor(img))

    # crop paths: detected frames only
    for s, tb in detected:
        img = cv2.imread(s.path, cv2.IMREAD_COLOR)
        sym_tensors.append(
            _to_tensor(crop_square_with_margin(img, tb, args.crop_margin)))
        fore_tensors.append(
            _to_tensor(crop_square_with_forearm_context(
                img, tb, DEFAULT_ASYMMETRIC_MARGINS)))
        cal_tensors.append(
            _to_tensor(crop_square_with_training_framing(img, tb, margins)))
        sym_bboxes.append(tuple(tb))
        fore_bboxes.append(asymmetric_context_bbox(
            img, tb, DEFAULT_ASYMMETRIC_MARGINS))
        cal_bboxes.append(training_calibrated_context_bbox(img, tb, margins))
    print(f"built {len(orig_tensors)} original + {len(sym_tensors)}x3 crop tensors "
          f"({time.time() - t0:.1f}s)")

    # ---- forward ---- #
    y_all = np.array([s.label for s in samples], dtype=np.int64)
    y_det = np.array([s.label for s, _ in detected], dtype=np.int64)
    p_orig = batch_predict(model, args.device, np.stack(orig_tensors), args.batch_size)
    p_sym = batch_predict(model, args.device, np.stack(sym_tensors), args.batch_size)
    p_fore = batch_predict(model, args.device, np.stack(fore_tensors), args.batch_size)
    p_cal = batch_predict(model, args.device, np.stack(cal_tensors), args.batch_size)

    # ---- metrics ---- #
    n_cls = len(CLASSES)
    cm_orig = confusion_matrix(y_all, p_orig, n_cls)
    cm_sym  = confusion_matrix(y_det, p_sym, n_cls)
    cm_fore = confusion_matrix(y_det, p_fore, n_cls)
    cm_cal  = confusion_matrix(y_det, p_cal, n_cls)

    acc_orig = float(cm_orig.trace() / max(1, cm_orig.sum()))
    acc_sym  = float(cm_sym.trace()  / max(1, cm_sym.sum()))
    acc_fore = float(cm_fore.trace() / max(1, cm_fore.sum()))
    acc_cal  = float(cm_cal.trace()  / max(1, cm_cal.sum()))

    f1_orig = float(macro_f1(cm_orig))
    f1_sym  = float(macro_f1(cm_sym))
    f1_fore = float(macro_f1(cm_fore))
    f1_cal  = float(macro_f1(cm_cal))

    rec_orig = per_class_accuracy(cm_orig)
    rec_sym  = per_class_accuracy(cm_sym)
    rec_fore = per_class_accuracy(cm_fore)
    rec_cal  = per_class_accuracy(cm_cal)
    n_det_per_class = Counter(s.class_name for s, _ in detected)

    # ---- print ---- #
    print(f"\noverall accuracy:")
    print(f"  original     (all {len(samples)}):  {acc_orig:.4f}  F1 {f1_orig:.4f}")
    print(f"  symmetric    (det, n={len(y_det)}): {acc_sym:.4f}  F1 {f1_sym:.4f}")
    print(f"  forearm      (det, n={len(y_det)}): {acc_fore:.4f}  F1 {f1_fore:.4f}")
    print(f"  calibrated   (det, n={len(y_det)}): {acc_cal:.4f}  F1 {f1_cal:.4f}")

    print(f"\n{'class':>8} {'orig':>8} {'sym':>8} {'fore':>8} {'cal':>8}  n_det")
    for c in CLASSES:
        o = rec_orig.get(c, float("nan"))
        s_v = rec_sym.get(c, float("nan"))
        f_v = rec_fore.get(c, float("nan"))
        cv = rec_cal.get(c, float("nan"))
        print(f"{c:>8} {o:8.3f} {s_v:8.3f} {f_v:8.3f} {cv:8.3f}  "
              f"{n_det_per_class.get(c, 0)}")

    print(f"\nfocus classes (S/E/M/N/K/V/X):")
    for c in FOCUS_CLASSES:
        o = rec_orig.get(c, float("nan"))
        print(f"  {c}: orig {o:.3f}  sym {rec_sym.get(c, float('nan')):.3f}  "
              f"fore {rec_fore.get(c, float('nan')):.3f}  "
              f"cal {rec_cal.get(c, float('nan')):.3f}  "
              f"(n_det {n_det_per_class.get(c, 0)})")

    # ---- report ---- #
    report = {
        "schema": REPORT_SCHEMA,
        "checkpoint": args.checkpoint,
        "device": args.device,
        "source": {"kind": "dev_val", "path": args.manifest},
        "crop": {
            "symmetric_margin": args.crop_margin,
            "forearm_margins": DEFAULT_ASYMMETRIC_MARGINS.__dict__,
            "calibrated_margins": margins.__dict__,
        },
        "n_total": len(samples),
        "detection": {
            "found": len(detected),
            "n": len(samples),
            "rate": len(detected) / max(1, len(samples)),
            "per_class": {c: {"found": n_det_per_class.get(c, 0),
                              "n": sum(1 for s in samples if s.class_name == c)}
                          for c in CLASSES},
        },
        "accuracy": {
            "original_all_frames": acc_orig,
            "symmetric_given_detection": acc_sym,
            "forearm_given_detection": acc_fore,
            "calibrated_given_detection": acc_cal,
        },
        "macro_f1": {
            "original_all_frames": f1_orig,
            "symmetric_given_detection": f1_sym,
            "forearm_given_detection": f1_fore,
            "calibrated_given_detection": f1_cal,
        },
        "per_class_recall": {
            "original": rec_orig,
            "symmetric_given_detection": rec_sym,
            "forearm_given_detection": rec_fore,
            "calibrated_given_detection": rec_cal,
        },
        "focus_classes": list(FOCUS_CLASSES),
        "notes": [
            "'given detection' excludes MediaPipe failures.",
            "original accuracy is over ALL frames.",
            "calibrated margins come from the full training set analysis.",
        ],
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"\nwrote {out}")

    if args.figures:
        figs = Path(args.figures)
        figs.mkdir(parents=True, exist_ok=True)
        for name, cm in [("original", cm_orig), ("symmetric", cm_sym),
                         ("forearm", cm_fore), ("calibrated", cm_cal)]:
            save_confusion_png(cm, str(figs / f"confusion_{name}.png"))
            save_confusion_csv(cm, str(figs / f"confusion_{name}.csv"))


def main() -> None:
    p = argparse.ArgumentParser(
        description="Benchmark calibrated vs other crops on the val split.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--checkpoint", default="checkpoints/efficientnet_b0_targeted.pt")
    p.add_argument("--manifest", default="data/split_manifest.json")
    p.add_argument("--device", default="auto",
                   help="auto | cpu | cuda | mps")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--crop-margin", type=float, default=0.62,
                   help="symmetric margin for the 'symmetric' path")
    p.add_argument("--limit-per-class", type=int, default=0,
                   help="subsample to N frames per class (0 = full val)")
    p.add_argument("--detection-cache",
                   default="data/forearm_crop_detection_records.json")
    p.add_argument("--calibrated-json",
                   default="data/training_framing_derived.json",
                   help="JSON with TRAIN_L/R/T/B keys")
    p.add_argument("--out",
                   default="docs/reports/calibrated_crop_benchmark.json")
    p.add_argument("--figures",
                   default="docs/figures/framing_comparison/calibrated_crop")
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
