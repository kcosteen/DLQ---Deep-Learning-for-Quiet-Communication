"""Forearm-crop ablation: original vs symmetric vs forearm crop on the val split.

The live webcam crop reproduces the training framing through a hand-centred
square margin (``--crop-margin 0.62``). The forearm-crop experiment asks whether
the asymmetric bottom-heavy crop (``crop_square_with_forearm_context``, margins
L/R/T/B = 0.7/0.7/0.4/1.8) matches the original 200x200 dataset framing better,
because those frames show the hand PLUS a substantial wrist/forearm segment.

For every val image where MediaPipe finds a hand this script scores three paths:

  original  : raw val image -> preprocess_bgr -> model
  symmetric : current hand crop (margin 0.62) -> preprocess_bgr -> model
  forearm   : asymmetric forearm-preserving crop -> preprocess_bgr -> model

Reported SEPARATELY (never mixed):

  * MediaPipe detection coverage
  * original accuracy            (all frames; the crop paths need a detection)
  * symmetric-crop accuracy given detection
  * forearm-crop accuracy given detection
  * per-class recall for each path
  * confusion matrices per path
  * end-to-end success: detected AND classified correctly / total

DIAGNOSTIC ONLY: margins are NOT tuned on validation labels here. MediaPipe
failures are counted in coverage and end-to-end, never as classifier errors in
the "given detection" metrics.

MediaPipe detection is cached per-path (``--detection-cache``) so re-runs only
re-classify. Inference is batched; ``--device mps`` is strongly recommended
(CPU is ~10x slower).

Usage::

    python scripts/benchmark_forearm_crop.py \\
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
    asymmetric_context_bbox,
    crop_square_with_forearm_context,
    crop_square_with_margin,
    preprocess_bgr,
)
from src.data import CLASSES, SplitManifest  # noqa: E402
from src.evaluate import (  # noqa: E402
    confusion_matrix,
    load_checkpoint,
    per_class_accuracy,
    save_confusion_csv,
    save_confusion_png,
)

FOCUS_CLASSES = ("S", "E", "M", "N", "K", "V", "X")
REPORT_SCHEMA = "asl-forearm-crop-benchmark/v1"
RECORD_SCHEMA = "asl-forearm-crop-records/v1"


# --------------------------------------------------------------------------- #
# Detection (cached)
# --------------------------------------------------------------------------- #
def detection_pass(
    paths: List[str], cache_path: Optional[str], min_detection: float
) -> Dict[str, Optional[Tuple[int, int, int, int]]]:
    """MediaPipe tight bbox per path; None for undetected. Cached to JSON."""
    cache: Dict[str, Optional[List[int]]] = {}
    if cache_path and Path(cache_path).is_file():
        cache = json.loads(Path(cache_path).read_text())
        kept = sum(1 for v in cache.values() if v)
        print(f"detection cache hit: {len(cache)} frames ({kept} with a hand)")

    missing = [p for p in paths if str(p) not in cache]
    if missing:
        from src.crop import HandCropper  # local import (mediapipe is heavy)

        with HandCropper(min_detection_confidence=min_detection) as cropper:
            for i, p in enumerate(missing):
                img = cv2.imread(p, cv2.IMREAD_COLOR)
                tb = cropper.tight_bbox(img) if img is not None else None
                cache[str(p)] = list(tb) if tb is not None else None
                if (i + 1) % 2000 == 0:
                    print(f"  detection {i + 1}/{len(missing)}")
        if cache_path:
            Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
            Path(cache_path).write_text(
                json.dumps(cache, indent=2, sort_keys=True) + "\n")
            print(f"wrote {len(missing)} new frames -> {cache_path}")
    return cache


# --------------------------------------------------------------------------- #
# Inference
# --------------------------------------------------------------------------- #
def _to_tensor(img_bgr: np.ndarray) -> np.ndarray:
    return preprocess_bgr(img_bgr)


def batch_predict(model, device, chunk_tensors, batch_size: int) -> np.ndarray:
    """argmax over a (N,3,224,224) float32 array in batches."""
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
    print(f"val split: {len(samples)} images "
          f"({args.manifest})"
          + (f", limited to {args.limit_per_class}/class" if args.limit_per_class else ""))

    paths = [s.path for s in samples]
    dets = detection_pass(paths, args.detection_cache, args.min_detection)

    detected = [(s, dets[s.path]) for s in samples if dets.get(s.path) is not None]
    undetected = [s for s in samples if dets.get(s.path) is None]
    print(f"MediaPipe detection: {len(detected)}/{len(samples)} "
          f"({100 * len(detected) / len(samples):.1f}%)")

    model = load_checkpoint(args.checkpoint, args.device)

    # ---- build tensors ----------------------------------------------------- #
    # original: every frame (crop paths need a detection).
    # symmetric / forearm: detected frames only.
    orig_tensors: List[np.ndarray] = []
    sym_tensors: List[np.ndarray] = []
    fore_tensors: List[np.ndarray] = []
    sym_bboxes: List[Tuple[int, int, int, int]] = []
    fore_bboxes: List[Tuple[int, int, int, int]] = []
    t0 = time.time()
    for s in samples:
        img = cv2.imread(s.path, cv2.IMREAD_COLOR)
        if img is None:
            raise SystemExit(f"unreadable image: {s.path}")
        orig_tensors.append(_to_tensor(img))
    for s, tb in detected:
        img = cv2.imread(s.path, cv2.IMREAD_COLOR)
        sym_tensors.append(_to_tensor(crop_square_with_margin(img, tb, args.crop_margin)))
        fore_tensors.append(
            _to_tensor(crop_square_with_forearm_context(img, tb, DEFAULT_ASYMMETRIC_MARGINS)))
        sym_bboxes.append(tuple(tb))
        fore_bboxes.append(asymmetric_context_bbox(img, tb, DEFAULT_ASYMMETRIC_MARGINS))
    print(f"built {len(orig_tensors)} original + {len(sym_tensors)}x2 crop tensors "
          f"({time.time() - t0:.1f}s)")

    # ---- forward ----------------------------------------------------------- #
    y_all = np.array([s.label for s in samples], dtype=np.int64)
    p_orig = batch_predict(model, args.device, np.stack(orig_tensors), args.batch_size)
    y_det = np.array([s.label for s, _ in detected], dtype=np.int64)
    p_sym = p_fore = None
    if sym_tensors:
        p_sym = batch_predict(model, args.device, np.stack(sym_tensors), args.batch_size)
        p_fore = batch_predict(model, args.device, np.stack(fore_tensors), args.batch_size)

    # ---- metrics ----------------------------------------------------------- #
    n = len(CLASSES)
    cm_orig = confusion_matrix(y_all, p_orig, n)
    cm_sym = confusion_matrix(y_det, p_sym, n) if p_sym is not None else None
    cm_fore = confusion_matrix(y_det, p_fore, n) if p_fore is not None else None

    acc_orig = float(cm_orig.trace() / max(1, cm_orig.sum()))
    acc_sym = float(cm_sym.trace() / max(1, cm_sym.sum())) if cm_sym is not None else None
    acc_fore = float(cm_fore.trace() / max(1, cm_fore.sum())) if cm_fore is not None else None

    # end-to-end: detected AND classified correctly / total.
    e2e = np.zeros((n, n), dtype=np.int64)
    for (s, _), p in zip(detected, p_fore):
        e2e[s.label, int(p)] += 1
    e2e_det_and_correct = int(e2e.trace())
    e2e_rate = e2e_det_and_correct / len(samples)

    # per-class recall: original over all, crops given detection.
    rec_orig = per_class_accuracy(cm_orig)
    rec_sym = per_class_accuracy(cm_sym) if cm_sym is not None else {}
    rec_fore = per_class_accuracy(cm_fore) if cm_fore is not None else {}
    n_det_per_class = Counter(s.class_name for s, _ in detected)

    report = {
        "schema": REPORT_SCHEMA,
        "checkpoint": args.checkpoint,
        "device": args.device,
        "source": {"kind": "dev_val", "path": args.manifest},
        "crop": {
            "symmetric_margin": args.crop_margin,
            "forearm_margins": DEFAULT_ASYMMETRIC_MARGINS.__dict__,
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
        },
        "end_to_end": {
            "detected_and_correct": e2e_det_and_correct,
            "n_total": len(samples),
            "rate": e2e_rate,
        },
        "per_class_recall": {
            "original": rec_orig,
            "symmetric_given_detection": rec_sym,
            "forearm_given_detection": rec_fore,
        },
        "focus_classes": list(FOCUS_CLASSES),
        "notes": [
            "'accuracy given detection' excludes MediaPipe failures; failures are "
            "counted only in coverage and end-to-end.",
            "original accuracy is over ALL frames (the crop paths need detection).",
            "DIAGNOSTIC ONLY: margins were not tuned on validation labels.",
        ],
    }

    # ---- print ------------------------------------------------------------- #
    print(f"\noverall accuracy:")
    print(f"  original (all {len(samples)} frames): {acc_orig:.4f}")
    if acc_sym is not None:
        print(f"  symmetric crop (given detection, n={len(y_det)}): {acc_sym:.4f}")
        print(f"  forearm crop   (given detection, n={len(y_det)}): {acc_fore:.4f}")
    print(f"end-to-end detected AND correct: {e2e_det_and_correct}/{len(samples)} "
          f"({100 * e2e_rate:.1f}%)")

    print(f"\n{'class':>8} {'orig':>8} {'sym':>8} {'fore':>8}   n_det")
    for c in CLASSES:
        o = rec_orig.get(c)
        row = f"{c:>8} {o if o is None else f'{o:8.3f}'}"
        row += f" {rec_sym.get(c, float('nan')):8.3f}" if cm_sym is not None else "        -"
        row += f" {rec_fore.get(c, float('nan')):8.3f}" if cm_fore is not None else "        -"
        print(f"{row}   {n_det_per_class.get(c, 0)}")
    print("\nfocus classes (S/E/M/N/K/V/X):")
    for c in FOCUS_CLASSES:
        o = rec_orig.get(c)
        print(f"  {c}: orig {o:.3f}  sym {rec_sym.get(c, float('nan')):.3f}  "
              f"fore {rec_fore.get(c, float('nan')):.3f}  "
              f"(n_det {n_det_per_class.get(c, 0)})")

    # ---- artifacts --------------------------------------------------------- #
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"\nwrote {out}")

    if args.figures:
        figs = Path(args.figures)
        figs.mkdir(parents=True, exist_ok=True)
        for name, cm in (("original", cm_orig), ("symmetric", cm_sym),
                         ("forearm", cm_fore)):
            if cm is None:
                continue
            save_confusion_png(cm, str(figs / f"confusion_{name}.png"))
            save_confusion_csv(cm, str(figs / f"confusion_{name}.csv"))

    # ---- per-frame records for the gallery --------------------------------- #
    j_map = {id(s): j for j, (s, _) in enumerate(detected)}
    recs: List[Dict] = []
    for i, s in enumerate(samples):
        tb = dets.get(s.path)
        rec = {
            "path": s.path,
            "class": s.class_name,
            "detected": tb is not None,
            "bbox": tb if tb is not None else None,
            "pred_original": int(p_orig[i]),
        }
        if tb is not None:
            j = j_map[id(s)]
            rec["pred_symmetric"] = int(p_sym[j]) if p_sym is not None else None
            rec["pred_forearm"] = int(p_fore[j]) if p_fore is not None else None
            rec["sym_bbox"] = sym_bboxes[j]
            rec["fore_bbox"] = fore_bboxes[j]
        recs.append(rec)
    if args.records:
        rp = Path(args.records)
        rp.parent.mkdir(parents=True, exist_ok=True)
        payload = {"schema": RECORD_SCHEMA, "classes": list(CLASSES), "records": recs}
        rp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        print(f"wrote {rp}")


def main() -> None:
    p = argparse.ArgumentParser(
        description="Ablation: original vs symmetric vs forearm crop on the val split.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--checkpoint", default="checkpoints/efficientnet_b0_targeted.pt")
    p.add_argument("--manifest", default="data/split_manifest.json")
    p.add_argument("--device", default="auto",
                   help="auto | cpu | cuda | mps (mps recommended on Apple Silicon)")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--crop-margin", type=float, default=0.62,
                   help="symmetric margin used for the 'symmetric' path")
    p.add_argument("--min-detection", type=float, default=0.5)
    p.add_argument("--limit-per-class", type=int, default=0,
                   help="subsample to N frames per class (0 = full val split)")
    p.add_argument("--detection-cache",
                   default="data/forearm_crop_detection_records.json")
    p.add_argument("--out", default="docs/reports/forearm_crop_benchmark.json")
    p.add_argument("--figures", default="docs/figures/framing_comparison/forearm_crop",
                   help="dir for confusion PNG/CSV ('' to skip)")
    p.add_argument("--records", default="data/forearm_crop_records.json",
                   help="per-frame predictions for the gallery ('' to skip)")
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
