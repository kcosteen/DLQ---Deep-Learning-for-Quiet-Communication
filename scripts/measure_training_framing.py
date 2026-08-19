"""Phase 1: measure the hand/image framing of the training images.

The webcam bridge (#17) must produce crops that look like the training images:
the hand has to occupy the same proportion of the frame, in the same part of the
frame, with the same amount of surrounding context. This script turns that fuzzy
goal into numbers, using MediaPipe *as a localizer only*:

  hand bbox width  / image width
  hand bbox height / image height
  hand bbox area   / image area
  hand center X / Y relative to the image
  context fraction around the hand
  the square-crop margin implied by each training frame

The "implied margin" is derived from the hand-fill fraction: a training image is
a full 200x200 camera frame in which the hand's longest side spans
``max(bbox_w, bbox_h) / image_side`` (median ~0.45). A webcam crop reproduces
that framing when ``bbox_side * (1 + 2m) == crop_side`` and the hand fills the
same fraction of the crop, i.e. ``m = (image_side / max(bbox_w, bbox_h) - 1) / 2``
(the ``fit_margin`` column). Median across detected frames is the recommended
``--crop-margin`` default for the live app.

``fit_margin_pos`` is kept as a *diagnostic only*: it is the largest square crop
centred on the hand that stays inside the image, which is dominated by how close
the hand sits to the frame edge (many training hands are near the edge) and so
is NOT a good basis for the default margin — that is what the first version of
this script got wrong.

HONESTY RULES (this is the point of the exercise):

* No centre fallback. A frame MediaPipe cannot localize is recorded as
  ``found: false`` and contributes NOTHING to the framing statistics — it is
  never silently treated as a hand at the image centre.
* Detection coverage is reported per class. ``nothing`` is an empty scene, so a
  low rate there is correct and it is excluded from the hand statistics.
* The number comes from the real landmarker, not from label priors.

Results are cached as a JSON of per-frame records so re-running does not re-run
MediaPipe (set ``--cache`` to a path; a fresh path is written after the first
run).

Usage::

    python scripts/measure_training_framing.py \\
        --root data/raw/asl_alphabet_train \\
        --per-class 20 \\
        --cache data/training_framing_records.json \\
        --out data/training_framing_stats.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import cv2

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data import CLASSES, list_class_files  # noqa: E402

NO_HAND_CLASS = "nothing"


# --------------------------------------------------------------------------- #
# Sampling
# --------------------------------------------------------------------------- #
def sample_paths(root: str, per_class: int) -> List[tuple]:
    """``per_class`` frames per class via deterministic stride across the range.

    A stride (not a random draw) so consecutive near-duplicate video frames do
    not crowd the sample and the result is reproducible across machines.
    """
    out = []
    for cls in CLASSES:
        files = list_class_files(root, cls)
        if not files:
            continue
        if len(files) > per_class:
            stride = len(files) / per_class
            files = [files[int(i * stride)] for i in range(per_class)]
        out += [(cls, f) for f in files]
    return out


# --------------------------------------------------------------------------- #
# Per-frame measurement
# --------------------------------------------------------------------------- #
def measure_frame(img: cv2.Mat, tight_bbox: Optional[tuple]) -> Dict:
    """Frame-level statistics. Only called with a *detected* hand (``tight_bbox``
    is not None) — there is no centre-fallback path here."""
    h, w = img.shape[:2]
    x0, y0, x1, y1 = tight_bbox
    bw, bh = x1 - x0, y1 - y0
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    img_side = min(h, w)
    # Largest square crop centred on the hand that stays inside the image.
    crop_side = 2.0 * min(cx, w - cx, cy, h - cy)
    return {
        "img_w": int(w), "img_h": int(h),
        "bbox_w": int(bw), "bbox_h": int(bh),
        "bw_ratio": bw / w,          # hand bbox width / image width
        "bh_ratio": bh / h,          # hand bbox height / image height
        "area_ratio": (bw * bh) / (w * h),  # hand bbox area / image area
        "center_x": cx / w,          # hand centre, fraction across image
        "center_y": cy / h,          # hand centre, fraction down image
        "context_w": 1.0 - bw / w,   # image width outside the hand
        "context_h": 1.0 - bh / h,   # image height outside the hand
        "fit_margin": (img_side / max(bw, bh) - 1.0) / 2.0,
        # ...ignoring where the hand sits
        "fit_margin_pos": (crop_side / max(bw, bh) - 1.0) / 2.0,
        # ...respecting the hand's actual position (honest one)
        "context_frac": 1.0 - max(bw, bh) / crop_side,
        # ...context inside that honest square crop
    }


def run(root: str, per_class: int, cache: Optional[str], quiet: bool) -> Dict:
    records: Dict[str, Dict] = {}
    if cache and Path(cache).is_file():
        records = json.loads(Path(cache).read_text())
        kept = sum(r["found"] for r in records.values())
        print(f"cache hit: {len(records)} frames ({kept} with a hand) from {cache}")

    paths = sample_paths(root, per_class)
    missing = [f for _, f in paths if str(f) not in records]
    if missing:
        from src.crop import HandCropper  # local import (mediapipe is heavy)

        with HandCropper() as cropper:
            for f in missing:
                if f in records:
                    continue
                cls = Path(f).parent.name
                img = cv2.imread(f, cv2.IMREAD_COLOR)
                if img is None:
                    records[f] = {"class": cls, "found": False, "unreadable": True}
                    continue
                tb = cropper.tight_bbox(img)
                rec = {"class": cls, "found": tb is not None,
                       "path": str(f)}
                if tb is not None:
                    rec.update(measure_frame(img, tb))
                records[f] = rec
        if cache:
            Path(cache).parent.mkdir(parents=True, exist_ok=True)
            Path(cache).write_text(json.dumps(records, indent=2, sort_keys=True))
            print(f"wrote {len(missing)} new frames -> {cache}")
    else:
        print("no new frames to measure (cache covers the sample)")

    stats = summarize(records)
    if not quiet:
        print_summary(stats)
    return stats


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #
def _pct(values: List[float], q: float) -> float:
    if not values:
        return float("nan")
    s = sorted(values)
    k = min(len(s) - 1, int(round(q * (len(s) - 1))))
    return float(s[k])


def _agg(values: List[float]) -> Dict:
    if not values:
        return {"n": 0, "mean": None, "median": None, "p25": None, "p75": None}
    return {
        "n": len(values),
        "mean": round(float(sum(values) / len(values)), 4),
        "median": round(_pct(values, 0.5), 4),
        "p25": round(_pct(values, 0.25), 4),
        "p75": round(_pct(values, 0.75), 4),
    }


def summarize(records: Dict[str, Dict]) -> Dict:
    by_class: Dict[str, List[int]] = {c: [] for c in CLASSES}
    hand_metrics: Dict[str, List[float]] = {
        "bw_ratio": [], "bh_ratio": [], "area_ratio": [],
        "center_x": [], "center_y": [], "context_w": [], "context_h": [],
        "fit_margin": [], "fit_margin_pos": [], "context_frac": [],
    }
    for f, rec in records.items():
        cls = rec["class"]
        found = rec.get("found", False)
        if cls in by_class and not rec.get("unreadable"):
            by_class[cls].append(1 if found else 0)
        if found and cls != NO_HAND_CLASS:
            for k in hand_metrics:
                v = rec.get(k)
                if v is not None:
                    hand_metrics[k].append(float(v))

    coverage = {
        c: {"found": sum(v), "n": len(v),
            "rate": (sum(v) / len(v)) if v else None}
        for c, v in by_class.items()
    }
    hand_classes = [c for c in CLASSES if c != NO_HAND_CLASS]
    det_total = sum(coverage[c]["found"] for c in hand_classes)
    det_n = sum(coverage[c]["n"] for c in hand_classes)

    # Base the recommendation on the hand-fill fraction (fit_margin), NOT on
    # fit_margin_pos (edge-dominated; see module docstring).
    margin_values = hand_metrics["fit_margin"]
    recommended_margin = _agg(margin_values)["median"]

    return {
        "root": None,  # filled by caller
        "sample": {"n_frames": len(records),
                   "per_class": None},  # filled by caller
        "detection_coverage": {
            "overall_hand_classes": {"found": det_total, "n": det_n,
                                     "rate": det_total / det_n if det_n else None},
            "per_class": coverage,
            "nothing": coverage.get(NO_HAND_CLASS),
        },
        "framing": {k: _agg(v) for k, v in hand_metrics.items()},
        "recommended_crop_margin": recommended_margin,
        "notes": [
            "framing statistics computed ONLY from frames MediaPipe detected a "
            "hand on; no centre fallback is counted as a hand.",
            "fit_margin = margin that reproduces the training hand-fill fraction "
            "(bbox_side/image_side) in a square webcam crop. median(fit_margin) "
            "is the recommended --crop-margin default.",
            "fit_margin_pos (largest square crop centred on the hand that fits "
            "in the image) is diagnostic only: it is dominated by edge placement "
            "— many training hands sit near the frame edge — so it must not "
            "drive the default margin.",
            "'nothing' is an empty scene: low detection there is correct and it "
            "is excluded from framing stats.",
        ],
    }


def print_summary(stats: Dict) -> None:
    det = stats["detection_coverage"]
    print(f"\ndetection coverage (hand classes): "
          f"{det['overall_hand_classes']['found']}/{det['overall_hand_classes']['n']} "
          f"({100*det['overall_hand_classes']['rate']:.1f}%)")
    print(f"{'class':>8} {'found':>7} {'n':>5}  rate")
    for c, d in det["per_class"].items():
        if not d["n"]:
            continue
        rate = d["rate"] or 0.0
        bar = "#" * int(30 * rate)
        print(f"{c:>8} {d['found']:>7} {d['n']:>5} {100*rate:5.0f}%  {bar}")

    f = stats["framing"]
    print("\nframing (detected hand frames only):")
    labels = {
        "bw_ratio": "bbox width / image width",
        "bh_ratio": "bbox height / image height",
        "area_ratio": "bbox area / image area",
        "center_x": "hand centre X (frac of width)",
        "center_y": "hand centre Y (frac of height)",
        "context_w": "context fraction, width",
        "context_h": "context fraction, height",
        "fit_margin": "implied margin (centred, pos ignored)",
        "fit_margin_pos": "implied margin (respects position)",
        "context_frac": "context fraction inside implied square crop",
    }
    print(f"{'metric':>40} {'mean':>8} {'median':>8} {'p25':>8} {'p75':>8}  n")
    for k, label in labels.items():
        a = f[k]
        row = f"{label:>40} "
        if a["n"]:
            row += f"{a['mean']:>8.4f} {a['median']:>8.4f} "
            row += f"{a['p25']:>8.4f} {a['p75']:>8.4f}  {a['n']}"
        else:
            row += "no detected samples"
        print(row)
    print(f"\nrecommended --crop-margin: "
          f"{stats['recommended_crop_margin']} "
          f"(median of the hand-fill-derived fit_margin)")


def main() -> None:
    p = argparse.ArgumentParser(
        description="Measure the hand/image framing of the training images (#17).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--root", required=True,
                   help="asl_alphabet_train/ (29 class folders)")
    p.add_argument("--per-class", type=int, default=20,
                   help="frames sampled per class (deterministic stride)")
    p.add_argument("--cache", default="data/training_framing_records.json",
                   help="JSON of per-frame records (written then reused)")
    p.add_argument("--out", default="data/training_framing_stats.json",
                   help="where to write the aggregate statistics")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()

    stats = run(args.root, args.per_class, args.cache, args.quiet)
    stats["root"] = args.root
    stats["sample"]["per_class"] = args.per_class
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(stats, indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
