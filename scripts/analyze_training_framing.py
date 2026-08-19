"""Step 1–2: Measure per-image framing margins on the FULL training dataset.

For every training image where MediaPipe detects a hand, record the context
margins (L, R, T, B) expressed as *multiples of the hand bbox width/height*:

    L = x0 / bw          (left margin in units of bbox width)
    R = (W - x1) / bw    (right margin)
    T = y0 / bh          (top margin in units of bbox height)
    B = (H - y1) / bh    (bottom margin)

These ratios are *portable*: they can be replayed on MediaPipe detections
from rectangular live camera frames to reproduce the training framing.

Artifacts:
    data/training_framing_all_records.json   per-image bounding-box + LRTB ratios
    data/training_framing_all_summary.json   global + per-class aggregate statistics

Usage::

    python scripts/analyze_training_framing.py \\
        --root data/raw/asl_alphabet_train \\
        --cache data/training_framing_all_records.json \\
        --out   data/training_framing_all_summary.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data import CLASSES, list_class_files  # noqa: E402

NO_HAND_CLASS = "nothing"


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _pct(values: Sequence[float], q: float) -> float:
    """Percentile (linear interpolation, 0..1)."""
    if not values:
        return float("nan")
    s = sorted(values)
    k = q * (len(s) - 1)
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    frac = k - lo
    return float(s[lo] * (1.0 - frac) + s[hi] * frac)


def _agg(values: Sequence[float]) -> Dict:
    """Mean, median, p10, p25, p75, p90 for a list of floats."""
    if not values:
        return {"n": 0, "mean": None, "median": None,
                "p10": None, "p25": None, "p75": None, "p90": None}
    return {
        "n": len(values),
        "mean": round(float(np.mean(values)), 4),
        "median": round(float(np.median(values)), 4),
        "p10": round(_pct(values, 0.10), 4),
        "p25": round(_pct(values, 0.25), 4),
        "p75": round(_pct(values, 0.75), 4),
        "p90": round(_pct(values, 0.90), 4),
    }


# --------------------------------------------------------------------------- #
# Per-frame analysis
# --------------------------------------------------------------------------- #

def analyze_frame(img: np.ndarray, tight_bbox: Tuple[int, int, int, int]) -> Dict:
    """Compute L/R/T/B margin ratios + hand-centre/scale metrics for one image."""
    h, w = img.shape[:2]
    x0, y0, x1, y1 = tight_bbox
    bw, bh = x1 - x0, y1 - y0
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0

    # Margin ratios in units of bbox width/height
    L = x0 / bw if bw > 0 else 0.0
    R = (w - x1) / bw if bw > 0 else 0.0
    T = y0 / bh if bh > 0 else 0.0
    B = (h - y1) / bh if bh > 0 else 0.0

    return {
        "img_w": int(w),
        "img_h": int(h),
        "bbox_x0": int(x0), "bbox_y0": int(y0),
        "bbox_x1": int(x1), "bbox_y1": int(y1),
        "bbox_w": int(bw),  "bbox_h": int(bh),
        "L": round(L, 4),
        "R": round(R, 4),
        "T": round(T, 4),
        "B": round(B, 4),
        "hand_center_x": round(cx / w, 4),
        "hand_center_y": round(cy / h, 4),
        "hand_w_over_img_w": round(bw / w, 4),
        "hand_h_over_img_h": round(bh / h, 4),
    }


# --------------------------------------------------------------------------- #
# Full-dataset pass
# --------------------------------------------------------------------------- #

def all_training_paths(root: str) -> List[Tuple[str, str]]:
    """Return [(class_name, absolute_path), ...] for every training image."""
    out: List[Tuple[str, str]] = []
    for cls in CLASSES:
        for f in list_class_files(root, cls):
            out.append((cls, f))
    return out


def detection_pass(
    paths: List[Tuple[str, str]],
    cache_path: Optional[str],
    min_detection: float,
) -> Dict[str, Optional[List[int]]]:
    """Run MediaPipe tight_bbox on every image; cache as {path: [x0,y0,x1,y1]|None}."""
    cache: Dict[str, Optional[List[int]]] = {}
    if cache_path and Path(cache_path).is_file():
        cache = json.loads(Path(cache_path).read_text())
        kept = sum(1 for v in cache.values() if v is not None)
        print(f"detection cache hit: {len(cache)} images ({kept} with hand)")

    missing = [p for _, p in paths if str(p) not in cache]
    if not missing:
        return cache

    from src.crop import HandCropper
    with HandCropper(min_detection_confidence=min_detection) as cropper:
        t0 = time.time()
        for i, p in enumerate(missing):
            img = cv2.imread(p, cv2.IMREAD_COLOR)
            tb = cropper.tight_bbox(img) if img is not None else None
            cache[str(p)] = list(tb) if tb is not None else None
            if (i + 1) % 5000 == 0:
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed
                eta = (len(missing) - i - 1) / rate
                print(f"  detection {i + 1}/{len(missing)} "
                      f"({rate:.1f} img/s, ETA {eta / 60:.1f} min)")
                # incremental save
                if cache_path:
                    Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
                    Path(cache_path).write_text(json.dumps(cache) + "\n")
    if cache_path:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        Path(cache_path).write_text(json.dumps(cache) + "\n")
        print(f"wrote {len(cache)} entries -> {cache_path}")
    return cache


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #

def aggregate(
    paths: List[Tuple[str, str]],
    det_cache: Dict[str, Optional[List[int]]],
) -> Dict:
    """Compute global + per-class L/R/T/B + hand-centre/scale statistics."""
    global_L, global_R, global_T, global_B = [], [], [], []
    global_cx, global_cy = [], []
    global_hw, global_hh = [], []

    by_class: Dict[str, Dict] = {
        c: {"found": 0, "n": 0,
            "L": [], "R": [], "T": [], "B": [],
            "cx": [], "cy": [], "hw": [], "hh": []}
        for c in CLASSES
    }

    for cls, path in paths:
        by_class[cls]["n"] += 1
        tb = det_cache.get(str(path))
        if tb is None:
            continue
        by_class[cls]["found"] += 1
        tb_t = tuple(tb)
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            continue
        m = analyze_frame(img, tb_t)
        _key_map = {
            "L": "L", "R": "R", "T": "T", "B": "B",
            "cx": "hand_center_x", "cy": "hand_center_y",
            "hw": "hand_w_over_img_w", "hh": "hand_h_over_img_h",
        }
        for short, g_list in [("L", global_L), ("R", global_R),
                              ("T", global_T), ("B", global_B),
                              ("cx", global_cx), ("cy", global_cy),
                              ("hw", global_hw), ("hh", global_hh)]:
            g_list.append(m[_key_map[short]])
        c = by_class[cls]
        for k in ("L", "R", "T", "B"):
            c[k].append(m[k])
        c["cx"].append(m["hand_center_x"])
        c["cy"].append(m["hand_center_y"])
        c["hw"].append(m["hand_w_over_img_w"])
        c["hh"].append(m["hand_h_over_img_h"])

    total_n = len(paths)
    total_found = sum(v["found"] for v in by_class.values())
    nothing = by_class.get(NO_HAND_CLASS, {"found": 0, "n": 0})
    hand_classes = [c for c in CLASSES if c != NO_HAND_CLASS]
    hand_found = sum(by_class[c]["found"] for c in hand_classes)
    hand_n = sum(by_class[c]["n"] for c in hand_classes)

    summary = {
        "root": None,
        "total_images": total_n,
        "total_detected": total_found,
        "detection_rate_overall": round(total_found / total_n, 4) if total_n else 0,
        "detection_rate_hand_classes": (
            round(hand_found / hand_n, 4) if hand_n else 0
        ),
        "hand_class_detection": {
            c: {"detected": by_class[c]["found"], "n": by_class[c]["n"],
                "rate": round(by_class[c]["found"] / by_class[c]["n"], 4)
                        if by_class[c]["n"] else None}
            for c in hand_classes
        },
        "nothing_detection": {
            "detected": nothing["found"], "n": nothing["n"],
            "rate": round(nothing["found"] / nothing["n"], 4)
                    if nothing["n"] else None,
        },
        "global_framing": {
            "L": _agg(global_L), "R": _agg(global_R),
            "T": _agg(global_T), "B": _agg(global_B),
            "hand_center_x": _agg(global_cx),
            "hand_center_y": _agg(global_cy),
            "hand_w_over_img_w": _agg(global_hw),
            "hand_h_over_img_h": _agg(global_hh),
        },
        "per_class_framing": {},
    }

    for c in CLASSES:
        bc = by_class[c]
        if bc["found"] == 0:
            summary["per_class_framing"][c] = {
                "detected": 0, "n": bc["n"], "rate": None,
                "L": None, "R": None, "T": None, "B": None,
                "hand_center_x": None, "hand_center_y": None,
                "hand_w_over_img_w": None, "hand_h_over_img_h": None,
            }
            continue
        summary["per_class_framing"][c] = {
            "detected": bc["found"], "n": bc["n"],
            "rate": round(bc["found"] / bc["n"], 4) if bc["n"] else None,
            "L": _agg(bc["L"]), "R": _agg(bc["R"]),
            "T": _agg(bc["T"]), "B": _agg(bc["B"]),
            "hand_center_x": _agg(bc["cx"]),
            "hand_center_y": _agg(bc["cy"]),
            "hand_w_over_img_w": _agg(bc["hw"]),
            "hand_h_over_img_h": _agg(bc["hh"]),
        }

    return summary


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main() -> None:
    p = argparse.ArgumentParser(
        description="Full-dataset training framing analysis (L/R/T/B as bbox ratios).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--root", required=True,
                   help="Path to asl_alphabet_train/ (29 class folders)")
    p.add_argument("--min-detection", type=float, default=0.5,
                   help="MediaPipe min detection confidence (matches runtime)")
    p.add_argument("--cache",
                   default="data/training_framing_all_records.json",
                   help="Per-image bbox cache (JSON, {path: [x0,y0,x1,y1]|None})")
    p.add_argument("--out",
                   default="data/training_framing_all_summary.json",
                   help="Aggregate summary output")
    args = p.parse_args()

    paths = all_training_paths(args.root)
    print(f"training images: {len(paths)} "
          f"({len(set(c for c, _ in paths))} classes)")

    t0 = time.time()
    det_cache = detection_pass(paths, args.cache, args.min_detection)
    print(f"detection pass: {time.time() - t0:.1f}s")

    summary = aggregate(paths, det_cache)
    summary["root"] = args.root

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"\nwrote {args.out}")

    # ---- pretty-print key stats ---- #
    g = summary["global_framing"]
    print("\n=== GLOBAL FRAMING (all detected hand-class images) ===")
    print(f"detected: {summary['total_detected']}/{summary['total_images']} "
          f"({100 * summary['detection_rate_overall']:.1f}%)")
    print(f"hand-class detection: "
          f"{summary['total_detected']}/{sum(v['n'] for v in summary['hand_class_detection'].values())} "
          f"({100 * summary['detection_rate_hand_classes']:.1f}%)")
    print(f"\n{'metric':>22} {'mean':>8} {'median':>8} {'p10':>8} {'p25':>8} "
          f"{'p75':>8} {'p90':>8}  n")
    for key, label in [("L", "L (left/bw)"), ("R", "R (right/bw)"),
                       ("T", "T (top/bh)"),   ("B", "B (bottom/bh)")]:
        a = g[key]
        print(f"{label:>22} {a['mean']:>8.4f} {a['median']:>8.4f} "
              f"{a['p10']:>8.4f} {a['p25']:>8.4f} {a['p75']:>8.4f} "
              f"{a['p90']:>8.4f}  {a['n']}")
    print(f"\n{'hand centre X':>22} {g['hand_center_x']['mean']:>8.4f} "
          f"{g['hand_center_x']['median']:>8.4f}")
    print(f"{'hand centre Y':>22} {g['hand_center_y']['mean']:>8.4f} "
          f"{g['hand_center_y']['median']:>8.4f}")
    print(f"{'hand w/img w':>22} {g['hand_w_over_img_w']['mean']:>8.4f} "
          f"{g['hand_w_over_img_w']['median']:>8.4f}")
    print(f"{'hand h/img h':>22} {g['hand_h_over_img_h']['mean']:>8.4f} "
          f"{g['hand_h_over_img_h']['median']:>8.4f}")

    print("\nper-class detection + median framing:")
    print(f"{'cls':>8} {'det':>5} {'n':>5} {'rate':>6}  "
          f"{'med_L':>7} {'med_R':>7} {'med_T':>7} {'med_B':>7}")
    for c in CLASSES:
        pc = summary["per_class_framing"][c]
        if pc["detected"] == 0:
            print(f"{c:>8} {pc['detected']:>5} {pc['n']:>5} {'--':>6}")
            continue
        Lm = pc["L"]["median"] if pc["L"] else float("nan")
        Rm = pc["R"]["median"] if pc["R"] else float("nan")
        Tm = pc["T"]["median"] if pc["T"] else float("nan")
        Bm = pc["B"]["median"] if pc["B"] else float("nan")
        rate = pc["rate"] or 0.0
        print(f"{c:>8} {pc['detected']:>5} {pc['n']:>5} {100*rate:5.1f}%  "
              f"{Lm:>7.3f} {Rm:>7.3f} {Tm:>7.3f} {Bm:>7.3f}")


if __name__ == "__main__":
    main()
