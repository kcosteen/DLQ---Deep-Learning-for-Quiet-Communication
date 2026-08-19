"""Build the forearm-crop ablation gallery (original | symmetric | forearm).

Reads the per-frame records written by ``scripts/benchmark_forearm_crop.py`` and
assembles a human-inspection montage:

    ORIGINAL | SYMMETRIC CROP | FOREARM CROP

with rows sampled from three buckets:

  * ``sym_fail_fore_ok`` — symmetric misclassifies, forearm classifies correctly
  * ``sym_ok_fore_fail`` — symmetric classifies correctly, forearm misclassifies
  * ``random_detected``  — deterministic spread of detected frames

The purpose is human inspection of hand scale, wrist visibility, forearm length,
centring and context differences between the two crops — not an accuracy claim.

Usage::

    python scripts/benchmark_forearm_crop.py --device mps   # writes records
    python scripts/build_forearm_ablation_gallery.py \\
        --records data/forearm_crop_records.json \\
        --out docs/figures/framing_comparison/forearm_crop_ablation.png
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.crop import (  # noqa: E402
    DEFAULT_ASYMMETRIC_MARGINS,
    crop_square_with_forearm_context,
    crop_square_with_margin,
    resize_square,
)
from src.data import CLASSES  # noqa: E402

DISPLAY = 224
GAP = 24
STRIP_H = 34
LEFT_PAD = 8
RIGHT_PAD = 8
TOP_PAD = 44  # room for the column header row


def _name(i) -> str:
    return CLASSES[i]


def _bucket(rec: Dict, true_idx: int) -> str:
    if not rec["detected"]:
        return "undetected"
    ps, pf = rec.get("pred_symmetric"), rec.get("pred_forearm")
    if ps is None or pf is None:
        return "undetected"
    sym_ok = ps == true_idx
    fore_ok = pf == true_idx
    if not sym_ok and fore_ok:
        return "sym_fail_fore_ok"
    if sym_ok and not fore_ok:
        return "sym_ok_fore_fail"
    return "random_detected"  # any detected record qualifies


def _sample_records(recs: List[Dict], classes: List[str], n_per: int) -> Dict[str, List[Dict]]:
    by_bucket: Dict[str, List[Dict]] = {"sym_fail_fore_ok": [], "sym_ok_fore_fail": [],
                                        "random_detected": []}
    for rec in recs:
        if not rec["detected"]:
            continue
        b = _bucket(rec, classes.index(rec["class"]))
        if b in by_bucket:
            by_bucket[b].append(rec)
    for b in by_bucket:
        bucket = by_bucket[b]
        # deterministic spread, not a random draw
        stride = max(1, len(bucket) / n_per)
        by_bucket[b] = [bucket[int(i * stride)] for i in range(min(n_per, len(bucket)))]
    return by_bucket


def _cell(img_bgr: np.ndarray) -> np.ndarray:
    return resize_square(img_bgr, DISPLAY)


def build_gallery(
    records_path: str,
    n_per_bucket: int,
    crop_margin: float,
    out_path: str,
) -> None:
    payload = json.loads(Path(records_path).read_text())
    classes = payload.get("classes", list(CLASSES))
    recs = payload["records"]
    picked = _sample_records(recs, classes, n_per_bucket)

    cell_w = DISPLAY
    row_w = LEFT_PAD + cell_w + GAP + cell_w + GAP + cell_w + RIGHT_PAD
    rows: List[np.ndarray] = []
    n_rows = 0

    for bucket, bucket_recs in picked.items():
        for rec in bucket_recs:
            img = cv2.imread(rec["path"], cv2.IMREAD_COLOR)
            if img is None:
                continue
            tb = tuple(rec["bbox"])
            sym_crop = crop_square_with_margin(img, tb, crop_margin)
            fore_crop = crop_square_with_forearm_context(
                img, tb, DEFAULT_ASYMMETRIC_MARGINS)

            row = np.zeros((STRIP_H + DISPLAY, row_w, 3), dtype=np.uint8)
            row[STRIP_H:, LEFT_PAD:LEFT_PAD + cell_w] = _cell(img)
            x = LEFT_PAD + cell_w + GAP
            row[STRIP_H:, x:x + cell_w] = _cell(sym_crop)
            x = LEFT_PAD + 2 * (cell_w + GAP)
            row[STRIP_H:, x:x + cell_w] = _cell(fore_crop)

            true_name = rec["class"]
            ps, pf = rec.get("pred_symmetric"), rec.get("pred_forearm")
            sym_name = _name(ps) if ps is not None else "-"
            fore_name = _name(pf) if pf is not None else "-"
            tag = (f"{bucket}  true={true_name}  "
                   f"sym={sym_name}  fore={fore_name}   {Path(rec['path']).name}")
            cv2.putText(row, tag, (LEFT_PAD, 22), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 255, 255), 1)
            rows.append(row)
            n_rows += 1

    if not rows:
        raise SystemExit("no records rendered; run benchmark_forearm_crop.py first")

    header = np.zeros((TOP_PAD, row_w, 3), dtype=np.uint8)
    titles = ["ORIGINAL", "SYMMETRIC CROP", "FOREARM CROP"]
    for i, t in enumerate(titles):
        x = LEFT_PAD + i * (cell_w + GAP)
        cv2.putText(header, t, (x, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 255, 0), 2)
    canvas = np.vstack([header] + rows)

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(out), canvas)
    if not ok:
        raise SystemExit(f"failed to write {out}")
    counts = {b: len(v) for b, v in picked.items()}
    print(f"wrote {out}  ({canvas.shape[1]}x{canvas.shape[0]})  rows per bucket: {counts}")


def main() -> None:
    p = argparse.ArgumentParser(
        description="Ablation gallery: ORIGINAL | SYMMETRIC | FOREARM crop.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--records", default="data/forearm_crop_records.json")
    p.add_argument("--out",
                   default="docs/figures/framing_comparison/forearm_crop_ablation.png")
    p.add_argument("--crop-margin", type=float, default=0.62,
                   help="symmetric margin used when rendering the symmetric cell")
    p.add_argument("--n-per-bucket", type=int, default=4)
    args = p.parse_args()
    build_gallery(args.records, args.n_per_bucket, args.crop_margin, args.out)


if __name__ == "__main__":
    main()
