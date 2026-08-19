"""Build a training-vs-webcam framing comparison montage (#17, Phase 2).

The point is human inspection: numeric bbox statistics cannot prove the webcam
crop framing matches the training framing. This script puts the two side by side
so the *eye* can judge hand/image size ratio, centring, surrounding context, and
wrist/arm visibility.

Pipeline: record webcam crops with the app first, then assemble:

    python -m app.webcam_speller --checkpoint checkpoints/efficientnet_b0_targeted.pt \\
        --save-crops data/webcam_crops --save-crops-max 24

    python scripts/build_framing_comparison.py \\
        --training-root data/raw/asl_alphabet_train \\
        --webcam-crops data/webcam_crops \\
        --out docs/figures/framing_comparison/training_vs_webcam.png

Output is a grid with one row per pair: left = training image (with its
MediaPipe tight hand box drawn), right = the webcam crop saved by the app.

Uses the app's real crop function (``crop_square_with_margin``), so what the
montage shows is exactly what the classifier sees at inference time.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.crop import crop_square_with_margin, resize_square  # noqa: E402
from src.data import CLASSES, list_class_files  # noqa: E402

# How the pair grid labels each cell.
DISPLAY = 224


def _training_samples(root: str, n: int, seed: int) -> List[Tuple[str, str]]:
    """``n`` training images spread across classes (one per class when possible)."""
    import random

    rng = random.Random(seed)
    out: List[Tuple[str, str]] = []
    classes = list(CLASSES)
    rng.shuffle(classes)
    per_class = max(1, n // len(classes))
    for cls in classes:
        files = list_class_files(root, cls)
        if not files:
            continue
        rng.shuffle(files)
        out += [(cls, f) for f in files[:per_class]]
        if len(out) >= n:
            break
    return out[:n]


def _webcam_crops(directory: str) -> List[str]:
    d = Path(directory)
    if not d.is_dir():
        return []
    files = sorted(p for p in d.iterdir()
                   if p.suffix.lower() in (".jpg", ".jpeg", ".png")
                   and p.stem.startswith("crop_"))
    return [str(p) for p in files]


def build_montage(
    training_root: str,
    webcam_crops_dir: str,
    n: int,
    seed: int,
    draw_boxes: bool,
) -> Optional[np.ndarray]:
    train = _training_samples(training_root, n, seed)
    crops = _webcam_crops(webcam_crops_dir)
    n_pairs = min(len(train), len(crops))
    if n_pairs == 0:
        print("nothing to show: need >= 1 training image and >= 1 webcam crop "
              f"(found {len(train)} training samples, {len(crops)} webcam crops)")
        return None

    from src.crop import HandCropper  # local import (mediapipe)

    gap = 24
    strip_h = 30
    cell_w = DISPLAY + gap + DISPLAY
    rows = []
    with HandCropper() as cropper:
        for i, (cls, tpath) in enumerate(train[:n_pairs]):
            t_img = cv2.imread(tpath, cv2.IMREAD_COLOR)
            w_path = crops[i]
            w_crop = cv2.imread(w_path, cv2.IMREAD_COLOR)
            if t_img is None or w_crop is None:
                continue
            tb = cropper.tight_bbox(t_img)
            if tb is not None and draw_boxes:
                x0, y0, x1, y1 = tb
                cv2.rectangle(t_img, (x0, y0), (x1, y1), (255, 0, 0), 1)
            # Both cells shown at the same display size so ratios are comparable.
            t_disp = resize_square(t_img, DISPLAY)
            w_disp = resize_square(w_crop, DISPLAY)

            row = np.zeros((DISPLAY + strip_h, cell_w, 3), dtype=np.uint8)
            row[strip_h : strip_h + DISPLAY, 0:DISPLAY] = t_disp
            row[strip_h : strip_h + DISPLAY, DISPLAY + gap :] = w_disp
            t_tag = f"{cls} {Path(tpath).name}"
            if tb is None:
                t_tag += " (no hand)"
            cv2.putText(row, t_tag, (4, 20), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 255, 255), 1)
            cv2.putText(row, f"webcam {Path(w_path).name}", (DISPLAY + gap + 4, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            rows.append(row)

    if not rows:
        print("no readable pairs assembled")
        return None
    return np.vstack(rows)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Side-by-side training vs webcam-crop framing comparison (#17).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=("Record webcam crops first with:\n"
                "  python -m app.webcam_speller --checkpoint "
                "checkpoints/efficientnet_b0_targeted.pt \\\n"
                "      --save-crops data/webcam_crops --save-crops-max 24"),
    )
    p.add_argument("--training-root", required=True,
                   help="asl_alphabet_train/ (29 class folders)")
    p.add_argument("--webcam-crops", required=True,
                   help="dir written by the app's --save-crops (crop_XXXX.jpg)")
    p.add_argument("--out", default="docs/figures/framing_comparison/"
                                    "training_vs_webcam.png")
    p.add_argument("--n", type=int, default=8, help="number of training/webcam pairs")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-boxes", action="store_true",
                   help="skip drawing the MediaPipe tight hand box on training images")
    args = p.parse_args()

    montage = build_montage(args.training_root, args.webcam_crops, args.n,
                            args.seed, draw_boxes=not args.no_boxes)
    if montage is None:
        raise SystemExit(1)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(out), montage)
    if not ok:
        raise SystemExit(f"failed to write {out}")
    print(f"wrote {out}  ({montage.shape[1]}x{montage.shape[0]})")


if __name__ == "__main__":
    main()
