"""Synthesize demo "webcam" frames for the training-vs-webcam framing montage.

The real comparison artifact is made from crops recorded with
``python -m app.webcam_speller --save-crops`` (see scripts/build_framing_comparison.py).
This script is the hardware-free proxy: it takes real training images, pastes
their hand region into a larger synthetic 640x480 background at varied scale and
position (the "webcam framing" the model has never seen), then runs that frame
through the EXACT app crop path (``HandCropper.tight_bbox`` + margin 0.62) and
saves the resulting 224x224 crops as ``crop_XXXX.jpg``.

The point is to demonstrate that MediaPipe localization + a training-derived
margin turns arbitrary framing into training-like framing — hand occupying
~45% of the crop, roughly centred — before real webcam captures exist.

Usage::

    python scripts/make_demo_webcam_crops.py \\
        --training-root data/raw/asl_alphabet_train \\
        --out data/webcam_crops_demo --n 12
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

from src.crop import HandCropper, crop_square_with_margin  # noqa: E402
from src.data import CLASSES, list_class_files  # noqa: E402

FRAME_W, FRAME_H = 640, 480


def _backgrounds(root: str, n: int) -> List[np.ndarray]:
    """Real room backgrounds: the 'nothing' class is an empty scene. Upscaled
    to webcam size so a pasted hand occupies a realistic fraction of the frame."""
    files = list_class_files(root, "nothing")
    if not files:
        raise SystemExit("no 'nothing' class found under --training-root; it is "
                         "needed as a handless background")
    out = []
    for i in range(min(n, len(files))):
        img = cv2.imread(files[int(i * len(files) / n)], cv2.IMREAD_COLOR)
        if img is not None:
            out.append(cv2.resize(img, (FRAME_W, FRAME_H),
                                  interpolation=cv2.INTER_LINEAR))
    return out


def _paste_hand(bg: np.ndarray, hand_bgr: np.ndarray, bbox, scale: float,
                pos_x: float, pos_y: float) -> np.ndarray:
    """Paste a training hand into a background with a feathered boundary.

    Returns the composited frame. ``pos_x/pos_y`` are fractions of the frame
    where the hand centre goes (0.5/0.5 = centred). Feathering matters: a hard
    rectangle around the hand is a shape the landmarker latches onto.
    """
    x0, y0, x1, y1 = bbox
    hand = hand_bgr[y0:y1, x0:x1].copy()
    h, w = hand.shape[:2]
    hw, hh = int(w * scale), int(h * scale)
    if hw < 8 or hh < 8:
        raise ValueError("hand too small after scaling")
    hand = cv2.resize(hand, (hw, hh), interpolation=cv2.INTER_AREA)

    frame = bg.copy()
    cx = int(pos_x * frame.shape[1])
    cy = int(pos_y * frame.shape[0])
    tx = max(0, min(cx - hw // 2, frame.shape[1] - hw))
    ty = max(0, min(cy - hh // 2, frame.shape[0] - hh))

    mask = np.zeros((hh, hw), dtype=np.float32)
    cv2.ellipse(mask, (hw // 2, hh // 2), (hw // 2, hh // 2),
                0, 0, 360, 1.0, -1)
    mask = cv2.GaussianBlur(mask, (0, 0), 8.0)
    m3 = mask[:, :, None]
    roi = frame[ty : ty + hh, tx : tx + hw].astype(np.float32)
    frame[ty : ty + hh, tx : tx + hw] = (
        roi * (1 - m3) + hand.astype(np.float32) * m3
    ).astype(np.uint8)
    return frame


def _pick_training(root: str, n: int) -> List[Tuple[str, str]]:
    """``n`` candidate paths, spread across classes (may exceed n; detection and
    synthetic-frame localization both drop some, so the loop retries)."""
    out = []
    per_class = max(1, n // len(CLASSES))
    for cls in CLASSES:
        files = list_class_files(root, cls)
        if not files:
            continue
        take = min(len(files), per_class * 3)
        stride = len(files) / per_class
        for i in range(take):
            out.append((cls, files[min(len(files) - 1, int(i * stride))]))
    return out[: n * 3]


def _synthetic_frame(hand_bgr: np.ndarray, bbox, scale: float, pos_x: float,
                     pos_y: float) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """Paste a training hand into a 640x480 background at ``scale``/position.

    Returns ``(frame, tight_bbox_in_frame)``. ``pos_x/pos_y`` are fractions of
    the frame where the hand centre goes (0.5/0.5 = centred).
    """
    rng = np.random.default_rng(0)
    bg = rng.integers(180, 210, (FRAME_H, FRAME_W, 3), dtype=np.uint8)
    # slight lighting gradient so the background is not flat
    ramp = np.linspace(0, 40, FRAME_W, dtype=np.uint8)
    bg += ramp[None, :, None].astype(np.uint8)

    x0, y0, x1, y1 = bbox
    hand = hand_bgr[y0:y1, x0:x1].copy()
    h, w = hand.shape[:2]
    hw, hh = int(w * scale), int(h * scale)
    if hw < 8 or hh < 8:
        raise ValueError("hand too small after scaling")
    hand = cv2.resize(hand, (hw, hh), interpolation=cv2.INTER_AREA)

    cx = int(pos_x * FRAME_W)
    cy = int(pos_y * FRAME_H)
    tx = max(0, min(cx - hw // 2, FRAME_W - hw))
    ty = max(0, min(cy - hh // 2, FRAME_H - hh))
    # use the alpha-ignored paste: replace the background rectangle
    frame = bg.copy()
    roi = frame[ty : ty + hh, tx : tx + hw]
    roi[:] = hand
    frame[ty : ty + hh, tx : tx + hw] = roi
    return frame, (tx, ty, tx + hw, ty + hh)


def main() -> None:
    p = argparse.ArgumentParser(description="Demo webcam-frame crops (#17).")
    p.add_argument("--training-root", required=True,
                   help="asl_alphabet_train/ (29 class folders)")
    p.add_argument("--out", default="data/webcam_crops_demo")
    p.add_argument("--n", type=int, default=12)
    p.add_argument("--crop-margin", type=float, default=0.62)
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    picks = _pick_training(args.training_root, args.n)
    backgrounds = _backgrounds(args.training_root, max(1, args.n))
    scales = [1.2, 1.6, 2.0]          # hand smaller/larger than training
    positions = [(0.5, 0.5), (0.35, 0.45), (0.65, 0.55), (0.5, 0.62)]
    written = 0
    with HandCropper() as cropper:
        for i, (cls, path) in enumerate(picks):
            if written >= args.n:
                break
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            if img is None:
                continue
            tb = cropper.tight_bbox(img)
            if tb is None:
                continue
            scale = scales[i % len(scales)]
            px, py = positions[i % len(positions)]
            bg = backgrounds[i % len(backgrounds)].copy()
            frame = _paste_hand(bg, img, tb, scale, px, py)
            # The app's real path: locate the hand in the NEW framing and crop.
            tb2 = cropper.tight_bbox(frame)
            if tb2 is None:
                continue
            crop = crop_square_with_margin(frame, tb2, args.crop_margin)
            cv2.imwrite(str(out / f"crop_{written:04d}.jpg"), crop)
            cv2.imwrite(str(out / f"frame_{written:04d}.jpg"), frame)
            written += 1
    print(f"wrote {written} demo webcam crops -> {out}")


if __name__ == "__main__":
    main()
