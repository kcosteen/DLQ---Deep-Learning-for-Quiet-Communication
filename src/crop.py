"""MediaPipe hand-crop bridge — the single shared preprocessing contract.

This module is imported by BOTH the training pipeline (``src/data.py`` /
``src/augment.py``) and the live webcam app (``app/webcam_speller.py``). The
whole point of the project depends on training framing and serving framing being
byte-for-byte identical, so all crop + resize + normalize logic lives here and
NOWHERE else. If you change a constant in this file, you change it for training
and inference at the same time — that is deliberate.

The parity is enforced by ``tests/test_preprocessing_parity.py``.

The MediaPipe crop step is mandatory: the 87k Kaggle training images are a single
signer in a single room. Cropping tightly to the detected hand bounding box is
what lets a model trained on that data generalise to a stranger's hand on a
different webcam.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

# MediaPipe Tasks model used by HandCropper. mediapipe>=1.0 removed the legacy
# `mp.solutions` API and requires an explicit model file (not bundled in the
# wheel). This file is committed to the repo; see models/README.md.
MODELS_DIR: Path = Path(__file__).resolve().parent.parent / "models"
HAND_LANDMARKER_MODEL: Path = MODELS_DIR / "hand_landmarker.task"

# =========================================================================== #
# THE PREPROCESSING CONTRACT — FROZEN. Do not change these values.
#
# These four constants define the exact tensor the network sees in training and
# in the live webcam app. Changing ANY of them silently invalidates every trained
# checkpoint, so this block is FROZEN as of Week 1 (issue #4).
#
# CHANGE-CONTROL RULE: any change requires sign-off from Person B (training) and
# a full re-run of tests/test_preprocessing_parity.py. Treat disagreement here as
# a blocker, not a tweak. See docs/PREPROCESSING_CONTRACT.md.
# =========================================================================== #
IMG_SIZE: int = 224
# ImageNet statistics (RGB order). timm/EfficientNet-B0 was pretrained with these.
IMAGENET_MEAN: Tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: Tuple[float, float, float] = (0.229, 0.224, 0.225)
# Fraction of the hand bbox side added as margin on every edge before cropping.
BBOX_MARGIN: float = 0.25


@dataclass(frozen=True)
class AsymmetricMargins:
    """Per-side context margins for the forearm-preserving crop.

    ``left``/``right`` are fractions of the tight bbox WIDTH, ``top``/``bottom``
    fractions of the tight bbox HEIGHT. The defaults (0.7/0.7/0.4/1.8) put far
    more context below the hand than above so a webcam crop keeps the forearm
    segment the original 200x200 dataset frames show. These are experiment
    defaults for the forearm-crop ablation, NOT frozen contract constants.
    """

    left: float = 0.7
    right: float = 0.7
    top: float = 0.4
    bottom: float = 1.8


DEFAULT_ASYMMETRIC_MARGINS: AsymmetricMargins = AsymmetricMargins()


@dataclass(frozen=True)
class TrainingFramingMargins:
    """Data-driven per-side margins derived from the training dataset.

    ``left``/``right`` are the *median* context-to-hand-bbox-width ratios
    measured across the full training set (``scripts/analyze_training_framing.py``).
    ``top``/``bottom`` are the corresponding height ratios.  These are *not*
    percentages of the image — they are multiples of the hand bbox width/height.

    Typical values from the Kaggle ASL 200x200 training set are close to
    L=R=0.5, T=0.5, B=0.5 (hand roughly centred with symmetric context),
    but the exact numbers come from the data and must never be hand-tuned.
    """

    left: float
    right: float
    top: float
    bottom: float


@dataclass
class CropResult:
    """Result of a hand-crop attempt on a single frame."""

    crop_bgr: np.ndarray  # H×W×3 uint8, square, resized to IMG_SIZE, BGR order
    bbox: Optional[Tuple[int, int, int, int]]  # x0, y0, x1, y1 in source pixels
    found_hand: bool  # False => MediaPipe found no hand; crop is a centre fallback


class HandCropper:
    """Wraps the MediaPipe Tasks HandLandmarker and produces a square hand crop.

    Uses the Tasks API (``mediapipe.tasks.python.vision.HandLandmarker``) — the
    legacy ``mp.solutions`` API was removed in mediapipe>=1.0. Runs in
    ``RunningMode.IMAGE`` (per-frame detection; no hand tracking), so the old
    ``static_image_mode`` / ``min_tracking_confidence`` knobs are gone. mediapipe
    is imported lazily so unit tests exercising the pure-numpy helpers (resize /
    normalize / letterbox) do not require the mediapipe wheel.
    """

    def __init__(
        self,
        max_num_hands: int = 1,
        min_detection_confidence: float = 0.5,
    ) -> None:
        # local import: keep tests light
        from mediapipe.tasks.python import vision
        from mediapipe.tasks.python.core.base_options import BaseOptions

        if not HAND_LANDMARKER_MODEL.is_file():
            raise FileNotFoundError(
                f"hand landmarker model not found at {HAND_LANDMARKER_MODEL}. "
                "Fetch it with the commands in models/README.md."
            )
        options = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(HAND_LANDMARKER_MODEL)),
            running_mode=vision.RunningMode.IMAGE,
            num_hands=max_num_hands,
            min_hand_detection_confidence=min_detection_confidence,
        )
        self._landmarker = vision.HandLandmarker.create_from_options(options)

    def hand_landmarks(
        self, frame_bgr: np.ndarray
    ) -> Optional[Tuple[List[float], List[float]]]:
        """First detected hand's landmarks as ``(xs, ys)`` in source pixels.

        Returns None if no hand is detected. Used by both ``tight_bbox`` and the
        framing-measurement tooling (Phase 1 / scripts/measure_training_framing.py)
        so the localization geometry has a single source of truth.
        """
        import mediapipe as mp

        h, w = frame_bgr.shape[:2]
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        result = self._landmarker.detect(mp_img)
        if not result.hand_landmarks:
            return None
        lm = result.hand_landmarks[0]
        return [p.x * w for p in lm], [p.y * h for p in lm]

    def tight_bbox(self, frame_bgr: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """Raw landmark min/max bbox in source pixels (no margin, not square).

        Returns None if no hand is detected. This is the tight hand box the
        webcam bridge expands with a configurable margin; ``hand_bbox`` wraps it
        with the fixed contract margin for backward compatibility.
        """
        det = self.hand_landmarks(frame_bgr)
        if det is None:
            return None
        xs, ys = det
        return (
            int(round(min(xs))), int(round(min(ys))),
            int(round(max(xs))), int(round(max(ys))),
        )

    def hand_bbox(self, frame_bgr: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """Return a margin-padded square bbox around the first detected hand.

        Returns None if no hand is detected. Uses the FROZEN ``BBOX_MARGIN``;
        the webcam bridge calls ``tight_bbox`` + ``crop_square_with_margin`` when
        it wants a configurable margin instead.
        """
        tb = self.tight_bbox(frame_bgr)
        if tb is None:
            return None
        x0, y0, x1, y1 = tb
        h, w = frame_bgr.shape[:2]
        return square_bbox_with_margin(x0, y0, x1, y1, w, h, BBOX_MARGIN)

    def crop(self, frame_bgr: np.ndarray) -> CropResult:
        """Crop a frame to the hand. Falls back to a centre square if no hand."""
        bbox = self.hand_bbox(frame_bgr)
        if bbox is None:
            crop = center_square(frame_bgr)
            return CropResult(resize_square(crop), None, found_hand=False)
        x0, y0, x1, y1 = bbox
        crop = frame_bgr[y0:y1, x0:x1]
        return CropResult(resize_square(crop), bbox, found_hand=True)

    def close(self) -> None:
        self._landmarker.close()

    def __enter__(self) -> "HandCropper":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


# --------------------------------------------------------------------------- #
# Pure functions — no mediapipe dependency. These define the resize/normalize
# half of the contract and are what the parity test pins down.
# --------------------------------------------------------------------------- #
def square_bbox_with_margin(
    x0: float, y0: float, x1: float, y1: float, w: int, h: int, margin: float
) -> Tuple[int, int, int, int]:
    """Expand a bbox to a square, add ``margin`` on each side, clamp to image.

    ``margin`` is the fraction of the tight bbox side added on every edge, e.g.
    ``0.25`` turns a 160-px hand box into a 160*1.5=240-px square crop. Clamping
    to the image keeps the box in-bounds but can make it non-square at the edges;
    ``resize_square`` handles that with a letterbox.
    """
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    side = max(x1 - x0, y1 - y0)
    side *= 1.0 + 2.0 * margin
    half = side / 2.0
    nx0 = int(max(0, round(cx - half)))
    ny0 = int(max(0, round(cy - half)))
    nx1 = int(min(w, round(cx + half)))
    ny1 = int(min(h, round(cy + half)))
    return nx0, ny0, nx1, ny1


def crop_square_with_margin(
    frame_bgr: np.ndarray,
    tight_bbox: Tuple[int, int, int, int],
    margin: float = BBOX_MARGIN,
) -> np.ndarray:
    """Configurable-margin square hand crop, bounds-safe, resized to IMG_SIZE.

    The live webcam bridge (#17) uses this instead of the fixed-margin
    ``HandCropper.crop`` so ``--crop-margin`` can reproduce the training framing
    (see scripts/measure_training_framing.py). ``tight_bbox`` is the raw landmark
    box from ``HandCropper.tight_bbox``. The output is a 224×224 BGR crop, ready
    for ``preprocess_bgr``.
    """
    x0, y0, x1, y1 = tight_bbox
    h, w = frame_bgr.shape[:2]
    bx0, by0, bx1, by1 = square_bbox_with_margin(x0, y0, x1, y1, w, h, margin)
    return resize_square(frame_bgr[by0:by1, bx0:bx1])


def asymmetric_context_bbox(
    frame_bgr: np.ndarray,
    tight_bbox: Tuple[int, int, int, int],
    margins: AsymmetricMargins = DEFAULT_ASYMMETRIC_MARGINS,
) -> Tuple[int, int, int, int]:
    """Requested per-side context box -> squared -> clamped to image bounds.

    The requested box is the tight hand bbox expanded by each side's own margin
    (left/right x bbox width, top/bottom x bbox height). It is made square by
    expanding the shorter side symmetrically around the requested box's centre —
    the requested box therefore stays FULLY inside the square (bottom/forearm
    context preserved) whenever the image has room, since the square side is
    ``max(requested_w, requested_h)``. Clamping to the image can trim the result
    (e.g. a hand near the bottom edge loses only context that does not exist in
    the frame) and can leave it non-square; ``resize_square`` letterboxes that,
    so the image is never stretched.
    """
    x0, y0, x1, y1 = tight_bbox
    bw, bh = x1 - x0, y1 - y0
    rx0 = x0 - margins.left * bw
    ry0 = y0 - margins.top * bh
    rx1 = x1 + margins.right * bw
    ry1 = y1 + margins.bottom * bh
    rw, rh = rx1 - rx0, ry1 - ry0
    side = max(rw, rh)
    cx, cy = (rx0 + rx1) / 2.0, (ry0 + ry1) / 2.0
    half = side / 2.0
    h, w = frame_bgr.shape[:2]
    nx0 = int(max(0, round(cx - half)))
    ny0 = int(max(0, round(cy - half)))
    nx1 = int(min(w, round(cx + half)))
    ny1 = int(min(h, round(cy + half)))
    return nx0, ny0, nx1, ny1


def crop_square_with_forearm_context(
    frame_bgr: np.ndarray,
    tight_bbox: Tuple[int, int, int, int],
    margins: AsymmetricMargins = DEFAULT_ASYMMETRIC_MARGINS,
) -> np.ndarray:
    """Asymmetric forearm-preserving hand crop, bounds-safe, resized to IMG_SIZE.

    The live webcam A/B comparison and the forearm-crop ablation use this instead
    of the symmetric ``crop_square_with_margin``: the top margin is kept small
    (0.4 x bbox height) while the bottom margin is large (1.8 x bbox height), so
    the wrist/forearm segment visible in the original 200x200 dataset frames is
    not cropped away. Output is a 224x224 BGR crop ready for ``preprocess_bgr``.
    """
    x0, y0, x1, y1 = asymmetric_context_bbox(frame_bgr, tight_bbox, margins)
    return resize_square(frame_bgr[y0:y1, x0:x1])


def training_calibrated_context_bbox(
    frame_bgr: np.ndarray,
    tight_bbox: Tuple[int, int, int, int],
    margins: TrainingFramingMargins,
) -> Tuple[int, int, int, int]:
    """Training-calibrated context box with smart shift-before-shrink square conversion.

    The requested training-like rectangle is constructed from the tight bbox
    expanded by each side's training-derived margin ratio (L x bw, R x bw,
    T x bh, B x bh).  This rectangle is then converted to a square that
    *contains* it, using shift-before-shrink boundary handling:

    1. If the requested square fits in the frame → no change.
    2. If one edge exceeds the frame, shift the square to stay within bounds
       while preserving the full side length whenever the frame is large enough.
    3. Only if the frame is physically smaller than the requested side length
       is the square shrunk — clamped to the frame dimensions.

    The centre of the *requested box* (not the hand bbox) is preserved by the
    shift, so the training-like framing is maintained as closely as possible.
    """
    x0, y0, x1, y1 = tight_bbox
    bw, bh = x1 - x0, y1 - y0
    h, w = frame_bgr.shape[:2]

    # Requested training-like rectangle
    rx0 = x0 - margins.left * bw
    ry0 = y0 - margins.top * bh
    rx1 = x1 + margins.right * bw
    ry1 = y1 + margins.bottom * bh
    rw, rh = rx1 - rx0, ry1 - ry0

    side = max(rw, rh)
    # Centre of the requested box
    cx = (rx0 + rx1) / 2.0
    cy = (ry0 + ry1) / 2.0
    half = side / 2.0

    # Initial square centred on requested box
    sx0 = cx - half
    sy0 = cy - half

    # --- shift before shrink ---
    # Horizontal: if the square exceeds a frame boundary, slide it back
    if sx0 < 0.0:
        sx0 = 0.0
    elif sx0 + side > w:
        sx0 = w - side

    # Vertical
    if sy0 < 0.0:
        sy0 = 0.0
    elif sy0 + side > h:
        sy0 = h - side

    # After shifting, if still out of bounds the frame is too small → shrink
    sx0 = max(0.0, sx0)
    sy0 = max(0.0, sy0)
    side = min(side, w, h)

    nx0 = int(round(sx0))
    ny0 = int(round(sy0))
    nx1 = int(round(sx0 + side))
    ny1 = int(round(sy0 + side))
    return nx0, ny0, nx1, ny1


def crop_square_with_training_framing(
    frame_bgr: np.ndarray,
    tight_bbox: Tuple[int, int, int, int],
    margins: TrainingFramingMargins,
) -> np.ndarray:
    """Training-calibrated hand crop, shift-before-shrink, resized to IMG_SIZE.

    The crop reproduces the framing the classifier was trained on, using
    data-derived L/R/T/B margin ratios.  Output is a 224x224 BGR crop
    ready for ``preprocess_bgr``.
    """
    x0, y0, x1, y1 = training_calibrated_context_bbox(
        frame_bgr, tight_bbox, margins)
    return resize_square(frame_bgr[y0:y1, x0:x1])


def center_square(frame_bgr: np.ndarray) -> np.ndarray:
    """Largest centred square crop of a frame (no-hand fallback)."""
    h, w = frame_bgr.shape[:2]
    side = min(h, w)
    y0 = (h - side) // 2
    x0 = (w - side) // 2
    return frame_bgr[y0 : y0 + side, x0 : x0 + side]


def resize_square(img_bgr: np.ndarray, size: int = IMG_SIZE) -> np.ndarray:
    """Resize any crop to ``size``×``size`` (letterbox to keep aspect ratio)."""
    if img_bgr.size == 0:
        return np.zeros((size, size, 3), dtype=np.uint8)
    h, w = img_bgr.shape[:2]
    scale = size / max(h, w)
    nh, nw = max(1, int(round(h * scale))), max(1, int(round(w * scale)))
    resized = cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    y0 = (size - nh) // 2
    x0 = (size - nw) // 2
    canvas[y0 : y0 + nh, x0 : x0 + nw] = resized
    return canvas


def normalize_chw(img_bgr: np.ndarray) -> np.ndarray:
    """BGR uint8 H×W×3 -> normalized RGB float32 C×H×W ready for the model.

    This is the exact tensor the network sees, in both training and serving.
    """
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    mean = np.asarray(IMAGENET_MEAN, dtype=np.float32)
    std = np.asarray(IMAGENET_STD, dtype=np.float32)
    rgb = (rgb - mean) / std
    return np.transpose(rgb, (2, 0, 1)).copy()


def preprocess_bgr(img_bgr: np.ndarray) -> np.ndarray:
    """Full deterministic contract: square-resize then normalize to C×H×W.

    ``img_bgr`` is assumed to already be a hand crop (or a full training image).
    Used by the parity test and by inference on already-cropped webcam images.
    """
    return normalize_chw(resize_square(img_bgr))


def _demo(args: argparse.Namespace) -> None:
    """Open the webcam and show live hand crops (visual smoke test of the crop)."""
    cap = cv2.VideoCapture(args.camera)
    with HandCropper(min_detection_confidence=args.min_confidence) as cropper:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            res = cropper.crop(frame)
            label = "hand" if res.found_hand else "no hand (center fallback)"
            if res.bbox is not None:
                x0, y0, x1, y1 = res.bbox
                cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 255, 0), 2)
            cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 255, 0), 2)
            cv2.imshow("frame", frame)
            cv2.imshow("crop", res.crop_bgr)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    cap.release()
    cv2.destroyAllWindows()


def main() -> None:
    p = argparse.ArgumentParser(description="MediaPipe hand-crop demo / contract.")
    p.add_argument("--camera", type=int, default=0, help="webcam index")
    p.add_argument("--min-confidence", type=float, default=0.5)
    _demo(p.parse_args())


if __name__ == "__main__":
    main()
