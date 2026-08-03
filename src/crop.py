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

    def hand_bbox(self, frame_bgr: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """Return a margin-padded square bbox around the first detected hand.

        Returns None if no hand is detected.
        """
        import mediapipe as mp

        h, w = frame_bgr.shape[:2]
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        result = self._landmarker.detect(mp_img)
        if not result.hand_landmarks:
            return None

        lm = result.hand_landmarks[0]
        xs = [p.x * w for p in lm]
        ys = [p.y * h for p in lm]
        x0, x1 = min(xs), max(xs)
        y0, y1 = min(ys), max(ys)
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
    """Expand a bbox to a square, add ``margin`` on each side, clamp to image."""
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    side = max(x1 - x0, y1 - y0)
    side *= 1.0 + 2.0 * margin
    half = side / 2.0
    nx0 = int(max(0, round(cx - half)))
    ny0 = int(max(0, round(cy - half)))
    nx1 = int(min(w, round(cx + half)))
    ny1 = int(min(h, round(cy + half)))
    return nx0, ny0, nx1, ny1


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
