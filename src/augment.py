"""Augmentation pipeline + background replacement.

Heavy augmentation is the single-signer antidote. The training images are one
person, one room, one light; to survive on a stranger's webcam the model must see
rotation, shift, scale, perspective, colour jitter, noise/blur, cutout, and — most
importantly — different backgrounds behind the same hand.

Both the train and val transforms END with the SAME deterministic
resize+normalize defined in ``src/crop.py``. Augmentation only ever adds random
ops BEFORE that shared tail; it never changes the final framing/normalisation, so
train and serving stay in parity. See ``tests/test_preprocessing_parity.py``.
"""

from __future__ import annotations

import argparse
import glob
import os
import random
from pathlib import Path
from typing import Callable, List, Optional

import cv2
import numpy as np

from .crop import IMAGENET_MEAN, IMAGENET_STD, IMG_SIZE, preprocess_bgr, resize_square


def _to_tensor(img_bgr: np.ndarray):
    """BGR uint8 -> normalized C×H×W torch tensor via the SHARED contract.

    Uses ``crop.preprocess_bgr`` (resize_square + normalize_chw) — the exact same
    function the webcam app applies — so train/val/serve stay in parity. Torch is
    imported lazily so the pure split logic stays importable without a GPU stack.
    """
    import torch

    return torch.from_numpy(preprocess_bgr(img_bgr))


def build_transforms(train: bool = True, backgrounds_dir: Optional[str] = None) -> Callable:
    """Return a callable ``img_bgr -> torch.FloatTensor`` (C×H×W).

    When ``train`` is False, only the deterministic resize+normalize contract runs
    (no randomness) — this is the val/serve path. When ``train`` is True, an
    Albumentations pipeline of geometric + photometric + structural augmentations
    runs first, optionally preceded by background replacement.
    """
    if not train:
        return lambda img_bgr: _to_tensor(img_bgr)

    import albumentations as A

    aug = A.Compose(
        [
            # --- geometric ---
            A.Affine(
                rotate=(-20, 20),
                translate_percent=(-0.08, 0.08),
                scale=(0.85, 1.15),
                shear=(-8, 8),
                fit_output=False,
                p=0.9,
            ),
            A.Perspective(scale=(0.02, 0.08), p=0.3),
            # --- photometric ---
            A.RandomBrightnessContrast(0.3, 0.3, p=0.7),
            A.HueSaturationValue(15, 25, 15, p=0.5),
            A.RandomGamma(gamma_limit=(80, 120), p=0.4),
            A.OneOf(
                [
                    A.GaussNoise(std_range=(0.01, 0.05)),
                    A.GaussianBlur(blur_limit=(3, 5)),
                    A.MotionBlur(blur_limit=5),
                ],
                p=0.4,
            ),
            # --- structural ---
            A.CoarseDropout(
                num_holes_range=(4, 6), hole_height_range=(0.05, 0.15),
                hole_width_range=(0.05, 0.15), fill=0, p=0.4
            ),
        ]
    )

    bg_replacer = BackgroundReplacer(backgrounds_dir) if backgrounds_dir else None

    def _transform(img_bgr: np.ndarray):
        if bg_replacer is not None and random.random() < 0.5:
            img_bgr = bg_replacer(img_bgr)
        img_bgr = aug(image=img_bgr)["image"]
        return _to_tensor(img_bgr)

    return _transform


class BackgroundReplacer:
    """Segment the hand with MediaPipe and composite it over random backgrounds.

    Structure augmentation: the model must learn the hand shape, not the room. A
    directory of arbitrary background images is sampled and the segmented hand is
    pasted on top. Falls back to the original image if segmentation is unavailable.

    Uses the MediaPipe Tasks ``ImageSegmenter`` (legacy ``mp.solutions`` was
    removed in mediapipe>=1.0); requires ``models/selfie_segmenter.tflite``.
    """

    def __init__(self, backgrounds_dir: str) -> None:
        self.backgrounds: List[str] = [
            p
            for ext in ("*.jpg", "*.jpeg", "*.png")
            for p in glob.glob(os.path.join(backgrounds_dir, ext))
        ]
        self._seg = None  # lazy ImageSegmenter

    def _segmenter(self):
        if self._seg is None:
            from mediapipe.tasks.python import vision
            from mediapipe.tasks.python.core.base_options import BaseOptions

            model = Path(__file__).resolve().parent.parent / "models" / "selfie_segmenter.tflite"
            if not model.is_file():
                raise FileNotFoundError(
                    f"selfie segmenter model not found at {model}. "
                    "Fetch it with the commands in models/README.md."
                )
            self._seg = vision.ImageSegmenter.create_from_options(
                vision.ImageSegmenterOptions(
                    base_options=BaseOptions(model_asset_path=str(model)),
                    running_mode=vision.RunningMode.IMAGE,
                    output_confidence_masks=True,
                )
            )
        return self._seg

    def _mp_image(self, img_bgr: np.ndarray):
        """Wrap a BGR array as a MediaPipe SRGB ``Image``.

        Isolates the only bare ``import mediapipe`` in the composite path so the
        segmentation step is fully monkeypatchable (segmenter + this wrapper) and
        the tests can run without the mediapipe wheel installed.
        """
        import mediapipe as mp

        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        return mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    def __call__(self, img_bgr: np.ndarray) -> np.ndarray:
        # Graceful fallbacks (both keep the tests light and the training run
        # robust): no backgrounds configured, or an unreadable background file,
        # returns the input untouched. Read the background BEFORE segmenting so
        # this fallback never needs mediapipe.
        if not self.backgrounds:
            return img_bgr
        bg_path = random.choice(self.backgrounds)
        bg = cv2.imread(bg_path, cv2.IMREAD_COLOR)
        if bg is None:
            return img_bgr

        result = self._segmenter().segment(self._mp_image(img_bgr))
        mask = result.confidence_masks[0].numpy_view().squeeze()
        fg = (mask > 0.5)[..., None]

        bg = cv2.resize(bg, (img_bgr.shape[1], img_bgr.shape[0]))
        # Composite at the input's native resolution; the shared contract tail
        # (resize_square + normalize_chw, applied later) does the final framing.
        return np.where(fg, img_bgr, bg).astype(np.uint8)


def _preview(args: argparse.Namespace) -> None:
    """Write a grid of augmented variants of one image, for eyeballing."""
    img = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if img is None:
        raise SystemExit(f"could not read {args.image}")
    import albumentations as A  # noqa: F401  (ensures dependency present)

    transform = build_transforms(train=True, backgrounds_dir=args.backgrounds)
    tiles = []
    for _ in range(args.n):
        t = transform(img)  # torch tensor C×H×W (normalized)
        arr = t.numpy()
        arr = (arr * np.asarray(IMAGENET_STD)[:, None, None]
               + np.asarray(IMAGENET_MEAN)[:, None, None])
        arr = np.clip(np.transpose(arr, (1, 2, 0)) * 255, 0, 255).astype(np.uint8)
        tiles.append(cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))
    grid = np.hstack([resize_square(t, IMG_SIZE) for t in tiles])
    cv2.imwrite(args.out, grid)
    print(f"wrote {args.n} augmented variants -> {args.out}")


def main() -> None:
    p = argparse.ArgumentParser(description="Preview the augmentation pipeline.")
    p.add_argument("--image", required=True, help="a single training image")
    p.add_argument("--backgrounds", default=None, help="dir of background images")
    p.add_argument("--n", type=int, default=8)
    p.add_argument("--out", default="docs/figures/augment_preview.png")
    _preview(p.parse_args())


if __name__ == "__main__":
    main()
