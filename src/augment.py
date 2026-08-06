"""Augmentation pipeline + background replacement.

Heavy augmentation is the single-signer antidote. The training images are one
person, one room, one light; to survive on a stranger's webcam the model must see
rotation, shift, scale, perspective, colour jitter, noise/blur, cutout, and — most
importantly — different backgrounds behind the same hand.

Both the train and val transforms END with the SAME deterministic
resize+normalize defined in ``src/crop.py``. Augmentation only ever adds random
ops BEFORE that shared tail; it never changes the final framing/normalisation, so
train and serving stay in parity. See ``tests/test_preprocessing_parity.py``.

Two policies (#12)
------------------
``default`` is the heavy pipeline above. ``geometry_safe`` softens only the ops
that move pixels — rotation, shear, perspective, scale — and the cutout that can
erase a whole finger, while leaving every photometric op at full strength. It
exists because the first transfer run (#6, ``docs/RESULTS.md``) put 86% of its
errors on three classes whose identity IS a small geometric detail: V vs K is
where the thumb sits, S vs E is how far the thumb folds over the fist, X vs
A/L/R is how far the index finger hooks. Those are precisely the details ±20°
rotation, ±8° shear and a 15%-of-frame dropout hole are licensed to destroy
while the label stays put — teaching the model that the distinguishing feature
is noise. Colour, gamma, blur and noise cannot move a thumb, so they keep full
strength; they are what the webcam set will actually demand.
"""

from __future__ import annotations

import argparse
import glob
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

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


@dataclass(frozen=True)
class AugPolicy:
    """Strength of the pixel-moving ops. Photometric strength is policy-independent.

    Only the geometry-destroying knobs live here, because they are the only ones
    that can change what a fingerspelling handshape *means*. See the module
    docstring for why ``geometry_safe`` exists.
    """

    name: str
    rotate: float
    translate: float
    scale: Tuple[float, float]
    shear: float
    perspective_scale: Tuple[float, float]
    perspective_p: float
    dropout_holes: Tuple[int, int]
    dropout_size: Tuple[float, float]
    dropout_p: float


POLICIES: Dict[str, AugPolicy] = {
    # The heavy single-signer antidote (#9) — unchanged from the #6 run.
    "default": AugPolicy(
        name="default",
        rotate=20.0, translate=0.08, scale=(0.85, 1.15), shear=8.0,
        perspective_scale=(0.02, 0.08), perspective_p=0.3,
        dropout_holes=(4, 6), dropout_size=(0.05, 0.15), dropout_p=0.4,
    ),
    # Same photometric strength, gentler geometry, and a dropout that can no
    # longer swallow a thumb: 6 holes at up to 15% of the frame each can cover
    # the one region that separates V from K.
    "geometry_safe": AugPolicy(
        name="geometry_safe",
        rotate=8.0, translate=0.05, scale=(0.92, 1.08), shear=3.0,
        perspective_scale=(0.02, 0.04), perspective_p=0.15,
        dropout_holes=(1, 2), dropout_size=(0.04, 0.08), dropout_p=0.25,
    ),
}


def resolve_policy(policy) -> AugPolicy:
    """Accept a policy name or an ``AugPolicy``; reject unknown names loudly."""
    if isinstance(policy, AugPolicy):
        return policy
    try:
        return POLICIES[policy]
    except KeyError:
        raise ValueError(
            f"unknown augmentation policy {policy!r}; "
            f"choose from {sorted(POLICIES)}"
        ) from None


def build_augmentation(policy: str = "default"):
    """The Albumentations pipeline for one policy, as an ``A.Compose``.

    Separate from ``build_transforms`` so the composed pipeline can be inspected —
    a policy whose fields never reach the transforms would still produce plausible
    augmented images, so the wiring is checked against the transforms themselves
    (``tests/test_aimed_augmentation.py``) rather than inferred from pixels.
    """
    import albumentations as A

    pol = resolve_policy(policy)

    return A.Compose(
        [
            # --- geometric (policy-controlled) ---
            A.Affine(
                rotate=(-pol.rotate, pol.rotate),
                translate_percent=(-pol.translate, pol.translate),
                scale=pol.scale,
                shear=(-pol.shear, pol.shear),
                fit_output=False,
                p=0.9,
            ),
            A.Perspective(scale=pol.perspective_scale, p=pol.perspective_p),
            # --- photometric (identical across policies: cannot move a thumb) ---
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
            # --- structural (policy-controlled) ---
            A.CoarseDropout(
                num_holes_range=pol.dropout_holes,
                hole_height_range=pol.dropout_size,
                hole_width_range=pol.dropout_size, fill=0, p=pol.dropout_p
            ),
        ]
    )


def build_transforms(
    train: bool = True,
    backgrounds_dir: Optional[str] = None,
    policy: str = "default",
) -> Callable:
    """Return a callable ``img_bgr -> torch.FloatTensor`` (C×H×W).

    When ``train`` is False, only the deterministic resize+normalize contract runs
    (no randomness) — this is the val/serve path, and ``policy`` is irrelevant
    there. When ``train`` is True, an Albumentations pipeline of geometric +
    photometric + structural augmentations runs first, optionally preceded by
    background replacement. ``policy`` selects the geometric strength (#12).
    """
    if not train:
        return lambda img_bgr: _to_tensor(img_bgr)

    aug = build_augmentation(policy)
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

    transform = build_transforms(
        train=True, backgrounds_dir=args.backgrounds, policy=args.policy
    )
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
    print(f"wrote {args.n} augmented variants ({args.policy}) -> {args.out}")


def main() -> None:
    p = argparse.ArgumentParser(description="Preview the augmentation pipeline.")
    p.add_argument("--image", required=True, help="a single training image")
    p.add_argument("--backgrounds", default=None, help="dir of background images")
    p.add_argument("--policy", choices=sorted(POLICIES), default="default",
                   help="geometric strength; 'geometry_safe' for the confusable "
                        "handshapes (#12)")
    p.add_argument("--n", type=int, default=8)
    p.add_argument("--out", default="docs/figures/augment_preview.png")
    _preview(p.parse_args())


if __name__ == "__main__":
    main()
