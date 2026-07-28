"""Proof that training and the webcam app preprocess pixels identically.

If the training pipeline and the live app resize/normalize differently, the model
silently loses accuracy at serving time with no error to point at. Both sides must
route through ``src.crop.preprocess_bgr`` (resize_square -> normalize_chw). These
tests pin that contract:

  1. the deterministic contract is stable (same input -> same output),
  2. the training val transform equals the app's ``preprocess_bgr`` byte-for-byte,
  3. output shape/normalization statistics match the model's expectations.

torch is required only for test (2); it is skipped if torch isn't installed.
"""

from __future__ import annotations

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from src.crop import (  # noqa: E402
    IMAGENET_MEAN,
    IMAGENET_STD,
    IMG_SIZE,
    normalize_chw,
    preprocess_bgr,
    resize_square,
)


def _sample_image(h=200, w=200):
    rng = np.random.default_rng(1234)
    return rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)


def test_contract_is_deterministic():
    img = _sample_image()
    a = preprocess_bgr(img.copy())
    b = preprocess_bgr(img.copy())
    assert np.array_equal(a, b)


def test_contract_output_shape_and_dtype():
    out = preprocess_bgr(_sample_image())
    assert out.shape == (3, IMG_SIZE, IMG_SIZE)
    assert out.dtype == np.float32


def test_resize_square_is_idempotent_on_square_input():
    img = _sample_image(IMG_SIZE, IMG_SIZE)
    assert np.array_equal(resize_square(img), img)


def test_non_square_input_is_letterboxed_not_stretched():
    img = _sample_image(100, 200)  # 1:2 aspect
    out = resize_square(img)
    assert out.shape[:2] == (IMG_SIZE, IMG_SIZE)
    # letterbox => some fully-black padding rows exist (top/bottom)
    row_is_black = (out.reshape(IMG_SIZE, -1) == 0).all(axis=1)
    assert row_is_black.any()


def test_normalization_uses_imagenet_stats():
    # A mid-grey image: normalized value == (0.5 - mean)/std per channel.
    grey = np.full((IMG_SIZE, IMG_SIZE, 3), 127, dtype=np.uint8)
    out = normalize_chw(grey)  # C×H×W, RGB order
    v = 127.0 / 255.0
    expected = [(v - m) / s for m, s in zip(IMAGENET_MEAN, IMAGENET_STD)]
    for c in range(3):
        assert np.allclose(out[c], expected[c], atol=1e-5)


def test_train_val_transform_matches_app_preprocessing():
    """The training val transform MUST equal the app's preprocess_bgr exactly."""
    torch = pytest.importorskip("torch")
    from src.augment import build_transforms

    img = _sample_image()
    val_transform = build_transforms(train=False)
    train_side = val_transform(img).numpy()      # what the model trains/validates on
    app_side = preprocess_bgr(img)               # what the webcam app feeds the model
    assert train_side.shape == app_side.shape
    assert np.array_equal(train_side, app_side), (
        "training preprocessing diverged from the app — silent accuracy loss"
    )


def test_train_transform_still_ends_in_contract_shape():
    """Even with augmentation, the train tensor has the contract shape."""
    pytest.importorskip("torch")
    pytest.importorskip("albumentations")
    from src.augment import build_transforms

    t = build_transforms(train=True)
    out = t(_sample_image()).numpy()
    assert out.shape == (3, IMG_SIZE, IMG_SIZE)
    assert out.dtype == np.float32
