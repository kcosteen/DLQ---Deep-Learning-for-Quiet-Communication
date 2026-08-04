"""Light tests for BackgroundReplacer — no mediapipe, no torch, no real segmentation.

Background replacement is a *structural* augmentation: composite the MediaPipe-
segmented hand over a random room so the model learns the hand, not the room. The
real op needs the mediapipe wheel + a tflite model, neither of which we want in the
unit-test path. These tests therefore:

  * exercise the graceful fallbacks (empty dir, unreadable background) that need no
    mediapipe at all,
  * assert the lazy-segmenter contract (construction imports nothing, builds nothing),
  * and verify the composite math by monkeypatching the segmenter + the mediapipe
    Image wrapper with fakes that yield a known confidence mask.

Only the last test (build_transforms shape) needs torch/albumentations; it is guarded
with importorskip. None of this weakens tests/test_preprocessing_parity.py.
"""

from __future__ import annotations

import sys

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from src.augment import BackgroundReplacer  # noqa: E402
from src.crop import IMG_SIZE  # noqa: E402


# --------------------------------------------------------------------------- #
# Fakes: stand in for the mediapipe ImageSegmenter without importing mediapipe.
# --------------------------------------------------------------------------- #
class _FakeMask:
    def __init__(self, mask: np.ndarray) -> None:
        self._mask = mask

    def numpy_view(self) -> np.ndarray:
        return self._mask


class _FakeResult:
    def __init__(self, mask: np.ndarray) -> None:
        self.confidence_masks = [_FakeMask(mask)]


class _FakeSegmenter:
    """Ignores its input and always returns the pre-baked confidence mask."""

    def __init__(self, mask: np.ndarray) -> None:
        self._mask = mask

    def segment(self, _mp_image) -> _FakeResult:
        return _FakeResult(self._mask)


def _install_fake_segmenter(monkeypatch, mask: np.ndarray) -> None:
    """Route _segmenter()/_mp_image() to fakes so no mediapipe is touched."""
    monkeypatch.setattr(BackgroundReplacer, "_segmenter", lambda self: _FakeSegmenter(mask))
    # Identity wrapper: the fake segmenter ignores the argument anyway, so we never
    # need a real mp.Image (and never import mediapipe).
    monkeypatch.setattr(BackgroundReplacer, "_mp_image", lambda self, img: img)


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #
def test_empty_backgrounds_dir_returns_input_unchanged(tmp_path):
    """No background images -> the hand image passes through untouched."""
    replacer = BackgroundReplacer(str(tmp_path))
    assert replacer.backgrounds == []
    img = np.full((32, 40, 3), 123, dtype=np.uint8)
    out = replacer(img)
    assert np.array_equal(out, img)


def test_construction_is_lazy_no_mediapipe_no_segmenter(tmp_path):
    """Constructing must import nothing heavy and must NOT build a segmenter."""
    # A background present so the list is non-empty — construction still must be lazy.
    cv2.imwrite(str(tmp_path / "bg.png"), np.zeros((8, 8, 3), dtype=np.uint8))
    had_mp = "mediapipe" in sys.modules  # may already be imported elsewhere
    replacer = BackgroundReplacer(str(tmp_path))
    assert replacer._seg is None  # segmenter not built at __init__ time
    if not had_mp:
        assert "mediapipe" not in sys.modules  # constructing didn't import mediapipe


def test_unreadable_background_returns_input_unchanged(tmp_path, monkeypatch):
    """cv2.imread returning None (corrupt/unreadable file) -> input untouched.

    This fallback runs BEFORE segmentation, so it must work with no mediapipe.
    """
    (tmp_path / "broken.png").write_bytes(b"not really a png")
    replacer = BackgroundReplacer(str(tmp_path))
    assert replacer.backgrounds  # the file is globbed...
    monkeypatch.setattr(cv2, "imread", lambda *a, **k: None)  # ...but unreadable
    img = np.full((16, 16, 3), 200, dtype=np.uint8)
    out = replacer(img)
    assert np.array_equal(out, img)


def test_composite_uses_foreground_and_background_regions(tmp_path, monkeypatch):
    """Foreground pixels come from the hand image, background pixels from the bg."""
    h, w = 20, 24
    hand = np.full((h, w, 3), (255, 0, 0), dtype=np.uint8)   # solid "hand" colour
    bg_color = (0, 255, 0)                                    # solid "room" colour
    bg_path = tmp_path / "room.png"
    cv2.imwrite(str(bg_path), np.full((h, w, 3), bg_color, dtype=np.uint8))

    # Confidence mask: top half is hand (>0.5), bottom half is room (<0.5).
    mask = np.zeros((h, w), dtype=np.float32)
    mask[: h // 2, :] = 0.9
    _install_fake_segmenter(monkeypatch, mask)

    replacer = BackgroundReplacer(str(tmp_path))
    out = replacer(hand)

    assert out.shape == hand.shape
    assert out.dtype == np.uint8
    # top half == hand, bottom half == background
    assert np.array_equal(out[: h // 2], hand[: h // 2])
    assert np.all(out[h // 2 :] == np.asarray(bg_color, dtype=np.uint8))


def test_composite_resizes_mismatched_background(tmp_path, monkeypatch):
    """A background of a different size is resized to the hand image's shape."""
    h, w = 30, 30
    hand = np.full((h, w, 3), (10, 20, 30), dtype=np.uint8)
    # bg deliberately a different size than the hand image.
    cv2.imwrite(str(tmp_path / "big.png"), np.full((64, 48, 3), (200, 100, 50), dtype=np.uint8))

    mask = np.full((h, w), 0.9, dtype=np.float32)  # everything is foreground
    _install_fake_segmenter(monkeypatch, mask)

    out = BackgroundReplacer(str(tmp_path))(hand)
    assert out.shape == hand.shape  # bg got resized to match; no broadcast error
    assert np.array_equal(out, hand)  # all-foreground -> pure hand


def test_build_transforms_with_backgrounds_returns_contract_tensor(tmp_path, monkeypatch):
    """build_transforms(train=True, backgrounds_dir=...) still yields the contract tensor.

    Exercised with the bg gate forced ON and the segmenter faked, so the full
    composite -> albumentations -> preprocess_bgr path runs without mediapipe.
    """
    pytest.importorskip("torch")
    pytest.importorskip("albumentations")
    import random as _random

    from src.augment import build_transforms

    cv2.imwrite(str(tmp_path / "room.png"), np.full((40, 40, 3), (0, 128, 255), dtype=np.uint8))
    _install_fake_segmenter(monkeypatch, np.full((32, 32), 0.9, dtype=np.float32))
    # Force the probabilistic bg step ON so we actually cover the composite path.
    monkeypatch.setattr(_random, "random", lambda: 0.0)

    transform = build_transforms(train=True, backgrounds_dir=str(tmp_path))
    out = transform(np.full((32, 32, 3), 127, dtype=np.uint8)).numpy()
    assert out.shape == (3, IMG_SIZE, IMG_SIZE)
    assert out.dtype == np.float32
