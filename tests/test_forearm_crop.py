"""Tests for the forearm-preserving asymmetric crop experiment.

Covers the pieces that are unit-testable without mediapipe:

* asymmetric margin math                      (asymmetric_context_bbox)
* image-bound clipping
* square conversion without stretching        (resize_square letterbox)
* preserving the requested bottom/forearm context where the image allows
* crop-mode routing in the app                (process_frame / build_parser)
* SYMMETRIC vs FOREARM A/B line               (--compare-crops)
* the legacy symmetric crop path staying unchanged

The integrated tests (real mediapipe + checkpoint) are gated on those assets
existing, same pattern as tests/test_webcam_bridge.py.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from src.crop import (
    AsymmetricMargins,
    DEFAULT_ASYMMETRIC_MARGINS,
    IMG_SIZE,
    asymmetric_context_bbox,
    crop_square_with_forearm_context,
    crop_square_with_margin,
    square_bbox_with_margin,
)
from app.webcam_speller import (
    DEFAULT_CROP_BOTTOM,
    DEFAULT_CROP_LEFT,
    DEFAULT_CROP_RIGHT,
    DEFAULT_CROP_TOP,
    build_parser,
    process_frame,
)

CHECKPOINT = Path("checkpoints/efficientnet_b0_targeted.pt")

HAS_MEDIAPIPE = pytest.importorskip("mediapipe") is not None
NEEDS_MP = pytest.mark.skipif(not HAS_MEDIAPIPE, reason="mediapipe not installed")
NEEDS_CKPT = pytest.mark.skipif(
    not CHECKPOINT.is_file(), reason=f"{CHECKPOINT} missing"
)


def _canvas(h=480, w=640, bg=200):
    return np.full((h, w, 3), bg, dtype=np.uint8)


# --------------------------------------------------------------------------- #
# Asymmetric margin math
# --------------------------------------------------------------------------- #
def test_default_margins_match_experiment_spec():
    m = DEFAULT_ASYMMETRIC_MARGINS
    assert m.left == 0.7 and m.right == 0.7
    assert m.top == 0.4 and m.bottom == 1.8


def test_asymmetric_requested_box_expands_per_side():
    # tight 100x100 at (100,100)-(200,200); requested box:
    #   L:100-70=30  R:200+70=270  T:100-40=60  B:200+180=380
    #   requested w=240, h=320 -> side 320, center (150,220)
    #   square x: 150-160=-10 -> 0 (clamped), y: 60..380
    img = _canvas()
    b = asymmetric_context_bbox(img, (100, 100, 200, 200))
    assert b == (0, 60, 310, 380)


def test_asymmetric_keeps_requested_bottom_edge_when_not_clipped():
    # Away from the image edge the requested bottom context must survive exactly:
    # the square side is the requested height, so crop bottom == requested bottom.
    img = _canvas()
    b = asymmetric_context_bbox(img, (200, 150, 300, 250))
    _, by0, _, by1 = b
    assert by1 == 250 + 1.8 * 100 == 430  # requested bottom
    assert by0 == 150 - 0.4 * 100 == 110  # requested top
    assert (by1 - by0) >= (250 + 1.8 * 100) - (150 - 0.4 * 100)  # no squeeze
    assert b[2] - b[0] >= 2.4 * 100


def test_asymmetric_square_never_smaller_than_requested_width():
    # Width must reach at least the requested width even when height dominates.
    img = _canvas()
    b = asymmetric_context_bbox(img, (200, 150, 300, 250))
    assert (b[2] - b[0]) >= (300 + 0.7 * 100) - (200 - 0.7 * 100)  # >= 240
    assert (b[3] - b[1]) == 430 - 110  # square side 320 (height dominates)


# --------------------------------------------------------------------------- #
# Clipping / bounds safety
# --------------------------------------------------------------------------- #
def test_asymmetric_clamps_to_image_bottom():
    # Hand near the bottom: requested bottom (580) is beyond the frame, so the
    # crop must clamp to h and never error or go out of bounds.
    img = _canvas(h=480)
    b = asymmetric_context_bbox(img, (100, 300, 200, 400))
    assert b[1] == 260 and b[3] == 480
    assert 0 <= b[0] < b[2] <= 640 and 0 <= b[1] < b[3] <= 480


def test_asymmetric_clamps_to_image_edges():
    # Hand in the corner: every side clamps in-bounds and the box never
    # degenerates to zero area.
    img = _canvas()
    b = asymmetric_context_bbox(img, (0, 0, 100, 100))
    assert b[0] == 0 and b[1] == 0
    assert b[2] <= 640 and b[3] <= 480
    assert b[2] > b[0] and b[3] > b[1]


def test_asymmetric_crop_output_is_224_square_and_nonempty():
    img = _canvas()
    crop = crop_square_with_forearm_context(img, (200, 150, 300, 250))
    assert crop.shape == (IMG_SIZE, IMG_SIZE, 3)
    assert crop.dtype == np.uint8
    assert crop.mean() > 0


def test_asymmetric_zero_margins_is_tight_bbox_crop():
    img = _canvas()
    tight = (200, 150, 300, 250)
    m = AsymmetricMargins(0.0, 0.0, 0.0, 0.0)
    b = asymmetric_context_bbox(img, tight, m)
    assert b == tight
    crop = crop_square_with_forearm_context(img, tight, m)
    assert crop.shape == (IMG_SIZE, IMG_SIZE, 3)


def test_asymmetric_keeps_forearm_context_relative_to_symmetric():
    # With equal total side (symmetric margin vs forearm square), the forearm
    # crop must reach further below the hand: its bottom edge is lower.
    img = _canvas()
    tight = (300, 120, 380, 220)  # 80x100, hand away from edges
    h, w = img.shape[:2]
    sym = square_bbox_with_margin(*tight, w, h, 0.62)
    fore = asymmetric_context_bbox(img, tight)
    assert fore[3] > sym[3]
    assert fore[1] < sym[1] + (fore[3] - sym[3])  # more of the extra is below


# --------------------------------------------------------------------------- #
# Legacy symmetric path unchanged
# --------------------------------------------------------------------------- #
def test_symmetric_crop_unchanged_after_forearm_additions():
    # The symmetric helper must still produce exactly the old bbox semantics.
    img = _canvas()
    tight = (50, 80, 150, 120)
    h, w = img.shape[:2]
    assert square_bbox_with_margin(*tight, w, h, 0.25) == (25, 25, 175, 175)
    crop = crop_square_with_margin(img, tight, 0.62)
    assert crop.shape == (IMG_SIZE, IMG_SIZE, 3)


# --------------------------------------------------------------------------- #
# Crop-mode routing (app)
# --------------------------------------------------------------------------- #
def test_crop_mode_cli_default_is_symmetric():
    args = build_parser().parse_args(["--checkpoint", str(CHECKPOINT)])
    assert args.crop_mode == "symmetric"


def test_crop_mode_cli_selects_forearm_with_default_margins():
    args = build_parser().parse_args(["--checkpoint", str(CHECKPOINT),
                                      "--crop-mode", "forearm"])
    assert args.crop_mode == "forearm"
    assert args.crop_left == DEFAULT_CROP_LEFT
    assert args.crop_right == DEFAULT_CROP_RIGHT
    assert args.crop_top == DEFAULT_CROP_TOP
    assert args.crop_bottom == DEFAULT_CROP_BOTTOM


def test_crop_margins_configurable_via_cli():
    args = build_parser().parse_args(["--checkpoint", str(CHECKPOINT),
                                      "--crop-left", "0.3", "--crop-right", "0.4",
                                      "--crop-top", "0.2", "--crop-bottom", "2.0"])
    assert (args.crop_left, args.crop_right) == (0.3, 0.4)
    assert (args.crop_top, args.crop_bottom) == (0.2, 2.0)


class _FakeClassifier:
    """Records every BGR image it was asked to classify; returns a fixed label."""

    def __init__(self, letter="S", conf=0.9):
        self.letter, self.conf = letter, conf
        self.seen = []

    def tensor_from_bgr(self, img_bgr):
        self.seen.append(img_bgr.copy())
        return img_bgr  # opaque to probs()

    def probs(self, x):
        import torch

        idx = ord(self.letter) - ord("A")
        if idx >= 26:
            idx = 29 - 1
        p = torch.zeros(29)
        p[idx] = self.conf
        p[(idx + 1) % 29] = 1.0 - self.conf
        return p


class _FakeCropper:
    def __init__(self, tight):
        self.tight = tight

    def tight_bbox(self, frame):
        return self.tight


def test_process_frame_routes_to_forearm_crop():
    img = _canvas()
    tight = (200, 150, 300, 250)
    clf = _FakeClassifier()
    out, _ = process_frame(img, _FakeCropper(tight), clf, 0.62, 0.30, "crop",
                           crop_mode="forearm")
    assert out.result.found_hand and out.crop is not None
    b = asymmetric_context_bbox(img, tight)
    assert out.margin_bbox == b
    assert len(clf.seen) == 1  # exactly one forward, the forearm crop


def test_process_frame_routes_to_symmetric_crop():
    img = _canvas()
    tight = (200, 150, 300, 250)
    clf = _FakeClassifier()
    out, _ = process_frame(img, _FakeCropper(tight), clf, 0.62, 0.30, "crop",
                           crop_mode="symmetric")
    h, w = img.shape[:2]
    assert out.margin_bbox == square_bbox_with_margin(*tight, w, h, 0.62)
    assert len(clf.seen) == 1


def test_process_frame_compare_crops_runs_both():
    img = _canvas()
    tight = (200, 150, 300, 250)
    clf = _FakeClassifier()
    out, _ = process_frame(img, _FakeCropper(tight), clf, 0.62, 0.30, "crop",
                           crop_mode="symmetric", compare_crops=True)
    assert len(clf.seen) == 3  # primary + symmetric + forearm forwards
    assert out.ab_line is not None
    assert "SYMMETRIC:" in out.ab_line and "FOREARM:" in out.ab_line


def test_process_frame_compare_crops_in_forearm_mode_runs_both():
    img = _canvas()
    tight = (200, 150, 300, 250)
    clf = _FakeClassifier()
    out, _ = process_frame(img, _FakeCropper(tight), clf, 0.62, 0.30, "crop",
                           crop_mode="forearm", compare_crops=True)
    assert len(clf.seen) == 3  # primary + symmetric + forearm forwards
    assert "SYMMETRIC:" in out.ab_line and "FOREARM:" in out.ab_line


def test_process_frame_no_hand_never_classifies_in_forearm_mode():
    class NeverCalled:
        def tensor_from_bgr(self, _):
            raise AssertionError("model called on a no-hand frame")

        def probs(self, _):
            raise AssertionError("model called on a no-hand frame")

    blank = np.full((480, 640, 3), 128, dtype=np.uint8)
    out, tight = process_frame(blank, _FakeCropper(None), NeverCalled(),
                               0.62, 0.30, "crop", crop_mode="forearm")
    assert tight is None
    assert out.result.status == "no_hand"
    assert out.crop is None


# --------------------------------------------------------------------------- #
# Integrated: real mediapipe (+ targeted checkpoint for A/B)
# --------------------------------------------------------------------------- #
@NEEDS_MP
def test_forearm_and_symmetric_crops_differ_on_real_hand():
    from src.crop import HandCropper

    img = cv2.imread("data/raw/asl_alphabet_train/S/S1.jpg", cv2.IMREAD_COLOR)
    with HandCropper() as cropper:
        tb = cropper.tight_bbox(img)
    assert tb is not None
    fore = crop_square_with_forearm_context(img, tb)
    sym = crop_square_with_margin(img, tb, 0.62)
    assert fore.shape == sym.shape == (IMG_SIZE, IMG_SIZE, 3)
    assert not np.array_equal(fore, sym)
    # The forearm crop must keep the bottom region the symmetric crop drops.
    assert np.abs(fore[180:, :].mean() - sym[180:, :].mean()) > 0


@NEEDS_MP
@NEEDS_CKPT
def test_process_frame_ab_line_real_checkpoint():
    from src.crop import HandCropper
    from app.webcam_speller import Classifier

    clf = Classifier(str(CHECKPOINT), device="cpu")
    img = cv2.imread("data/raw/asl_alphabet_train/S/S1.jpg", cv2.IMREAD_COLOR)
    with HandCropper() as cropper:
        out, _ = process_frame(img, cropper, clf, 0.62, 0.30, "crop",
                               crop_mode="symmetric", compare_crops=True)
    assert out.result.found_hand
    assert out.ab_line is not None
    assert "SYMMETRIC:" in out.ab_line and "FOREARM:" in out.ab_line
