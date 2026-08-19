"""Tests for the live webcam bridge (#17): MediaPipe localizer -> crop -> net.

Covers the pieces that are unit-testable without a physical webcam:

* square bbox + margin behaviour            (src.crop.square_bbox_with_margin)
* configurable crop margin                  (src.crop.crop_square_with_margin)
* image-boundary clipping
* no-hand suppression (never classify a centre fallback)
* confidence gate
* full-frame vs localized-crop routing      (app.webcam_speller.pipeline_mode)
* the A/B and no-hand per-frame pipeline    (app.webcam_speller.process_frame)

The integrated tests (real MediaPipe + the targeted checkpoint) are gated on the
model file and checkpoint existing. No hardware is touched.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from src.crop import (
    BBOX_MARGIN,
    IMG_SIZE,
    crop_square_with_margin,
    square_bbox_with_margin,
)
from src.data import CLASSES
from app.webcam_speller import (
    DEFAULT_CONFIDENCE_THRESHOLD,
    TRAINING_DERIVED_CROP_MARGIN,
    HIT_TOLERANCE,
    DeleteButton,
    FrameResult,
    click_in_delete,
    decide_prediction,
    draw_delete_button,
    handle_delete_key,
    on_mouse,
    pipeline_mode,
    process_frame,
    _status_lines,
    build_parser,
)

CHECKPOINT = Path("checkpoints/efficientnet_b0_targeted.pt")

HAS_MEDIAPIPE = pytest.importorskip("mediapipe") is not None
NEEDS_MP = pytest.mark.skipif(not HAS_MEDIAPIPE, reason="mediapipe not installed")
NEEDS_CKPT = pytest.mark.skipif(
    not CHECKPOINT.is_file(), reason=f"{CHECKPOINT} missing"
)


# --------------------------------------------------------------------------- #
# Square bbox + margin (pure)
# --------------------------------------------------------------------------- #
def test_square_bbox_margin_grows_side_relative_to_bbox():
    # tight box 100 wide x 40 tall -> square side 100; margin 0.25 -> side 150.
    bx0, by0, bx1, by1 = square_bbox_with_margin(50, 80, 150, 120, 500, 500, 0.25)
    assert bx1 - bx0 == 150
    assert by1 - by0 == 150


def test_square_bbox_zero_margin_is_square_of_max_side():
    bx0, by0, bx1, by1 = square_bbox_with_margin(10, 10, 40, 50, 500, 500, 0.0)
    assert bx1 - bx0 == 40 and by1 - by0 == 40


def test_square_bbox_preserves_center():
    bx0, by0, bx1, by1 = square_bbox_with_margin(100, 100, 200, 160, 1000, 1000, 0.5)
    # tight center (150, 130); side = 100*2 = 200 -> box (50, 30, 250, 230)
    assert (bx0 + bx1) / 2.0 == 150.0
    assert (by0 + by1) / 2.0 == 130.0


def test_square_bbox_larger_margin_larger_box():
    _, _, a1, _ = square_bbox_with_margin(10, 10, 110, 110, 1000, 1000, 0.25)
    _, _, b1, _ = square_bbox_with_margin(10, 10, 110, 110, 1000, 1000, 0.75)
    assert b1 - 10 > a1 - 10


def test_square_bbox_clips_to_image_boundary():
    # hand against the left edge: margin box would extend to x < 0.
    bx0, by0, bx1, by1 = square_bbox_with_margin(0, 40, 100, 140, 200, 200, 1.0)
    assert bx0 == 0 and by0 == 0 and bx1 <= 200 and by1 <= 200
    assert bx1 > bx0 and by1 > by0  # never degenerate


def test_square_bbox_hand_corner_clamps_both_axes():
    bx0, by0, bx1, by1 = square_bbox_with_margin(2, 2, 60, 70, 200, 200, 2.0)
    assert bx0 == 0 and by0 == 0
    assert bx1 <= 200 and by1 <= 200


def test_square_bbox_hand_right_edge_clamps_high_side():
    bx0, by0, bx1, by1 = square_bbox_with_margin(170, 60, 199, 140, 200, 200, 1.0)
    assert bx1 == 200
    assert 0 <= bx0 < bx1


# --------------------------------------------------------------------------- #
# Configurable crop margin (pure, synthetic images)
# --------------------------------------------------------------------------- #
def _handless_canvas(size=(400, 640)):
    """A plain frame with a dark 'hand' rectangle at a known location."""
    img = np.full((size[0], size[1], 3), 220, dtype=np.uint8)
    cv2.rectangle(img, (200, 120), (320, 300), (90, 90, 90), -1)
    return img


def test_crop_square_with_margin_is_224_square():
    img = _handless_canvas()
    crop = crop_square_with_margin(img, (200, 120, 320, 300), 0.62)
    assert crop.shape == (IMG_SIZE, IMG_SIZE, 3)
    assert crop.dtype == np.uint8
    assert crop.mean() > 0  # not empty/black


def test_crop_square_with_margin_respects_margin_argument():
    img = _handless_canvas()
    # tight box 120x180 -> side 180. margin 0.25 -> side 270 crop around center.
    tight = (200, 120, 320, 300)
    h, w = img.shape[:2]
    b_lo = square_bbox_with_margin(*tight, w, h, 0.1)
    b_hi = square_bbox_with_margin(*tight, w, h, 0.9)
    crop_lo = crop_square_with_margin(img, tight, 0.1)
    crop_hi = crop_square_with_margin(img, tight, 0.9)
    assert (b_hi[2] - b_hi[0]) > (b_lo[2] - b_lo[0])
    # A bigger margin includes more background -> brighter crop.
    assert crop_hi.mean() > crop_lo.mean()


def test_crop_square_with_margin_clips_without_error():
    img = _handless_canvas()
    # hand touching the right edge; margin crop would run off the frame.
    tight = (560, 120, 639, 300)
    crop = crop_square_with_margin(img, tight, 0.62)
    assert crop.shape == (IMG_SIZE, IMG_SIZE, 3)
    assert crop.mean() > 0


# --------------------------------------------------------------------------- #
# No-hand suppression + confidence gate
# --------------------------------------------------------------------------- #
def test_no_hand_never_yields_a_letter():
    r = decide_prediction("S", 0.99, found_hand=False, threshold=0.70)
    assert r.status == "no_hand"
    assert r.letter is None and r.confidence == 0.0
    assert r.top3 == []


def test_no_hand_wins_even_over_high_confidence():
    # The guard must apply regardless of what a model would have said.
    r = decide_prediction("A", 0.999, found_hand=False, threshold=0.70)
    assert r.status == "no_hand" and r.letter is None


def test_confidence_gate_below_threshold_is_uncertain():
    r = decide_prediction("S", 0.54, found_hand=True, threshold=0.70)
    assert r.status == "uncertain"
    assert r.letter is None
    assert r.confidence == pytest.approx(0.54)  # kept for the HUD line


def test_confidence_gate_at_threshold_is_ok():
    r = decide_prediction("S", 0.70, found_hand=True, threshold=0.70)
    assert r.status == "ok" and r.letter == "S"
    r2 = decide_prediction("S", 0.87, found_hand=True, threshold=0.70)
    assert r2.status == "ok" and r2.letter == "S" and r2.confidence == pytest.approx(0.87)


def test_uncertain_keeps_top3_for_display():
    top3 = [("S", 0.54), ("X", 0.1), ("V", 0.05)]
    r = decide_prediction("S", 0.54, True, 0.70, top3)
    assert r.status == "uncertain"
    assert len(r.top3) == 3


def test_status_lines_match_spec():
    assert _status_lines(FrameResult("no_hand", False)) == ("No hand", "")
    assert _status_lines(FrameResult("uncertain", True, confidence=0.54)) == \
        ("Uncertain", "Confidence: 0.54")
    assert _status_lines(FrameResult("ok", True, letter="S", confidence=0.87)) == \
        ("Prediction: S", "Confidence: 0.87")


# --------------------------------------------------------------------------- #
# Routing
# --------------------------------------------------------------------------- #
def test_pipeline_mode_routing():
    assert pipeline_mode(False) == "crop"
    assert pipeline_mode(True) == "both"


def test_app_defaults_come_from_training_and_spec():
    assert TRAINING_DERIVED_CROP_MARGIN == 0.62
    assert DEFAULT_CONFIDENCE_THRESHOLD == 0.30
    args = build_parser().parse_args(
        ["--checkpoint", str(CHECKPOINT)])
    assert args.crop_margin == TRAINING_DERIVED_CROP_MARGIN
    assert args.confidence_threshold == DEFAULT_CONFIDENCE_THRESHOLD
    assert args.confidence_threshold == 0.30  # CLI default matches the constant


def test_crop_margin_is_configurable_via_cli():
    args = build_parser().parse_args(
        ["--checkpoint", str(CHECKPOINT), "--crop-margin", "0.35",
         "--confidence-threshold", "0.5"])
    assert args.crop_margin == 0.35
    assert args.confidence_threshold == 0.5


# --------------------------------------------------------------------------- #
# Integrated: real MediaPipe (+ targeted checkpoint for A/B)
# --------------------------------------------------------------------------- #
@NEEDS_MP
def test_hand_bbox_is_tight_plus_contract_margin():
    from src.crop import HandCropper

    img = cv2.imread("data/raw/asl_alphabet_train/S/S1.jpg",
                     cv2.IMREAD_COLOR)
    with HandCropper() as cropper:
        tb = cropper.tight_bbox(img)
    assert tb is not None
    with HandCropper() as cropper:
        hb = cropper.hand_bbox(img)
    h, w = img.shape[:2]
    expect = square_bbox_with_margin(*tb, w, h, BBOX_MARGIN)
    assert hb == expect


@NEEDS_MP
def test_tight_bbox_none_on_empty_frame():
    from src.crop import HandCropper

    blank = np.full((480, 640, 3), 128, dtype=np.uint8)
    with HandCropper() as cropper:
        assert cropper.tight_bbox(blank) is None


@NEEDS_MP
def test_process_frame_no_hand_never_calls_model():
    """A frame with no hand must resolve to no_hand and cost no inference."""
    from src.crop import HandCropper

    class NeverCalled:
        def tensor_from_bgr(self, _):
            raise AssertionError("model path called on a no-hand frame")

        def probs(self, _):
            raise AssertionError("model path called on a no-hand frame")

    blank = np.full((480, 640, 3), 128, dtype=np.uint8)
    with HandCropper() as cropper:
        out, tight = process_frame(blank, cropper, NeverCalled(), 0.62, 0.70,
                                   "crop")
    assert tight is None
    assert out.result.status == "no_hand"
    assert out.crop is None
    assert out.timing["localization"] > 0
    assert out.timing["inference"] == 0.0
    assert out.timing["total"] > 0


@NEEDS_MP
@NEEDS_CKPT
def test_process_frame_crop_and_ab_modes():
    """A detected hand produces a crop path; A/B mode additionally runs full."""
    from src.crop import HandCropper
    from app.webcam_speller import Classifier

    img = cv2.imread("data/raw/asl_alphabet_train/S/S1.jpg", cv2.IMREAD_COLOR)
    clf = Classifier(str(CHECKPOINT), device="cpu")
    with HandCropper() as cropper:
        out_crop, tight = process_frame(img, cropper, clf, 0.62, 0.70, "crop")
        out_ab, _ = process_frame(img, cropper, clf, 0.62, 0.70, "both")
    assert tight is not None
    assert out_crop.crop is not None and out_crop.crop.shape[0] == IMG_SIZE
    assert out_crop.result.found_hand
    assert out_crop.ab_line is None
    assert out_ab.ab_line is not None
    assert out_ab.ab_line.startswith("FULL: ") and "CROP: " in out_ab.ab_line
    for stage in ("localization", "crop_preprocess", "inference", "total"):
        assert out_crop.timing[stage] > 0


@NEEDS_MP
@NEEDS_CKPT
def test_classifier_topk_letters_are_valid_classes():
    from src.crop import HandCropper
    from app.webcam_speller import Classifier

    clf = Classifier(str(CHECKPOINT), device="cpu")
    with HandCropper() as cropper:
        tb = cropper.tight_bbox(
            cv2.imread("data/raw/asl_alphabet_train/S/S1.jpg"))
    assert tb is not None
    crop = crop_square_with_margin(
        cv2.imread("data/raw/asl_alphabet_train/S/S1.jpg"), tb, 0.62)
    top3 = clf.topk(crop, k=3)
    assert len(top3) == 3
    assert all(letter in CLASSES for letter, _ in top3)
    assert all(0.0 <= prob <= 1.0 for _, prob in top3)


# --------------------------------------------------------------------------- #
# HUD DELETE button
# --------------------------------------------------------------------------- #
def test_delete_button_strip_is_full_width_and_above_word_bar():
    frame = np.zeros((480, 640, 3), np.uint8)
    x0, y0, x1, y1 = draw_delete_button(frame)
    assert x0 == 0 and x1 == 640          # full-width strip
    assert y0 >= 0 and y1 < 480           # inside the frame
    assert y1 <= 480 - 50                 # above the word bar (h-50..h)
    assert y1 > y0


def test_click_in_delete_inside_tolerance_and_outside():
    rect = (0, 360, 640, 424)  # frame_h=480 strip
    assert click_in_delete(rect, 480, 100, 390)      # click inside the strip
    assert click_in_delete(rect, 480, 100, 390 - HIT_TOLERANCE)  # top-edge tol
    assert click_in_delete(rect, 480, 100, 424 + HIT_TOLERANCE)  # bottom-edge tol
    assert not click_in_delete(rect, 480, 100, 300)  # too far above
    assert not click_in_delete(None, 480, 100, 390)


def test_on_mouse_click_marks_button():
    btn = DeleteButton()
    btn.rect = (0, 360, 640, 424)
    btn.frame_h = 480
    on_mouse(cv2.EVENT_LBUTTONDOWN, 320, 390, 0, btn)
    assert btn.clicked
    btn.clicked = False
    on_mouse(cv2.EVENT_LBUTTONDBLCLK, 320, 390, 0, btn)  # double-click works
    assert btn.clicked
    btn.clicked = False
    on_mouse(cv2.EVENT_LBUTTONDOWN, 320, 100, 0, btn)
    assert not btn.clicked
    on_mouse(cv2.EVENT_MOUSEMOVE, 320, 390, 0, btn)
    assert not btn.clicked
