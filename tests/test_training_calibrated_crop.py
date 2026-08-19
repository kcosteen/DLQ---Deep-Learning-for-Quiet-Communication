"""Tests for the training-calibrated crop (shift-before-shrink geometry).

Covers:

* TrainingFramingMargins dataclass
* training_calibrated_context_bbox shift-before-shrink logic
* crop_square_with_training_framing output contract
* hand centred / near each edge
* requested crop wider than tall / taller than wide
* requested square larger than one frame dimension
* non-square frames (4:3, 16:9)
* shift preserves square size whenever the frame has room
* legacy functions unchanged
"""

from __future__ import annotations

import numpy as np
import pytest

from src.crop import (
    IMG_SIZE,
    TrainingFramingMargins,
    asymmetric_context_bbox,
    crop_square_with_forearm_context,
    crop_square_with_margin,
    crop_square_with_training_framing,
    resize_square,
    training_calibrated_context_bbox,
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _canvas(h=480, w=640, bg=200):
    return np.full((h, w, 3), bg, dtype=np.uint8)


def _square_side(bbox):
    x0, y0, x1, y1 = bbox
    return x1 - x0


# --------------------------------------------------------------------------- #
# TrainingFramingMargins
# --------------------------------------------------------------------------- #

def test_margins_frozen_dataclass():
    m = TrainingFramingMargins(left=0.5, right=0.5, top=0.5, bottom=0.5)
    assert m.left == 0.5
    assert m.right == 0.5
    assert m.top == 0.5
    assert m.bottom == 0.5


def test_margins_are_frozen():
    m = TrainingFramingMargins(left=0.6, right=0.4, top=0.3, bottom=1.0)
    with pytest.raises(AttributeError):
        m.left = 0.7  # type: ignore[misc]


def test_asymmetric_margins_still_work():
    """Legacy asymmetric crop is untouched."""
    from src.crop import AsymmetricMargins, DEFAULT_ASYMMETRIC_MARGINS
    assert DEFAULT_ASYMMETRIC_MARGINS.left == 0.7
    m = AsymmetricMargins(left=0.7, right=0.7, top=0.4, bottom=1.8)
    assert m.bottom == 1.8


# --------------------------------------------------------------------------- #
# training_calibrated_context_bbox — basic math
# --------------------------------------------------------------------------- #

def test_centred_hand_symmetric_margins():
    """Hand centered with L=R=T=B=0.5: requested box is the full image."""
    img = _canvas(h=200, w=200)
    # hand at (50,50)-(150,150), bw=100, bh=100
    # L=0.5: x0=50-50=0;  R=0.5: x1=150+50=200
    # T=0.5: y0=50-50=0;  B=0.5: y1=150+50=200
    m = TrainingFramingMargins(left=0.5, right=0.5, top=0.5, bottom=0.5)
    b = training_calibrated_context_bbox(img, (50, 50, 150, 150), m)
    assert b == (0, 0, 200, 200)


def test_requested_box_wider_than_tall():
    """Requested box wider than tall → side = requested width, square from height."""
    img = _canvas(h=600, w=800)
    # hand (100,200)-(300,300), bw=200, bh=100
    # L=0.8: x0=100-160=-60  R=0.3: x1=300+60=360  → rw=420
    # T=0.6: y0=200-60=140   B=0.6: y1=300+60=360  → rh=220
    # side = 420, cx=150, cy=250
    # sx0=150-210=-60 → shift to 0, sx1=420 (fits in w=800)
    # sy0=250-210=40, sy1=660 > h=600 → shift to 600-420=180
    m = TrainingFramingMargins(left=0.8, right=0.3, top=0.6, bottom=0.6)
    b = training_calibrated_context_bbox(img, (100, 200, 300, 300), m)
    x0, y0, x1, y1 = b
    assert x1 - x0 == 420  # side preserved by shift (h=600 is large enough)
    assert y1 - y0 == 420
    assert x0 >= 0 and x1 <= 800
    assert y0 >= 0 and y1 <= 600


def test_requested_crop_taller_than_wide():
    """Requested box taller than wide → side = requested height."""
    img = _canvas(h=800, w=600)
    # hand (100,200)-(200,400), bw=100, bh=200
    # L=0.5: x0=100-50=50   R=0.5: x1=200+50=250   → rw=200
    # T=0.8: y0=200-160=40   B=0.4: y1=400+80=480   → rh=440
    # side = 440, cx=150, cy=260
    # sx0=150-220=-70 → shift to 0, sy0=260-220=40 (fits in h=800)
    m = TrainingFramingMargins(left=0.5, right=0.5, top=0.8, bottom=0.4)
    b = training_calibrated_context_bbox(img, (100, 200, 200, 400), m)
    x0, y0, x1, y1 = b
    side = x1 - x0
    assert side == 440
    assert y1 - y0 == 440
    assert x0 >= 0 and x1 <= 600
    assert y0 >= 0 and y1 <= 800


# --------------------------------------------------------------------------- #
# Edge cases — shift before shrink
# --------------------------------------------------------------------------- #

def test_shift_preserves_side_at_left_edge():
    """Square pushed past left edge → shift right, side preserved."""
    img = _canvas(h=400, w=600)
    # hand at (10,100)-(50,300), bw=40, bh=200
    # L=1.0: x0=10-40=-30  R=1.0: x1=50+40=90  → rw=120
    # T=0.3: y0=100-60=40  B=0.3: y1=300+60=360 → rh=320
    # side=320, cx=30, cy=200 → sx0=30-160=-130 → shift to 0
    m = TrainingFramingMargins(left=1.0, right=1.0, top=0.3, bottom=0.3)
    b = training_calibrated_context_bbox(img, (10, 100, 50, 300), m)
    x0, y0, x1, y1 = b
    assert x0 == 0
    assert x1 - x0 == 320  # full side preserved
    assert y0 >= 0 and y1 <= 400


def test_shift_preserves_side_at_right_edge():
    """Square pushed past right edge → shift left, side preserved."""
    img = _canvas(h=400, w=600)
    # hand at (560,100)-(590,300), bw=30, bh=200
    # L=1.0: x0=560-30=530  R=1.0: x1=590+30=620 → rw=90
    # T=0.3: y0=100-60=40   B=0.3: y1=300+60=360 → rh=320
    # side=320, cx=575, cy=200 → sx0=575-160=415, sx0+320=735>600 → shift to 600-320=280
    m = TrainingFramingMargins(left=1.0, right=1.0, top=0.3, bottom=0.3)
    b = training_calibrated_context_bbox(img, (560, 100, 590, 300), m)
    x0, y0, x1, y1 = b
    assert x1 == 600
    assert x1 - x0 == 320  # full side preserved
    assert x0 >= 0


def test_shift_preserves_side_at_top_edge():
    """Square pushed past top edge → shift down, side preserved."""
    img = _canvas(h=400, w=600)
    # hand at (200,10)-(400,50), bw=200, bh=40
    # L=0.3: x0=200-60=140  R=0.3: x1=400+60=460 → rw=320
    # T=1.0: y0=10-40=-30   B=1.0: y1=50+40=90   → rh=120
    # side=320, cx=300, cy=30 → sy0=30-160=-130 → shift to 0
    m = TrainingFramingMargins(left=0.3, right=0.3, top=1.0, bottom=1.0)
    b = training_calibrated_context_bbox(img, (200, 10, 400, 50), m)
    x0, y0, x1, y1 = b
    assert y0 == 0
    assert y1 - y0 == 320
    assert x0 >= 0 and x1 <= 600


def test_shift_preserves_side_at_bottom_edge():
    """Square pushed past bottom edge → shift up, side preserved."""
    img = _canvas(h=400, w=600)
    # hand at (200,360)-(400,390), bw=200, bh=30
    # L=0.3: x0=200-60=140  R=0.3: x1=400+60=460 → rw=320
    # T=1.0: y0=360-30=330  B=1.0: y1=390+30=420 → rh=90
    # side=320, cx=300, cy=375 → sy0=375-160=215, sy0+320=535>400 → shift to 400-320=80
    m = TrainingFramingMargins(left=0.3, right=0.3, top=1.0, bottom=1.0)
    b = training_calibrated_context_bbox(img, (200, 360, 400, 390), m)
    x0, y0, x1, y1 = b
    assert y1 == 400
    assert y1 - y0 == 320
    assert y0 >= 0


# --------------------------------------------------------------------------- #
# Shrink — frame too small
# --------------------------------------------------------------------------- #

def test_shrink_when_frame_smaller_than_side():
    """If the frame is smaller than the requested side, square shrinks to frame."""
    img = _canvas(h=200, w=200)
    # hand at (50,50)-(150,150), bw=100, bh=100
    # L=R=T=B=1.5: requested rw = 100+150+150=400 > w=200
    m = TrainingFramingMargins(left=1.5, right=1.5, top=1.5, bottom=1.5)
    b = training_calibrated_context_bbox(img, (50, 50, 150, 150), m)
    x0, y0, x1, y1 = b
    # side = max(400,400) = 400 → shrunk to min(400,200,200)=200
    assert x1 - x0 == 200
    assert y1 - y0 == 200
    assert x0 == 0 and y0 == 0


# --------------------------------------------------------------------------- #
# Non-square frames
# --------------------------------------------------------------------------- #

def test_4_3_frame():
    """4:3 frame (640x480) — square conversion works correctly."""
    img = _canvas(h=480, w=640)
    m = TrainingFramingMargins(left=0.6, right=0.6, top=0.6, bottom=0.6)
    # hand centred: (220, 140)-(420, 340), bw=200, bh=200
    b = training_calibrated_context_bbox(img, (220, 140, 420, 340), m)
    x0, y0, x1, y1 = b
    side = x1 - x0
    assert side == y1 - y0  # square
    assert side >= 200 + 2 * 0.6 * 200  # >= requested width = 440
    assert x0 >= 0 and x1 <= 640
    assert y0 >= 0 and y1 <= 480


def test_16_9_frame():
    """16:9 frame (1280x720) — shift-before-shrink on a wide frame."""
    img = _canvas(h=720, w=1280)
    m = TrainingFramingMargins(left=0.6, right=0.6, top=0.6, bottom=0.6)
    # hand at (500,260)-(780,460), bw=280, bh=200
    # rw = 280 + 0.6*280*2 = 616, rh = 200 + 0.6*200*2 = 440 → side=616
    b = training_calibrated_context_bbox(img, (500, 260, 780, 460), m)
    x0, y0, x1, y1 = b
    side = x1 - x0
    assert side == 616
    assert side == y1 - y0
    assert x0 >= 0 and x1 <= 1280
    assert y0 >= 0 and y1 <= 720


# --------------------------------------------------------------------------- #
# crop_square_with_training_framing output
# --------------------------------------------------------------------------- #

def test_calibrated_crop_is_224x224():
    img = _canvas(h=480, w=640)
    m = TrainingFramingMargins(left=0.6, right=0.6, top=0.6, bottom=0.6)
    crop = crop_square_with_training_framing(img, (200, 150, 400, 350), m)
    assert crop.shape == (IMG_SIZE, IMG_SIZE, 3)
    assert crop.dtype == np.uint8


def test_calibrated_crop_near_edges_is_224x224():
    """Even when clamped at edges, output is always 224x224."""
    img = _canvas(h=480, w=640)
    m = TrainingFramingMargins(left=1.5, right=1.5, top=1.5, bottom=1.5)
    crop = crop_square_with_training_framing(img, (10, 10, 50, 50), m)
    assert crop.shape == (IMG_SIZE, IMG_SIZE, 3)


def test_calibrated_crop_nonempty():
    """Crop is never all-black (letterbox pads, but content is present)."""
    img = _canvas(h=480, w=640, bg=128)
    m = TrainingFramingMargins(left=0.6, right=0.6, top=0.6, bottom=0.6)
    crop = crop_square_with_training_framing(img, (200, 150, 400, 350), m)
    assert crop.mean() > 0


# --------------------------------------------------------------------------- #
# Legacy functions unchanged
# --------------------------------------------------------------------------- #

def test_symmetric_crop_unchanged():
    """crop_square_with_margin still works as before."""
    img = _canvas()
    crop = crop_square_with_margin(img, (200, 150, 400, 350), margin=0.62)
    assert crop.shape == (IMG_SIZE, IMG_SIZE, 3)


def test_forearm_crop_unchanged():
    """crop_square_with_forearm_context still works as before."""
    img = _canvas()
    crop = crop_square_with_forearm_context(img, (200, 150, 400, 350))
    assert crop.shape == (IMG_SIZE, IMG_SIZE, 3)


def test_asymmetric_context_bbox_unchanged():
    """asymmetric_context_bbox produces the same result as before."""
    img = _canvas()
    b = asymmetric_context_bbox(img, (100, 100, 200, 200))
    assert b == (0, 60, 310, 380)


# --------------------------------------------------------------------------- #
# Integration: training_calibrated + preprocessing pipeline
# --------------------------------------------------------------------------- #

def test_calibrated_crop_preprocess_pipeline():
    """Calibrated crop → preprocess_bgr produces a valid (3,224,224) float32."""
    from src.crop import preprocess_bgr
    img = _canvas(h=480, w=640, bg=128)
    m = TrainingFramingMargins(left=0.6, right=0.6, top=0.6, bottom=0.6)
    crop = crop_square_with_training_framing(img, (200, 150, 400, 350), m)
    tensor = preprocess_bgr(crop)
    assert tensor.shape == (3, IMG_SIZE, IMG_SIZE)
    assert tensor.dtype == np.float32


def test_calibrated_context_bbox_request_contains_hand():
    """The square crop always fully contains the hand bbox (L,R,T,B > 0)."""
    img = _canvas(h=600, w=800)
    tb = (200, 150, 400, 350)
    m = TrainingFramingMargins(left=0.6, right=0.6, top=0.8, bottom=1.2)
    b = training_calibrated_context_bbox(img, tb, m)
    # The crop must always contain the hand bbox
    assert b[0] <= tb[0]  # crop left <= hand left
    assert b[1] <= tb[1]  # crop top  <= hand top
    assert b[2] >= tb[2]  # crop right >= hand right
    assert b[3] >= tb[3]  # crop bottom >= hand bottom


def test_calibrated_shift_direction_left_then_right():
    """When shifted right then left, the shift is always minimal."""
    img = _canvas(h=300, w=500)
    m = TrainingFramingMargins(left=0.6, right=0.6, top=0.6, bottom=0.6)
    # hand near left: should shift right
    b_left = training_calibrated_context_bbox(img, (10, 100, 110, 200), m)
    # hand near right: should shift left
    b_right = training_calibrated_context_bbox(img, (390, 100, 490, 200), m)
    assert b_left[0] >= 0
    assert b_right[2] <= 500
    # both are full-size squares
    assert b_left[2] - b_left[0] == b_left[3] - b_left[1]
    assert b_right[2] - b_right[0] == b_right[3] - b_right[1]
