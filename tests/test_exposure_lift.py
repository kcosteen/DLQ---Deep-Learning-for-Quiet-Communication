"""The adaptive shadow lift (#12).

S sits at 0.770 against the 85% bar because the dataset is backlit and S vs E is
the one confusable pair with no silhouette cue — only interior shading, which the
exposure crushes. ``ExposureLift`` recovers that tonal range inside the model.

Two properties carry the design and are worth guarding:

* **It is a no-op on well-exposed input.** This is the whole reason the gamma is
  self-calibrating rather than fixed. A constant gamma 0.45 measurably *hurt* the
  classes that were already correctly exposed, and would wash out a well-lit
  webcam frame at serve time.
* **Its checkpoints cannot be confused with plain efficientnet's.** The lift lives
  in the model so it travels with the weights; that guarantee is only real if the
  arch name makes the two non-interchangeable.

Tests run on synthetic tensors — the arithmetic is what matters and a real photo
cannot make it any more true.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.crop import IMAGENET_MEAN, IMAGENET_STD
from src.model import (
    GAMMA_MAX,
    GAMMA_MIN,
    LIFT_TARGET,
    ExposureLift,
    build_model,
)

MEAN = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
STD = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)


def normalized(level: float, n: int = 1, size: int = 32) -> torch.Tensor:
    """A flat image at ``level`` in [0,1], put through the contract's normalization."""
    p = torch.full((n, 3, size, size), float(level))
    return (p - MEAN) / STD


def denormalized(x: torch.Tensor) -> torch.Tensor:
    return x * STD + MEAN


def backlit(dark: float, bright: float, frac_bright: float = 0.34) -> torch.Tensor:
    """A frame shaped like this dataset: dark subject, blown-out background."""
    size = 32
    p = torch.full((1, 3, size, size), float(dark))
    cut = int(size * frac_bright)
    p[:, :, :cut, :] = float(bright)
    return (p - MEAN) / STD


# --- the no-op property -------------------------------------------------------

def test_well_exposed_input_is_left_alone():
    """Anchor already at/above target -> gamma clamps to 1.0 -> exact no-op."""
    x = normalized(0.6)
    lift = ExposureLift()

    assert torch.allclose(lift.gamma_for(x), torch.tensor(GAMMA_MAX))
    assert torch.allclose(lift(x), x, atol=1e-5)


def test_bright_input_is_never_darkened():
    """gamma_max = 1.0, so an over-exposed frame cannot be pushed down."""
    lift = ExposureLift()
    for level in (0.5, 0.7, 0.9, 0.99):
        x = normalized(level)
        assert lift.gamma_for(x).item() == pytest.approx(GAMMA_MAX)
        out = denormalized(lift(x))
        assert out.min().item() >= level - 1e-4, f"darkened at {level}"


# --- the lift actually lifts --------------------------------------------------

def test_dark_input_is_brightened():
    x = normalized(0.10)
    lift = ExposureLift()

    assert lift.gamma_for(x).item() < 1.0
    assert denormalized(lift(x)).mean().item() > 0.10


def test_anchor_lands_on_target():
    """The defining property: the percentile anchor is mapped to LIFT_TARGET."""
    level = 0.10
    out = denormalized(ExposureLift()(normalized(level)))
    # A flat image puts every percentile at the same place, so the whole frame
    # should land on the target.
    assert out.mean().item() == pytest.approx(LIFT_TARGET, abs=0.02)


def test_gamma_is_clamped_on_a_near_black_frame():
    """Without the floor, an almost-black anchor would demand an absurd gamma."""
    lift = ExposureLift()
    assert lift.gamma_for(normalized(0.001)).item() == pytest.approx(GAMMA_MIN)


def test_backlit_frame_lifts_the_subject_not_just_the_mean():
    """The case that motivated this: dark hand, blown background.

    A median-anchored lift would no-op here -- the blown third drags the median to
    mid-grey. The 25th-percentile anchor has to reach the dark region instead.
    """
    x = backlit(dark=0.12, bright=0.96)
    lift = ExposureLift()

    assert lift.gamma_for(x).item() < 1.0, "must not no-op on a backlit frame"

    before, after = denormalized(x), denormalized(lift(x))
    dark_before = before[before < 0.5].mean().item()
    dark_after = after[before < 0.5].mean().item()
    assert dark_after > dark_before * 1.5, "shadow region should lift substantially"


def test_median_anchored_lift_would_have_no_opped():
    """Pins the reason the anchor is the 25th percentile and not the median.

    Documents the trap rather than the fix: on a frame that is a third blown out,
    the median sits near mid-grey and a median-anchored gamma is ~1.0.
    """
    x = backlit(dark=0.12, bright=0.96, frac_bright=0.5)
    median_anchored = ExposureLift(percentile=0.5, target=LIFT_TARGET)
    assert median_anchored.gamma_for(x).item() == pytest.approx(GAMMA_MAX)


# --- per-image, not per-batch -------------------------------------------------

def test_gamma_is_computed_per_image():
    """A dark frame batched with a bright one must not contaminate it."""
    x = torch.cat([normalized(0.08), normalized(0.75)], dim=0)
    gammas = ExposureLift().gamma_for(x)

    assert gammas.shape == (2,)
    assert gammas[0].item() < 1.0
    assert gammas[1].item() == pytest.approx(GAMMA_MAX)


def test_batching_does_not_change_a_single_image_result():
    solo = normalized(0.12)
    batched = torch.cat([solo, normalized(0.9)], dim=0)
    lift = ExposureLift()
    assert torch.allclose(lift(solo)[0], lift(batched)[0], atol=1e-5)


# --- determinism, shape, config ----------------------------------------------

def test_is_deterministic():
    x = normalized(0.2)
    lift = ExposureLift()
    assert torch.allclose(lift(x), lift(x))


def test_preserves_shape_and_dtype():
    x = normalized(0.3, n=4, size=224)
    out = ExposureLift()(x)
    assert out.shape == x.shape and out.dtype == x.dtype


def test_has_no_trainable_parameters():
    """It is a fixed transform; nothing here should receive gradient."""
    assert list(ExposureLift().parameters()) == []


def test_rejects_nonsense_config():
    for bad in (0.0, 1.0, -0.1, 1.5):
        with pytest.raises(ValueError):
            ExposureLift(percentile=bad)
        with pytest.raises(ValueError):
            ExposureLift(target=bad)


# --- the arch contract --------------------------------------------------------

def test_lift_arch_puts_the_lift_first():
    model = build_model("efficientnet_lift", pretrained=False)
    assert isinstance(model[0], ExposureLift)


def test_plain_arch_is_unchanged():
    """The #6 / #12 baseline checkpoints must still load, so keep it byte-identical."""
    model = build_model("efficientnet", pretrained=False)
    assert not isinstance(model[0], ExposureLift)


def test_the_two_arches_have_incompatible_state_dicts():
    """This is what makes 'the lift travels with the weights' true rather than hoped.

    train.py records `arch` and load_resume_state refuses a mismatch; this pins the
    underlying fact that the two really are not interchangeable.
    """
    plain = build_model("efficientnet", pretrained=False).state_dict()
    lifted = build_model("efficientnet_lift", pretrained=False)
    with pytest.raises(RuntimeError):
        lifted.load_state_dict(plain, strict=True)


def test_freeze_backbone_finds_the_backbone_through_the_lift():
    from src.model import freeze_backbone, unfreeze_backbone

    model = build_model("efficientnet_lift", pretrained=False)
    freeze_backbone(model)
    backbone = model[1]
    assert all(not p.requires_grad for p in backbone.parameters())
    head = model[2]
    assert all(p.requires_grad for p in head.parameters()), "head must stay trainable"

    unfreeze_backbone(model)
    assert all(p.requires_grad for p in model.parameters())


def test_forward_runs_end_to_end():
    model = build_model("efficientnet_lift", pretrained=False).eval()
    with torch.no_grad():
        out = model(normalized(0.15, n=2, size=224))
    assert out.shape == (2, 29)


# --- the serve path -----------------------------------------------------------
# The lift is only a real guarantee if it survives export. The percentile is taken
# with torch.sort rather than torch.quantile for exactly this reason: quantile has
# no ONNX lowering, sort does.

def test_lift_is_torchscript_traceable():
    lift = ExposureLift().eval()
    x = normalized(0.12, n=2, size=64)
    traced = torch.jit.trace(lift, x)
    assert torch.allclose(traced(x), lift(x), atol=1e-5)


def test_traced_lift_still_adapts_to_a_different_exposure():
    """Guards against the gamma being frozen into the trace as a constant."""
    lift = ExposureLift().eval()
    traced = torch.jit.trace(lift, normalized(0.12, n=1, size=64))

    bright = normalized(0.8, n=1, size=64)
    assert torch.allclose(traced(bright), lift(bright), atol=1e-5)
    assert torch.allclose(traced(bright), bright, atol=1e-4), "should no-op when bright"


def test_lift_exports_to_onnx(tmp_path):
    """torch.sort lowers to ONNX; torch.quantile would not."""
    pytest.importorskip("onnx")
    lift = ExposureLift().eval()
    dst = tmp_path / "lift.onnx"
    try:
        torch.onnx.export(
            lift, normalized(0.12, n=1, size=64), str(dst),
            opset_version=17, dynamo=False,
        )
    except TypeError:  # torch < 2.6 has no `dynamo` kwarg; legacy tracer is default
        torch.onnx.export(lift, normalized(0.12, n=1, size=64), str(dst), opset_version=17)
    assert dst.exists() and dst.stat().st_size > 0
