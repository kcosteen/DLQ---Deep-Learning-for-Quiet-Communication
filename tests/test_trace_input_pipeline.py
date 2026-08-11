"""Light tests for the input-pipeline diagnostic helpers (#12 / Task 2).

The diagnostic must report tensor stats without altering the tensor and must
reconstruct a normalized tensor back into a viewable RGB image.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from scripts.trace_input_pipeline import denormalize, describe_batch


def test_describe_batch_stats_on_known_tensor():
    x = torch.full((2, 3, 4, 4), 0.5)
    stats = describe_batch(x)
    assert stats["shape"] == (2, 3, 4, 4)
    assert stats["dtype"] == "torch.float32"
    assert stats["min"] == pytest.approx(0.5)
    assert stats["max"] == pytest.approx(0.5)
    assert stats["mean"] == pytest.approx(0.5)
    assert stats["std"] == pytest.approx(0.0)
    assert sorted(stats["per_channel"]) == ["ch0", "ch1", "ch2"]


def test_describe_batch_rejects_non_nchw():
    with pytest.raises(ValueError):
        describe_batch(torch.zeros(3, 4, 4))  # CHW, not NCHW


def test_describe_batch_does_not_mutate_tensor():
    x = torch.randn(2, 3, 8, 8)
    before = x.clone()
    describe_batch(x)
    assert torch.equal(x, before)


def test_denormalize_zero_returns_imagenet_mean():
    x = torch.zeros(3, 4, 4)  # normalized zero == raw RGB mean
    rgb = denormalize(x)
    assert rgb.shape == (4, 4, 3)
    assert rgb.dtype == np.uint8
    # mean (0.485, 0.456, 0.406) * 255 -> (123.675, 116.28, 103.53), uint8 truncates
    assert tuple(rgb[0, 0].tolist()) == (123, 116, 103)


def test_denormalize_roundtrip_channel_order():
    x = torch.full((3, 4, 4), 0.0)
    x[0] = 1.0  # red channel saturated -> raw value = mean + std = 0.714
    rgb = denormalize(x)
    assert rgb[0, 0, 0] == pytest.approx((0.485 + 0.229) * 255, abs=1)
    assert rgb[0, 0, 1] == 116  # green stays at the mean
    assert rgb[0, 0, 2] == 103  # blue stays at the mean
