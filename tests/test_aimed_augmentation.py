"""Aimed augmentation: the geometry-safe policy and per-class routing (#12).

The premise is that the #6 run's three broken classes were broken *by* the
default augmentation, not for want of it. V vs K is where the thumb sits, S vs E
is how far it folds over the fist, X vs A/L/R is how far the index finger hooks —
and ±20° rotation, ±8° shear and six dropout holes at up to 15% of the frame each
are all free to erase exactly that while the label stays put.

So the tests here are about containment as much as about strength: the gentle
policy must actually be gentler on every pixel-moving knob, it must reach only
the classes it was aimed at, and it must never touch the val path — a val image
augmented differently from its neighbours is not a val image any more.
"""

from __future__ import annotations

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")
pytest.importorskip("torch")
pytest.importorskip("albumentations")

from src.augment import POLICIES, AugPolicy, build_transforms, resolve_policy  # noqa: E402
from src.crop import IMG_SIZE  # noqa: E402
from src.data import (  # noqa: E402
    CLASS_TO_IDX,
    CLASSES,
    ASLImageDataset,
    Sample,
    SplitManifest,
    build_datasets,
)

DEFAULT, GENTLE = POLICIES["default"], POLICIES["geometry_safe"]


def _hand_ish(size: int = 200) -> np.ndarray:
    """A blob with a small off-centre 'thumb' — the detail the policies argue over."""
    img = np.zeros((size, size, 3), np.uint8)
    cv2.circle(img, (size // 2, size // 2), size // 4, (180, 150, 140), -1)
    cv2.circle(img, (size // 2 + size // 5, size // 2), size // 12, (200, 170, 160), -1)
    return img


# --------------------------------------------------------------------------
# the policies themselves
# --------------------------------------------------------------------------


def test_gentle_policy_is_gentler_on_every_geometric_knob():
    """One knob left at full strength is enough to smear the thumb back out."""
    assert GENTLE.rotate < DEFAULT.rotate
    assert GENTLE.translate < DEFAULT.translate
    assert GENTLE.shear < DEFAULT.shear
    assert GENTLE.perspective_p < DEFAULT.perspective_p
    assert max(GENTLE.perspective_scale) < max(DEFAULT.perspective_scale)
    # Narrower zoom range, i.e. strictly inside the default's.
    assert GENTLE.scale[0] > DEFAULT.scale[0]
    assert GENTLE.scale[1] < DEFAULT.scale[1]


def test_gentle_dropout_cannot_swallow_a_thumb():
    """6 holes at 15% of the frame can cover the one region separating V from K."""
    assert max(GENTLE.dropout_holes) < min(DEFAULT.dropout_holes)
    assert max(GENTLE.dropout_size) < max(DEFAULT.dropout_size)
    assert GENTLE.dropout_p < DEFAULT.dropout_p


def test_default_policy_still_matches_the_run_it_is_compared_against():
    """#12's before/after is only meaningful if 'before' is unchanged.

    These are the values the #6 run trained with (docs/RESULTS.md). Softening the
    default would silently make every future comparison against that run invalid.
    """
    assert (DEFAULT.rotate, DEFAULT.translate, DEFAULT.shear) == (20.0, 0.08, 8.0)
    assert DEFAULT.scale == (0.85, 1.15)
    assert DEFAULT.dropout_holes == (4, 6)
    assert DEFAULT.dropout_size == (0.05, 0.15)


def test_unknown_policy_is_rejected_with_the_alternatives():
    with pytest.raises(ValueError, match="geometry_safe"):
        resolve_policy("gentle")  # plausible guess, not a real name


def test_resolve_policy_passes_through_an_object():
    custom = AugPolicy("custom", 1, 0.01, (0.99, 1.01), 1, (0.01, 0.02), 0.1,
                       (1, 1), (0.01, 0.02), 0.1)
    assert resolve_policy(custom) is custom


@pytest.mark.parametrize("policy", sorted(POLICIES))
def test_both_policies_end_in_the_shared_contract(policy):
    """Augmentation may only ADD ops before the shared resize+normalize tail.

    Same guarantee as tests/test_preprocessing_parity.py, re-checked per policy:
    whatever a policy does to the pixels, the tensor handed to the model must have
    the shape and scale the webcam app produces.
    """
    out = build_transforms(train=True, policy=policy)(_hand_ish())
    assert tuple(out.shape) == (3, IMG_SIZE, IMG_SIZE)
    assert out.dtype.is_floating_point
    assert -5.0 < float(out.min()) and float(out.max()) < 5.0  # normalized, not 0-255


def test_val_path_ignores_the_policy_and_stays_deterministic():
    """The gentle policy must not leak into evaluation."""
    img = _hand_ish()
    a = build_transforms(train=False, policy="geometry_safe")(img)
    b = build_transforms(train=False, policy="default")(img)
    np.testing.assert_array_equal(a.numpy(), b.numpy())


def test_gentle_policy_displaces_fewer_pixels():
    """The behavioural claim, not just the config: less geometry actually moves.

    Averaged over many draws because both policies are random; the gap between
    ±8° and ±20° rotation (plus shear and perspective) is large enough that the
    mean absolute difference from the unaugmented tensor separates them clearly.
    """
    img = _hand_ish()
    baseline = build_transforms(train=False)(img).numpy()

    def mean_displacement(policy: str) -> float:
        # Photometric ops are identical across policies and add the same noise to
        # both sides, so the comparison stays about geometry.
        rng = [build_transforms(train=True, policy=policy)(img).numpy()
               for _ in range(25)]
        return float(np.mean([np.abs(t - baseline).mean() for t in rng]))

    assert mean_displacement("geometry_safe") < mean_displacement("default")


# --------------------------------------------------------------------------
# per-class routing
# --------------------------------------------------------------------------


def _manifest(tmp_path) -> SplitManifest:
    root = tmp_path / "asl_alphabet_train"
    manifest = SplitManifest(train_frac=0.8, seed=0, root=str(root))
    for cls in CLASSES:
        d = root / cls
        d.mkdir(parents=True)
        for i in range(1, 6):
            path = d / f"{cls}{i}.jpg"
            cv2.imwrite(str(path), _hand_ish(32))
            sample = Sample(str(path), CLASS_TO_IDX[cls], cls)
            (manifest.train if i <= 4 else manifest.val).append(sample)
    return manifest


def test_only_the_named_classes_are_routed_to_the_gentle_policy(tmp_path):
    train_ds, _ = build_datasets(_manifest(tmp_path), geometry_safe_classes=["V", "K"])

    assert sorted(train_ds.class_transforms) == ["K", "V"]
    # One shared gentle transform, not one per class: building an Albumentations
    # pipeline per class would cost 29 of them for no behavioural difference.
    assert len({id(t) for t in train_ds.class_transforms.values()}) == 1
    assert train_ds.class_transforms["V"] is not train_ds.transform


def test_val_dataset_is_never_routed(tmp_path):
    """Even when the caller asks for gentle classes, val stays the contract."""
    _, val_ds = build_datasets(_manifest(tmp_path), geometry_safe_classes=["V", "K"])
    assert val_ds.class_transforms == {}


def test_no_gentle_classes_leaves_the_dataset_untouched(tmp_path):
    """The default path must be exactly what it was before #12."""
    train_ds, val_ds = build_datasets(_manifest(tmp_path))
    assert train_ds.class_transforms == {}
    assert val_ds.class_transforms == {}


def test_routing_dispatches_on_the_sample_class(tmp_path):
    """The mechanism: two frames, two policies, chosen by class name.

    Uses sentinel transforms so the assertion is about which callable ran, not
    about pixels — a real policy could coincidentally produce a similar tensor.
    """
    manifest = _manifest(tmp_path)
    v_sample = next(s for s in manifest.train if s.class_name == "V")
    q_sample = next(s for s in manifest.train if s.class_name == "Q")

    import torch

    seen = []

    def default_t(img):
        seen.append("default")
        return torch.zeros(3, 4, 4)

    def gentle_t(img):
        seen.append("gentle")
        return torch.ones(3, 4, 4)

    ds = ASLImageDataset([v_sample, q_sample], transform=default_t,
                         class_transforms={"V": gentle_t})
    assert float(ds[0][0].mean()) == 1.0  # V -> gentle
    assert float(ds[1][0].mean()) == 0.0  # Q -> default
    assert seen == ["gentle", "default"]


def test_labels_survive_routing(tmp_path):
    """A per-class transform must not disturb the label it is keyed on."""
    manifest = _manifest(tmp_path)
    train_ds, _ = build_datasets(manifest, geometry_safe_classes=["V"])
    for i, sample in enumerate(train_ds.samples[:8]):
        assert train_ds[i][1] == CLASS_TO_IDX[sample.class_name]


def test_typo_in_gentle_classes_is_rejected_at_dataset_build(tmp_path):
    """Guarded here too, not only in the CLI: a silently-empty aim looks like a win."""
    with pytest.raises(ValueError, match="not ASL classes"):
        build_datasets(_manifest(tmp_path), geometry_safe_classes=["V", "thumb"])
