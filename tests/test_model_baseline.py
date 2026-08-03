"""The from-scratch baseline and the logistic-regression sanity floor (#5).

Two reference numbers bracket every later result:

  * the **floor** — logistic regression on 32x32 grayscale pixels. Anything that
    cannot beat this is not learning; it is a wiring bug.
  * the **baseline** — CompactCNN trained from scratch. The EfficientNet transfer
    model has to beat *this* to justify itself (#6/#8).

These tests use a synthetic on-disk dataset (per-class geometric patterns, no
Kaggle download) so they run anywhere. They assert the machinery is correct and
that the floor separates a learnable signal from noise — not that any particular
accuracy is reached on the real data.
"""

from __future__ import annotations

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")
torch = pytest.importorskip("torch")
pytest.importorskip("sklearn")

from src.data import CLASSES, frame_range_split  # noqa: E402
from src.model import (  # noqa: E402
    NUM_CLASSES,
    PIXEL_SUBSET,
    CompactCNN,
    build_model,
    count_parameters,
    fit_logreg_sanity,
    pixel_features,
)

SYNTH_CLASSES = ("A", "B", "space")


def _patterned_image(class_idx: int, frame: int) -> np.ndarray:
    """A 32x32 BGR image whose structure depends on the class, plus frame drift.

    Each class gets a distinct filled region, so the pattern is linearly
    separable from raw pixels. The slow per-frame brightness drift mimics the
    real dataset's consecutive-video-frame character, so neighbouring frames are
    near-duplicates of each other.
    """
    img = np.zeros((32, 32, 3), dtype=np.uint8)
    drift = 40 + (frame % 10)
    if class_idx == 0:
        img[:16, :, :] = 200 + (frame % 5)  # top half
    elif class_idx == 1:
        img[:, :16, :] = 200 + (frame % 5)  # left half
    else:
        img[8:24, 8:24, :] = 200 + (frame % 5)  # centre block
    img[img == 0] = drift
    return img


@pytest.fixture
def synth_dataset(tmp_path):
    """3 classes x 60 Kaggle-named frames of class-dependent patterns."""
    root = tmp_path / "asl_alphabet_train"
    for idx, cls in enumerate(SYNTH_CLASSES):
        d = root / cls
        d.mkdir(parents=True)
        for i in range(1, 61):
            cv2.imwrite(str(d / f"{cls}{i}.jpg"), _patterned_image(idx, i))
    return str(root)


# --- model definitions -------------------------------------------------------


def test_compact_cnn_output_shape():
    model = CompactCNN()
    out = model(torch.zeros(2, 3, 224, 224))
    assert out.shape == (2, NUM_CLASSES)


def test_num_classes_matches_the_class_list():
    """The head width and the label-index order must not drift apart."""
    assert NUM_CLASSES == len(CLASSES) == 29


def test_build_model_dispatches_compact():
    assert isinstance(build_model("compact"), CompactCNN)


def test_build_model_rejects_unknown_name():
    with pytest.raises(ValueError):
        build_model("resnet50")


def test_compact_cnn_is_small_enough_to_be_a_baseline():
    """The baseline must stay far smaller than EfficientNet-B0 (~5.3M params)."""
    assert count_parameters(CompactCNN()) < 1_000_000


# --- pixel features ----------------------------------------------------------


def test_pixel_features_shape_and_range():
    feats = pixel_features(_patterned_image(0, 1))
    assert feats.shape == (PIXEL_SUBSET * PIXEL_SUBSET,)
    assert feats.dtype == np.float32
    assert 0.0 <= feats.min() and feats.max() <= 1.0


def test_pixel_features_separate_the_classes():
    """Different classes must not collapse to the same feature vector."""
    a = pixel_features(_patterned_image(0, 1))
    b = pixel_features(_patterned_image(1, 1))
    assert not np.allclose(a, b)


# --- the sanity floor --------------------------------------------------------


def _featurize(paths_labels):
    X = np.stack([pixel_features(cv2.imread(p, cv2.IMREAD_COLOR)) for p, _ in paths_labels])
    y = np.asarray([lab for _, lab in paths_labels])
    return X, y


def test_logreg_floor_learns_a_separable_signal(synth_dataset):
    """On a learnable task the floor must land well above chance (1/3)."""
    m = frame_range_split(synth_dataset, train_frac=0.8, classes=SYNTH_CLASSES)
    X_train, y_train = _featurize([(s.path, s.label) for s in m.train])
    X_val, y_val = _featurize([(s.path, s.label) for s in m.val])

    _, acc = fit_logreg_sanity(X_train, y_train, X_val, y_val)
    assert acc > 0.8, f"floor failed to learn a separable pattern (acc={acc})"


def test_logreg_floor_is_near_chance_on_shuffled_labels(synth_dataset):
    """Sanity check on the floor itself: no signal => no accuracy.

    If this passed with a high score the floor would be leaking, and every
    'we beat the floor' claim built on it would be meaningless.
    """
    m = frame_range_split(synth_dataset, train_frac=0.8, classes=SYNTH_CLASSES)
    X_train, y_train = _featurize([(s.path, s.label) for s in m.train])
    X_val, y_val = _featurize([(s.path, s.label) for s in m.val])

    rng = np.random.default_rng(0)
    y_train_shuffled = rng.permutation(y_train)

    _, acc = fit_logreg_sanity(X_train, y_train_shuffled, X_val, y_val)
    assert acc < 0.6, f"floor scored {acc} on shuffled labels — it is leaking"


def test_compute_sanity_floor_end_to_end(synth_dataset):
    """The train.py entry point wires split -> features -> floor correctly."""
    from src.train import compute_sanity_floor

    m = frame_range_split(synth_dataset, train_frac=0.8, classes=SYNTH_CLASSES)
    acc = compute_sanity_floor(m, max_per_class=40)
    assert 0.0 <= acc <= 1.0
    assert acc > 0.8


def test_sanity_floor_subsample_is_deterministic(synth_dataset):
    """Same split in, same floor out — the number has to be reproducible."""
    from src.train import compute_sanity_floor

    m = frame_range_split(synth_dataset, train_frac=0.8, classes=SYNTH_CLASSES)
    assert compute_sanity_floor(m, max_per_class=20) == compute_sanity_floor(
        m, max_per_class=20
    )
