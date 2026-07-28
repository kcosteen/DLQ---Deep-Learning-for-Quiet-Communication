"""Proof that the frame-range split does not leak near-duplicate frames.

The ASL Alphabet images are consecutive video frames of a single signer. A random
split scatters neighbouring (near-identical) frames across train and val, inflating
accuracy to a fake ~99.9%. The frame-range split must instead:

  1. put every file in exactly ONE of train / val (no file in both),
  2. keep frames in capture order so the split is a clean temporal cut,
  3. never place a train frame and a val frame adjacent in capture order.

These tests build a synthetic on-disk dataset (tiny blank images with realistic
Kaggle-style names) and assert all three properties. No torch/mediapipe needed.
"""

from __future__ import annotations

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from src.data import (  # noqa: E402
    CLASSES,
    _frame_group_id,
    frame_range_split,
    list_class_files,
)


@pytest.fixture
def fake_dataset(tmp_path):
    """A 3-class dataset, 100 frames each, named like the Kaggle files."""
    root = tmp_path / "asl_alphabet_train"
    blank = np.zeros((8, 8, 3), dtype=np.uint8)
    for cls in ("A", "B", "space"):
        d = root / cls
        d.mkdir(parents=True)
        for i in range(1, 101):  # A1.jpg .. A100.jpg
            cv2.imwrite(str(d / f"{cls}{i}.jpg"), blank)
    return str(root)


def test_filenames_sorted_in_capture_order(fake_dataset):
    files = list_class_files(fake_dataset, "A")
    # numeric suffix sort => 1,2,...,100  (not lexical 1,10,100,11,...)
    nums = [int("".join(c for c in f.rsplit("/", 1)[-1] if c.isdigit())) for f in files]
    assert nums == sorted(nums) == list(range(1, 101))


def test_no_file_appears_in_both_splits(fake_dataset):
    m = frame_range_split(fake_dataset, train_frac=0.8)
    train_paths = {s.path for s in m.train}
    val_paths = {s.path for s in m.val}
    assert train_paths.isdisjoint(val_paths)


def test_no_frame_group_leaks_across_splits(fake_dataset):
    """Stronger statement: train and val share no physical frame group."""
    m = frame_range_split(fake_dataset, train_frac=0.8)
    train_groups = {_frame_group_id(s.path) for s in m.train}
    val_groups = {_frame_group_id(s.path) for s in m.val}
    assert train_groups.isdisjoint(val_groups)


def test_split_is_a_clean_temporal_cut(fake_dataset):
    """Every train frame precedes every val frame within each class (no interleave).

    This is exactly what a random split would violate — and why it leaks.
    """
    m = frame_range_split(fake_dataset, train_frac=0.8)
    for cls in ("A", "B", "space"):
        order = {p: i for i, p in enumerate(list_class_files(fake_dataset, cls))}
        train_idx = [order[s.path] for s in m.train if s.class_name == cls]
        val_idx = [order[s.path] for s in m.val if s.class_name == cls]
        assert max(train_idx) < min(val_idx), f"train/val interleave in class {cls}"


def test_split_respects_fraction(fake_dataset):
    m = frame_range_split(fake_dataset, train_frac=0.8)
    for cls in ("A", "B", "space"):
        n_train = sum(1 for s in m.train if s.class_name == cls)
        assert n_train == 80  # 80% of 100 frames


def test_all_classes_present_both_sides(fake_dataset):
    m = frame_range_split(fake_dataset, train_frac=0.8)
    train_classes = {s.class_name for s in m.train}
    val_classes = {s.class_name for s in m.val}
    assert train_classes == val_classes == {"A", "B", "space"}


def test_class_index_ordering_is_stable():
    assert CLASSES[0] == "A" and CLASSES[25] == "Z"
    assert CLASSES[26:] == ("space", "delete", "nothing")
    assert len(CLASSES) == 29
