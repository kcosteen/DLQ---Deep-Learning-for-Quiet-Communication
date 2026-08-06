"""Pre-flight guards on the training entry point (#6).

The transfer run costs GPU hours, so the checks that matter are the ones that
fail *before* it starts. The specific trap guarded here: ``frame_range_split``
skips a class whose folder is absent (deliberate — a partial webcam capture
should still split), which means a full training run can quietly fit a 29-way
head to 28 classes and still report a believable dev-val accuracy.
"""

from __future__ import annotations

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")
pytest.importorskip("torch")

from src.data import CLASSES, Sample, SplitManifest, frame_range_split  # noqa: E402
from src.train import assert_all_classes_present  # noqa: E402


def _write_class(root, cls: str, n: int = 10) -> None:
    """``n`` Kaggle-named frames (``A1.jpg``..) of a flat colour."""
    d = root / cls
    d.mkdir(parents=True, exist_ok=True)
    for i in range(1, n + 1):
        cv2.imwrite(str(d / f"{cls}{i}.jpg"), np.full((8, 8, 3), 20 + i, np.uint8))


@pytest.fixture
def full_root(tmp_path):
    """A root covering all 29 classes."""
    root = tmp_path / "asl_alphabet_train"
    for cls in CLASSES:
        _write_class(root, cls)
    return root


def test_complete_split_passes(full_root):
    """All 29 present -> no complaint."""
    assert_all_classes_present(frame_range_split(str(full_root), 0.8))


def test_kaggle_del_folder_is_rejected(full_root):
    """Kaggle ships the delete gesture as 'del'; CLASSES expects 'delete'.

    This is the read-only ``/kaggle/input`` case: the folder cannot be renamed
    in place, so without the guard the class is silently dropped.
    """
    (full_root / "delete").rename(full_root / "del")

    with pytest.raises(ValueError) as excinfo:
        assert_all_classes_present(frame_range_split(str(full_root), 0.8))

    msg = str(excinfo.value)
    assert "delete" in msg
    assert "del" in msg  # the hint names the Kaggle folder
    assert str(full_root) in msg  # points at the offending root


def test_missing_letter_is_rejected(full_root):
    """Not delete-specific — any absent class stops the run."""
    for f in (full_root / "Q").iterdir():
        f.unlink()
    (full_root / "Q").rmdir()

    with pytest.raises(ValueError, match="Q"):
        assert_all_classes_present(frame_range_split(str(full_root), 0.8))


def test_single_frame_class_is_rejected(full_root):
    """One frame -> ``cut = int(1 * 0.8) = 0``, so the class is all val, no train.

    A class the model never sees a training example of is exactly as broken as
    an absent one, and just as invisible in the reported accuracy.
    """
    for f in (full_root / "Z").iterdir():
        f.unlink()
    _write_class(full_root, "Z", n=1)

    with pytest.raises(ValueError, match=r"absent from train.*'Z'"):
        assert_all_classes_present(frame_range_split(str(full_root), 0.8))


def test_val_side_is_checked_too():
    """The val branch is unreachable via ``frame_range_split``.

    Every file is its own frame group (``_frame_group_id`` hashes the path), so
    ``cut = int(n * train_frac) < n`` always leaves val non-empty. It stays
    reachable through a hand-written ``--manifest``, so it is checked directly.
    """
    manifest = SplitManifest(train_frac=0.8, seed=0, root="synthetic")
    manifest.train = [
        Sample(f"{c}.jpg", i, c) for i, c in enumerate(CLASSES)
    ]
    manifest.val = manifest.train[:-1]  # last class missing from val only

    with pytest.raises(ValueError, match="absent from val"):
        assert_all_classes_present(manifest)
