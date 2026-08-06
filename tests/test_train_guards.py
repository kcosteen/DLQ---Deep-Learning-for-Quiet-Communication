"""Pre-flight guards on the training entry point (#6, #12).

The transfer run costs GPU hours, so the checks that matter are the ones that
fail *before* it starts. The specific trap guarded here: ``frame_range_split``
skips a class whose folder is absent (deliberate — a partial webcam capture
should still split), which means a full training run can quietly fit a 29-way
head to 28 classes and still report a believable dev-val accuracy.

The #12 guards are about the other irreplaceable thing a run can destroy: the
baseline checkpoint it is being compared against.
"""

from __future__ import annotations

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")
pytest.importorskip("torch")

import torch  # noqa: E402

from src.data import CLASSES, Sample, SplitManifest, frame_range_split  # noqa: E402
from src.model import build_model  # noqa: E402
from src.train import (  # noqa: E402
    assert_all_classes_present,
    assert_out_does_not_clobber_resume,
    default_checkpoint_path,
    load_resume_state,
)


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


# --------------------------------------------------------------------------
# #12: do not destroy the checkpoint you are being compared against
# --------------------------------------------------------------------------


def test_resuming_onto_the_same_path_is_refused():
    """``_run_stage`` saves on the first epoch that beats best=0.0 — i.e. always."""
    with pytest.raises(SystemExit, match="overwrite the checkpoint being resumed"):
        assert_out_does_not_clobber_resume(
            "checkpoints/efficientnet_b0.pt", "checkpoints/efficientnet_b0.pt"
        )


def test_the_same_path_spelled_differently_is_still_the_same_path():
    """A relative --out and an absolute --resume must not slip past the guard."""
    import os

    target = os.path.abspath("checkpoints/efficientnet_b0.pt")
    with pytest.raises(SystemExit):
        assert_out_does_not_clobber_resume(
            "checkpoints/./efficientnet_b0.pt", target
        )


def test_overwrite_can_be_asked_for_explicitly():
    assert_out_does_not_clobber_resume("a.pt", "a.pt", allow_overwrite=True)


def test_distinct_paths_and_plain_runs_pass():
    assert_out_does_not_clobber_resume("b.pt", "a.pt")
    assert_out_does_not_clobber_resume("a.pt", None)  # not resuming at all


def test_targeted_runs_get_their_own_default_filename():
    """Otherwise a targeted re-run lands on the baseline it must be compared to."""
    baseline = default_checkpoint_path("efficientnet", targeted=False)
    targeted = default_checkpoint_path("efficientnet", targeted=True)
    assert baseline == "checkpoints/efficientnet_b0.pt"
    assert targeted != baseline
    assert default_checkpoint_path("compact", targeted=False) == (
        "checkpoints/compact_cnn.pt"
    )


def _ckpt(tmp_path, arch: str, val_acc: float = 0.9545):
    payload = {"model": build_model(arch, pretrained=False).state_dict(),
               "val_acc": val_acc, "arch": arch}
    path = tmp_path / f"{arch}.pt"
    torch.save(payload, path)
    return str(path)


def test_resume_loads_weights_and_reports_the_recorded_accuracy(tmp_path):
    """The recorded dev-val is what the targeted run has to beat."""
    path = _ckpt(tmp_path, "compact", val_acc=0.9545)
    model = build_model("compact", pretrained=False)

    assert load_resume_state(model, path, "compact") == pytest.approx(0.9545)

    saved = torch.load(path, map_location="cpu")["model"]
    for name, param in model.state_dict().items():
        torch.testing.assert_close(param, saved[name])


def test_resuming_across_architectures_is_refused(tmp_path):
    """Otherwise torch reports it as thousands of unmatched parameter names."""
    path = _ckpt(tmp_path, "compact")
    model = build_model("efficientnet", pretrained=False)

    with pytest.raises(SystemExit, match="'compact' checkpoint but --model is"):
        load_resume_state(model, path, "efficientnet")
