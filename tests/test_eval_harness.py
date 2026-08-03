"""Checkpoint loading in the eval harness (#6).

The harness has to score two architectures on the same split — the transfer
model and the compact-CNN baseline (#5) — so it must rebuild whichever one the
checkpoint actually holds. Loading a compact state dict into an EfficientNet
does not produce a wrong number, it raises; the risk being guarded here is the
opposite one, where an architecture is *assumed* and the mismatch surfaces as an
unreadable key dump instead of a one-line fix.
"""

from __future__ import annotations

import os

import pytest

pytest.importorskip("torch")

import torch  # noqa: E402

from src.data import SplitManifest  # noqa: E402
from src.evaluate import load_checkpoint, resolve_arch  # noqa: E402
from src.model import build_model  # noqa: E402
from src.train import split_manifest_path  # noqa: E402


def _save(tmp_path, name: str, arch: str | None):
    """A real state dict for ``name``, tagged with ``arch`` when given."""
    model = build_model(name, pretrained=False)
    payload = {"model": model.state_dict(), "val_acc": 0.5}
    if arch is not None:
        payload["arch"] = arch
    path = tmp_path / f"{name}.pt"
    torch.save(payload, path)
    return str(path)


def test_arch_comes_from_the_checkpoint():
    assert resolve_arch({"arch": "compact"}, None) == "compact"
    assert resolve_arch({"arch": "efficientnet"}, None) == "efficientnet"


def test_legacy_checkpoint_defaults_to_efficientnet():
    """Checkpoints written before #6 carry no ``arch``.

    The pre-#6 harness could only build EfficientNet, so defaulting to it keeps
    every command that worked against those checkpoints working unchanged.
    """
    assert resolve_arch({"val_acc": 0.9545}, None) == "efficientnet"


def test_explicit_flag_wins_and_warns(capsys):
    assert resolve_arch({"arch": "efficientnet"}, "compact") == "compact"
    assert "overrides" in capsys.readouterr().out


def test_explicit_flag_on_legacy_checkpoint_is_silent(capsys):
    """Nothing to disagree with, so no warning."""
    assert resolve_arch({}, "compact") == "compact"
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize("arch", ["efficientnet", "compact"])
def test_round_trips_both_architectures(tmp_path, arch):
    """A tagged checkpoint loads with no flag, whichever architecture it is."""
    model = load_checkpoint(_save(tmp_path, arch, arch), "cpu")
    out = model(torch.zeros(1, 3, 224, 224))
    assert out.shape == (1, 29)


def test_wrong_architecture_fails_with_a_usable_message(tmp_path):
    """The compact baseline's weights cannot be read as an EfficientNet."""
    path = _save(tmp_path, "compact", arch=None)  # legacy -> assumed efficientnet

    with pytest.raises(SystemExit) as excinfo:
        load_checkpoint(path, "cpu")

    assert "--model" in str(excinfo.value)


def test_manifest_path_sits_beside_the_checkpoint():
    assert split_manifest_path("checkpoints/efficientnet_b0.pt") == (
        "checkpoints/efficientnet_b0.split.json"
    )
    assert split_manifest_path("run.pt") == "run.split.json"


def test_manifest_round_trip_preserves_membership(tmp_path):
    """train.py writes this file and evaluate.py --manifest reads it back.

    ``to_json`` stores paths relative to ``root`` for portability, so the split
    survives the trip only if ``from_json`` rebuilds them against the same root.
    A silent failure here scores the checkpoint on the wrong files — the exact
    outcome the manifest exists to rule out.
    """
    cv2 = pytest.importorskip("cv2")
    import numpy as np

    from src.data import CLASSES, frame_range_split

    root = tmp_path / "asl_alphabet_train"
    for cls in CLASSES:
        d = root / cls
        d.mkdir(parents=True)
        for i in range(1, 11):
            cv2.imwrite(str(d / f"{cls}{i}.jpg"), np.full((8, 8, 3), 20 + i, np.uint8))

    original = frame_range_split(str(root), 0.8)
    path = tmp_path / "run.split.json"
    original.to_json(str(path))
    reloaded = SplitManifest.from_json(str(path))

    # Compared normalised: the manifest stores '/' separators for portability, so
    # on Windows a reloaded path is 'A/A9.jpg' where the original was 'A\A9.jpg'.
    # Both open the same file — the guarantee being checked is which files are in
    # which split, not the spelling of the separator.
    def norm(samples):
        return [os.path.normpath(s.path) for s in samples]

    assert norm(reloaded.val) == norm(original.val)
    assert norm(reloaded.train) == norm(original.train)
    assert [s.label for s in reloaded.train] == [s.label for s in original.train]
    assert len(reloaded.val) == len(original.val) == 2 * len(CLASSES)
    # the paths must still resolve, not merely match as strings
    assert all(os.path.exists(s.path) for s in reloaded.val)
