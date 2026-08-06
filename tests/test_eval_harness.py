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

from src.data import CLASS_TO_IDX, Sample, SplitManifest  # noqa: E402
from src.evaluate import (  # noqa: E402
    WATCH_CONFUSIONS,
    checkpoint_arch,
    dump_errors,
    load_checkpoint,
    resolve_arch,
    top_confusions,
)
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


@pytest.mark.parametrize("arch", ["efficientnet", "compact"])
def test_checkpoint_arch_reads_the_tag_without_building_the_model(tmp_path, arch):
    """Report provenance: a report that does not name the architecture is not
    comparable against one from a different model."""
    assert checkpoint_arch(_save(tmp_path, arch, arch)) == arch


def test_checkpoint_arch_falls_back_for_legacy_checkpoints(tmp_path):
    assert checkpoint_arch(_save(tmp_path, "compact", arch=None)) == "efficientnet"


def test_dump_errors_writes_only_mistakes_grouped_by_pair(tmp_path):
    """X's 296 errors spread over A/L/R/I with no single culprit (#6).

    That is the one failure shape a confusion matrix cannot explain, so the next
    step is looking at the frames — this is what makes that a one-liner.
    """
    import numpy as np

    src_dir = tmp_path / "frames"
    src_dir.mkdir()
    samples, y_true, y_pred = [], [], []
    for i, (true, pred) in enumerate(
        [("X", "A"), ("X", "A"), ("X", "L"), ("V", "K"), ("Q", "Q")]
    ):
        path = src_dir / f"{true}{i}.jpg"
        path.write_bytes(b"not-really-a-jpeg")  # dump_errors copies, never decodes
        samples.append(Sample(str(path), CLASS_TO_IDX[true], true))
        y_true.append(CLASS_TO_IDX[true])
        y_pred.append(CLASS_TO_IDX[pred])

    out = tmp_path / "errors"
    written = dump_errors(samples, np.array(y_true), np.array(y_pred), str(out))

    assert written == 4  # the Q->Q frame is a correct answer, not an error
    assert sorted(p.name for p in out.iterdir()) == ["V_to_K", "X_to_A", "X_to_L"]
    assert len(list((out / "X_to_A").iterdir())) == 2
    # The ORIGINAL file, not a normalised tensor: the question is what the frame
    # contains, not what the loader did to it.
    assert (out / "V_to_K" / "V3.jpg").read_bytes() == b"not-really-a-jpeg"


def test_dump_errors_caps_each_pair(tmp_path):
    """Without a cap, one 219-error pair buries every other pair in the dir."""
    import numpy as np

    src_dir = tmp_path / "frames"
    src_dir.mkdir()
    samples = []
    for i in range(10):
        path = src_dir / f"V{i}.jpg"
        path.write_bytes(b"x")
        samples.append(Sample(str(path), CLASS_TO_IDX["V"], "V"))
    y_true = np.full(10, CLASS_TO_IDX["V"])
    y_pred = np.full(10, CLASS_TO_IDX["K"])

    out = tmp_path / "errors"
    assert dump_errors(samples, y_true, y_pred, str(out), max_per_pair=3) == 3
    assert len(list((out / "V_to_K").iterdir())) == 3


def test_manifest_path_sits_beside_the_checkpoint():
    assert split_manifest_path("checkpoints/efficientnet_b0.pt") == (
        "checkpoints/efficientnet_b0.split.json"
    )
    assert split_manifest_path("run.pt") == "run.split.json"


def _cm(pairs: dict):
    """Confusion matrix from {(true, pred): count}, diagonal filled to 600."""
    import numpy as np

    n = len(CLASS_TO_IDX)
    cm = np.zeros((n, n), dtype=np.int64)
    for i in range(n):
        cm[i, i] = 600
    for (t, p), count in pairs.items():
        cm[CLASS_TO_IDX[t], CLASS_TO_IDX[p]] = count
        cm[CLASS_TO_IDX[t], CLASS_TO_IDX[t]] -= count
    return cm


def test_top_confusions_ranks_by_size():
    cm = _cm({("V", "K"): 219, ("S", "E"): 160, ("X", "A"): 120, ("M", "N"): 41})

    assert top_confusions(cm, k=3) == [
        (219, "V", "K"), (160, "S", "E"), (120, "X", "A")
    ]


def test_top_confusions_ignores_the_diagonal():
    """A perfect matrix has nothing to report — correct answers are not errors."""
    assert top_confusions(_cm({})) == []


def test_top_confusions_finds_an_unwatched_pair():
    """The point of the ranking: it does not depend on WATCH_CONFUSIONS.

    On the #6 run the worst class (X) was in no watch group, and S->E was split
    across two groups so no group contained both. Both were invisible to the
    group report while being the two largest failures in the matrix.
    """
    unwatched = ("Q", "Z")
    assert not any(set(unwatched) <= set(g) for g in WATCH_CONFUSIONS)

    cm = _cm({unwatched: 300, ("V", "K"): 10})

    assert top_confusions(cm, k=1) == [(300, "Q", "Z")]


def test_measured_confusions_are_watched():
    """The pairs #6 actually measured must each sit inside some group.

    S->E is the regression guard: it was the second-largest failure and the
    original list could not report it, because S was grouped with M/N/T and E
    with A only.
    """
    for pair in [("S", "E"), ("V", "K"), ("X", "A"), ("Y", "I"), ("M", "N")]:
        assert any(set(pair) <= set(g) for g in WATCH_CONFUSIONS), pair


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
