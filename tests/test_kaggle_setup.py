"""Resolving a Kaggle mount into a usable class root (#6).

The mount path is not stable across attach methods, and the depth limit in
``find_class_dir`` is the part that silently breaks: too shallow and it reports
"dataset not attached" for a dataset that is, in fact, attached.
"""

from __future__ import annotations

import pytest

from scripts.kaggle_setup import (  # noqa: E402
    EXPECTED_DELETE_DIR,
    KAGGLE_DELETE_DIR,
    find_class_dir,
    link_classes,
)

CLASS_DIRS = (
    [chr(c) for c in range(ord("A"), ord("Z") + 1)]
    + ["space", "nothing", KAGGLE_DELETE_DIR]
)


def _build_mount(root, nesting: str):
    """Create ``<root>/<nesting>/{A..Z,space,nothing,del}`` with one file each."""
    base = root
    for part in nesting.split("/"):
        base = base / part
    for cls in CLASS_DIRS:
        d = base / cls
        d.mkdir(parents=True)
        (d / f"{cls}1.jpg").write_bytes(b"x")
    return base


# The two layouts seen in practice: the classic per-slug mount, and the deeper
# kagglehub-style one that broke the first version of the finder.
@pytest.mark.parametrize("nesting", [
    "asl-alphabet/asl_alphabet_train/asl_alphabet_train",
    "datasets/grassknoted/asl-alphabet/asl_alphabet_train/asl_alphabet_train",
])
def test_finds_class_dir_at_both_mount_depths(tmp_path, nesting):
    expected = _build_mount(tmp_path, nesting)

    found = find_class_dir(tmp_path)

    assert found is not None, f"did not find the class dir under {nesting}"
    where, subs = found
    assert where == expected
    assert len(subs) == 29


def test_returns_none_when_nothing_is_attached(tmp_path):
    (tmp_path / "some-other-dataset").mkdir()
    assert find_class_dir(tmp_path) is None


def test_does_not_descend_into_class_folders(tmp_path):
    """A folder of images must not be mistaken for the class root."""
    base = _build_mount(tmp_path, "asl-alphabet/asl_alphabet_train")
    found = find_class_dir(tmp_path)
    assert found is not None
    assert found[0] == base  # the parent of A/B/C, not A itself


def test_del_is_linked_as_delete(tmp_path, monkeypatch):
    """The whole point: Kaggle's 'del' must arrive as 'delete'.

    Symlinks need elevation on Windows, so the link call is stubbed — what is
    under test is the name mapping, not the OS primitive.
    """
    base = _build_mount(tmp_path, "asl-alphabet/asl_alphabet_train")
    made = {}

    def fake_symlink(self, target, target_is_directory=False):
        made[self.name] = target.name

    monkeypatch.setattr("pathlib.Path.symlink_to", fake_symlink)
    linked = link_classes(sorted(p for p in base.iterdir()), tmp_path / "out")

    assert EXPECTED_DELETE_DIR in linked
    assert KAGGLE_DELETE_DIR not in linked
    assert made[EXPECTED_DELETE_DIR] == KAGGLE_DELETE_DIR  # points at 'del'
    assert len(linked) == 29
    assert made["A"] == "A"  # every other class passes through unchanged
