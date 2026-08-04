"""The webcam test set's layout, naming and coverage rules (#15).

The webcam set is the REAL benchmark, and its worst failure mode is silent:
``src.evaluate._webcam_samples`` lists files directly under ``<root>/<CLASS>/``
via ``data.list_class_files`` and does NOT recurse, so a tidy-looking per-person
folder contributes zero images to the reported number while the eval still prints
a plausible accuracy. The layout tests below pin that down from both sides — what
the scorer sees, and what the coverage checker refuses to pass.

No mediapipe, no torch: the coverage logic is deliberately import-light so it can
gate CI. Only the two capture-loop tests need cv2, for ``imwrite``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from src.data import CLASSES, list_class_files
from src.webcam_testset import (
    MIN_CONDITIONS_PER_PERSON,
    CaptureNameError,
    capture_filename,
    find_shadowed_dirs,
    next_seq,
    parse_capture_name,
    resolve_class,
    scan,
)

PEOPLE = ("qdoan", "kcosteen", "jsmith")


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")  # content irrelevant: these tests only walk the layout


def _flat_set(root: Path, people=PEOPLE, classes=CLASSES, condition="lamp-wall"):
    """The layout evaluate.py scores: <root>/<CLASS>/<person>_<cond>_<seq>.jpg."""
    for cls in classes:
        for person in people:
            _touch(root / cls / capture_filename(person, 1, condition))
    return root


# --------------------------------------------------------------------------- #
# Naming
# --------------------------------------------------------------------------- #
def test_name_round_trips_with_and_without_a_condition():
    assert capture_filename("qdoan", 7) == "qdoan_007.jpg"
    assert capture_filename("qdoan", 7, "desklamp-plainwall") == \
        "qdoan_desklamp-plainwall_007.jpg"
    assert parse_capture_name("qdoan_007.jpg") == ("qdoan", None)
    assert parse_capture_name("qdoan_desklamp-plainwall_007.jpg") == \
        ("qdoan", "desklamp-plainwall")


def test_unconventional_names_are_reported_not_guessed():
    """A file we cannot attribute must not be silently credited to someone."""
    assert parse_capture_name("IMG_20260804.jpg")[0] is None
    assert parse_capture_name("A1.jpg") == (None, None)


def test_person_id_cannot_contain_the_separator():
    with pytest.raises(CaptureNameError):
        capture_filename("q_doan", 1)
    with pytest.raises(CaptureNameError):
        capture_filename("qdoan", 1, "bad_tag")


def test_class_names_come_from_src_data_classes():
    assert resolve_class("a") == "A"
    assert resolve_class("SPACE") == "space"
    assert resolve_class("delete") == "delete"
    with pytest.raises(CaptureNameError):
        resolve_class("del")  # the Kaggle name; CLASSES says 'delete'
    with pytest.raises(CaptureNameError):
        resolve_class("QQ")


def test_next_seq_appends_and_never_reuses(tmp_path):
    d = tmp_path / "A"
    assert next_seq(d, "qdoan") == 1
    _touch(d / capture_filename("qdoan", 1, "lamp-wall"))
    _touch(d / capture_filename("qdoan", 2, "lamp-wall"))
    _touch(d / capture_filename("kcosteen", 9, "sun-kitchen"))
    assert next_seq(d, "qdoan") == 3          # per person, not per folder
    assert next_seq(d, "kcosteen") == 10
    (d / capture_filename("qdoan", 1, "lamp-wall")).unlink()  # a deleted bad take
    assert next_seq(d, "qdoan") == 3          # max+1, so 002 is never overwritten


# --------------------------------------------------------------------------- #
# Layout — the flat class folders evaluate.py scores
# --------------------------------------------------------------------------- #
def test_scan_sees_exactly_what_the_scorer_lists(tmp_path):
    root = _flat_set(tmp_path / "webcam_testset")
    listed = sum(len(list_class_files(str(root), c)) for c in CLASSES)
    cov = scan(str(root))
    assert cov.total == listed == len(CLASSES) * len(PEOPLE)
    assert cov.people == sorted(PEOPLE)
    assert all(cov.count(p, c) == 1 for p in PEOPLE for c in CLASSES)


def test_per_person_folders_are_invisible_to_the_scorer(tmp_path):
    """The layout trap #15 must not fall into, proven against the real lister."""
    root = tmp_path / "webcam_testset"
    for cls in ("A", "B"):
        _touch(root / "qdoan" / cls / "qdoan_lamp-wall_001.jpg")   # top-level person
        _touch(root / cls / "kcosteen" / "kcosteen_sun-kitchen_001.jpg")  # nested
    assert sum(len(list_class_files(str(root), c)) for c in CLASSES) == 0
    assert scan(str(root)).total == 0
    assert set(find_shadowed_dirs(str(root))) == {"qdoan", "A/kcosteen", "B/kcosteen"}


def test_a_correct_set_shadows_nothing(tmp_path):
    root = _flat_set(tmp_path / "webcam_testset")
    _touch(root / "samples" / "illustration.jpg")  # whitelisted, not a class
    assert find_shadowed_dirs(str(root)) == []


# --------------------------------------------------------------------------- #
# Coverage — the #15 Definition of Done
# --------------------------------------------------------------------------- #
def test_missing_class_for_one_person_is_detected(tmp_path):
    root = _flat_set(tmp_path / "webcam_testset")
    (root / "Z" / capture_filename("jsmith", 1, "lamp-wall")).unlink()
    cov = scan(str(root))
    assert cov.classes_missing_for("jsmith", 1) == ["Z"]
    assert cov.classes_missing_for("qdoan", 1) == []
    # the class still has images, so a class-level count alone would miss this
    assert sum(cov.count(p, "Z") for p in PEOPLE) == 2


def test_conditions_are_tracked_per_person(tmp_path):
    root = _flat_set(tmp_path / "webcam_testset", condition="lamp-wall")
    _touch(root / "A" / capture_filename("qdoan", 50, "outdoor-daylight"))
    cov = scan(str(root))
    assert cov.conditions["qdoan"] == {"lamp-wall", "outdoor-daylight"}
    # everyone else is still single-condition -> below the protocol floor
    assert len(cov.conditions["jsmith"]) < MIN_CONDITIONS_PER_PERSON


def test_untagged_files_do_not_count_as_a_condition(tmp_path):
    root = tmp_path / "webcam_testset"
    _touch(root / "A" / capture_filename("qdoan", 1))
    cov = scan(str(root))
    assert cov.count("qdoan", "A") == 1
    assert cov.conditions.get("qdoan", set()) == set()


# --------------------------------------------------------------------------- #
# The capture loop: what actually lands on disk
# --------------------------------------------------------------------------- #
cv2 = pytest.importorskip("cv2")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import record_webcam_testset as rec  # noqa: E402


class _FakeResult:
    def __init__(self, found_hand: bool) -> None:
        self.crop_bgr = np.full((224, 224, 3), 128, np.uint8)
        self.bbox = (1, 2, 3, 4) if found_hand else None
        self.found_hand = found_hand


class _FakeCropper:
    """Stands in for HandCropper — no mediapipe, no camera."""

    def __init__(self, found_hand: bool) -> None:
        self.found_hand = found_hand

    def crop(self, frame):
        return _FakeResult(self.found_hand)


class _FakeCap:
    def read(self):
        return True, np.zeros((480, 640, 3), np.uint8)


def _args(tmp_path, **over):
    ns = rec.build_parser().parse_args(
        ["--person", "qdoan", "--class", "A", "--condition", "lamp-wall",
         "--out", str(tmp_path), "--count", "2"]
    )
    for k, v in over.items():
        setattr(ns, k, v)
    return ns


def _drive(monkeypatch, keys):
    """Feed a fixed key sequence to the capture loop, with no GUI."""
    it = iter(keys)
    monkeypatch.setattr(rec.cv2, "imshow", lambda *a, **k: None)
    monkeypatch.setattr(rec.cv2, "waitKey", lambda *a, **k: next(it, ord("q")))


def test_capture_writes_the_crop_where_evaluate_looks(tmp_path, monkeypatch):
    _drive(monkeypatch, [ord(" "), ord(" "), ord("n")])
    n, quit_now = rec.record_class(_FakeCap(), _FakeCropper(True), "A",
                                   _args(tmp_path))
    assert (n, quit_now) == (2, False)
    files = list_class_files(str(tmp_path), "A")  # the scorer's own lister
    assert [Path(f).name for f in files] == [
        "qdoan_lamp-wall_001.jpg", "qdoan_lamp-wall_002.jpg"
    ]
    img = cv2.imread(files[0], cv2.IMREAD_COLOR)
    assert img.shape == (224, 224, 3)  # the frozen src/crop.py contract


def test_no_hand_frames_are_not_saved(tmp_path, monkeypatch):
    """A no-hand crop is a centre-square fallback, not a sample of the sign."""
    _drive(monkeypatch, [ord(" "), ord(" "), ord("n")])
    n, _ = rec.record_class(_FakeCap(), _FakeCropper(False), "A", _args(tmp_path))
    assert n == 0
    assert list_class_files(str(tmp_path), "A") == []


def test_nothing_inverts_the_rule(tmp_path, monkeypatch):
    """`nothing` is the idle gesture: the empty frame IS the correct sample."""
    _drive(monkeypatch, [ord(" "), ord("n")])
    n, _ = rec.record_class(_FakeCap(), _FakeCropper(False), "nothing",
                            _args(tmp_path))
    assert n == 1
    _drive(monkeypatch, [ord(" "), ord("n")])
    n, _ = rec.record_class(_FakeCap(), _FakeCropper(True), "nothing",
                            _args(tmp_path))
    assert n == 0  # a hand in frame is not `nothing`


def test_undo_removes_the_last_capture(tmp_path, monkeypatch):
    _drive(monkeypatch, [ord(" "), ord(" "), ord("u"), ord("n")])
    n, _ = rec.record_class(_FakeCap(), _FakeCropper(True), "A", _args(tmp_path))
    assert n == 1
    assert len(list_class_files(str(tmp_path), "A")) == 1
