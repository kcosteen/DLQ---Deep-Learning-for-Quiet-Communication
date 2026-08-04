"""Naming + coverage logic for the shared webcam test set (#15).

``data/webcam_testset`` is the project's REAL benchmark: crops the three of us
record ourselves, scored by ``src.evaluate --webcam``. This module holds the two
things the recorder (``scripts/record_webcam_testset.py``) and the coverage
checker (``scripts/check_webcam_coverage.py``) must agree on:

1. **The filename convention.** ``<person>_<condition>_<seq>.jpg`` (the condition
   token is optional: ``<person>_<seq>.jpg``). Provenance lives in the *filename*,
   not in a folder, because of the layout constraint below.
2. **What "covered" means.** Every one of the 29 classes needs samples from all
   three teammates — the Definition of Done of #15.

Layout — why it is flat
-----------------------
``src.evaluate._webcam_samples(root)`` loops over ``CLASSES`` and calls
``data.list_class_files(root, c)``, which lists files DIRECTLY under
``root/<CLASS>/`` and does not recurse. So the scored layout is::

    data/webcam_testset/<CLASS>/<person>_<condition>_<seq>.jpg

A per-person folder (``data/webcam_testset/<person>/<CLASS>/...``) would be
*silently skipped* by the scorer — the eval would report a number over a subset of
the data and never say so. ``find_shadowed_dirs`` exists to catch exactly that.

QUARANTINE: this set never enters training and is never used for model selection.
See ``docs/WEBCAM_TESTSET_PROTOCOL.md``.

Kept import-light (stdlib + ``src.data``): no cv2, no mediapipe, no torch, so the
coverage checker runs anywhere and the tests stay fast.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from .data import CLASSES, IMG_EXTS, list_class_files

DEFAULT_ROOT = "data/webcam_testset"

# How many people must cover every class (the three teammates) and how many
# images each of them needs per class. Both are the #15 Definition of Done.
EXPECTED_PEOPLE = 3
MIN_PER_PERSON_PER_CLASS = 1
# Distinct lighting/background tags per person. Two is the floor the protocol
# asks for; one means the whole capture came from one room under one lamp, which
# is the failure mode this set exists to avoid.
MIN_CONDITIONS_PER_PERSON = 2

# Person ids: lowercase alnum + '-' only. '_' is the field separator in the
# filename, so allowing it in an id would make the prefix ambiguous.
PERSON_RE = re.compile(r"^[a-z0-9][a-z0-9-]*$")
# Condition tags follow the same rule, for the same reason.
CONDITION_RE = PERSON_RE

# The one class where MediaPipe finding no hand is the CORRECT capture: `nothing`
# is the idle gesture, i.e. an empty frame. Everywhere else a no-hand frame means
# the crop fell back to a centre square and the sample is junk.
NO_HAND_CLASS = "nothing"

# Top-level folders that are allowed to hold images without being part of the
# benchmark. ``samples/`` is the one the repo's .gitignore whitelists for a small
# curated illustration; it is not a class folder, so the scorer ignores it and so
# does the layout check.
NON_CLASS_DIRS = frozenset({"samples"})


class CaptureNameError(ValueError):
    """A person id / condition tag / class name that cannot be used in a path."""


def resolve_class(name: str) -> str:
    """Map user input to a canonical ``src.data.CLASSES`` entry.

    Case-insensitive (``a`` -> ``A``, ``SPACE`` -> ``space``) because the shell
    makes exact case tedious. ``CLASSES`` is the only source of truth — nothing
    here hard-codes the 29 names.
    """
    lookup = {c.lower(): c for c in CLASSES}
    key = str(name).strip().lower()
    if key not in lookup:
        raise CaptureNameError(
            f"{name!r} is not one of the 29 classes in src.data.CLASSES. "
            f"Valid: {', '.join(CLASSES)}"
        )
    return lookup[key]


def validate_person(person: str) -> str:
    """Check a person id against ``PERSON_RE`` and return it unchanged."""
    if not PERSON_RE.match(person or ""):
        raise CaptureNameError(
            f"person id {person!r} must be lowercase letters/digits/'-' "
            "(no '_': it separates the fields of the filename)"
        )
    return person


def validate_condition(condition: Optional[str]) -> Optional[str]:
    """Check an optional condition tag; ``None``/empty means 'not recorded'."""
    if condition in (None, ""):
        return None
    if not CONDITION_RE.match(condition):
        raise CaptureNameError(
            f"condition tag {condition!r} must be lowercase letters/digits/'-' "
            "(no '_'), e.g. 'desk-lamp-plainwall'"
        )
    return condition


def capture_filename(
    person: str, seq: int, condition: Optional[str] = None, ext: str = ".jpg"
) -> str:
    """``<person>_<condition>_<seq>.jpg`` (condition omitted when absent)."""
    validate_person(person)
    validate_condition(condition)
    parts = [person] + ([condition] if condition else []) + [f"{seq:03d}"]
    return "_".join(parts) + ext


def parse_capture_name(filename: str) -> Tuple[Optional[str], Optional[str]]:
    """Inverse of :func:`capture_filename`: ``(person, condition)``.

    Three or more tokens => the middle ones are the condition tag; two tokens =>
    no condition. Returns ``(None, None)`` for a name that does not follow the
    convention at all, so the checker can list those files instead of guessing a
    person for them.
    """
    stem = Path(filename).stem
    parts = [p for p in stem.split("_") if p]
    if len(parts) < 2 or not PERSON_RE.match(parts[0]):
        return None, None
    if len(parts) == 2:
        return parts[0], None
    return parts[0], "_".join(parts[1:-1])


def next_seq(out_dir: Path, person: str) -> int:
    """Next free sequence number for ``person`` in a class folder.

    Scans the existing files rather than counting them, so re-running the
    recorder appends to a session instead of overwriting it (and so a deleted
    bad take does not cause a collision).
    """
    highest = 0
    if out_dir.is_dir():
        for p in out_dir.iterdir():
            if p.suffix.lower() not in IMG_EXTS:
                continue
            who, _ = parse_capture_name(p.name)
            if who != person:
                continue
            tail = p.stem.split("_")[-1]
            if tail.isdigit():
                highest = max(highest, int(tail))
    return highest + 1


# --------------------------------------------------------------------------- #
# Coverage
# --------------------------------------------------------------------------- #
@dataclass
class Coverage:
    """(person x class) counts over a webcam-test-set root."""

    root: str
    counts: Dict[str, Dict[str, int]] = field(default_factory=dict)  # person->class->n
    conditions: Dict[str, set] = field(default_factory=dict)         # person->{tags}
    unparsed: List[str] = field(default_factory=list)  # files not following the rule
    total: int = 0

    @property
    def people(self) -> List[str]:
        return sorted(self.counts)

    def count(self, person: str, class_name: str) -> int:
        return self.counts.get(person, {}).get(class_name, 0)

    def classes_missing_for(self, person: str, minimum: int) -> List[str]:
        return [c for c in CLASSES if self.count(person, c) < minimum]


def scan(root: str, classes: Sequence[str] = CLASSES) -> Coverage:
    """Build a :class:`Coverage` from ``root``.

    Deliberately walks with ``data.list_class_files`` — the exact call
    ``src.evaluate._webcam_samples`` makes — so what this reports is what the
    benchmark would score, not an independent re-derivation of the layout.
    """
    cov = Coverage(root=str(root))
    for class_name in classes:
        for path in list_class_files(str(root), class_name):
            cov.total += 1
            person, condition = parse_capture_name(Path(path).name)
            if person is None:
                cov.unparsed.append(path)
                continue
            cov.counts.setdefault(person, {}).setdefault(class_name, 0)
            cov.counts[person][class_name] += 1
            if condition:
                cov.conditions.setdefault(person, set()).add(condition)
    return cov


def find_shadowed_dirs(root: str, classes: Sequence[str] = CLASSES) -> List[str]:
    """Directories holding images that the scorer will never see.

    Two mistakes produce them, and both look fine in a file browser:

    * a per-person folder at the top level (``<root>/<person>/A/...``) — not a
      class name, so ``_webcam_samples`` never opens it;
    * a subfolder inside a class folder (``<root>/A/<person>/...``) — real, but
      ``list_class_files`` does not recurse.

    Returned as paths relative to ``root``.
    """
    root_path = Path(root)
    if not root_path.is_dir():
        return []
    class_names = set(classes)
    shadowed: List[str] = []

    def holds_images(d: Path) -> bool:
        return any(
            p.is_file() and p.suffix.lower() in IMG_EXTS for p in _walk_files(d)
        )

    for entry in sorted(root_path.iterdir()):
        if not entry.is_dir() or entry.name in NON_CLASS_DIRS:
            continue
        if entry.name not in class_names:
            if holds_images(entry):
                shadowed.append(entry.name)
            continue
        for sub in sorted(p for p in entry.iterdir() if p.is_dir()):
            if holds_images(sub):
                shadowed.append(f"{entry.name}/{sub.name}")
    return shadowed


def _walk_files(d: Path):
    try:
        for p in d.rglob("*"):
            yield p
    except OSError:  # unreadable dir: not our problem to report here
        return
