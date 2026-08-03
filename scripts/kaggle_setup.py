"""Build a writable, correctly-named class root from a read-only Kaggle mount.

Two things make ``/kaggle/input`` unusable as ``--root`` directly:

* **It is read-only.** The README's documented fixup for the delete gesture
  (``mv del delete``) cannot be applied in place.
* **The mount path is not stable.** Depending on how the dataset was attached,
  the 29 class folders sit under ``/kaggle/input/asl-alphabet/...`` or under
  ``/kaggle/input/datasets/<owner>/<slug>/...``, with one or two levels of
  ``asl_alphabet_train`` nesting.

So: search the mount for the directory that actually holds the class folders,
then build a directory of per-class symlinks in a writable location, renaming
Kaggle's ``del`` to the ``delete`` that ``src.data.CLASSES`` expects. Symlinks
mean no copy of the 87k images.

Usage (Kaggle notebook)::

    !python scripts/kaggle_setup.py

Prints the root to pass to ``python -m src.train --root ...``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Tuple

# Kaggle ships the delete gesture under this name; src.data.CLASSES wants the
# spelled-out one, and the label-index order depends on it.
KAGGLE_DELETE_DIR = "del"
EXPECTED_DELETE_DIR = "delete"

# A class root must contain at least the letters; the full set is 29 but we
# match loosely so a partially-uploaded mirror still resolves.
MIN_CLASS_DIRS = 26
PROBE_NAMES = {"A", "B", "C"}


def find_class_dir(root: Path, max_depth: int = 6) -> Optional[Tuple[Path, List[Path]]]:
    """Find the directory holding the per-class folders.

    Breadth-first and depth-limited, checking each directory *before* pushing
    its children, so it never descends into a 3000-image class folder.

    ``max_depth`` has to clear the deepest observed layout, which is the
    kagglehub-style mount at depth 5 --
    ``input/datasets/<owner>/<slug>/asl_alphabet_train/asl_alphabet_train`` --
    plus a level of headroom.
    """
    queue: List[Tuple[Path, int]] = [(root, 0)]
    while queue:
        d, depth = queue.pop(0)
        if depth > max_depth:
            continue
        try:
            subs = sorted(p for p in d.iterdir() if p.is_dir())
        except (PermissionError, FileNotFoundError):
            continue
        names = {p.name for p in subs}
        if PROBE_NAMES <= names and len(subs) >= MIN_CLASS_DIRS:
            return d, subs
        queue += [(p, depth + 1) for p in subs]
    return None


def link_classes(subs: List[Path], out: Path) -> List[str]:
    """Symlink each class folder into ``out``, renaming ``del`` -> ``delete``."""
    out.mkdir(parents=True, exist_ok=True)
    linked = []
    for p in subs:
        name = EXPECTED_DELETE_DIR if p.name == KAGGLE_DELETE_DIR else p.name
        link = out / name
        if not link.exists():
            try:
                link.symlink_to(p)
            except OSError as e:  # Windows needs admin / Developer Mode
                raise OSError(
                    f"could not symlink {link} -> {p}: {e}\n"
                    "This script targets the Linux Kaggle/Colab runtime. On "
                    "Windows, enable Developer Mode or just copy the class "
                    "folders and rename 'del' to 'delete' by hand."
                ) from e
        linked.append(name)
    return sorted(linked)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Resolve a Kaggle ASL mount into a writable class root."
    )
    ap.add_argument("--input", default="/kaggle/input",
                    help="read-only mount to search (default: /kaggle/input)")
    ap.add_argument("--out", default="/kaggle/working/asl_train",
                    help="writable root of symlinks to create")
    args = ap.parse_args()

    found = find_class_dir(Path(args.input))
    if found is None:
        print(f"ERROR: no directory with >= {MIN_CLASS_DIRS} class folders under "
              f"{args.input}", file=sys.stderr)
        print("Is the dataset attached? (+ Add Input -> grassknoted/asl-alphabet)",
              file=sys.stderr)
        return 1

    src, subs = found
    print(f"class dir : {src}")
    print(f"found     : {len(subs)} class folders")

    linked = link_classes(subs, Path(args.out))
    renamed = KAGGLE_DELETE_DIR in {p.name for p in subs}
    if renamed:
        print(f"renamed   : {KAGGLE_DELETE_DIR!r} -> {EXPECTED_DELETE_DIR!r}")
    print(f"linked    : {len(linked)} classes -> {args.out}")
    print(f"classes   : {linked}")
    print()
    print(f"Now run:  python -m src.train --root {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
