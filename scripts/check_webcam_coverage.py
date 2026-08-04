#!/usr/bin/env python3
"""Verify the shared webcam test set: coverage, layout, and quarantine (#15).

Walks ``data/webcam_testset`` with ``src.data.list_class_files`` — the exact call
``src.evaluate._webcam_samples`` makes — parses the person prefix out of each
filename, and prints a (class x person) count matrix.

It then reports the two Definition-of-Done conditions of issue #15:

  (a) every one of the 29 classes has samples from all three teammates;
  (b) the files sit in the layout ``src.evaluate --webcam`` actually scores.

(b) is the one that fails quietly in real life. ``list_class_files`` does not
recurse, so a per-person folder (``<root>/qdoan/A/...``) or a nested subfolder
(``<root>/A/qdoan/...``) contributes ZERO images to the benchmark while looking
perfectly organised in a file browser. Those directories are reported as errors,
not warnings.

**Exit code is 0 only when both conditions hold**, so this can gate CI once the
captures exist.

QUARANTINE: this set never enters training and is never used for model selection.
See ``docs/WEBCAM_TESTSET_PROTOCOL.md``.

Examples::

    python scripts/check_webcam_coverage.py
    python scripts/check_webcam_coverage.py --people qdoan kcosteen jsmith
    python scripts/check_webcam_coverage.py --min-per-person 5 --quiet
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:  # allow `python scripts/check_webcam_coverage.py`
    sys.path.insert(0, str(REPO_ROOT))

from src.crop import IMG_SIZE  # noqa: E402
from src.data import CLASSES  # noqa: E402
from src.webcam_testset import (  # noqa: E402
    DEFAULT_ROOT,
    EXPECTED_PEOPLE,
    MIN_CONDITIONS_PER_PERSON,
    MIN_PER_PERSON_PER_CLASS,
    Coverage,
    find_shadowed_dirs,
    scan,
)

QUARANTINE_LINE = (
    "QUARANTINE: never used for training or model selection — "
    "see docs/WEBCAM_TESTSET_PROTOCOL.md"
)


def print_matrix(cov: Coverage, people: Sequence[str], minimum: int) -> None:
    """(class x person) counts. Rows are classes so 29 of them fit a terminal."""
    width = max(8, max((len(p) for p in people), default=8) + 1)
    header = f"{'class':>8} " + "".join(f"{p:>{width}}" for p in people) + f"{'TOTAL':>8}"
    print(header)
    print("-" * len(header))
    for c in CLASSES:
        cells = ""
        for p in people:
            n = cov.count(p, c)
            cells += f"{(str(n) if n else '.') + ('' if n >= minimum else '!'):>{width}}"
        total = sum(cov.count(p, c) for p in people)
        print(f"{c:>8} {cells}{total:>8}")
    print("-" * len(header))
    totals = "".join(
        f"{sum(cov.count(p, c) for c in CLASSES):>{width}}" for p in people
    )
    print(f"{'TOTAL':>8} {totals}{cov.total:>8}")
    print("\n('.' = none, '!' = below the per-person minimum of "
          f"{minimum} image(s) for that class)")


def check_image_sizes(root: str) -> List[str]:
    """Paths whose image is not the contract's ``IMG_SIZE`` square.

    Anything produced by ``HandCropper.crop(...).crop_bgr`` is 224x224 by
    construction, so an odd size means a file got in without going through the
    shared crop — i.e. the test framing no longer matches the training framing.
    Returns [] (and skips the check) when cv2 is unavailable.
    """
    try:
        import cv2
    except ImportError:
        print("note: cv2 unavailable — skipped the 224x224 crop-contract check")
        return []
    from src.data import list_class_files

    bad = []
    for class_name in CLASSES:
        for path in list_class_files(root, class_name):
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            if img is None:
                bad.append(f"{path} (unreadable)")
            elif img.shape[0] != IMG_SIZE or img.shape[1] != IMG_SIZE:
                bad.append(f"{path} ({img.shape[1]}x{img.shape[0]})")
    return bad


def crosscheck_with_evaluate(root: str, expected_total: int) -> Optional[str]:
    """Confirm ``src.evaluate._webcam_samples`` sees the same files we counted.

    The direct proof of DoD (b). ``src.evaluate`` imports torch at module level,
    so this is best-effort: without torch installed the layout is still verified,
    just via ``list_class_files`` (the same function the scorer calls).
    """
    try:
        from src.evaluate import _webcam_samples
    except Exception as e:  # torch missing, etc.
        return f"skipped (could not import src.evaluate: {type(e).__name__})"
    n = len(_webcam_samples(root))
    if n != expected_total:
        return (f"MISMATCH: evaluate._webcam_samples sees {n} images, this scan "
                f"counted {expected_total}")
    return f"OK: evaluate._webcam_samples sees the same {n} images"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Check webcam-test-set coverage + layout (#15). Exits non-zero "
                    "if the Definition of Done is not met, so it can gate CI.",
        epilog="Record with: python scripts/record_webcam_testset.py --help",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--root", default=DEFAULT_ROOT,
                   help=f"test-set root (default: {DEFAULT_ROOT})")
    p.add_argument("--people", nargs="+", default=None, metavar="ID",
                   help="the expected roster of person ids. Default: whoever is "
                        "found in the filenames (then --expect-people decides if "
                        "that is enough).")
    p.add_argument("--expect-people", type=int, default=EXPECTED_PEOPLE,
                   help=f"how many teammates must cover every class "
                        f"(default: {EXPECTED_PEOPLE})")
    p.add_argument("--min-per-person", type=int, default=MIN_PER_PERSON_PER_CLASS,
                   help="images each person needs per class "
                        f"(default: {MIN_PER_PERSON_PER_CLASS})")
    p.add_argument("--min-conditions", type=int, default=MIN_CONDITIONS_PER_PERSON,
                   help="distinct lighting/background tags each person must have "
                        f"(default: {MIN_CONDITIONS_PER_PERSON})")
    p.add_argument("--no-size-check", dest="size_check", action="store_false",
                   help=f"skip verifying every image is {IMG_SIZE}x{IMG_SIZE}")
    p.add_argument("--quiet", action="store_true",
                   help="only print the verdict, not the matrix")
    return p


def main() -> int:
    args = build_parser().parse_args()
    root = args.root
    if not Path(root).is_dir():
        print(f"ERROR: {root} is not a directory", file=sys.stderr)
        return 2

    cov = scan(root)
    found = cov.people
    roster = list(args.people) if args.people else found
    failures: List[str] = []

    print(f"webcam test set: {root}")
    print(QUARANTINE_LINE)
    print(f"images: {cov.total}   people found: {', '.join(found) or '(none)'}\n")

    if not args.quiet and roster:
        print_matrix(cov, roster, args.min_per_person)
        print()

    # --- condition (a): 29 classes x all three teammates ------------------- #
    if len(roster) < args.expect_people:
        failures.append(
            f"only {len(roster)} person id(s) present ({', '.join(roster) or 'none'}); "
            f"the DoD needs {args.expect_people}"
        )
    for person in roster:
        missing = cov.classes_missing_for(person, args.min_per_person)
        if missing:
            failures.append(
                f"{person}: {len(missing)}/{len(CLASSES)} classes below "
                f"{args.min_per_person} image(s) — {', '.join(missing)}"
            )
    unexpected = [p for p in found if p not in roster]
    if unexpected:
        print(f"note: person id(s) not in the roster: {', '.join(unexpected)}")

    # --- varied conditions (the point of the set, not a nicety) ------------ #
    for person in roster:
        tags = sorted(cov.conditions.get(person, ()))
        label = ", ".join(tags) if tags else "(none tagged)"
        if len(tags) < args.min_conditions:
            failures.append(
                f"{person}: {len(tags)} condition tag(s) — {label}. Needs "
                f">= {args.min_conditions}: same-room/same-lighting captures "
                "defeat the purpose of this set (record with --condition)."
            )
        else:
            print(f"ok   {person}: {len(tags)} condition tags — {label}")
    print()

    # --- condition (b): the layout evaluate.py actually scores -------------- #
    shadowed = find_shadowed_dirs(root)
    if shadowed:
        failures.append(
            "images in directories the scorer never reads (evaluate._webcam_samples "
            "does not recurse): " + ", ".join(shadowed) +
            " — move them to <root>/<CLASS>/<person>_<seq>.jpg"
        )
    if cov.unparsed:
        failures.append(
            f"{len(cov.unparsed)} file(s) do not follow <person>_<seq>.jpg, so "
            "their provenance cannot be checked: "
            + ", ".join(str(p) for p in cov.unparsed[:5])
            + (" ..." if len(cov.unparsed) > 5 else "")
        )
    print(f"layout: {crosscheck_with_evaluate(root, cov.total)}")

    if args.size_check and cov.total:
        bad = check_image_sizes(root)
        if bad:
            failures.append(
                f"{len(bad)} image(s) are not {IMG_SIZE}x{IMG_SIZE} — they did not "
                "come through src.crop.HandCropper: " + ", ".join(bad[:5])
                + (" ..." if len(bad) > 5 else "")
            )
        else:
            print(f"crops:  OK: every image is {IMG_SIZE}x{IMG_SIZE} "
                  "(the frozen src/crop.py contract)")

    print()
    if failures:
        print("FAIL — the #15 Definition of Done is not met:")
        for f in failures:
            print(f"  - {f}")
        print("\nRecord more with: python scripts/record_webcam_testset.py "
              "--person <id> --all-classes --condition <tag>")
        return 1

    print(f"PASS — all {len(CLASSES)} classes covered by {len(roster)} people "
          f"({', '.join(roster)}), in the layout src.evaluate --webcam scores.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
