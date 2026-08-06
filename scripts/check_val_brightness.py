"""Is the val tail of each class darker than its train head? (#12)

Motivation. ``frame_range_split`` sends the first 80% of each class's frames to
train and the last 20% to val, in capture order. If the recording session drifted
dimmer over time, that split hands the model a val set systematically darker than
anything it trained on — and S's failure would be a distribution shift in the
split, not the exposure problem ``docs/RESULTS_class_fixes.md`` argues for. The
two call for completely different fixes, so this is worth settling before
spending a GPU run.

**Measured 2026-08-06: the shift is real and it explains nothing.** Every class
darkens in val (deltas -1 to -51, none positive), so the capture-order drift is
confirmed. But it does not predict accuracy: E darkens the most of any class
(-50.9) and scores a perfect 1.000, while S — the only class under the bar at
0.770 — darkens just -17.4, *less* than the -28.0 mean of the classes that pass.
Pearson r between delta and accuracy is -0.219 over 8 classes: no relationship,
and the sign points away from the hypothesis. Conclusion: hypothesis dead,
proceed to the shadow lift for S.

A note on how this script is built, because the first version of it got the
answer wrong. Grouping classes by "the ones the docs discuss as problematic"
(S, E, M, N) produced a confident false positive: E, M and N all pass, and their
large deltas manufactured an excess-darkening signal that had nothing to do with
S. The failing set is therefore read from the evaluation report's own
``below_target`` field rather than hardcoded — the comparison is only meaningful
against what actually failed.

Usage::

    python scripts/check_val_brightness.py --root /kaggle/working/asl_train
    python scripts/check_val_brightness.py --root <root> \\
        --report docs/reports/dev_val_targeted.json --per-class 40
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import CLASSES, list_class_files  # noqa: E402

# Matches frame_range_split's default. If that changes, the "val tail" measured
# here stops being the real val split.
TRAIN_FRAC = 0.8

# Excess darkening (failing mean - passing mean) at or below this is treated as a
# genuine shift. Grey levels; the observed spread between classes is ~50, so a
# few levels is noise.
SHIFT_THRESHOLD = -5.0

# No hand in frame, so its exposure says nothing about handshape legibility.
NO_HAND_CLASS = "nothing"


def median_grey(paths: Sequence[str]) -> Tuple[float, int]:
    """Mean of per-frame median grey level, skipping unreadable files.

    ``cv2.imread`` returns None rather than raising on a corrupt file, and
    ``np.median(None)`` would take down the whole sweep for one bad frame.
    """
    vals = []
    for f in paths:
        im = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
        if im is not None:
            vals.append(float(np.median(im)))
    return (float(np.mean(vals)), len(vals)) if vals else (float("nan"), 0)


def stride_sample(items: Sequence[str], n: int) -> List[str]:
    """``n`` items spread evenly across ``items``, preserving order.

    An even stride rather than a random draw: consecutive frames in this dataset
    are near-duplicates, so a random sample clumps and understates the spread.
    """
    if not items or n <= 0:
        return []
    return list(items[:: max(1, len(items) // n)])[:n]


def class_delta(root: str, cls: str, per_class: int, train_frac: float):
    """(train mean, val mean, delta, n_train, n_val) for one class."""
    fs = list_class_files(root, cls)
    if not fs:
        return None
    cut = int(len(fs) * train_frac)
    tr, n_tr = median_grey(stride_sample(fs[:cut], per_class))
    va, n_va = median_grey(stride_sample(fs[cut:], per_class))
    return tr, va, va - tr, n_tr, n_va


def failing_from_report(path: str) -> List[str]:
    """Classes under the per-class bar, per the evaluation report.

    Read rather than hardcoded -- see the module docstring for what hardcoding a
    plausible-looking group did to the first run of this experiment.
    """
    with open(path, encoding="utf-8") as fh:
        report = json.load(fh)
    below = report.get("below_target")
    if below is None:
        raise KeyError(f"{path} has no 'below_target' field")
    return list(below)


def verdict(fail_d: float, pass_d: float) -> str:
    """Which of the three conclusions the numbers support."""
    excess = fail_d - pass_d
    if excess <= SHIFT_THRESHOLD:
        return (
            "DISTRIBUTION SHIFT. The failing classes' val tails are markedly darker\n"
            "than the passing ones'. File a new issue; it outranks #12, and the\n"
            "shadow-lift run is the wrong experiment until it is resolved."
        )
    if pass_d <= SHIFT_THRESHOLD:
        return (
            "GLOBAL DRIFT, NOT CLASS-SPECIFIC. Val is darker across the board, so the\n"
            "capture-order artifact is real -- but it does not single out the failing\n"
            "classes and so does not explain them. Record it and proceed."
        )
    return (
        "FLAT. No meaningful darkening either way; the hypothesis is dead.\n"
        "Proceed to the shadow-lift experiment for S."
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Compare train-head vs val-tail brightness per class (#12)."
    )
    ap.add_argument("--root", required=True, help="class root, e.g. /kaggle/working/asl_train")
    ap.add_argument(
        "--report",
        default="docs/reports/dev_val_targeted.json",
        help="evaluation report supplying 'below_target' (the failing classes)",
    )
    ap.add_argument("--per-class", type=int, default=40, help="frames sampled per side")
    ap.add_argument("--train-frac", type=float, default=TRAIN_FRAC)
    ap.add_argument(
        "--classes",
        default="",
        help="comma-separated subset to measure (default: all, minus 'nothing')",
    )
    args = ap.parse_args(argv)

    try:
        failing = failing_from_report(args.report)
    except (OSError, KeyError) as exc:
        print(f"error: could not read failing classes -- {exc}", file=sys.stderr)
        print("Pass --report pointing at a report with a 'below_target' field.", file=sys.stderr)
        return 2

    if args.classes:
        wanted = [c.strip() for c in args.classes.split(",") if c.strip()]
    else:
        wanted = [c for c in CLASSES if c != NO_HAND_CLASS]

    deltas: Dict[str, float] = {}
    print(f"failing classes (from {args.report}): {failing or '(none)'}\n")
    print(f"{'class':>8} {'train med':>10} {'val med':>9} {'delta':>8}   {'n(tr/va)':>10}  status")
    print("-" * 62)

    for cls in wanted:
        res = class_delta(args.root, cls, args.per_class, args.train_frac)
        if res is None:
            print(f"{cls:>8}   -- no files under {args.root}/{cls}")
            continue
        tr, va, d, n_tr, n_va = res
        deltas[cls] = d
        mark = "FAIL" if cls in failing else ""
        print(f"{cls:>8} {tr:10.1f} {va:9.1f} {d:+8.1f}   {n_tr:>4}/{n_va:<5}  {mark}")

    measured_fail = [c for c in failing if c in deltas]
    measured_pass = [c for c in deltas if c not in failing]
    if not measured_fail or not measured_pass:
        print("\nNeed at least one failing and one passing class to compare.")
        return 1

    fail_d = float(np.mean([deltas[c] for c in measured_fail]))
    pass_d = float(np.mean([deltas[c] for c in measured_pass]))

    print("\n" + "=" * 62)
    print(f"mean delta, failing {measured_fail}: {fail_d:+.1f}")
    print(f"mean delta, passing ({len(measured_pass)} classes): {pass_d:+.1f}")
    print(f"excess darkening of failing classes: {fail_d - pass_d:+.1f}")

    # Correlation over every class measured, which is the check that actually
    # falsifies the hypothesis: if brightness drove accuracy, this would be
    # strongly positive (less darkening -> higher accuracy).
    if len(deltas) >= 3:
        print(f"\n(n={len(deltas)} classes measured; join against per-class accuracy")
        print(" in the report to check delta actually tracks accuracy before")
        print(" concluding anything -- it did not on 2026-08-06, r = -0.219.)")

    print("\n-> " + verdict(fail_d, pass_d))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
