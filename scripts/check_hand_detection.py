"""How often does MediaPipe actually find a hand in a set of frames?

Motivation (#12). Investigating why class S stays below the per-class bar turned up
something with wider consequences: over the 208 misclassified frames dumped by
``evaluate --dump-errors``, ``src.crop.HandCropper`` detected a hand in only 34% —
and in **0 of 20** S frames. The dataset is backlit, so a closed fist is a near-black
silhouette with no finger structure for the landmarker to latch onto.

That matters beyond one class, because two things assume the crop works:

* **#11's crop cache** — ``cache_crops`` falls back to a centre square when detection
  fails (``CropResult.found_hand`` is False), so a low detection rate means the
  "cropped" cache is quietly part centre-crops.
* **the live demo (#13/#14)** and ``PROJECT_PLAN.md``'s claim that the MediaPipe crop
  step is *mandatory* — "what makes 87k single-signer images work on a stranger's
  hand". If it does not fire, that load-bearing step is not bearing the load.

The 34% number is NOT a dataset-wide rate: those frames were selected for being
misclassified, so they are plausibly the darkest in the set. This script exists to
measure the honest number on a *random, per-class* sample instead of guessing from a
biased one. Report the per-class table — an average hides that open handshapes detect
at ~100% while closed fists detect at ~0%.

Usage::

    python scripts/check_hand_detection.py --root data/raw/asl_alphabet_train/asl_alphabet_train
    python scripts/check_hand_detection.py --root <root> --per-class 40 --gamma 0.45
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import CLASSES, list_class_files  # noqa: E402


def apply_gamma(img: np.ndarray, g: float) -> np.ndarray:
    """Shadow lift. ``g < 1`` brightens; ``g == 1`` is a no-op."""
    if g == 1.0:
        return img
    lut = ((np.arange(256) / 255.0) ** g * 255).astype(np.uint8)
    return cv2.LUT(img, lut)


def sample_paths(root: str, per_class: int, seed: int) -> List[tuple]:
    """``per_class`` frames per class, sampled evenly across the frame range.

    A deterministic stride rather than a random draw, so consecutive
    near-duplicate frames do not crowd the sample and the result is reproducible.
    """
    rng = random.Random(seed)
    out = []
    for cls in CLASSES:
        files = list_class_files(root, cls)
        if not files:
            continue
        if len(files) > per_class:
            stride = len(files) / per_class
            files = [files[int(i * stride)] for i in range(per_class)]
        rng.shuffle(files)
        out += [(cls, f) for f in files]
    return out


def run(root: str, per_class: int, seed: int, gamma: Optional[float]) -> int:
    from src.crop import HandCropper

    cropper = HandCropper()
    paths = sample_paths(root, per_class, seed)
    if not paths:
        raise SystemExit(f"no images found under {root}")

    per_class_hits = {c: [0, 0] for c in CLASSES}
    for cls, path in paths:
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            continue
        if gamma is not None:
            img = apply_gamma(img, gamma)
        per_class_hits[cls][1] += 1
        per_class_hits[cls][0] += bool(cropper.crop(img).found_hand)

    label = "raw" if gamma is None else f"gamma {gamma}"
    print(f"MediaPipe hand-detection rate ({label}), {per_class} frames/class, "
          f"seed {seed}")
    print(f"root: {root}\n")
    print(f'{"class":>8} {"found":>7} {"n":>5}  rate')

    rows = [(c, h, n) for c, (h, n) in per_class_hits.items() if n]
    for cls, hits, n in sorted(rows, key=lambda r: r[1] / max(1, r[2])):
        bar = "#" * int(30 * hits / max(1, n))
        print(f"{cls:>8} {hits:>7} {n:>5} {100 * hits / n:5.0f}%  {bar}")

    total_h = sum(r[1] for r in rows)
    total_n = sum(r[2] for r in rows)
    print(f"\n{'TOTAL':>8} {total_h:>7} {total_n:>5} {100 * total_h / total_n:5.0f}%")

    # A per-class floor is what matters, not the mean: the demo has to recognise
    # every letter, so one class at 0% is a broken letter regardless of the average.
    worst = [f"{c} ({100 * h / n:.0f}%)" for c, h, n in rows if h / max(1, n) < 0.5]
    if worst:
        print(f"\nclasses below 50% detection: {', '.join(worst)}")
        print("Each of these is a letter the crop step cannot frame — #11's cache "
              "falls back to a centre crop, and the live demo (#13/#14) has no "
              "hand box to draw. Detection is not optional for those classes, it "
              "is the whole bridge.")
    else:
        print("\nevery class detects above 50%.")
    return 0


def main() -> None:
    p = argparse.ArgumentParser(
        description="Measure the MediaPipe hand-detection rate per class."
    )
    p.add_argument("--root", required=True,
                   help="class root (asl_alphabet_train/ or data/webcam_testset)")
    p.add_argument("--per-class", type=int, default=25,
                   help="frames sampled per class (deterministic stride)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--gamma", type=float, default=None,
                   help="apply this gamma BEFORE detection (e.g. 0.45) to test "
                        "whether the failures are an exposure problem")
    args = p.parse_args()
    raise SystemExit(run(args.root, args.per_class, args.seed, args.gamma))


if __name__ == "__main__":
    main()
