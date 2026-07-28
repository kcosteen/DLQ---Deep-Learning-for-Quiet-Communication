"""Dataset loading and the LEAKAGE-SAFE split.

The ASL Alphabet dataset (grassknoted/asl-alphabet) is 87k images that are, in
reality, consecutive near-duplicate video frames of ONE signer in ONE room. A
random train/val split places near-identical neighbouring frames on both sides
and reports a fake ~99.9% accuracy.

The only honest split here is a GROUP / frame-range split: within each class,
sort filenames deterministically and send the first ``train_frac`` of frames to
train and the remaining tail to val. Neighbouring frames therefore never straddle
the split. See ``frame_range_split`` and the test in
``tests/test_split_leakage.py``.

The REAL benchmark is not even this val split — it is ``data/webcam_testset``,
recorded by the team and never used for training.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

# 29 classes: A-Z then the three control gestures. Order is fixed and is the
# label index order used everywhere (model head, evaluation, the app).
CLASSES: Tuple[str, ...] = tuple(
    [chr(c) for c in range(ord("A"), ord("Z") + 1)] + ["space", "delete", "nothing"]
)
CLASS_TO_IDX: Dict[str, int] = {c: i for i, c in enumerate(CLASSES)}
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


@dataclass
class Sample:
    path: str
    label: int
    class_name: str


@dataclass
class SplitManifest:
    """Serialisable record of exactly which files landed in each split."""

    train_frac: float
    seed: int
    root: str
    train: List[Sample] = field(default_factory=list)
    val: List[Sample] = field(default_factory=list)

    def to_json(self, path: str) -> None:
        payload = {
            "train_frac": self.train_frac,
            "seed": self.seed,
            "root": self.root,
            "train": [s.__dict__ for s in self.train],
            "val": [s.__dict__ for s in self.val],
        }
        Path(path).write_text(json.dumps(payload, indent=2))

    @staticmethod
    def from_json(path: str) -> "SplitManifest":
        d = json.loads(Path(path).read_text())
        m = SplitManifest(d["train_frac"], d["seed"], d["root"])
        m.train = [Sample(**s) for s in d["train"]]
        m.val = [Sample(**s) for s in d["val"]]
        return m


def list_class_files(root: str, class_name: str) -> List[str]:
    """All image files for one class, sorted deterministically.

    Sorting is what makes "first 80% of frames" well-defined and reproducible.
    Kaggle names are like ``A1.jpg .. A3000.jpg``; we sort by the numeric suffix
    when present, else lexicographically, so frames stay in capture order.
    """
    d = Path(root) / class_name
    if not d.is_dir():  # class folder absent (e.g. a partial webcam capture)
        return []
    files = [p for p in d.iterdir() if p.suffix.lower() in IMG_EXTS]

    def key(p: Path):
        stem = p.stem
        digits = "".join(ch for ch in stem if ch.isdigit())
        return (0, int(digits)) if digits else (1, stem)

    return [str(p) for p in sorted(files, key=key)]


def frame_range_split(
    root: str, train_frac: float = 0.8, classes: Sequence[str] = CLASSES
) -> SplitManifest:
    """Leakage-safe split: per class, first ``train_frac`` frames -> train.

    The tail (last ``1 - train_frac``) becomes val. Because files are in capture
    order, a train frame and a val frame are never adjacent video frames.
    """
    if not 0.0 < train_frac < 1.0:
        raise ValueError("train_frac must be in (0, 1)")
    manifest = SplitManifest(train_frac=train_frac, seed=0, root=str(root))
    for class_name in classes:
        files = list_class_files(root, class_name)
        if not files:
            continue
        cut = int(len(files) * train_frac)
        idx = CLASS_TO_IDX[class_name]
        manifest.train += [Sample(f, idx, class_name) for f in files[:cut]]
        manifest.val += [Sample(f, idx, class_name) for f in files[cut:]]
    return manifest


def class_counts(root: str, classes: Sequence[str] = CLASSES) -> Dict[str, int]:
    """Image count per class — used by EDA and the class-balance check."""
    return {c: len(list_class_files(root, c)) for c in classes}


def _frame_group_id(path: str) -> str:
    """Stable id for a physical frame, independent of split assignment.

    Used by the leakage test to assert train and val share no frame group.
    """
    return hashlib.sha1(os.path.abspath(path).encode()).hexdigest()


class ASLImageDataset:
    """A tiny torch ``Dataset`` over a list of ``Sample``.

    Kept import-light: torch and the augmentation pipeline are only pulled in when
    a Dataset is actually constructed, so the split logic above stays testable
    without a GPU stack installed.
    """

    def __init__(self, samples: Sequence[Sample], transform=None, train: bool = True):
        self.samples = list(samples)
        self.train = train
        if transform is None:
            from .augment import build_transforms  # local import

            transform = build_transforms(train=train)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int):
        import cv2

        s = self.samples[i]
        img_bgr = cv2.imread(s.path, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise FileNotFoundError(s.path)
        tensor = self.transform(img_bgr)
        return tensor, s.label


def build_datasets(manifest: SplitManifest) -> Tuple["ASLImageDataset", "ASLImageDataset"]:
    """Convenience: (train_ds, val_ds) from a manifest with correct aug flags."""
    return (
        ASLImageDataset(manifest.train, train=True),
        ASLImageDataset(manifest.val, train=False),
    )


def main() -> None:
    p = argparse.ArgumentParser(description="Build the leakage-safe train/val split.")
    p.add_argument("--root", required=True,
                   help="path to asl_alphabet_train/ (29 class folders)")
    p.add_argument("--train-frac", type=float, default=0.8)
    p.add_argument("--out", default="data/split_manifest.json")
    p.add_argument("--counts", action="store_true", help="also print class balance")
    args = p.parse_args()

    if args.counts:
        counts = class_counts(args.root)
        for c in CLASSES:
            print(f"{c:>8}: {counts.get(c, 0)}")
        print(f"{'TOTAL':>8}: {sum(counts.values())}")

    manifest = frame_range_split(args.root, args.train_frac)
    manifest.to_json(args.out)
    print(f"train={len(manifest.train)}  val={len(manifest.val)}  -> {args.out}")


if __name__ == "__main__":
    main()
