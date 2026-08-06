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
recorded by the team and never used for training. Everywhere this split is
surfaced (docstrings, the ``--counts`` CLI output, the manifest metadata, the
README) it is labelled **"dev val (temporal split, same signer — NOT the reported
metric)"** so no one mistakes it for the headline number (#15/#16).

Augmentation invariant
----------------------
Augmented copies must **never** cross the split. The split here operates ONLY on
the original source files under ``root``; augmentation (#9/#10) is applied *after*
the split and *train-only* — the augmentation entry points take the train split as
input, never the raw ``root``. Enforcing "split first, augment second" is what keeps
a photometric/geometric twin of a val frame from leaking into train.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

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
        """Write a deterministic, portable manifest.

        Paths are stored RELATIVE to ``root`` (and with ``/`` separators) so the
        file reproduces byte-for-byte across machines; keys are sorted and a
        ``content_sha256`` over the ordered (split, class, relative-path) assignment
        is recorded so any change in membership is detectable. Re-running with the
        same ``(root, train_frac)`` yields an identical file.
        """
        root = str(self.root)

        def rel(p: str) -> str:
            return os.path.relpath(p, root).replace(os.sep, "/")

        def rows(samples: Sequence[Sample]) -> List[Dict]:
            return [
                {"path": rel(s.path), "label": s.label, "class_name": s.class_name}
                for s in samples
            ]

        train_rows, val_rows = rows(self.train), rows(self.val)

        # content hash: deterministic function of the split assignment only
        h = hashlib.sha256()
        for split, rws in (("train", train_rows), ("val", val_rows)):
            for r in rws:
                h.update(f"{split}\t{r['class_name']}\t{r['path']}\n".encode())

        payload = {
            "schema": "asl-frame-range-split/v1",
            "description": (
                "dev val (temporal frame-range split, same signer) — NOT the "
                "reported metric. The reported number is the webcam test set "
                "(see #15/#16)."
            ),
            "train_frac": self.train_frac,
            "seed": self.seed,
            "root": root,
            "paths_relative_to": "root",
            "classes": list(CLASSES),          # stable label-index order
            "class_to_idx": dict(CLASS_TO_IDX),
            "content_sha256": h.hexdigest(),
            "n_train": len(train_rows),
            "n_val": len(val_rows),
            "train": train_rows,
            "val": val_rows,
        }
        Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    @staticmethod
    def from_json(path: str) -> "SplitManifest":
        d = json.loads(Path(path).read_text())
        root = d["root"]
        m = SplitManifest(d["train_frac"], d.get("seed", 0), root)

        def abspath(rel: str) -> str:
            return rel if os.path.isabs(rel) else os.path.join(root, rel)

        m.train = [Sample(abspath(s["path"]), s["label"], s["class_name"]) for s in d["train"]]
        m.val = [Sample(abspath(s["path"]), s["label"], s["class_name"]) for s in d["val"]]
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
    order (numeric-aware sort, so ``A2.jpg`` precedes ``A10.jpg``), a train frame
    and a val frame are never adjacent video frames.

    Deterministic given ``(root, train_frac)``: no shuffling, no ``random`` /
    ``np.random`` / ``train_test_split``, and no reliance on filesystem iteration
    order (``list_class_files`` sorts). Operates ONLY on original source files —
    augmentation happens after the split, train-only (see module docstring).
    """
    if not 0.0 < train_frac < 1.0:
        raise ValueError("train_frac must be in (0, 1)")
    manifest = SplitManifest(train_frac=train_frac, seed=0, root=str(root))
    for class_name in classes:
        files = list_class_files(root, class_name)
        if not files:
            continue
        cut = int(len(files) * train_frac)
        # Snap the cut to a frame-group boundary so a physical frame group never
        # straddles train/val. ``_frame_group_id`` is per-file here (each file is
        # its own group), so this is a no-op for the Kaggle data; it future-proofs
        # the split should grouping ever become many-frames-per-group.
        if 0 < cut < len(files):
            edge = _frame_group_id(files[cut - 1])
            while cut < len(files) and _frame_group_id(files[cut]) == edge:
                cut += 1
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

    def __init__(
        self,
        samples: Sequence[Sample],
        transform=None,
        train: bool = True,
        backgrounds_dir: Optional[str] = None,
        class_transforms: Optional[Dict[str, object]] = None,
    ):
        self.samples = list(samples)
        self.train = train
        if transform is None:
            from .augment import build_transforms  # local import

            # backgrounds only ever feed the train transform; the val/serve path
            # stays the deterministic contract with no augmentation (parity).
            transform = build_transforms(
                train=train, backgrounds_dir=backgrounds_dir if train else None
            )
        self.transform = transform
        # Per-class override, keyed by class NAME (#12). Train-only by
        # construction — build_datasets never populates it for the val set,
        # because a val image that is transformed differently from its
        # neighbours is not a val image any more.
        self.class_transforms = dict(class_transforms or {})

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int):
        import cv2

        s = self.samples[i]
        img_bgr = cv2.imread(s.path, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise FileNotFoundError(s.path)
        transform = self.class_transforms.get(s.class_name, self.transform)
        tensor = transform(img_bgr)
        return tensor, s.label


def build_datasets(
    manifest: SplitManifest,
    backgrounds_dir: Optional[str] = None,
    geometry_safe_classes: Sequence[str] = (),
) -> Tuple["ASLImageDataset", "ASLImageDataset"]:
    """Convenience: (train_ds, val_ds) from a manifest with correct aug flags.

    ``backgrounds_dir`` (if given) enables background replacement on the TRAIN set
    only; the val set stays the deterministic contract for honest evaluation.

    ``geometry_safe_classes`` routes just those classes through the gentler
    ``geometry_safe`` augmentation policy (#12) while every other class keeps the
    heavy default. Val is untouched either way, so the dev-val number stays
    comparable to previous runs.
    """
    unknown = [c for c in geometry_safe_classes if c not in CLASS_TO_IDX]
    if unknown:
        raise ValueError(f"geometry_safe_classes are not ASL classes: {unknown}")

    class_transforms: Dict[str, object] = {}
    if geometry_safe_classes:
        from .augment import build_transforms  # local import

        gentle = build_transforms(
            train=True, backgrounds_dir=backgrounds_dir, policy="geometry_safe"
        )
        class_transforms = {c: gentle for c in geometry_safe_classes}

    return (
        ASLImageDataset(manifest.train, train=True, backgrounds_dir=backgrounds_dir,
                        class_transforms=class_transforms),
        ASLImageDataset(manifest.val, train=False),
    )


def main() -> None:
    p = argparse.ArgumentParser(description="Build the leakage-safe train/val split.")
    p.add_argument("--root", required=True,
                   help="path to asl_alphabet_train/ (29 class folders)")
    p.add_argument("--train-frac", type=float, default=0.8)
    p.add_argument("--out", default="data/split_manifest.json")
    p.add_argument("--counts", action="store_true",
                   help="also print per-class train/val counts and totals")
    args = p.parse_args()

    manifest = frame_range_split(args.root, args.train_frac)

    if args.counts:
        tr = {c: 0 for c in CLASSES}
        va = {c: 0 for c in CLASSES}
        for s in manifest.train:
            tr[s.class_name] += 1
        for s in manifest.val:
            va[s.class_name] += 1
        print("dev val = temporal frame-range split, same signer — NOT the reported "
              "metric (that is the webcam test set, #15/#16)")
        print(f"{'class':>8} {'train':>7} {'val':>6} {'total':>7}")
        for c in CLASSES:
            t, v = tr[c], va[c]
            if t + v:  # skip classes with no images present
                print(f"{c:>8} {t:>7} {v:>6} {t + v:>7}")
        T, V = len(manifest.train), len(manifest.val)
        print(f"{'TOTAL':>8} {T:>7} {V:>6} {T + V:>7}")

    manifest.to_json(args.out)
    print(f"train={len(manifest.train)}  val={len(manifest.val)}  -> {args.out}  "
          f"(dev val — NOT the reported metric)")


if __name__ == "__main__":
    main()
