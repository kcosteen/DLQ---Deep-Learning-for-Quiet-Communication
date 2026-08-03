"""Pre-apply the shared MediaPipe hand-crop to the whole training set (#11).

The 87k Kaggle images are read, cropped once with the SAME ``HandCropper`` the
webcam app uses, and the 224² BGR crops are written to a parallel folder-per-class
tree. Training then points ``--root`` at that cache and reads pre-cropped images,
so the expensive MediaPipe crop is paid once offline instead of being re-derivable
every epoch.

Parity is preserved on purpose. The crop is cached **un-normalized** (a plain
224² BGR uint8 image on disk); normalization still happens in the load-time
transform. ``crop.resize_square`` is idempotent on a 224² square (proven in
``tests/test_preprocessing_parity.py``), so feeding a cached crop back through the
train/val transform is a no-op resize followed by the usual augment/normalize —
byte-for-byte the same tensor the app would produce from a live crop.

Wiring into training (no loader change):

    python -m src.cache_crops \
        --src-root data/raw/asl_alphabet_train/asl_alphabet_train \
        --cache-root data/cache_crops
    python -m src.train --root data/cache_crops --manifest ""   # train off the cache

``frame_range_split`` runs over the cache's identical filenames/layout, so the
leakage-safe split is unchanged. (Pass a fresh ``--manifest`` path or none, since a
manifest baked against the raw root records that root.)

The cache is as large as the raw set — it is git-ignored and must never be
committed (see ``.gitignore``).
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Sequence

import cv2

from .data import CLASSES, list_class_files


@dataclass
class CacheStats:
    """What happened during a caching run — logged at the end."""

    total: int = 0          # images seen across all class folders
    written: int = 0        # crops newly written this run
    skipped: int = 0        # already present in the cache (resumable no-op)
    unreadable: int = 0     # cv2.imread returned None; left uncached
    no_hand: int = 0        # crops that fell back to a centre square (no hand)
    no_hand_by_class: Dict[str, int] = field(default_factory=dict)


def cache_dataset(
    src_root: str,
    cache_root: str,
    cropper,
    classes: Sequence[str] = CLASSES,
    log_every: int = 1000,
    log: Callable[[str], None] = print,
) -> CacheStats:
    """Crop every image under ``src_root`` and mirror it into ``cache_root``.

    ``cropper`` is anything exposing ``crop(img_bgr) -> CropResult`` with
    ``.crop_bgr`` (a 224² BGR uint8 array) and ``.found_hand`` — i.e. a
    ``crop.HandCropper``. It is passed in (not constructed here) so the ONE
    instance is created once by the caller and reused for all 87k images, and so
    the loop is testable without the mediapipe wheel.

    Resumable/idempotent: an image whose destination file already exists is
    skipped, so an interrupted run can simply be re-invoked.
    """
    src = Path(src_root)
    cache = Path(cache_root)

    # One deterministic pass to size the job, so progress can show count / total.
    files_by_class = {c: list_class_files(str(src), c) for c in classes}
    total = sum(len(v) for v in files_by_class.values())
    stats = CacheStats(total=total)
    if total == 0:
        log(f"no images found under {src} — nothing to cache")
        return stats

    log(f"caching {total} images from {src} -> {cache}")
    seen = 0
    start = time.perf_counter()
    for class_name in classes:
        files = files_by_class[class_name]
        if not files:
            continue
        out_dir = cache / class_name
        out_dir.mkdir(parents=True, exist_ok=True)
        for path in files:
            seen += 1
            dest = out_dir / Path(path).name
            if dest.exists():  # resumable: already cached
                stats.skipped += 1
            else:
                img = cv2.imread(path, cv2.IMREAD_COLOR)
                if img is None:
                    stats.unreadable += 1
                    log(f"  WARN unreadable, skipped: {path}")
                else:
                    result = cropper.crop(img)
                    if not result.found_hand:
                        stats.no_hand += 1
                        stats.no_hand_by_class[class_name] = (
                            stats.no_hand_by_class.get(class_name, 0) + 1
                        )
                    cv2.imwrite(str(dest), result.crop_bgr)
                    stats.written += 1

            if log_every and seen % log_every == 0:
                rate = seen / max(1e-9, time.perf_counter() - start)
                log(f"  {seen}/{total}  ({rate:.0f} img/s)")

    _log_summary(stats, log)
    return stats


def _log_summary(stats: CacheStats, log: Callable[[str], None]) -> None:
    log(
        f"done: {stats.written} written, {stats.skipped} skipped (already cached), "
        f"{stats.unreadable} unreadable, {stats.total} total"
    )
    log(
        f"no-hand fallbacks (centre-square crop): {stats.no_hand}"
        + (f" of {stats.written} newly cropped" if stats.written else "")
    )
    if stats.no_hand_by_class:
        for cls in sorted(stats.no_hand_by_class):
            log(f"    {cls:>8}: {stats.no_hand_by_class[cls]}")


def main() -> None:
    p = argparse.ArgumentParser(
        description=(
            "Cache MediaPipe hand-crops of the training set (#11). Reads every "
            "image under --src-root, applies the shared HandCropper, and writes "
            "224² BGR crops to a folder-per-class tree at --cache-root. Then train "
            "with: python -m src.train --root <cache-root> --manifest ''"
        )
    )
    p.add_argument("--src-root", required=True,
                   help="raw asl_alphabet_train/ (29 class folders)")
    p.add_argument("--cache-root", required=True,
                   help="output dir for cropped crops (git-ignored; never commit)")
    p.add_argument("--min-detection-confidence", type=float, default=0.5,
                   help="MediaPipe hand detection threshold (crop.HandCropper)")
    p.add_argument("--log-every", type=int, default=1000,
                   help="images between progress lines (0 = only the final summary)")
    args = p.parse_args()

    from .crop import HandCropper

    # static_image_mode=True: these are independent stills, not a video stream, so
    # MediaPipe must detect afresh per image rather than track across frames. ONE
    # cropper, reused for all images (constructing it per image is very slow).
    with HandCropper(
        static_image_mode=True,
        min_detection_confidence=args.min_detection_confidence,
    ) as cropper:
        cache_dataset(args.src_root, args.cache_root, cropper,
                      log_every=args.log_every)


if __name__ == "__main__":
    main()
