"""Generate a tiny, committable set of synthetic background textures (issue #10).

These are NOT meant to be good backgrounds — they exist so the demo command

    python -m src.augment --image <img> --backgrounds data/backgrounds

works out of the box and the ablation is reproducible without downloading
anything. For a real training run, drop varied *photographic* indoor/outdoor
scenes into this folder (see README.md); those are git-ignored by default.

Run from the repo root:

    python data/backgrounds/generate_samples.py
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

OUT_DIR = Path(__file__).resolve().parent
SIZE = 96  # tiny on purpose: each PNG is a few KB


def _linear_gradient(c0, c1) -> np.ndarray:
    t = np.linspace(0.0, 1.0, SIZE, dtype=np.float32)[None, :, None]
    return (np.asarray(c0, np.float32) * (1 - t) + np.asarray(c1, np.float32) * t).astype(np.uint8) \
        * np.ones((SIZE, 1, 3), np.uint8)


def _checker(c0, c1, cells=6) -> np.ndarray:
    step = SIZE // cells
    idx = (np.arange(SIZE) // step)[:, None] + (np.arange(SIZE) // step)[None, :]
    mask = (idx % 2).astype(bool)[..., None]
    img = np.where(mask, np.asarray(c0, np.uint8), np.asarray(c1, np.uint8))
    return img.astype(np.uint8)


def _noise(base, spread=40, seed=0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = rng.normal(0, spread, (SIZE, SIZE, 3))
    return np.clip(np.asarray(base, np.float32) + n, 0, 255).astype(np.uint8)


def main() -> None:
    # BGR colours (cv2). A small spread of indoor/outdoor-ish tones + textures.
    samples = {
        "sample_sky_gradient": _linear_gradient((200, 150, 90), (240, 220, 180)),
        "sample_warm_wall": _noise((110, 140, 175), spread=18, seed=1),   # beige wall
        "sample_wood_floor": _noise((60, 90, 130), spread=28, seed=2),    # brown wood
        "sample_foliage": _noise((60, 120, 60), spread=45, seed=3),       # green outdoors
        "sample_tile_checker": _checker((210, 210, 210), (150, 150, 150)),
        "sample_curtain": _linear_gradient((70, 40, 120), (160, 120, 200)),  # purple
    }
    for name, img in samples.items():
        path = OUT_DIR / f"{name}.png"
        cv2.imwrite(str(path), img)
        print(f"wrote {path.relative_to(OUT_DIR.parent.parent)}  ({path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
