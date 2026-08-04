# `data/webcam_testset/` — the reported benchmark

Crops the three of us record ourselves. This is the **only accuracy number this
project reports** (target ≥ 90%, no class below 85%); the frame-range val split is
a dev number from the same signer in the same room.

Full capture protocol: **[`docs/WEBCAM_TESTSET_PROTOCOL.md`](../../docs/WEBCAM_TESTSET_PROTOCOL.md)** (#15).

## 🚫 Quarantine

**This data never enters training and is never used for model selection** — not as
a training root, not in a split manifest, not for choosing a checkpoint, epoch,
threshold, or augmentation setting. It is read by one command:

```bash
python -m src.evaluate --checkpoint checkpoints/<model>.pt --webcam data/webcam_testset
```

Tuning against it turns "generalises to new people" into "fitted three specific
people", irreversibly. If the number disappoints, fix it on the training side and
measure on the **dev val split**, then re-score here once.

## Layout

```
data/webcam_testset/<CLASS>/<person>_<condition>_<seq>.jpg
```

e.g. `data/webcam_testset/A/qdoan_desklamp-plainwall_001.jpg`

Folder-per-class with the person in the **filename**, because
`src.evaluate._webcam_samples` lists files directly under `<root>/<CLASS>/` and
does not recurse — a per-person folder would be silently scored as zero images.
Class names are exactly `src.data.CLASSES` (29: A–Z, `space`, `delete`,
`nothing` — note `delete`, not Kaggle's `del`).

## Commands

```bash
# record (all 29 classes, one lighting/background condition per session)
python scripts/record_webcam_testset.py --person qdoan --all-classes \
    --count 10 --condition desklamp-plainwall

# verify coverage + layout (exit 0 only when the #15 DoD is met)
python scripts/check_webcam_coverage.py --root data/webcam_testset
```

## What is committed here

Nothing but `.gitkeep`, `.gitignore` and this README. The captures are
git-ignored: they live on the team's shared drive and everyone syncs a local copy.

Do **not** add synthetic or placeholder images to a class folder to smoke-test the
pipeline — unlike `data/backgrounds/`, everything under a class folder here is
scored as the reported benchmark. Point the recorder at `--out /tmp/...` instead.
