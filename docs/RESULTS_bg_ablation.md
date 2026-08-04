# Ablation — Background replacement on segmented hands (#10)

**Question.** Does compositing the MediaPipe-segmented hand over random backgrounds
(`src.augment.BackgroundReplacer`) improve **webcam robustness** — accuracy on
`data/webcam_testset`, the *real* reported benchmark — over the same recipe without
it? The hypothesis: the 87k training images are one signer in one room, so a model
that never sees a different background over-fits the room; background replacement is
the structural augmentation that most directly attacks that.

> ⚠️ **Status: placeholder — no numbers filled in yet.** This environment has no GPU
> and no training dataset, so no real training run was performed here. Every metric
> cell below is a clearly-marked `TODO`; **do not quote any number from this table**
> until a run fills it in. The commands are exact and reproducible on Colab / a GPU
> box. (Per issue #10's Definition of Done, a documented ablation with the exact
> command + honest placeholders is acceptable when real numbers can't be produced.)

## Design

Two otherwise-identical two-stage EfficientNet-B0 runs on the **same leakage-safe
frame-range split** (`data/split_manifest.json`), differing in exactly one flag:

| Arm | Command flag | What changes |
|---|---|---|
| **A — control** | *(no `--backgrounds`)* | augmentation as in #9, no bg replacement |
| **B — treatment** | `--backgrounds data/backgrounds` | + train-only bg replacement (~50% of samples) |

Background replacement is **train-only** and runs *before* the frozen
`preprocess_bgr` contract, so the val/serve path is byte-identical between arms —
any accuracy delta is attributable to the augmentation, not to a preprocessing
change. The dev-val (temporal, same-signer) number is expected to move little or
even *down*; the metric that matters is the **webcam** number and the
**train→webcam gap**.

For a real ablation, point `--backgrounds` at a **varied photographic** set (see
`data/backgrounds/README.md`), not just the tiny committed `sample_*.png` synthetic
textures — those exist only so the command runs out of the box.

## Reproduce

```bash
# 0. (once) MediaPipe wheel + models are required for bg replacement to actually
#    segment; the tflite model is already committed (models/selfie_segmenter.tflite).
pip install -r requirements.txt

# 1. Build the leakage-safe split once (shared by both arms).
python -m src.data --root data/raw/asl_alphabet_train/asl_alphabet_train --counts

# 2. Arm A — control (no background replacement).
python -m src.train \
    --root data/raw/asl_alphabet_train/asl_alphabet_train \
    --manifest data/split_manifest.json \
    --wandb-mode online --out checkpoints/effb0_noBG.pt

# 3. Arm B — treatment (background replacement ON).
python -m src.train \
    --root data/raw/asl_alphabet_train/asl_alphabet_train \
    --manifest data/split_manifest.json \
    --backgrounds data/backgrounds \
    --wandb-mode online --out checkpoints/effb0_BG.pt

# 4. Evaluate BOTH on the reported benchmark — the webcam test set.
python -m src.evaluate --checkpoint checkpoints/effb0_noBG.pt --webcam data/webcam_testset
python -m src.evaluate --checkpoint checkpoints/effb0_BG.pt   --webcam data/webcam_testset
#    (also --split data/raw/... for the dev-val number, labelled dev-only)
```

Keep everything else fixed between the two runs (epochs, batch size, LR schedule,
seed). Use the same `--manifest` for both so the split is identical.

## Results

**Reported benchmark — webcam test set (`data/webcam_testset`):**

| Arm | Webcam acc | Macro-F1 | Min per-class | Dev-val acc (dev-only) |
|---|---|---|---|---|
| A — no background replacement | `TODO` | `TODO` | `TODO` | `TODO` |
| B — + background replacement  | `TODO` | `TODO` | `TODO` | `TODO` |
| **Δ (B − A)** | `TODO` | `TODO` | `TODO` | `TODO` |

Targets for context (from the README honest-metrics policy): webcam ≥ 90%, macro-F1
high, **no class below 85%**. The bg-replacement win, if any, should show up most in
the **webcam** column and in lifting the **worst-confused / worst-background**
classes, not in the dev-val column.

## How to read the delta (when filled in)

- **Δ webcam acc > 0** → background replacement helped the model ignore the room;
  keep it on. Report the exact number as *the* result of this ablation.
- **Δ webcam acc ≈ 0 but Δ min-per-class > 0** → it didn't move the average but
  rescued the worst classes (often the ones whose training frames share a
  distinctive background); still worth keeping.
- **Δ webcam acc < 0** → over-augmentation or segmentation artefacts hurt more than
  the room-invariance helped; lower the apply probability (currently ~0.5 in
  `build_transforms`) or improve the background set before dropping the feature.

## Notes / honesty

- The tiny committed `data/backgrounds/sample_*.png` are **synthetic** and only make
  the demo/command runnable; a real run must use varied photographic scenes.
- No metric in this doc is real yet. When a run completes, replace every `TODO`,
  delete this line, and cite the **webcam** delta as the headline.
- Parity is unaffected: `tests/test_preprocessing_parity.py` stays green with and
  without `--backgrounds` (bg replacement is train-only, before `preprocess_bgr`).
