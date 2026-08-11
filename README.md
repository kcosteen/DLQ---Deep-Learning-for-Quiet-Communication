# ASL Fingerspelling Recognition

A computer-vision model that reads the **American Sign Language hand alphabet**
and, live from a webcam, spells words as **text and speech**. Python + PyTorch,
transfer learning on EfficientNet-B0, trainable on a Colab free GPU. 3-person
team, ~8 weeks.

> **Read this first — what this project is (and is not).**
> This is an **alphabet / fingerspelling recognizer**, *not* "sign language
> translation." ASL is a full language with its own grammar and facial grammar;
> recognizing 26 static handshapes + a few control gestures is a narrow slice of
> it. Please read the [Ethics & honest-limits](#ethics--honest-limits) section
> before demoing this to anyone.

**29 classes:** A–Z plus `space`, `delete`, `nothing`.

---

## ⚠️ The critical trap: never trust a random split

The primary dataset is 87,000 images — but they are **consecutive near-duplicate
video frames of ONE signer in ONE room**. A random train/val split puts
near-identical neighbouring frames on both sides and reports a **fake ~99.9%**
accuracy.

The project is built around two consequences:

1. **We split *each class by frame ranges*** (first 80% of frames → train, last
   20% → val), never randomly. Implemented in `src/data.py::frame_range_split`,
   **enforced by `tests/test_split_leakage.py`**, and recorded in
   `data/split_manifest.json` (deterministic, byte-for-byte reproducible).
   Augmentation is applied *after* the split, train-only, so no augmented twin of
   a val frame ever leaks into train.
2. **This frame-range val is a dev number — NOT the reported metric.** The real
   benchmark is our own **webcam test set** (`data/webcam_testset/`, recorded by
   the team, see [protocol](docs/WEBCAM_TESTSET_PROTOCOL.md)). That set is
   **quarantined**: it never enters training and is never used for model
   selection. **The number we report is webcam-set accuracy — never the dev-val
   number, and never the random-split number.**

Targets:

| Metric | Target |
|---|---|
| Leakage-safe frame-range split (dev) | > 95% |
| **Own webcam test set** (the real, reported number) | ≥ 90% |
| Per-class accuracy / macro-F1 | no class below 85% |

---

## Pipelines — three different things, don't mix them up

The trained checkpoints, the live demo, and the experiments use **different
framing**. The only shared invariant is the deterministic **resize + normalize
tail** (`src/crop.py::preprocess_bgr`, FROZEN: 224², ImageNet mean/std) — guarded
by `tests/test_preprocessing_parity.py`.

### 1. Current trained model pipeline (what the checkpoints saw)

```
raw 200×200 BGR frame (Kaggle) → resize_square → 224² (letterbox, aspect kept)
→ BGR→RGB → ImageNet normalize → EfficientNet-B0 → 29 logits
```

**No MediaPipe, no hand crop.** `ASLImageDataset` reads the full frame and runs
the transform (train: heavy Albumentations first; val: the plain contract). The
checkpoints below were trained and evaluated this way — confirmed by a runtime
input trace ([`docs/reports/model_input_pipeline_ablate.md`](docs/reports/model_input_pipeline_ablate.md)).

### 2. Live / demo pipeline (`app/webcam_speller.py`)

```
webcam frame → mirror → MediaPipe HandCropper (bbox + 0.25 margin, centre fallback)
→ preprocess_bgr (same resize + normalize as training) → model → speller → TTS
```

This is where MediaPipe hand-cropping **is** used — and it is a **different
framing than training** (tight hand crop vs full frame). The webcam benchmark
exists precisely to measure the effect of that mismatch on strangers' hands;
until it is scored, the demo's real-world accuracy is unmeasured.

### 3. Experimental / future paths (not in the current checkpoints)

- **`src/cache_crops.py`** — optional pre-cropped 224² training cache (train with
  `--root data/cache_crops`). Resize is a no-op on 224² squares, so parity holds.
- **`BackgroundReplacer`** (`src/augment.py`) — train-only structural
  augmentation (`--backgrounds data/backgrounds`).
- **`efficientnet_lift`** (`src/model.py::ExposureLift`) — exposure normalization
  built into the model. **Tested and not adopted**: it lifted S (0.770 → 0.82)
  but collapsed X (→ 0.72) and dropped overall accuracy
  ([`docs/reports/dev_val_lift.json`](docs/reports/dev_val_lift.json)); an
  ablation found the lift "was never helping". See
  [`docs/RESULTS_class_fixes.md`](docs/RESULTS_class_fixes.md).
- **Landmark-MLP comparison** — scoped in
  [`docs/PROJECT_PLAN.md`](docs/PROJECT_PLAN.md), not built.

---

## Current model & results

**Architecture:** EfficientNet-B0 (ImageNet-pretrained via timm), head =
`Dropout(0.3) → Linear(1280, 29)`. Two-stage transfer: freeze backbone → train
head (LR 1e-3), then fine-tune all (LR 1e-4). Loss = cross-entropy with label
smoothing 0.1, AdamW (wd 0.01), cosine + warmup, AMP, early stopping on the
leakage-safe val.

**Deployable checkpoints** (all dev-val, leakage-safe split, 600/class):

| checkpoint | acc | macro-F1 | S recall | S→E | V→K | verdict |
|---|---|---|---|---|---|---|
| `efficientnet_b0_targeted.pt` | 0.9775 | 0.9775 | 0.770 | 138 | 13 | baseline |
| `efficientnet_b0_head_ft.pt` | 0.9776 | 0.9776 | 0.777 | 134 | 14 | no material change |
| `efficientnet_b0_partial_ft.pt` | 0.9771 | 0.9772 | **0.837** | **98** | **54** | **TRADEOFF** |

`efficientnet_b0_partial_ft.pt` is the most recent deployable model. It improves
S (recall 0.770 → 0.837, S→E 138 → 98) **but regresses V→K (13 → 54) and Y→I
(+7)** — so overall accuracy/F1 are flat and it is **not an unconditional win**.
28/29 classes meet the 85% bar; S is the only one below at 0.837. Numbers are
MPS raw-image eval (MPS-vs-CPU float flips shift a handful of S/E boundary
frames). Full details, per-class tables, and reproduce commands:
**[`docs/CURRENT_RESULTS.md`](docs/CURRENT_RESULTS.md)** and
[`docs/reports/partial_finetune_targeted.md`](docs/reports/partial_finetune_targeted.md).

**The reported webcam number does not exist yet** — the test set has not been
scored for any of these checkpoints.

---

## Repository layout

```
├── src/       # crop (preprocess contract) · data (leakage-safe split) · augment
│              # model · train · evaluate · export · cache_crops · webcam_testset
├── app/       # webcam_speller.py (live demo) · tts.py
├── scripts/   # record_webcam_testset.py, analyze_s_to_e*.py, train_*_cached.py, …
├── tests/     # split-leakage, preprocessing-parity, S→E/FT suites, …
├── models/    # MediaPipe Tasks assets (hand_landmarker.task, selfie_segmenter.tflite)
├── data/      # raw/ (gitignored) · webcam_testset/ (gitignored captures) · caches
├── checkpoints/   # efficientnet_b0_{targeted,head_ft,partial_ft}.pt
└── docs/      # see "Docs index" below
```

---

## Setup (short version)

Requires **Python 3.10**, then:

```bash
pip install -r requirements.txt        # mediapipe>=1.0, timm, torch, albumentations
```

MediaPipe needs `models/hand_landmarker.task` + `models/selfie_segmenter.tflite`
(committed; fetch with [`models/README.md`](models/README.md) if missing). Get
the Kaggle dataset, fix the `del`→`delete` rename, and build the split:

```bash
kaggle datasets download -d grassknoted/asl-alphabet
unzip -q asl-alphabet.zip -d data/raw/
mv data/raw/asl_alphabet_train/asl_alphabet_train/del \
   data/raw/asl_alphabet_train/asl_alphabet_train/delete
python -m src.data --root data/raw/asl_alphabet_train/asl_alphabet_train --counts
```

Expect 87,000 images (29 × 3000), `train=69600 val=17400 -> data/split_manifest.json`.
Full conda/venv/Colab instructions: **[`docs/SETUP.md`](docs/SETUP.md)**.

---

## Usage

```bash
# Inspect a model definition
python -m src.model --model efficientnet

# Train (two-stage transfer learning; logs to W&B if configured)
python -m src.train --root data/raw/asl_alphabet_train/asl_alphabet_train \
    --manifest data/split_manifest.json --wandb-mode online

# Evaluate — dev val split (NOT the reported number)
python -m src.evaluate --checkpoint checkpoints/efficientnet_b0_targeted.pt \
    --split data/raw/asl_alphabet_train/asl_alphabet_train \
    --manifest data/split_manifest.json

# Evaluate — THE reported benchmark (needs the webcam test set first)
python scripts/record_webcam_testset.py --person <your-id> --all-classes \
    --count 10 --condition desklamp-plainwall
python scripts/check_webcam_coverage.py --root data/webcam_testset
python -m src.evaluate --checkpoint checkpoints/efficientnet_b0_partial_ft.pt \
    --webcam data/webcam_testset --figures docs/figures

# Export for CPU demo inference
python -m src.export --checkpoint checkpoints/efficientnet_b0_partial_ft.pt --format both

# Live demo
python -m app.webcam_speller --checkpoint checkpoints/efficientnet_b0_partial_ft.pt
```

Advanced experiment commands (`--rebalance-from`, `--geometry-safe-classes auto`,
cache-crop training, partial-FT reproduce): see the docs index below.

---

## Docs index

- **[`docs/CURRENT_RESULTS.md`](docs/CURRENT_RESULTS.md)** — current state: checkpoints, the three-way dev-val comparison, honest gaps.
- **[`docs/reports/partial_finetune_targeted.md`](docs/reports/partial_finetune_targeted.md)** — the partial-FT experiment (verdict: TRADEOFF).
- **[`docs/reports/partial_finetune_training.md`](docs/reports/partial_finetune_training.md)** — the partial-FT training sweep.
- **[`docs/reports/s_to_e_investigation_timeline.md`](docs/reports/s_to_e_investigation_timeline.md)** — the full S→E diagnosis (crop → exposure → embeddings → head), what was ruled out and why.
- **[`docs/reports/head_only_cached_embeddings.md`](docs/reports/head_only_cached_embeddings.md)** — the head-only FT experiment.
- **[`docs/RESULTS.md`](docs/RESULTS.md)** — transfer-learning v1 (#6) results.
- **[`docs/RESULTS_class_fixes.md`](docs/RESULTS_class_fixes.md)** — the aimed-augmentation class-fix work (V, X), the S/exposure investigation, reproduce recipe.
- **[`docs/RESULTS_bg_ablation.md`](docs/RESULTS_bg_ablation.md)** — background-replacement ablation.
- **[`docs/PREPROCESSING_CONTRACT.md`](docs/PREPROCESSING_CONTRACT.md)** — the FROZEN resize+normalize contract and its change-control rule.
- **[`docs/WEBCAM_TESTSET_PROTOCOL.md`](docs/WEBCAM_TESTSET_PROTOCOL.md)** — how to record the reported benchmark; quarantine rules.
- **[`docs/SETUP.md`](docs/SETUP.md)** — full setup, Kaggle, and Colab recipes.
- **[`docs/PROJECT_PLAN.md`](docs/PROJECT_PLAN.md)** — plan, milestones, ethics.
- **[`docs/REPORT_TEMPLATE.md`](docs/REPORT_TEMPLATE.md)** — final-report skeleton.

---

## Tests

```bash
pytest            # runs from the repo root
```

Two suites are load-bearing:

- **`tests/test_split_leakage.py`** — proves no near-duplicate frame leaks
  between train and val (the anti-99.9%-fantasy guard).
- **`tests/test_preprocessing_parity.py`** — proves training and live-app
  resize+normalize are identical (the anti-silent-accuracy-loss guard).

The S→E diagnosis and cached fine-tunes are guarded too (`test_s_to_e_embeddings.py`,
`test_s_to_e_head.py`, `test_head_only_cached.py`, `test_partial_finetune.py`).
Tests run without a GPU: mediapipe/torch-heavy assertions skip when those packages
are missing.

---

## Ethics & honest-limits

- This is an **alphabet recognizer, not sign-language translation.** Say so, out
  loud, whenever you demo it.
- **Involve Deaf / signing users in testing.** A tool for a community should be
  evaluated by that community; hearing developers are not the arbiters of whether
  it works.
- **Be explicit about accuracy limits.** Report the honest webcam number and its
  failure modes (lighting, skin tone, hand size, left/right hand, M/N/S/T and
  K/V confusions), not the flattering split number.
- Fingerspelling is used in ASL mainly for names and loanwords; a fingerspelling
  reader is a small building block, not a communication replacement.

---

## License note

Code in this repo: see `LICENSE` (team's choice). **The ASL Alphabet dataset is
distributed under GPL-2.0** — mind its terms (notably around derivative works and
distribution) **before any commercial use**. The dataset is *not* redistributed
in this repo; download it yourself from Kaggle.

---

## References

- ASL Alphabet dataset — https://kaggle.com/datasets/grassknoted/asl-alphabet
- Mirror — https://kaggle.com/datasets/debashishsau
- Google ASL Signs (word-level upgrade) — https://kaggle.com/competitions/asl-signs
- Sign Language MNIST — https://kaggle.com/datasets/datamunge/sign-language-mnist
- timm (PyTorch Image Models) — https://github.com/huggingface/pytorch-image-models
- MediaPipe — https://developers.google.com/mediapipe
- Albumentations — https://albumentations.ai
