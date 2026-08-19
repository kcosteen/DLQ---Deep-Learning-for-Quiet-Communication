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

## Data split warning

The 87k images are **consecutive near-duplicate video frames of one signer in
one room**. A random split leaks near-identical neighbours → fake ~99.9%.
We use **frame-range splits** (enforced by `tests/test_split_leakage.py`) and
report **webcam-set accuracy only** — never the dev-val number. See
[PROJECT_PLAN.md](docs/PROJECT_PLAN.md#critical-trap) for the full explanation.

| Metric | Target |
|---|---|
| Leakage-safe frame-range split (dev) | > 95% |
| **Own webcam test set** (reported) | ≥ 90% |
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
webcam frame → mirror → MediaPipe HandLandmarker (tight bbox, no centre fallback)
→ square crop + 0.62 training-derived margin → resize_square → 224²
→ preprocess_bgr (same resize + normalize as training) → model → speller → TTS
```

MediaPipe is a **localizer only** — it finds the hand bbox, and the crop
reproduces the training-set framing (hand fills ~45% of the crop, measured by
`scripts/measure_training_framing.py`). The 0.62 margin was selected by an
offline 4-way crop ablation (symmetric 0.62 vs forearm vs training-calibrated);
symmetric won. No-hand frames are **never** classified. The crop formula:

```python
# src/crop.py::crop_square_with_margin
side = max(bw, bh) * (1 + 2 * margin)   # margin = 0.62
cx, cy = bbox centre
square centred on (cx, cy), clamped to frame, resize to 224²
```

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

### Crop ablation (val split, 6,282 MediaPipe-detected frames)

4-way comparison on `efficientnet_b0_targeted.pt`:

| Method | Accuracy | Macro-F1 | Notes |
|---|---|---|---|
| original (all 17,400) | 0.9775 | 0.9775 | raw 200×200, no crop |
| **symmetric (0.62)** | **0.9823** | **0.9845** | **winner — used in production** |
| forearm (L=0.7 R=0.7 T=0.4 B=1.8) | 0.9769 | 0.9780 | helps S, hurts M/K/X |
| calibrated (L=1.15 R=0.88 T=0.71 B=0.57) | 0.9787 | 0.9799 | best training-framing match, hurts N |

Per-class focus (given detection):

| Class | orig | sym | fore | cal |
|---|---|---|---|---|
| S | 0.770 | 0.830 | **0.881** | 0.877 |
| E | 1.000 | 0.995 | 1.000 | 1.000 |
| M | 0.947 | 0.953 | 0.839 | **0.987** |
| N | 0.960 | **1.000** | **1.000** | 0.864 |
| K | 0.943 | **0.947** | 0.902 | 0.918 |
| V | 0.978 | **1.000** | **1.000** | **1.000** |
| X | 0.933 | **0.970** | 0.910 | 0.932 |

Derived framing (global median from 63,581 training-set detections):
L=1.15, R=0.88, T=0.71, B=0.57. Framing reconstruction analysis confirms
calibrated best matches original hand vertical centre (0.538 vs 0.534) but
the tighter symmetric crop produces better classifier accuracy.

Reports: [`docs/reports/calibrated_crop_benchmark.json`](docs/reports/calibrated_crop_benchmark.json),
[`docs/reports/forearm_crop_benchmark.json`](docs/reports/forearm_crop_benchmark.json),
[`docs/reports/framing_reconstruction_error.json`](docs/reports/framing_reconstruction_error.json).

### Live webcam test (B, C only — 46 samples)

First live-camera benchmark on `efficientnet_b0_targeted.pt`:

| Class | Samples | Accuracy |
|---|---|---|
| B | 21 | 1.000 |
| C | 25 | 1.000 |
| **Overall** | **46** | **1.000** |

A was also tested (19 samples) but scored 0% — all confused as S (15) or E (4).
This matches the known A→S/E confusion from the val split and is a classifier
weakness, not a crop/preprocessing issue. Recorded with `scripts/webcam_testset_generator.py`,
evaluated with `python -m src.evaluate --webcam data/livetest --num-workers 0`.

---

## Repository layout

```
├── src/       # crop (preprocess contract) · data (leakage-safe split) · augment
│              # model · train · evaluate · export · cache_crops · webcam_testset
├── app/       # webcam_speller.py (live demo) · tts.py
├── scripts/   # recording, analysis, benchmarking, and framing measurement tools
│              # webcam_testset_generator.py  — simple live sample collector
│              # record_webcam_testset.py     — full protocol-compliant recorder
│              # analyze_training_framing.py  — 87K training-set MediaPipe analysis
│              # benchmark_calibrated_crop.py — 4-way crop ablation
│              # benchmark_pipeline.py        — per-stage latency measurement
│              # measure_framing_reconstruction.py — training distribution matching
├── tests/     # split-leakage, preprocessing-parity, crop geometry, webcam bridge, …
├── models/    # MediaPipe Tasks assets (hand_landmarker.task, selfie_segmenter.tflite)
├── data/      # raw/ (gitignored) · webcam_testset/ · livetest/ · framing caches
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

# Collect live samples — simple (trackbar + space to record)
python scripts/webcam_testset_generator.py --camera 0

# Collect live samples — full protocol (person ID, condition tags)
python scripts/record_webcam_testset.py --person <your-id> --all-classes \
    --count 10 --condition desklamp-plainwall
python scripts/check_webcam_coverage.py --root data/webcam_testset

# Evaluate — THE reported benchmark (needs the webcam test set first)
python -m src.evaluate --checkpoint checkpoints/efficientnet_b0_targeted.pt \
    --webcam data/webcam_testset --num-workers 0

# Evaluate — quick live test
python -m src.evaluate --checkpoint checkpoints/efficientnet_b0_targeted.pt \
    --webcam data/livetest --num-workers 0

# Export for CPU demo inference
python -m src.export --checkpoint checkpoints/efficientnet_b0_partial_ft.pt --format both

# Live demo
python -m app.webcam_speller --checkpoint checkpoints/efficientnet_b0_targeted.pt \
    --device mps --debug --compare-full-frame
```

Advanced experiment commands (`--rebalance-from`, `--geometry-safe-classes auto`,
cache-crop training, partial-FT reproduce): see the docs index below.

---

## Docs index

- **[`docs/CURRENT_RESULTS.md`](docs/CURRENT_RESULTS.md)** — current state: checkpoints, the three-way dev-val comparison, honest gaps.
- **[`docs/reports/calibrated_crop_benchmark.json`](docs/reports/calibrated_crop_benchmark.json)** — 4-way crop ablation (original/symmetric/forearm/calibrated).
- **[`docs/reports/framing_reconstruction_error.json`](docs/reports/framing_reconstruction_error.json)** — which crop best reproduces training framing.
- **[`docs/reports/partial_finetune_targeted.md`](docs/reports/partial_finetune_targeted.md)** — the partial-FT experiment (verdict: TRADEOFF).
- **[`docs/reports/partial_finetune_training.md`](docs/reports/partial_finetune_training.md)** — the partial-FT training sweep.
- **[`docs/reports/s_to_e_investigation_timeline.md`](docs/reports/s_to_e_investigation_timeline.md)** — the full S→E diagnosis (crop → exposure → embeddings → head), what was ruled out and why.
- **[`docs/reports/head_only_cached_embeddings.md`](docs/reports/head_only_cached_embeddings.md)** — the head-only FT experiment.
- **[`docs/RESULTS.md`](docs/RESULTS.md)** — transfer-learning v1 (#6) results.
- **[`docs/RESULTS_class_fixes.md`](docs/RESULTS_class_fixes.md)** — the aimed-augmentation class-fix work (V, X), the S/exposure investigation, reproduce recipe.
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
