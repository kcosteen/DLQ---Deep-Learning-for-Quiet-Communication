# ASL Fingerspelling Recognition — Project Plan

**Version 3.0 · 2026-07-27**

> This document is the authoritative project plan. The README is the quick start;
> this is the full spec.

## Goal

A computer-vision model that reads the ASL hand alphabet and, live from a webcam,
spells words as text and speech. 3-person team, ~8 weeks. Python + PyTorch,
transfer learning (EfficientNet-B0), trains on Colab free GPU.

## Data

**Primary dataset — ASL Alphabet** (Kaggle):
`kaggle datasets download -d grassknoted/asl-alphabet`
- 87,000 RGB 200×200 images, 29 classes (A–Z + `space`/`delete`/`nothing`),
  folder-per-class (~3,000 each), zero missing values, GPL-2.0.
- Mirror: `kaggle.com/datasets/debashishsau`.

**Upgrade path (documented, not built now) — Google ASL Signs:**
`kaggle.com/competitions/asl-signs` — 94,744 clips, 250 words, 21 signers;
landmark sequences → small Transformer.

### Critical trap
The 87k images are **consecutive near-duplicate video frames of ONE signer in ONE
room**. **Never split randomly** — a random split leaks near-identical frames and
reports a fake ~99.9%. Split each class **by frame ranges** (e.g. first 80% of
frames → train, last 20% → val). The **real benchmark is our own webcam test set**:
each teammate records all 29 signs in varied lighting/backgrounds; none of it enters
training. **Never let augmented copies cross the split.**

## Scope

29-class fingerspelling classifier wrapped in a live webcam word-speller (stable
letters → words; `space`/`delete` gestures edit the word; `nothing` = idle) with
TTS.

## Model & training

- **EfficientNet-B0**, ImageNet-pretrained (timm), head =
  `Dropout(0.3) → Linear(1280 → 29)`.
  - **Stage 1:** freeze backbone, train head.
  - **Stage 2:** unfreeze, fine-tune whole net at low LR.
- **Baseline:** compact 3-block CNN from scratch. **Sanity floor:** logistic
  regression on a pixel subset.
- **Loss:** cross-entropy, label smoothing 0.1. **Optimizer:** AdamW (LR 1e-3 head
  stage, 1e-4 fine-tune, weight decay 0.01). Cosine decay + warmup; early stopping
  on the leakage-safe val split; mixed precision (AMP); batch 32–64 at 224×224.
- **Preprocessing:** resize 224×224, keep RGB, ImageNet mean/std normalization.
  Recommended: MediaPipe hand-crop the training images too, so train and webcam
  framing match. **The exact same pipeline must run in the webcam app** — mismatch
  = silent accuracy loss (guard with a preprocessing-parity test).
- **Augmentation (the single-signer antidote):**
  - *geometric:* rotation ±15–20°, shift, scale/zoom, perspective, small shear.
  - *photometric:* brightness/contrast/hue jitter, gamma, Gaussian noise & blur.
  - *background/structure:* random erasing/cutout, random background replacement on
    MediaPipe-segmented hands.
  - Use **Albumentations**.
- **Tooling:** Python 3.10+, PyTorch + timm, torchvision/Albumentations,
  OpenCV + MediaPipe, pyttsx3, Weights & Biases (free), Kaggle API,
  conda + requirements.txt. Colab free T4: ~15–40 min/epoch at 224×224 with AMP;
  cache resized images to Drive. Demo inference runs on CPU.

## Targets

- Leakage-safe split accuracy **> 95%**.
- **Own webcam test set ≥ 90%** (the real, reported number).
- **No class below 85%** per-class accuracy / macro-F1.
- **Never report the naïve random-split number.**
- Watch confusions: **M/N/S/T, A/E, K/V**.

## Real-time demo pipeline

OpenCV capture → **MediaPipe Hands detect & crop to hand bbox** → identical
resize/normalize → EfficientNet predicts letter + confidence → speller (commit a
letter after ~0.5 s stable prediction; confidence gate ignores low-confidence/no-hand
frames; `space`/`delete` edit the word; `nothing` = idle) → on word completion
pyttsx3 speaks it. Show the crop, top-3 letters + confidence, and the word being
built. Threads: capture / inference / TTS separate. Target **15–30 FPS**. "One
letter" demo by **Week 4**, full word-speller by **Week 6**. **The MediaPipe crop
step is mandatory** — it's what makes 87k single-signer images work on a stranger's
hand.

## Ethics note (also in README)

This is an **alphabet recognizer, not "sign language translation."** Involve
Deaf/signing users in testing; be clear about accuracy limits.

## Weekly milestones

1. **Week 1 — Foundations:** repo, env, Kaggle download, EDA, leakage-safe split +
   split-test, lock the crop/preprocessing contract, compact-CNN baseline.
2. **Week 2 — Transfer learning v1:** EfficientNet-B0 two-stage training, honest
   validation number on the frame-range split, eval harness + confusion matrix.
3. **Week 3 — Augmentation & robustness:** heavy augmentation + background
   replacement, MediaPipe hand-crop the training data, fix worst-confused classes.
4. **Week 4 — Webcam bridge + letter demo:** MediaPipe crop → model live; first
   "one letter at a time" demo.
5. **Week 5 — Capture our own test set:** all three record every sign in varied
   conditions; measure real accuracy; diagnose the gap.
6. **Week 6 — Close the gap + word-speller:** augment/optionally fine-tune toward
   ≥90% webcam accuracy; add word-building + `space`/`delete` + TTS. **Milestone.**
7. **Week 7 — Polish & benchmark:** optional landmark-MLP comparison, export model,
   UX polish, freeze final metrics.
8. **Week 8 — Report & ship:** final report, figures, README, recorded demo video,
   retro + scope the word-level upgrade (Google ASL Signs).

## Team split

- **Person A · Data & Augmentation:** Kaggle download, EDA, class balance;
  leakage-safe split + the split-leakage test; augmentation + background
  replacement; coordinates the shared webcam test set.
- **Person B · Modeling & Training:** compact-CNN baseline → EfficientNet transfer
  learning; two-stage training, W&B, tuning; optional landmark-MLP robustness model;
  model export (TorchScript/ONNX).
- **Person C · Demo & Evaluation:** MediaPipe crop bridge (shared with training);
  webcam app, word-building, debounce, TTS; evaluation harness; runs the webcam
  benchmark; final report, figures, README, demo video.

## Sync points

- Lock the crop + preprocessing contract in Week 1 so the MediaPipe bridge is
  identical in training and the app — guarded by the preprocessing-parity test.
- Twice-weekly 30-min syncs; PRs on the shared repo; shared W&B project.

## References

- ASL Alphabet — `kaggle.com/datasets/grassknoted/asl-alphabet`
- Mirror — `kaggle.com/datasets/debashishsau`
- Google ASL Signs — `kaggle.com/competitions/asl-signs`
- Sign Language MNIST — `kaggle.com/datasets/datamunge/sign-language-mnist`
- timm — `github.com/huggingface/pytorch-image-models`
- MediaPipe — `developers.google.com/mediapipe`
- Albumentations — `albumentations.ai`
