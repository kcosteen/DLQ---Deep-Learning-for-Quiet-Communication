# ASL Fingerspelling Recognition

A computer-vision model that reads the **American Sign Language hand alphabet** and,
live from a webcam, spells words as **text and speech**. Python + PyTorch, transfer
learning on EfficientNet-B0, trainable on a Colab free GPU. 3-person team, ~8 weeks.

> **Read this first — what this project is (and is not).**
> This is an **alphabet / fingerspelling recognizer**, *not* "sign language
> translation." ASL is a full language with its own grammar, facial grammar, and
> movement; recognizing 26 static handshapes + a few control gestures is a narrow
> slice of it. Please read the [Ethics & honest-limits](#ethics--honest-limits)
> section before demoing this to anyone.

---

## What it does

Live pipeline:

```
OpenCV capture
  → MediaPipe Hands detect & crop to the hand bounding box   (mandatory — see below)
  → identical resize(224²) + ImageNet normalize
  → EfficientNet-B0 predicts a letter + confidence
  → speller: commit a letter after ~0.5 s of stable, confident prediction;
     `space` / `delete` gestures edit the word; `nothing` = idle
  → on word completion, pyttsx3 speaks the word aloud
```

The HUD shows the live hand crop, the top-3 letters with confidence, and the word
being built. Capture / inference / TTS run decoupled; target **15–30 FPS** on CPU.

**29 classes:** A–Z plus `space`, `delete`, `nothing`.

---

## ⚠️ The critical trap: never trust a random split

The primary dataset is 87,000 images — but they are **consecutive near-duplicate
video frames of ONE signer in ONE room**. Two consequences drive the whole design:

1. **Never split randomly.** A random train/val split puts near-identical
   neighbouring frames on both sides and reports a **fake ~99.9%** accuracy. We
   split *each class by frame ranges* (first 80% of frames → train, last 20% →
   val). This is implemented in `src/data.py::frame_range_split`, **enforced by
   `tests/test_split_leakage.py`**, and recorded in `data/split_manifest.json`
   (deterministic, byte-for-byte reproducible from `data/raw`). Even so, this
   frame-range val is only a **dev val (temporal split, same signer — NOT the
   reported metric)**; augmentation is applied *after* the split, train-only, so no
   augmented twin of a val frame ever leaks into train.
2. **The real benchmark is our own webcam test set.** Each teammate records all 29
   signs in varied lighting/backgrounds (`data/webcam_testset/`); none of it enters
   training. **The number we report is webcam-set accuracy — never the naïve
   random-split number.**

The **MediaPipe hand-crop is mandatory**: cropping tightly to the detected hand is
what lets a model trained on one signer in one room work on a stranger's hand on a
different webcam. Training images are cropped with the *same* code the app uses
(`src/crop.py`) — a mismatch causes silent accuracy loss, so it's guarded by
`tests/test_preprocessing_parity.py`.

---

## Repository layout

```
asl-fingerspelling/
├── README.md · CONTRIBUTING.md · requirements.txt
├── data/
│   ├── raw/              # Kaggle asl_alphabet_train/ (29 folders) — gitignored
│   └── webcam_testset/   # our own captured crops — the REAL benchmark
├── notebooks/            # EDA, confusion-matrix analysis
├── src/
│   ├── crop.py           # MediaPipe hand-crop bridge + shared preprocessing contract
│   ├── data.py           # folder loader, leakage-safe frame-range split, Dataset
│   ├── augment.py        # augmentation + background replacement
│   ├── model.py          # EfficientNet-B0 head + compact-CNN baseline
│   ├── train.py          # two-stage transfer learning + W&B
│   ├── evaluate.py       # metrics, confusion matrix, webcam-set eval
│   └── export.py         # TorchScript / ONNX export
├── app/
│   ├── webcam_speller.py # OpenCV + MediaPipe → model → word building → TTS
│   └── tts.py
├── tests/                # split-leakage + preprocessing-parity tests
└── docs/                 # PROJECT_PLAN.md, REPORT_TEMPLATE.md, figures
```

---

## Setup

Requires **Python 3.10**. Create the environment with conda (recommended) *or* a
venv, then install the pinned dependencies.

**Option A — conda (recommended):**

```bash
conda create -n dlq python=3.10 -y
conda activate dlq
pip install -r requirements.txt
```

**Option B — venv (if you don't have conda).** Needs a Python 3.10 interpreter; if
your system Python isn't 3.10, install one with `pyenv` first:

```bash
pyenv install 3.10.20        # skip if you already have Python 3.10
pyenv local 3.10.20          # or: use your own python3.10
python3.10 -m venv .venv
source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

> Option B (venv on `pyenv`-provided Python 3.10.20) is the path verified on macOS
> for this repo. `.venv/` is git-ignored.

**MediaPipe Tasks model assets:** the crop bridge uses `mediapipe>=1.0` (Tasks
API; the legacy `mp.solutions` API was removed in 1.x). It needs
`models/hand_landmarker.task` and `models/selfie_segmenter.tflite`, which are
committed to the repo. If they're missing, fetch them with the `curl` commands in
[`models/README.md`](models/README.md).

### 1. Configure the Kaggle API

Create a token at **kaggle.com → Settings → API → Create New Token** — this
downloads `kaggle.json` (do not rename it). Then place and lock it:

```bash
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/kaggle.json    # adjust source path if needed
chmod 600 ~/.kaggle/kaggle.json
```

### 2. Download & extract the dataset

```bash
kaggle datasets download -d grassknoted/asl-alphabet
mkdir -p data/raw
unzip -q asl-alphabet.zip -d data/raw/
rm asl-alphabet.zip                                  # ~1 GB — remove after extracting
```

The archive extracts to a **doubly-nested** folder; the training images land at
`data/raw/asl_alphabet_train/asl_alphabet_train/{A..Z,space,delete,nothing}/`.

**One required fixup:** Kaggle ships the delete-gesture folder as `del`, but the
canonical class name (`src.data.CLASSES`) is `delete`. Rename it so the class
counts line up:

```bash
mv data/raw/asl_alphabet_train/asl_alphabet_train/del \
   data/raw/asl_alphabet_train/asl_alphabet_train/delete
```

> **Mirror** if the primary 404s: `kaggle datasets download -d
> debashishsau/aslamerican-sign-language-aplhabet-dataset`, and adjust the `unzip`
> destination so the final layout matches the path above.

### 3. Verify the dataset

```bash
python -m src.data --root data/raw/asl_alphabet_train/asl_alphabet_train --counts
```

Expect **29 classes × 3,000 images = 87,000 total** (each image 200×200 RGB). The
output ends with:

```
       A: 3000
       ...
   space: 3000
  delete: 3000
 nothing: 3000
   TOTAL: 87000
train=69600  val=17400  -> data/split_manifest.json
```

If `delete` shows `0` and the total is `84000`, you skipped the `del`→`delete`
rename in step 2. (This command also writes the leakage-safe split to
`data/split_manifest.json`.)

> **Never commit the dataset.** Everything under `data/raw/` is git-ignored (see
> `data/raw/.gitignore`); the 87k images and `data/split_manifest.json` must never
> be staged.

### Colab (first-cell setup)

On Colab, run the same steps in the **first notebook cell** and mount Google Drive
so the unzipped dataset is cached across sessions instead of re-downloaded:

```python
from google.colab import drive
drive.mount('/content/drive')
DATA = '/content/drive/MyDrive/dlq/data/raw'         # persists across sessions
import os; os.makedirs(DATA, exist_ok=True)

# upload kaggle.json to the session first, then:
!pip install -q -r requirements.txt
!mkdir -p ~/.kaggle && cp kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
!if [ ! -d "$DATA/asl_alphabet_train" ]; then \
    kaggle datasets download -d grassknoted/asl-alphabet && \
    unzip -q asl-alphabet.zip -d "$DATA/" && rm asl-alphabet.zip && \
    mv "$DATA/asl_alphabet_train/asl_alphabet_train/del" \
       "$DATA/asl_alphabet_train/asl_alphabet_train/delete"; fi
!python -m src.data --root "$DATA/asl_alphabet_train/asl_alphabet_train" --counts
```

(Torch/torchvision are preinstalled on Colab; the `pip install` still pulls the rest.)

---

## Usage

```bash
# 1. Inspect a model
python -m src.model --model efficientnet

# 2. Train (two-stage transfer learning; logs to W&B if configured)
python -m src.train --root data/raw/asl_alphabet_train/asl_alphabet_train \
    --manifest data/split_manifest.json --wandb-mode online

# 3. Evaluate — dev val split (NOT the reported number)
python -m src.evaluate --checkpoint checkpoints/efficientnet_b0.pt \
    --split data/raw/asl_alphabet_train/asl_alphabet_train --figures docs/figures

# 4. Evaluate — THE reported benchmark (our webcam test set)
python -m src.evaluate --checkpoint checkpoints/efficientnet_b0.pt \
    --webcam data/webcam_testset --figures docs/figures

# 5. Export for CPU demo inference
python -m src.export --checkpoint checkpoints/efficientnet_b0.pt --format both

# 6. Live demo
python -m app.webcam_speller --checkpoint checkpoints/efficientnet_b0.pt
```

Preview the augmentation pipeline / smoke-test the crop & TTS:

```bash
python -m src.augment --image data/raw/asl_alphabet_train/asl_alphabet_train/A/A1.jpg --n 8
python -m src.crop            # live crop viewer
python -m app.tts "hello world"
```

### Colab notes
Free **T4**: ~15–40 min/epoch at 224² with mixed precision (AMP). Cache the resized
images to Google Drive to cut epoch time. Demo inference runs on **CPU**.

---

## Model & training (summary)

- **EfficientNet-B0**, ImageNet-pretrained (`timm`); head = `Dropout(0.3) → Linear(1280 → 29)`.
  - **Stage 1:** freeze backbone, train head (LR 1e-3).
  - **Stage 2:** unfreeze, fine-tune whole net (LR 1e-4).
- **Baselines:** compact 3-block CNN from scratch; logistic-regression-on-pixels
  sanity floor.
- **Loss:** cross-entropy, label smoothing 0.1. **Optimizer:** AdamW (wd 0.01),
  cosine decay + warmup, early stopping on the **leakage-safe** val split, AMP,
  batch 32–64 at 224².
- **Preprocessing:** resize 224², RGB, ImageNet mean/std — the **exact same
  `src/crop.py` pipeline** runs in training and the app (parity-tested).
- **Augmentation** (the single-signer antidote): geometric (rotation ±15–20°,
  shift, scale, perspective, shear); photometric (brightness/contrast/hue jitter,
  gamma, noise, blur); structural (cutout, background replacement on
  MediaPipe-segmented hands). Albumentations.

### Targets & honest-metrics policy
| Metric | Target |
|---|---|
| Leakage-safe frame-range split accuracy | **> 95%** |
| **Own webcam test set** (the real, reported number) | **≥ 90%** |
| Per-class accuracy / macro-F1 | **no class below 85%** |

**We never report the naïve random-split number.** Watch known confusions:
**M/N/S/T, A/E, K/V**.

---

## Milestones (8 weeks)

1. **Foundations** — repo, env, Kaggle download, EDA, leakage-safe split + split-test,
   lock the crop/preprocessing contract, compact-CNN baseline.
2. **Transfer learning v1** — EfficientNet-B0 two-stage training, honest validation
   number, eval harness + confusion matrix.
3. **Augmentation & robustness** — heavy augmentation + background replacement,
   MediaPipe-crop the training data, fix worst-confused classes.
4. **Webcam bridge + letter demo** — MediaPipe crop → model live; first
   "one letter at a time" demo.
5. **Capture our own test set** — everyone records every sign; measure real
   accuracy; diagnose the gap.
6. **Close the gap + word-speller** — reach ≥90% webcam accuracy; word-building +
   space/delete + TTS. **★ Milestone demo.**
7. **Polish & benchmark** — optional landmark-MLP comparison, export model, UX
   polish, freeze final metrics.
8. **Report & ship** — final report, figures, README, recorded demo video, retro +
   scope the word-level upgrade.

Full plan: [`docs/PROJECT_PLAN.md`](docs/PROJECT_PLAN.md). Report skeleton:
[`docs/REPORT_TEMPLATE.md`](docs/REPORT_TEMPLATE.md).

---

## Tests

```bash
pytest            # runs from the repo root
```

Two tests are load-bearing for this project's integrity:

- **`tests/test_split_leakage.py`** — proves no near-duplicate frame leaks between
  train and val (the anti-99.9%-fantasy guard).
- **`tests/test_preprocessing_parity.py`** — proves the training and the live-app
  preprocessing are byte-for-byte identical (the anti-silent-accuracy-loss guard).

Tests are written to run without a GPU: torch/mediapipe-dependent assertions are
skipped automatically if those packages aren't installed, but the split-leakage and
core contract tests only need numpy + OpenCV.

The preprocessing contract (`IMG_SIZE`, ImageNet mean/std, `BBOX_MARGIN`) is
**frozen** — any change requires Person B sign-off. Full spec and change-control
rule: [`docs/PREPROCESSING_CONTRACT.md`](docs/PREPROCESSING_CONTRACT.md).

---

## Upgrade path (documented, not built now)

From fingerspelling to **word-level** recognition: **Google ASL Signs**
(`kaggle.com/competitions/asl-signs`) — 94,744 clips, 250 words, 21 signers,
landmark sequences → a small Transformer. Scoped in Week 8's retro; see
`docs/PROJECT_PLAN.md`.

---

## Ethics & honest-limits

- This is an **alphabet recognizer, not sign-language translation.** Say so, out
  loud, whenever you demo it.
- **Involve Deaf / signing users in testing.** A tool for a community should be
  evaluated by that community; hearing developers are not the arbiters of whether
  it works.
- **Be explicit about accuracy limits.** Report the honest webcam number and its
  failure modes (lighting, skin tone, hand size, left/right hand, the M/N/S/T etc.
  confusions), not the flattering split number.
- Fingerspelling is used in ASL mainly for names and loanwords; a fingerspelling
  reader is a small building block, not a communication replacement.

---

## License note

Code in this repo: see `LICENSE` (team's choice). **The ASL Alphabet dataset is
distributed under GPL-2.0** — mind its terms (notably around derivative works and
distribution) **before any commercial use**. The dataset is *not* redistributed in
this repo; download it yourself from Kaggle.

---

## References

- ASL Alphabet dataset — https://kaggle.com/datasets/grassknoted/asl-alphabet
- Mirror — https://kaggle.com/datasets/debashishsau
- Google ASL Signs (word-level upgrade) — https://kaggle.com/competitions/asl-signs
- Sign Language MNIST — https://kaggle.com/datasets/datamunge/sign-language-mnist
- timm (PyTorch Image Models) — https://github.com/huggingface/pytorch-image-models
- MediaPipe — https://developers.google.com/mediapipe
- Albumentations — https://albumentations.ai
