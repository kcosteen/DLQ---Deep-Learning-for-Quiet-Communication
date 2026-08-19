# Setup

Requirements, dataset download, and the Colab recipe. The README links here for
the long version; this is the whole current content, moved out of the README so
the entry point stays short.

## Requirements

**Python 3.10.** Create the environment with conda (recommended) *or* a venv,
then install the pinned dependencies.

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
[`models/README.md`](../models/README.md).

## 1. Configure the Kaggle API

Create a token at **kaggle.com → Settings → API → Create New Token** — this
downloads `kaggle.json` (do not rename it). Then place and lock it:

```bash
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/kaggle.json    # adjust source path if needed
chmod 600 ~/.kaggle/kaggle.json
```

## 2. Download & extract the dataset

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

## 3. Verify the dataset

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

## Colab (first-cell setup)

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

## Colab training notes

Free **T4**: ~15–40 min/epoch at 224² with mixed precision (AMP). Cache the
cropped images to Google Drive once to cut epoch time, then train off the cache:

```bash
python -m src.cache_crops --src-root "$DATA/asl_alphabet_train/asl_alphabet_train" \
    --cache-root /content/drive/MyDrive/dlq/data/cache_crops   # persists across sessions
python -m src.train --root /content/drive/MyDrive/dlq/data/cache_crops --wandb-mode online
```

`cache_crops` is resumable — re-run it after a disconnect and it skips images
already cached. Demo inference runs on **CPU**.

## Experiment commands

The advanced targeting commands (`--rebalance-from`, `--geometry-safe-classes
auto`, resume-based fine-tunes, cache-crop training) are documented with the
experiments they belong to:

- Class-fix recipe + reproduce block: [`docs/RESULTS_class_fixes.md`](RESULTS_class_fixes.md)
- Partial-FT / head-only FT reproduce block: [`docs/CURRENT_RESULTS.md`](CURRENT_RESULTS.md)
- Augmentation preview: `python -m src.augment --image <img> --n 8`
  (add `--backgrounds data/backgrounds` to preview background replacement)
