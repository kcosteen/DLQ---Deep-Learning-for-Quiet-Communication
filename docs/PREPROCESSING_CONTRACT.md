# Preprocessing Contract — `src/crop.py`

**Status: FROZEN** (Week 1, issue #4). This document is the change-control record.

The whole crop + resize + normalize path for **both training and live serving**
lives in `src/crop.py` and **nowhere else**. If train framing and serve framing
diverge by even one operation, the model silently loses accuracy at webcam time
with no error to point at.

## What the contract is

```
raw BGR frame
  └─ HandCropper.crop()      → square hand-crop, resized to IMG_SIZE (BGR uint8)
       └─ preprocess_bgr()   → normalize_chw(resize_square(img))
            = BGR → RGB, /255, ImageNet mean/std normalize, H×W×C → C×H×W float32
```

`preprocess_bgr(img_bgr)` is **the exact tensor the network sees** in training,
validation, and the webcam app. Nothing else produces model input.

## Frozen constants

| Constant | Value | Meaning |
|---|---|---|
| `IMG_SIZE` | `224` | square input side for the model (EfficientNet-B0) |
| `IMAGENET_MEAN` | `(0.485, 0.456, 0.406)` | RGB order; timm pretrained stats |
| `IMAGENET_STD` | `(0.229, 0.224, 0.225)` | RGB order; timm pretrained stats |
| `BBOX_MARGIN` | `0.25` | fraction of hand-bbox side added as margin per edge |

## Importers (all must route through this module)

- `src/data.py` — `ASLImageDataset` builds loaders; images read BGR via `cv2.imread`.
- `src/augment.py` — `build_transforms(train=...)`; the train/val transform **ends**
  with the same deterministic `preprocess_bgr` tail. Augmentation only adds random
  ops *before* that tail.
- `app/webcam_speller.py` — `Classifier.topk` feeds `preprocess_bgr(HandCropper
  .crop(frame).crop_bgr)` to the model.

## Model assets (MediaPipe Tasks)

The crop bridge runs on the MediaPipe **Tasks API** (`mediapipe>=1.0`; the legacy
`mp.solutions` API no longer exists). It needs two model files that are committed
to the repo and are **not** bundled in the pip wheel:

- `models/hand_landmarker.task` — used by `HandCropper`.
- `models/selfie_segmenter.tflite` — used by `BackgroundReplacer`.

If either file is missing, `HandCropper`/`BackgroundReplacer` raise a
`FileNotFoundError` pointing at `models/README.md` for the download commands.

## Change-control rule

Any change to the constants or the pure helpers (`square_bbox_with_margin`,
`center_square`, `resize_square`, `normalize_chw`, `preprocess_bgr`) requires:

1. **Sign-off from Person B (training)** — they consume the contract in every run;
2. a full re-run of `tests/test_preprocessing_parity.py` (must stay green).

Treat a proposed change as a blocker, not a tweak. Changing a constant here after a
model is trained invalidates that checkpoint's serving behavior.

## How the tests enforce it

`tests/test_preprocessing_parity.py` pins:

- determinism (same input → byte-identical output);
- output shape `(3, IMG_SIZE, IMG_SIZE)` and `float32` dtype;
- letterbox-not-stretch resize;
- ImageNet normalization statistics;
- the **critical** `test_train_val_transform_matches_app_preprocessing` — the
  training val transform equals the app's `preprocess_bgr` byte-for-byte;
- `test_app_path_is_idempotent` — the app's double `resize_square` (hand crop is
  already 224²) is a no-op, so serving still equals training.

## Why the MediaPipe crop is mandatory

The 87k Kaggle training images are a single signer in a single room. Cropping
tightly to the detected hand bbox is what lets a model trained on that data
generalise to a stranger's hand on a different webcam. The crop step is therefore
part of the contract for serving, not an optional extra.

## Smoke test (acceptance)

```
python -m src.crop   # green hand bbox + live crop window; "no hand (center
                     # fallback)" when the hand leaves the frame
```
