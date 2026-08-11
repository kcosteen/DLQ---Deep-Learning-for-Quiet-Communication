# Model Input Pipeline — Ablation Checkpoint

> **Scope:** `efficientnet_b0_ablate.pt` (arch `efficientnet`, no-lift fresh
> train, dev-val 0.9783 — `docs/reports/dev_val_ablate.json`). Everything below is
> traced from code and runtime-verified with
> `scripts/trace_input_pipeline.py`. The runtime probe ran against the derived
> frame-range split whose `content_sha256` exactly matches the ablation
> manifest (`454ca31b…`), so the batch examined is from the same val set the
> checkpoint was scored on. The forward-hook run used
> `checkpoints/efficientnet_b0_targeted.pt` as a same-arch stand-in because the
> ablate `.pt` is not present locally; the *input tensor* is
> checkpoint-independent, and both checkpoints are arch `efficientnet` with the
> identical preprocessing contract.

---

## Source image

* `.jpg` files (only `.jpg` present), **200 × 200 × 3**, `uint8` (verified on
  `data/raw/asl_alphabet_train/S/S1.jpg`).
* 29 classes (`A`–`Z`, `space`, `delete`, `nothing`); dev val = the temporal
  tail of each class (frame-range split), 17,400 images total
  (`src/data.py:150` `frame_range_split`).

## Dataset loading

* `ASLImageDataset.__getitem__` — `src/data.py:265`.
* `cv2.imread(s.path, cv2.IMREAD_COLOR)` → **BGR** `uint8` H×W×3
  (`src/data.py:269`); a failed read raises `FileNotFoundError` (271).
* Returns `(tensor, label)`; the label is **not** part of the tensor (274).

## Spatial preprocessing

* Transform chosen per class: `self.class_transforms.get(class_name,
  self.transform)` — `src/data.py:272`. For the val set
  `class_transforms` is **empty** (`build_datasets`, `src/data.py:308`), so the
  transform is `build_transforms(train=False)`.
* `build_transforms(train=False)` is exactly
  `lambda img_bgr: _to_tensor(img_bgr)` — `src/augment.py:169-170`.
* `_to_tensor` = `torch.from_numpy(preprocess_bgr(img_bgr))` —
  `src/augment.py:53`.
* `preprocess_bgr` = `normalize_chw(resize_square(img))` — `src/crop.py:197`.
* `resize_square` — `src/crop.py:164`: letterbox. `scale = 224 / max(h, w)`,
  `cv2.resize(..., INTER_AREA)`, then pasted centered onto a **224 × 224 black
  (zero) canvas**. 200 × 200 input → scale 1.12 → no padding actually needed.
  Aspect ratio **preserved**; zero padding **is** added for non-square input.
* No center crop, no hand crop, no other resize in this path.

## Augmentation

* **Validation: none.** The val transform is the deterministic
  resize+normalize contract only (`src/augment.py:169`, `src/data.py:308`).
* **Training:** `build_transforms(train=True)` — `src/augment.py:156-181` —
  runs an Albumentations pipeline (Affine rotate/perspective/translate/scale/
  shear, brightness/contrast/hue/gamma, noise/blur, CoarseDropout) under the
  `default` or `geometry_safe` policy, then the same `_to_tensor` tail.
  Background replacement is train-only and off unless `--backgrounds` is set.
* `geometry_safe` classes: applied **train-only** per-class override
  (`src/data.py:296-308`); the val set never gets a class transform.
* **Exposure/lift:** none for this model. `ExposureLift` exists only under the
  `efficientnet_lift` arch (`src/model.py:205-217`); the ablation checkpoint
  records `arch: efficientnet`.

## Tensor conversion

* `normalize_chw` — `src/crop.py:179-188`: **BGR → RGB**
  (`cv2.cvtColor(..., COLOR_BGR2RGB)`, line 184), `/255.0` to `float32`,
  per-channel `(x − mean) / std`, transpose to **C×H×W**, `.copy()` (line 188).

## Normalization

* ImageNet statistics: mean `(0.485, 0.456, 0.406)`, std `(0.229, 0.224, 0.225)`
  — frozen in `src/crop.py:47-48`, applied at lines 185-187.

## Final model input

* Per sample: `(3, 224, 224)` `float32`. Per batch: `(N, 3, 224, 224)` `float32`
  (default torch collate).
* Batch → model at `model(x)` — `src/evaluate.py:129` (under `@torch.no_grad()`)
  and `src/train.py:501` (under AMP autocast). Loader:
  `DataLoader(batch_size=64, shuffle=False)` — `src/evaluate.py:488`.
* Architecture: `build_model("efficientnet")` = `nn.Sequential(timm
  efficientnet_b0 num_classes=0 global_pool="avg", Dropout(0.3),
  Linear(1280, 29))` — `src/model.py:52-64`. Rebuilt from the checkpoint's
  recorded `arch` in `load_checkpoint` (`src/evaluate.py:90-111`).
* **No hidden preprocessing in the model wrapper** — timm does not normalize
  internally, and there is no `ExposureLift` in front of `efficientnet`
  (runtime-confirmed below).

## Runtime verification

Command:

```bash
python scripts/trace_input_pipeline.py \
  --manifest None \
  --checkpoint checkpoints/efficientnet_b0_targeted.pt
```

Results (first batch of 8, val split `A2401…`, sha `454ca31b…`):

```
batch before model: shape=(8, 3, 224, 224) dtype=torch.float32
  min=-2.1179 max=2.6400 mean=-0.3854 std=1.1083
  ch0 (R): min=-2.1179 max=1.7694 mean=-0.4444 std=0.9861
  ch1 (G): min=-2.0357 max=2.3235 mean=-0.5159 std=1.1179
  ch2 (B): min=-1.8044 max=2.6400 mean=-0.1960 std=1.1860
labels in first batch: [0, 0, 0, 0, 0, 0, 0, 0]        # class A
first 3 source files:
  data/raw/asl_alphabet_train/A/A2401.jpg
  data/raw/asl_alphabet_train/A/A2402.jpg
  data/raw/asl_alphabet_train/A/A2403.jpg
```

Forward-pre-hook on the model reports **identical** stats
(min/max/mean/std match to <1e-6), so no hidden preprocessing runs between the
loader and the first layer of EfficientNet.

Channel-order probe images (reconstructed from the normalized tensor):

* `docs/figures/model_input_pipeline_probe.png` — tensor reconstructed assuming
  **RGB**; correct skin tones on this file confirm the RGB order.
* `docs/figures/model_input_pipeline_probe_channels_swapped.png` — channels
  swapped, for comparison.

## Does `crop.py` affect this model?

**YES** — but only the pure-function half.

### Evidence

| Path | Involved? | Code |
| --- | --- | --- |
| `preprocess_bgr` / `resize_square` / `normalize_chw` | **Yes — this is the transform** | `src/crop.py:164-197`, called from `src/augment.py:53` |
| Contract constants `IMG_SIZE` / `IMAGENET_MEAN/STD` | **Yes** — define the tensor | `src/crop.py:45-48` |
| `HandCropper` / `hand_bbox` / `square_bbox_with_margin` | **No** | `src/crop.py:62-134`; not imported by `src/data.py` / `src/augment.py` |
| `BBOX_MARGIN` | **No** — only used by `hand_bbox` | `src/crop.py:50,114` |
| MediaPipe (`hand_landmarker.task`) | **No** | only `src/crop.py:79-93` (`HandCropper`) |
| `src/cache_crops.py` (HandCropper cache) | **No** — this run trained on raw `/kaggle/working/asl_train` | `docs/reports/dev_val_ablate.json` source path |

---

## Does modifying `src/crop.py` change the input distribution used by `efficientnet_b0_ablate.pt`?

**ONLY IN SOME PATHS.**

Modifying the shared contract tail — `preprocess_bgr`, `resize_square`,
`normalize_chw`, or the frozen constants (`IMG_SIZE`, `IMAGENET_MEAN/STD`) —
would change what this checkpoint receives and silently invalidate it, because
those functions are the val/train/serve transform itself. But the MediaPipe
hand-crop half of `crop.py` (`HandCropper`, `hand_bbox`, `BBOX_MARGIN`, the
centre-square no-hand fallback) is **not** in this model's input path at all —
training read full frames with no detection. So the crop-related errors in the
ablation confusion ranking are not produced by a crop step; the hand-crop code
is simply never executed for this checkpoint.
