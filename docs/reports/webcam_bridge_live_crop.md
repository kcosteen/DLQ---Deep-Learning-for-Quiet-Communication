# Live webcam bridge: MediaPipe-as-localizer crop vs the targeted checkpoint (#17)

Experiment date: 2026-08-14. No training happened; the checkpoint is unchanged.

## Question

Can MediaPipe localization turn arbitrary webcam framing into an input similar
enough to the 200x200 training framing that the existing targeted EfficientNet-B0
checkpoint works live?

## Checkpoint

`checkpoints/efficientnet_b0_targeted.pt` — `efficientnet` arch, dev-val 0.9775.
Bit-identical to the state used by every other 2026-08 run; not retrained, not
fine-tuned, not touched.

## Phase 1 — training-image framing (measured)

`scripts/measure_training_framing.py`, 20 frames/class sampled by deterministic
stride (580 frames). Framing statistics use ONLY the 426 MediaPipe-detected hand
frames; no centre fallback is counted as a hand.

| metric | median | p25–p75 |
|---|---|---|
| hand bbox width / image width | 0.31 | 0.24–0.42 |
| hand bbox height / image height | 0.40 | 0.32–0.52 |
| hand bbox area / image area | 0.127 | 0.086–0.183 |
| hand centre X (frac of width) | 0.53 | 0.43–0.65 |
| hand centre Y (frac of height) | 0.52 | 0.43–0.62 |
| context fraction (width / height) | 0.69 / 0.60 | — |
| hand fill = max(bbox_side)/img_side | **0.445** | 0.36–0.56 |
| implied margin = (1/fill − 1)/2 | **0.62** | 0.39–0.89 |

Detection coverage on hand classes: **426/560 = 76.1%** (F 100%, M 35%, N 45%,
space 45%, delete 60%). `nothing` 0/20 — correct (empty scene).

## Derived crop margin

**Default `--crop-margin` = 0.62**, the median hand-fill-implied margin: a square
crop with margin `m` makes the hand fill `1/(1+2m)` of the crop, and 0.62 maps to
~0.445 — the median training framing. This is deliberately NOT the frozen
`BBOX_MARGIN=0.25`, which would frame the hand at ~2/3 of the crop, a framing the
training data never had.

## Phase 2 — comparison artifact

`docs/figures/framing_comparison/training_vs_webcam.png` — 8 pairs (training
image with tight hand box | webcam crop). Webcam crops were synthesized
(`scripts/make_demo_webcam_crops.py`: real training hand pasted into a real
`nothing`-class background at varied scale/position, then run through the exact
app crop path) because no real webcam capture was possible this session. Demo
crops' hand-fill fraction median **0.455** ≈ training 0.445.

Caveat: numeric bbox agreement is evidence, not proof. The montage exists for
human inspection of hand/image ratio, centring, context, and wrist/arm
visibility; eyeball the artifact before trusting the default margin.

## Phase 3/4 — live pipeline + A/B

`app/webcam_speller.py` pipeline:
localize (`HandCropper.tight_bbox`, no margin, no fallback) → `crop_square_with_margin`
→ `preprocess_bgr` → checkpoint → confidence gate (`--confidence-threshold`,
default 0.70) → HUD (`Prediction: S` / `Uncertain` / `No hand`).

Example A/B (`--compare-full-frame`) on a demo webcam frame:

    FULL: nothing 0.48   CROP: A 0.72

The full-frame path answers "background", the localized crop answers the correct
hand shape. On the S1 training frame both paths agree (S 0.911 / S 0.913).

## Phase 5 — measured latency

`scripts/benchmark_pipeline.py`, 58 sampled training frames, steady state.

| stage | CPU median | CPU mean | MPS median | MPS mean |
|---|---|---|---|---|
| MediaPipe localization | 42 ms | 42 ms | 41 ms | 38 ms |
| crop + preprocess | ~1.5 ms | ~1.5 ms | ~2.7 ms | ~2.2 ms |
| EfficientNet forward | 172 ms | 146 ms | **10.9 ms** | 22 ms |
| total | 216 ms | 189 ms | **54 ms** | 63 ms |
| pipeline FPS | ~4.6 | ~5.3 | **~18.5** | ~16 |

The 15–30 FPS CPU target is not met on this machine with this checkpoint
(EfficientNet-B0 forward on CPU is the bottleneck); `--device mps` reaches it
(~16–18 FPS). Reported honestly; nothing was forced to hit the target.

## Live verification (real webcam, 2026-08-14)

Ran `python -m app.webcam_speller --checkpoint checkpoints/efficientnet_b0_targeted.pt --device mps --width 1280 --height 720 --compare-full-frame --save-crops data/webcam_crops`. 12 real crops (224x224) saved and the framing-comparison montage regenerated from them.

Measured live at 1280x720, A/B mode (`kill -TERM` clean exit):

| stage | live mean ms |
|---|---|
| MediaPipe localization | 66.5 |
| crop + preprocess | 6.3 |
| EfficientNet forward (crop) | 22.6 |
| full-frame fwd (A/B only) | 35.8 |
| total pipeline | 149.0 |
| rolling FPS / wall-clock FPS | 6.7 / 5.5 |

A/B mode doubles the forward pass; single-path would be ~113 ms total (~8.8 FPS). Localization is slower than the 200x200 benchmark because the live capture is 1280x720. Confidence stayed below the 0.70 gate for the untrained shapes shown — nothing committed spuriously; the gate worked as designed.

Two bugs found and fixed during the live run: `Timings.add` KeyError on the A/B stage (missing `setdefault`), and `mean_ms` KeyError when a stage never fired (no hand) — plus `total` was never recorded, so the FPS line never printed. Added a SIGTERM handler so `kill`/Ctrl-C exits cleanly and prints the timing report.

## Safeguards honoured

- No retraining / fine-tuning; checkpoint and frozen preprocessing contract
  unchanged; ImageNet normalization and 224x224 input untouched.
- MediaPipe is a localizer only; landmarks are not classifier features.
- No-hand frames are never classified (no centre fallback); verified by a test
  whose model stub raises if called on a no-hand frame.
- No word building or temporal smoothing added (the debounced `WordSpeller`
  state machine already in the app is untouched).
