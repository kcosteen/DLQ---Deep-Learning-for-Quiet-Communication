# S -> E visual notes — efficientnet_b0_targeted (diagnosis only)

Descriptive comparison between the 138 S->E errors and the 462 correct-S
recognitions of `checkpoints/efficientnet_b0_targeted.pt` on the exact dev-val
split (`data/split_manifest.json`, sha256 `454ca31b…`, frames S2401..S3000).

This document is **descriptive only**. It records what is observed in the frames
and in the model's outputs; it proposes hypotheses, not causes, and it proposes
no fix.

## Method and limitation

- Contact sheets: `docs/figures/error_gallery_targeted/S_to_E/`
  - `A_highest_pE.png` — 25 errors with highest P(E)
  - `B_boundary_margin.png` — 25 errors with smallest positive E−S margin
  - `C_random_errors.png` — 25 deterministic random errors (seed 42)
  - `D_correct_S_control.png` — 25 deterministic random correct S (seed 42)
  Each tile is the actual model input (200×200 → `resize_square` → 224×224 RGB),
  annotated `S -> pred | E: 0.xx | S: 0.xx | margin: 0.xx | S####.jpg`.
- The authoring model cannot render the contact sheets, so the observations below
  are grounded in (1) pixel statistics measured over **all 600** S val frames,
  (2) the softmax probability records, and (3) frame-ID clustering — not in a
  visual reading of the tiles. Dimensions that can only be judged by eye
  (thumb, finger closure, occlusion, orientation) are explicitly labelled
  *requires visual inspection*; measured proxies are labelled *Observed* /
  *Possible hypothesis* / *No obvious difference*.
- Pixel stats used the same inputs the model sees (BGR->grey; background is
  pinned ~240-255 in this dataset, hand region ~9-89).

## Group summary (measured)

- Correct S: **462 / 600** (recall **0.770**); S->E errors: **138**
- P(E) on errors: mean **0.569**, median **0.549**, **max 0.833**;
  **0** errors with P(E) ≥ 0.9, **4** with P(E) ≥ 0.8, **30** with P(E) ≥ 0.7
- E−S margin on errors: median **0.360**; **20** of 138 with margin < 0.1
- S is the **2nd prediction on all 138** errors; the 3rd-place classes are
  X (66), N (37), A/V (11-13) → the S/E/X/N fist family
- Frame clustering: **27** contiguous runs; longest **41** frames (29.7% of
  errors); top-3 runs contain **63.0%** of errors

## Headline observations (no fix proposed)

1. **The model is never confident these are E.** The strongest error is only
   0.833 P(E); most errors sit at 0.45-0.68. S->E is a moderate-confidence
   near-family decision (S/E/X/N), not a confident misread.
2. **The errors are strongly temporally clustered.** ~63% fall inside three
   capture segments: ≈2575-2619 (44 frames), ≈2801-2814 (17), ≈2831-2877 (50).
   A ~157-frame error-free stretch (2620-2776) sits between the first two
   clusters. This is a handful of repeated fist poses failing across their
   near-duplicate video frames — a different problem from 138 independent
   failures.
3. **Measured pixels differ only modestly.** Error frames are marginally
   brighter (median grey 125 vs 115; mean 141 vs 133) with less dark-pixel area
   (0.32 vs 0.40); background median and sharpness are essentially identical.
   The direction is consistent with the existing #12 exposure hypothesis but the
   groups overlap heavily and the gap is ~3-4% — not decisive on its own.

## Per-dimension comparison (S->E vs S->S)

### Hand orientation
- **Requires visual inspection.** No reliable proxy was measurable: the dark
  pixels (hand+arm+shadows) span the full frame, so a bbox-based orientation
  estimate degenerated to the whole image. The frame clustering shows the errors
  come from distinct capture segments, which may correspond to distinct poses,
  but orientation itself is not measured here.

### Thumb position / visibility
- **Requires visual inspection.** No pixel proxy distinguishes thumb state from
  finger closure or arm pixels.

### Finger closure
- **Possible hypothesis.** Error frames have ~8 percentage points less dark-pixel
  area (0.317 vs 0.395) — consistent with a less fully closed fist (more bright
  background visible around/through the hand). This overlaps heavily with the
  correct group and cannot be separated from arm/sleeve/shadow effects, so it is
  a hypothesis, not a finding.

### Occlusion
- **Requires visual inspection.** Nothing measurable separates the groups.

### Hand position
- **No obvious difference.** The dark-region centroid was not a usable proxy (the
  largest dark component spans the whole frame), and no other measured statistic
  separated the groups.

### Hand scale
- **Possible hypothesis.** Less dark-pixel area in the error group (0.317 vs
  0.395) is consistent with a smaller or thinner hand silhouette, with the same
  caveats as finger closure.

### Brightness / exposure
- **Observed.** Error frames are brighter: median grey **124.7 vs 115.4**
  (+9.3), mean **141.1 vs 133.2** (+8.0). The lower tail (p25 grey) differs
  little (68.2 vs 64.7). Distributions overlap substantially. The direction
  matches the #12 exposure story (a brighter, less shadowed fist reads as E),
  but the effect is small and not decisive.

### Shadows
- **Observed.** Very-dark fraction is slightly *lower* in errors (0.179 vs
  0.197) — fewer deep-shadow pixels, consistent with the brighter-exposure
  observation. Small effect.

### Blur
- **No obvious difference.** Laplacian variance 1276 vs 1204 — errors are if
  anything marginally sharper; both groups are in focus.

### Background
- **No obvious difference.** Background grey median ~251 in both groups (pinned
  bright); the mean gap (+8) is driven by a few correct frames with slightly
  darker backgrounds, not a systematic difference.

### Repeated pose / frame sequences
- **Observed (strongest signal).** 27 contiguous runs; longest run 41 frames
  (29.7%); top-3 runs 63.0%. Two dominant segments (≈2575-2619, ≈2831-2877)
  plus ≈2801-2814, with an error-free 2620-2776 stretch between them. The 138
  errors are concentrated in a few repeated poses, not spread evenly across the
  600 S frames.

## Bottom line (diagnosis only)

S->E is not "the model confidently reads an S handshape as E." It is (a) a
moderate-confidence decision inside the S/E/X/N fist family (max P(E) 0.833,
S always 2nd) and (b) concentrated in a few contiguous capture segments — a
small number of repeated fist poses that sit near the S/E boundary. Measured
pixel differences (brighter, less dark area in errors) are consistent with the
existing exposure hypothesis but are small and overlapping. No fix is proposed
in this document.
