# S -> E transition-run visual notes — efficientnet_b0_targeted

Follow-up to `s_to_e_visual_notes.md`. Instead of group-level averages, this document watches each of the three dominant contiguous error segments over time: what the frames and the model's probabilities do right before a run begins, across the run, and when prediction returns to S.

Generated 2026-08-09T15:52:55+00:00 by `scripts/inspect_s_to_e_runs.py` (diagnosis only; no fixes).

## Runs studied

- **Run 1** S2575-S2619 — errors S2575-S2582, S2584-S2619 (44 errors; correct blip S2583)
- **Run 2** S2801-S2814 — errors except the correct blip S2811 (13 errors)
- **Run 3** S2831-S2877 — errors S2831-S2871, S2875-S2877 (44 errors; correct blips S2872-S2874)
- Long error-free stretch between Run 1 and Run 2: S2620-S2776 (157 frames)

## Galleries (for human review)

- `docs/figures/error_gallery_targeted/S_to_E/E_transition_runs.png` — one chronological filmstrip per run. Tile order is last correct-S frames before the run -> first error -> evenly sampled run frames -> last error -> first correct-S after the run. Each tile is the actual model input (200x200 -> `resize_square` -> 224x224 RGB), annotated `S2575 | pred=E | S=.39 E=.52`.
- `docs/figures/error_gallery_targeted/S_to_E/F_pose_comparison.png` — 4 rows x 5 tiles: high-confidence correct S, dominant-run S->E, boundary-margin S->E, correct S inside S2620-2776.

## Method and limitation

As in `s_to_e_visual_notes.md`: the authoring model cannot render the images, so the observations below are grounded in (1) the softmax probability records, (2) measured pixel signals at the run boundaries (grey mean, very-dark fraction `<100` of the raw frame), and (3) frame clustering. Hand-geometry dimensions (thumb, finger closure, orientation, occlusion) are **not** measurable without a clean hand silhouette (dark pixels span the whole frame) and are labelled *requires visual inspection*.

Tables: `pred` is the argmax class, `P(S)` / `P(E)` are softmax probabilities, `grey` is mean grey level of the raw frame, `dark` is the fraction of pixels below grey 100.

## Run 1: S2575-S2619

Errors: **44** frames (31.9% of all S->E errors); correct blips inside the range: S2583. P(E) within run: 0.33 - 0.83.

| frame | pred | P(S) | P(E) | grey | dark |
|-------|------|------|------|------|------|
| S2572 | S | 0.50 | 0.24 | 118 | 0.43 |
| S2573 | S | 0.41 | 0.21 | 120 | 0.42 |
| S2574 | S | 0.47 | 0.31 | 121 | 0.41 |
| S2575 | E | 0.20 | 0.54 | 121 | 0.41 |
| S2601 | E | 0.03 | 0.83 | 125 | 0.41 |
| S2619 | E | 0.30 | 0.33 | 128 | 0.42 |
| S2620 | S | 0.56 | 0.11 | 129 | 0.41 |
| S2621 | S | 0.70 | 0.09 | 131 | 0.41 |
| S2622 | S | 0.81 | 0.04 | 133 | 0.41 |

- **What changes immediately before S->E begins (measured):** the 3 last correct-S frames average grey 120 and dark 0.42; the first error S2575 is grey 121 and dark 0.41 (brighter by 2). P(S) drops from 0.47 to 0.20 while P(E) rises to 0.54 — the decision flips in one frame.
- **What changes immediately before S->E begins (visual):** **requires visual inspection** of the boundary tiles (`S2572`..`S2619`) in `E_transition_runs.png` — thumb position, finger closure and orientation cannot be measured by this tool.

- **What stays consistent through the run:** S remains the 2nd prediction on every error frame and P(E) stays moderate (0.33-0.83, never a confident read). The run is one repeated pose captured across many near-duplicate video frames, not 44 independent failures.

- **What changes when prediction returns to S:** the first correct-S frame after the run is S2620 (P(S) 0.56); the last error S2619 had P(S) 0.30. Measured pixels: grey 128 -> 129 (brighter by 2). Whether the hand *shape* changed is **requires visual inspection**.

## Run 2: S2801-S2814

Errors: **13** frames (9.4% of all S->E errors); correct blips inside the range: S2811. P(E) within run: 0.37 - 0.72.

| frame | pred | P(S) | P(E) | grey | dark |
|-------|------|------|------|------|------|
| S2797 | S | 0.45 | 0.26 | 163 | 0.19 |
| S2798 | S | 0.48 | 0.23 | 163 | 0.19 |
| S2800 | S | 0.37 | 0.32 | 164 | 0.19 |
| S2801 | E | 0.31 | 0.37 | 164 | 0.19 |
| S2813 | E | 0.08 | 0.72 | 162 | 0.19 |
| S2814 | E | 0.32 | 0.42 | 161 | 0.20 |
| S2815 | S | 0.77 | 0.06 | 161 | 0.20 |
| S2816 | S | 0.63 | 0.13 | 160 | 0.20 |
| S2817 | S | 0.46 | 0.29 | 159 | 0.21 |

- **What changes immediately before S->E begins (measured):** the 3 last correct-S frames average grey 163 and dark 0.19; the first error S2801 is grey 164 and dark 0.19 (brighter by 0). P(S) drops from 0.37 to 0.31 while P(E) rises to 0.37 — the decision flips in one frame.
- **What changes immediately before S->E begins (visual):** **requires visual inspection** of the boundary tiles (`S2797`..`S2814`) in `E_transition_runs.png` — thumb position, finger closure and orientation cannot be measured by this tool.

- **What stays consistent through the run:** S remains the 2nd prediction on every error frame and P(E) stays moderate (0.37-0.72, never a confident read). The run is one repeated pose captured across many near-duplicate video frames, not 13 independent failures.

- **What changes when prediction returns to S:** the first correct-S frame after the run is S2815 (P(S) 0.77); the last error S2814 had P(S) 0.32. Measured pixels: grey 161 -> 161 (darker by 1). Whether the hand *shape* changed is **requires visual inspection**.

## Run 3: S2831-S2877

Errors: **44** frames (31.9% of all S->E errors); correct blips inside the range: S2872, S2873, S2874. P(E) within run: 0.35 - 0.78.

| frame | pred | P(S) | P(E) | grey | dark |
|-------|------|------|------|------|------|
| S2828 | S | 0.76 | 0.06 | 157 | 0.22 |
| S2829 | S | 0.45 | 0.21 | 157 | 0.22 |
| S2830 | S | 0.36 | 0.29 | 156 | 0.22 |
| S2831 | E | 0.14 | 0.54 | 156 | 0.22 |
| S2864 | E | 0.06 | 0.78 | 155 | 0.24 |
| S2877 | E | 0.34 | 0.43 | 153 | 0.26 |
| S2878 | S | 0.46 | 0.29 | 153 | 0.26 |
| S2879 | S | 0.41 | 0.33 | 153 | 0.26 |
| S2880 | S | 0.56 | 0.21 | 153 | 0.26 |

- **What changes immediately before S->E begins (measured):** the 3 last correct-S frames average grey 157 and dark 0.22; the first error S2831 is grey 156 and dark 0.22 (darker by 0). P(S) drops from 0.36 to 0.14 while P(E) rises to 0.54 — the decision flips in one frame.
- **What changes immediately before S->E begins (visual):** **requires visual inspection** of the boundary tiles (`S2828`..`S2877`) in `E_transition_runs.png` — thumb position, finger closure and orientation cannot be measured by this tool.

- **What stays consistent through the run:** S remains the 2nd prediction on every error frame and P(E) stays moderate (0.35-0.78, never a confident read). The run is one repeated pose captured across many near-duplicate video frames, not 44 independent failures.

- **What changes when prediction returns to S:** the first correct-S frame after the run is S2878 (P(S) 0.46); the last error S2877 had P(S) 0.34. Measured pixels: grey 153 -> 153 (darker by 0). Whether the hand *shape* changed is **requires visual inspection**.

## Cross-run pattern (measured)

- All three runs share the same probability signature: moderate P(E), S always 2nd, max P(E) 0.83 (Run 1 0.83, Run 2 0.72, Run 3 0.78).
- Segment grey means: Run 1 124, Run 2 163, Run 3 155 (vs 133 whole-set mean, 141 mean over all errors) — the runs are not an exposure outlier, so brightness is at most a contributing factor.
- The three runs cover 101 of 138 errors (73.2%) and each starts at a clean correct-S boundary: correct -> error in a single frame.
- **Cross-run geometry (visual):** **requires visual inspection.** If the three runs share one hand-shape feature (e.g. thumb visible in the same spot), `F_pose_comparison.png` row 2 should show it; this tool cannot confirm it.

## Decision gate

Is there a repeatable hand-geometry subtype that separates these S->E frames from correct S? **UNCLEAR (measured evidence only).** The temporal clustering and the identical per-run probability profile strongly indicate a handful of *repeated* poses near the S/E boundary — not pervasive S failure — but whether the separating feature is hand geometry (rather than exposure/background or mere pose repetition) cannot be established from the measurements this tool can produce. Confirm or refute from `E_transition_runs.png` and `F_pose_comparison.png` (rows 2-4): if the dominant-run tiles share a visible hand feature that correct-S tiles lack, answer YES; if they do not, answer NO.

