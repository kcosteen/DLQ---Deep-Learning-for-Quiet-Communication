# Partial fine-tuning of efficientnet_b0_targeted — final report

Experiment: image-space training of the FINAL MBConv stage `blocks[6]` + the existing `Linear(1280, 29)` head (differential LRs, BN frozen). Parent backbone bit-identical except `blocks[6]` conv/SE and the head.

**VERDICT: TRADEOFF**

- S->E 138 -> 98 (-40); material = True.
- Accuracy 97.75% -> 97.71% (preserved = True).
- Macro-F1 0.9775 -> 0.9772 (preserved = True).
- E recall 1.000 -> 1.000 (not regressed = True).
- Watch-pair regressions: ['V->K', 'Y->I'].

## Setup

- Parent: `checkpoints/efficientnet_b0_targeted.pt` (file sha256 `3827f8797d0f822f…`, backbone sd `5c3921cb4410c631…`), untouched.
- Candidate (training phase): `checkpoints/partial_ft/partial_ft_best.pt`.
- Manifest `454ca31b1d6a2b8a…` — leakage-safe dev val, NOT the reported metric.
- Training: device mps; differential LRs backbone `1e-05` / head `1e-04`, best epoch 1; CrossEntropy (label smoothing 0.1, checkpoint class weights), AdamW wd 0.01, cosine warmup (1 epoch); BN running stats + affine frozen.
- Final evaluation: raw-image `preprocess_bgr` pipeline on CPU (mps (raw-image preprocess_bgr)), 17,400 = 600/class.

## Sanity gate

- Raw-image eval of the parent reproduced the baseline (acc 97.75%, macro-F1 0.9775, errors 391, S->E 138).

## Before / after (dev val)

| metric | before | after | delta |
|--------|--------|-------|-------|
| S->E | 138 | 98 | -40 |
| E->S | 0 | 0 | +0 |
| T->L | 45 | 49 | +4 |
| K->V | 34 | 22 | -12 |
| V->K | 13 | 54 | +41 |
| M->N | 32 | 32 | +0 |
| N->M | 24 | 24 | +0 |
| Y->I | 30 | 37 | +7 |
| X->K | 17 | 24 | +7 |
| X->I | 10 | 12 | +2 |
| X->R | 9 | 17 | +8 |
| total errors | 391 | 398 | +7 |
| accuracy | 97.75% | 97.71% | -0.04% |
| macro-F1 | 0.9775 | 0.9772 | -0.0004 |

## S / E per class

| class | metric | before | after |
|-------|--------|--------|-------|
| S | accuracy | 0.770 | 0.837 |
| S | precision | 0.991 | 0.992 |
| S | recall | 0.770 | 0.837 |
| S | f1 | 0.867 | 0.908 |
| E | accuracy | 1.000 | 1.000 |
| E | precision | 0.797 | 0.846 |
| E | recall | 1.000 | 1.000 |
| E | f1 | 0.887 | 0.917 |

## Three-way comparison (targeted / head-only / partial-FT)

| metric | targeted | head-only | partial-FT |
|--------|----------|-----------|------------|
| accuracy | 0.9775 | 0.9776 | 0.9771 |
| macro_f1 | 0.9775 | 0.9776 | 0.9772 |
| total_errors | 391 | 390 | 398 |
| S_to_E | 138 | 134 | 98 |
| E_to_S | 0 | 0 | 0 |
| T_to_L | 45 | 42 | 49 |
| K_to_V | 34 | 33 | 22 |
| V_to_K | 13 | 14 | 54 |
| M_to_N | 32 | 34 | 32 |
| N_to_M | 24 | 24 | 24 |
| Y_to_I | 30 | 31 | 37 |
| X_to_K | 17 | 17 | 24 |
| X_to_I | 10 | 10 | 12 |
| X_to_R | 9 | 13 | 17 |

## Representation change (pooled 1280-D, L2-normalized)

Embedding movement `1 - cosine(before, after)` by group (higher = moved more):

| group | n | movement mean | movement med | cosine mean |
|-------|---|---------------|--------------|-------------|
| S->E errors | 138 | 0.0049 | 0.0048 | 0.9951 |
| correct S | 462 | 0.0043 | 0.0045 | 0.9957 |
| true E | 600 | 0.0039 | 0.0038 | 0.9961 |
| all val | 17400 | 0.0037 | 0.0034 | 0.9963 |

- Original 138 S->E errors now correct: 40 (138 total); still wrong: 98.
- New predictions of the original 138: {'E': 98, 'S': 40}.
- Movement of the corrected subset (mean 0.0051) vs still-wrong (0.0048).

## S-vs-E nearest neighbor in the NEW embedding space

- Original 138 errors: nearest-correct-S cosine 1.000 vs nearest-actual-E 0.595; closer to S 138/138, closer to E 0/138.
- Corrected subset of the 138: 40/40 closer to S, 0/40 closer to E.

## S/E boundary (margin_SE = logit_S - logit_E, fixed groups)

| group | n | margin mean | margin med | dist mean | dist med | on S side |
|-------|---|-------------|------------|-----------|----------|-----------|
| before · S->E errors | 138 | -1.208 | -1.047 | -0.3399 | -0.2946 | 0 |
| before · correct S | 462 | +2.880 | +2.919 | +0.8101 | +0.8211 | 462 |
| before · true E | 600 | -5.317 | -5.321 | -1.4954 | -1.4964 | 0 |
| after · S->E errors | 138 | -0.744 | -0.552 | -0.2125 | -0.1575 | 40 |
| after · correct S | 462 | +3.177 | +3.205 | +0.9070 | +0.9152 | 462 |
| after · true E | 600 | -5.299 | -5.292 | -1.5132 | -1.5112 | 0 |

- `w_S - w_E` norm: before 3.556, after 3.502.

## Representative frames

| frame | before pred | after pred | margin before | margin after | movement |
|-------|-------------|------------|---------------|--------------|----------|
| 2575 | nothing | nothing | +0.854 | +0.903 | 0.0040 |
| 2601 | nothing | nothing | +0.788 | +0.866 | 0.0043 |
| 2805 | nothing | nothing | +0.375 | +0.403 | 0.0049 |
| 2864 | nothing | nothing | +0.694 | +0.688 | 0.0047 |
| 2832 | nothing | nothing | +0.447 | +0.445 | 0.0045 |
| 2876 | nothing | nothing | +0.588 | +0.602 | 0.0047 |

## Artifacts

- Deployable checkpoint: `checkpoints/efficientnet_b0_partial_ft.pt` (file sha256 `46d86662b3ac3409…`, verdict embedded).
- Confusion PNG: `docs/figures/confusion_matrix_partial_ft.png`
- Confusion CSV: `docs/figures/confusion_matrix_partial_ft.csv`
- Representation-change figure: `docs/figures/error_gallery_targeted/S_to_E/I_partial_finetune/rep_change.png`
- Embedding cache (before/after, val): `data/partial_ft_embeddings_val.npz`
- Full JSON: `docs/reports/partial_finetune_targeted.json`

## Interpretation (conservative)

- Verdict per deterministic criteria (CLEAN IMPROVEMENT: S->E down >= 15% AND acc/macro-F1 within 0.001 of baseline AND E recall >= baseline-0.02 AND no watch pair falls > max(3, 20%) below baseline. TRADEOFF: S improved but errors moved elsewhere. REGRESSION: acc or macro-F1 down >= 0.005. Watch pairs for the verdict: T->L, K->V, V->K, M->N, N->M, Y->I (per task spec).).
- Margin/nearest-neighbour numbers are descriptive two-class S-vs-E quantities, not the full 29-class decision.
- Training-time epoch selection used the mps val macro-F1; the numbers above are the raw-image mps (raw-image preprocess_bgr) eval. MPS-vs-CPU float differences can flip a handful of S/E top-2 boundary frames (seen in the head-only run); reported eval device: mps (raw-image preprocess_bgr).

