# Head-only fine-tuning on cached embeddings — efficientnet_b0_targeted

Experiment: train ONLY the final `Linear(1280, 29)` on the cached 1280-D pooled embeddings of the frozen EfficientNet-B0 backbone; the backbone is bit-identical before/after (`c30a904f3898bf7b…`). No new loss, no re-weighting, no oversampling, no backbone unfreezing.

**VERDICT: NO MEANINGFUL CHANGE**

- S->E 138 -> 134 (-4); material = False (threshold: 15% of baseline).
- Accuracy 97.75% -> 97.76% (preserved = True, tol 0.001).
- Macro-F1 0.9775 -> 0.9776 (preserved = True, tol 0.001).
- E recall 1.000 -> 1.000 (not regressed = True, tol 0.02).
- Watch-pair regressions: none.

## Setup

- Checkpoint: `checkpoints/efficientnet_b0_targeted.pt` (parent; untouched).
- Manifest: `data/split_manifest.json` (`454ca31b1d6a2b8a…`), leakage-safe frame-range split — dev val, NOT the reported metric.
- Device: mps. Loss: `CrossEntropyLoss` (label smoothing 0.1, class weights from the checkpoint), AdamW wd 0.01, cosine warmup (1 epoch), mirror of src/train.py.
- Cached train embeddings: `targeted_train_embeddings.pt` (69600 x 1280). Augmentation: one deterministic realization of the existing stochastic train transform (seed 20260809, backgrounds None); not claimed identical to image-space training.
- Cached val embeddings: `targeted_val_embeddings.pt` (17400 x 1280), deterministic `preprocess_bgr` contract.
- Cached-logits verification: max |logit diff| 2.21e-06 over 12 sampled frames.

## Sanity gate

- Original head on cached val embeddings reproduced the baseline **exactly**: acc 97.75%, macro-F1 0.9775, S->E 138, confusion matrix `True`.

## LR search

| LR | best val macro-F1 | best epoch | epochs run |
|----|-------------------|------------|------------|
| 1e-04 | 0.9776 | 1 | 6 |
| 3e-04 | 0.9770 | 1 | 6 |
| 1e-03 | 0.9764 | 1 | 6 |

Selected: **lr=1e-04**, epoch 1, val macro-F1 0.9776. Trainable parameters: 37149 (Linear only). Seed 20260809.

## Before / after (dev val, 17,400 = 600/class)

| metric | before | after | delta |
|--------|--------|-------|-------|
| S->E | 138 | 134 | -4 |
| E->S | 0 | 0 | +0 |
| T->L | 45 | 42 | -3 |
| K->V | 34 | 33 | -1 |
| V->K | 13 | 14 | +1 |
| M->N | 32 | 34 | +2 |
| N->M | 24 | 24 | +0 |
| Y->I | 30 | 31 | +1 |
| X->K | 17 | 17 | +0 |
| X->I | 10 | 10 | +0 |
| X->R | 9 | 13 | +4 |
| total errors | 391 | 390 | -1 |
| accuracy | 97.75% | 97.76% | +0.01% |
| macro-F1 | 0.9775 | 0.9776 | +0.0001 |

## S/E boundary (fixed diagnostic groups, baseline-head assignment)

| group | n | margin mean | margin med | dist mean | dist med | on S side |
|-------|---|-------------|------------|-----------|----------|-----------|
| before · S->E errors | 138 | -1.208 | -1.047 | -0.3399 | -0.2946 | 0 |
| before · correct S | 462 | +2.880 | +2.919 | +0.8101 | +0.8211 | 462 |
| before · true E | 600 | -5.317 | -5.321 | -1.4954 | -1.4964 | 0 |
| after · S->E errors | 134 | -1.151 | -0.974 | -0.3257 | -0.2756 | 0 |
| after · correct S | 466 | +2.901 | +2.955 | +0.8206 | +0.8359 | 466 |
| after · true E | 600 | -5.275 | -5.269 | -1.4924 | -1.4906 | 0 |

- `w_S - w_E` norm: before 3.556, after 3.535.

## S / E per class

| class | metric | before | after |
|-------|--------|--------|-------|
| S | accuracy | 0.770 | 0.777 |
| S | precision | 0.991 | 0.991 |
| S | recall | 0.770 | 0.777 |
| S | f1 | 0.867 | 0.871 |
| E | accuracy | 1.000 | 1.000 |
| E | precision | 0.797 | 0.802 |
| E | recall | 1.000 | 1.000 |
| E | f1 | 0.887 | 0.890 |

## Artifacts

- Head checkpoint: `checkpoints/head_only/candidate_head_best.pt`
- Deployable checkpoint: `checkpoints/efficientnet_b0_head_ft.pt` (file sha256 `b3d47fa4d3a46381…`; verdict embedded in metadata)
- Confusion matrix PNG: `docs/figures/confusion_matrix_head_only_cached.png`
- Confusion matrix CSV: `docs/figures/confusion_matrix_head_only_cached.csv`
- Full JSON: `docs/reports/head_only_cached_embeddings.json`

## Interpretation (conservative)

- The verdict is deterministic per CLEAN IMPROVEMENT: S->E down >= 15% AND acc/macro-F1 within 0.001 of baseline AND E recall >= baseline-0.02 AND no watch pair falls > max(3, 20%) below baseline. TRADEOFF: S improved but errors moved elsewhere. REGRESSION: acc or macro-F1 down >= 0.005. Watch pairs for the verdict: T->L, K->V, V->K, M->N, N->M, Y->I (per task spec).
- This experiment optimizes the head on ONE fixed augmented realization of the train split; it is not a claim about the full image-space training procedure.
- Boundary numbers are descriptive (margin/distance = S-vs-E two-class quantities, not the full 29-class decision).

