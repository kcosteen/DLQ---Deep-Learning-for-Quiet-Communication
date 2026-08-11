# Partial fine-tuning of efficientnet_b0_targeted — training phase

Experiment: image-space training of ONLY the final MBConv stage `blocks[6]` (717,232 params, a single InvertedResidual) + the existing `Linear(1280, 29)` head. All other backbone tensors and every BatchNorm statistic stay bit-identical to the parent.

**Selected candidate: backbone_lr=1e-05, head_lr=1e-04, epoch 1, val macro-F1 0.9772**

## Provenance

- Parent: `checkpoints/efficientnet_b0_targeted.pt` (file sha256 `3827f8797d0f822f…`, backbone sd sha256 `5c3921cb4410c631…`, recorded val_acc 0.9775).
- Rebalance: loss; geometry-safe classes ['A', 'E', 'K', 'L', 'R', 'S', 'V', 'X']; backgrounds None; resumed_from /kaggle/input/models/duketruong/dlq-effb0-baseline/pytorch/default/1/efficientnet_b0.pt.
- Manifest: `data/split_manifest.json` (`454ca31b1d6a2b8a…`), leakage-safe frame-range split: 69600 train / 17400 val (600/class).

## Model structure (inspected)

- container nn.Sequential(backbone, head); backbone EfficientNet; 7 MBConv stages.
- blocks[0]: 1448 params
- blocks[1]: 16714 params
- blocks[2]: 46640 params
- blocks[3]: 242930 params
- blocks[4]: 543148 params
- blocks[5]: 2026348 params
- blocks[6]: 717232 params  <- TRAINABLE
- conv_stem 864 | bn1 64 | conv_head 409600 | bn2 2560 | global_pool SelectAdaptivePool2d (all frozen).
- head: Dropout(p=0.3) + Linear(1280, 29) (trainable).

## Freeze policy

- Total 4044697 params; trainable 749133 (18.5214%); frozen 3295564.
- BN affine frozen: True; running mean/var frozen (all BatchNorm modules eval); pretrained stats preserved.

## Sanity gate

- Raw-image val eval of the parent reproduced the baseline: acc 0.9775 (expected 0.977529), macro-F1 0.9775 (expected 0.977508), total errors 391 (expected 391), S->E 138 (expected 138). PASS.

## Device

- torch 2.13.0, MPS available True; selected mps.
- Per-epoch selection uses mps val macro-F1 (full 17,400-frame val). Final reported numbers come from the raw-image CPU pipeline in the analysis phase.

## Sweep

| backbone_lr | head_lr | best macro-F1 | best epoch | epochs run |
|-------------|---------|---------------|------------|------------|
| 3e-06 | 3e-05 | 0.9764 | 2 | 4 |
| 1e-05 | 1e-04 | 0.9772 | 1 | 3 |
| 3e-05 | 3e-04 | 0.9747 | 2 | 4 |

Per-epoch records (train loss / val acc / val macro-F1 / S->E):

| backbone_lr | head_lr | epoch | train loss | val acc | val macro-F1 | S->E |
|-------------|---------|-------|------------|---------|--------------|------|
| 3e-06 | 3e-05 | 1 | 0.6508 | 0.9763 | 0.9762 | 132 |
| 3e-06 | 3e-05 | 2 | 0.6495 | 0.9764 | 0.9764 | 116 |
| 3e-06 | 3e-05 | 3 | 0.6492 | 0.9757 | 0.9757 | 113 |
| 3e-06 | 3e-05 | 4 | 0.6499 | 0.9755 | 0.9755 | 114 |
| 1e-05 | 1e-04 | 1 | 0.6506 | 0.9771 | 0.9772 | 98 |
| 1e-05 | 1e-04 | 2 | 0.6493 | 0.9754 | 0.9754 | 91 |
| 1e-05 | 1e-04 | 3 | 0.6488 | 0.9745 | 0.9745 | 88 |
| 3e-05 | 3e-04 | 1 | 0.6502 | 0.9718 | 0.9717 | 106 |
| 3e-05 | 3e-04 | 2 | 0.6484 | 0.9748 | 0.9747 | 110 |
| 3e-05 | 3e-04 | 3 | 0.6475 | 0.9728 | 0.9727 | 114 |
| 3e-05 | 3e-04 | 4 | 0.6473 | 0.9731 | 0.9730 | 115 |

## Parameter-change verification

- Candidate vs parent: 9 tensors changed (all in the trainable set `blocks[6]` conv/SE + `1.1` head); 351 tensors bit-identical (stem, blocks[0..5], conv_head, bn2, all BatchNorm tensors).
- BatchNorm tensors unchanged: True.

## Artifacts (training phase)

- Candidate state dict: `checkpoints/partial_ft/partial_ft_best.pt` (file sha256 `ec79c1fb19470105…`)
- Training report: `docs/reports/partial_finetune_training.json`
- This markdown: `docs/reports/partial_finetune_training.md`

## Next phase

- Run `scripts/analyze_partial_finetune.py` for the final raw-image eval, representation-change diagnostics, three-way comparison (targeted / head-only / partial-FT), verdict, and the deployable checkpoint.

