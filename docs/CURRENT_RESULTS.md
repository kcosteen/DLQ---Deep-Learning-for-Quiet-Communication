# Current results (2026-08-15)

Where the project stands *right now*: what we trained, where the checkpoints are,
and what the numbers honestly say. Everything here is the **leakage-safe dev val
(temporal split, same signer)** — the *reported* metric is our quarantined webcam
test set, which has partial results on B/C (see README).

## The short version

- We attacked the one class still below the 85% per-class bar — **S** — from the
  classifier side, on cached embeddings.
- **Head-only fine-tune:** no meaningful change (S→E 138→134).
- **Partial fine-tune** (last MBConv stage `blocks[6]` + head @ 1e-4):
  **S→E 138→98, S recall 0.770→0.837** — just shy of the 85% bar — **but** it
  regresses V→K (13→54) and Y→I (+7), so overall **accuracy/F1 are flat**.
- Verdict (per the deterministic criteria in
  `docs/reports/partial_finetune_targeted.md`): **TRADEOFF, not a clean win.**
- Deployable checkpoint: **`checkpoints/efficientnet_b0_partial_ft.pt`** — that is
  the `.pt` we currently ship.

## Checkpoints on disk

| checkpoint | role | dev-val acc | macro-F1 | S recall | S→E |
|---|---|---|---|---|---|
| `checkpoints/efficientnet_b0_targeted.pt` | parent / baseline (#12 targeted) | 0.9775 | 0.9775 | 0.770 | 138 |
| `checkpoints/efficientnet_b0_head_ft.pt` | head-only FT experiment | 0.9776 | 0.9776 | 0.777 | 134 |
| `checkpoints/efficientnet_b0_partial_ft.pt` | **deployable (current best)** | 0.9771 | 0.9772 | **0.837** | **98** |
| `checkpoints/partial_ft/partial_ft_best.pt` | training-phase candidate (intermediate) | — | — | — | — |

The parent `efficientnet_b0_targeted.pt` is untouched; the partial-FT model is the
parent's backbone with only `blocks[6]` conv/SE + the head `Linear(1280,29)`
updated (BN affine + running stats frozen). Training details, LR search, and
reproducibility caveats: `docs/reports/partial_finetune_training.md`.

## The partial-FT tradeoff (dev val, MPS raw-image eval)

| metric | before (targeted) | after (partial-FT) | delta |
|---|---|---|---|
| S→E | 138 | 98 | **−40** |
| S recall | 0.770 | 0.837 | **+0.067** |
| V→K | 13 | 54 | **+41** (regression) |
| Y→I | 30 | 37 | +7 (regression) |
| K→V | 34 | 22 | −12 |
| T→L | 45 | 49 | +4 |
| X→R | 9 | 17 | +8 |
| total errors | 391 | 398 | +7 |
| accuracy | 97.75% | 97.71% | −0.04% |
| macro-F1 | 0.9775 | 0.9772 | −0.0004 |

S and E per-class (before → after): S acc/precision/recall/F1
0.770/0.991/0.770/0.867 → **0.837/0.992/0.837/0.908**; E recall stays **1.000**
(E prec/F1 0.797/0.887 → 0.846/0.917).

Why it's a tradeoff: S improved (and E→S is still 0), but the S errors moved into
the V↔K family — V→K jumped 13→54, which the watch-pair rule flags as a material
regression. 28/29 classes ≥ 85%; **S is the only one below, at 0.837** (MPS eval —
MPS-vs-CPU float flips mean the CPU number can differ by a handful of S/E
boundary frames, so call it "near 85%, not over it").

Full tables, boundary-margin analysis, and representative frames:
`docs/reports/partial_finetune_targeted.md`.

## Why the classifier-side attack exists

Prior work established S is *hard* because of the dataset, not just the model: the
frames are backlit, the hand sits at grey 9–89 against a 240–255 background, and
S vs E is the one pair where the only cue is interior shading — which the exposure
crushes. That pointed at an exposure-normalising preprocessing fix (a Person-B
change to the FROZEN contract — not done). Meanwhile the classifier-side attack
above is the follow-up that *doesn't* touch the contract. See
`docs/reports/s_to_e_visual_notes.md` and `docs/reports/s_to_e_head_notes.md`.

## Reproduce

```bash
# Deployable model eval on the dev val (leakage-safe split, NOT the reported metric)
python -m src.evaluate --checkpoint checkpoints/efficientnet_b0_partial_ft.pt \
    --split data/raw/asl_alphabet_train/asl_alphabet_train \
    --manifest data/split_manifest.json \
    --report-json docs/reports/partial_finetune_targeted.json \
    --figures docs/figures
```

Rebuild/train/analyze (needs the embedding cache):
`scripts/build_embedding_cache.py`, `scripts/train_head_only_cached.py`,
`scripts/train_partial_finetune.py`, `scripts/analyze_partial_finetune.py`.

## Known gaps

- **Webcam-set scores exist for B and C only** (46 samples, 100% acc on
  `efficientnet_b0_targeted.pt`). A scored 0% (19 samples, all confused as S/E).
  Full 29-class webcam benchmark still pending.
- S sits at 0.837 dev-val — above the baseline 0.770 but **not over the 85% bar**.
- Crop ablation (symmetric 0.62 vs forearm vs calibrated) completed; symmetric
  remains the production default. See README crop ablation table.

## Artifacts index

- `checkpoints/efficientnet_b0_partial_ft.pt` — deployable, sha256 `46d86662b3ac3409…`
- `docs/reports/partial_finetune_targeted.{json,md}` — final verdict + tables
- `docs/reports/partial_finetune_training.{json,md}` — training sweep + best-epoch pick
- `docs/figures/confusion_matrix_partial_ft.{png,csv}` — confusion matrix
- `docs/figures/error_gallery_targeted/S_to_E/I_partial_finetune/rep_change.png`
- `data/partial_ft_embeddings_val.npz` — before/after val embeddings (raw-image pipeline)
