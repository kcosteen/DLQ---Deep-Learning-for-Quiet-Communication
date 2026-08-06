# Fixing the worst-confused classes (#12)

**Question.** The #6 transfer run cleared the overall dev-val target (0.9545 > 95%)
and **missed the per-class bar**: S, V and X sat below 85%, and between them held
**684 of 791 errors (86.5%)**. Can those three be lifted above 85% without pushing
a fourth class below it?

> ⚠️ **Status: "before" is real, "after" is not filled in yet.** The before column
> is measured — it comes from the #6 run and is pinned as
> `docs/reports/dev_val_baseline_6.json`. This environment has **no GPU and no
> dataset**, so no targeted run was performed here; every `TODO` cell is unfilled.
> **Do not quote any after-number until a run fills it in.** The commands below are
> exact and reproducible on Kaggle/Colab.

## Before — what actually failed

From [`RESULTS.md`](RESULTS.md) / `docs/reports/dev_val_baseline_6.json`. Every
class has exactly 600 val frames, so error counts compare directly.

| Class | Accuracy | Errors | Where they go | Shape of the failure |
|---|---|---|---|---|
| **X** | 0.507 | 296 | A 120, L 78, R 65, I 16, K 11 | diffuse — no single culprit |
| **V** | 0.630 | 222 | **K 219**, W 3 | clean collision |
| **S** | 0.723 | 166 | **E 160**, X 4, A 2 | clean collision |
| M | 0.932 | 41 | N 41 | above the bar |
| N | 0.972 | 17 | M 17 | above the bar |
| *21 others* | 1.000 | 0 | — | solved |

Overall 0.9545 · macro-F1 0.9518 · worst class 0.507.

## Diagnosis

**The default augmentation is the leading suspect, not the missing ingredient.**
`RESULTS.md` originally attributed the plateau to having "no augmentation yet";
that was wrong, and correcting it inverts the plan. The heavy Albumentations
pipeline (#9) has been active in `build_transforms(train=True)` since 2026-07-28,
five days *before* the #6 run. So the model failed these three classes **while**
seeing ±20° rotation, ±8° shear, perspective warp, 0.85–1.15 zoom and 4–6 coarse
dropout holes of up to 15% of the frame each.

That matters because of *which* classes failed. Every one of them is a handshape
whose identity is a small geometric detail:

- **V vs K** — where the thumb sits (219 errors, one wrong answer nearly every time).
- **S vs E** — how far the thumb folds across a closed fist (160 errors).
- **X vs A/L/R** — how far the index finger hooks (296 errors, spread).

Those are exactly the details that rotation, shear and a dropout hole are free to
smear or erase while the label stays put. Train long enough on that and the model
learns the distinguishing feature is noise. Meanwhile the 21 classes at 1.000 are
ones a human tells apart from the silhouette alone, which survives any amount of
geometric jitter.

Two things this diagnosis does **not** explain, and the honest limits of it:

- **X is diffuse, and diffuse is different.** V→K and S→E are single wrong
  answers; X leaks into four classes at once, which reads less like "confused with
  a lookalike" and more like "never formed a stable X at all". That could be
  framing/data rather than augmentation, so `--dump-errors` exists to look at the
  frames before spending another run guessing. #11's MediaPipe crop cache, which
  landed after this run, is the other candidate for X.
- **A/E never confused at all**, and M/N is mild (58 errors, both above the bar) —
  so the project plan's `M/N/S/T, A/E, K/V` watch list was measured wrong. Nothing
  here is aimed by that list; everything is aimed by the confusion matrix.

## What was built

Every lever is opt-in and derives its target from a *measured* report, never from
a hardcoded class list.

| Lever | Flag | Why |
|---|---|---|
| Gentle geometry for named classes | `--geometry-safe-classes auto` | Softens only rotation/shear/perspective/zoom/dropout, for the failing classes **and the classes they are confused with** — the S/E boundary is learned from both. Photometric strength is untouched. |
| Error-driven class weights | `--rebalance-from <report.json>` | `weight = 1 + alpha*(1 - per_class_accuracy)`. Inverse-frequency weighting is useless here (600 val / ~2,400 train frames per class, every class); the imbalance is difficulty, not count. |
| Applied to loss **or** sampler | `--rebalance loss` / `sampler` | Same weights, two mechanisms; exactly one applies, because both would square the emphasis. |
| Focused fine-tune | `--resume <ckpt>` | Skips the frozen-head stage: a targeted attempt costs one stage instead of two. |
| Machine-readable per-class report | `evaluate --report-json` | Carries the full confusion matrix, so a run's result is also the next run's targeting input. |
| Before/after table | `evaluate --baseline <report>` | The #12 deliverable. Flags **fixed**, **still below**, and **new regressions** separately. |
| Look at the frames | `evaluate --dump-errors <dir>` | Copies misclassified source frames into `TRUE_to_PRED/` dirs. For X, where the matrix has said all it can. |

`auto` on the #6 baseline resolves to **A, E, K, L, R, S, V, X** — the three
failing classes plus the four they leak into (I is excluded at 5% of X's errors;
the partner threshold is 20% of that class's errors).

## Reproduce

Run on Kaggle (T4). The `#6` checkpoint must be available at
`checkpoints/efficientnet_b0.pt` — attach it as a Kaggle dataset input and copy it
in, since `--resume` needs the weights, not just the report.

```bash
# 0. Resolve the read-only Kaggle mount into a usable class root (also renames
#    Kaggle's 'del' to the 'delete' that src.data.CLASSES expects). It prints
#    what it linked and where; ROOT is its --out, default /kaggle/working/asl_train.
python scripts/kaggle_setup.py
ROOT=/kaggle/working/asl_train

# 1. Build the split manifest ONCE and use it for both runs.
#    The #6 checkpoint predates checkpoints/<name>.split.json, so there is no
#    manifest from that run to reuse. frame_range_split is deterministic given
#    (root, train_frac), so rebuilding it here reproduces the same split #6
#    trained on — the #6 eval re-derived it the same way and matched at 17,400
#    val images. Passing it explicitly is what makes that checkable
#    (content_sha256) instead of assumed.
python -m src.data --root "$ROOT" --train-frac 0.8 \
    --out data/split_manifest.json --counts

# 2. Targeted fine-tune: aimed augmentation + error-driven class weights.
#    --finetune-epochs 6 because the #6 run early-stopped at 8 of 15 and never
#    reached its cosine anneal (see RESULTS.md); sizing the schedule to what the
#    model actually uses lets it complete.
python -m src.train \
    --root "$ROOT" \
    --manifest data/split_manifest.json \
    --resume checkpoints/efficientnet_b0.pt \
    --rebalance-from docs/reports/dev_val_baseline_6.json \
    --rebalance loss --rebalance-alpha 1.0 \
    --geometry-safe-classes auto \
    --finetune-epochs 6 --wandb-mode online \
    --out checkpoints/efficientnet_b0_targeted.pt
#    (--out defaults to the _targeted name anyway; overwriting --resume is refused,
#     because the baseline checkpoint is the 'before' half of this comparison)

# 3. Score it on the SAME split and diff against the baseline.
python -m src.evaluate \
    --checkpoint checkpoints/efficientnet_b0_targeted.pt \
    --split "$ROOT" \
    --manifest checkpoints/efficientnet_b0_targeted.split.json \
    --report-json docs/reports/dev_val_targeted.json \
    --baseline docs/reports/dev_val_baseline_6.json \
    --label "targeted (#12)" --figures /tmp/figs_targeted
#    The training run writes <out>.split.json, which is a copy of what step 1 built;
#    scoring against it means the val frames are provably the ones held out.
#    --figures /tmp/... keeps docs/figures/confusion_matrix.{png,csv} pointing at the
#    baseline until this run is accepted.

# 4. If X is still below the bar, stop guessing and look at its frames.
python -m src.evaluate \
    --checkpoint checkpoints/efficientnet_b0_targeted.pt \
    --split "$ROOT" \
    --manifest checkpoints/efficientnet_b0_targeted.split.json \
    --dump-errors /tmp/x_errors --dump-errors-max 20 --figures ""
#    then eyeball /tmp/x_errors/X_to_A, X_to_L, X_to_R
```

A `--manifest` path that does not exist is a hard error rather than a silent
re-derive, so a typo here cannot quietly score a different set of frames than the
flag claims. Preview what the gentle policy actually does:

```bash
python -m src.augment --image <a V frame> --policy geometry_safe --n 8 \
    --out /tmp/v_gentle.png
```

## Results

**Dev-val, leakage-safe frame-range split — DEV ONLY, not the reported number.**

| Class | Before | After | Δ | ≥ 85%? |
|---|---|---|---|---|
| X | 0.507 | `TODO` | `TODO` | `TODO` |
| V | 0.630 | `TODO` | `TODO` | `TODO` |
| S | 0.723 | `TODO` | `TODO` | `TODO` |
| M | 0.932 | `TODO` | `TODO` | `TODO` |
| N | 0.972 | `TODO` | `TODO` | `TODO` |
| Y | 0.957 | `TODO` | `TODO` | `TODO` |
| worst class | 0.507 | `TODO` | `TODO` | — |
| overall | 0.9545 | `TODO` | `TODO` | — |
| macro-F1 | 0.9518 | `TODO` | `TODO` | — |

Run step 2 above and paste its before/after table here; it prints exactly these
rows, plus any class that moved by ≥ 0.005.

## How to read the delta (when filled in)

- **All three ≥ 85%, nothing new below** → #12 is done. Record the table, promote
  the targeted checkpoint, and re-point `docs/figures/confusion_matrix.*` at it.
- **V and S fixed, X still below** → the augmentation hypothesis held for the clean
  collisions and not for the diffuse one. Do **not** just raise `--rebalance-alpha`;
  run step 3 and look at X's frames first. If they are poorly framed, the fix is
  #11's crop cache (`--root data/cache_crops`), not more weighting.
- **A new class below 85%** → the comparison did its job. That is over-emphasis:
  drop `--rebalance-alpha` (try 0.5), or remove the regressed class's partner from
  the geometry-safe set. Fixing V by breaking K is not progress.
- **Overall accuracy up but the worst class flat** → ignore the overall number. It
  is 26 solved classes diluting three broken ones, which is exactly how the #6 run
  passed a 95% target while getting X right barely half the time.
- **Nothing moves at all** → the resumed fine-tune may be too far into a minimum
  that has these weaknesses baked in. The fallback is a fresh two-stage run with
  the same aimed flags (drop `--resume`), which costs both stages.

## Notes / honesty

- **Every number in this doc is dev-val**, i.e. the same signer in the same room,
  tail 20% of each class's frames. The reported metric is the webcam test set
  (#15/#16), which does not exist yet. Nothing here may be quoted as "the accuracy
  of the model".
- **Deriving weights from dev-val makes dev-val a tuning target**, so the after-number
  is optimistically biased in a way the before-number is not. That price is
  acceptable only because dev-val is explicitly not the reported metric; the webcam
  set stays quarantined and never feeds `--rebalance-from`. That is enforced, not
  just agreed: a webcam report passed to `--rebalance-from` is a hard error, since
  class weights and the geometry-safe list are exactly the "augmentation setting"
  the quarantine rule forbids the webcam set from choosing.
- **Softening geometry for 8 classes is a real trade against webcam robustness.**
  Less geometric augmentation means less invariance to how a stranger holds their
  hand, and dev-val *cannot see that* — it is the same signer at the same angle. So
  a dev-val win here could be partly borrowed from #16. Mitigations: only geometry
  is softened (colour/noise/blur and `--backgrounds` stay at full strength for
  those classes), and the geometry-safe set must be re-examined against the webcam
  set at #16 rather than treated as settled.
- **The per-class bar is the DoD, not the mean.** `evaluate` fails loudly with
  `FAIL: classes below 85%`, and `compare_reports` separates "fixed" from "new
  regression" so an improved average cannot hide a broken class.
- The default policy's numbers are frozen by a test
  (`tests/test_aimed_augmentation.py::test_default_policy_still_matches_the_run_it_is_compared_against`)
  — softening the default would silently invalidate every comparison against #6.
