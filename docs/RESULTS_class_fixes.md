# Fixing the worst-confused classes (#12)

**Question.** The #6 transfer run cleared the overall dev-val target (0.9545 > 95%)
and **missed the per-class bar**: S, V and X sat below 85%, and between them held
**684 of 791 errors (86.5%)**. Can those three be lifted above 85% without pushing
a fourth class below it?

> **Status: measured — and #12 is NOT closed.** Run 2026-08-06 on Kaggle T4.
> **V and X are fixed** (0.630 → 0.978 and 0.507 → 0.933); total errors nearly
> halved, 791 → 391. **S is still below the bar at 0.770**, so the Definition of
> Done — *no class below 85%* — is not met. Every number here is dev-val: the same
> signer, the same room. The reported metric is still the webcam set (#15/#16),
> which does not exist yet.

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

## Results — run 2026-08-06, Kaggle T4

Six fine-tune epochs resumed from the #6 checkpoint, `--rebalance loss
--rebalance-alpha 1.0`, `--geometry-safe-classes auto` (resolved to A, E, K, L, R,
S, V, X). Scored on the same 17,400 val frames.
Report: `docs/reports/dev_val_targeted.json`.

**Dev-val, leakage-safe frame-range split — DEV ONLY, not the reported number.**

| Class | Before | After | Δ | ≥ 85%? |
|---|---|---|---|---|
| **X** | 0.507 | **0.933** | **+0.427** | ✅ fixed |
| **V** | 0.630 | **0.978** | **+0.348** | ✅ fixed |
| **S** | 0.723 | **0.770** | +0.047 | ❌ **still below** |
| T | 0.988 | 0.925 | −0.063 | ✅ (regressed) |
| K | 1.000 | 0.943 | −0.057 | ✅ (regressed) |
| I | 1.000 | 0.982 | −0.018 | ✅ |
| M | 0.932 | 0.947 | +0.015 | ✅ |
| N | 0.972 | 0.960 | −0.012 | ✅ |
| R | 0.998 | 0.987 | −0.012 | ✅ |
| F | 0.977 | 0.983 | +0.007 | ✅ |
| Y | 0.957 | 0.950 | −0.007 | ✅ |
| worst class | 0.507 | **0.770** (S) | +0.263 | — |
| total errors | 791 | **391** | −400 | — |
| overall | 0.9545 | 0.9775 | +0.0230 | — |
| macro-F1 | 0.9518 | 0.9775 | +0.0257 | — |

**The augmentation hypothesis was right about V and X and wrong about S.** V and X
were the two classes the diagnosis predicted would respond, and they moved +0.348
and +0.427 — V's 222 errors became 13, X's 296 became 40. Nothing about the data
changed between the runs; only how hard the augmentation was allowed to deform it.
That is about as direct a confirmation as this setup can produce: those two classes
were not hard, they were being taught that their distinguishing feature was noise.

### S is a different problem, and now the only one

| | Before | After |
|---|---|---|
| S accuracy | 0.723 | 0.770 |
| S → E | 160 | **138** |
| E → S | 0 | **0** |
| E accuracy | 1.000 | 1.000 |

S got the *same* treatment that fixed V and X — both S and E were in the
geometry-safe set — and moved +0.047. So S→E is **not** an augmentation artefact.
Two things say what it is instead:

- **It is one-directional.** 138 S frames are called E; not one E frame is called
  S, and E sits at 1.000 in both runs. S is not symmetrically ambiguous with E — S
  is being *absorbed* into E. The model has a confident E and a weak S.
- **S is now 138 of 391 total errors — 35%**, on its own. Every other class in the
  matrix is at 0.925 or better.

S and E are both closed fists; the difference is whether the thumb lies across the
fingers (S) or tucks under them (E). In a 200×200 full-frame image the hand is a
fraction of the pixels and the thumb is a handful of them. That points at
**resolution on the discriminating region**, not at augmentation or weighting.

### The regressions are real and worth reading

Nothing fell below the bar, but two classes moved enough to explain:

- **K: 1.000 → 0.943** (K→V, 34). This is the V/K boundary shifting, which is the
  cost of up-weighting V 1.31×. It is a good trade and not close: V's 222 errors
  became 13 while K gained 34. But it is the mechanism to watch if `--rebalance-alpha`
  is ever raised.
- **T: 0.988 → 0.925** (T→L, 45, up from 7) — and this one is a **flaw in the `auto`
  rule**, not a trade. Every one of T's errors went to L, both before and after. L
  was softened because it is one of X's error partners; **T was not, because `auto`
  only considers classes that are themselves below the bar and the classes *those*
  leak into.** T was passing at 0.988, so nothing pulled it in — and it ended up on
  the heavy policy while the class it confuses with was on the gentle one. Asymmetric
  augmentation across a shared boundary is exactly what the S/E comment in
  `auto_geometry_safe_classes` warns about, applied in the direction the code does
  not check.

  **Fix:** after choosing the set, also pull in any class whose errors go
  predominantly *into* the set. On this matrix that would have caught T (100% of its
  errors → L). Until that lands, pass T explicitly.

## Why S fails: the frames are backlit silhouettes

`--dump-errors` was run on the targeted checkpoint and the S→E frames inspected
directly. The finding is not what the pre-run guesses in this doc predicted, so
those have been replaced with what was measured.

**The hand is a silhouette.** In every S→E frame the subject is backlit against a
blown-out window: measured over the 20 dumped frames, the hand occupies ~38% of the
frame at grey levels **9–89**, while ~32% of the frame is background pinned at
**240–255**. The thumb — the only thing separating S from E — is a few pixels of
shading inside a near-black blob.

**The information is present, not lost.** Only **0.2%** of hand pixels are clipped
at ≤10, and interior standard deviation is 23. Applying a shadow lift (gamma 0.45)
raises it to ~30 and makes the thumb lying across the fingers **plainly visible to
the eye**. So this is a *tonal range* problem, not a resolution or labelling one.

**This explains the whole results pattern**, not just S:

| Handshape group | Outline distinguishes them? | Result |
|---|---|---|
| B, C, L, O, W, … (21 classes) | yes | 1.000 — solved from the silhouette alone |
| V vs K | yes — K's thumb breaks the outline | fixed by gentler geometry (+0.348) |
| X vs A/L/R | yes — hook changes the contour | fixed by gentler geometry (+0.427) |
| **S vs E** | **no — both are compact fists** | **stuck at 0.770** |

Softening the augmentation worked exactly where the cue survives in the outline. S/E
is the one pair where it does not, so no augmentation policy can rescue it.

### The MediaPipe crop cannot fix this — it does not fire

The obvious next lever looked like #11's crop cache: spend all 224² on the hand.
Measured on the dumped frames with `src.crop.HandCropper`, it does not work, because
**MediaPipe finds no hand to crop**:

| | Hand detected |
|---|---|
| S→E frames | **0 / 20** |
| V→K, X→K, X→I, X→R, Y→I | **0 / 79** |
| M→N | 1 / 20 · T→L | 7 / 20 |
| K→V, F→*, U→X, W→V, X→S | 20/20, 100% |
| **all 208 dumped frames** | **71 / 208 = 34%** |

Lifting the shadows *before* detection helps some classes a lot and S not at all:

| Pre-processing | Detection over the 110 failing frames | V | T | S |
|---|---|---|---|---|
| raw (current) | 7% | 0/13 | 7/20 | 0/20 |
| gamma 0.45 | 26% | 8/13 | 19/20 | 1/20 |
| gamma 0.30 | 31% | 11/13 | 18/20 | **2/20** |
| CLAHE | 6% (worse) | 0/13 | 7/20 | 0/20 |

So `cache_crops` would silently fall back to a **centre crop** on exactly the frames
that need help — `CropResult.found_hand=False` returns the centre square. Caching
crops for S would change nothing.

> ⚠️ **Those 208 frames are a biased sample** — every one is a *misclassified*
> frame, so they are plausibly the darkest and hardest in the set. The 34% figure is
> **not** a dataset-wide detection rate. It was measured properly instead; see below.

### Measured properly: 74% dataset-wide, and the bias was real

`scripts/check_hand_detection.py --root <class root> --per-class 40` over a
deterministic stride sample of 1,160 frames (40 per class, seed 0):

| | Raw | With `--gamma 0.45` |
|---|---|---|
| **All hand-bearing classes** | **854/1160 = 74%** | **917/1160 = 79%** |
| M | 45% | **62%** |
| N | 42% | 48% |
| S | 88% | 90% |
| V | 88% | 90% |
| X | 75% | 72% |
| best (F) | 98% | 98% |
| `nothing` (empty scene) | 0% | 0% — correct, and scored inverted |

Three conclusions, and the first one corrects this document:

1. **74%, not 34%.** The error-frame sample was badly biased, so #11's crop cache
   and the demo bridge are in substantially better shape than the first measurement
   implied. Recorded here rather than quietly dropped, because the 34% was used to
   argue against the crop cache.
2. **S's detection was never the problem.** 88% dataset-wide against **0/20** on its
   val error frames. A 0-for-20 at an 88% base rate is not chance — the frames the
   classifier gets wrong are the same frames the detector cannot see. Something is
   different about those specific frames, which is what the val-tail question below
   is about.
3. **M and N are the genuine detection failures** (45%/42%) — and M↔N is also a
   confusion pair in the results table (32 + 24 errors). A fist with the fingers
   folded over the thumb gives the landmarker no structure to grip and the
   classifier no feature to separate. Same cause, two symptoms.

**Contrast helps detection, but partially, and a fixed gamma is the wrong shape.**
Gamma 0.45 lifts the total 5 points and M by 17, but leaves N under 50% and makes
X, W and `delete` slightly *worse* — exactly what a constant would do to frames
that were already correctly exposed. It supports the adaptive lift recommended
below and argues against a hardcoded one. Note this measures **detection** only;
whether recovered tonal range helps the classifier separate S from E is a separate
question that only a training run answers.

### Ruled out: the val tail is darker, and it explains nothing

Before spending a GPU run on the shadow lift, one competing explanation had to go.
`frame_range_split` sends the first 80% of each class to train and the last 20% to
val *in capture order*. If the recording session dimmed over time, val would be
systematically darker than anything the model trained on, and S's failure would be a
distribution shift in the split rather than an exposure problem — a different bug,
with a different fix, outranking #12.

Measured 2026-08-06 with `scripts/check_val_brightness.py`, 40 frames per side:

| class | train med | val med | delta | val acc |
|---|---|---|---|---|
| **S** | 134.8 | 117.5 | **−17.4** | **0.770** ✗ |
| E | 141.7 | 90.7 | −50.9 | 1.000 |
| M | 138.8 | 93.4 | −45.4 | 0.947 |
| N | 133.9 | 92.6 | −41.3 | 0.960 |
| B | 143.1 | 115.8 | −27.3 | 1.000 |
| L | 140.7 | 122.7 | −18.0 | 1.000 |
| X | 123.9 | 111.8 | −12.1 | 0.933 |
| V | 126.0 | 125.0 | −1.0 | 0.978 |

**The drift is real: every class darkens, none brightens.** The capture-order
artifact exists and is worth knowing about. **It does not explain S.** E darkens the
most of any class measured (−50.9) and scores a perfect 1.000; S darkens −17.4,
*less* than the −28.0 mean of the seven classes that pass. Pearson r between delta
and per-class accuracy is **−0.219** over 8 classes — no relationship, and the sign
points away from the hypothesis. Brightness shift is not what separates the class
that fails from the classes that do not.

So the shadow lift below stands as the next experiment, and the darkening is
recorded here as a property of the split rather than filed as a blocker.

> **Methodological note, because the first run of this got it wrong.** Grouping the
> "failing" classes as S, E, M, N — the four this doc discusses — produced a
> confident false positive: a −24.2 excess darkening and a "distribution shift"
> verdict. But E, M and N all *pass* (1.000, 0.947, 0.960); only S is under the bar.
> Three passing classes with large deltas manufactured a signal about a class whose
> delta is unremarkable. `check_val_brightness.py` now reads the failing set from the
> evaluation report's `below_target` field instead of a hardcoded list, and
> `tests/test_val_brightness.py` pins that behaviour. The general lesson: when a
> comparison groups classes, the grouping is the experiment — get it from the data,
> not from which names appear together in prose.

## What to do next — closing S

#12 stays open until S ≥ 0.85. Revised in light of the above:

1. **Put an adaptive shadow lift in the preprocessing contract.** This is the fix the
   evidence points at, and it needs no MediaPipe: the CNN can use recovered tonal
   range whether or not a hand was detected. It belongs in `src/crop.py`'s
   `preprocess_bgr` — *not* in augmentation — because train and serve must apply it
   identically, which is exactly what the contract exists to guarantee. A fixed gamma
   is wrong (it would wash out a well-exposed webcam frame); make it self-calibrating,
   e.g. the gamma that maps the frame's median luminance to a target.

   Consequences to accept up front: this is Person C's locked file (#4) and needs
   their sign-off; `tests/test_preprocessing_parity.py` must stay green; and it
   **invalidates every existing checkpoint**, so the comparison run is a fresh
   two-stage train, not a `--resume`. Worth it — an exposure-normalising contract is
   also one of the better things that could happen to the *webcam* number (#16),
   since a stranger's lighting is the variable it neutralises.

2. **Raise the weight on S only, in the same run.** The failure is one-directional —
   138 S frames are called E, zero E frames are called S — so moving the S/E boundary
   may buy real accuracy cheaply. `--rebalance-alpha 2.0` roughly doubles S's
   emphasis. Weighting cannot create information the pixels lack, so treat it as a
   supplement to (1), and read E's row in the before/after table: E is at 1.000 and
   has the most to lose.

3. **If S still fails, say so and close #12 as measured.** If a contract-level
   contrast fix plus targeted weighting cannot separate two handshapes that are
   genuinely near-identical under this signer's backlighting, then "S cannot reach 85%
   on this dataset" is a legitimate, reportable finding — and a strong argument for
   the landmark-MLP route (#20), which reads geometry instead of pixels. What would
   not be legitimate is quietly dropping the per-class bar.

What **not** to do: raise `--rebalance-alpha` globally (it pushes V harder, and K is
already down 0.057), or cache crops expecting S to improve (MediaPipe does not fire
on those frames).

### Implemented: the lift lives in the model, for now

Built as `ExposureLift` in `src/model.py`, reachable as the arch
**`efficientnet_lift`** — *not* in `src/crop.py` as recommended above. The reasoning,
and the debt this takes on:

**Why the model.** `src/crop.py` is Person C's locked contract (#4) and a change
there needs their sign-off, which blocks the experiment on a review. Putting the
lift inside the model gets the number now, and it turns out to give a *stronger*
parity guarantee rather than a weaker one: `train.py` records `arch` in the
checkpoint, `evaluate.py` rebuilds from it, and `load_resume_state` refuses a
checkpoint whose arch disagrees. So the weights and the preprocessing they expect
cannot be separated — `tests/test_exposure_lift.py` pins that the two arches have
incompatible `state_dict`s. A contract-side lift, by contrast, relies on both sides
remembering to call it. It also leaves #13 completely untouched: Person C's bridge
loads the new checkpoint and the exposure handling comes with it.

**The debt.** Exposure handling is now invisible in `crop.py`, which is where a
future reader will look for it. **If this closes #12, migrate it into
`preprocess_bgr` with Person C's sign-off** — at that point the migration is an
easy sell backed by a number, instead of a request to approve a locked-file change
on spec.

**The anchor is the 25th percentile, not the median** — and this is the part that
would have silently failed. The frame median on this dataset is already ~120–140/255
(≈0.5, see the brightness table above) because the blown-out background drags it up.
Solving for a gamma that maps the *median* to a mid target is therefore very nearly a
no-op and leaves the hand exactly as dark as it was. `test_median_anchored_lift_would_have_no_opped`
pins that trap. The lift is also clamped to γ ≤ 1.0 so it can only brighten: a
well-exposed webcam frame gets γ = 1.0, an exact no-op. That is the property a fixed
γ = 0.45 lacks — it measurably hurt X, W and delete, which were already exposed
correctly.

Export is verified, not assumed: the percentile is taken with `torch.sort` rather
than `torch.quantile` precisely because `quantile` has no ONNX lowering. TorchScript
tracing and ONNX export are both covered, including a test that the traced module
still *adapts* rather than baking one gamma in as a constant.

```bash
# Fresh two-stage train -- NO --resume. The lift changes what the network sees, so
# resuming would train under one preprocessing and evaluate under another; the arch
# guard refuses it anyway, which is the guard working as intended.
python -m src.train \
    --root "$ROOT" \
    --manifest data/split_manifest.json \
    --model efficientnet_lift \
    --rebalance-from docs/reports/dev_val_targeted.json \
    --rebalance loss --rebalance-alpha 2.0 \
    --geometry-safe-classes auto \
    --baseline-epochs 20 --finetune-epochs 15 --wandb-mode online \
    --out checkpoints/efficientnet_b0_lift.pt

python -m src.evaluate \
    --checkpoint checkpoints/efficientnet_b0_lift.pt \
    --split "$ROOT" \
    --manifest checkpoints/efficientnet_b0_lift.split.json \
    --report-json docs/reports/dev_val_lift.json \
    --baseline docs/reports/dev_val_targeted.json \
    --label "adaptive lift + S weight (#12)" --figures /tmp/figs_lift
```

Read **E's row first** in the diff, not S's. E is at 1.000 and the S→E boundary is
being pushed at from both sides — by the recovered shading and by `--rebalance-alpha
2.0`. A gain on S bought by breaking E is not progress, and the per-class bar is
what #12 is measured against.

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
