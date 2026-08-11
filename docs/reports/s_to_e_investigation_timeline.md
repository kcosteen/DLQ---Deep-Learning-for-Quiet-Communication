# S→E Investigation Timeline — targeted EfficientNet-B0

This document is the investigation history for the **S → E** confusion of
`checkpoints/efficientnet_b0_targeted.pt`: what was observed first, which
hypotheses were tested, which were weakened or ruled out, what quantitative
evidence moved the investigation, why the current evidence points to the final
classifier head rather than preprocessing / exposure / representation collapse,
and what remains unproven.

It is **documentation only**. No new experiments were run for this document, and
no fix is proposed. Every number below is traced to an existing generated
artifact (filenames are listed in the References section); where a number came
from memory in an earlier draft, it has been re-checked against the artifact.

---

## 1. Current baseline

| Property | Value | Source |
| --- | --- | --- |
| Checkpoint | `checkpoints/efficientnet_b0_targeted.pt` | — |
| Architecture | EfficientNet-B0 (arch `efficientnet`) | checkpoint `arch` field |
| Validation split | 17,400 samples, 600 / class (leakage-safe frame-range split, same signer — **not** the reported metric) | `dev_val_targeted_current.json` |
| Overall accuracy | **0.9775** | `dev_val_targeted_current.json` |
| Macro-F1 | **0.9775** | `dev_val_targeted_current.json` |
| Total errors | **391** | `dev_val_targeted_current.json` (17400 − trace) |

Per-class headline numbers:

| Class | F1 | Recall | Source |
| --- | --- | --- | --- |
| S | **0.8668** | **0.770** (462/600) | `dev_val_targeted_current.json` |
| E | **0.8869** | **1.000** (600/600) | `dev_val_targeted_current.json` |

Confusion-matrix cells that define the problem:

| Cell | Count |
| --- | --- |
| S → E | **138** |
| S → S | **462** |
| E → S | **0** (exact matrix value; effectively absent) |

S → E alone accounts for **138 / 391 ≈ 35.3%** of all model errors — the single
largest directional confusion (the next largest is T→L at 45,
`confusion_ranking_targeted_current.md`).

---

## 2. Initial suspicion: preprocessing / crop / exposure

The investigation started from the project's known weak spots:

- The MediaPipe hand-crop path (`HandCropper`) had known reliability issues
  elsewhere in the project.
- Closed-fist classes (S is a fist) had raised concern about crop reliability.
- Exposure was already on the table (the #12 exposure hypothesis; the
  `efficientnet_lift` arch existed as a candidate fix).

The **exact input-pipeline trace** (`docs/reports/model_input_pipeline_ablate.md`,
runtime-verified with `scripts/trace_input_pipeline.py`) showed what the
targeted checkpoint actually receives:

```
raw 200×200 BGR JPEG
→ resize_square → 224×224 (letterbox, aspect preserved, INTER_AREA)
→ BGR → RGB
→ ImageNet normalization (0.485, 0.456, 0.406 / 0.229, 0.224, 0.225)
→ EfficientNet-B0
```

Important conclusions from that trace:

- **HandCropper is NOT in the targeted checkpoint's train/val path.**
- **MediaPipe is NOT in the path.**
- **BBOX_MARGIN / the center-square no-hand crop fallback are NOT involved.**
- `src/crop.py` is involved **only through its deterministic pure-function half**
  (`resize_square`, `normalize_chw`, `preprocess_bgr`, and the frozen contract
  constants) — the hand-detection half is simply never executed for this model.

This **weakened** the hypothesis that S→E was caused by the MediaPipe hand-crop
path. Note the pipeline contract is shared (`preprocess_bgr` is the frozen
single contract used by train, val, and the webcam app); the runtime forward
hook used the targeted checkpoint as a same-arch stand-in because the ablate
`.pt` was not present locally, and the *input tensor* is checkpoint-independent.

---

## 3. Visual/statistical S→E diagnosis

`docs/reports/s_to_e_error_analysis_targeted.json` and
`docs/reports/s_to_e_visual_notes.md`:

| Quantity | Value |
| --- | --- |
| S val samples | **600** |
| S → E errors | **138** |
| Correct S | **462** |
| S recall | **0.770** |

Probability characteristics of the 138 errors:

- mean P(E) ≈ **0.569**, median ≈ **0.549**
- max P(E) ≈ **0.833**
- **0** errors with P(E) ≥ 0.9 (only 4 with P(E) ≥ 0.8, 30 with P(E) ≥ 0.7)
- S is the **second prediction on all 138** errors; the third-place classes are
  X/N/A/V — the S/E/X/N fist family

Temporal clustering (`s_to_e_error_analysis_targeted.json` →
`frame_clustering`):

- **27** contiguous runs
- longest run 41 frames (29.7% of errors)
- top three *dominant* runs (visually grouped segments, `s_to_e_transition_visual_notes.md`)
  contain **101 / 138 errors ≈ 73.2%**:
  - **S2575–S2619** (44 errors; correct blip S2583)
  - **S2801–S2814** (13 errors; correct blip S2811)
  - **S2831–S2877** (44 errors; correct blips S2872–S2874)
- long clean stretch between the first two dominant segments: **S2620–S2776**
  (157 frames)

> Note on two "top-3" figures: the raw contiguous-run clustering (gap = 1 frame)
> reports 63.0% for its top-3 runs (`error_analysis` JSON); the 73.2% figure
> merges adjacent runs into the three visually dominant segments (Run 1 =
> S2575–2582 + S2584–2619, etc.) exactly as `s_to_e_transition_visual_notes.md`
> defines them. Both figures come from the same 138 errors; they differ only in
> how runs are grouped.

**Conservative interpretation:** the 138 failures are not 138 independent poses.
A large fraction are repeated / near-duplicate frames from a few temporal pose
segments.

---

## 4. Exposure hypothesis weakened

Measured image statistics (`s_to_e_visual_notes.md`, pixels measured on the
same 200×200 → 224×224 inputs the model sees):

- Error frames are **slightly brighter on average**: median grey 124.7 vs 115.4
  (+9.3), mean 141.1 vs 133.2 (+8.0) over all 600 S frames.
- But the dominant error runs have **very different mean brightness** from each
  other (`s_to_e_transition_visual_notes.md`): Run 1 ≈ **124**, Run 2 ≈ **163**,
  Run 3 ≈ **155** (whole-set mean 133; all-errors mean 141).
- At the correct → error boundaries, grey / dark-pixel changes are
  **approximately zero to very small**:

| Boundary | last correct-S grey | first error grey | change |
| --- | --- | --- | --- |
| S2574 → S2575 | 121 | 121 | ≈ 0 |
| S2800 → S2801 | 164 | 164 | ≈ 0 |
| S2830 → S2831 | 156 | 156 | ≈ 0 |

**Key observation:** prediction can flip from S to E **in a single frame** with
essentially unchanged global brightness/background statistics. Exposure may
contribute, but the current evidence does not support it as the main driver.

**Prior ablation evidence (supporting, historical — a different checkpoint):**
the exposure/lift study (`docs/RESULTS_class_fixes.md`, "The lift was never
helping", 2026-08-06) was a **negative result**: the no-lift ablation
(`efficientnet_b0_ablate.pt`, arch `efficientnet`, dev-val 0.9783) reached
**S = 0.9617 without the lift**, versus 0.8200 *with* it — the lift's apparent S
gain belonged to the training regime, and `efficientnet_lift` must **not** be
migrated into `src/crop.py`. This is historical context from the *ablate*
checkpoint, not the *targeted* checkpoint this timeline is about; the two must
not be conflated.

---

## 5. Human visual inspection: geometry / pose suspicion

Generated galleries (human-review artifacts, `docs/figures/error_gallery_targeted/S_to_E/`):

- Transition-run galleries: `E_transition_runs.png` +
  `E_transition_run{1,2,3}_human.png` + combined `E_transition_runs_human.png`
  (one chronological filmstrip per dominant run, boundary → run → recovery)
- Pose-comparison galleries: `F_pose_comparison.png` +
  `F_pose_comparison_human.png` (high-confidence correct S vs dominant-run S→E
  vs boundary-margin S→E vs clean correct S)
- Highest-confidence / boundary / control S: `A_highest_pE.png`,
  `B_boundary_margin.png`, `C_random_errors.png`, `D_correct_S_control.png`

Observed qualitative pattern, stated conservatively (the authoring tool could
not render the images; the notes explicitly label hand-geometry dimensions
*requires visual inspection*):

- The dominant S→E runs appear associated with **small changes in fist
  orientation / articulation** across near-duplicate frames.
- Correct and incorrect predictions can **alternate within nearly identical
  temporal sequences** (e.g. correct blips S2583, S2811, S2872–S2874 inside the
  dominant error ranges).
- **Scale and background alone do not explain the flips** (background median
  ~251 in both groups; error vs correct sharpness 1276 vs 1204 — no separation).
- **Thumb / finger geometry remains a plausible visual factor**, but the
  transition notes' decision gate records the repeatable-geometry question as
  **UNCLEAR on measured evidence alone** — confirm/refute is left to human
  review of the galleries. No specific anatomical feature is claimed here.

---

## 6. Embedding-neighbor diagnosis

**This was the turning point.** `scripts/analyze_s_to_e_embeddings.py`,
`docs/reports/s_to_e_embedding_neighbors_targeted.json`.

Penultimate feature used:

- the pooled EfficientNet-B0 **1280-D** vector (the timm backbone output
  `model[0]`, which sits immediately before `Dropout(0.3) + Linear(1280, 29)`)
- identity of the layer verified numerically (`head(backbone(x)) == model(x)`)
- **L2-normalized** before any cosine similarity

For **all 138 S→E errors**:

- **138 / 138 are closer to a correctly classified S than to an actual E** in
  embedding space
- mean nearest-correct-S cosine ≈ **0.900**
- mean nearest-E cosine ≈ **0.606**

Correct-S control (462 samples):

- mean nearest-S ≈ **0.975**
- mean nearest-E ≈ **0.365**

**Careful conclusion:** the S→E failures are **not showing obvious S/E
representation collapse** in the penultimate EfficientNet feature space. The
backbone representation *appears substantially S-like under this diagnostic*
even though the final prediction is E. (Not "the backbone is proven correct" —
only that the nearest-neighbor geometry of these errors is S-like.)

---

## 7. Final linear-head diagnosis

`scripts/analyze_s_to_e_head.py`, `docs/reports/s_to_e_head_analysis_targeted.json`.

Exact classifier:

```
EfficientNet pooled 1280-D
→ Dropout(p=0.3)
→ Linear(1280, 29)
```

Under `model.eval()` dropout is the identity, so `logit_c = w_c^T x + b_c` on
the pooled embedding `x`. The head was extracted and verified against the
checkpoint; every S prediction/probability recomputed from the cached
embeddings + head weights matched the stored inference (0/600 pred mismatches,
max |ΔP| 1.4e-6).

Head parameter statistics:

| Parameter | Value |
| --- | --- |
| `\|\|w_S\|\|₂` | **2.779** (rank 8/29) |
| `\|\|w_E\|\|₂` | **2.850** (rank 12/29) |
| median class weight norm | ≈ **2.858** |
| `b_S` | **+0.118** (rank 28/29) |
| `b_E` | **−0.008** (rank 13/29) |
| `b_S − b_E` | **+0.126** |
| `cos(w_S, w_E)` | **0.202** |
| `\|\|w_S − w_E\|\|₂` | **3.556** |

The S and E weight norms are **not obviously anomalous** relative to the other
29 classes (both sit near the median). The bias does not obviously favor E;
if anything `b_S − b_E` is positive (i.e. mildly S-favoring).

---

## 8. S/E logit-margin results

Define

```
margin_SE = logit_S − logit_E
```

(margin > 0 → S wins over E; margin < 0 → E wins over S. This is the two-class
S/E quantity, not the complete 29-class decision.)

| Group | mean | median | std | min | max |
| --- | --- | --- | --- | --- | --- |
| S→E errors (138) | **−1.208** | **−1.047** | 0.839 | −3.350 | −0.025 |
| Correct S (462) | **+2.880** | **+2.919** | 1.668 | +0.028 | +5.802 |
| True E (600) | **−5.317** | **−5.321** | 0.238 | −6.102 | −4.349 |

Error margin quantiles (5 / 25 / 50 / 75 / 95%): **−2.574 / −1.874 / −1.047 /
−0.436 / −0.147**.

Reading:

- **many errors are only moderately across** the S/E boundary (25% sit within
  ~0.44 of it; 95th percentile is −0.147)
- **29 / 138 errors are within |distance| < 0.1** in the normalized geometric
  boundary distance (Section 9) — genuine near-boundary cases
- some errors are deeper across the E side (min −3.350)
- compared with true E (mean −5.317), even the deepest errors are ~2 logits
  shallower — the errors are not E-like *by margin depth*.

---

## 9. Signed geometric boundary distance

Define

```
distance_SE = margin_SE / ||w_S − w_E||₂
```

(the signed distance to the S/E hyperplane `(w_S − w_E)^T x + (b_S − b_E) = 0`,
in units of the weight-difference direction; raw logits depend on weight scale,
this does not).

| Group | mean | median | min | max |
| --- | --- | --- | --- | --- |
| S→E errors (138) | **−0.340** | **−0.295** | −0.942 | −0.007 |
| Correct S (462) | **+0.810** | +0.821 | +0.008 | +1.632 |
| True E (600) | **−1.495** | −1.496 | −1.716 | −1.223 |

Error distance quantiles: q5 −0.724, q95 −0.041.

**Conservative interpretation:** the errors occupy the E side of the S/E
hyperplane, but are **substantially closer to the boundary than true E examples
are** (median −0.295 vs −1.496 — roughly one fifth of the distance true E sits
at).

---

## 10. Strongest combined evidence

Why the investigation ended at a head / decision-boundary diagnosis — the
combined result (`s_to_e_embedding_neighbors_targeted.json`,
`s_to_e_head_analysis_targeted.json`):

- **138 / 138** S→E errors are closer to a correct S than to an actual E in the
  penultimate embedding space, **and**
- **138 / 138** nevertheless have **margin_SE < 0** at the final linear head,
  **and**
- **93 / 138** satisfy the stricter condition
  `nearest-S − nearest-E similarity > 0.25` while still lying on the E side
  (0 of 138 exceed 0.5),
- with **nearest-S − nearest-E vs margin_SE: Pearson r ≈ +0.869** (n = 138),
  **Spearman r ≈ +0.844**.

What this means:

- The embedding and the head are **not random or contradictory**: the more
  S-like an embedding is, the more S-favoring its margin tends to be
  (positive correlation).
- However, the learned **linear S/E hyperplane crosses a region containing
  legitimate S-like embeddings** — embeddings that are closer to correct S than
  to any real E still fall on the E side of the boundary.

Cautious wording: **"Current evidence is most consistent with final-head /
decision-boundary behavior."** It does not warrant "the linear head is broken."

---

## 11. Representative frames

Six representative S→E errors (from `s_to_e_head_analysis_targeted.json`;
nearest-S/E are cosine similarities to the nearest correct-S / actual-E
embedding, self-excluded):

| Frame | nearest S | nearest E | sim diff | margin_SE | distance_SE | P(S) / P(E) |
| --- | --- | --- | --- | --- | --- | --- |
| S2575 | 0.955 | 0.528 | +0.427 | −0.994 | −0.280 | 0.20 / 0.55 |
| S2601 | 0.773 | 0.694 | +0.078 | −3.270 | −0.920 | 0.03 / 0.83 |
| S2805 | 0.933 | 0.734 | +0.199 | −1.812 | −0.510 | 0.10 / 0.63 |
| S2864 | 0.859 | 0.687 | +0.172 | −2.526 | −0.710 | 0.06 / 0.78 |
| S2832 | 0.964 | 0.490 | +0.475 | −0.025 | −0.007 | 0.34 / 0.35 |
| S2876 | 0.976 | 0.480 | +0.496 | −0.087 | −0.024 | 0.36 / 0.39 |

Highlighted cases:

- **S2832 / S2876** — extremely S-like (nearest-S 0.964 / 0.976) yet sitting
  within a hair of the S/E boundary (distance −0.007 / −0.024): the cleanest
  "legitimate S-like embedding pushed just over the line" examples.
- **S2601** — S-like by nearest-neighbor (0.773 > 0.694) but **much deeper
  across** the E side (margin −3.270, the deepest of the six): shows that
  "S-like in embedding space" and "deep on the E side of the boundary" can
  coexist.

---

## 12. Hypotheses considered and current status

| Hypothesis | Current status | Evidence |
| --- | --- | --- |
| MediaPipe / hand-crop failure | **Unlikely for this checkpoint** | Not in the input path (`model_input_pipeline_ablate.md`); crop half of `crop.py` never executed |
| Exposure / brightness | **Weak / secondary** | Boundary flips happen with ~0 brightness change (Run boundaries, §4); errors only ~+8 grey mean; lift ablation negative (`RESULTS_class_fixes.md`) |
| Background / blur | **Weak evidence** | No strong measured separation (background medians ~251 both; Laplacian 1276 vs 1204) |
| Insufficient S pose diversity | **Still plausible** | Errors cluster in a few repeated temporal pose segments (101/138 in 3 dominant runs) |
| EfficientNet representation collapse S→E | **Weakened strongly** | 138/138 embeddings closer to correct S (0.900 vs 0.606); control 0.975 vs 0.365 |
| Final linear S/E boundary behavior | **Strongest current explanation** | S-like embeddings fall on the E side; 138/138 margin < 0; errors much closer to the boundary than true E (−0.295 vs −1.496) |
| Head bias specifically favoring E | **Not supported** | `b_S − b_E` is positive (+0.126); S/E weight norms near median (2.779 / 2.850 vs median 2.858) |

---

## 13. Why we did NOT immediately change the loss or retrain

We deliberately did **not**:

- add class weights immediately,
- switch to focal loss immediately,
- retrain the head immediately,
- add landmarks immediately,
- modify `crop.py` immediately.

Each of these would alter multiple parts of the system **before the failure
location was understood**. The investigation therefore followed a deliberate
sequence:

```
confusion ranking
→ exact input trace
→ visual / temporal diagnosis
→ embedding-space diagnosis
→ classifier-head margin diagnosis
```

This sequence progressively narrowed the failure from broad
preprocessing/data/exposure hypotheses toward a specific
decision-boundary behavior at the final linear head.

---

## 14. Current conclusion

Conservative wording:

> Current evidence is most consistent with a final classifier-head / S-vs-E
> decision-boundary issue affecting a subset of legitimate S-like embeddings,
> especially a few repeated pose subtypes. The EfficientNet penultimate
> representation remains substantially S-like for all observed S→E failures. No
> single head parameter is obviously pathological, and the analysis does not
> prove the head is defective; the boundary may reflect the trade-offs learned
> by the current 29-class objective.

**No fix was implemented as part of this investigation.**

---

## 15. Suggested next experiment — documentation only

Not implemented; not a recommendation proven by this analysis. The next
controlled question would be:

> If the S/E boundary is moved or the head is fine-tuned to recover S-like
> boundary cases, what accuracy/regression cost appears on E and the other 27
> classes?

Possible future controlled experiments (each marked as a *future experiment*,
not a proven path):

- head-only fine-tune
- current-loss fine-tune (full model)
- focal-loss comparison
- targeted hard-example sampling

Any of these must be evaluated on the **webcam test set** (the reported
benchmark) and must not treat dev-val as the reported number.

---

## References / artifacts

- Baseline report: `docs/reports/dev_val_targeted_current.json`
- Confusion ranking (targeted): `docs/reports/confusion_ranking_targeted_current.md`
- Confusion ranking (ablation, historical): `docs/reports/confusion_ranking_ablate.md`
- Model input pipeline: `docs/reports/model_input_pipeline_ablate.md`
- S→E error analysis: `docs/reports/s_to_e_error_analysis_targeted.json`
- S→E visual notes: `docs/reports/s_to_e_visual_notes.md`
- S→E transition-run notes: `docs/reports/s_to_e_transition_visual_notes.md`
- Embedding-neighbor report: `docs/reports/s_to_e_embedding_neighbors_targeted.json`
- Head-analysis report: `docs/reports/s_to_e_head_analysis_targeted.json`
- Head-analysis notes: `docs/reports/s_to_e_head_notes.md`
- Exposure/lift ablation history: `docs/RESULTS_class_fixes.md` (sections
  "Result — lift run" and "Ablation — 2026-08-06")
- Galleries: `docs/figures/error_gallery_targeted/S_to_E/` (`A_highest_pE.png`,
  `B_boundary_margin.png`, `C_random_errors.png`, `D_correct_S_control.png`,
  `E_transition_runs.png`, `E_transition_run{1,2,3}_human.png`,
  `E_transition_runs_human.png`, `F_pose_comparison.png`,
  `F_pose_comparison_human.png`, `G_embedding_neighbors_human/`,
  `H_head_boundary/`)
- Analysis scripts: `scripts/analyze_s_to_e.py`, `scripts/inspect_s_to_e_runs.py`,
  `scripts/build_human_galleries.py`, `scripts/analyze_s_to_e_embeddings.py`,
  `scripts/analyze_s_to_e_head.py`, `scripts/trace_input_pipeline.py`
