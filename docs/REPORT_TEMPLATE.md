# ASL Fingerspelling Recognition — Final Report

*(Skeleton — fill in as the project progresses. Delete the italic prompts.)*

**Team:** Person A, Person B, Person C · **Date:** ____ · **Repo:** ____

---

## 1. Abstract
*One paragraph: the problem (read the ASL hand alphabet from a webcam and spell
words + speech), the approach (EfficientNet-B0 transfer learning + MediaPipe crop),
and the headline honest result — the **webcam-test-set** accuracy, not the split
number.*

## 2. Problem & scope
- What fingerspelling is; what the 29 classes are (A–Z, `space`, `delete`,
  `nothing`).
- Explicit scope limit: alphabet recognition, **not** sign-language translation.

## 3. Data
- Dataset: ASL Alphabet (87k images, 29 classes, GPL-2.0).
- **The single-signer trap** and why a random split is dishonest.
- Our leakage-safe frame-range split (how; the split-leakage test).
- The webcam test set: who recorded, how many samples, conditions covered.

*Figure: class-balance bar chart. Figure: example frames + hand crops.*

## 4. Method
- Preprocessing / crop contract (`src/crop.py`) and why train == serve.
- Augmentation pipeline (list the ops; background replacement).
- Models: logistic-regression sanity floor → compact CNN baseline →
  EfficientNet-B0 two-stage transfer learning.
- Training config: loss (label smoothing), AdamW, cosine+warmup, AMP, early stop.

## 5. Experiments & results

| Model | Leakage-safe val acc | Webcam test acc (reported) | Macro-F1 | Min per-class |
|---|---|---|---|---|
| LogReg (pixels) sanity floor | | | | |
| Compact CNN (scratch) | | | | |
| EfficientNet-B0 (no aug) | | | | |
| EfficientNet-B0 (+ aug + bg) | | | | |

- **Report the webcam number as the result.** Note the val number separately and
  label it dev-only.
- Targets: val > 95%, webcam ≥ 90%, no class < 85%.

*Figure: confusion matrix (`docs/figures/confusion_matrix.png`).*
*Figure: training curves from W&B.*

## 6. Error analysis
- The known-confusable groups: **M/N/S/T, A/E, K/V** — did they show up? by how much?
- The train→webcam gap: how big, and what closed it (augmentation? crop parity?
  fine-tuning?).
- Failure modes: lighting, background, skin tone, hand size, left vs right hand.

## 7. Live demo
- Pipeline recap, measured FPS, debounce/confidence-gate settings.
- Link to the recorded demo video.

## 8. Ethics & limitations
- Alphabet recognizer, not translation.
- Deaf/signing-user testing: what was done, what feedback surfaced.
- Honest accuracy limits and who the tool does / doesn't work for.

## 9. Future work
- Word-level upgrade: Google ASL Signs (landmark sequences → Transformer). Scope,
  cost, and expected payoff.

## 10. Contributions
*Who did what (A / B / C).*

## 11. References
*Same list as README / PROJECT_PLAN.*
