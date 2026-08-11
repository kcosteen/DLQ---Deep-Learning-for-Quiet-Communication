# S->E head analysis notes — efficientnet_b0_targeted (diagnosis only)

All quantities are computed on the cached pooled embeddings (`data/s_e_embeddings_targeted.npz`) recombined with the checkpoint head weights; every S prediction/probability is verified against `data/s_to_e_all_s_records.json` (0 mismatches, max |dp| 1.37e-06).

## Q1. Are the 138 S->E errors close to or far across the S/E boundary?

- margin_SE distribution (logit_S - logit_E): errors mean **-1.208**, median **-1.047** (std 0.839, min -3.350, max -0.025).
- margin_SE quantiles (errors): -2.574 / -1.874 / -1.047 / -0.436 / -0.147 (5/25/50/75/95%).
- As a fraction of the S/E weight direction ||w_S - w_E|| = 3.556: signed distance mean **-0.3399**, median **-0.2946** (std 0.2361, min -0.9421, max -0.0072).
- 71 errors with margin < -1.0; 6 errors with |margin| < 0.1; 29 errors with |distance| < 0.1.
- For context, correct-S margin mean **+2.880** (median +2.919, min +0.028) and true-E margin mean **-5.317** (median -5.321, max -4.349).

## Q2. Do their embeddings remain strongly S-like even when the final margin favors E?

- All 138 errors have nearest-correct-S cosine greater than nearest-actual-E cosine (True); mean similarity difference +0.293 (median +0.325).
- Embedding-similarity-difference vs margin_SE: Pearson r = +0.869 (p=2.05e-43, n=138); Spearman r = +0.844. vs distance_SE Pearson r = +0.869.
- The correlation is descriptive; it does not establish whether the head or the representation causes the flip.

## Q3. Anything numerically unusual about the S or E head parameters?

- ||w_S||₂ = **2.779** (rank 8/29), ||w_E||₂ = **2.850** (rank 12/29); median over all classes 2.858.
- b_S = **+0.118** (rank 28/29), b_E = **-0.008** (rank 13/29); median bias +0.001.
- b_S - b_E = **+0.126**; cos(w_S, w_E) = **+0.202**.
- None of these are inherently bad: a norm or bias outside the median is a description, not a defect.

## Q4. Representation overlap, head/boundary behavior, or ambiguous?

- The errors' embeddings are overwhelmingly S-like (Q2: all 138 have nearest-correct-S closer than nearest-E, mean difference +0.293), yet their S/E margin is negative because of the head's linear decision. This profile points at the final-head/decision-boundary region rather than feature-space S/E overlap.
- The decomposition margin_SE = feature_term + bias_term shows feature_term (errors) mean **-1.335**, median **-1.174**, against a constant bias_term +0.126 — the S-vs-E weight direction dominates the sign, with the bias as the constant offset.
- Representative check: of the six query errors, 6/6 are closer to correct S yet lie on the E side of the hyperplane.
- This remains **descriptive and to be read conservatively**: the nearest-neighbor result plus the negative margins are consistent with head/boundary behavior, but they do not *prove* the head is defective. **No fix is proposed** in this document.

## Representative samples

| frame | nearest-S | nearest-E | sim diff | logit_S | logit_E | margin_SE | distance_SE | P(S) | P(E) |
|-------|-----------|-----------|----------|---------|---------|------------|--------------|------|------|
| S2575 | 0.955 | 0.528 | +0.427 | +2.62 | +3.61 | -0.994 | -0.2795 | 0.202 | 0.545 |
| S2601 | 0.773 | 0.694 | +0.078 | +1.87 | +5.14 | -3.270 | -0.9196 | 0.032 | 0.833 |
| S2805 | 0.933 | 0.734 | +0.199 | +2.59 | +4.40 | -1.812 | -0.5096 | 0.103 | 0.632 |
| S2864 | 0.859 | 0.687 | +0.172 | +2.24 | +4.77 | -2.526 | -0.7104 | 0.063 | 0.783 |
| S2832 | 0.964 | 0.490 | +0.475 | +3.24 | +3.27 | -0.025 | -0.0072 | 0.337 | 0.346 |
| S2876 | 0.976 | 0.480 | +0.496 | +3.38 | +3.47 | -0.087 | -0.0244 | 0.359 | 0.392 |
