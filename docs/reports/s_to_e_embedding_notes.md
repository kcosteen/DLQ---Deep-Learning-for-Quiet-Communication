# S->E embedding-neighbor notes — efficientnet_b0_targeted (diagnosis only)

> **For the full investigation summary, see
> [`s_to_e_investigation_timeline.md`](s_to_e_investigation_timeline.md).**
> This document is a source report for that synthesis.

## Method

- Embedding: penultimate EfficientNet-B0 pooled feature vector (`model[0]`, 1280-D), the input to `Dropout(0.3) + Linear(1280, 29)`. Location verified against the head structure and `model(x) == head(backbone(x))`.
- Normalization: row-wise L2; similarity is cosine (dot product).
- Pools: 462 correctly classified S, 600 actual E validation samples; queries self-excluded.
- This is descriptive. Nearest-to-E vs nearest-to-correct-S splits a *possible* representation/data-overlap problem from a *possible* head/boundary problem, but neither is established as a cause here.

## Q1: Are S->E errors generally embedded closer to E than correct S?

- **0** of 138 S->E errors have a nearest actual-E closer than their nearest correct S; **138** have nearest correct S closer (0 ties).
- Mean nearest-E similarity: **0.606**; mean nearest-correct-S similarity: **0.900**.
- Distribution of nearest-E minus nearest-S: mean **-0.293**, std 0.125, min -0.496, median -0.325, max -0.013.

## Q2: Are the representative errors visually surrounded by real E?

- The six query galleries (`S2575_neighbors.png`, `S2601_neighbors.png`, `S2805_neighbors.png`, `S2864_neighbors.png`, `S2832_neighbors.png`, `S2876_neighbors.png`) show top-5 nearest correct S (left), the query (center), top-5 nearest actual E (right). Per-query nearest similarities:

| query | nearest S | nearest E | | query | nearest S | nearest E |
|-------|-----------|-----------|-|-------|-----------|-----------|
| S2575 | 0.955 | 0.528 | | S2864 | 0.859 | 0.687 |
| S2601 | 0.773 | 0.694 | | S2832 | 0.964 | 0.490 |
| S2805 | 0.933 | 0.734 | | S2876 | 0.976 | 0.480 |

Whether the neighboring E tiles *look* like the query is for a human reviewer to decide from the galleries; the similarities above are measured, the visual judgment is not.

## Q3: Are some failures embedded close to correct S but misclassified?

- **138** of 138 errors sit closer to correct S than to actual E in embedding space yet still predict E — the profile of a head/decision-boundary problem. **0** sit closer to E — the profile of a representation/data-overlap problem.
- Control (correctly classified S): 0 of 462 have nearest-E closer than nearest-other-S (mean nearest-other-S 0.975, nearest-E 0.365).

## Bottom line (descriptive)

The 138 S->E errors split 0 (nearer to E) vs 138 (nearer to correct S). No fix is proposed in this document; the split is reported for the human reviewer to weigh.

