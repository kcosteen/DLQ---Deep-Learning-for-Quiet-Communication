# Confusion ranking — targeted checkpoint (current baseline)

This report is based on the **targeted checkpoint** (`checkpoints/efficientnet_b0_targeted.pt`, arch `efficientnet`) and its freshly evaluated dev-val confusion matrix. Directional rows are kept separate (`M -> N` and `N -> M` are distinct).

- Total validation samples: **17400**
- Total misclassifications: **391**
- Overall accuracy: **0.9775**
- Macro-F1: **0.9775**

Source: `docs/figures/confusion_matrix_targeted_current.csv` and report `docs/reports/dev_val_targeted_current.json` (dev val = leakage-safe frame-range split, same signer — NOT the reported metric).

## Top 20 directional confusions

| Rank | True | Pred | Errors |
| ---: | :--- | :--- | -----: |
| 1 | S | E | 138 |
| 2 | T | L | 45 |
| 3 | K | V | 34 |
| 4 | M | N | 32 |
| 5 | Y | I | 30 |
| 6 | N | M | 24 |
| 7 | X | K | 17 |
| 8 | V | K | 13 |
| 9 | I | E | 11 |
| 10 | X | I | 10 |
| 11 | X | R | 9 |
| 12 | R | X | 7 |
| 13 | F | W | 6 |
| 14 | X | S | 4 |
| 15 | F | E | 3 |
| 16 | J | I | 2 |
| 17 | U | X | 2 |
| 18 | A | M | 1 |
| 19 | F | V | 1 |
| 20 | R | E | 1 |

## Top 10 symmetric pairs

| Rank | Pair | Total errors |
| ---: | :--- | -----------: |
| 1 | E <-> S | 138 |
| 2 | M <-> N | 56 |
| 3 | K <-> V | 47 |
| 4 | L <-> T | 45 |
| 5 | I <-> Y | 30 |
| 6 | K <-> X | 17 |
| 7 | R <-> X | 16 |
| 8 | E <-> I | 11 |
| 9 | I <-> X | 10 |
| 10 | F <-> W | 6 |

## Five lowest-F1 classes

| Rank | Class | F1 |
| ---: | :--- | ---: |
| 1 | S | 0.8668 |
| 2 | E | 0.8869 |
| 3 | K | 0.9465 |
| 4 | M | 0.9522 |
| 5 | N | 0.9536 |

## Five highest-F1 classes

| Rank | Class | F1 |
| ---: | :--- | ---: |
| 1 | B | 1.0000 |
| 2 | C | 1.0000 |
| 3 | D | 1.0000 |
| 4 | G | 1.0000 |
| 5 | H | 1.0000 |
