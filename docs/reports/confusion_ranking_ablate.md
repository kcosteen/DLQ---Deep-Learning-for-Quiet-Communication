# Confusion ranking — ablation results

This report is based on the **ablation checkpoint** (`efficientnet_b0_ablate.pt`, no-lift fresh train) and its confusion matrix — **not** the older targeted checkpoint. Directional rows are kept separate (`M -> N` and `N -> M` are distinct).

- Total validation samples: **17400**
- Total misclassifications: **378**
- Overall accuracy (derived): **0.9783**

Source: `docs/figures/confusion_matrix_ablate.csv` and report `docs/reports/dev_val_ablate.json` (for sample count only)

## Top 20 directional confusions

| Rank | True | Pred | Errors |
| ---: | :--- | :--- | -----: |
| 1 | V | K | 160 |
| 2 | X | R | 60 |
| 3 | M | N | 36 |
| 4 | Y | I | 25 |
| 5 | S | E | 23 |
| 6 | X | I | 16 |
| 7 | N | M | 15 |
| 8 | X | K | 14 |
| 9 | T | L | 12 |
| 10 | F | V | 5 |
| 11 | Y | L | 4 |
| 12 | X | L | 3 |
| 13 | D | I | 1 |
| 14 | F | E | 1 |
| 15 | F | W | 1 |
| 16 | J | I | 1 |
| 17 | X | S | 1 |

## Top 10 symmetric pairs

| Rank | Pair | Total errors |
| ---: | :--- | -----------: |
| 1 | K <-> V | 160 |
| 2 | R <-> X | 60 |
| 3 | M <-> N | 51 |
| 4 | I <-> Y | 25 |
| 5 | E <-> S | 23 |
| 6 | I <-> X | 16 |
| 7 | K <-> X | 14 |
| 8 | L <-> T | 12 |
| 9 | F <-> V | 5 |
| 10 | L <-> Y | 4 |
