# MediaPipe vs RGB Experiment Artifacts

## Provenance

- **Generated on:** Kaggle (notebook)
- **Dataset:** `grassknoted/asl-alphabet`
- **Samples per class:** 100 (deterministic, seed 42)
- **Classes:** 29 (A–Z, del, nothing, space)
- **MediaPipe:** Tasks API HandLandmarker (same `hand_landmarker.task` as the project)
- **RGB checkpoint:** `efficientnet_b0_targeted.pt`
- **Pipeline:** raw image → `preprocess_bgr` → EfficientNet backbone → classifier head → top-1 prediction (no hand crop before EfficientNet)

## Files

| File | Description |
|------|-------------|
| `mediapipe_sample_seed42.csv` | Sample manifest: class + file path (2,900 rows) |
| `mediapipe_vs_rgb_sample_records.csv` | Per-sample records: class, path, found_hand, prediction, confidence, correct (2,900 rows) — **primary data source** |
| `mediapipe_vs_rgb_summary.csv` | Pre-computed per-class summary (29 rows) — derived, do not use as source of truth |

## Analysis

Run `python scripts/analyze_mediapipe_vs_rgb.py` to regenerate reports and figures from the per-sample CSV.

Reports: `docs/reports/mediapipe_miss_vs_rgb_accuracy.{md,json}`
Figures: `docs/figures/mediapipe_detection_coverage_by_class.png`, `docs/figures/mediapipe_rgb_accuracy_comparison.png`

## Special-class note

The Kaggle notebook used an unverified manual output-label mapping (`A–Z, del, nothing, space`) that does not match the checkpoint's training order (`A–Z, space, delete, nothing`). Special-class RGB metrics are invalid. Analysis is restricted to A–Z.
