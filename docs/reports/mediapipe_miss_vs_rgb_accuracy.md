# MediaPipe HandLandmarker Miss vs RGB EfficientNet Accuracy

## Experimental question

> Does failure of MediaPipe HandLandmarker imply that an ASL image is not
> classifiable by the RGB EfficientNet model?

## Setup

| Parameter | Value |
| --------- | ----- |
| Dataset | `grassknoted/asl-alphabet` (Kaggle) |
| Samples per class | 100 (deterministic, seed 42) |
| Classes | A–Z only (26 classes, 2600 images) |
| MediaPipe | Tasks API HandLandmarker (`hand_landmarker.task`) |
| RGB model | `efficientnet_b0_targeted.pt` (EfficientNet-B0, 29-class head) |
| RGB path | raw image → `preprocess_bgr` → backbone → head → top-1 |
| Hand crop before EfficientNet | **No** |

**Note:** The three special classes (`del`, `nothing`, `space`) are excluded because
the Kaggle notebook used an unverified manual output-label mapping that does not match
the checkpoint’s training order (`space`/`delete`/`nothing`). Their RGB metrics are
invalid and are not reported here.

## Detection coverage

Overall MediaPipe coverage on A–Z: **77.8%** (2023/2600).

| Class | Detected | Missed | Coverage |
| ----- | -------: | -----: | -------: |
| A | 75 | 25 | 75.0% |
| B | 71 | 29 | 71.0% |
| C | 67 | 33 | 67.0% |
| D | 80 | 20 | 80.0% |
| E | 84 | 16 | 84.0% |
| F | 96 | 4 | 96.0% |
| G | 81 | 19 | 81.0% |
| H | 83 | 17 | 83.0% |
| I | 79 | 21 | 79.0% |
| J | 91 | 9 | 91.0% |
| K | 91 | 9 | 91.0% |
| L | 82 | 18 | 82.0% |
| M | 46 | 54 | 46.0% |
| N | 48 | 52 | 48.0% |
| O | 80 | 20 | 80.0% |
| P | 63 | 37 | 63.0% |
| Q | 73 | 27 | 73.0% |
| R | 86 | 14 | 86.0% |
| S | 78 | 22 | 78.0% |
| T | 75 | 25 | 75.0% |
| U | 83 | 17 | 83.0% |
| V | 91 | 9 | 91.0% |
| W | 85 | 15 | 85.0% |
| X | 71 | 29 | 71.0% |
| Y | 88 | 12 | 88.0% |
| Z | 76 | 24 | 76.0% |

## RGB accuracy: detected vs missed (A–Z)

Overall RGB accuracy: **99.27%** (2581/2600).

| Condition | Accuracy |
| --------- | -------: |
| All A–Z frames | 99.27% |
| MediaPipe detected | 99.65% (2016/2023) |
| MediaPipe missed | 97.92% (565/577) |

## 2×2 outcome breakdown (A–Z)

| | RGB Correct | RGB Wrong | Total |
| - | ----------: | --------: | ----: |
| **MP Detected** | 2016 | 7 | 2023 |
| **MP Missed** | 565 | 12 | 577 |
| **Total** | 2581 | 19 | 2600 |

**565** correctly classified frames (21.9% of all correct)
would be discarded by a mandatory MediaPipe gate.

## Per-class detail

| Class | MP coverage | RGB acc all | RGB acc detected | RGB acc missed | RGB-correct discarded |
| ----- | ----------: | ----------: | ---------------: | -------------: | ---------------------: |
| A | 75.0% | 99.0% | 100.0% | 96.0% | 24 |
| B | 71.0% | 100.0% | 100.0% | 100.0% | 29 |
| C | 67.0% | 100.0% | 100.0% | 100.0% | 33 |
| D | 80.0% | 100.0% | 100.0% | 100.0% | 20 |
| E | 84.0% | 100.0% | 100.0% | 100.0% | 16 |
| F | 96.0% | 100.0% | 100.0% | 100.0% | 4 |
| G | 81.0% | 100.0% | 100.0% | 100.0% | 19 |
| H | 83.0% | 100.0% | 100.0% | 100.0% | 17 |
| I | 79.0% | 100.0% | 100.0% | 100.0% | 21 |
| J | 91.0% | 100.0% | 100.0% | 100.0% | 9 |
| K | 91.0% | 97.0% | 96.7% | 100.0% | 9 |
| L | 82.0% | 100.0% | 100.0% | 100.0% | 18 |
| M | 46.0% | 97.0% | 100.0% | 94.44% | 51 |
| N | 48.0% | 100.0% | 100.0% | 100.0% | 52 |
| O | 80.0% | 100.0% | 100.0% | 100.0% | 20 |
| P | 63.0% | 100.0% | 100.0% | 100.0% | 37 |
| Q | 73.0% | 100.0% | 100.0% | 100.0% | 27 |
| R | 86.0% | 99.0% | 98.84% | 100.0% | 14 |
| S | 78.0% | 91.0% | 96.15% | 72.73% | 16 |
| T | 75.0% | 98.0% | 100.0% | 92.0% | 23 |
| U | 83.0% | 100.0% | 100.0% | 100.0% | 17 |
| V | 91.0% | 100.0% | 100.0% | 100.0% | 9 |
| W | 85.0% | 100.0% | 100.0% | 100.0% | 15 |
| X | 71.0% | 100.0% | 100.0% | 100.0% | 29 |
| Y | 88.0% | 100.0% | 100.0% | 100.0% | 12 |
| Z | 76.0% | 100.0% | 100.0% | 100.0% | 24 |

## M/N case study

M and N are among the hardest classes for MediaPipe hand detection, likely due to
similar hand shapes where landmarks are harder to distinguish.

| | M | N |
| - | -: | -: |
| MP coverage | 46.0% (46/100) | 48.0% (48/100) |
| RGB acc (detected) | 100.0% | 100.0% |
| RGB acc (missed) | 94.44% | 100.0% |
| RGB-correct discarded | 51 | 52 |

Despite low MediaPipe coverage, the RGB model still classifies the majority of missed
M/N frames correctly, reinforcing that MediaPipe failure is not equivalent to
classification failure.

## Interpretation

> MediaPipe failure is not equivalent to image unclassifiability for the RGB model.

On this dataset and checkpoint, 97.9% of frames where MediaPipe
fails to detect a hand are still classified correctly by the RGB EfficientNet
(565/577 missed frames). A mandatory HandLandmarker gate
before classification would discard these correctly classifiable frames.

The discarding is **class-dependent**: classes like M and N have much lower MediaPipe
coverage (~46–48%) than classes like E (84%), meaning a mandatory gate would
disproportionately lose information for specific hand shapes.

**Scope:** These findings apply only to this MediaPipe configuration, this dataset,
this deterministic 100-sample-per-class subset, and this checkpoint. They do not
imply that MediaPipe is universally unreliable.

## Provenance

Raw experiment data: `mediapipe_vs_rgb/` (generated on Kaggle,
dataset `grassknoted/asl-alphabet`, seed 42, 100 samples/class).
Original CSVs preserved; this report computed from `mediapipe_vs_rgb_sample_records.csv`.

