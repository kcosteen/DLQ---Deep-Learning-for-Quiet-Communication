# MediaPipe Tasks model assets

The crop bridge (`src/crop.py`) and background replacer (`src/augment.py`) run on
the modern **MediaPipe Tasks API** (`mediapipe>=1.0.0`), which requires the model
files below — they are **not** bundled inside the pip wheel.

These files are committed to the repo so the demo works out of the box. To
re-download them (e.g. after a version bump):

```bash
curl -L -o models/hand_landmarker.task \
  "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
curl -L -o models/selfie_segmenter.tflite \
  "https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter/float16/latest/selfie_segmenter.tflite"
```

| File | Used by | Purpose |
|---|---|---|
| `hand_landmarker.task` | `src/crop.HandCropper` | detect hand → normalized landmarks → bbox crop |
| `selfie_segmenter.tflite` | `src.augment.BackgroundReplacer` | segment person/hand → background replacement |
