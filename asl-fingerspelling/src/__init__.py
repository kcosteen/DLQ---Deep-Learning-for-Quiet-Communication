"""ASL fingerspelling recognition — source package.

Modules:
    crop      MediaPipe hand-crop bridge + the shared preprocessing contract
    data      folder loader + leakage-safe frame-range split + Dataset
    augment   augmentation pipeline + background replacement
    model     EfficientNet-B0 head + compact-CNN baseline
    train     two-stage transfer-learning trainer (+ W&B)
    evaluate  metrics, confusion matrix, webcam-set eval
    export    TorchScript / ONNX export
"""

__version__ = "3.0.0"
