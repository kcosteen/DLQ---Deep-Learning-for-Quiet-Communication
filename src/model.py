"""Models: EfficientNet-B0 transfer-learning net + compact-CNN baseline.

- ``build_efficientnet``: ImageNet-pretrained EfficientNet-B0 (timm) with a fresh
  head ``Dropout(0.3) -> Linear(1280, 29)``. Two-stage training freezes then
  unfreezes the backbone (see ``freeze_backbone`` / ``unfreeze_backbone``).
- ``CompactCNN``: a 3-block conv net trained from scratch — the honest baseline
  the transfer-learning model must beat.
- ``fit_logreg_sanity``: logistic regression on a pixel subset — the sanity FLOOR;
  if the pipeline can't beat this, something is wired wrong.
"""

from __future__ import annotations

import argparse

import numpy as np
import torch
import torch.nn as nn

NUM_CLASSES = 29
EFFICIENTNET_FEATURES = 1280

# --- adaptive shadow lift (#12) ----------------------------------------------
# S sits at 0.770 against the 85% bar and 138 of its frames are called E, with
# zero E frames called S. The cause is exposure, not modelling: the dataset is
# backlit, hand at grey 9-89 against a background pinned at 240-255. Every class
# solved by augmentation is one the *silhouette* distinguishes; S vs E is the one
# pair where both handshapes are compact fists, so the only cue is interior
# shading, and the exposure crushes it. See docs/RESULTS_class_fixes.md.
#
# Why the anchor is a low percentile and NOT the median. The frame median on this
# dataset is already ~120-140/255 (~0.5) because the blown-out background drags it
# up -- measured in docs/RESULTS_class_fixes.md's brightness table. Solving for a
# gamma that maps the *median* to a mid target is therefore very nearly a no-op
# and would leave the hand exactly as dark as it was. Anchoring on the 25th
# percentile targets the shadow region the hand actually lives in.
LIFT_PERCENTILE = 0.25
LIFT_TARGET = 0.35

# Clamped so the lift can only brighten, never darken. A well-exposed webcam frame
# whose anchor already sits at or above the target gets gamma 1.0 -- an exact
# no-op. This is the property a fixed gamma lacks: gamma 0.45 measurably hurt the
# already-correctly-exposed classes (X, W, delete) in the detection sweep.
GAMMA_MIN, GAMMA_MAX = 0.25, 1.0

# BT.601 luma, matching cv2.COLOR_RGB2GRAY so this agrees with the offline
# brightness measurements.
_LUMA = (0.299, 0.587, 0.114)
_EPS = 1e-6


def build_efficientnet(
    num_classes: int = NUM_CLASSES, pretrained: bool = True, dropout: float = 0.3
) -> nn.Module:
    """EfficientNet-B0 (timm) with a fresh classification head."""
    import timm

    model = timm.create_model(
        "efficientnet_b0", pretrained=pretrained, num_classes=0, global_pool="avg"
    )
    head = nn.Sequential(
        nn.Dropout(dropout), nn.Linear(EFFICIENTNET_FEATURES, num_classes)
    )
    return nn.Sequential(model, head)


class ExposureLift(nn.Module):
    """Per-image adaptive shadow lift, applied to the contract's output tensor.

    Takes the ImageNet-normalized C×H×W tensor ``preprocess_bgr`` produces, undoes
    the normalization to get back to [0, 1], solves for the gamma that maps this
    image's ``percentile`` luminance to ``target``, applies it, and re-normalizes.
    Parameter-free and deterministic.

    It lives in the *model* rather than in ``src/crop.py`` so that the fix travels
    with the checkpoint: ``arch`` is recorded when training saves, ``evaluate.py``
    rebuilds from it, and ``load_resume_state`` refuses a checkpoint whose arch
    disagrees. There is therefore no path to these weights that skips the lift —
    a stronger train/serve guarantee than a contract both sides must remember to
    call. The tradeoff is that exposure handling is no longer visible in
    ``crop.py``; if this closes #12, it should be migrated into the contract
    proper with Person C's sign-off (#4).
    """

    def __init__(
        self,
        percentile: float = LIFT_PERCENTILE,
        target: float = LIFT_TARGET,
        gamma_min: float = GAMMA_MIN,
        gamma_max: float = GAMMA_MAX,
    ) -> None:
        super().__init__()
        if not 0.0 < percentile < 1.0:
            raise ValueError(f"percentile must be in (0, 1), got {percentile}")
        if not 0.0 < target < 1.0:
            raise ValueError(f"target must be in (0, 1), got {target}")
        self.percentile = float(percentile)
        self.target = float(target)
        self.gamma_min = float(gamma_min)
        self.gamma_max = float(gamma_max)
        # Buffers, not parameters: they must move with .to(device) but never train.
        # Imported from the contract so there is one source of truth for them.
        from .crop import IMAGENET_MEAN, IMAGENET_STD

        self.register_buffer("_mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("_std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))
        self.register_buffer("_luma", torch.tensor(_LUMA).view(1, 3, 1, 1))

    def gamma_for(self, x: torch.Tensor) -> torch.Tensor:
        """The per-image gamma this input would get. Exposed for testing."""
        p = (x * self._std + self._mean).clamp(0.0, 1.0)
        lum = (p * self._luma).sum(dim=1).flatten(1)  # (N, H*W)
        # Percentile via sort rather than torch.quantile: quantile has no ONNX
        # export path, sort does. k is a Python int so it traces as a constant.
        k = min(int(self.percentile * lum.shape[1]), lum.shape[1] - 1)
        anchor = torch.sort(lum, dim=1).values[:, k].clamp(_EPS, 1.0 - _EPS)
        gamma = torch.log(torch.tensor(self.target, device=x.device)) / torch.log(anchor)
        return gamma.clamp(self.gamma_min, self.gamma_max)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        p = (x * self._std + self._mean).clamp(_EPS, 1.0)
        lifted = p ** self.gamma_for(x).view(-1, 1, 1, 1)
        return (lifted - self._mean) / self._std


def _backbone_of(model: nn.Module) -> nn.Module:
    """The timm backbone, whether or not an ExposureLift sits in front of it."""
    return model[1] if isinstance(model[0], ExposureLift) else model[0]


def freeze_backbone(model: nn.Module) -> None:
    """Stage 1: freeze the timm backbone, leave the head trainable."""
    backbone = _backbone_of(model)
    for p in backbone.parameters():
        p.requires_grad = False


def unfreeze_backbone(model: nn.Module) -> None:
    """Stage 2: unfreeze everything for low-LR fine-tuning."""
    for p in model.parameters():
        p.requires_grad = True


class CompactCNN(nn.Module):
    """A small 3-block CNN trained from scratch — the from-scratch baseline."""

    def __init__(self, num_classes: int = NUM_CLASSES) -> None:
        super().__init__()

        def block(cin: int, cout: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1),
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )

        self.features = nn.Sequential(
            block(3, 32), block(32, 64), block(64, 128)
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Dropout(0.3), nn.Linear(128, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.pool(self.features(x)))


PIXEL_SUBSET = 32  # floor features = 32x32 grayscale = 1024 dims


def pixel_features(img_bgr: "np.ndarray") -> "np.ndarray":
    """BGR uint8 image -> the sanity floor's flat feature vector.

    Deliberately bypasses the 224x224 preprocessing contract in ``src/crop.py``:
    the floor is meant to be the dumbest reasonable model, so it sees raw
    downscaled grayscale pixels in [0, 1] and nothing else. Using the real
    pipeline here would make the floor look better than it should and blunt the
    comparison it exists to provide.
    """
    import cv2

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(
        gray, (PIXEL_SUBSET, PIXEL_SUBSET), interpolation=cv2.INTER_AREA
    )
    return (small.astype("float32") / 255.0).reshape(-1)


def fit_logreg_sanity(X_train, y_train, X_val, y_val, seed: int = 0):
    """Fit multinomial logistic regression on pixel features.

    Returns ``(classifier, dev_val_accuracy)``. This is the sanity FLOOR: if the
    CompactCNN baseline cannot beat it on the same leakage-safe split, the data
    pipeline or the training loop is wired wrong.
    """
    from sklearn.linear_model import LogisticRegression

    clf = LogisticRegression(max_iter=1000, random_state=seed)
    clf.fit(X_train, y_train)
    return clf, float(clf.score(X_val, y_val))


def build_efficientnet_lift(
    num_classes: int = NUM_CLASSES, pretrained: bool = True, dropout: float = 0.3
) -> nn.Module:
    """``build_efficientnet`` behind an :class:`ExposureLift` (#12).

    A separate arch name rather than a flag on the existing one, deliberately:
    the name is what gets written into the checkpoint and checked on load, so
    "these weights expect a lifted input" becomes unforgeable rather than a
    convention. It also leaves plain ``efficientnet`` byte-identical, so the #6
    and #12 baseline checkpoints still load for comparison.
    """
    base = build_efficientnet(num_classes=num_classes, pretrained=pretrained, dropout=dropout)
    return nn.Sequential(ExposureLift(), *base)


def build_model(name: str, pretrained: bool = True) -> nn.Module:
    """Factory used by ``train.py`` / ``export.py``."""
    if name == "efficientnet":
        return build_efficientnet(pretrained=pretrained)
    if name == "efficientnet_lift":
        return build_efficientnet_lift(pretrained=pretrained)
    if name == "compact":
        return CompactCNN()
    raise ValueError(
        f"unknown model '{name}' (use efficientnet | efficientnet_lift | compact)"
    )


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main() -> None:
    p = argparse.ArgumentParser(description="Inspect a model definition.")
    p.add_argument(
        "--model",
        choices=["efficientnet", "efficientnet_lift", "compact"],
        default="efficientnet",
    )
    p.add_argument("--no-pretrained", action="store_true")
    args = p.parse_args()
    model = build_model(args.model, pretrained=not args.no_pretrained)
    dummy = torch.zeros(2, 3, 224, 224)
    out = model(dummy)
    print(f"model={args.model}  trainable_params={count_parameters(model):,}")
    print(f"output shape for batch of 2: {tuple(out.shape)}")


if __name__ == "__main__":
    main()
