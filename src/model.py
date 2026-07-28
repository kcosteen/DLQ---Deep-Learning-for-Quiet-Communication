"""Models: EfficientNet-B0 transfer-learning net + compact-CNN baseline.

- ``build_efficientnet``: ImageNet-pretrained EfficientNet-B0 (timm) with a fresh
  head ``Dropout(0.3) -> Linear(1280, 29)``. Two-stage training freezes then
  unfreezes the backbone (see ``freeze_backbone`` / ``unfreeze_backbone``).
- ``CompactCNN``: a 3-block conv net trained from scratch — the honest baseline
  the transfer-learning model must beat.
- ``logreg_sanity``: logistic regression on a pixel subset — the sanity FLOOR;
  if the pipeline can't beat this, something is wired wrong.
"""

from __future__ import annotations

import argparse

import torch
import torch.nn as nn

NUM_CLASSES = 29
EFFICIENTNET_FEATURES = 1280


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


def freeze_backbone(model: nn.Module) -> None:
    """Stage 1: freeze the timm backbone, leave the head trainable."""
    backbone = model[0]
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


def logreg_sanity(x: torch.Tensor, num_classes: int = NUM_CLASSES) -> nn.Module:
    """Logistic regression on flattened, sub-sampled pixels — the sanity floor."""
    n_features = x.reshape(x.shape[0], -1).shape[1]
    return nn.Linear(n_features, num_classes)


def build_model(name: str, pretrained: bool = True) -> nn.Module:
    """Factory used by ``train.py`` / ``export.py``."""
    if name == "efficientnet":
        return build_efficientnet(pretrained=pretrained)
    if name == "compact":
        return CompactCNN()
    raise ValueError(f"unknown model '{name}' (use efficientnet | compact)")


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main() -> None:
    p = argparse.ArgumentParser(description="Inspect a model definition.")
    p.add_argument("--model", choices=["efficientnet", "compact"], default="efficientnet")
    p.add_argument("--no-pretrained", action="store_true")
    args = p.parse_args()
    model = build_model(args.model, pretrained=not args.no_pretrained)
    dummy = torch.zeros(2, 3, 224, 224)
    out = model(dummy)
    print(f"model={args.model}  trainable_params={count_parameters(model):,}")
    print(f"output shape for batch of 2: {tuple(out.shape)}")


if __name__ == "__main__":
    main()
