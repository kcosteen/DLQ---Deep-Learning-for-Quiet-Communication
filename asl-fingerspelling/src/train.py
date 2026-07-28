"""Two-stage transfer-learning trainer with W&B logging.

Stage 1: freeze the EfficientNet-B0 backbone, train only the new head (LR 1e-3).
Stage 2: unfreeze and fine-tune the whole network at a low LR (1e-4).

Both stages use cross-entropy with label smoothing 0.1, AdamW (weight decay
0.01), cosine LR decay with warmup, mixed precision (AMP), and early stopping on
the LEAKAGE-SAFE frame-range val split — never a random split.

Colab free T4: ~15-40 min/epoch at 224×224 with AMP. Cache resized images to
Drive to speed up epochs.
"""

from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .data import SplitManifest, build_datasets, frame_range_split
from .model import build_model, freeze_backbone, unfreeze_backbone


@dataclass
class TrainConfig:
    root: str
    manifest: Optional[str] = None
    train_frac: float = 0.8
    head_epochs: int = 5
    finetune_epochs: int = 15
    batch_size: int = 48
    head_lr: float = 1e-3
    finetune_lr: float = 1e-4
    weight_decay: float = 0.01
    label_smoothing: float = 0.1
    warmup_epochs: int = 1
    patience: int = 5
    num_workers: int = 4
    amp: bool = True
    wandb_project: str = "asl-fingerspelling"
    wandb_mode: str = "online"  # set "disabled" for local runs
    out: str = "checkpoints/efficientnet_b0.pt"


def cosine_warmup_lr(step: int, total_steps: int, warmup_steps: int, base_lr: float) -> float:
    """Linear warmup then cosine decay to 0."""
    if step < warmup_steps:
        return base_lr * (step + 1) / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * base_lr * (1.0 + math.cos(math.pi * progress))


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: str) -> float:
    """Top-1 accuracy on the leakage-safe val split."""
    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x).argmax(1)
        correct += (pred == y).sum().item()
        total += y.numel()
    return correct / max(1, total)


def _run_stage(
    model, loader, val_loader, device, epochs, base_lr, cfg, wandb, stage, best
):
    """Run one training stage; returns updated best val accuracy."""
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=base_lr, weight_decay=cfg.weight_decay)
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.amp and device == "cuda")

    total_steps = epochs * max(1, len(loader))
    warmup_steps = cfg.warmup_epochs * max(1, len(loader))
    step = 0
    epochs_no_improve = 0

    for epoch in range(epochs):
        model.train()
        running = 0.0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            lr = cosine_warmup_lr(step, total_steps, warmup_steps, base_lr)
            for g in opt.param_groups:
                g["lr"] = lr
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=cfg.amp and device == "cuda"):
                loss = criterion(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            running += loss.item()
            step += 1

        val_acc = evaluate(model, val_loader, device)
        train_loss = running / max(1, len(loader))
        print(f"[{stage}] epoch {epoch + 1}/{epochs} "
              f"loss={train_loss:.4f} val_acc={val_acc:.4f} lr={lr:.2e}")
        if wandb is not None:
            wandb.log({f"{stage}/loss": train_loss, f"{stage}/val_acc": val_acc,
                       "lr": lr})

        if val_acc > best:
            best = val_acc
            epochs_no_improve = 0
            os.makedirs(os.path.dirname(cfg.out) or ".", exist_ok=True)
            torch.save({"model": model.state_dict(), "val_acc": best}, cfg.out)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= cfg.patience:
                print(f"[{stage}] early stopping (no val improvement)")
                break
    return best


def train(cfg: TrainConfig) -> float:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}")

    if cfg.manifest and os.path.exists(cfg.manifest):
        manifest = SplitManifest.from_json(cfg.manifest)
    else:
        manifest = frame_range_split(cfg.root, cfg.train_frac)
    train_ds, val_ds = build_datasets(manifest)
    print(f"train={len(train_ds)} val={len(val_ds)}  (leakage-safe frame-range split)")

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=cfg.num_workers, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                            num_workers=cfg.num_workers)

    try:
        import wandb
        wandb.init(project=cfg.wandb_project, mode=cfg.wandb_mode, config=cfg.__dict__)
    except Exception as e:  # W&B is optional / offline-friendly
        print(f"W&B disabled: {e}")
        wandb = None

    model = build_model("efficientnet", pretrained=True).to(device)

    best = 0.0
    freeze_backbone(model)
    print("Stage 1: training head (backbone frozen)")
    best = _run_stage(model, train_loader, val_loader, device,
                      cfg.head_epochs, cfg.head_lr, cfg, wandb, "head", best)

    unfreeze_backbone(model)
    print("Stage 2: fine-tuning whole network")
    best = _run_stage(model, train_loader, val_loader, device,
                      cfg.finetune_epochs, cfg.finetune_lr, cfg, wandb, "finetune", best)

    print(f"best leakage-safe val acc = {best:.4f}  (checkpoint: {cfg.out})")
    print("NOTE: the REAL reported number is the webcam test set, not this val "
          "accuracy. Run src/evaluate.py on data/webcam_testset.")
    if wandb is not None:
        wandb.finish()
    return best


def main() -> None:
    p = argparse.ArgumentParser(description="Two-stage EfficientNet-B0 training.")
    p.add_argument("--root", required=True, help="asl_alphabet_train/ dir")
    p.add_argument("--manifest", default=None, help="reuse a saved split manifest")
    p.add_argument("--train-frac", type=float, default=0.8)
    p.add_argument("--head-epochs", type=int, default=5)
    p.add_argument("--finetune-epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=48)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--wandb-mode", default="online",
                   choices=["online", "offline", "disabled"])
    p.add_argument("--out", default="checkpoints/efficientnet_b0.pt")
    args = p.parse_args()
    cfg = TrainConfig(
        root=args.root, manifest=args.manifest, train_frac=args.train_frac,
        head_epochs=args.head_epochs, finetune_epochs=args.finetune_epochs,
        batch_size=args.batch_size, num_workers=args.num_workers,
        amp=not args.no_amp, wandb_mode=args.wandb_mode, out=args.out,
    )
    train(cfg)


if __name__ == "__main__":
    main()
