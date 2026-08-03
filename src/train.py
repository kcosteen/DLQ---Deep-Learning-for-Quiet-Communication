"""Trainer for the EfficientNet-B0 transfer model and the CompactCNN baseline.

``--model efficientnet`` (default) runs the two-stage schedule:
Stage 1: freeze the EfficientNet-B0 backbone, train only the new head (LR 1e-3).
Stage 2: unfreeze and fine-tune the whole network at a low LR (1e-4).

``--model compact`` trains the from-scratch 3-block CNN in a single stage — the
honest baseline the transfer model must beat (#5). Pair it with ``--sanity-floor``
to also fit the logistic-regression pixel floor and assert baseline > floor.

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
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .data import CLASSES, SplitManifest, build_datasets, frame_range_split
from .model import build_model, freeze_backbone, unfreeze_backbone


@dataclass
class TrainConfig:
    root: str
    manifest: Optional[str] = None
    model: str = "efficientnet"  # "compact" = from-scratch baseline (#5)
    train_frac: float = 0.8
    head_epochs: int = 5
    finetune_epochs: int = 15
    baseline_epochs: int = 20  # compact CNN: single stage, nothing to freeze
    baseline_lr: float = 1e-3
    sanity_floor: bool = False  # also fit the logreg floor (#5)
    floor_max_per_class: int = 200
    batch_size: int = 48
    head_lr: float = 1e-3
    finetune_lr: float = 1e-4
    weight_decay: float = 0.01
    label_smoothing: float = 0.1
    warmup_epochs: int = 1
    patience: int = 5
    num_workers: int = 4
    amp: bool = True
    limit_batches: Optional[int] = None  # cap batches/epoch: smoke runs, not real ones
    log_every: int = 50  # batches between progress lines; 0 disables
    wandb_project: str = "asl-fingerspelling"
    wandb_mode: str = "online"  # set "disabled" for local runs
    out: str = "checkpoints/efficientnet_b0.pt"


def cosine_warmup_lr(step: int, total_steps: int, warmup_steps: int, base_lr: float) -> float:
    """Linear warmup then cosine decay to 0."""
    if step < warmup_steps:
        return base_lr * (step + 1) / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * base_lr * (1.0 + math.cos(math.pi * progress))


def assert_all_classes_present(manifest: SplitManifest) -> None:
    """Refuse to train on a split that is missing any of the 29 classes.

    ``list_class_files`` returns ``[]`` for an absent class folder and
    ``frame_range_split`` skips it. That leniency is deliberate — it lets a
    partial webcam capture still split — but for a full training run it is a
    trap: the head always has ``len(CLASSES)`` outputs, so a missing folder
    silently trains a 29-way classifier on 28 classes and still reports a
    plausible dev-val accuracy. Nothing downstream notices until the demo's
    missing gesture never fires.

    The usual cause is Kaggle's delete gesture, shipped as ``del`` while
    ``CLASSES`` expects ``delete``.
    """
    in_train = {s.class_name for s in manifest.train}
    in_val = {s.class_name for s in manifest.val}
    missing_train = [c for c in CLASSES if c not in in_train]
    missing_val = [c for c in CLASSES if c not in in_val]
    if not missing_train and not missing_val:
        return

    lines = [f"split does not cover all {len(CLASSES)} classes:"]
    if missing_train:
        lines.append(f"  absent from train: {missing_train}")
    if missing_val:
        lines.append(f"  absent from val:   {missing_val}")
    if "delete" in set(missing_train) | set(missing_val):
        lines.append(
            "  hint: Kaggle ships the delete gesture as 'del', but CLASSES "
            "expects 'delete'.\n"
            "        Writable root: rename the folder (see README).\n"
            "        Read-only root (e.g. /kaggle/input): point --root at a "
            "directory of\n"
            "        per-class symlinks instead, with 'del' linked as 'delete'."
        )
    lines.append(f"  root: {manifest.root}")
    raise ValueError("\n".join(lines))


def compute_sanity_floor(
    manifest: SplitManifest, max_per_class: int = 200, seed: int = 0
) -> float:
    """Logistic regression on downscaled pixels — the floor the baseline must beat.

    Subsamples ``max_per_class`` frames per class from each side of the split;
    fitting on all 87k images buys nothing and costs far more than the floor is
    worth. The subsample is a deterministic stride over the already-sorted frame
    range, not a random draw, so it stays reproducible and spans the whole range
    instead of clustering at one end.
    """
    from collections import defaultdict

    import cv2

    from .model import fit_logreg_sanity, pixel_features

    def subsample(samples):
        by_class = defaultdict(list)
        for s in samples:
            by_class[s.class_name].append(s)
        chosen = []
        for cls in sorted(by_class):
            group = by_class[cls]
            stride = max(1, len(group) // max_per_class)
            chosen += group[::stride][:max_per_class]
        return chosen

    def featurize(samples):
        X, y = [], []
        for s in samples:
            img = cv2.imread(s.path, cv2.IMREAD_COLOR)
            if img is None:
                raise FileNotFoundError(s.path)
            X.append(pixel_features(img))
            y.append(s.label)
        return np.stack(X), np.asarray(y)

    X_train, y_train = featurize(subsample(manifest.train))
    X_val, y_val = featurize(subsample(manifest.val))
    _, acc = fit_logreg_sanity(X_train, y_train, X_val, y_val, seed=seed)
    return acc


@torch.no_grad()
def evaluate(
    model: nn.Module, loader: DataLoader, device: str,
    limit_batches: Optional[int] = None,
) -> float:
    """Top-1 accuracy on the leakage-safe val split.

    ``limit_batches`` caps the number of val batches so a ``--limit-batches``
    smoke run is not dominated by a full pass over the val split.
    """
    model.eval()
    correct = total = 0
    for i, (x, y) in enumerate(loader):
        if limit_batches is not None and i >= limit_batches:
            break
        x, y = x.to(device), y.to(device)
        pred = model(x).argmax(1)
        correct += (pred == y).sum().item()
        total += y.numel()
    return correct / max(1, total)


def _steps_per_epoch(loader: DataLoader, limit_batches: Optional[int]) -> int:
    """Batches actually run per epoch, honouring ``--limit-batches``."""
    n = max(1, len(loader))
    return min(n, limit_batches) if limit_batches else n


def _fmt_eta(seconds: float) -> str:
    if seconds < 90:
        return f"{seconds:.0f}s"
    if seconds < 5400:
        return f"{seconds / 60:.0f}m"
    return f"{seconds / 3600:.1f}h"


def _run_stage(
    model, loader, val_loader, device, epochs, base_lr, cfg, wandb, stage, best
):
    """Run one training stage; returns updated best val accuracy."""
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=base_lr, weight_decay=cfg.weight_decay)
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.amp and device == "cuda")

    steps_per_epoch = _steps_per_epoch(loader, cfg.limit_batches)
    total_steps = epochs * steps_per_epoch
    warmup_steps = cfg.warmup_epochs * steps_per_epoch
    step = 0
    epochs_no_improve = 0

    for epoch in range(epochs):
        model.train()
        running = 0.0
        seen = 0
        epoch_start = time.perf_counter()
        for i, (x, y) in enumerate(loader):
            if cfg.limit_batches is not None and i >= cfg.limit_batches:
                break
            x, y = x.to(device), y.to(device)
            lr = cosine_warmup_lr(step, total_steps, warmup_steps, base_lr)
            for g in opt.param_groups:
                g["lr"] = lr
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=cfg.amp and device == "cuda"):
                loss = criterion(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            running += loss.item()
            seen += 1
            step += 1

            # An epoch is 20-40 min on a T4 and the only other output is at the
            # end of it, so without this a long run is indistinguishable from a
            # hung one. flush=True because Kaggle/Colab buffer subprocess stdout.
            if cfg.log_every and seen % cfg.log_every == 0:
                elapsed = time.perf_counter() - epoch_start
                rate = seen / max(1e-9, elapsed)
                eta = (steps_per_epoch - seen) / max(1e-9, rate)
                print(f"[{stage}] epoch {epoch + 1}/{epochs} "
                      f"batch {seen}/{steps_per_epoch} "
                      f"loss={running / seen:.4f} lr={lr:.2e} "
                      f"{rate * loader.batch_size:.0f} img/s "
                      f"eta {_fmt_eta(eta)}", flush=True)

        val_acc = evaluate(model, val_loader, device, cfg.limit_batches)
        train_loss = running / max(1, seen)
        print(f"[{stage}] epoch {epoch + 1}/{epochs} "
              f"loss={train_loss:.4f} dev_val_acc={val_acc:.4f} lr={lr:.2e} "
              f"({_fmt_eta(time.perf_counter() - epoch_start)})", flush=True)
        if wandb is not None:
            # "dev_val", never plain "val": this is the same-signer temporal
            # split, not the reported webcam number (#15/#16).
            wandb.log({f"{stage}/loss": train_loss,
                       f"{stage}/dev_val_acc": val_acc, "lr": lr})

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
    # Fail before the GPU hours, not after: a missing class folder splits and
    # trains without complaint (see assert_all_classes_present).
    assert_all_classes_present(manifest)
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

    floor = None
    if cfg.sanity_floor:
        floor = compute_sanity_floor(manifest, cfg.floor_max_per_class)
        print(f"logreg sanity floor (dev val) = {floor:.4f}")
        if wandb is not None:
            wandb.summary["dev_val/logreg_floor"] = floor

    model = build_model(cfg.model, pretrained=True).to(device)

    best = 0.0
    if cfg.model == "compact":
        # Trained from scratch: there is no pretrained backbone to freeze, so the
        # two-stage schedule does not apply — one stage at ``baseline_lr``.
        print("Baseline: CompactCNN from scratch (single stage)")
        best = _run_stage(model, train_loader, val_loader, device,
                          cfg.baseline_epochs, cfg.baseline_lr, cfg, wandb,
                          "baseline", best)
    else:
        freeze_backbone(model)
        print("Stage 1: training head (backbone frozen)")
        best = _run_stage(model, train_loader, val_loader, device,
                          cfg.head_epochs, cfg.head_lr, cfg, wandb, "head", best)

        unfreeze_backbone(model)
        print("Stage 2: fine-tuning whole network")
        best = _run_stage(model, train_loader, val_loader, device,
                          cfg.finetune_epochs, cfg.finetune_lr, cfg, wandb,
                          "finetune", best)

    if cfg.limit_batches:
        print(f"SMOKE RUN (--limit-batches {cfg.limit_batches}): the accuracy "
              "below is from a truncated pass over a fraction of the data. It "
              "is a wiring check, NOT a result - do not report or compare it, "
              "and re-run without --limit-batches for a real number.")
    print(f"best dev val acc = {best:.4f}  (checkpoint: {cfg.out})")
    if wandb is not None:
        wandb.summary["dev_val/best_acc"] = best

    if floor is not None:
        gap = best - floor
        verdict = "PASS" if gap > 0 else "FAIL"
        print(f"[{verdict}] {cfg.model} {best:.4f} vs logreg floor {floor:.4f} "
              f"(gap {gap:+.4f})")
        if verdict == "FAIL":
            print("The model did not beat the sanity floor — something in the "
                  "data pipeline or training loop is wired wrong.")

    print("NOTE: the REAL reported number is the webcam test set, not this dev "
          "val accuracy (same signer, temporal split). Run src/evaluate.py on "
          "data/webcam_testset.")
    if wandb is not None:
        wandb.finish()
    return best


def main() -> None:
    p = argparse.ArgumentParser(
        description="Train EfficientNet-B0 (two-stage) or the CompactCNN baseline."
    )
    p.add_argument("--root", required=True, help="asl_alphabet_train/ dir")
    p.add_argument("--manifest", default=None, help="reuse a saved split manifest")
    p.add_argument("--model", choices=["efficientnet", "compact"],
                   default="efficientnet",
                   help="'compact' = from-scratch baseline (single stage)")
    p.add_argument("--sanity-floor", action="store_true",
                   help="also fit the logreg pixel floor and compare against it")
    p.add_argument("--floor-max-per-class", type=int, default=200)
    p.add_argument("--train-frac", type=float, default=0.8)
    p.add_argument("--head-epochs", type=int, default=5)
    p.add_argument("--finetune-epochs", type=int, default=15)
    p.add_argument("--baseline-epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=48)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--limit-batches", type=int, default=None,
                   help="cap batches per epoch (train and val) — smoke tests "
                        "only; the resulting accuracy is not a real number")
    p.add_argument("--log-every", type=int, default=50,
                   help="batches between progress lines (0 = only per-epoch)")
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--wandb-mode", default="online",
                   choices=["online", "offline", "disabled"])
    p.add_argument("--out", default=None,
                   help="checkpoint path (default: named after --model)")
    args = p.parse_args()
    if args.out is None:
        stem = "compact_cnn" if args.model == "compact" else "efficientnet_b0"
        args.out = f"checkpoints/{stem}.pt"
    cfg = TrainConfig(
        root=args.root, manifest=args.manifest, model=args.model,
        train_frac=args.train_frac,
        head_epochs=args.head_epochs, finetune_epochs=args.finetune_epochs,
        baseline_epochs=args.baseline_epochs,
        sanity_floor=args.sanity_floor,
        floor_max_per_class=args.floor_max_per_class,
        batch_size=args.batch_size, num_workers=args.num_workers,
        limit_batches=args.limit_batches, log_every=args.log_every,
        amp=not args.no_amp, wandb_mode=args.wandb_mode, out=args.out,
    )
    train(cfg)


if __name__ == "__main__":
    main()
