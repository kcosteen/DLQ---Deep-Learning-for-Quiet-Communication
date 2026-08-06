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

Targeting the failing classes (#12)
-----------------------------------
``--rebalance-from`` takes a report from ``src.evaluate --report-json`` (or the
``confusion_matrix.csv`` from an older run) and aims the next run at whatever
actually failed, rather than at the classes a textbook predicted would:

* ``--rebalance loss`` weights the cross-entropy per class, ``--rebalance sampler``
  oversamples instead; weights come from measured per-class error.
* ``--geometry-safe-classes auto`` routes the failing classes and the classes they
  are confused WITH through the gentler ``geometry_safe`` augmentation policy.
* ``--resume`` fine-tunes an existing checkpoint instead of retraining from
  ImageNet, so a targeted fix costs one stage instead of two.

Colab free T4: ~15-40 min/epoch at 224×224 with AMP. Cache resized images to
Drive to speed up epochs.
"""

from __future__ import annotations

import argparse
import math
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from .data import CLASS_TO_IDX, CLASSES, SplitManifest, build_datasets, frame_range_split
from .model import build_model, freeze_backbone, unfreeze_backbone

# Share of a failing class's errors a predicted class must absorb to be pulled
# into the geometry-safe set alongside it. The decision boundary between S and E
# is learned from both classes' frames, so softening only S's augmentation while
# E's thumb is still being sheared away fixes half the pair.
PARTNER_ERROR_SHARE = 0.2


@dataclass
class TrainConfig:
    root: str
    manifest: Optional[str] = None
    backgrounds: Optional[str] = None  # dir of bg textures -> train-only bg replacement (#10)
    model: str = "efficientnet"  # "compact" = from-scratch baseline (#5)
    resume: Optional[str] = None  # fine-tune this checkpoint instead of ImageNet (#12)
    # --- targeted fixes (#12); all no-ops unless set ---
    class_weights: Optional[List[float]] = None  # per-class, CLASSES order
    rebalance: str = "loss"  # how class_weights are applied: "loss" | "sampler"
    geometry_safe_classes: Sequence[str] = field(default_factory=tuple)
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


def split_manifest_path(checkpoint_path: str) -> str:
    """Where the split manifest lands for a given checkpoint path."""
    return os.path.splitext(checkpoint_path)[0] + ".split.json"


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


def derive_class_weights(
    report: dict, alpha: float = 1.0, cap: float = 3.0
) -> List[float]:
    """Per-class loss weights from *measured* per-class accuracy: ``1 + alpha*(1-acc)``.

    Inverse-frequency weighting — the usual reflex — does nothing here: the split
    is 600 val frames and ~2,400 train frames for every one of the 29 classes.
    The imbalance is not in how many examples a class has, it is in how hard the
    class is: X sits at 0.507 while 21 classes sit at 1.000. So the weight is
    driven by error, not by count. A solved class gets 1.0 and X gets 1.49 at
    ``alpha=1``; raise ``alpha`` to lean harder.

    Weights are scaled to mean 1.0 purely so the printed numbers are readable —
    ``CrossEntropyLoss(reduction="mean")`` divides by the sum of the batch's
    weights, so a global multiplier already cancels out.

    Honest caveat, also recorded in ``docs/RESULTS_class_fixes.md``: these weights
    come from dev-val accuracy, which turns dev-val into a tuning target and makes
    it optimistically biased. That is an acceptable price only because dev-val is
    explicitly not the reported metric — the webcam set (#15/#16) stays quarantined
    and never feeds this.
    """
    if alpha < 0:
        raise ValueError("rebalance alpha must be >= 0")
    if cap < 1:
        raise ValueError("rebalance cap must be >= 1")
    per_class = report["per_class"]
    weights = []
    for c in CLASSES:
        acc = (per_class.get(c) or {}).get("accuracy")
        # A class absent from the report gets the neutral weight; guessing would
        # be worse than leaving it alone.
        weights.append(1.0 if acc is None else min(cap, 1.0 + alpha * (1.0 - acc)))
    mean = sum(weights) / len(weights)
    return [w / mean for w in weights]


def auto_geometry_safe_classes(
    report: dict, target: Optional[float] = None
) -> List[str]:
    """Classes below the per-class bar, plus the classes they are confused WITH.

    Derived from the confusion matrix rather than from the project plan's
    M/N/S/T, A/E, K/V list, because that list was measured wrong on the first run
    (#6): A/E never confused at all, while S->E and X->A/L/R — the two biggest
    failures — were not on it. A set taken from the matrix cannot miss a
    confusion for being unanticipated.
    """
    from .evaluate import PER_CLASS_TARGET, report_matrix

    if target is None:
        target = report.get("per_class_target") or PER_CLASS_TARGET
    cm = report_matrix(report)
    chosen = set()
    for i, c in enumerate(CLASSES):
        total = int(cm[i].sum())
        if not total:
            continue
        acc = cm[i, i] / total
        if acc >= target:
            continue
        chosen.add(c)
        errors = total - int(cm[i, i])
        if not errors:
            continue
        for j, other in enumerate(CLASSES):
            if j != i and cm[i, j] / errors >= PARTNER_ERROR_SHARE:
                chosen.add(other)
    return [c for c in CLASSES if c in chosen]  # CLASSES order, not set order


def resolve_geometry_safe_classes(
    spec: Optional[str], report: Optional[dict]
) -> List[str]:
    """Parse ``--geometry-safe-classes``: empty, ``auto``, or a comma-separated list."""
    if not spec or spec.lower() in ("none", ""):
        return []
    if spec.lower() == "auto":
        if report is None:
            raise ValueError(
                "--geometry-safe-classes auto needs --rebalance-from <report> "
                "to know which classes are failing"
            )
        return auto_geometry_safe_classes(report)
    names = [s.strip() for s in spec.split(",") if s.strip()]
    unknown = [n for n in names if n not in CLASS_TO_IDX]
    if unknown:
        raise ValueError(
            f"--geometry-safe-classes: not ASL class names: {unknown}\n"
            f"  valid: {list(CLASSES)}"
        )
    return [c for c in CLASSES if c in set(names)]


def assert_not_webcam_derived(report: dict, path: str) -> None:
    """Refuse to aim a training run using the quarantined webcam test set.

    CONTRIBUTING's quarantine rule: the webcam set "never trains, and it never
    picks a checkpoint, epoch, threshold, or augmentation setting". Class weights
    and the geometry-safe class list are augmentation settings, so a webcam report
    reaching ``--rebalance-from`` would spend the one honest number this project
    has — and it would do so invisibly, because the resulting run looks exactly
    like a legitimate one. The report records its own source, so this is cheap to
    enforce rather than merely documented.

    A confusion-matrix CSV records no source, so it can only be warned about — the
    check is as strong as the provenance it is given, and ``--report-json`` is the
    form that carries any.
    """
    kind = (report.get("source") or {}).get("kind")
    if report.get("is_reported_metric") or kind == "webcam":
        raise SystemExit(
            f"--rebalance-from {path} is a WEBCAM report.\n"
            f"  The webcam set never selects an augmentation setting — that is the\n"
            f"  quarantine rule that keeps it the reported number (CONTRIBUTING.md,\n"
            f"  docs/WEBCAM_TESTSET_PROTOCOL.md). Target the dev-val report instead."
        )
    if kind not in ("dev_val", "webcam"):
        print(f"warning: {path} does not record which split it came from, so the "
              f"webcam-quarantine check cannot verify it. Confirm it is a DEV-VAL "
              f"result; prefer 'evaluate --report-json', which records its source.")


def default_checkpoint_path(model: str, targeted: bool) -> str:
    """Where a run's checkpoint lands when ``--out`` is not given.

    A targeted re-run (#12) gets its own filename rather than the baseline's,
    because the #12 deliverable is a before/after comparison and the baseline
    checkpoint is half of it. Landing on the same default would destroy the
    "before" on the first epoch of the "after".
    """
    stem = "compact_cnn" if model == "compact" else "efficientnet_b0"
    return f"checkpoints/{stem}{'_targeted' if targeted else ''}.pt"


def assert_out_does_not_clobber_resume(
    out: str, resume: Optional[str], allow_overwrite: bool = False
) -> None:
    """Refuse a run whose output would overwrite the checkpoint it resumes from.

    ``_run_stage`` saves whenever the epoch beats ``best``, which starts at 0.0 —
    so the first epoch always writes, however bad it is. If that write lands on
    the resumed file, the baseline is gone before anyone knows whether the
    targeted run improved on it.
    """
    if not resume or allow_overwrite:
        return
    if os.path.abspath(resume) == os.path.abspath(out):
        raise SystemExit(
            f"--out would overwrite the checkpoint being resumed ({out}).\n"
            f"  The baseline is the 'before' half of the #12 comparison — keep it.\n"
            f"  Pass a different --out, or --allow-overwrite to accept the loss."
        )


def loss_class_weights(cfg: "TrainConfig") -> Optional[List[float]]:
    """Weights for the loss — ``None`` under ``--rebalance sampler``.

    The two mechanisms express the same weights: one makes each mistake on a hard
    class count more, the other shows that class more often. Applying both would
    square the emphasis, which at ``alpha=1`` would push X from 1.43x to 2.0x
    without anyone asking for it.
    """
    if cfg.class_weights is None or cfg.rebalance != "loss":
        return None
    return list(cfg.class_weights)


def sampler_weights(
    cfg: "TrainConfig", samples: Sequence
) -> Optional[List[float]]:
    """Per-SAMPLE draw weights under ``--rebalance sampler``, else ``None``.

    Every frame inherits its class's weight, so a class weighted 1.5x is drawn
    ~1.5x as often.
    """
    if cfg.class_weights is None or cfg.rebalance != "sampler":
        return None
    return [cfg.class_weights[s.label] for s in samples]


def load_resume_state(model: nn.Module, path: str, arch: str) -> Optional[float]:
    """Load weights into ``model`` for a focused fine-tune; return recorded dev-val.

    Refuses a checkpoint from a different architecture: ``load_state_dict`` would
    otherwise fail with thousands of unmatched parameter names, which buries the
    one-line cause the same way ``evaluate.load_checkpoint`` guards against.
    """
    ckpt = torch.load(path, map_location="cpu")
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    recorded = ckpt.get("arch") if isinstance(ckpt, dict) else None
    if recorded is not None and recorded != arch:
        raise SystemExit(
            f"--resume {path} is a '{recorded}' checkpoint but --model is '{arch}'."
        )
    model.load_state_dict(state)
    return ckpt.get("val_acc") if isinstance(ckpt, dict) else None


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


def _checkpoint_meta(cfg: TrainConfig) -> Dict:
    """The targeting choices a checkpoint has to carry to stay interpretable (#12)."""
    return {
        "resumed_from": cfg.resume,
        "rebalance": cfg.rebalance if cfg.class_weights is not None else None,
        "class_weights": cfg.class_weights,
        "geometry_safe_classes": list(cfg.geometry_safe_classes),
        "backgrounds": cfg.backgrounds,
    }


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
    weights = loss_class_weights(cfg)
    weight = (torch.tensor(weights, dtype=torch.float32, device=device)
              if weights is not None else None)
    criterion = nn.CrossEntropyLoss(weight=weight,
                                    label_smoothing=cfg.label_smoothing)
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
            # "arch" makes the checkpoint self-describing: evaluate.py rebuilds
            # the right architecture without being told which one to expect. The
            # #12 fields go with it for the same reason — two checkpoints with
            # different rebalancing are not comparable, and a file that does not
            # say which it was cannot be told apart later.
            torch.save({"model": model.state_dict(), "val_acc": best,
                        "arch": cfg.model, **_checkpoint_meta(cfg)}, cfg.out)
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

    # Pin the split next to the checkpoint. Re-deriving it later works only while
    # ``root`` holds byte-identical contents (frame_range_split is deterministic
    # given (root, train_frac) — src/data.py), so evaluating a checkpoint against
    # a root that has since gained or lost frames would silently score it on a
    # different val set. That would undermine the one claim this split exists to
    # support: that the reported frames were never trained on. The manifest also
    # records a content_sha256 of the assignment, so a drifted split is
    # detectable rather than merely suspected.
    manifest_out = split_manifest_path(cfg.out)
    os.makedirs(os.path.dirname(manifest_out) or ".", exist_ok=True)
    manifest.to_json(manifest_out)

    train_ds, val_ds = build_datasets(
        manifest, backgrounds_dir=cfg.backgrounds,
        geometry_safe_classes=cfg.geometry_safe_classes,
    )
    bg_note = f"bg-replacement ON ({cfg.backgrounds})" if cfg.backgrounds else "bg-replacement off"
    print(f"train={len(train_ds)} val={len(val_ds)}  (leakage-safe frame-range split; {bg_note})")
    print(f"split manifest -> {manifest_out}  (pass it to src/evaluate.py --manifest)")
    if cfg.geometry_safe_classes:
        print(f"geometry-safe augmentation for {list(cfg.geometry_safe_classes)} "
              f"(train only; val stays the deterministic contract)")
    if cfg.class_weights is not None:
        emphasised = sorted(
            ((w, c) for c, w in zip(CLASSES, cfg.class_weights) if w > 1.05),
            reverse=True,
        )
        print(f"rebalance={cfg.rebalance}; up-weighted: "
              f"{[(c, round(w, 2)) for w, c in emphasised] or 'none'}")

    sampler = None
    per_sample = sampler_weights(cfg, manifest.train)
    if per_sample is not None:
        # replacement=True is required — without it a "weighted" sampler over
        # num_samples == len(dataset) is just a permutation.
        sampler = WeightedRandomSampler(per_sample, num_samples=len(per_sample),
                                        replacement=True)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size,
                              shuffle=sampler is None, sampler=sampler,
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

    # pretrained=False when resuming: the ImageNet weights would be overwritten by
    # the checkpoint anyway, and skipping the download makes the run reproducible
    # offline.
    model = build_model(cfg.model, pretrained=cfg.resume is None).to(device)
    resumed_acc = None
    if cfg.resume:
        resumed_acc = load_resume_state(model, cfg.resume, cfg.model)
        shown = f"{resumed_acc:.4f}" if resumed_acc is not None else "unrecorded"
        print(f"resumed {cfg.resume} (recorded dev-val {shown}) — skipping the "
              f"head stage, fine-tuning directly")

    best = 0.0
    if cfg.model == "compact":
        # Trained from scratch: there is no pretrained backbone to freeze, so the
        # two-stage schedule does not apply — one stage at ``baseline_lr``.
        origin = "resumed" if cfg.resume else "from scratch"
        print(f"Baseline: CompactCNN {origin} (single stage)")
        best = _run_stage(model, train_loader, val_loader, device,
                          cfg.baseline_epochs, cfg.baseline_lr, cfg, wandb,
                          "baseline", best)
    else:
        if cfg.resume is None:
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

    if resumed_acc is not None:
        delta = best - resumed_acc
        print(f"vs resumed checkpoint: {resumed_acc:.4f} -> {best:.4f} ({delta:+.4f})")
        # Overall accuracy is the wrong scoreboard for #12 — the whole point is
        # three classes, and 26 solved ones dilute them. Say so where the number
        # is printed, not only in the docs.
        print("Overall accuracy is not the #12 criterion: run src/evaluate.py "
              "--baseline <previous report.json> for the per-class before/after.")

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
    p.add_argument("--backgrounds", default=None,
                   help="dir of background textures (e.g. data/backgrounds); enables "
                        "train-only background replacement (#10). Omit to disable.")
    p.add_argument("--model", choices=["efficientnet", "compact"],
                   default="efficientnet",
                   help="'compact' = from-scratch baseline (single stage)")
    p.add_argument("--resume", default=None,
                   help="fine-tune this checkpoint instead of ImageNet weights; "
                        "skips the frozen-head stage (#12)")
    p.add_argument("--allow-overwrite", action="store_true",
                   help="permit --out to overwrite --resume (refused otherwise)")
    p.add_argument("--rebalance-from", default=None,
                   help="a report from 'src.evaluate --report-json' (or a "
                        "confusion_matrix.csv): weight training toward the "
                        "classes measured to be failing (#12)")
    p.add_argument("--rebalance", choices=["loss", "sampler"], default="loss",
                   help="apply the weights to the loss, or oversample instead")
    p.add_argument("--rebalance-alpha", type=float, default=1.0,
                   help="weight = 1 + alpha*(1 - per_class_accuracy); 0 disables")
    p.add_argument("--rebalance-max", type=float, default=3.0,
                   help="cap on any single class weight before normalisation")
    p.add_argument("--geometry-safe-classes", default=None,
                   help="comma-separated classes to augment gently, or 'auto' to "
                        "take them from --rebalance-from (#12)")
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
        args.out = default_checkpoint_path(
            args.model, targeted=bool(args.resume or args.rebalance_from)
        )
    assert_out_does_not_clobber_resume(args.out, args.resume, args.allow_overwrite)

    report = None
    if args.rebalance_from:
        from .evaluate import load_report
        report = load_report(args.rebalance_from)
        assert_not_webcam_derived(report, args.rebalance_from)
        print(f"targeting from {args.rebalance_from} "
              f"(below target: {report.get('below_target') or 'n/a'})")

    # ``!= 0``, not ``> 0``: a negative alpha is a mistake, and skipping the
    # weighting for it would produce a run that silently ignored the flag it was
    # given. derive_class_weights rejects it instead.
    class_weights = None
    if report is not None and args.rebalance_alpha != 0:
        class_weights = derive_class_weights(
            report, args.rebalance_alpha, args.rebalance_max
        )

    cfg = TrainConfig(
        root=args.root, manifest=args.manifest, backgrounds=args.backgrounds,
        model=args.model, resume=args.resume,
        class_weights=class_weights, rebalance=args.rebalance,
        geometry_safe_classes=resolve_geometry_safe_classes(
            args.geometry_safe_classes, report),
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
