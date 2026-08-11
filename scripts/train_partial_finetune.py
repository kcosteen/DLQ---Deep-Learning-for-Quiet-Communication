"""Partial fine-tuning of ``checkpoints/efficientnet_b0_targeted.pt``.

Controlled experiment: unfreeze ONLY the final EfficientNet-B0 MBConv stage
(``blocks[6]``, a single ``InvertedResidual``) plus the EXISTING
``Dropout(0.3) + Linear(1280, 29)`` classifier head. Everything else
(``conv_stem``, ``bn1``, ``blocks[0..5]``, ``conv_head``, ``bn2``,
``global_pool``) stays frozen and bit-identical to the parent. BatchNorm running
statistics are frozen everywhere (all ``_BatchNorm`` modules forced to eval), so
pretrained normalization is preserved exactly.

Question (one clean answer): can a small adaptation of the FINAL representation
block (on top of the head-only run) materially improve the 138 S->E errors
without regressing the strong global model?

This script runs the TRAINING phase only:

1. Verify manifest + parent checkpoint provenance (hashes, arch, rebalance,
   class weights, geometry-safe classes, backgrounds).
2. Inspect the ACTUAL loaded model structure and assert the trainable set.
3. Sanity gate: the parent must reproduce the baseline (acc 0.9775, macro-F1
   0.9775, errors 391, S->E 138) on the raw-image val pipeline — checked
   against the fresh gate report produced by ``src.evaluate``.
4. Differential-LR sweep over {backbone-block LR x head LR}:
   (3e-6, 3e-5), (1e-5, 1e-4), (3e-5, 3e-4). Image-space training (the train
   transform is the checkpoint's own), frozen early/middle backbone, trainable
   ``blocks[6]``, pooled 1280-D features, Dropout, trainable head. Loss =
   CrossEntropyLoss(label_smoothing=0.1, weight=<checkpoint's class weights>),
   AdamW (wd 0.01), cosine schedule with 1 warmup epoch (mirrors src/train.py).
   Early stopping on validation macro-F1 (MPS eval, full val split), patience 2.
5. Parameter-change verification: candidate vs parent state dicts — only the
   allowlisted final-block conv/SE + head params may differ; every BN tensor and
   every frozen tensor must be bit-identical.

The final evaluation, representation-change diagnostics, three-way comparison,
verdict, and the deployable checkpoint are produced by
``scripts/analyze_partial_finetune.py`` (the verdict must exist before a
deployable checkpoint is written).

Loss/augmentation semantics (documented difference vs head-only)
--------------------------------------------------------------
This is IMAGE-space training: the model sees fresh random augmentation every
epoch through the checkpoint's own train transform (geometry-safe policy for
the checkpoint's geometry_safe_classes, no background replacement). Unlike the
head-only cached-embedding run, this is the real training procedure with a
trainable final block.

Outputs (training phase)
-----------------------
- ``checkpoints/partial_ft/partial_ft_best.pt``  (full state dict of the best
  candidate across the sweep, with metadata)
- ``docs/reports/partial_finetune_training.{json,md}``
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from scripts.analyze_s_to_e import _resolve_device  # noqa: E402
from scripts.build_embedding_cache import file_sha256, state_dict_sha256  # noqa: E402
from scripts.train_head_only_cached import evaluate_cm, s_e_metrics  # noqa: E402
from src.data import CLASSES, SplitManifest, build_datasets  # noqa: E402
from src.evaluate import (  # noqa: E402
    confusion_matrix,
    load_checkpoint,
)
from src.train import cosine_warmup_lr  # noqa: E402

DEFAULT_PARENT = "checkpoints/efficientnet_b0_targeted.pt"
DEFAULT_MANIFEST = "data/split_manifest.json"
DEFAULT_GATE = "docs/reports/partial_ft_baseline_gate.json"
DEFAULT_CANDIDATE_OUT = "checkpoints/partial_ft/partial_ft_best.pt"
DEFAULT_TRAIN_JSON = "docs/reports/partial_finetune_training.json"
DEFAULT_TRAIN_MD = "docs/reports/partial_finetune_training.md"

PARENT_FILE_SHA = "3827f8797d0f822f"
MANIFEST_SHA = "454ca31b1d6a2b8a"
N_CLASSES = 29
FEATURE_DIM = 1280
EXPECTED_GEOMETRY_SAFE = ["A", "E", "K", "L", "R", "S", "V", "X"]
LABEL_SMOOTHING = 0.1
WEIGHT_DECAY = 0.01
WARMUP_EPOCHS = 1

# --- baseline gate tolerances (from docs/reports/dev_val_targeted_current.json)
GATE_ACC = 0.9775287356321839
GATE_MACRO_F1 = 0.9775082322752521
GATE_TOTAL_ERRORS = 391
GATE_S_TO_E = 138


# --------------------------------------------------------------------------- #
# Provenance + structure
# --------------------------------------------------------------------------- #
def verify_manifest(manifest_path: str, manifest: SplitManifest) -> Dict:
    sha = json.loads(Path(manifest_path).read_text()).get("content_sha256") or ""
    if not sha.startswith(MANIFEST_SHA):
        raise SystemExit(
            f"manifest content_sha256 {sha[:16]}… does not start with the "
            f"expected {MANIFEST_SHA}… — the split changed; STOP."
        )
    per_class = {c: sum(1 for s in manifest.val if s.class_name == c)
                 for c in CLASSES}
    bad = {c: n for c, n in per_class.items() if n != 600}
    if bad or len(manifest.train) != 69600 or len(manifest.val) != 17400:
        raise SystemExit(
            f"manifest counts unexpected: train {len(manifest.train)}, val "
            f"{len(manifest.val)}, non-600 val classes {bad} — STOP."
        )
    return {"manifest_sha256": sha, "n_train": len(manifest.train),
            "n_val": len(manifest.val), "val_per_class": per_class}


def verify_parent(checkpoint_path: str) -> Dict:
    sha = file_sha256(checkpoint_path)
    if not sha.startswith(PARENT_FILE_SHA):
        raise SystemExit(
            f"parent checkpoint file sha256 {sha[:16]}… does not start with "
            f"{PARENT_FILE_SHA}… — this is not the expected parent; STOP."
        )
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    if ckpt.get("arch") != "efficientnet":
        raise SystemExit(f"parent arch {ckpt.get('arch')!r} != 'efficientnet'")
    cw = list(ckpt.get("class_weights") or [])
    if len(cw) != N_CLASSES:
        raise SystemExit(f"parent has {len(cw)} class weights, expected 29")
    if ckpt.get("rebalance") != "loss":
        raise SystemExit(f"parent rebalance {ckpt.get('rebalance')!r} != 'loss'")
    gs = list(ckpt.get("geometry_safe_classes") or [])
    if gs != EXPECTED_GEOMETRY_SAFE:
        raise SystemExit(f"parent geometry_safe_classes {gs} != "
                         f"{EXPECTED_GEOMETRY_SAFE}")
    if ckpt.get("backgrounds") is not None:
        raise SystemExit("parent was trained with background replacement; "
                         "this experiment assumes backgrounds=None")
    sd = ckpt["model"]
    return {
        "path": checkpoint_path, "file_sha256": sha,
        "backbone_state_dict_sha256": state_dict_sha256(sd),
        "recorded_val_acc": ckpt.get("val_acc"),
        "rebalance": ckpt.get("rebalance"),
        "class_weights": cw,
        "geometry_safe_classes": gs,
        "backgrounds": ckpt.get("backgrounds"),
        "resumed_from": ckpt.get("resumed_from"),
    }


def is_bn_module(mod: nn.Module) -> bool:
    """All BatchNorm variants, incl. timm's BatchNormAct2d (subclass of BatchNorm2d)."""
    import torch.nn.modules.batchnorm as bn
    return isinstance(mod, bn._BatchNorm)


def bn_param_names(model: nn.Module) -> set:
    """Names of every trainable-affine BN tensor in the model."""
    names = set()
    for mname, mod in model.named_modules():
        if is_bn_module(mod):
            for pname, _ in mod.named_parameters(prefix=mname, recurse=False):
                names.add(pname)
    return names


def inspect_structure(model: nn.Module) -> Dict:
    """Print + assert the exact model structure; return the inspection record."""
    if not (isinstance(model, nn.Sequential) and len(model) == 2):
        raise SystemExit(f"model is not Sequential(backbone, head): {model}")
    backbone = model[0]
    for attr in ("conv_stem", "bn1", "blocks", "conv_head", "bn2",
                 "global_pool", "classifier"):
        if not hasattr(backbone, attr):
            raise SystemExit(f"backbone missing {attr}")
    blocks = backbone.blocks
    if len(blocks) != 7:
        raise SystemExit(f"expected 7 MBConv stages, got {len(blocks)}")
    if len(blocks[6]) != 1:
        raise SystemExit(f"blocks[6] should be a single InvertedResidual, "
                         f"got {len(blocks[6])} sub-blocks")
    head = model[1]
    if not (isinstance(head, nn.Sequential) and len(head) == 2
            and isinstance(head[0], nn.Dropout)
            and abs(head[0].p - 0.3) < 1e-9
            and isinstance(head[1], nn.Linear)
            and head[1].in_features == FEATURE_DIM
            and head[1].out_features == N_CLASSES):
        raise SystemExit(f"head is not Dropout(0.3)+Linear({FEATURE_DIM},"
                         f"{N_CLASSES}): {head}")

    def stage_params(stage) -> int:
        return sum(p.numel() for p in stage.parameters())

    rec = {
        "container": "nn.Sequential(backbone, head)",
        "backbone_type": type(backbone).__name__,
        "n_blocks": len(blocks),
        "blocks_params": [stage_params(b) for b in blocks],
        "conv_stem_params": stage_params(backbone.conv_stem),
        "bn1_params": stage_params(backbone.bn1),
        "conv_head_params": stage_params(backbone.conv_head),
        "bn2_params": stage_params(backbone.bn2),
        "global_pool_type": type(backbone.global_pool).__name__,
        "head_layers": ["Dropout(p=0.3)", f"Linear({FEATURE_DIM}, {N_CLASSES})"],
    }
    print("=== model structure ===")
    print(f"top-level: {rec['container']}  backbone={rec['backbone_type']}")
    for i, n in enumerate(rec["blocks_params"]):
        print(f"  blocks[{i}]: {n} params")
    print(f"  conv_stem {rec['conv_stem_params']} | bn1 {rec['bn1_params']} | "
          f"conv_head {rec['conv_head_params']} | bn2 {rec['bn2_params']} | "
          f"global_pool {rec['global_pool_type']}")
    print(f"  head: {' + '.join(rec['head_layers'])}")
    b6 = blocks[6][0]
    print("  blocks[6][0] (final InvertedResidual) modules:")
    for mname, mod in b6.named_modules():
        if mname:
            print(f"    {mname}: {type(mod).__name__} "
                  f"({sum(p.numel() for p in mod.parameters())} params)")
    return rec


# --------------------------------------------------------------------------- #
# Freeze policy
# --------------------------------------------------------------------------- #
def apply_freeze_policy(model: nn.Module) -> Dict:
    """Freeze everything; enable ONLY blocks[6] conv/SE weights + head Linear.

    BN affine stays frozen (conservative) and BN running stats are preserved by
    forcing every _BatchNorm module to eval mode (see bn_eval). Fails if any
    unexpected module is trainable.
    """
    bns = bn_param_names(model)
    for n, p in model.named_parameters():
        p.requires_grad = False

    trainable: List[str] = []
    for n, p in model.named_parameters():
        if n in bns:
            continue
        if n.startswith("0.blocks.6."):
            p.requires_grad = True
            trainable.append(n)
    for n in ("1.1.weight", "1.1.bias"):
        p = model.get_parameter(n)
        p.requires_grad = True
        trainable.append(n)

    # Fail on ANY trainable param outside the allowlist.
    allowed = set(trainable)
    bad = [n for n, p in model.named_parameters()
           if p.requires_grad and n not in allowed]
    if bad:
        raise SystemExit(f"unexpected trainable parameters: {bad}")
    if any(n.startswith("0.blocks.6.") and ".bn" in n for n in trainable):
        raise SystemExit("BN affine in blocks[6] must stay frozen")
    if any("0.blocks.6" not in n and n not in ("1.1.weight", "1.1.bias")
           for n in trainable):
        raise SystemExit(f"trainable set outside blocks[6]+head: {trainable}")

    total = sum(p.numel() for p in model.parameters())
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert total == frozen + tr
    print(f"trainable params: {tr} / {total} ({100.0 * tr / total:.4f}%); "
          f"frozen {frozen}")
    print(f"trainable module names: {[n for n in trainable if n not in ('1.1.weight', '1.1.bias')][:3]}… "
          f"(head: {'1.1.weight','1.1.bias'})")
    return {"total_params": total, "frozen_params": frozen,
            "trainable_params": tr,
            "trainable_fraction": float(tr) / float(total),
            "trainable_names": trainable,
            "bn_affine_frozen": True}


def bn_eval(model: nn.Module) -> None:
    """Force every BatchNorm module to eval mode: running stats stay frozen."""
    for mod in model.modules():
        if is_bn_module(mod):
            mod.eval()


# --------------------------------------------------------------------------- #
# Validation metrics (MPS forward, full val split)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def val_metrics(model: nn.Module, loader: DataLoader, device: str) -> Dict:
    model.eval()
    ys, ps = [], []
    for x, y in loader:
        x = x.to(device)
        ps.append(model(x).argmax(1).cpu().numpy())
        ys.append(y.numpy())
    y_true = np.concatenate(ys)
    y_pred = np.concatenate(ps)
    cm = confusion_matrix(y_true, y_pred, N_CLASSES)
    ev = evaluate_cm(y_true, y_pred)
    sm = s_e_metrics(cm)
    return {
        "cm": cm,
        "accuracy": ev["accuracy"],
        "macro_f1": ev["macro_f1"],
        "total_errors": ev["total_errors"],
        "S_recall": sm["S recall"],
        "S_to_E": sm["S->E"],
        "E_to_S": sm["E->S"],
    }


# --------------------------------------------------------------------------- #
# Candidate training
# --------------------------------------------------------------------------- #
def train_candidate(
    parent: str, train_loader: DataLoader, val_loader: DataLoader,
    class_weights: Sequence[float], backbone_lr: float, head_lr: float, *,
    epochs: int, patience: int, batch_size: int, seed: int, device: str,
) -> Dict:
    model = load_checkpoint(parent, device)
    freeze = apply_freeze_policy(model)
    bn_eval(model)

    block_params = [p for n, p in model.named_parameters()
                    if n.startswith("0.blocks.6.") and p.requires_grad]
    head_params = [p for n, p in model.named_parameters()
                   if n.startswith("1.1.")]
    assert block_params and len(head_params) == 2
    opt = torch.optim.AdamW([
        {"params": block_params, "lr": backbone_lr},
        {"params": head_params, "lr": head_lr},
    ], lr=0.0, weight_decay=WEIGHT_DECAY)

    weight = torch.tensor(class_weights, dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=weight,
                                    label_smoothing=LABEL_SMOOTHING)

    steps_per_epoch = len(train_loader)
    total_steps = epochs * steps_per_epoch
    warmup_steps = WARMUP_EPOCHS * steps_per_epoch
    step = 0
    best_state = None
    best_macro_f1 = -1.0
    best_epoch = 0
    no_improve = 0
    runs: List[Dict] = []
    torch.manual_seed(seed)
    np.random.seed(seed)

    for epoch in range(1, epochs + 1):
        model.train()
        bn_eval(model)
        running = 0.0
        seen = 0
        t0 = time.perf_counter()
        for i, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            lr_b = cosine_warmup_lr(step, total_steps, warmup_steps, backbone_lr)
            lr_h = cosine_warmup_lr(step, total_steps, warmup_steps, head_lr)
            opt.param_groups[0]["lr"] = lr_b
            opt.param_groups[1]["lr"] = lr_h
            opt.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)
            loss.backward()
            opt.step()
            running += float(loss.item())
            seen += 1
            step += 1
            if i % 50 == 49:
                dt = time.perf_counter() - t0
                rate = seen / max(1e-9, dt)
                print(f"  [bb {backbone_lr:.0e}|h {head_lr:.0e}] "
                      f"epoch {epoch}/{epochs} batch {i + 1}/{steps_per_epoch} "
                      f"loss {running / seen:.4f} "
                      f"{rate * batch_size:.0f} img/s", flush=True)

        vm = val_metrics(model, val_loader, device)
        train_loss = running / max(1, seen)
        runs.append({
            "epoch": epoch, "train_loss": train_loss,
            "lr_backbone": backbone_lr, "lr_head": head_lr,
            "val_accuracy": vm["accuracy"], "val_macro_f1": vm["macro_f1"],
            "total_errors": vm["total_errors"], "S_recall": vm["S_recall"],
            "S_to_E": vm["S_to_E"], "E_to_S": vm["E_to_S"],
        })
        improved = vm["macro_f1"] > best_macro_f1 + 1e-9
        if improved:
            best_macro_f1 = vm["macro_f1"]
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
        print(f"  [bb {backbone_lr:.0e}|h {head_lr:.0e}] epoch {epoch:2d} "
              f"loss {train_loss:.4f} acc {vm['accuracy']:.4f} "
              f"macro-F1 {vm['macro_f1']:.4f} S->E {vm['S_to_E']} "
              f"(best ep {best_epoch} @ {best_macro_f1:.4f}) "
              f"epoch {time.perf_counter() - t0:.0f}s", flush=True)
        if no_improve >= patience:
            print(f"  [bb {backbone_lr:.0e}|h {head_lr:.0e}] early stop", flush=True)
            break

    if best_state is None:
        raise SystemExit("no improving epoch for candidate")
    return {
        "lr_backbone": backbone_lr, "lr_head": head_lr,
        "best_epoch": best_epoch, "best_macro_f1": best_macro_f1,
        "epochs_run": len(runs), "runs": runs,
        "state_dict": best_state,
        "freeze": freeze,
    }


def select_best(candidates: Sequence[Dict]) -> Dict:
    best = max(candidates, key=lambda c: c["best_macro_f1"])
    print(f"best candidate: backbone_lr={best['lr_backbone']:.0e} "
          f"head_lr={best['lr_head']:.0e} macro-F1={best['best_macro_f1']:.4f} "
          f"at epoch {best['best_epoch']}")
    return best


# --------------------------------------------------------------------------- #
# Parameter-change verification
# --------------------------------------------------------------------------- #
def verify_param_changes(parent_sd: Dict, candidate_sd: Dict,
                         trainable_names: Sequence[str]) -> Dict:
    """Only the allowlisted params may differ; BN + frozen tensors bit-identical."""
    parent_keys = set(parent_sd)
    cand_keys = set(candidate_sd)
    if parent_keys != cand_keys:
        raise SystemExit("state dict key sets differ — structure changed")

    trainable = set(trainable_names)
    changed = []
    for n in sorted(parent_keys):
        if not torch.equal(parent_sd[n], candidate_sd[n]):
            changed.append(n)
    changed_set = set(changed)

    bn_changed = [n for n in changed if is_bn_tensor(n)
                  or ".running_mean" in n or ".running_var" in n
                  or ".num_batches_tracked" in n]
    if bn_changed:
        raise SystemExit(
            f"BatchNorm tensors changed: {bn_changed} — BN statistics and "
            "affine params must stay frozen"
        )
    unexpected = changed_set - trainable
    if unexpected:
        raise SystemExit(
            f"tensors changed outside the trainable set: {sorted(unexpected)}"
        )
    missing = trainable - changed_set
    if missing:
        raise SystemExit(f"trainable tensors did not change: {sorted(missing)}")

    bn_total = sum(1 for n in parent_keys
                   if ".running_mean" in n or ".running_var" in n
                   or ".num_batches_tracked" in n or is_bn_tensor(n))
    frozen_unchanged = len(parent_keys) - len(changed)
    print(f"param-change check: {len(changed)} tensors changed (all in the "
          f"trainable set), {frozen_unchanged} frozen tensors bit-identical "
          f"({bn_total} BatchNorm tensors all unchanged)")
    return {
        "n_changed": len(changed), "n_unchanged": frozen_unchanged,
        "changed": sorted(changed),
        "n_bn_tensors_total": bn_total,
        "bn_tensors_unchanged": True,
    }


def is_bn_tensor(name: str) -> bool:
    """Heuristic for BN affine params in the state dict (bnN/bn in path)."""
    head = name.split(".")[-2] if "." in name else ""
    return head.startswith("bn") or ".bn" in name or name.startswith("bn")


# --------------------------------------------------------------------------- #
# Reports
# --------------------------------------------------------------------------- #
def _fmt(v):
    try:
        return f"{float(v):.4f}"
    except (TypeError, ValueError):
        return str(v)


def build_markdown_report(report: Dict) -> str:
    lines = [
        "# Partial fine-tuning of efficientnet_b0_targeted — training phase",
        "",
        f"Experiment: image-space training of ONLY the final MBConv stage "
        f"`blocks[6]` (717,232 params, a single InvertedResidual) + the existing "
        f"`Linear(1280, 29)` head. All other backbone tensors and every "
        f"BatchNorm statistic stay bit-identical to the parent.",
        "",
        f"**Selected candidate: backbone_lr={report['selected']['lr_backbone']:.0e}, "
        f"head_lr={report['selected']['lr_head']:.0e}, "
        f"epoch {report['selected']['best_epoch']}, "
        f"val macro-F1 {report['selected']['best_macro_f1']:.4f}**",
        "",
        "## Provenance",
        "",
        f"- Parent: `{report['parent']['path']}` "
        f"(file sha256 `{report['parent']['file_sha256'][:16]}…`, "
        f"backbone sd sha256 `{report['parent']['backbone_state_dict_sha256'][:16]}…`, "
        f"recorded val_acc {report['parent']['recorded_val_acc']:.4f}).",
        f"- Rebalance: {report['parent']['rebalance']}; geometry-safe classes "
        f"{report['parent']['geometry_safe_classes']}; backgrounds "
        f"{report['parent']['backgrounds']}; resumed_from "
        f"{report['parent']['resumed_from']}.",
        f"- Manifest: `{report['manifest']}` "
        f"(`{report['manifest_sha256'][:16]}…`), leakage-safe frame-range split: "
        f"{report['n_train']} train / {report['n_val']} val (600/class).",
        "",
        "## Model structure (inspected)",
        "",
    ]
    rec = report["structure"]
    lines.append(f"- container {rec['container']}; backbone {rec['backbone_type']}; "
                 f"{rec['n_blocks']} MBConv stages.")
    for i, n in enumerate(rec["blocks_params"]):
        star = "  <- TRAINABLE" if i == 6 else ""
        lines.append(f"- blocks[{i}]: {n} params{star}")
    lines += [
        f"- conv_stem {rec['conv_stem_params']} | bn1 {rec['bn1_params']} | "
        f"conv_head {rec['conv_head_params']} | bn2 {rec['bn2_params']} | "
        f"global_pool {rec['global_pool_type']} (all frozen).",
        f"- head: {' + '.join(rec['head_layers'])} (trainable).",
        "",
        "## Freeze policy",
        "",
        f"- Total {report['freeze']['total_params']} params; trainable "
        f"{report['freeze']['trainable_params']} "
        f"({100.0 * report['freeze']['trainable_fraction']:.4f}%); frozen "
        f"{report['freeze']['frozen_params']}.",
        f"- BN affine frozen: {report['freeze']['bn_affine_frozen']}; running "
        f"mean/var frozen (all BatchNorm modules eval); pretrained stats "
        f"preserved.",
        "",
        "## Sanity gate",
        "",
        f"- Raw-image val eval of the parent reproduced the baseline: "
        f"acc {_fmt(report['gate']['accuracy'])} (expected {GATE_ACC:.6f}), "
        f"macro-F1 {_fmt(report['gate']['macro_f1'])} (expected "
        f"{GATE_MACRO_F1:.6f}), total errors {report['gate']['total_errors']} "
        f"(expected {GATE_TOTAL_ERRORS}), S->E {report['gate']['S_to_E']} "
        f"(expected {GATE_S_TO_E}). PASS.",
        "",
        "## Device",
        "",
        f"- torch {report['device']['torch_version']}, MPS available "
        f"{report['device']['mps_available']}; selected {report['device']['device']}.",
        f"- Per-epoch selection uses {report['device']['device']} val macro-F1 "
        f"(full 17,400-frame val). Final reported numbers come from the raw-image "
        f"CPU pipeline in the analysis phase.",
        "",
        "## Sweep",
        "",
        "| backbone_lr | head_lr | best macro-F1 | best epoch | epochs run |",
        "|-------------|---------|---------------|------------|------------|",
    ]
    for c in report["sweep"]["candidates"]:
        lines.append(
            f"| {c['lr_backbone']:.0e} | {c['lr_head']:.0e} | "
            f"{c['best_macro_f1']:.4f} | {c['best_epoch']} | "
            f"{c['epochs_run']} |"
        )
    lines += [
        "",
        "Per-epoch records (train loss / val acc / val macro-F1 / S->E):",
        "",
        "| backbone_lr | head_lr | epoch | train loss | val acc | val macro-F1 | S->E |",
        "|-------------|---------|-------|------------|---------|--------------|------|",
    ]
    for c in report["sweep"]["candidates"]:
        for r in c["runs"]:
            lines.append(
                f"| {c['lr_backbone']:.0e} | {c['lr_head']:.0e} | "
                f"{r['epoch']} | {r['train_loss']:.4f} | {r['val_accuracy']:.4f} | "
                f"{r['val_macro_f1']:.4f} | {r['S_to_E']} |"
            )
    lines += [
        "",
        "## Parameter-change verification",
        "",
        f"- Candidate vs parent: {report['param_change']['n_changed']} tensors "
        f"changed (all in the trainable set `blocks[6]` conv/SE + `1.1` head); "
        f"{report['param_change']['n_unchanged']} tensors bit-identical "
        f"(stem, blocks[0..5], conv_head, bn2, all BatchNorm tensors).",
        f"- BatchNorm tensors unchanged: "
        f"{report['param_change']['bn_tensors_unchanged']}.",
        "",
        "## Artifacts (training phase)",
        "",
        f"- Candidate state dict: `{report['artifacts']['candidate_out']}` "
        f"(file sha256 `{report['artifacts']['candidate_sha256'][:16]}…`)",
        f"- Training report: `{report['artifacts']['report_json']}`",
        f"- This markdown: `{report['artifacts']['report_md']}`",
        "",
        "## Next phase",
        "",
        "- Run `scripts/analyze_partial_finetune.py` for the final raw-image "
        "eval, representation-change diagnostics, three-way comparison "
        "(targeted / head-only / partial-FT), verdict, and the deployable "
        "checkpoint.",
        "",
    ]
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
# run
# --------------------------------------------------------------------------- #
def run(args) -> Dict:
    device = _resolve_device(args.device)
    print(f"torch {torch.__version__}, MPS available "
          f"{torch.backends.mps.is_available()}, selected device {device}",
          flush=True)
    if device not in ("mps", "cuda") and args.device == "auto":
        raise SystemExit(
            "training requires MPS (or CUDA); refusing to silently train on CPU"
        )

    manifest = SplitManifest.from_json(args.manifest)
    mver = verify_manifest(args.manifest, manifest)
    parent = verify_parent(args.checkpoint)

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    class_weights = list(ckpt["class_weights"])
    parent_sd = ckpt["model"]
    del ckpt

    # ---- sanity gate: raw-image val eval must reproduce the baseline ----
    gate = json.loads(Path(args.gate_report).read_text())
    gcm = np.asarray(gate["confusion_matrix"], dtype=np.int64)
    g_s_e = int(gcm[CLASSES.index("S"), CLASSES.index("E")])
    g_errors = int(gcm.sum() - gcm.trace())
    ok = (abs(gate["overall_accuracy"] - GATE_ACC) < 1e-9
          and abs(gate["macro_f1"] - GATE_MACRO_F1) < 1e-9
          and g_errors == GATE_TOTAL_ERRORS and g_s_e == GATE_S_TO_E)
    if not ok:
        raise SystemExit(
            f"SANITY GATE FAILED: raw-image eval of {args.checkpoint} gave "
            f"acc {gate['overall_accuracy']:.6f}, macro-F1 {gate['macro_f1']:.6f}, "
            f"errors {g_errors}, S->E {g_s_e} — expected "
            f"{GATE_ACC:.6f}/{GATE_MACRO_F1:.6f}/{GATE_TOTAL_ERRORS}/"
            f"{GATE_S_TO_E}. STOP before training."
        )
    gate_metrics = {"accuracy": gate["overall_accuracy"],
                    "macro_f1": gate["macro_f1"],
                    "total_errors": g_errors, "S_to_E": g_s_e}

    model = load_checkpoint(args.checkpoint, "cpu")
    structure = inspect_structure(model)
    del model
    print(f"SANITY GATE PASSED: raw-image eval of parent reproduced acc "
          f"{gate_metrics['accuracy']:.4f}, macro-F1 {gate_metrics['macro_f1']:.4f}, "
          f"errors {g_errors}, S->E {g_s_e}", flush=True)

    train_ds, val_ds = build_datasets(
        manifest, backgrounds_dir=None,
        geometry_safe_classes=parent["geometry_safe_classes"],
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=0, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=0)
    print(f"train={len(train_ds)} val={len(val_ds)} "
          f"(geometry-safe for {list(parent['geometry_safe_classes'])})",
          flush=True)

    if len(args.backbone_lrs) != len(args.head_lrs):
        raise SystemExit("--backbone-lrs and --head-lrs must be same length")
    pairs = list(zip(args.backbone_lrs, args.head_lrs))

    candidates: List[Dict] = []
    for bb_lr, h_lr in pairs:
        print(f"=== candidate backbone_lr={bb_lr:.0e} head_lr={h_lr:.0e} ===",
              flush=True)
        cand = train_candidate(
            args.checkpoint, train_loader, val_loader, class_weights,
            bb_lr, h_lr, epochs=args.epochs, patience=args.patience,
            batch_size=args.batch_size, seed=args.seed, device=device,
        )
        candidates.append(cand)

    best = select_best(candidates)

    cand_sd = best["state_dict"]
    verify = verify_param_changes(parent_sd, cand_sd, best["freeze"]["trainable_names"])

    os.makedirs(os.path.dirname(args.candidate_out) or ".", exist_ok=True)
    torch.save({
        "schema": "asl-partial-ft-candidate/v1",
        "parent_checkpoint": args.checkpoint,
        "parent_checkpoint_sha256": parent["file_sha256"],
        "parent_backbone_state_dict_sha256": parent["backbone_state_dict_sha256"],
        "lr_backbone": best["lr_backbone"], "lr_head": best["lr_head"],
        "best_epoch": best["best_epoch"],
        "best_macro_f1": best["best_macro_f1"],
        "epochs_run": best["epochs_run"],
        "seed": args.seed,
        "freeze": {k: v for k, v in best["freeze"].items()
                   if k != "trainable_names"},
        "trainable_names": best["freeze"]["trainable_names"],
        "model": cand_sd,
        "note": "training-phase output; final eval + verdict in the analyze phase",
    }, args.candidate_out)
    cand_sha = file_sha256(args.candidate_out)
    print(f"wrote {args.candidate_out} (sha {cand_sha[:16]}…)", flush=True)

    report = {
        "schema": "asl-partial-ft-training/v1",
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "parent": parent,
        "manifest": args.manifest,
        "manifest_sha256": mver["manifest_sha256"],
        "n_train": mver["n_train"], "n_val": mver["n_val"],
        "structure": structure,
        "freeze": {k: v for k, v in best["freeze"].items()
                   if k != "trainable_names"},
        "gate": gate_metrics,
        "gate_report": args.gate_report,
        "device": {
            "torch_version": torch.__version__,
            "mps_available": torch.backends.mps.is_available(),
            "device": device,
        },
        "loss": {
            "criterion": "CrossEntropyLoss",
            "label_smoothing": LABEL_SMOOTHING,
            "class_weights_source": "parent checkpoint (rebalance: loss)",
            "optimizer": "AdamW", "weight_decay": WEIGHT_DECAY,
            "schedule": "cosine warmup (1 epoch), mirror of src/train.py",
        },
        "training": {
            "seed": args.seed, "epochs_max": args.epochs,
            "patience": args.patience, "batch_size": args.batch_size,
            "device": device,
            "data_contract": (
                "same manifest + split as parent; geometry_safe_classes = "
                f"{parent['geometry_safe_classes']}; backgrounds=None; "
                "IMAGE-space training (fresh augmentation per epoch)"
            ),
        },
        "sweep": {
            "pairs": [[a, b] for a, b in pairs],
            "candidates": [
                {k: v for k, v in c.items() if k != "state_dict"}
                for c in candidates
            ],
        },
        "selected": {
            "lr_backbone": best["lr_backbone"], "lr_head": best["lr_head"],
            "best_epoch": best["best_epoch"],
            "best_macro_f1": best["best_macro_f1"],
            "epochs_run": best["epochs_run"],
        },
        "param_change": verify,
        "artifacts": {
            "candidate_out": args.candidate_out,
            "candidate_sha256": cand_sha,
            "report_json": args.train_json,
            "report_md": args.train_md,
        },
    }
    Path(args.train_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.train_json).write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {args.train_json}")
    Path(args.train_md).parent.mkdir(parents=True, exist_ok=True)
    Path(args.train_md).write_text(build_markdown_report(report))
    print(f"wrote {args.train_md}")
    return report


def main() -> None:
    p = argparse.ArgumentParser(
        description="Partial fine-tuning (final block + head) of the targeted "
                    "checkpoint — training phase."
    )
    p.add_argument("--checkpoint", default=DEFAULT_PARENT)
    p.add_argument("--manifest", default=DEFAULT_MANIFEST)
    p.add_argument("--gate-report", default=DEFAULT_GATE)
    p.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"],
                   default="auto")
    p.add_argument("--backbone-lrs", type=float, nargs="+",
                   default=[3e-6, 1e-5, 3e-5])
    p.add_argument("--head-lrs", type=float, nargs="+",
                   default=[3e-5, 1e-4, 3e-4])
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--patience", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--seed", type=int, default=20260810)
    p.add_argument("--candidate-out", default=DEFAULT_CANDIDATE_OUT)
    p.add_argument("--train-json", default=DEFAULT_TRAIN_JSON)
    p.add_argument("--train-md", default=DEFAULT_TRAIN_MD)
    run(p.parse_args())


if __name__ == "__main__":
    main()
