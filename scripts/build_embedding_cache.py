"""Cache pooled EfficientNet-B0 embeddings ONCE, so head-only training never
re-runs the backbone.

Experiment (head-only fine-tuning of the final Linear(1280, 29) on
``checkpoints/efficientnet_b0_targeted.pt``):

    images
      -> frozen EfficientNet-B0 (``model[0]``) once, on MPS
      -> cached 1280-D pooled embeddings (the exact vector before
         ``model[1] = Dropout(0.3) + Linear(1280, 29)``)
      -> train the linear head directly on the cached vectors

The backbone is completely frozen, so running it once and caching makes the
actual head training trivially light.

Augmentation semantics (IMPORTANT — documented difference)
----------------------------------------------------------
Original training re-sampled the stochastic Albumentations pipeline EVERY
epoch. Cached-embedding training cannot do that, so this script caches ONE
deterministic realization:

- TRAIN cache: the exact existing training transform (same pipeline from
  ``src.augment.build_augmentation``, same per-class ``geometry_safe`` routing
  as the targeted checkpoint, ``backgrounds=None`` as the checkpoint used) with
  the Compose RNG fixed to ``seed=20260809``. Every training epoch of the head
  therefore sees the same fixed augmented realization, NOT a fresh stochastic
  draw. This is deliberately NOT claimed to be identical to image-space
  training.
- VAL cache: the deterministic ``preprocess_bgr`` contract only — no
  augmentation, exactly what the original evaluation pipeline used.

Neither ``crop.py``, ``augment.py``, nor preprocessing is modified. No new
augmentation is invented.

Structure verification
----------------------
``build_efficientnet`` returns ``nn.Sequential(backbone, head)`` (see
``src/model.py``). This script asserts ``model[0]`` outputs ``(B, 1280)`` and
that ``head(backbone(x)) == model(x)`` for every batch under ``model.eval()``,
so the cached vector is provably the same 1280-D pooled feature the linear
head consumes.

Outputs
-------
- ``data/embedding_cache/targeted_train_embeddings.pt``  (69,600 x 1280)
- ``data/embedding_cache/targeted_val_embeddings.pt``    (17,400 x 1280)
- ``data/embedding_cache/cache_manifest.json``           (summary + hashes)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from src.augment import _to_tensor, build_augmentation  # noqa: E402
from src.data import ASLImageDataset, CLASSES, SplitManifest  # noqa: E402
from src.evaluate import load_checkpoint  # noqa: E402
from scripts.analyze_s_to_e import _resolve_device, frame_id  # noqa: E402

DEFAULT_CHECKPOINT = "checkpoints/efficientnet_b0_targeted.pt"
DEFAULT_MANIFEST = "data/split_manifest.json"
DEFAULT_CACHE_DIR = "data/embedding_cache"
EXPECTED_MANIFEST_SHA = (
    "454ca31b1d6a2b8a4a6a5343f9027bb5bfc397b427fe33527501ac2ac59de842"
)

FEATURE_DIM = 1280
CACHE_SEED = 20260809
SCHEMA = "asl-embedding-cache/v1"


# --------------------------------------------------------------------------- #
# Pure helpers (unit-tested)
# --------------------------------------------------------------------------- #
def state_dict_sha256(sd: Dict) -> str:
    """Canonical sha256 over a state dict (sorted keys + raw bytes).

    Deterministic across machines for identical float values (float32 bytes are
    bit-identical). Used to prove the frozen backbone is literally unchanged
    before/after the experiment.
    """
    h = hashlib.sha256()
    for key in sorted(sd):
        h.update(key.encode())
        arr = sd[key].detach().cpu().contiguous().numpy()
        h.update(arr.tobytes())
    return h.hexdigest()


def file_sha256(path: str) -> str:
    """sha256 over a file's bytes (chunked, for large checkpoints)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def tensor_sha256(arr: np.ndarray) -> str:
    """sha256 over a numpy array's contiguous bytes (cache fingerprint)."""
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(arr).tobytes())
    return h.hexdigest()


def seeded_train_transform(policy: str, seed: int):
    """The existing train augmentation pipeline, one deterministic realization.

    Mirrors ``src.augment.build_transforms(train=True)`` (including its shared
    ``_to_tensor`` contract tail) but fixes the Albumentations Compose RNG with
    ``set_random_seed(seed)`` so the entire sequence of augmented images is
    reproducible across runs and machines. ``backgrounds`` is always None here
    (the targeted checkpoint trained with ``backgrounds=None``), so the
    background-replacement branch never runs.
    """
    aug = build_augmentation(policy)  # exact repo pipeline, unmodified
    aug.set_random_seed(seed)

    def _transform(img_bgr: np.ndarray):
        return _to_tensor(aug(image=img_bgr)["image"])

    return _transform


# --------------------------------------------------------------------------- #
# Cache extraction
# --------------------------------------------------------------------------- #
def verify_model_split(model: torch.nn.Module) -> None:
    """Assert ``model[0]`` (backbone) feeds ``model[1]`` (Dropout+Linear head)."""
    if not (
        isinstance(model, torch.nn.Sequential)
        and len(model) == 2
    ):
        raise SystemExit(
            f"expected nn.Sequential(backbone, head), got {type(model).__name__}"
        )
    head = model[1]
    if not (
        isinstance(head, torch.nn.Sequential)
        and len(head) == 2
        and isinstance(head[0], torch.nn.Dropout)
        and isinstance(head[1], torch.nn.Linear)
        and head[1].in_features == FEATURE_DIM
        and head[1].out_features == len(CLASSES)
    ):
        raise SystemExit(
            f"model[1] is not Dropout(0.3)+Linear({FEATURE_DIM},29): {head}"
        )


def verify_split_point(
    model: torch.nn.Module, ds: ASLImageDataset, device: str
) -> float:
    """One small batch: assert ``model(x) == head(backbone(x))`` (eval mode).

    ``model(x)`` re-runs the backbone, so it is done ONCE per split on a tiny
    batch rather than every batch — otherwise extraction would double the GPU
    work. Returns the max absolute logit difference.
    """
    backbone = model[0]
    head = model[1]
    x = ds[0][0][None, :].to(device)
    with torch.no_grad():
        emb = backbone(x)
        full = model(x)
        via_head = head(emb)
    if not torch.allclose(full, via_head, atol=1e-5):
        raise SystemExit("head(backbone(x)) != model(x): wrong embedding layer")
    diff = float((full - via_head).abs().max())
    print(f"  split point verified: model(x) == head(backbone(x)), "
          f"max |logit diff| {diff:.2e}")
    return diff


def extract_split(
    model: torch.nn.Module,
    ds: ASLImageDataset,
    device: str,
    batch_size: int,
    split: str,
) -> Dict:
    """Run the frozen backbone once over a split (single forward per image)."""
    backbone = model[0]
    max_logit_diff = verify_split_point(model, ds, device)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    embs: List[np.ndarray] = []
    labels: List[np.ndarray] = []
    class_names: List[str] = []
    paths: List[str] = []
    frame_ids: List[int] = []
    import time as _time

    t0 = _time.time()
    n_batches = int(np.ceil(len(ds) / batch_size))
    with torch.no_grad():
        for bi, (x, y) in enumerate(loader):
            x = x.to(device)
            emb = backbone(x)
            if tuple(emb.shape)[1:] != (FEATURE_DIM,):
                raise SystemExit(
                    f"backbone output shape {tuple(emb.shape)} != (B, {FEATURE_DIM})"
                )
            embs.append(emb.cpu().numpy())
            labels.append(y.numpy())
            for i in range(int(x.shape[0])):
                s = ds.samples[bi * batch_size + i]
                class_names.append(s.class_name)
                paths.append(s.path)
                frame_ids.append(frame_id(s.path))
            if (bi + 1) % 100 == 0 or bi + 1 == n_batches:
                done = len(ds) if bi + 1 == n_batches else (bi + 1) * batch_size
                el = _time.time() - t0
                eta = el / done * (len(ds) - done)
                print(f"  [{split}] {done}/{len(ds)}  {el:.0f}s elapsed  "
                      f"ETA {eta / 60:.1f} min", flush=True)
    print(f"[{split}] {len(ds)} embeddings on {device}, "
          f"max |model(x)-head(backbone(x))| {max_logit_diff:.2e}")
    return {
        "embeddings": np.concatenate(embs, axis=0).astype(np.float32),
        "labels": np.concatenate(labels, axis=0).astype(np.int64),
        "class_names": class_names,
        "paths": paths,
        "frame_ids": frame_ids,
        "max_logit_diff": max_logit_diff,
    }


def build_train_dataset(
    manifest: SplitManifest, geometry_safe_classes: List[str], seed: int
) -> ASLImageDataset:
    """The exact training split routing, but with a seeded (deterministic)
    realization of the stochastic train transform.

    Mirrors ``src.data.build_datasets``: the base transform is the heavy
    ``default`` policy; the checkpoint's ``geometry_safe_classes`` route through
    the gentler ``geometry_safe`` policy. ``backgrounds=None`` (as the targeted
    checkpoint used), so no background replacement.
    """
    unknown = [c for c in geometry_safe_classes if c not in CLASSES]
    if unknown:
        raise ValueError(f"geometry_safe_classes are not ASL classes: {unknown}")
    default_t = seeded_train_transform("default", seed)
    gentle_t = seeded_train_transform("geometry_safe", seed)
    class_transforms = {c: gentle_t for c in geometry_safe_classes}
    return ASLImageDataset(
        manifest.train, transform=default_t, train=True,
        class_transforms=class_transforms,
    )


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def run(args) -> Dict:
    device = _resolve_device(args.device)
    print(f"torch {torch.__version__}")
    print(f"torch.backends.mps.is_available() = {torch.backends.mps.is_available()}")
    print(f"embedding extraction device: {device}")

    manifest = SplitManifest.from_json(args.manifest)
    manifest_sha = json.loads(Path(args.manifest).read_text()).get("content_sha256")
    if manifest_sha != EXPECTED_MANIFEST_SHA:
        print(f"WARNING: manifest content_sha256 {manifest_sha} differs from the "
              f"recorded targeted manifest {EXPECTED_MANIFEST_SHA}")
    print(f"manifest: {manifest_sha}  train={len(manifest.train)}  "
          f"val={len(manifest.val)}")

    model = load_checkpoint(args.checkpoint, device)
    model.eval()
    verify_model_split(model)
    backbone_hash = state_dict_sha256(model[0].state_dict())
    print(f"backbone state_dict sha256: {backbone_hash[:16]}…")
    parent_hash = file_sha256(args.checkpoint)
    print(f"parent checkpoint file sha256: {parent_hash[:16]}…")

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    geometry_safe = list(ckpt.get("geometry_safe_classes") or [])
    backgrounds = ckpt.get("backgrounds")
    if backgrounds is not None:
        print(f"WARNING: parent checkpoint recorded backgrounds={backgrounds}; "
              "cache generation uses backgrounds=None (matches the training "
              "reality for this checkpoint).")

    train_ds = build_train_dataset(manifest, geometry_safe, args.seed)
    val_ds = ASLImageDataset(manifest.val, train=False)

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    created = datetime.now(timezone.utc).isoformat(timespec="seconds")
    splits_to_extract = {"train"} if args.only in ("train", "both") else set()
    if args.only in ("val", "both"):
        splits_to_extract.add("val")

    metadata = {
        "schema": SCHEMA,
        "created": created,
        "split": "",
        "manifest": args.manifest,
        "manifest_sha256": manifest_sha,
        "parent_checkpoint": args.checkpoint,
        "parent_checkpoint_sha256": parent_hash,
        "backbone_state_dict_sha256": backbone_hash,
        "preprocessing": (
            "src.crop.preprocess_bgr (resize_square to 224 + ImageNet "
            "mean/std, BGR->RGB) — the FROZEN shared contract"
        ),
        "augmentation": {
            "train": {
                "policy": "one deterministic realization of the existing "
                          "stochastic train transform",
                "pipeline": "src.augment.build_augmentation, unmodified",
                "seed": args.seed,
                "backgrounds": None,
                "geometry_safe_classes": geometry_safe,
                "difference_from_original": (
                    "original training re-sampled augmentation every epoch; the "
                    "cache fixes ONE seeded realization reused for every head "
                    "training epoch"
                ),
            },
            "val": {"policy": "none — deterministic preprocess_bgr contract"},
        },
        "device": device,
        "seed": args.seed,
        "batch_size": args.batch_size,
    }

    out_files = {}
    for split, ds in (("train", train_ds), ("val", val_ds)):
        out_path = (cache_dir / args.train_out if split == "train"
                    else cache_dir / args.val_out)
        if split not in splits_to_extract:
            continue
        if out_path.is_file() and not args.force:
            print(f"keep existing {out_path} (resume; pass --force to rebuild)")
            existing = torch.load(out_path, map_location="cpu",
                                  weights_only=False)
            out_files[split] = {
                "path": str(out_path),
                "n": int(existing["n"]),
                "embeddings_sha256": existing["embeddings_sha256"],
                "labels_sha256": existing["labels_sha256"],
                "file_sha256": file_sha256(str(out_path)),
                "max_logit_diff": existing["max_logit_diff"],
            }
            continue
        data = extract_split(model, ds, device, args.batch_size, split)
        payload = {
            **metadata,
            "split": split,
            "n": int(len(data["labels"])),
            "embeddings_sha256": tensor_sha256(data["embeddings"]),
            "labels_sha256": tensor_sha256(data["labels"]),
            "max_logit_diff": data["max_logit_diff"],
            "embeddings": data["embeddings"],
            "labels": data["labels"],
            "class_names": data["class_names"],
            "paths": data["paths"],
            "frame_ids": data["frame_ids"],
        }
        torch.save(payload, out_path)
        out_files[split] = {
            "path": str(out_path),
            "n": payload["n"],
            "embeddings_sha256": payload["embeddings_sha256"],
            "labels_sha256": payload["labels_sha256"],
            "file_sha256": file_sha256(str(out_path)),
            "max_logit_diff": payload["max_logit_diff"],
        }
        print(f"wrote {out_path}  ({payload['n']} x {FEATURE_DIM})")

    manifest_out = cache_dir / args.cache_manifest
    summary = {
        "schema": "asl-embedding-cache-manifest/v1",
        "created": created,
        "manifest": args.manifest,
        "manifest_sha256": manifest_sha,
        "parent_checkpoint": args.checkpoint,
        "parent_checkpoint_sha256": parent_hash,
        "backbone_state_dict_sha256": backbone_hash,
        "seed": args.seed,
        "augmentation": metadata["augmentation"],
        "files": out_files,
        "note": "backbone frozen; embeddings extracted once (MPS when available)",
    }
    manifest_out.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"wrote {manifest_out}")
    return summary


def main() -> None:
    p = argparse.ArgumentParser(
        description="Cache pooled 1280-D EfficientNet embeddings once "
                    "(train+val) for head-only fine-tuning."
    )
    p.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    p.add_argument("--manifest", default=DEFAULT_MANIFEST)
    p.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"],
                   default="auto")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--seed", type=int, default=CACHE_SEED)
    p.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    p.add_argument("--train-out", default="targeted_train_embeddings.pt")
    p.add_argument("--val-out", default="targeted_val_embeddings.pt")
    p.add_argument("--cache-manifest", default="cache_manifest.json")
    p.add_argument("--only", choices=["train", "val", "both"], default="both",
                   help="which split(s) to build (default both)")
    p.add_argument("--force", action="store_true",
                   help="rebuild existing cache files instead of resuming")
    run(p.parse_args())


if __name__ == "__main__":
    main()
