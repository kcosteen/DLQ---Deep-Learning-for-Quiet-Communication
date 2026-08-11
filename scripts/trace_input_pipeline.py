"""Trace exactly what a trained checkpoint receives at inference time.

Rebuilds the same val data path that ``src/evaluate.py`` uses and prints the
runtime properties of one real batch — shape, dtype, min/max/mean/std — plus the
label and source file path of the first samples, and reconstructs the first
tensor back into a viewable image so the channel order can be eyeballed.

This is a diagnostic only:

* it does NOT modify the tensor,
* it does NOT touch ``src/crop.py``, preprocessing, or any training code,
* it does NOT write into normal training logs.

Optional ``--checkpoint`` registers a ``forward_pre_hook`` on the model and
prints the identical stats at the exact point the tensor enters the network,
which proves whether the model wrapper performs any hidden preprocessing
(``efficientnet`` does not; ``efficientnet_lift`` runs an ``ExposureLift``).

Usage::

    python scripts/trace_input_pipeline.py
    python scripts/trace_input_pipeline.py --checkpoint checkpoints/efficientnet_b0_targeted.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from src.crop import IMAGENET_MEAN, IMAGENET_STD  # noqa: E402
from src.data import ASLImageDataset, load_or_derive_split  # noqa: E402

DEFAULT_SPLIT = "data/raw/asl_alphabet_train"
DEFAULT_MANIFEST = "data/split_manifest.json"


def describe_batch(x) -> Dict:
    """Runtime properties of a normalized batch, as a JSON-serialisable dict."""
    if not isinstance(x, torch.Tensor) or x.dim() != 4:
        raise ValueError(f"expected a NCHW tensor, got {type(x).__name__} {getattr(x, 'shape', None)}")
    xf = x.float()
    per_channel = {
        f"ch{i}": {
            "min": float(xf[:, i].min()),
            "max": float(xf[:, i].max()),
            "mean": float(xf[:, i].mean()),
            "std": float(xf[:, i].std()),
        }
        for i in range(xf.shape[1])
    }
    return {
        "shape": tuple(x.shape),
        "dtype": str(x.dtype),
        "min": float(xf.min()),
        "max": float(xf.max()),
        "mean": float(xf.mean()),
        "std": float(xf.std()),
        "per_channel": per_channel,
    }


def denormalize(
    x, mean: Sequence[float] = IMAGENET_MEAN, std: Sequence[float] = IMAGENET_STD
) -> np.ndarray:
    """Undo ``crop.normalize_chw`` for one C×H×W tensor -> H×W×3 RGB uint8."""
    rgb = x.detach().cpu().float().numpy().transpose(1, 2, 0)
    rgb = rgb * np.asarray(std, dtype=np.float32) + np.asarray(mean, dtype=np.float32)
    return (np.clip(rgb, 0.0, 1.0) * 255).astype(np.uint8)


def _print_stats(title: str, stats: Dict) -> None:
    pc = stats["per_channel"]
    print(f"{title}: shape={stats['shape']} dtype={stats['dtype']}")
    print(f"  min={stats['min']:.4f} max={stats['max']:.4f} "
          f"mean={stats['mean']:.4f} std={stats['std']:.4f}")
    for i, (ch, s) in enumerate(pc.items()):
        print(f"  {ch}: min={s['min']:.4f} max={s['max']:.4f} "
              f"mean={s['mean']:.4f} std={s['std']:.4f}")


def _sample_info(ds, n: int) -> str:
    lines = []
    for i in range(min(n, len(ds))):
        s = ds.samples[i]
        lines.append(f"  label={s.label} ({s.class_name})  {s.path}")
    return "\n".join(lines)


def trace(args) -> None:
    manifest = load_or_derive_split(args.manifest, args.split, args.train_frac)
    ds = ASLImageDataset(manifest.val, train=False)
    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=False, num_workers=0
    )

    x, y = next(iter(loader))
    stats = describe_batch(x)
    print(f"val split: {len(ds)} images (leakage-safe frame-range split)")
    print(f"val transform: build_transforms(train=False) = preprocess_bgr only")
    _print_stats("batch before model", stats)
    print(f"labels in first batch: {y.tolist()}")
    print(f"first {args.sample_info} samples (source files):")
    print(_sample_info(ds, args.sample_info))

    out = Path(args.out_png)
    out.parent.mkdir(parents=True, exist_ok=True)
    rgb = denormalize(x[0])
    cv2.imwrite(str(out), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    cv2.imwrite(
        str(out.with_name(out.stem + "_channels_swapped" + out.suffix)),
        cv2.cvtColor(rgb[:, :, ::-1], cv2.COLOR_RGB2BGR),
    )
    print(f"reconstructed first tensor -> {out}")
    print(f"channel-order probe (RGB assumed) -> "
          f"{out.with_name(out.stem + '_channels_swapped' + out.suffix)}")

    if args.checkpoint:
        import torch

        from src.evaluate import resolve_arch
        from src.model import build_model

        ckpt = torch.load(args.checkpoint, map_location="cpu")
        arch = resolve_arch(ckpt if isinstance(ckpt, dict) else {}, None)
        model = build_model(arch, pretrained=False)
        model.load_state_dict(
            ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        )
        model.eval()

        hook_stats: Dict = {}

        def _hook(module, args_):
            hook_stats.update(describe_batch(args_[0]))

        handle = model.register_forward_pre_hook(_hook)
        try:
            with torch.no_grad():
                model(x)
        finally:
            handle.remove()
        print(f"\ncheckpoint: {args.checkpoint}  arch={arch}")
        _print_stats("tensor at model forward_pre_hook", hook_stats)
        same = all(
            abs(float(hook_stats[k]) - float(stats[k])) < 1e-6
            for k in ("min", "max", "mean", "std")
        )
        print("hidden preprocessing in model wrapper: "
              + ("NONE (hook input == loader output)" if same
                 else "DETECTED (loader output was transformed)"))
        if arch == "efficientnet_lift":
            print("note: arch is efficientnet_lift — an ExposureLift runs INSIDE "
                  "the model, after this hook.")


def main() -> None:
    p = argparse.ArgumentParser(description="Trace the val input pipeline at runtime.")
    p.add_argument("--split", default=DEFAULT_SPLIT,
                   help="asl_alphabet_train/ dir (default: %(default)s)")
    p.add_argument("--manifest", default=DEFAULT_MANIFEST,
                   help="split manifest; pass --manifest None to derive")
    p.add_argument("--train-frac", type=float, default=0.8)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--checkpoint", default=None,
                   help="checkpoint to attach a forward_pre_hook to (optional)")
    p.add_argument("--out-png", default="docs/figures/model_input_pipeline_probe.png")
    p.add_argument("--sample-info", type=int, default=3,
                   help="how many source-file rows to print (default: %(default)s)")
    args = p.parse_args()
    if str(args.manifest).lower() == "none":
        args.manifest = None
    if not Path(args.split).is_dir():
        raise SystemExit(f"--split {args.split} is not a directory")
    if args.manifest is not None and not Path(args.manifest).is_file():
        print(f"warning: --manifest {args.manifest} not found; deriving the split "
              f"(deterministic given root + train_frac)")
        args.manifest = None
    trace(args)


if __name__ == "__main__":
    main()
