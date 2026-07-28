"""Export a trained checkpoint to TorchScript and/or ONNX for CPU demo inference.

The webcam demo runs on CPU. Exporting to TorchScript (or ONNX) gives a portable,
dependency-light artifact and a small latency win. The exported graph expects the
SAME normalized C×H×W input produced by ``src/crop.preprocess_bgr`` — the export
does not bake in preprocessing, so the app must keep using the shared contract.
"""

from __future__ import annotations

import argparse
import os

import torch

from .evaluate import load_checkpoint


def export_torchscript(model: torch.nn.Module, out: str) -> None:
    model.eval()
    example = torch.zeros(1, 3, 224, 224)
    traced = torch.jit.trace(model, example)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    traced.save(out)
    print(f"wrote TorchScript -> {out}")


def export_onnx(model: torch.nn.Module, out: str, opset: int = 17) -> None:
    model.eval()
    example = torch.zeros(1, 3, 224, 224)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    torch.onnx.export(
        model, example, out,
        input_names=["input"], output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=opset,
    )
    print(f"wrote ONNX -> {out}")


def main() -> None:
    p = argparse.ArgumentParser(description="Export a checkpoint for CPU inference.")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--format", choices=["torchscript", "onnx", "both"], default="both")
    p.add_argument("--out-dir", default="checkpoints")
    args = p.parse_args()

    model = load_checkpoint(args.checkpoint, device="cpu")
    if args.format in ("torchscript", "both"):
        export_torchscript(model, os.path.join(args.out_dir, "model.ts"))
    if args.format in ("onnx", "both"):
        export_onnx(model, os.path.join(args.out_dir, "model.onnx"))


if __name__ == "__main__":
    main()
