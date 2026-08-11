"""Diagnose the ``S -> E`` confusion of ``efficientnet_b0_targeted.pt``.

This is a **diagnosis only** script: no training, no fine-tuning, no changes to
``src/crop.py`` / preprocessing / augmentation, no exposure correction, no
MediaPipe, no checkpoint changes. It only *reads* the existing pipeline and the
existing checkpoint.

What it does:

1. Runs inference over the exact val split used for
   ``docs/reports/dev_val_targeted_current.json`` — the same
   ``data/split_manifest.json`` (content-sha verified) and the same
   ``ASLImageDataset(val, train=False)`` (``preprocess_bgr`` contract) — and
   records softmax probabilities (not just argmax) for every ``S`` val sample.
2. Verifies the ``S -> E`` count is exactly 138, as in the baseline report.
3. Writes four 5x5 contact sheets into
   ``docs/figures/error_gallery_targeted/S_to_E/``:
   A highest-``P(E)``, B smallest positive ``E-S`` margin, C deterministic random
   errors, D deterministic random correct-``S`` control. Tiles show the actual
   model input (``resize_square`` -> 224x224) with an ``S -> pred | E | S |
   margin | file`` annotation.
4. Runs contiguous-run clustering over the error frame IDs.
5. Writes ``docs/reports/s_to_e_error_analysis_targeted.json`` and
   ``docs/reports/s_to_e_visual_notes.md``.

The pure helpers (``frame_id``, ``top_k``, ``boundary_k``, ``random_k``,
``contiguous_runs``, ``run_stats``) are unit-tested in
``tests/test_analyze_s_to_e.py``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Sequence, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from src.crop import IMG_SIZE, resize_square  # noqa: E402
from src.data import ASLImageDataset, CLASSES, SplitManifest  # noqa: E402
from src.evaluate import load_checkpoint  # noqa: E402

DEFAULT_CHECKPOINT = "checkpoints/efficientnet_b0_targeted.pt"
DEFAULT_MANIFEST = "data/split_manifest.json"
DEFAULT_SPLIT = "data/raw/asl_alphabet_train"
DEFAULT_GALLERY_DIR = "docs/figures/error_gallery_targeted/S_to_E"
DEFAULT_OUT_JSON = "docs/reports/s_to_e_error_analysis_targeted.json"
DEFAULT_OUT_MD = "docs/reports/s_to_e_visual_notes.md"
DEFAULT_BASELINE_REPORT = "docs/reports/dev_val_targeted_current.json"
GALLERY_K = 25
RANDOM_SEED = 42

SCHEMA = "asl-s_to_e-error-analysis/v1"


# --------------------------------------------------------------------------- #
# Pure helpers (unit-tested)
# --------------------------------------------------------------------------- #
def frame_id(path: str) -> int:
    """Numeric suffix of a Kaggle filename: ``S2451.jpg`` -> ``2451``."""
    digits = "".join(ch for ch in Path(path).stem if ch.isdigit())
    if not digits:
        raise ValueError(f"no numeric frame id in {path!r}")
    return int(digits)


def top_k(records: Sequence, k: int, key: Callable) -> List:
    """The ``k`` records with the largest ``key`` value, descending (stable)."""
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    return sorted(records, key=key, reverse=True)[:k]


def boundary_k(records: Sequence, k: int, key: Callable) -> List:
    """The ``k`` records with the smallest *positive* ``key``, ascending.

    Zero and negative values are excluded (an error's margin is positive by
    definition of argmax; a zero margin is not a clear error). Stable sort keeps
    the input (frame) order among ties, so output is deterministic.
    """
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    return sorted((r for r in records if key(r) > 0), key=key)[:k]


def random_k(records: Sequence, k: int, seed: int) -> List:
    """A deterministic subsample of ``k`` records via a seeded generator."""
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    rng = np.random.default_rng(seed)
    size = min(k, len(records))
    idx = rng.choice(len(records), size=size, replace=False)
    return [records[i] for i in idx]


def contiguous_runs(ids: Iterable[int]) -> List[Tuple[int, int]]:
    """Inclusive ``(start, end)`` runs of consecutive integers, sorted by start.

    Duplicate ids are deduped (frame ids are unique per file, but the helper is
    safe either way).
    """
    runs: List[Tuple[int, int]] = []
    for i in sorted(set(int(v) for v in ids)):
        if runs and i == runs[-1][1] + 1:
            runs[-1] = (runs[-1][0], i)
        else:
            runs.append((i, i))
    return runs


def run_stats(runs: Sequence[Tuple[int, int]], total: int) -> Dict:
    """Clustering stats over ``runs`` of error frame ids.

    ``total`` is the number of errors being clustered, so the percentages answer
    "how much of the S->E error mass sits inside the largest run(s)" — the
    question that separates a few repeated poses from pervasive failure.
    """
    if not runs:
        return {
            "n_runs": 0,
            "longest_run": 0,
            "longest_run_pct": 0.0,
            "top3_runs_pct": 0.0,
            "runs": [],
        }
    lengths = sorted((end - start + 1 for start, end in runs), reverse=True)
    top3 = sum(lengths[:3])
    return {
        "n_runs": len(runs),
        "longest_run": int(lengths[0]),
        "longest_run_pct": 100.0 * lengths[0] / total,
        "top3_runs_pct": 100.0 * top3 / total,
        "runs": [
            {"start": int(start), "end": int(end), "length": int(end - start + 1)}
            for start, end in runs
        ],
    }


# --------------------------------------------------------------------------- #
# Inference + record building
# --------------------------------------------------------------------------- #
def _resolve_device(requested: str) -> str:
    if requested == "auto":
        return "mps" if torch.backends.mps.is_available() else "cpu"
    return requested


def infer_s_records(
    checkpoint: str, manifest_path: str, device: str, batch_size: int
) -> List[Dict]:
    """Softmax-probability records for every ``S`` sample in the val split."""
    manifest = SplitManifest.from_json(manifest_path)
    ds = ASLImageDataset(manifest.val, train=False)
    model = load_checkpoint(checkpoint, device)
    model.eval()
    model.to(device)

    classes = list(CLASSES)
    s_idx = classes.index("S")
    e_idx = classes.index("E")
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)

    records: List[Dict] = []
    with torch.no_grad():
        for batch_i, (x, y) in enumerate(loader):
            probs = torch.softmax(model(x.to(device)), dim=1).cpu().numpy()
            top3 = np.argsort(probs, axis=1)[:, ::-1][:, :3]
            for i in range(int(x.shape[0])):
                sample = ds.samples[batch_i * batch_size + i]
                if sample.class_name != "S":
                    continue
                p_e = float(probs[i, e_idx])
                p_s = float(probs[i, s_idx])
                records.append(
                    {
                        "path": sample.path,
                        "frame_id": frame_id(sample.path),
                        "true": "S",
                        "pred": classes[int(top3[i, 0])],
                        "p_s": p_s,
                        "p_e": p_e,
                        "margin": p_e - p_s,
                        "top3": [
                            {"class": classes[j], "prob": float(probs[i, j])}
                            for j in top3[i]
                        ],
                    }
                )
    print(f"inference: {len(records)} S val samples scored "
          f"({len(ds)} total val images, device={device})")
    return records


def verify_counts(errors: List[Dict], correct: List[Dict], baseline_report: str) -> None:
    """The 138/462 check, against both inference and the baseline report."""
    n_err, n_ok = len(errors), len(correct)
    if n_err != 138 or n_ok != 462:
        raise SystemExit(
            f"S->E count mismatch: got {n_err} errors / {n_ok} correct "
            f"(expected 138 / 462). This script is tied to "
            f"efficientnet_b0_targeted.pt on the exact val split; "
            f"re-verify before proceeding."
        )
    if baseline_report and Path(baseline_report).is_file():
        rep = json.loads(Path(baseline_report).read_text())
        classes = rep["classes"]
        i, j = classes.index("S"), classes.index("E")
        reported = rep["confusion_matrix"][i][j]
        if reported != n_err:
            raise SystemExit(
                f"S->E count {n_err} disagrees with baseline report "
                f"{baseline_report} ({reported})"
            )
        print(f"verified: S->E = {n_err}, S->S = {n_ok}, "
              f"S recall = {n_ok / 600:.3f} "
              f"(matches {baseline_report})")


# --------------------------------------------------------------------------- #
# Galleries
# --------------------------------------------------------------------------- #
def render_model_input(path: str, size: int = IMG_SIZE) -> np.ndarray:
    """The exact deterministic pixels the model sees: square-resized RGB."""
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    resized = resize_square(img, size)
    return cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)


def _annotate(rec: Dict) -> str:
    base = Path(rec["path"]).name
    return (f"{rec['true']} -> {rec['pred']} | "
            f"E: {rec['p_e']:.2f} | S: {rec['p_s']:.2f} | "
            f"margin: {rec['margin']:.2f} | {base}")


def write_contact_sheet(
    records: Sequence[Dict], out_path: Path, title: str, cols: int = 5
) -> None:
    """A ``rows x cols`` grid of model-input tiles, one record each."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = (len(records) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 3.2 * rows))
    for ax, rec in zip(np.asarray(axes).flat, records):
        ax.imshow(render_model_input(rec["path"]))
        ax.axis("off")
        ax.set_title(_annotate(rec), fontsize=6.5)
    for ax in np.asarray(axes).flat[len(records):]:
        ax.axis("off")
    fig.suptitle(title, fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"wrote {out_path}")


def write_galleries(
    errors: List[Dict], correct: List[Dict], gallery_dir: Path, seed: int
) -> Dict[str, str]:
    a = top_k(errors, GALLERY_K, key=lambda r: r["p_e"])
    b = boundary_k(errors, GALLERY_K, key=lambda r: r["margin"])
    c = random_k(errors, GALLERY_K, seed=seed)
    d = random_k(correct, GALLERY_K, seed=seed)

    files = {
        "A_highest_pE.png": a,
        "B_boundary_margin.png": b,
        "C_random_errors.png": c,
        "D_correct_S_control.png": d,
    }
    paths: Dict[str, str] = {}
    for name, recs in files.items():
        out = gallery_dir / name
        write_contact_sheet(
            recs, out,
            title=name.replace(".png", "") + f"  ({len(recs)} tiles)",
        )
        paths[name] = str(out)
    return paths


# --------------------------------------------------------------------------- #
# Stats + outputs
# --------------------------------------------------------------------------- #
def error_stats(errors: List[Dict]) -> Dict:
    p_e = np.array([r["p_e"] for r in errors])
    p_s = np.array([r["p_s"] for r in errors])
    margin = np.array([r["margin"] for r in errors])
    return {
        "p_e_mean": float(p_e.mean()),
        "p_e_median": float(np.median(p_e)),
        "p_s_mean": float(p_s.mean()),
        "p_s_median": float(np.median(p_s)),
        "margin_mean": float(margin.mean()),
        "margin_median": float(np.median(margin)),
        "count_p_e_ge_0.9": int((p_e >= 0.9).sum()),
        "count_margin_lt_0.1": int((margin < 0.1).sum()),
    }


def build_json(
    records: List[Dict],
    errors: List[Dict],
    correct: List[Dict],
    checkpoint: str,
    manifest_path: str,
    gallery_files: Dict[str, str],
    created: str,
) -> Dict:
    clustering = run_stats(contiguous_runs([r["frame_id"] for r in errors]),
                           total=len(errors))
    return {
        "schema": SCHEMA,
        "created": created,
        "checkpoint": checkpoint,
        "arch": "efficientnet",
        "manifest": manifest_path,
        "manifest_sha256": json.loads(
            Path(manifest_path).read_text()
        ).get("content_sha256"),
        "split_note": (
            "dev val = leakage-safe frame-range split, same signer, same room "
            "(S2401..S3000). NOT the reported metric."
        ),
        "totals": {
            "s_val_samples": len(records),
            "correct_s": len(correct),
            "s_to_e": len(errors),
            "s_recall": len(correct) / len(records),
        },
        "error_probs": error_stats(errors),
        "frame_clustering": clustering,
        "gallery_files": gallery_files,
        "errors": errors,
    }


def write_visual_notes(
    out_path: Path,
    errors: List[Dict],
    correct: List[Dict],
    clustering: Dict,
    gallery_dir: Path,
) -> None:
    """Draft of the visual comparison; filled in after inspecting galleries."""
    p_e = np.array([r["p_e"] for r in errors])
    margin = np.array([r["margin"] for r in errors])
    runs = clustering["runs"]
    major = [r for r in runs if r["length"] >= 5] or runs[:3]
    run_lines = ", ".join(
        f"{r['start']}-{r['end']} (n={r['length']})" for r in major
    )
    lines = [
        "# S -> E visual notes — efficientnet_b0_targeted (diagnosis only)",
        "",
        "Descriptive comparison between the 138 S->E errors and the 462 correct "
        "S recognitions, based on the contact sheets in "
        f"`{gallery_dir}` and the probability records in "
        "`s_to_e_error_analysis_targeted.json`. This document is "
        "**descriptive only** — it records what is observed in the frames and "
        "in the model's outputs, not a causal claim.",
        "",
        "## Group summary",
        "",
        f"- Correct S: **{len(correct)}** / 600 (recall 0.770)",
        f"- S->E errors: **{len(errors)}**",
        f"- Mean P(E) on errors: **{p_e.mean():.3f}**; "
        f"median **{np.median(p_e):.3f}**",
        f"- Mean E-S margin on errors: **{margin.mean():.3f}**",
        f"- Errors with P(E) >= 0.9: "
        f"**{(p_e >= 0.9).sum()}** of {len(errors)}",
        f"- Errors with margin < 0.1: "
        f"**{(margin < 0.1).sum()}** of {len(errors)}",
        f"- Contiguous error runs: **{clustering['n_runs']}**; "
        f"longest **{clustering['longest_run']}** frames "
        f"({clustering['longest_run_pct']:.1f}% of errors); "
        f"top-3 runs {clustering['top3_runs_pct']:.1f}% of errors",
        f"- Major error ranges: {run_lines}",
        "",
        "## Per-dimension comparison (S->E vs S->S)",
        "",
        "Each dimension is labelled *Observed* (what the frames show), "
        "*Possible hypothesis* (a mechanism the data is consistent with, not "
        "proof), or *No obvious difference*.",
        "",
    ]
    dims = [
        ("Hand orientation", ""),
        ("Thumb position / visibility", ""),
        ("Finger closure", ""),
        ("Occlusion", ""),
        ("Hand position", ""),
        ("Hand scale", ""),
        ("Brightness / exposure", ""),
        ("Shadows", ""),
        ("Blur", ""),
        ("Background", ""),
        ("Repeated pose / frame sequences", ""),
    ]
    for name, _ in dims:
        lines.append(f"### {name}\n")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")
    print(f"wrote draft {out_path}")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def analyze(args) -> Dict:
    device = _resolve_device(args.device)
    manifest_sha = json.loads(Path(args.manifest).read_text()).get("content_sha256")
    print(f"manifest: {args.manifest}  sha256={manifest_sha}")

    records = infer_s_records(args.checkpoint, args.manifest, device, args.batch_size)
    errors = [r for r in records if r["pred"] == "E"]
    correct = [r for r in records if r["pred"] == "S"]
    verify_counts(errors, correct, args.baseline_report)

    clustering = run_stats(contiguous_runs([r["frame_id"] for r in errors]),
                           total=len(errors))
    stats = error_stats(errors)
    print("\nS totals: " + json.dumps(
        {"s_val": len(records), "correct_s": len(correct),
         "s_to_e": len(errors), "s_recall": round(len(correct) / len(records), 4)}
    ))
    print("error probs: " + json.dumps({k: round(v, 4)
                                        for k, v in stats.items()}))
    print("clustering: " + json.dumps(
        {k: v for k, v in clustering.items() if k != "runs"}
    ))

    gallery_dir = Path(args.gallery_dir)
    gallery_files = write_galleries(errors, correct, gallery_dir, args.seed)

    created = datetime.now(timezone.utc).isoformat(timespec="seconds")
    report = build_json(records, errors, correct, args.checkpoint,
                        args.manifest, gallery_files, created)
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {args.out_json}")

    md = Path(args.out_md)
    if md.exists():
        # The curated visual notes are written once, after a human inspects the
        # galleries; a re-run must not clobber them with the empty draft.
        print(f"kept existing {md} (curated; re-run will not overwrite)")
    else:
        write_visual_notes(md, errors, correct, clustering, gallery_dir)
    return report

def main() -> None:
    p = argparse.ArgumentParser(
        description="Diagnose S->E on efficientnet_b0_targeted.pt (no fixes)."
    )
    p.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    p.add_argument("--manifest", default=DEFAULT_MANIFEST)
    p.add_argument("--baseline-report", default=DEFAULT_BASELINE_REPORT)
    p.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--seed", type=int, default=RANDOM_SEED)
    p.add_argument("--gallery-dir", default=DEFAULT_GALLERY_DIR)
    p.add_argument("--out-json", default=DEFAULT_OUT_JSON)
    p.add_argument("--out-md", default=DEFAULT_OUT_MD)
    analyze(p.parse_args())


if __name__ == "__main__":
    main()
