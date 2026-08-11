"""Build the human-inspection transition galleries for the S->E diagnosis.

Follow-up to ``scripts/inspect_s_to_e_runs.py``: the earlier ``E_transition_runs.png``
crammed 20 tiles per strip into ~1.5 in each. These artifacts are sized for a
human reviewer to inspect thumb/finger geometry, so each tile keeps ~2.3 in.

Outputs (in ``docs/figures/error_gallery_targeted/S_to_E/``):

1. ``E_transition_run{1,2,3}_human.png`` — one high-resolution chronological
   strip per dominant run: 4 correct-S before the first error -> first 8 S->E
   frames -> 2-3 representative middle-run errors -> final 4 errors -> 4
   correct-S after recovery. Internal correct "blips" (S2583, S2811,
   S2872-S2874) stay in chronological place.
2. ``E_transition_runs_human.png`` — the same three strips at overview size
   (one row per run) for quick scanning.
3. ``F_pose_comparison_human.png`` — 6 groups x 5 large tiles: strong correct S,
   first-frame S->E, high-confidence S->E, boundary-margin S->E, correct-S
   blips inside the runs, correct S from the clean S2620-S2776 stretch.
4. ``transition_human_manifest.json`` — exact frames per run (id, phase, pred,
   P(S), P(E)), pose-group memberships, image dimensions, missing-frame report.

Tile annotation is exactly ``S2575 | pred=E | S=.20 | E=.54`` (leading zero
dropped, matching the requested format). Colors are the deterministic model
input (``render_model_input``: 200x200 BGR -> ``resize_square`` -> 224x224 RGB).
Prediction probabilities are read directly from the cached inference records, so
every annotation matches them by construction.

No geometry is inferred anywhere in this script. No segmentation, MediaPipe,
landmarks, or other CV analysis. This is material for a human reviewer only.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.inspect_s_to_e_runs import (  # noqa: E402
    _resolve_device,
    boundary_k,
    even_sample,
    infer_s_records,
    render_model_input,
    top_k,
    verify_counts,
)

DEFAULT_CHECKPOINT = "checkpoints/efficientnet_b0_targeted.pt"
DEFAULT_MANIFEST = "data/split_manifest.json"
DEFAULT_GALLERY_DIR = "docs/figures/error_gallery_targeted/S_to_E"
DEFAULT_CACHE = "data/s_to_e_all_s_records.json"

SEGMENTS: List[Tuple[str, int, int]] = [
    ("Run 1", 2575, 2619),
    ("Run 2", 2801, 2814),
    ("Run 3", 2831, 2877),
]
CLEAN_RANGE = (2620, 2776)

BEFORE_K = 4
FIRST_K = 8
MIDDLE_K = 3
FINAL_K = 4
AFTER_K = 4

POSES_PER_GROUP = 5
STRIP_TILES_PER_ROW = 12
RUN_DPI = 150
POSES_DPI = 150
RUN_TILE_IN = 2.3
POSES_TILE_IN = 2.2
OVERVIEW_TILE_IN = 1.4
OVERVIEW_DPI = 150


# --------------------------------------------------------------------------- #
# Selection
# --------------------------------------------------------------------------- #
def _phase(tag: str, rec: Dict) -> Dict:
    return {"rec": rec, "phase": tag}


def transition_strip(
    records: Sequence[Dict],
    seg_start: int,
    seg_end: int,
    before_k: int = BEFORE_K,
    first_k: int = FIRST_K,
    middle_k: int = MIDDLE_K,
    final_k: int = FINAL_K,
    after_k: int = AFTER_K,
) -> List[Dict]:
    """Chronological, deduplicated boundary-focused tile list for one run.

    Returns ``[{rec, phase}]`` with ``phase`` in
    ``before | first | blip | middle | final | after``, ordered by frame id.
    The first selected run frame is ``seg_start`` (the first S->E error).
    """
    by_id = {r["frame_id"]: r for r in records}

    def correct_before(start_id: int) -> List[int]:
        cand = sorted(
            (r["frame_id"] for r in records
             if r["pred"] == "S" and r["frame_id"] < start_id)
        )
        return cand[-before_k:]

    def correct_after(end_id: int) -> List[int]:
        cand = sorted(
            (r["frame_id"] for r in records
             if r["pred"] == "S" and r["frame_id"] > end_id)
        )
        return cand[:after_k]

    seg_ids = sorted(
        r["frame_id"] for r in records if seg_start <= r["frame_id"] <= seg_end
    )
    err_ids = [i for i in seg_ids if by_id[i]["pred"] == "E"]
    blip_ids = [i for i in seg_ids if by_id[i]["pred"] == "S"]
    if not err_ids:
        raise ValueError(f"no S->E error frames in S{seg_start}-S{seg_end}")

    first_ids = err_ids[:first_k]          # first FIRST_K errors (from seg_start)
    final_ids = err_ids[-final_k:]         # last FINAL_K errors
    middle_pool = [i for i in err_ids
                   if i not in first_ids and i not in final_ids]

    # Representative middle errors: even sample, force-include max-P(E).
    middle_ids: List[int] = []
    if middle_pool:
        base = even_sample(middle_pool, middle_k)
        max_pe = max(middle_pool, key=lambda i: by_id[i]["p_e"])
        if max_pe not in base:
            nearest = min(base, key=lambda i: abs(i - max_pe))
            base = [max_pe if i == nearest else i for i in base]
        middle_ids = sorted(base)

    chosen = set()
    tiles: List[Dict] = []
    for i in correct_before(seg_start):
        tiles.append(_phase("before", by_id[i]))
        chosen.add(i)
    for i in first_ids + middle_ids + blip_ids + final_ids:
        if i in chosen:
            continue
        if i in first_ids:
            phase = "first"
        elif i in middle_ids:
            phase = "middle"
        elif i in blip_ids:
            phase = "blip"
        else:
            phase = "final"
        tiles.append(_phase(phase, by_id[i]))
        chosen.add(i)
    for i in correct_after(seg_end):
        tiles.append(_phase("after", by_id[i]))

    tiles.sort(key=lambda t: t["rec"]["frame_id"])
    return tiles


# --------------------------------------------------------------------------- #
# Annotation (matches the requested format exactly)
# --------------------------------------------------------------------------- #
def _fmt_p(p: float) -> str:
    """``0.54`` -> ``.54`` to match the requested ``S=.20 | E=.54`` format."""
    s = f"{p:.2f}"
    return s[1:] if s.startswith("0") else s


def annotate(rec: Dict) -> str:
    stem = Path(rec["path"]).stem
    return (f"{stem} | pred={rec['pred']} | "
            f"S={_fmt_p(rec['p_s'])} | E={_fmt_p(rec['p_e'])}")


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #
def _verify_tile(rec: Dict) -> bool:
    """True if the frame file exists and renders as an RGB model input."""
    if not Path(rec["path"]).is_file():
        return False
    img = render_model_input(rec["path"])
    return img.shape == (224, 224, 3) and img.dtype == np.uint8


def _tile_axes(plt, rows: int, cols: int, tile_in: float):
    fig, axes = plt.subplots(rows, cols,
                             figsize=(cols * tile_in, rows * tile_in))
    return fig, np.asarray(axes).reshape(rows, cols)


def _paint(ax, rec: Dict, edge: str) -> None:
    ax.imshow(render_model_input(rec["path"]))
    ax.axis("off")
    for sp in ax.spines.values():
        sp.set_edgecolor(edge)
        sp.set_linewidth(2.5)
    ax.set_title(annotate(rec), fontsize=8.5)


def write_human_strip(
    out_path: Path, tiles: Sequence[Dict], tile_in: float, dpi: int
) -> Tuple[int, int]:
    """Wrapped chronological strip. Returns (width_px, height_px)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(tiles)
    ncols = STRIP_TILES_PER_ROW
    nrows = (n + ncols - 1) // ncols
    fig, axes = _tile_axes(plt, nrows, ncols, tile_in)
    for idx, t in enumerate(tiles):
        ax = axes[idx // ncols, idx % ncols]
        rec = t["rec"]
        edge = "#2e7d32" if rec["pred"] == "S" else "#c62828"
        _paint(ax, rec, edge)
    for ax in axes.flat[n:]:
        ax.axis("off")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return int(ncols * tile_in * dpi), int(nrows * tile_in * dpi)


def write_combined_overview(
    out_path: Path, strips_by_run: Sequence[Tuple[str, Sequence[Dict]]]
) -> Tuple[int, int]:
    """One row per run at overview size (smaller tiles, quick scan)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ncols = max(len(tiles) for _, tiles in strips_by_run)
    nrows = len(strips_by_run)
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * OVERVIEW_TILE_IN,
                                      nrows * OVERVIEW_TILE_IN))
    for row, (label, tiles) in enumerate(strips_by_run):
        for col, t in enumerate(tiles):
            rec = t["rec"]
            edge = "#2e7d32" if rec["pred"] == "S" else "#c62828"
            _paint(axes[row, col], rec, edge)
        for col in range(len(tiles), ncols):
            axes[row, col].axis("off")
        fig.text(
            0.002,
            1.0 - (row + 0.5) / nrows,
            label,
            rotation=90,
            va="center",
            ha="left",
            fontsize=11,
        )
    fig.tight_layout(rect=(0.015, 0, 1, 0.97))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=OVERVIEW_DPI)
    plt.close(fig)
    return int(ncols * OVERVIEW_TILE_IN * OVERVIEW_DPI), int(
        nrows * OVERVIEW_TILE_IN * OVERVIEW_DPI)


def pose_groups(errors: List[Dict], correct: List[Dict]) -> List[Tuple[str, List[Dict]]]:
    run_bounds = [(s, e) for _, s, e in SEGMENTS]
    first_frames: List[Dict] = []
    for _, s, e in SEGMENTS:
        first_frames.append(next(r for r in errors if r["frame_id"] == s))
        second = next((r for r in errors if r["frame_id"] == s + 1), None)
        if second is not None and s + 1 <= e:
            first_frames.append(second)
    blips = sorted(
        (r for r in correct if any(s <= r["frame_id"] <= e
                                   for s, e in run_bounds)),
        key=lambda r: r["frame_id"],
    )
    clean = sorted(
        (r for r in correct
         if CLEAN_RANGE[0] <= r["frame_id"] <= CLEAN_RANGE[1]),
        key=lambda r: r["frame_id"],
    )
    return [
        ("Strong correct-S", top_k(correct, POSES_PER_GROUP,
                                   key=lambda r: r["p_s"])),
        ("First-frame S->E", first_frames[:POSES_PER_GROUP]),
        ("High-confidence S->E", top_k(errors, POSES_PER_GROUP,
                                       key=lambda r: r["p_e"])),
        ("Boundary S->E", boundary_k(errors, POSES_PER_GROUP,
                                     key=lambda r: r["margin"])),
        ("Correct-S blips in runs", even_sample(blips, POSES_PER_GROUP)),
        ("Correct-S S2620-2776", even_sample(clean, POSES_PER_GROUP)),
    ]


def write_pose_sheet_human(
    out_path: Path, groups: Sequence[Tuple[str, List[Dict]]]
) -> Tuple[int, int]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    nrows, ncols = len(groups), POSES_PER_GROUP
    fig, axes = _tile_axes(plt, nrows, ncols, POSES_TILE_IN)
    for row, (label, recs) in enumerate(groups):
        for col in range(ncols):
            if col < len(recs):
                rec = recs[col]
                edge = "#2e7d32" if rec["pred"] == "S" else "#c62828"
                _paint(axes[row, col], rec, edge)
            else:
                axes[row, col].axis("off")
        fig.text(
            0.002,
            1.0 - (row + 0.5) / nrows,
            label,
            rotation=90,
            va="center",
            ha="left",
            fontsize=9,
        )
    fig.tight_layout(rect=(0.02, 0, 1, 0.98))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=POSES_DPI)
    plt.close(fig)
    return int(ncols * POSES_TILE_IN * POSES_DPI), int(
        nrows * POSES_TILE_IN * POSES_DPI)


# --------------------------------------------------------------------------- #
# Manifest + CLI
# --------------------------------------------------------------------------- #
def load_cached_records(
    cache: str, checkpoint: str, manifest: str, device: str,
    batch_size: int, baseline_report: str,
) -> List[Dict]:
    """Read the cached 600-record inference, or re-infer and cache it."""
    cache_path = Path(cache)
    if cache_path.is_file():
        records = json.loads(cache_path.read_text())
        n_err = sum(1 for r in records if r["pred"] == "E")
        n_ok = sum(1 for r in records if r["pred"] == "S")
        if n_err != 138 or n_ok != 462:
            raise SystemExit(
                f"cached records disagree with the targeted baseline "
                f"({n_err} errors / {n_ok} correct); delete {cache} to re-infer"
            )
        print(f"loaded {len(records)} S records from {cache}")
        return records
    records = infer_s_records(checkpoint, manifest, device, batch_size)
    errors = [r for r in records if r["pred"] == "E"]
    correct = [r for r in records if r["pred"] == "S"]
    verify_counts(errors, correct, baseline_report)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(records))
    print(f"inferred + cached {len(records)} S records to {cache}")
    return records


def build_manifest(records: List[Dict]) -> Dict:
    """Per-run/group frame lists + phase + predictions for the report."""
    manifest: Dict = {
        "schema": "asl-s_to_e-human-galleries/v1",
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "checkpoint": DEFAULT_CHECKPOINT,
        "cache": DEFAULT_CACHE,
        "note": (
            "Frames are model val inputs rendered via the shared preprocessing "
            "contract; predictions/probabilities are copied verbatim from the "
            "cached inference records. No geometry is inferred."
        ),
        "runs": {},
        "pose_groups": {},
        "missing_frames": [],
    }
    for label, s, e in SEGMENTS:
        tiles = transition_strip(records, s, e)
        manifest["runs"][label] = {
            "frame_range": [s, e],
            "frames": [
                {
                    "frame_id": t["rec"]["frame_id"],
                    "phase": t["phase"],
                    "pred": t["rec"]["pred"],
                    "p_s": t["rec"]["p_s"],
                    "p_e": t["rec"]["p_e"],
                }
                for t in tiles
            ],
        }
    errors = [r for r in records if r["pred"] == "E"]
    correct = [r for r in records if r["pred"] == "S"]
    for label, recs in pose_groups(errors, correct):
        manifest["pose_groups"][label] = [
            {
                "frame_id": r["frame_id"],
                "pred": r["pred"],
                "p_s": r["p_s"],
                "p_e": r["p_e"],
            }
            for r in recs
        ]
    for r in records:
        if not Path(r["path"]).is_file():
            manifest["missing_frames"].append(r["frame_id"])
    return manifest


def run(args) -> Dict:
    device = _resolve_device(args.device)
    records = load_cached_records(
        args.cache, args.checkpoint, args.manifest, device,
        args.batch_size, args.baseline_report)
    errors = [r for r in records if r["pred"] == "E"]
    correct = [r for r in records if r["pred"] == "S"]
    gallery_dir = Path(args.gallery_dir)

    # Verify every selected frame renders before writing anything.
    strips_by_run: List[Tuple[str, List[Dict]]] = []
    for label, s, e in SEGMENTS:
        tiles = transition_strip(records, s, e)
        bad = [t for t in tiles if not _verify_tile(t["rec"])]
        if bad:
            raise SystemExit(
                f"{label}: {len(bad)} frames could not be rendered: "
                + ", ".join(str(t["rec"]["frame_id"]) for t in bad))
        strips_by_run.append((label, tiles))
    pose_groups_list = pose_groups(errors, correct)
    bad_pose = [
        rec for _, recs in pose_groups_list for rec in recs
        if not _verify_tile(rec)
    ]
    if bad_pose:
        raise SystemExit(
            "pose sheet: frames could not be rendered: "
            + ", ".join(str(r["frame_id"]) for r in bad_pose))

    dims: Dict[str, List[int]] = {}
    for (label, _, _), (_, tiles) in zip(SEGMENTS, strips_by_run):
        run_no = label.split()[-1]
        out = gallery_dir / f"E_transition_run{run_no}_human.png"
        dims[out.name] = list(write_human_strip(out, tiles, RUN_TILE_IN, RUN_DPI))
    overview = gallery_dir / "E_transition_runs_human.png"
    dims[overview.name] = list(write_combined_overview(overview, strips_by_run))
    pose_out = gallery_dir / "F_pose_comparison_human.png"
    dims[pose_out.name] = list(write_pose_sheet_human(pose_out, pose_groups_list))

    manifest = build_manifest(records)
    manifest["image_dimensions_px"] = dims
    manifest["image_tile_size_in"] = {
        "run_strips": RUN_TILE_IN,
        "overview": OVERVIEW_TILE_IN,
        "pose_sheet": POSES_TILE_IN,
    }
    manifest_out = Path(args.out_manifest)
    manifest_out.parent.mkdir(parents=True, exist_ok=True)
    manifest_out.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {manifest_out}")
    print("dims: " + json.dumps(dims))
    return manifest


def main() -> None:
    p = argparse.ArgumentParser(
        description="Human-inspection S->E transition galleries (no analysis)."
    )
    p.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    p.add_argument("--manifest", default=DEFAULT_MANIFEST)
    p.add_argument("--baseline-report",
                   default="docs/reports/dev_val_targeted_current.json")
    p.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"],
                   default="auto")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--cache", default=DEFAULT_CACHE)
    p.add_argument("--gallery-dir", default=DEFAULT_GALLERY_DIR)
    p.add_argument("--out-manifest",
                   default=str(Path(DEFAULT_GALLERY_DIR) / "transition_human_manifest.json"))
    run(p.parse_args())


if __name__ == "__main__":
    main()
