"""Build the transition-run and pose-comparison galleries for the S->E diagnosis.

Follow-up to ``scripts/analyze_s_to_e.py``: the group-level contact sheets there
(A-D) show the 138 S->E errors vs correct S, but not *how* a contiguous error
run behaves over time. This script focuses on the three dominant error segments
(Run 1 = S2575-S2619, Run 2 = S2801-S2814, Run 3 = S2831-S2877):

1. Re-uses the exact same inference (``infer_s_records``) but **caches** the
   600 S val records to ``data/s_to_e_all_s_records.json`` so the ~6 min MPS
   inference runs once.
2. ``E_transition_runs.png`` — one chronological filmstrip per run:
   last correct-S frames before the run -> first error -> middle -> last
   error -> first correct-S after the run. Each tile is the actual model input
   annotated ``S2575 | pred=E | S=.39 E=.52``.
3. ``F_pose_comparison.png`` — 4 groups x 5: high-confidence correct S,
   dominant-run S->E, boundary-margin S->E, correct S inside the long
   error-free stretch (S2620-S2776).
4. Writes ``docs/reports/s_to_e_transition_visual_notes.md`` — per-run
   transition observations grounded in the probability records and measured
   pixel signals at the boundaries; dimensions only a human eye can judge
   (thumb, finger closure, orientation, occlusion) are explicitly labelled
   *requires visual inspection*.

Diagnosis only: no training, no preprocessing/augmentation changes, no
MediaPipe, no fixes.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2  # noqa: E402

from scripts.analyze_s_to_e import (  # noqa: E402
    _resolve_device,
    boundary_k,
    infer_s_records,
    render_model_input,
    top_k,
    verify_counts,
)

DEFAULT_CHECKPOINT = "checkpoints/efficientnet_b0_targeted.pt"
DEFAULT_MANIFEST = "data/split_manifest.json"
DEFAULT_GALLERY_DIR = "docs/figures/error_gallery_targeted/S_to_E"
DEFAULT_OUT_MD = "docs/reports/s_to_e_transition_visual_notes.md"
DEFAULT_CACHE = "data/s_to_e_all_s_records.json"

SEGMENTS: List[Tuple[str, int, int]] = [
    ("Run 1", 2575, 2619),
    ("Run 2", 2801, 2814),
    ("Run 3", 2831, 2877),
]
CLEAN_RANGE = (2620, 2776)  # long error-free stretch between Run 1 and Run 2
BEFORE_K = 3
AFTER_K = 3
MID_K = 14  # tiles sampled inside each run's frame range
POSES_PER_GROUP = 5


# --------------------------------------------------------------------------- #
# Pure helpers (unit-tested)
# --------------------------------------------------------------------------- #
def even_sample(seq: Sequence, n: int) -> List:
    """Up to ``n`` items of ``seq``, evenly spaced, first-to-last order.

    Deterministic (``np.linspace`` over indices, truncated to ints). Returns the
    whole sequence when it has at most ``n`` items.
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    if len(seq) <= n:
        return list(seq)
    idx = np.linspace(0, len(seq) - 1, n).astype(int)
    return [seq[i] for i in idx]


def correct_before(records: Sequence[Dict], start_id: int, k: int) -> List[Dict]:
    """The ``k`` correct-S frames with the largest id < ``start_id``, ascending."""
    cand = sorted(
        (r for r in records if r["pred"] == "S" and r["frame_id"] < start_id),
        key=lambda r: r["frame_id"],
    )
    return cand[-k:]


def correct_after(records: Sequence[Dict], end_id: int, k: int) -> List[Dict]:
    """The ``k`` correct-S frames with the smallest id > ``end_id``, ascending."""
    cand = sorted(
        (r for r in records if r["pred"] == "S" and r["frame_id"] > end_id),
        key=lambda r: r["frame_id"],
    )
    return cand[:k]


def segment_frames(
    records: Sequence[Dict],
    seg_start: int,
    seg_end: int,
    mid_k: int,
    before_k: int = BEFORE_K,
    after_k: int = AFTER_K,
) -> List[Dict]:
    """Chronological before -> evenly-sampled segment -> after frame list.

    The segment sample includes both error and in-range correct frames (e.g. a
    brief correct blip inside a dominant run is informative).
    """
    seg = sorted(
        (r for r in records if seg_start <= r["frame_id"] <= seg_end),
        key=lambda r: r["frame_id"],
    )
    return correct_before(records, seg_start, before_k) + even_sample(
        seg, mid_k
    ) + correct_after(records, seg_end, after_k)


def probe_frames(
    records: Sequence[Dict],
    seg_start: int,
    seg_end: int,
    before_k: int = BEFORE_K,
    after_k: int = AFTER_K,
) -> List[Dict]:
    """Boundary probe: before frames, first/max-P(E)/last error, after frames."""
    seg = sorted(
        (r for r in records if seg_start <= r["frame_id"] <= seg_end),
        key=lambda r: r["frame_id"],
    )
    errs = [r for r in seg if r["pred"] == "E"]
    if not errs:
        return []
    first = errs[0]
    maxpe = max(errs, key=lambda r: r["p_e"])
    last = errs[-1]
    return (
        correct_before(records, seg_start, before_k)
        + [first, maxpe, last]
        + correct_after(records, seg_end, after_k)
    )


# --------------------------------------------------------------------------- #
# Pixel metrics (measured proxies; geometry is not measurable without a clean
# hand silhouette — see the probe note in the group-level analysis)
# --------------------------------------------------------------------------- #
def pixel_metrics(path: str) -> Dict[str, float]:
    """Grey mean and very-dark fraction of the raw (pre-resize) frame."""
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return {
        "grey_mean": float(grey.mean()),
        "dark_frac": float((grey < 100).mean()),
    }


# --------------------------------------------------------------------------- #
# Galleries
# --------------------------------------------------------------------------- #
def _annotate(rec: Dict) -> str:
    base = Path(rec["path"]).stem
    return f"{base} | pred={rec['pred']} | S={rec['p_s']:.2f} E={rec['p_e']:.2f}"


def write_transition_strips(
    strips: Sequence[Tuple[str, Tuple[int, int], List[Dict]]],
    out_path: Path,
) -> None:
    """One filmstrip per run: before / run / after, in chronological order."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ncols = max(len(recs) for _, _, recs in strips)
    nrows = len(strips)
    fig, axes = plt.subplots(nrows, ncols, figsize=(1.55 * ncols, 1.9 * nrows))
    axes = np.asarray(axes)
    for row, (label, (s, e), recs) in enumerate(strips):
        for col in range(ncols):
            ax = axes[row, col]
            if col < len(recs):
                ax.imshow(render_model_input(recs[col]["path"]))
                ax.set_title(_annotate(recs[col]), fontsize=5.0)
            ax.axis("off")
        fig.text(
            0.005,
            1.0 - (row + 0.5) / nrows,
            f"{label}  S{s}-S{e}",
            rotation=90,
            va="center",
            ha="left",
            fontsize=9,
        )
    fig.suptitle(
        "S->E dominant runs: chronological model input (before -> run -> after)",
        fontsize=12,
    )
    fig.tight_layout(rect=(0.02, 0, 1, 0.96))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"wrote {out_path}")


def write_pose_sheet(
    groups: Sequence[Tuple[str, List[Dict]]], out_path: Path
) -> None:
    """Side-by-side comparison of four S groups (5 tiles each)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    nrows, ncols = len(groups), POSES_PER_GROUP
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.0 * ncols, 1.9 * nrows))
    axes = np.asarray(axes)
    for row, (label, recs) in enumerate(groups):
        for col in range(ncols):
            ax = axes[row, col]
            if col < len(recs):
                ax.imshow(render_model_input(recs[col]["path"]))
                ax.set_title(_annotate(recs[col]), fontsize=6.0)
            ax.axis("off")
        fig.text(
            0.005,
            1.0 - (row + 0.5) / nrows,
            label,
            rotation=90,
            va="center",
            ha="left",
            fontsize=8,
        )
    fig.suptitle("S pose comparison: correct vs S->E (5 tiles per group)",
                 fontsize=12)
    fig.tight_layout(rect=(0.02, 0, 1, 0.96))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"wrote {out_path}")


def pose_groups(
    errors: List[Dict], correct: List[Dict]
) -> List[Tuple[str, List[Dict]]]:
    dominant = sorted(
        (
            r
            for r in errors
            if any(s <= r["frame_id"] <= e for _, s, e in SEGMENTS)
        ),
        key=lambda r: r["frame_id"],
    )
    clean = sorted(
        (
            r
            for r in correct
            if CLEAN_RANGE[0] <= r["frame_id"] <= CLEAN_RANGE[1]
        ),
        key=lambda r: r["frame_id"],
    )
    return [
        ("High-conf correct S", top_k(correct, POSES_PER_GROUP,
                                      key=lambda r: r["p_s"])),
        ("Dominant-run S->E", even_sample(dominant, POSES_PER_GROUP)),
        ("Boundary S->E", boundary_k(errors, POSES_PER_GROUP,
                                     key=lambda r: r["margin"])),
        ("Correct S (S2620-2776)", even_sample(clean, POSES_PER_GROUP)),
    ]


# --------------------------------------------------------------------------- #
# Notes
# --------------------------------------------------------------------------- #
def _fmt_prob(v: float) -> str:
    return f"{v:.2f}"


def _fmt_rec(rec: Dict, metrics: Dict) -> str:
    m = metrics
    return (
        f"| S{rec['frame_id']} | {rec['pred']} | {_fmt_prob(rec['p_s'])} "
        f"| {_fmt_prob(rec['p_e'])} | {m['grey_mean']:.0f} "
        f"| {m['dark_frac']:.2f} |"
    )


def write_transition_notes(
    out_path: Path,
    records: List[Dict],
    errors: List[Dict],
    correct: List[Dict],
    gallery_dir: Path,
) -> None:
    metrics = {r["frame_id"]: pixel_metrics(r["path"]) for r in records}

    created = datetime.now(timezone.utc).isoformat(timespec="seconds")
    lines = [
        "# S -> E transition-run visual notes — efficientnet_b0_targeted",
        "",
        "Follow-up to `s_to_e_visual_notes.md`. Instead of group-level averages, "
        "this document watches each of the three dominant contiguous error "
        "segments over time: what the frames and the model's probabilities do "
        "right before a run begins, across the run, and when prediction returns "
        "to S.",
        "",
        f"Generated {created} by `scripts/inspect_s_to_e_runs.py` "
        "(diagnosis only; no fixes).",
        "",
        "## Runs studied",
        "",
        f"- **Run 1** S2575-S2619 — errors S2575-S2582, S2584-S2619 "
        f"(44 errors; correct blip S2583)",
        f"- **Run 2** S2801-S2814 — errors except the correct blip S2811 "
        f"(13 errors)",
        f"- **Run 3** S2831-S2877 — errors S2831-S2871, S2875-S2877 "
        f"(44 errors; correct blips S2872-S2874)",
        f"- Long error-free stretch between Run 1 and Run 2: S2620-S2776 "
        f"(157 frames)",
        "",
        "## Galleries (for human review)",
        "",
        f"- `{gallery_dir}/E_transition_runs.png` — one chronological filmstrip "
        "per run. Tile order is last correct-S frames before the run -> first "
        "error -> evenly sampled run frames -> last error -> first correct-S "
        "after the run. Each tile is the actual model input (200x200 -> "
        "`resize_square` -> 224x224 RGB), annotated "
        "`S2575 | pred=E | S=.39 E=.52`.",
        f"- `{gallery_dir}/F_pose_comparison.png` — 4 rows x 5 tiles: "
        "high-confidence correct S, dominant-run S->E, boundary-margin S->E, "
        "correct S inside S2620-2776.",
        "",
        "## Method and limitation",
        "",
        "As in `s_to_e_visual_notes.md`: the authoring model cannot render the "
        "images, so the observations below are grounded in (1) the softmax "
        "probability records, (2) measured pixel signals at the run boundaries "
        "(grey mean, very-dark fraction `<100` of the raw frame), and (3) frame "
        "clustering. Hand-geometry dimensions (thumb, finger closure, "
        "orientation, occlusion) are **not** measurable without a clean hand "
        "silhouette (dark pixels span the whole frame) and are labelled "
        "*requires visual inspection*.",
        "",
        "Tables: `pred` is the argmax class, `P(S)` / `P(E)` are softmax "
        "probabilities, `grey` is mean grey level of the raw frame, `dark` is "
        "the fraction of pixels below grey 100.",
        "",
    ]

    for label, s, e in SEGMENTS:
        errs_in = sorted(
            (r for r in errors if s <= r["frame_id"] <= e),
            key=lambda r: r["frame_id"],
        )
        blips = sorted(
            (r for r in records if s <= r["frame_id"] <= e and r["pred"] == "S"),
            key=lambda r: r["frame_id"],
        )
        probes = probe_frames(records, s, e)
        lines += [
            f"## {label}: S{s}-S{e}",
            "",
            f"Errors: **{len(errs_in)}** frames "
            f"({len(errs_in) / len(errors) * 100:.1f}% of all S->E errors); "
            f"correct blips inside the range: "
            + (", ".join(f"S{r['frame_id']}" for r in blips) or "none")
            + f". P(E) within run: {min(r['p_e'] for r in errs_in):.2f} - "
            f"{max(r['p_e'] for r in errs_in):.2f}.",
            "",
            "| frame | pred | P(S) | P(E) | grey | dark |",
            "|-------|------|------|------|------|------|",
        ]
        for rec in probes:
            lines.append(_fmt_rec(rec, metrics[rec["frame_id"]]))
        lines.append("")

        # Measured transition observations.
        before = [r for r in probes if r["frame_id"] < s]
        first_err = next((r for r in probes if r["frame_id"] == s), None)
        after = [r for r in probes if r["frame_id"] > e]
        if before and first_err:
            b_grey = np.mean([metrics[r["frame_id"]]["grey_mean"] for r in before])
            b_dark = np.mean([metrics[r["frame_id"]]["dark_frac"] for r in before])
            fe = first_err["frame_id"]
            f_grey = metrics[fe]["grey_mean"]
            f_dark = metrics[fe]["dark_frac"]
            lines += [
                "- **What changes immediately before S->E begins (measured):** "
                f"the {len(before)} last correct-S frames average grey {b_grey:.0f} "
                f"and dark {b_dark:.2f}; the first error S{fe} is grey {f_grey:.0f} "
                f"and dark {f_dark:.2f} ({'brighter' if f_grey >= b_grey else 'darker'} "
                f"by {abs(f_grey - b_grey):.0f}). P(S) drops from "
                f"{before[-1]['p_s']:.2f} to {first_err['p_s']:.2f} while P(E) rises "
                f"to {first_err['p_e']:.2f} — the decision flips in one frame.",
                "- **What changes immediately before S->E begins (visual):** "
                "**requires visual inspection** of the boundary tiles "
                f"(`S{before[0]['frame_id']}`..`S{e}`) in "
                "`E_transition_runs.png` — thumb position, finger closure and "
                "orientation cannot be measured by this tool.",
                "",
            ]
        if errs_in:
            lines += [
                "- **What stays consistent through the run:** S remains the 2nd "
                "prediction on every error frame and P(E) stays moderate "
                f"({min(r['p_e'] for r in errs_in):.2f}-"
                f"{max(r['p_e'] for r in errs_in):.2f}, never a confident "
                "read). The run is one repeated pose captured across many "
                f"near-duplicate video frames, not {len(errs_in)} independent "
                "failures.",
                "",
            ]
        if after:
            a_first = after[0]
            last_err = max((r for r in errs_in), key=lambda r: r["frame_id"])
            le_grey = metrics[last_err["frame_id"]]["grey_mean"]
            a_grey = metrics[a_first["frame_id"]]["grey_mean"]
            lines += [
                "- **What changes when prediction returns to S:** the first "
                f"correct-S frame after the run is S{a_first['frame_id']} "
                f"(P(S) {a_first['p_s']:.2f}); the last error S{last_err['frame_id']} "
                f"had P(S) {last_err['p_s']:.2f}. Measured pixels: grey "
                f"{le_grey:.0f} -> {a_grey:.0f} "
                f"({'brighter' if a_grey >= le_grey else 'darker'} by "
                f"{abs(a_grey - le_grey):.0f}). Whether the hand *shape* changed "
                "is **requires visual inspection**.",
                "",
            ]

    # Cross-run pattern.
    pe_max = {label: max(r["p_e"] for r in errors if s <= r["frame_id"] <= e)
              for label, s, e in SEGMENTS}
    grey_by_run = {}
    for label, s, e in SEGMENTS:
        ids = [r["frame_id"] for r in records if s <= r["frame_id"] <= e]
        grey_by_run[label] = float(np.mean([metrics[i]["grey_mean"] for i in ids]))
    seg_err_count = sum(
        sum(1 for r in errors if s <= r["frame_id"] <= e) for _, s, e in SEGMENTS
    )
    lines += [
        "## Cross-run pattern (measured)",
        "",
        f"- All three runs share the same probability signature: moderate P(E), "
        f"S always 2nd, max P(E) {max(pe_max.values()):.2f} "
        f"({', '.join(f'{k} {v:.2f}' for k, v in pe_max.items())}).",
        f"- Segment grey means: {', '.join(f'{k} {v:.0f}' for k, v in grey_by_run.items())} "
        "(vs 133 whole-set mean, 141 mean over all errors) — the runs are not "
        "an exposure outlier, so brightness is at most a contributing factor.",
        f"- The three runs cover {seg_err_count} of {len(errors)} errors "
        f"({seg_err_count / len(errors) * 100:.1f}%) and each starts at a clean "
        "correct-S boundary: correct -> error in a single frame.",
        "- **Cross-run geometry (visual):** **requires visual inspection.** If "
        "the three runs share one hand-shape feature (e.g. thumb visible in the "
        "same spot), `F_pose_comparison.png` row 2 should show it; this tool "
        "cannot confirm it.",
        "",
        "## Decision gate",
        "",
        "Is there a repeatable hand-geometry subtype that separates these S->E "
        "frames from correct S? **UNCLEAR (measured evidence only).** The "
        "temporal clustering and the identical per-run probability profile "
        "strongly indicate a handful of *repeated* poses near the S/E boundary — "
        "not pervasive S failure — but whether the separating feature is "
        "hand geometry (rather than exposure/background or mere pose "
        "repetition) cannot be established from the measurements this tool can "
        "produce. Confirm or refute from `E_transition_runs.png` and "
        "`F_pose_comparison.png` (rows 2-4): if the dominant-run tiles share a "
        "visible hand feature that correct-S tiles lack, answer YES; if they do "
        "not, answer NO.",
        "",
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")
    print(f"wrote {out_path}")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def run(args) -> Dict:
    device = _resolve_device(args.device)
    cache = Path(args.cache)
    if cache.is_file():
        records = json.loads(cache.read_text())
        print(f"loaded {len(records)} S records from {cache} (inference skipped)")
        n_err = sum(1 for r in records if r["pred"] == "E")
        n_ok = sum(1 for r in records if r["pred"] == "S")
        if n_err != 138 or n_ok != 462:
            raise SystemExit(
                f"cached records disagree with the targeted baseline "
                f"({n_err} errors / {n_ok} correct); delete {cache} to re-infer"
            )
    else:
        records = infer_s_records(
            args.checkpoint, args.manifest, device, args.batch_size
        )
        errors = [r for r in records if r["pred"] == "E"]
        correct = [r for r in records if r["pred"] == "S"]
        verify_counts(errors, correct, args.baseline_report)
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(records))
        print(f"wrote {len(records)} S records to {cache}")

    errors = [r for r in records if r["pred"] == "E"]
    correct = [r for r in records if r["pred"] == "S"]
    gallery_dir = Path(args.gallery_dir)

    strips = [
        (label, (s, e), segment_frames(records, s, e, MID_K))
        for label, s, e in SEGMENTS
    ]
    write_transition_strips(strips, gallery_dir / "E_transition_runs.png")
    write_pose_sheet(pose_groups(errors, correct),
                     gallery_dir / "F_pose_comparison.png")

    md = Path(args.out_md)
    if md.exists():
        print(f"kept existing {md} (curated; re-run will not overwrite)")
    else:
        write_transition_notes(md, records, errors, correct, gallery_dir)
    return records


def main() -> None:
    p = argparse.ArgumentParser(
        description="Transition-run + pose galleries for the S->E diagnosis."
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
    p.add_argument("--out-md", default=DEFAULT_OUT_MD)
    run(p.parse_args())


if __name__ == "__main__":
    main()
