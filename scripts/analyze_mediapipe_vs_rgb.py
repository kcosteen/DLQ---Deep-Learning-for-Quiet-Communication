"""Analyze MediaPipe HandLandmarker vs RGB EfficientNet on ASL alphabet images.

Reads the per-sample CSV from the Kaggle experiment and computes:
  - A-Z detection coverage
  - A-Z RGB accuracy (overall, detected, missed)
  - 2x2 outcome breakdown
  - Per-class metrics
  - Markdown + JSON reports
  - Bar-chart figures

No model loading, no inference, no torch/mediapipe needed.
"""

from __future__ import annotations

import json
import string
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = ROOT / "mediapipe_vs_rgb" / "mediapipe_vs_rgb_sample_records.csv"
REPORT_MD = ROOT / "docs" / "reports" / "mediapipe_miss_vs_rgb_accuracy.md"
REPORT_JSON = ROOT / "docs" / "reports" / "mediapipe_miss_vs_rgb_accuracy.json"
FIG_COVERAGE = ROOT / "docs" / "figures" / "mediapipe_detection_coverage_by_class.png"
FIG_ACCURACY = ROOT / "docs" / "figures" / "mediapipe_rgb_accuracy_comparison.png"

AZ_CLASSES: Tuple[str, ...] = tuple(c for c in string.ascii_uppercase)
EXPECTED_PER_CLASS = 100


# ---------------------------------------------------------------------------
# Core helpers (importable by tests)
# ---------------------------------------------------------------------------

def load_records(csv_path: Path = CSV_PATH) -> pd.DataFrame:
    """Load per-sample CSV and validate basic structure."""
    df = pd.read_csv(csv_path)
    required = {"class", "path", "found_hand", "prediction", "confidence", "correct"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    if df["path"].duplicated().any():
        raise ValueError("Duplicate file paths in records")
    df["found_hand"] = df["found_hand"].astype(str).str.lower().map({"true": True, "false": False})
    df["correct"] = df["correct"].astype(str).str.lower().map({"true": True, "false": False})
    return df


def filter_az(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only A-Z classes."""
    return df[df["class"].isin(AZ_CLASSES)].reset_index(drop=True)


def validate_az(df: pd.DataFrame, expected_per_class: int = EXPECTED_PER_CLASS) -> None:
    """Assert A-Z subset has expected shape and balance."""
    if len(df) != 26 * expected_per_class:
        raise ValueError(f"Expected {26 * expected_per_class} A-Z rows, got {len(df)}")
    counts = df["class"].value_counts()
    for cls in AZ_CLASSES:
        if counts.get(cls, 0) != expected_per_class:
            raise ValueError(f"Class {cls}: expected {expected_per_class}, got {counts.get(cls, 0)}")
    if df["path"].duplicated().any():
        raise ValueError("Duplicate paths in A-Z subset")


def compute_overall(df: pd.DataFrame) -> Dict[str, Any]:
    """Overall metrics for the A-Z subset."""
    n = len(df)
    detected = df["found_hand"].sum()
    missed = n - detected
    correct = df["correct"].sum()
    detected_df = df[df["found_hand"]]
    missed_df = df[~df["found_hand"]]
    correct_detected = detected_df["correct"].sum()
    correct_missed = missed_df["correct"].sum()
    return {
        "n": int(n),
        "detected": int(detected),
        "missed": int(missed),
        "mp_coverage_pct": round(100.0 * detected / n, 1),
        "rgb_correct": int(correct),
        "rgb_acc_pct": round(100.0 * correct / n, 2),
        "rgb_acc_detected_pct": round(100.0 * correct_detected / detected, 2) if detected else None,
        "rgb_acc_missed_pct": round(100.0 * correct_missed / missed, 2) if missed else None,
        "correct_detected": int(correct_detected),
        "correct_missed": int(correct_missed),
        "wrong_detected": int(detected - correct_detected),
        "wrong_missed": int(missed - correct_missed),
        "discarded_correct": int(correct_missed),
        "discarded_correct_pct_of_missed": round(100.0 * correct_missed / missed, 1) if missed else None,
        "discarded_correct_pct_of_total_correct": round(100.0 * correct_missed / correct, 1) if correct else None,
    }


def compute_per_class(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Per-class metrics for A-Z. Skips classes absent from df."""
    rows = []
    present = set(df["class"].unique())
    for cls in AZ_CLASSES:
        if cls not in present:
            continue
        cdf = df[df["class"] == cls]
        n = len(cdf)
        detected = int(cdf["found_hand"].sum())
        missed = n - detected
        correct = int(cdf["correct"].sum())
        det_df = cdf[cdf["found_hand"]]
        mis_df = cdf[~cdf["found_hand"]]
        correct_det = int(det_df["correct"].sum())
        correct_mis = int(mis_df["correct"].sum())
        rows.append({
            "class": cls,
            "n": n,
            "detected": detected,
            "missed": missed,
            "mp_coverage_pct": round(100.0 * detected / n, 1),
            "rgb_acc_pct": round(100.0 * correct / n, 2),
            "rgb_acc_detected_pct": round(100.0 * correct_det / detected, 2) if detected else None,
            "rgb_acc_missed_pct": round(100.0 * correct_mis / missed, 2) if missed else None,
            "correct_missed": correct_mis,
        })
    return rows


# ---------------------------------------------------------------------------
# Report writers
# ---------------------------------------------------------------------------

def write_json_report(overall: Dict, per_class: List[Dict], out: Path = REPORT_JSON) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "description": "MediaPipe HandLandmarker miss vs RGB EfficientNet accuracy on A-Z ASL alphabet",
        "dataset": "grassknoted/asl-alphabet (100 deterministic samples/class, seed 42)",
        "checkpoint": "checkpoints/efficientnet_b0_targeted.pt",
        "scope": "A-Z only (26 classes, 2600 images)",
        "overall": overall,
        "per_class": per_class,
    }
    out.write_text(json.dumps(payload, indent=2) + "\n")


def write_markdown_report(
    overall: Dict,
    per_class: List[Dict],
    out: Path = REPORT_MD,
) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    lines: List[str] = []

    def w(s: str = "") -> None:
        lines.append(s)

    w("# MediaPipe HandLandmarker Miss vs RGB EfficientNet Accuracy")
    w()
    w("## Experimental question")
    w()
    w("> Does failure of MediaPipe HandLandmarker imply that an ASL image is not")
    w("> classifiable by the RGB EfficientNet model?")
    w()
    w("## Setup")
    w()
    w("| Parameter | Value |")
    w("| --------- | ----- |")
    w("| Dataset | `grassknoted/asl-alphabet` (Kaggle) |")
    w("| Samples per class | 100 (deterministic, seed 42) |")
    w("| Classes | A\u2013Z only (26 classes, 2600 images) |")
    w("| MediaPipe | Tasks API HandLandmarker (`hand_landmarker.task`) |")
    w("| RGB model | `efficientnet_b0_targeted.pt` (EfficientNet-B0, 29-class head) |")
    w("| RGB path | raw image \u2192 `preprocess_bgr` \u2192 backbone \u2192 head \u2192 top-1 |")
    w("| Hand crop before EfficientNet | **No** |")
    w()
    w("**Note:** The three special classes (`del`, `nothing`, `space`) are excluded because")
    w("the Kaggle notebook used an unverified manual output-label mapping that does not match")
    w("the checkpoint\u2019s training order (`space`/`delete`/`nothing`). Their RGB metrics are")
    w("invalid and are not reported here.")
    w()

    # Detection coverage
    w("## Detection coverage")
    w()
    w(f"Overall MediaPipe coverage on A\u2013Z: **{overall['mp_coverage_pct']}%** ({overall['detected']}/{overall['n']}).")
    w()
    w("| Class | Detected | Missed | Coverage |")
    w("| ----- | -------: | -----: | -------: |")
    for pc in per_class:
        w(f"| {pc['class']} | {pc['detected']} | {pc['missed']} | {pc['mp_coverage_pct']}% |")
    w()

    # RGB accuracy
    w("## RGB accuracy: detected vs missed (A\u2013Z)")
    w()
    w(f"Overall RGB accuracy: **{overall['rgb_acc_pct']}%** ({overall['rgb_correct']}/{overall['n']}).")
    w()
    w(f"| Condition | Accuracy |")
    w(f"| --------- | -------: |")
    w(f"| All A\u2013Z frames | {overall['rgb_acc_pct']}% |")
    w(f"| MediaPipe detected | {overall['rgb_acc_detected_pct']}% ({overall['correct_detected']}/{overall['detected']}) |")
    w(f"| MediaPipe missed | {overall['rgb_acc_missed_pct']}% ({overall['correct_missed']}/{overall['missed']}) |")
    w()

    # 2x2
    w("## 2\u00d72 outcome breakdown (A\u2013Z)")
    w()
    w("| | RGB Correct | RGB Wrong | Total |")
    w("| - | ----------: | --------: | ----: |")
    w(f"| **MP Detected** | {overall['correct_detected']} | {overall['wrong_detected']} | {overall['detected']} |")
    w(f"| **MP Missed** | {overall['correct_missed']} | {overall['wrong_missed']} | {overall['missed']} |")
    w(f"| **Total** | {overall['rgb_correct']} | {overall['n'] - overall['rgb_correct']} | {overall['n']} |")
    w()
    w(f"**{overall['discarded_correct']}** correctly classified frames ({overall['discarded_correct_pct_of_total_correct']}% of all correct)")
    w(f"would be discarded by a mandatory MediaPipe gate.")
    w()

    # Per-class table
    w("## Per-class detail")
    w()
    w("| Class | MP coverage | RGB acc all | RGB acc detected | RGB acc missed | RGB-correct discarded |")
    w("| ----- | ----------: | ----------: | ---------------: | -------------: | ---------------------: |")
    for pc in per_class:
        det_str = f"{pc['rgb_acc_detected_pct']}%" if pc['detected'] else "\u2014"
        mis_str = f"{pc['rgb_acc_missed_pct']}%" if pc['missed'] else "\u2014"
        w(f"| {pc['class']} | {pc['mp_coverage_pct']}% | {pc['rgb_acc_pct']}% | {det_str} | {mis_str} | {pc['correct_missed']} |")
    w()

    # M/N case study
    m = next(pc for pc in per_class if pc["class"] == "M")
    n = next(pc for pc in per_class if pc["class"] == "N")
    w("## M/N case study")
    w()
    w("M and N are among the hardest classes for MediaPipe hand detection, likely due to")
    w("similar hand shapes where landmarks are harder to distinguish.")
    w()
    w(f"| | M | N |")
    w(f"| - | -: | -: |")
    w(f"| MP coverage | {m['mp_coverage_pct']}% ({m['detected']}/{m['n']}) | {n['mp_coverage_pct']}% ({n['detected']}/{n['n']}) |")
    w(f"| RGB acc (detected) | {m['rgb_acc_detected_pct']}% | {n['rgb_acc_detected_pct']}% |")
    w(f"| RGB acc (missed) | {m['rgb_acc_missed_pct']}% | {n['rgb_acc_missed_pct']}% |")
    w(f"| RGB-correct discarded | {m['correct_missed']} | {n['correct_missed']} |")
    w()
    w("Despite low MediaPipe coverage, the RGB model still classifies the majority of missed")
    w("M/N frames correctly, reinforcing that MediaPipe failure is not equivalent to")
    w("classification failure.")
    w()

    # Interpretation
    w("## Interpretation")
    w()
    w("> MediaPipe failure is not equivalent to image unclassifiability for the RGB model.")
    w()
    w(f"On this dataset and checkpoint, {overall['discarded_correct_pct_of_missed']}% of frames where MediaPipe")
    w(f"fails to detect a hand are still classified correctly by the RGB EfficientNet")
    w(f"({overall['correct_missed']}/{overall['missed']} missed frames). A mandatory HandLandmarker gate")
    w("before classification would discard these correctly classifiable frames.")
    w()
    w("The discarding is **class-dependent**: classes like M and N have much lower MediaPipe")
    w("coverage (~46\u201348%) than classes like E (84%), meaning a mandatory gate would")
    w("disproportionately lose information for specific hand shapes.")
    w()
    w("**Scope:** These findings apply only to this MediaPipe configuration, this dataset,")
    w("this deterministic 100-sample-per-class subset, and this checkpoint. They do not")
    w("imply that MediaPipe is universally unreliable.")
    w()

    # Provenance
    w("## Provenance")
    w()
    w("Raw experiment data: `mediapipe_vs_rgb/` (generated on Kaggle,")
    w("dataset `grassknoted/asl-alphabet`, seed 42, 100 samples/class).")
    w("Original CSVs preserved; this report computed from `mediapipe_vs_rgb_sample_records.csv`.")
    w()

    out.write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def plot_coverage(per_class: List[Dict], out: Path = FIG_COVERAGE) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    classes = [pc["class"] for pc in per_class]
    coverage = [pc["mp_coverage_pct"] for pc in per_class]
    colors = ["#e74c3c" if c < 50 else "#f39c12" if c < 70 else "#2ecc71" for c in coverage]

    fig, ax = plt.subplots(figsize=(14, 5))
    bars = ax.bar(classes, coverage, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_ylabel("MediaPipe detection coverage (%)")
    ax.set_xlabel("Class")
    ax.set_title("MediaPipe HandLandmarker Detection Coverage by ASL Class (A\u2013Z)")
    ax.set_ylim(0, 105)
    ax.axhline(y=np.mean(coverage), color="#3498db", linestyle="--", linewidth=1.2, label=f"Mean ({np.mean(coverage):.1f}%)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    for bar, val in zip(bars, coverage):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.2, f"{val:.0f}",
                ha="center", va="bottom", fontsize=7)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_accuracy_comparison(per_class: List[Dict], out: Path = FIG_ACCURACY) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    classes = [pc["class"] for pc in per_class]
    acc_det = [pc["rgb_acc_detected_pct"] for pc in per_class]
    acc_mis = [pc["rgb_acc_missed_pct"] for pc in per_class]

    x = np.arange(len(classes))
    width = 0.38

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(x - width / 2, acc_det, width, label="RGB acc (MP detected)", color="#2ecc71", edgecolor="white", linewidth=0.5)
    ax.bar(x + width / 2, acc_mis, width, label="RGB acc (MP missed)", color="#e67e22", edgecolor="white", linewidth=0.5)
    ax.set_ylabel("RGB accuracy (%)")
    ax.set_xlabel("Class")
    ax.set_title("RGB EfficientNet Accuracy: MediaPipe-Detected vs MediaPipe-Missed Frames")
    ax.set_xticks(x)
    ax.set_xticklabels(classes)
    ax.set_ylim(70, 105)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    df = load_records()
    az = filter_az(df)
    validate_az(az)
    overall = compute_overall(az)
    per_class = compute_per_class(az)

    write_json_report(overall, per_class)
    write_markdown_report(overall, per_class)
    plot_coverage(per_class)
    plot_accuracy_comparison(per_class)

    print(f"A-Z samples:       {overall['n']}")
    print(f"MP detected:       {overall['detected']}")
    print(f"MP missed:         {overall['missed']}")
    print(f"MP coverage:       {overall['mp_coverage_pct']}%")
    print(f"RGB acc (all):     {overall['rgb_acc_pct']}%")
    print(f"RGB acc (detected):{overall['rgb_acc_detected_pct']}%")
    print(f"RGB acc (missed):  {overall['rgb_acc_missed_pct']}%")
    print(f"Discarded correct: {overall['discarded_correct']} ({overall['discarded_correct_pct_of_total_correct']}% of all correct)")
    print(f"Reports written to {REPORT_MD}")
    print(f"  and {REPORT_JSON}")
    print(f"Figures: {FIG_COVERAGE}")
    print(f"  and   {FIG_ACCURACY}")


if __name__ == "__main__":
    main()
