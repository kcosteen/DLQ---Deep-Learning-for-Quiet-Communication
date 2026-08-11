"""Rank directional class confusions from an ablation confusion matrix.

Answers: *which directional confusions cause the most errors in the best /
current ablation model?*

Source of truth is the ablation confusion matrix only. By default this is
``docs/figures/confusion_matrix_ablate.csv``; an optional ``--report`` JSON is
read **only** for its ``n_images`` field and provenance label — confusion
counts are never mixed with results from the older targeted checkpoint.

Directionality is preserved: ``M -> N`` and ``N -> M`` are separate rows.

This task is purely descriptive — no model is loaded, no preprocessing or
training code is touched.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd

DEFAULT_CONFUSION = Path("docs/figures/confusion_matrix_ablate.csv")
DEFAULT_REPORT = Path("docs/reports/dev_val_ablate.json")
DEFAULT_OUT = Path("docs/reports/confusion_ranking_ablate.md")

# (true_class, pred_class, count)
Confusion = Tuple[str, str, int]


def load_confusion_csv(path: Path) -> Tuple[List[str], np.ndarray]:
    """Load a ``*,A,B,...,nothing`` CSV into (classes, square int matrix).

    The header's first cell is empty (row-label column); the first column
    holds the true-class labels. Rows are true classes, columns predictions.
    """
    frame = pd.read_csv(path, index_col=0)
    if not frame.columns.equals(frame.index):
        raise ValueError(
            f"{path}: row labels do not match column labels "
            f"(got {list(frame.columns)} vs {list(frame.index)})"
        )
    if frame.shape[0] != frame.shape[1]:
        raise ValueError(f"{path}: confusion matrix is not square ({frame.shape})")
    classes = [str(c) for c in frame.index]
    matrix = frame.to_numpy(dtype=int)
    if matrix.min() < 0:
        raise ValueError(f"{path}: confusion matrix contains negative counts")
    return classes, matrix


def off_diagonal_total(matrix: np.ndarray) -> int:
    """Sum of every cell off the main diagonal."""
    return int(matrix.sum() - np.trace(matrix))


def rank_confusions(
    matrix: np.ndarray, classes: Sequence[str], top: int = 20
) -> List[Confusion]:
    """Rank directional (true, pred) confusions by raw error count.

    Diagonal cells and zero-count pairs are excluded. Ties are broken by
    (true, pred) so output is deterministic.
    """
    n = len(classes)
    counts = [
        (int(matrix[i, j]), classes[i], classes[j])
        for i in range(n)
        for j in range(n)
        if i != j and matrix[i, j] > 0
    ]
    counts.sort(key=lambda row: (-row[0], row[1], row[2]))
    return [(true, pred, count) for count, true, pred in counts][:top]


def symmetric_pairs(
    matrix: np.ndarray, classes: Sequence[str], top: int = 10
) -> List[Confusion]:
    """Combine both directions: ``(a, b)`` errors = ``a->b`` + ``b->a``."""
    n = len(classes)
    pairs = [
        (
            classes[i],
            classes[j],
            int(matrix[i, j]) + int(matrix[j, i]),
        )
        for i in range(n)
        for j in range(i + 1, n)
        if matrix[i, j] + matrix[j, i] > 0
    ]
    pairs.sort(key=lambda row: (-row[2], row[0], row[1]))
    return pairs[:top]


def print_table(rows: List[Confusion]) -> None:
    if not rows:
        print("  (none)")
        return
    width = max(len(true) for true, _, _ in rows) if rows else 0
    width = max(width, len(str(len(rows))))
    print(f"{'Rank':<4}  {'True':<{width}}  {'Pred':<{width}}  {'Errors':>6}")
    for rank, (true, pred, count) in enumerate(rows, start=1):
        print(f"{rank:<4}  {true:<{width}}  {pred:<{width}}  {count:>6}")


def write_markdown(
    out: Path,
    matrix: np.ndarray,
    directional: List[Confusion],
    symmetric: List[Confusion],
    total_samples: int,
    source: Path,
    report_path: Path | None,
) -> None:
    total_errors = off_diagonal_total(matrix)
    provenance = f"`{source}`"
    if report_path is not None:
        provenance += f" and report `{report_path}` (for sample count only)"
    header = [
        "# Confusion ranking — ablation results",
        "",
        "This report is based on the **ablation checkpoint** "
        "(`efficientnet_b0_ablate.pt`, no-lift fresh train) and its confusion "
        "matrix — **not** the older targeted checkpoint. Directional rows are "
        "kept separate (`M -> N` and `N -> M` are distinct).",
        "",
        f"- Total validation samples: **{total_samples}**",
        f"- Total misclassifications: **{total_errors}**",
        f"- Overall accuracy (derived): **{1.0 - total_errors / total_samples:.4f}**",
        "",
        f"Source: {provenance}",
        "",
    ]
    body = [
        "## Top 20 directional confusions",
        "",
        "| Rank | True | Pred | Errors |",
        "| ---: | :--- | :--- | -----: |",
    ]
    for rank, (true, pred, count) in enumerate(directional, start=1):
        body.append(f"| {rank} | {true} | {pred} | {count} |")
    body += [
        "",
        "## Top 10 symmetric pairs",
        "",
        "| Rank | Pair | Total errors |",
        "| ---: | :--- | -----------: |",
    ]
    for rank, (a, b, count) in enumerate(symmetric, start=1):
        body.append(f"| {rank} | {a} <-> {b} | {count} |")
    body += [""]

    out.write_text("\n".join(header + body))
    print(f"Wrote {out}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rank directional confusions from the ablation confusion matrix."
    )
    parser.add_argument(
        "--confusion", type=Path, default=DEFAULT_CONFUSION,
        help="ablation confusion matrix CSV (default: %(default)s)",
    )
    parser.add_argument(
        "--report", type=Path, default=DEFAULT_REPORT, nargs="?",
        help="ablation report JSON for the validation-sample count "
             "(default: %(default)s); pass --report None to skip",
    )
    parser.add_argument("--top", type=int, default=20,
                        help="how many directional rows to print (default: %(default)s)")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT,
                        help="markdown artifact (default: %(default)s)")
    args = parser.parse_args()

    if args.top < 1:
        parser.error("--top must be >= 1")

    classes, matrix = load_confusion_csv(args.confusion)
    total_samples = int(matrix.sum())

    report_path = args.report
    if args.report is not None and str(args.report).lower() != "none" and args.report.exists():
        report = json.loads(Path(args.report).read_text())
        reported_n = report.get("n_images")
        if reported_n is not None and int(reported_n) != total_samples:
            print(
                f"WARNING: report n_images ({reported_n}) disagrees with CSV sum "
                f"({total_samples})"
            )
        total_samples = int(reported_n or total_samples)
    elif args.report is not None and str(args.report).lower() != "none":
        print(f"WARNING: --report {args.report} not found; using CSV sum only")

    directional = rank_confusions(matrix, classes, top=args.top)
    symmetric = symmetric_pairs(matrix, classes, top=10)

    print("Top directional confusions\n")
    print(f"{'Rank':<4}  {'True':<5}  {'Pred':<5}  {'Errors':>6}")
    for rank, (true, pred, count) in enumerate(directional, start=1):
        print(f"{rank:<4}  {true:<5}  {pred:<5}  {count:>6}")
    print(f"\nTotal off-diagonal errors: {off_diagonal_total(matrix)}")
    print(f"Total validation samples:  {total_samples}")
    print("\nTop 10 symmetric pairs")
    print_table(symmetric)

    write_markdown(
        out=args.out,
        matrix=matrix,
        directional=directional,
        symmetric=symmetric,
        total_samples=total_samples,
        source=args.confusion,
        report_path=report_path if report_path.exists() else None,
    )


if __name__ == "__main__":
    main()
