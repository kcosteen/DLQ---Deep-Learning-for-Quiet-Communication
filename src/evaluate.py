"""Evaluation harness: metrics, confusion matrix, and the webcam-set eval.

Two evaluation modes:

1. ``--split``  : accuracy on the leakage-safe frame-range val split. Useful for
   development, but NOT the number we report.
2. ``--webcam`` : accuracy on ``data/webcam_testset`` — folder-per-class crops the
   team recorded. THIS is the real, reported benchmark.

Emits: overall accuracy, macro-F1, per-class accuracy, and a confusion matrix
(PNG + CSV). Flags the classic confusions to watch: M/N/S/T, A/E, K/V.

Works on either architecture, so the transfer model can be scored against the
compact-CNN baseline (#5) on the same split: the architecture comes from the
checkpoint's ``arch`` field, or from ``--model`` for checkpoints predating it.
Pass ``--manifest`` to score the exact split that was trained on rather than
re-deriving it from ``root``.

``--report-json`` writes the whole result — overall, macro-F1, per-class
accuracy and the full confusion matrix — as one machine-readable file. Feed a
previous one back with ``--baseline`` to get a before/after table (#12), or to
``train.py --rebalance-from`` to weight training toward the classes that are
actually failing.

Targets: leakage-safe val > 95%; webcam test >= 90%; no class below 85%.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from collections import defaultdict
from datetime import datetime, timezone
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import CLASSES, ASLImageDataset, Sample, load_or_derive_split
from .model import build_model

PER_CLASS_TARGET = 0.85  # docs/PROJECT_PLAN.md: no class below 85%
REPORT_SCHEMA = "asl-class-report/v1"

# Groups to inspect explicitly. The first three are the textbook fingerspelling
# confusions; the rest were measured on the first transfer run (#6, see
# docs/RESULTS.md), where the textbook list missed the two largest failures.
WATCH_CONFUSIONS = [
    ("M", "N", "S", "T"),       # textbook — mild in practice (58 errors)
    ("A", "E", "S"),            # S->E was 160 errors. S and E previously sat in
                                # separate groups, so the pair could not be
                                # reported no matter how large it got.
    ("K", "V"),                 # V->K, 219 errors: the single worst collision
    ("X", "A", "L", "R", "I"),  # X spreads its errors instead of picking one
    ("I", "Y"),                 # Y->I, 26
]


def resolve_arch(ckpt: dict, requested: Optional[str]) -> str:
    """Which architecture to rebuild: explicit ``--model`` beats the checkpoint.

    Checkpoints written since #6 record their own ``arch``, so the flag is
    normally unnecessary. It stays available for the checkpoints trained before
    that (which carry no ``arch``) and defaults to ``efficientnet`` — the only
    architecture this harness could load previously, so old commands keep working.
    """
    if requested is not None:
        recorded = ckpt.get("arch")
        if recorded is not None and recorded != requested:
            print(f"warning: --model {requested} overrides the architecture "
                  f"recorded in the checkpoint ({recorded})")
        return requested
    return ckpt.get("arch", "efficientnet")


def checkpoint_arch(path: str) -> str:
    """The architecture a checkpoint records, without building the model.

    Used for provenance in ``--report-json``: a report that does not name the
    architecture cannot be compared against one from a different model.
    """
    ckpt = torch.load(path, map_location="cpu")
    return resolve_arch(ckpt if isinstance(ckpt, dict) else {}, None)


def load_checkpoint(
    path: str, device: str, arch: Optional[str] = None
) -> torch.nn.Module:
    ckpt = torch.load(path, map_location=device)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    name = resolve_arch(ckpt if isinstance(ckpt, dict) else {}, arch)
    model = build_model(name, pretrained=False).to(device)
    try:
        model.load_state_dict(state)
    except RuntimeError as e:
        # The usual cause is the wrong architecture. torch reports that as every
        # unmatched parameter name — thousands of them for EfficientNet — which
        # buries the one-line fix, so only the summary line is kept.
        summary = str(e).splitlines()[0]
        raise SystemExit(
            f"checkpoint does not fit a '{name}' model.\n"
            f"  Pass --model to name the right architecture "
            f"(efficientnet | compact).\n  {summary}"
        ) from e
    model.eval()
    print(f"model: {name}  (checkpoint {path})")
    return model


def _webcam_samples(root: str) -> List[Sample]:
    from .data import CLASS_TO_IDX, list_class_files

    samples: List[Sample] = []
    for c in CLASSES:
        for f in list_class_files(root, c):
            samples.append(Sample(f, CLASS_TO_IDX[c], c))
    return samples


@torch.no_grad()
def predict_all(model, loader, device) -> Tuple[np.ndarray, np.ndarray]:
    ys, ps = [], []
    for x, y in loader:
        x = x.to(device)
        ps.append(model(x).argmax(1).cpu().numpy())
        ys.append(y.numpy())
    return np.concatenate(ys), np.concatenate(ps)


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n: int) -> np.ndarray:
    cm = np.zeros((n, n), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def per_class_accuracy(cm: np.ndarray) -> Dict[str, float]:
    accs = {}
    for i, c in enumerate(CLASSES):
        total = cm[i].sum()
        accs[c] = float(cm[i, i] / total) if total else float("nan")
    return accs


def macro_f1(cm: np.ndarray) -> float:
    f1s = []
    for i in range(len(CLASSES)):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        f1s.append(f1)
    return float(np.mean(f1s))


def top_confusions(cm: np.ndarray, k: int = 10) -> List[Tuple[int, str, str]]:
    """The ``k`` largest off-diagonal cells, worst first, as (count, true, pred).

    Reported unconditionally, because ``WATCH_CONFUSIONS`` only finds what it was
    told to look for. On the #6 run the two biggest failures were X (spread over
    A/L/R/I, and in no group at all) and S->E (split across two groups, so
    invisible to both) — the classic list named neither. A ranking derived from
    the matrix itself cannot miss a confusion for being unanticipated.
    """
    pairs = [
        (int(cm[i, j]), CLASSES[i], CLASSES[j])
        for i in range(len(CLASSES))
        for j in range(len(CLASSES))
        if i != j and cm[i, j]
    ]
    pairs.sort(reverse=True)
    return pairs[:k]


def save_confusion_png(cm: np.ndarray, path: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"skip PNG (matplotlib unavailable: {e})")
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig, ax = plt.subplots(figsize=(11, 10))
    im = ax.imshow(cm, cmap="viridis")
    ax.set_xticks(range(len(CLASSES)))
    ax.set_yticks(range(len(CLASSES)))
    ax.set_xticklabels(CLASSES, rotation=90, fontsize=7)
    ax.set_yticklabels(CLASSES, fontsize=7)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title("ASL fingerspelling confusion matrix")
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"wrote {path}")


def save_confusion_csv(cm: np.ndarray, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([""] + list(CLASSES))
        for i, c in enumerate(CLASSES):
            w.writerow([c] + cm[i].tolist())
    print(f"wrote {path}")


def build_report(
    cm: np.ndarray,
    *,
    checkpoint: str = "",
    arch: str = "",
    source_kind: str = "dev_val",
    source_path: str = "",
    manifest: Optional[str] = None,
    manifest_sha256: Optional[str] = None,
    target_per_class: float = PER_CLASS_TARGET,
    label: Optional[str] = None,
) -> Dict:
    """Everything the printed report says, as a JSON-serialisable dict.

    Carries the full confusion matrix, not just the summary, so a saved report is
    enough to re-derive any metric later — and enough for ``train.py`` to work out
    which classes to weight and which to augment gently without re-running the
    evaluation.

    ``source_kind`` is recorded because the two numbers this project produces mean
    completely different things: ``dev_val`` is the same-signer temporal split and
    is NOT reportable; ``webcam`` is the benchmark. A file that does not say which
    one it is would eventually be quoted as the wrong one.
    """
    accs = per_class_accuracy(cm)
    return {
        "schema": REPORT_SCHEMA,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "label": label or source_kind,
        "checkpoint": checkpoint,
        "arch": arch,
        "source": {
            "kind": source_kind,
            "path": source_path,
            "manifest": manifest,
            "manifest_sha256": manifest_sha256,
        },
        "is_reported_metric": source_kind == "webcam",
        "note": (
            "REPORTED BENCHMARK: own webcam test set (#15/#16)."
            if source_kind == "webcam"
            else "DEV ONLY — leakage-safe frame-range split, same signer, same "
                 "room. NOT the reported number (that is the webcam set, #15/#16)."
        ),
        "n_images": int(cm.sum()),
        "overall_accuracy": float(cm.trace() / max(1, cm.sum())),
        "macro_f1": macro_f1(cm),
        "per_class_target": target_per_class,
        "classes": list(CLASSES),
        "per_class": {
            c: {"accuracy": accs[c] if not np.isnan(accs[c]) else None,
                "n": int(cm[i].sum())}
            for i, c in enumerate(CLASSES)
        },
        "below_target": [
            c for i, c in enumerate(CLASSES)
            if cm[i].sum() and not np.isnan(accs[c]) and accs[c] < target_per_class
        ],
        "top_confusions": [
            {"true": t, "pred": p, "n": n} for n, t, p in top_confusions(cm, 15)
        ],
        "confusion_matrix": cm.astype(int).tolist(),
    }


def write_report(report_dict: Dict, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        json.dump(report_dict, f, indent=2, sort_keys=True)
        f.write("\n")
    print(f"wrote {path}")


def _report_from_matrix(cm: np.ndarray, label: str) -> Dict:
    return build_report(cm, label=label, source_kind="unknown")


def load_confusion_csv(path: str) -> np.ndarray:
    """Read a ``confusion_matrix.csv`` written by ``save_confusion_csv``.

    Needed because the #6 run predates ``--report-json`` and left only the CSV,
    which is the sole record of the "before" numbers #12 has to improve on.
    Rows/columns are matched by class name rather than position, so a CSV written
    with a different class order still loads correctly.
    """
    with open(path, newline="") as f:
        rows = list(csv.reader(f))
    if not rows:
        raise ValueError(f"{path} is empty")
    header = [h.strip() for h in rows[0][1:]]
    idx = {c: i for i, c in enumerate(CLASSES)}
    missing = [c for c in header if c not in idx]
    if missing:
        raise ValueError(f"{path} has unknown classes: {missing}")
    cm = np.zeros((len(CLASSES), len(CLASSES)), dtype=np.int64)
    for row in rows[1:]:
        if not row or not row[0].strip():
            continue
        true = row[0].strip()
        if true not in idx:
            raise ValueError(f"{path} has unknown class row: {true!r}")
        for col, value in zip(header, row[1:]):
            cm[idx[true], idx[col]] = int(value or 0)
    return cm


def load_report(path: str) -> Dict:
    """Load a report from ``--report-json`` output, or from a confusion-matrix CSV."""
    if path.lower().endswith(".csv"):
        return _report_from_matrix(load_confusion_csv(path), label=os.path.basename(path))
    with open(path) as f:
        data = json.load(f)
    if data.get("schema") != REPORT_SCHEMA:
        raise ValueError(
            f"{path}: expected schema {REPORT_SCHEMA}, got {data.get('schema')!r}"
        )
    return data


def report_matrix(report_dict: Dict) -> np.ndarray:
    """The confusion matrix from a loaded report, reindexed onto ``CLASSES``."""
    cm = np.asarray(report_dict["confusion_matrix"], dtype=np.int64)
    classes = report_dict.get("classes", list(CLASSES))
    if list(classes) == list(CLASSES):
        return cm
    order = [classes.index(c) for c in CLASSES]
    return cm[np.ix_(order, order)]


def compare_reports(
    before: Dict, after: Dict, target: float = PER_CLASS_TARGET
) -> Dict[str, List[str]]:
    """Print a before/after per-class table; return the classes in each bucket.

    This is the #12 deliverable that a single accuracy number cannot express: the
    point is not "did the mean go up" but "did the three broken classes get fixed
    without breaking a fourth". Regressions are called out explicitly because the
    obvious way to fix S, V and X — weight them harder — pays for it somewhere
    else, and a mean that improved would hide that.
    """
    b, a = before["per_class"], after["per_class"]
    fixed, regressed, still_below, newly_below = [], [], [], []

    print(f"\nbefore/after per-class accuracy ({before.get('label', 'before')} -> "
          f"{after.get('label', 'after')}):")
    print(f"  {'class':>8} {'before':>8} {'after':>8} {'delta':>8}")
    for c in CLASSES:
        bv = (b.get(c) or {}).get("accuracy")
        av = (a.get(c) or {}).get("accuracy")
        if bv is None or av is None:
            continue
        delta = av - bv
        was_below, is_below = bv < target, av < target
        flag = ""
        if was_below and not is_below:
            fixed.append(c)
            flag = "  FIXED"
        elif was_below:
            still_below.append(c)
            flag = "  still below"
        elif is_below:
            newly_below.append(c)
            flag = "  <-- NEW REGRESSION below target"
        if delta < -0.005 and not is_below:
            regressed.append(c)
        # Unchanged, already-passing classes are the majority (21 sat at 1.000);
        # printing them all buries the four lines that matter.
        if flag or abs(delta) >= 0.005:
            print(f"  {c:>8} {bv:>8.3f} {av:>8.3f} {delta:>+8.3f}{flag}")

    for key in ("overall_accuracy", "macro_f1"):
        bv, av = before.get(key), after.get(key)
        if bv is not None and av is not None:
            print(f"  {key:>18}: {bv:.4f} -> {av:.4f} ({av - bv:+.4f})")

    if newly_below:
        print(f"\nREGRESSION: {newly_below} fell below the "
              f"{target:.0%} bar that they previously met.")
    if fixed:
        print(f"fixed (now >= {target:.0%}): {fixed}")
    if still_below:
        print(f"still below {target:.0%}: {still_below}")
    if not still_below and not newly_below:
        print(f"\nall classes meet the {target:.0%} per-class bar.")
    return {"fixed": fixed, "still_below": still_below,
            "newly_below": newly_below, "softly_regressed": regressed}


def dump_errors(
    samples: Sequence[Sample], y_true: np.ndarray, y_pred: np.ndarray,
    out_dir: str, max_per_pair: int = 12,
) -> int:
    """Copy misclassified source frames into ``out_dir/TRUE_to_PRED/`` for eyeballing.

    X's 296 errors spread over A, L, R and I with no single culprit (#6), which is
    the one failure shape a confusion matrix cannot explain — it says the model
    never formed a stable X, but not why. Looking at the actual frames is the
    cheapest next step, and this is what makes that a one-liner. Copies the
    ORIGINAL file, not the normalised tensor: the question is what the frame
    contains, not what the loader did to it.
    """
    per_pair: Dict[Tuple[int, int], int] = defaultdict(int)
    written = 0
    for sample, t, p in zip(samples, y_true, y_pred):
        if t == p:
            continue
        key = (int(t), int(p))
        if per_pair[key] >= max_per_pair:
            continue
        per_pair[key] += 1
        pair_dir = os.path.join(out_dir, f"{CLASSES[t]}_to_{CLASSES[p]}")
        os.makedirs(pair_dir, exist_ok=True)
        shutil.copy(sample.path, os.path.join(pair_dir, os.path.basename(sample.path)))
        written += 1
    print(f"wrote {written} misclassified frames (<= {max_per_pair} per pair) "
          f"-> {out_dir}")
    return written


def report(cm: np.ndarray, target_per_class: float = PER_CLASS_TARGET) -> None:
    overall = cm.trace() / max(1, cm.sum())
    print(f"\noverall accuracy: {overall:.4f}")
    print(f"macro-F1:         {macro_f1(cm):.4f}")
    accs = per_class_accuracy(cm)
    print("\nper-class accuracy:")
    below = []
    for c in CLASSES:
        flag = "" if (np.isnan(accs[c]) or accs[c] >= target_per_class) else "  <-- BELOW 85%"
        if flag:
            below.append(c)
        print(f"  {c:>8}: {accs[c]:.3f}{flag}")
    top = top_confusions(cm)
    if top:
        print("\nlargest confusions (true -> predicted):")
        for n, t, p in top:
            print(f"  {t} -> {p}: {n}")

    print("\nwatch these known-confusable groups:")
    for group in WATCH_CONFUSIONS:
        idxs = [CLASSES.index(g) for g in group if g in CLASSES]
        for i in idxs:
            confused = {CLASSES[j]: int(cm[i, j]) for j in idxs if j != i and cm[i, j]}
            if confused:
                print(f"  {CLASSES[i]} -> {confused}")
    if below:
        print(f"\nFAIL: classes below 85%: {below}")
    else:
        print("\nall classes meet the 85% per-class bar.")


def run(args: argparse.Namespace) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_checkpoint(args.checkpoint, device, args.model)
    arch = args.model or checkpoint_arch(args.checkpoint)

    manifest_path = manifest_sha = None
    if args.webcam:
        samples = _webcam_samples(args.webcam)
        if not samples:
            raise SystemExit(f"no images under {args.webcam}")
        ds = ASLImageDataset(samples, train=False)
        source_kind, source_path = "webcam", args.webcam
        print(f"WEBCAM TEST SET (the reported benchmark): {len(ds)} images")
    else:
        manifest = load_or_derive_split(args.manifest, args.split, args.train_frac)
        if args.manifest:
            manifest_path = args.manifest
            with open(args.manifest) as f:
                manifest_sha = json.load(f).get("content_sha256")
        ds = ASLImageDataset(manifest.val, train=False)
        source_kind, source_path = "dev_val", args.split
        print(f"LEAKAGE-SAFE VAL SPLIT (dev only, not reported): {len(ds)} images")

    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers)
    y_true, y_pred = predict_all(model, loader, device)
    cm = confusion_matrix(y_true, y_pred, len(CLASSES))
    report(cm)
    if args.figures:
        save_confusion_png(cm, os.path.join(args.figures, "confusion_matrix.png"))
        save_confusion_csv(cm, os.path.join(args.figures, "confusion_matrix.csv"))

    current = build_report(
        cm, checkpoint=args.checkpoint, arch=arch,
        source_kind=source_kind, source_path=source_path or "",
        manifest=manifest_path, manifest_sha256=manifest_sha, label=args.label,
    )
    if args.report_json:
        write_report(current, args.report_json)
    if args.baseline:
        compare_reports(load_report(args.baseline), current)
    if args.dump_errors:
        dump_errors(ds.samples, y_true, y_pred, args.dump_errors, args.dump_errors_max)


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate a checkpoint.")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--model", choices=["efficientnet", "compact"], default=None,
                   help="architecture to rebuild; default: whatever the "
                        "checkpoint records, else efficientnet")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--split", help="asl_alphabet_train/ dir -> dev val split")
    g.add_argument("--webcam", help="data/webcam_testset dir -> reported benchmark")
    p.add_argument("--manifest", default=None)
    p.add_argument("--train-frac", type=float, default=0.8)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--figures", default="docs/figures",
                   help="dir for confusion PNG/CSV (omit to skip)")
    p.add_argument("--report-json", default=None,
                   help="write the full per-class report + confusion matrix here; "
                        "feed it to --baseline or to train.py --rebalance-from")
    p.add_argument("--baseline", default=None,
                   help="a previous --report-json (or confusion_matrix.csv) to "
                        "diff against; prints the before/after table (#12)")
    p.add_argument("--label", default=None,
                   help="short name for this run in before/after tables")
    p.add_argument("--dump-errors", default=None,
                   help="copy misclassified source frames into this dir, grouped "
                        "by TRUE_to_PRED, for eyeballing")
    p.add_argument("--dump-errors-max", type=int, default=12,
                   help="max frames per confusion pair for --dump-errors")
    run(p.parse_args())


if __name__ == "__main__":
    main()
