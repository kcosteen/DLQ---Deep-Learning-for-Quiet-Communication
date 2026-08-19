#!/usr/bin/env python3
"""Record the shared webcam test set — the project's REAL benchmark (#15).

Opens the webcam, runs every frame through the SHARED crop contract
(``src.crop.HandCropper``), shows the live 224² crop, and writes the captures the
scorer expects::

    data/webcam_testset/<CLASS>/<person>_<condition>_<seq>.jpg

The layout is flat-per-class on purpose: ``src.evaluate._webcam_samples`` lists
files directly under ``<root>/<CLASS>/`` and does not recurse, so provenance goes
in the FILENAME, never in a per-person folder. See
``docs/WEBCAM_TESTSET_PROTOCOL.md``.

Nothing about the framing is re-implemented here. What lands on disk is exactly
``HandCropper.crop(frame).crop_bgr`` — the same 224² BGR array the training cache
(``src.cache_crops``) and the live app (``app.webcam_speller``) produce — and the
frame is mirrored before cropping because ``app/webcam_speller.py`` mirrors before
cropping. Test framing == serving framing == training framing.

QUARANTINE: what this script records NEVER enters training and is NEVER used for
model selection. It is the honest number, and it only stays honest if no one
tunes against it.

Examples::

    # one class, ten shots, tagged with the lighting/background condition
    python scripts/record_webcam_testset.py --person qdoan --class A \\
        --condition desklamp-plainwall --count 10

    # walk all 29 classes in one sitting (press n to advance, q to stop)
    python scripts/record_webcam_testset.py --person qdoan --all-classes \\
        --condition window-daylight-kitchen --count 8

Keys in the capture window:
    SPACE / c   capture the current crop
    a           toggle auto-capture (one every --auto-delay seconds)
    u           undo (delete) the last capture of this session
    n           next class
    q / ESC     quit
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:  # allow `python scripts/record_webcam_testset.py`
    sys.path.insert(0, str(REPO_ROOT))

import cv2  # noqa: E402  (same placement as src/crop.py: module-level cv2)

from src.crop import IMG_SIZE, HandCropper  # noqa: E402
from src.data import CLASSES  # noqa: E402  — the ONLY source of truth for classes
from src.webcam_testset import (  # noqa: E402
    DEFAULT_ROOT,
    NO_HAND_CLASS,
    CaptureNameError,
    capture_filename,
    next_seq,
    resolve_class,
    validate_condition,
    validate_person,
)

QUARANTINE_BANNER = (
    "QUARANTINE: this set is the reported benchmark. It never enters training and\n"
    "is never used for model selection. Do not tune against it."
)

GREEN = (0, 200, 0)
RED = (0, 0, 255)
WHITE = (255, 255, 255)


def annotate(frame, lines: List[tuple], bbox: Optional[tuple]) -> None:
    """Draw the bbox and a stack of ``(text, colour)`` lines onto ``frame``."""
    if bbox is not None:
        x0, y0, x1, y1 = bbox
        cv2.rectangle(frame, (x0, y0), (x1, y1), GREEN, 2)
    for i, (text, colour) in enumerate(lines):
        cv2.putText(frame, text, (10, 28 + 26 * i), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, colour, 2)


def save_capture(crop_bgr, out_dir: Path, person: str, condition: Optional[str],
                 quality: int) -> Path:
    """Write one crop under ``out_dir`` with the next free sequence number."""
    out_dir.mkdir(parents=True, exist_ok=True)
    seq = next_seq(out_dir, person)
    dest = out_dir / capture_filename(person, seq, condition)
    if dest.exists():  # next_seq scans the folder, so this should be impossible
        raise FileExistsError(f"refusing to overwrite {dest}")
    if not cv2.imwrite(str(dest), crop_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality]):
        raise OSError(f"cv2.imwrite failed for {dest}")
    return dest


def record_class(cap, cropper, class_name: str, args) -> tuple:
    """Capture loop for ONE class. Returns ``(n_saved, quit_requested)``.

    ``found_hand`` gates every save: a no-hand frame means MediaPipe found nothing
    and the crop fell back to a centre square, which is a junk sample for every
    class except ``nothing`` (the idle gesture, where an empty frame IS the
    sample — the same convention ``src.cache_crops`` records for the training
    images).
    """
    out_dir = Path(args.out) / class_name
    expect_hand = class_name != NO_HAND_CLASS
    saved: List[Path] = []
    auto = False
    last_auto = 0.0

    print(f"\n=== {class_name} === -> {out_dir}   "
          f"(target {args.count}; SPACE=capture a=auto u=undo n=next q=quit)")
    if not expect_hand:
        print(f"  '{NO_HAND_CLASS}' is the idle gesture: keep your hands OUT of "
              "frame; a no-hand centre crop is the correct sample here.")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("  camera read failed — stopping", file=sys.stderr)
            return len(saved), True
        if args.mirror:
            frame = cv2.flip(frame, 1)  # matches app/webcam_speller.py

        result = cropper.crop(frame)
        # For every class but `nothing` a hand must be found; for `nothing` the
        # correct sample is precisely the frame where none is.
        usable = result.found_hand if expect_hand else not result.found_hand
        done = len(saved) >= args.count

        status = ("hand" if result.found_hand else "NO HAND — will not save")
        if not expect_hand:
            status = "no hand (expected for 'nothing')" if not result.found_hand \
                else "hand in frame — move it out"
        lines = [
            (f"{class_name}   {len(saved)}/{args.count} saved", WHITE),
            (f"person={args.person}  cond={args.condition or '-'}", WHITE),
            (status, GREEN if usable else RED),
            ("AUTO" if auto else "manual  (SPACE=capture, n=next, q=quit)",
             GREEN if auto else WHITE),
        ]
        if done:
            lines.append(("target reached — press n for the next class", GREEN))
        annotate(frame, lines, result.bbox)

        try:
            cv2.imshow("record webcam test set", frame)
            cv2.imshow("crop (what the model sees)", result.crop_bgr)
        except cv2.error as e:  # headless build / no display
            raise SystemExit(
                f"cannot open a preview window ({e}).\nThis script needs a display "
                "— run it on the machine with the webcam, not over a headless shell."
            ) from e

        key = cv2.waitKey(1) & 0xFF
        now = time.time()
        shoot = key in (ord(" "), ord("c"))
        if auto and not done and now - last_auto >= args.auto_delay:
            shoot = True

        if key in (ord("q"), 27):
            return len(saved), True
        if key == ord("n"):
            return len(saved), False
        if key == ord("a"):
            auto = not auto
            last_auto = now
        if key == ord("u"):
            if saved:
                gone = saved.pop()
                gone.unlink(missing_ok=True)
                print(f"  undone: {gone.name}")
        elif shoot:
            if not usable:
                print(f"  skipped: {status}")
                continue
            dest = save_capture(result.crop_bgr, out_dir, args.person,
                                args.condition, args.jpeg_quality)
            saved.append(dest)
            last_auto = now
            print(f"  saved {dest}  ({len(saved)}/{args.count})")
            if len(saved) >= args.count:
                auto = False
                print(f"  target reached for {class_name} — press n for the next "
                      "class, or keep capturing.")


def resolve_labels(args) -> List[str]:
    """The ordered class queue for this session."""
    if args.all_classes:
        return list(CLASSES)
    return [resolve_class(c) for c in args.labels]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Record webcam test-set crops through the shared crop contract "
                    "(#15). This data never trains and is never used for model "
                    "selection.",
        epilog="Protocol: docs/WEBCAM_TESTSET_PROTOCOL.md   "
               "Coverage: python scripts/check_webcam_coverage.py",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--person", required=True,
                   help="your id; becomes the filename prefix (lowercase "
                        "letters/digits/'-', no '_'), e.g. qdoan")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--class", "--label", dest="labels", nargs="+", metavar="CLASS",
                   help="one or more classes from src.data.CLASSES "
                        "(case-insensitive: 'a' == 'A')")
    g.add_argument("--all-classes", action="store_true",
                   help="walk all 29 classes in order in one session")
    p.add_argument("--condition", required=True,
                   help="lighting+background tag recorded in the filename, e.g. "
                        "'desklamp-plainwall'. REQUIRED, and it must change "
                        "between sessions: varied conditions are the entire point "
                        "of this set, and the coverage checker fails a person with "
                        "fewer than two distinct tags.")
    p.add_argument("--out", default=DEFAULT_ROOT,
                   help=f"test-set root (default: {DEFAULT_ROOT})")
    p.add_argument("--camera", type=int, default=0, help="webcam index")
    p.add_argument("--count", type=int, default=10,
                   help="target captures per class (default: 10)")
    p.add_argument("--min-confidence", type=float, default=0.5,
                   help="MediaPipe min hand-detection confidence (default: 0.5, "
                        "the value app/webcam_speller.py uses)")
    p.add_argument("--auto-delay", type=float, default=10.0,
                   help="seconds between shots while auto-capture is on")
    p.add_argument("--jpeg-quality", type=int, default=95,
                   help="JPEG quality for the saved crops (default: 95)")
    p.add_argument("--no-mirror", dest="mirror", action="store_false",
                   help="do NOT mirror the frame before cropping. Default is to "
                        "mirror, because app/webcam_speller.py mirrors before it "
                        "crops — keep them the same or test framing drifts from "
                        "serving framing.")
    return p


def main() -> int:
    args = build_parser().parse_args()
    print(QUARANTINE_BANNER)
    try:
        validate_person(args.person)
        validate_condition(args.condition)
        labels = resolve_labels(args)
    except CaptureNameError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"ERROR: could not open camera {args.camera}", file=sys.stderr)
        return 1

    print(f"person={args.person}  condition={args.condition or '(none)'}  "
          f"out={args.out}  classes={len(labels)}  crop={IMG_SIZE}x{IMG_SIZE}")
    totals = {}
    try:
        with HandCropper(min_detection_confidence=args.min_confidence) as cropper:
            for class_name in labels:
                n, quit_now = record_class(cap, cropper, class_name, args)
                totals[class_name] = n
                if quit_now:
                    break
    finally:
        cap.release()
        cv2.destroyAllWindows()

    written = sum(totals.values())
    print(f"\nsaved {written} crops across {len(totals)} class(es) -> {args.out}")
    short = [c for c, n in totals.items() if n < args.count]
    if short:
        print(f"below target ({args.count}): {', '.join(short)}")
    print("\nCheck coverage:  python scripts/check_webcam_coverage.py "
          f"--root {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
