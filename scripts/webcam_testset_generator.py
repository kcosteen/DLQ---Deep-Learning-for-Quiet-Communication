"""Simple webcam test-set generator: trackbar to select class, space to record.

Opens the webcam, shows the live 224x224 crop the model would see, and saves
crops to ``data/livetest/<CLASS>/`` when recording is active.

Keys:
    SPACE   toggle recording on/off
    q       quit

Trackbar:
    slide to select class (0=A .. 28=nothing)

Usage::

    python scripts/webcam_testset_generator.py --camera 0
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import cv2
import numpy as np

from src.crop import IMG_SIZE, HandCropper
from src.data import CLASSES

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

WINDOW = "webcam test set generator"
CROP_WINDOW = "crop (224x224)"
OUT_DIR = "data/livetest"
NO_HAND_CLASS = "nothing"

GREEN = (0, 200, 0)
RED = (0, 0, 255)
WHITE = (255, 255, 255)
YELLOW = (0, 255, 255)
BLACK = (0, 0, 0)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def count_existing(class_dir: Path) -> int:
    """Count existing .jpg files in a class folder (for sequence numbering)."""
    if not class_dir.exists():
        return 0
    return sum(1 for _ in class_dir.glob("*.jpg"))


def save_crop(crop_bgr: np.ndarray, class_dir: Path, seq: int,
              quality: int = 95) -> Path:
    """Write one 224x224 crop as a JPEG. Returns the path."""
    class_dir.mkdir(parents=True, exist_ok=True)
    dest = class_dir / f"{seq:03d}.jpg"
    cv2.imwrite(str(dest), crop_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return dest


# --------------------------------------------------------------------------- #
# HUD
# --------------------------------------------------------------------------- #

def draw_hud(
    frame: np.ndarray,
    class_name: str,
    recording: bool,
    saved: int,
    found_hand: bool,
) -> None:
    """Draw status text on the frame."""
    h, w = frame.shape[:2]
    rec_text = "REC" if recording else "PAUSED"
    rec_color = RED if recording else WHITE

    lines = [
        (f"class: {class_name}", WHITE),
        (f"status: {rec_text}", rec_color),
        (f"saved: {saved}", WHITE),
        (f"hand: {'detected' if found_hand else 'NOT detected'}",
         GREEN if found_hand else RED),
        ("SPACE=record  q=quit", WHITE),
    ]
    for i, (text, colour) in enumerate(lines):
        cv2.putText(frame, text, (10, 28 + 26 * i),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, colour, 2)


def on_trackbar(val: int, state: dict) -> None:
    """Callback when the trackbar is moved."""
    state["class_idx"] = val
    state["recording"] = False
    state["saved"] = count_existing(Path(OUT_DIR) / CLASSES[val])


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def run(args: argparse.Namespace) -> None:
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"could not open camera {args.camera}")
    if args.width:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    if args.height:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    n_classes = len(CLASSES)
    state = {"class_idx": 0, "recording": False, "saved": 0}

    cv2.namedWindow(WINDOW)
    cv2.createTrackbar("class", WINDOW, 0, n_classes - 1,
                       lambda val: on_trackbar(val, state))

    with HandCropper(min_detection_confidence=0.5) as cropper:
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                frame = cv2.flip(frame, 1)

                result = cropper.crop(frame)
                class_name = CLASSES[state["class_idx"]]
                expect_hand = class_name != NO_HAND_CLASS
                usable = result.found_hand if expect_hand else not result.found_hand

                # Save if recording and usable
                if state["recording"] and usable:
                    class_dir = Path(OUT_DIR) / class_name
                    save_crop(result.crop_bgr, class_dir, state["saved"])
                    state["saved"] += 1

                # Show crop in separate window
                cv2.imshow(CROP_WINDOW, result.crop_bgr)

                # Annotate main frame
                if result.bbox is not None:
                    x0, y0, x1, y1 = result.bbox
                    cv2.rectangle(frame, (x0, y0), (x1, y1), GREEN, 2)

                draw_hud(frame, class_name, state["recording"],
                         state["saved"], result.found_hand)

                cv2.imshow(WINDOW, frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord(" "):
                    state["recording"] = not state["recording"]
                    if state["recording"]:
                        # Re-count in case files were added externally
                        state["saved"] = count_existing(
                            Path(OUT_DIR) / class_name)
                elif key == ord("q"):
                    break
        except KeyboardInterrupt:
            pass

    cap.release()
    cv2.destroyAllWindows()
    total = sum(count_existing(Path(OUT_DIR) / c) for c in CLASSES)
    print(f"\nsaved {total} crops -> {OUT_DIR}/")


def main() -> None:
    p = argparse.ArgumentParser(
        description="Simple webcam test-set generator: trackbar to select "
                    "class, space to record.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--camera", type=int, default=0,
                   help="webcam index (default: 0)")
    p.add_argument("--width", type=int, default=0,
                   help="capture width (0 = camera default)")
    p.add_argument("--height", type=int, default=0,
                   help="capture height (0 = camera default)")
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
