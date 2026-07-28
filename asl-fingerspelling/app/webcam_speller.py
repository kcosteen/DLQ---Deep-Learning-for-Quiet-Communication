"""Live webcam word-speller.

Pipeline (all sharing the ``src/crop`` contract so serving == training):

    OpenCV capture
      -> MediaPipe Hands detect & crop to hand bbox   (src/crop.HandCropper)
      -> identical resize + normalize                 (src/crop.preprocess_bgr)
      -> EfficientNet predicts letter + confidence
      -> speller: commit a letter after ~0.5 s of stable, confident prediction;
         'space'/'delete' edit the word; 'nothing' = idle
      -> on word completion, pyttsx3 speaks it        (app/tts.Speaker)

The HUD shows the hand crop, the top-3 letters + confidence, and the word being
built. Capture / inference / TTS are decoupled: TTS runs on its own thread and the
debouncer smooths the per-frame predictions so a single noisy frame never commits.

Usage:
    python -m app.webcam_speller --checkpoint checkpoints/efficientnet_b0.pt
"""

from __future__ import annotations

import argparse
import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Optional, Tuple

import cv2
import numpy as np

from app.tts import Speaker
from src.crop import HandCropper, preprocess_bgr
from src.data import CLASSES


# --------------------------------------------------------------------------- #
# Debounced speller state machine
# --------------------------------------------------------------------------- #
@dataclass
class SpellerConfig:
    confidence_gate: float = 0.6     # ignore predictions below this confidence
    stable_seconds: float = 0.5      # hold a letter this long before committing
    cooldown_seconds: float = 0.8    # min gap between commits of the same letter


class WordSpeller:
    """Turns a stream of (letter, confidence) into an edited, spoken word."""

    def __init__(self, cfg: SpellerConfig, speaker: Optional[Speaker] = None) -> None:
        self.cfg = cfg
        self.speaker = speaker
        self.word: str = ""
        self.spoken: List[str] = []          # completed words this session
        self._candidate: Optional[str] = None
        self._candidate_since: float = 0.0
        self._last_commit_time: float = 0.0
        self._last_committed: Optional[str] = None

    def update(self, letter: str, confidence: float, now: float) -> Optional[str]:
        """Feed one frame's prediction. Returns a committed token, or None."""
        if confidence < self.cfg.confidence_gate or letter == "nothing":
            self._candidate = None
            return None

        if letter != self._candidate:
            self._candidate = letter
            self._candidate_since = now
            return None

        held = now - self._candidate_since
        cooling = (now - self._last_commit_time) < self.cfg.cooldown_seconds
        same_as_last = letter == self._last_committed
        if held >= self.cfg.stable_seconds and not (cooling and same_as_last):
            self._last_commit_time = now
            self._last_committed = letter
            self._candidate_since = now  # require re-hold for a repeat
            return self._commit(letter)
        return None

    def _commit(self, token: str) -> str:
        if token == "space":
            if self.word:
                self.spoken.append(self.word)
                if self.speaker:
                    self.speaker.say(self.word)
                self.word = ""
        elif token == "delete":
            self.word = self.word[:-1]
        else:  # a letter A-Z
            self.word += token
        return token


# --------------------------------------------------------------------------- #
# Inference
# --------------------------------------------------------------------------- #
class Classifier:
    """Loads a checkpoint and returns top-k (letter, prob) for a hand crop."""

    def __init__(self, checkpoint: str, device: str = "cpu") -> None:
        import torch

        from src.evaluate import load_checkpoint

        self.torch = torch
        self.device = device
        self.model = load_checkpoint(checkpoint, device)

    def topk(self, crop_bgr: np.ndarray, k: int = 3) -> List[Tuple[str, float]]:
        x = self.torch.from_numpy(preprocess_bgr(crop_bgr)).unsqueeze(0).to(self.device)
        with self.torch.no_grad():
            probs = self.torch.softmax(self.model(x), dim=1)[0]
        vals, idxs = probs.topk(k)
        return [(CLASSES[i], float(v)) for v, i in zip(vals.tolist(), idxs.tolist())]


# --------------------------------------------------------------------------- #
# HUD
# --------------------------------------------------------------------------- #
def draw_hud(frame, crop, top3, word, fps, found_hand) -> np.ndarray:
    h, w = frame.shape[:2]
    thumb = cv2.resize(crop, (140, 140))
    frame[10:150, w - 150 : w - 10] = thumb
    cv2.rectangle(frame, (w - 150, 10), (w - 10, 150), (0, 255, 0), 1)

    y = 30
    cv2.putText(frame, f"FPS {fps:4.1f}", (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (0, 255, 0), 2)
    if not found_hand:
        cv2.putText(frame, "no hand", (10, y + 24), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 165, 255), 2)
    for i, (letter, prob) in enumerate(top3):
        color = (0, 255, 0) if i == 0 else (200, 200, 200)
        cv2.putText(frame, f"{letter:>7}  {prob:5.2f}", (10, y + 24 * (i + 2)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.rectangle(frame, (0, h - 50), (w, h), (0, 0, 0), -1)
    cv2.putText(frame, f"word: {word}_", (10, h - 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    return frame


def run(args: argparse.Namespace) -> None:
    speaker = Speaker(enabled=not args.no_tts)
    speller = WordSpeller(
        SpellerConfig(confidence_gate=args.confidence,
                      stable_seconds=args.stable_seconds),
        speaker,
    )
    clf = Classifier(args.checkpoint, device=args.device)
    cap = cv2.VideoCapture(args.camera)

    fps = 0.0
    last = time.time()
    with HandCropper(min_detection_confidence=0.5) as cropper:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)  # mirror for natural interaction
            res = cropper.crop(frame)
            top3 = clf.topk(res.crop_bgr, k=3) if res.found_hand else [("nothing", 1.0)]

            now = time.time()
            letter, conf = top3[0]
            committed = speller.update(letter, conf, now)
            if committed:
                print(f"committed: {committed!r}   word={speller.word!r}")

            if res.bbox is not None:
                x0, y0, x1, y1 = res.bbox
                cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 255, 0), 2)
            dt = now - last
            fps = 0.9 * fps + 0.1 * (1.0 / dt if dt > 0 else 0.0)
            last = now
            frame = draw_hud(frame, res.crop_bgr, top3, speller.word, fps,
                             res.found_hand)

            cv2.imshow("ASL fingerspelling speller  (q=quit, c=clear)", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("c"):
                speller.word = ""

    cap.release()
    cv2.destroyAllWindows()
    speaker.stop()
    if speller.spoken:
        print("spoken words this session:", speller.spoken)


def main() -> None:
    p = argparse.ArgumentParser(description="Live ASL fingerspelling word-speller.")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--device", default="cpu", help="cpu or cuda")
    p.add_argument("--confidence", type=float, default=0.6,
                   help="min confidence to consider a prediction")
    p.add_argument("--stable-seconds", type=float, default=0.5,
                   help="hold time before a letter commits")
    p.add_argument("--no-tts", action="store_true", help="disable speech output")
    run(p.parse_args())


if __name__ == "__main__":
    main()
