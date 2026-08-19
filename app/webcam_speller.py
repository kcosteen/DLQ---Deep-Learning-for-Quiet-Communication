"""Live webcam bridge (#17): MediaPipe as a *localizer*, not a representation.

The checkpoint ``efficientnet_b0_targeted.pt`` was trained on the original 200x200
training frames. Those frames pass through ``resize_square -> normalize_chw ->
EfficientNet``; the hand fills ~45% of the image (median ``max(bbox)/img_side``,
measured by ``scripts/measure_training_framing.py``). For the live demo to work,
the webcam crop that arrives at ``preprocess_bgr`` has to reproduce that framing —
the same hand/image size ratio, centring, and context — rather than a tight hand
crop where the hand fills nearly the whole image.

Pipeline (all sharing the frozen ``src.crop`` contract):

    OpenCV capture
      -> MediaPipe HandLandmarker        (localizer ONLY: tight landmark bbox)
      -> square bbox + configurable margin   (default = training-derived 0.62)
      -> bounds-safe crop -> resize        (src.crop.crop_square_with_margin)
      -> identical resize + normalize     (src.crop.preprocess_bgr)
      -> EfficientNet-B0 targeted.pt      (Classifier)
      -> confidence gate -> HUD           (Prediction / Uncertain / No hand)

Guardrails (deliberate):

* No-hand frames are NEVER classified. The MediaPipe centre fallback that
  ``HandCropper.crop`` uses for training is not used in live mode: no hand means
  the "No hand" HUD state and an idle speller tick, nothing else.
* The crop margin is configurable (``--crop-margin``) and defaults to the
  training-derived value rather than the fixed ``BBOX_MARGIN=0.25`` contract
  constant, which produces a far tighter framing than the training images.
* ``--compare-full-frame`` runs both the full-frame path and the localized-crop
  path (A/B, diagnostic only).
* Per-stage latency is tracked (localization / crop+preprocess / inference) and
  reported on exit — measured, not optimised-to-target.

Usage:
    python -m app.webcam_speller --checkpoint checkpoints/efficientnet_b0_targeted.pt
"""

from __future__ import annotations

import argparse
import signal
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from app.tts import Speaker
from src.crop import (
    AsymmetricMargins,
    DEFAULT_ASYMMETRIC_MARGINS,
    HandCropper,
    asymmetric_context_bbox,
    crop_square_with_forearm_context,
    crop_square_with_margin,
    preprocess_bgr,
    square_bbox_with_margin,
)
from src.data import CLASSES

# --------------------------------------------------------------------------- #
# Training-derived webcam crop margin
# --------------------------------------------------------------------------- #
# Measured by scripts/measure_training_framing.py on 580 sampled training frames
# (426 with a MediaPipe-detected hand): the hand's longest side spans a median
# of ~44.5% of the 200px training image, so a square webcam crop reproduces that
# framing with margin m = (1/0.445 - 1)/2 = 0.62. This is intentionally NOT the
# frozen BBOX_MARGIN=0.25 contract constant: that tightens the hand to ~2/3 of
# the crop, which is framing the training data never had.
TRAINING_DERIVED_CROP_MARGIN: float = 0.62

# Forearm-crop experiment defaults (see src.crop.AsymmetricMargins). These are
# per-side fractions of the tight hand bbox; the big bottom margin preserves the
# wrist/forearm segment the original 200x200 dataset frames show. Experimental,
# NOT frozen.
DEFAULT_CROP_LEFT: float = 0.7
DEFAULT_CROP_RIGHT: float = 0.7
DEFAULT_CROP_TOP: float = 0.4
DEFAULT_CROP_BOTTOM: float = 1.8

# Default confidence gate, configurable via --confidence-threshold. Lowered from
# the #17 spec's 0.70 to 0.40 after the 2026-08-14 live measurement, then to 0.30
# (2026-08-15): crop-path top-1 confidence was median ~0.24 / p90 ~0.47, so 0.70
# suppressed essentially every frame and nothing could ever commit.
DEFAULT_CONFIDENCE_THRESHOLD: float = 0.30

# Rolling FPS window and per-stage timing lists.
STAGE_NAMES = ("localization", "crop_preprocess", "inference")

# OpenCV window title; the DELETE button is clickable inside this window.
WINDOW_NAME = "ASL webcam bridge  (q=quit, c=clear, click DELETE)"


# --------------------------------------------------------------------------- #
# Debounced speller state machine
# --------------------------------------------------------------------------- #
@dataclass
class SpellerConfig:
    confidence_gate: float = DEFAULT_CONFIDENCE_THRESHOLD  # ignore below this
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

    def delete_last(self) -> Optional[str]:
        """Backspace one letter from the word. Used by the HUD DELETE button;
        no hold/cooldown logic (it is an explicit click, not a model token)."""
        if not self.word:
            return None
        removed = self.word[-1]
        self.word = self.word[:-1]
        return removed


# --------------------------------------------------------------------------- #
# Inference
# --------------------------------------------------------------------------- #
class Classifier:
    """Loads the targeted checkpoint; stages are separable so latency is measured
    honestly (normalize vs forward are different pipeline steps)."""

    def __init__(self, checkpoint: str, device: str = "cpu") -> None:
        import torch

        from src.evaluate import load_checkpoint

        self.torch = torch
        self.device = device
        self.model = load_checkpoint(checkpoint, device)

    def tensor_from_bgr(self, img_bgr: np.ndarray) -> "torch.Tensor":
        """The exact training tensor for any BGR image (crop OR full frame)."""
        x = self.torch.from_numpy(preprocess_bgr(img_bgr)).unsqueeze(0).to(self.device)
        return x

    def probs(self, x: "torch.Tensor") -> "torch.Tensor":
        """Softmax logits over the 29 classes for a (1,3,224,224) tensor."""
        with self.torch.no_grad():
            return self.torch.softmax(self.model(x), dim=1)[0]

    def topk(self, img_bgr: np.ndarray, k: int = 3) -> List[Tuple[str, float]]:
        """Preprocess + forward + top-k, for a single BGR image."""
        probs = self.probs(self.tensor_from_bgr(img_bgr))
        vals, idxs = probs.topk(k)
        return [(CLASSES[i], float(v)) for v, i in zip(vals.tolist(), idxs.tolist())]


# --------------------------------------------------------------------------- #
# Per-frame decision logic (pure, testable)
# --------------------------------------------------------------------------- #
@dataclass
class FrameResult:
    """What a frame resolves to after localization + confidence gate."""

    status: str                      # "ok" | "uncertain" | "no_hand"
    found_hand: bool
    letter: Optional[str] = None     # top-1 letter when confident
    confidence: float = 0.0          # top-1 confidence
    top3: List[Tuple[str, float]] = field(default_factory=list)


def decide_prediction(
    letter: Optional[str],
    confidence: float,
    found_hand: bool,
    threshold: float,
    top3: Optional[List[Tuple[str, float]]] = None,
) -> FrameResult:
    """Apply the two live-mode gates to one frame's raw classification.

    * ``found_hand=False``  -> ``no_hand`` (the crop path never runs, and a
      centre fallback is never classified — this is the #17 no-hand guard).
    * ``confidence < threshold`` -> ``uncertain`` (letter is withheld).
    * otherwise -> ``ok`` with the letter.
    """
    if not found_hand:
        return FrameResult("no_hand", found_hand=False, letter=None,
                           confidence=0.0, top3=[])
    if letter is None or confidence < threshold:
        return FrameResult("uncertain", found_hand=True, letter=None,
                           confidence=float(confidence), top3=list(top3 or []))
    return FrameResult("ok", found_hand=True, letter=letter,
                       confidence=float(confidence), top3=list(top3 or []))


def pipeline_mode(compare_full_frame: bool) -> str:
    """Routing: 'crop' (default) or 'both' (A/B when --compare-full-frame).

    Kept as a tiny function so the routing choice is testable and the CLI flag
    cannot silently mean different things in different places.
    """
    if compare_full_frame:
        return "both"
    return "crop"


@dataclass
class FrameOutcome:
    """Everything one frame resolves to, before any HUD/speller/save logic."""

    result: FrameResult
    crop: Optional[np.ndarray] = None
    margin_bbox: Optional[Tuple[int, int, int, int]] = None
    ab_line: Optional[str] = None
    timing: Dict[str, float] = field(default_factory=dict)  # stage -> dt (s)


def process_frame(
    frame: np.ndarray,
    cropper: HandCropper,
    clf: "Classifier",
    crop_margin: float,
    threshold: float,
    mode: str,
    crop_mode: str = "symmetric",
    forearm_margins: AsymmetricMargins = DEFAULT_ASYMMETRIC_MARGINS,
    compare_crops: bool = False,
) -> Tuple[FrameOutcome, Optional[Tuple[int, int, int, int]]]:
    """One frame through localization -> crop -> classify -> gates.

    This is the full live pipeline minus hardware/HUD: it is what the webcam
    loop runs per frame and what the no-hand / A/B tests exercise. The centre
    fallback of ``HandCropper.crop`` is deliberately NOT used here: when
    ``tight_bbox`` is None the frame is never fed to the model.

    ``crop_mode`` selects the framing fed to the net: ``symmetric`` (hand-centred
    margin, the #17 default) or ``forearm`` (asymmetric bottom-heavy crop that
    preserves the wrist/forearm segment, experimental). ``compare_crops`` runs
    BOTH crops and reports a ``SYMMETRIC: ... FOREARM: ...`` line per detected
    hand; ``mode="both"`` additionally runs the full frame (``FULL: ... CROP``).
    The frame's primary decision always comes from the active ``crop_mode``.
    """
    t_frame = time.perf_counter()
    tight = cropper.tight_bbox(frame)
    t1 = time.perf_counter()
    timing = {"localization": t1 - t_frame}

    if tight is None:
        timing.update({"crop_preprocess": 0.0, "inference": 0.0,
                       "total": time.perf_counter() - t_frame})
        return FrameOutcome(FrameResult("no_hand", found_hand=False),
                            timing=timing), None

    h, w = frame.shape[:2]
    if crop_mode == "forearm":
        crop = crop_square_with_forearm_context(frame, tight, forearm_margins)
        margin_bbox = asymmetric_context_bbox(frame, tight, forearm_margins)
    else:
        crop = crop_square_with_margin(frame, tight, crop_margin)
        margin_bbox = square_bbox_with_margin(*tight, w, h, crop_margin)
    x = clf.tensor_from_bgr(crop)
    t2 = time.perf_counter()
    timing["crop_preprocess"] = t2 - t1
    probs = clf.probs(x)
    t3 = time.perf_counter()
    timing["inference"] = t3 - t2
    vals, idxs = probs.topk(3)
    top3 = [(CLASSES[i], float(v)) for v, i in zip(vals.tolist(), idxs.tolist())]
    letter, conf = top3[0]
    result = decide_prediction(letter, conf, True, threshold, top3)
    ab_line: Optional[str] = None
    ab_lines: List[str] = []

    if mode == "both":
        # A/B diagnostic: also run the full frame through the same net.
        ta = time.perf_counter()
        probs_full = clf.probs(clf.tensor_from_bgr(frame))
        timing["inference_ab_full"] = time.perf_counter() - ta
        vf, if_ = probs_full.topk(1)
        fl, fc = CLASSES[int(if_[0])], float(vf[0])
        ab_lines.append(f"FULL: {fl} {fc:.2f}   CROP: {letter} {conf:.2f}")

    if compare_crops:
        # A/B diagnostic: run BOTH the symmetric and the forearm crop.
        if crop_mode == "forearm":
            sym_crop = crop_square_with_margin(frame, tight, crop_margin)
            fore_crop = crop
        else:
            sym_crop = crop
            fore_crop = crop_square_with_forearm_context(frame, tight, forearm_margins)
        ta = time.perf_counter()
        probs_sym = clf.probs(clf.tensor_from_bgr(sym_crop))
        probs_fore = clf.probs(clf.tensor_from_bgr(fore_crop))
        timing["inference_ab_crop"] = time.perf_counter() - ta
        vs, is_ = probs_sym.topk(1)
        vf, if_ = probs_fore.topk(1)
        sl, sc = CLASSES[int(is_[0])], float(vs[0])
        fl, fc = CLASSES[int(if_[0])], float(vf[0])
        ab_lines.append(f"SYMMETRIC: {sl} {sc:.2f}   FOREARM: {fl} {fc:.2f}")

    if ab_lines:
        ab_line = "   ".join(ab_lines)

    timing["total"] = time.perf_counter() - t_frame
    return FrameOutcome(result, crop=crop, margin_bbox=margin_bbox,
                        ab_line=ab_line, timing=timing), tight


# --------------------------------------------------------------------------- #
# HUD
# --------------------------------------------------------------------------- #
def _status_lines(result: FrameResult) -> Tuple[str, str]:
    if result.status == "no_hand":
        return "No hand", ""
    if result.status == "uncertain":
        return "Uncertain", f"Confidence: {result.confidence:.2f}"
    return f"Prediction: {result.letter}", f"Confidence: {result.confidence:.2f}"


def draw_hud(
    frame: np.ndarray,
    result: FrameResult,
    crop: Optional[np.ndarray],
    word: str,
    fps: float,
    ab_line: Optional[str] = None,
) -> np.ndarray:
    """Overlay the #17 HUD: status (Prediction/Uncertain/No hand), confidence,
    word bar, FPS, optional A/B line, and the crop thumbnail."""
    h, w = frame.shape[:2]
    if crop is not None:
        thumb = cv2.resize(crop, (140, 140))
        frame[10:150, w - 150 : w - 10] = thumb
        cv2.rectangle(frame, (w - 150, 10), (w - 10, 150), (0, 255, 0), 1)

    line1, line2 = _status_lines(result)
    color = {"ok": (0, 255, 0), "uncertain": (0, 165, 255), "no_hand": (0, 0, 255)}
    y = 30
    cv2.putText(frame, f"FPS {fps:4.1f}", (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (0, 255, 0), 2)
    cv2.putText(frame, line1, (10, y + 24), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, color[result.status], 2)
    if line2:
        cv2.putText(frame, line2, (10, y + 48), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (200, 200, 200), 2)
    if ab_line is not None:
        cv2.putText(frame, ab_line, (10, y + 72), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 0), 2)

    cv2.rectangle(frame, (0, h - 50), (w, h), (0, 0, 0), -1)
    cv2.putText(frame, f"word: {word}_", (10, h - 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    return frame


def draw_debug(
    frame: np.ndarray,
    tight_bbox: Optional[Tuple[int, int, int, int]],
    margin_bbox: Optional[Tuple[int, int, int, int]],
    crop: Optional[np.ndarray],
    info: str,
) -> None:
    """Debug overlay: tight hand box (blue) + crop box (green).

    ``info`` is a short framing label (e.g. ``symmetric margin 0.62`` or the
    forearm per-side margins) drawn under the boxes.
    """
    if tight_bbox is not None:
        x0, y0, x1, y1 = tight_bbox
        cv2.rectangle(frame, (x0, y0), (x1, y1), (255, 0, 0), 1)
    if margin_bbox is not None:
        x0, y0, x1, y1 = margin_bbox
        cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 255, 0), 2)
    cv2.putText(frame, info, (10, 150),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    if crop is not None:
        cv2.imshow("crop (224, margin-applied)", crop)


# --------------------------------------------------------------------------- #
# HUD DELETE button (click to backspace a wrong letter)
# --------------------------------------------------------------------------- #
@dataclass
class DeleteButton:
    """Shared click-target for the HUD DELETE button. OpenCV runs the mouse
    callback on its own thread, so the run loop polls ``clicked`` per frame."""

    rect: Optional[Tuple[int, int, int, int]] = None
    frame_h: Optional[int] = None
    clicked: bool = False


def draw_delete_button(frame: np.ndarray) -> Tuple[int, int, int, int]:
    """Draw a full-width DELETE strip above the word bar; return its rect."""
    h, w = frame.shape[:2]
    x0, y0, x1, y1 = 0, h - 120, w, h - 56
    cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 0, 40), -1)
    cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 0, 255), 2)
    cv2.putText(frame, "DELETE", (w // 2 - 95, y0 + 44),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    return x0, y0, x1, y1


HIT_TOLERANCE = 30  # px; absorbs the classic macOS title-bar y-offset


def click_in_delete(
    rect: Optional[Tuple[int, int, int, int]],
    frame_h: Optional[int],
    x: int,
    y: int,
) -> bool:
    """Tolerant hit test for the DELETE strip.

    The strip is full-width, so only the y coordinate matters; ``HIT_TOLERANCE``
    absorbs title-bar / scale offsets that the Cocoa backend may introduce.
    ``frame_h`` is kept for diagnostics and for a possible y-origin fix if the
    printout in ``on_mouse`` ever shows inverted coordinates.
    """
    if rect is None:
        return False
    x0, y0, x1, y1 = rect
    return (x0 - HIT_TOLERANCE <= x <= x1 + HIT_TOLERANCE
            and y0 - HIT_TOLERANCE <= y <= y1 + HIT_TOLERANCE)


def on_mouse(event: int, x: int, y: int, flags: int, userdata: DeleteButton) -> None:
    """OpenCV mouse callback: a left-click on the DELETE strip marks it.

    Prints every click (with reported coordinates and the strip rect) so a
    still-broken button can be diagnosed in one run instead of guessed at.
    """
    btn = userdata
    if event in (cv2.EVENT_LBUTTONDOWN, cv2.EVENT_LBUTTONDBLCLK):
        print(f"mouse {event} at ({x},{y})  rect={btn.rect}  h={btn.frame_h}")
        if click_in_delete(btn.rect, btn.frame_h, x, y):
            btn.clicked = True


def handle_delete_key(key: int, speller: "WordSpeller") -> Optional[str]:
    """Backspace/delete via keyboard (Backspace or 'b'), same as the button."""
    if key in (8, ord("b")):
        return speller.delete_last()
    return None


# --------------------------------------------------------------------------- #
# Timing
# --------------------------------------------------------------------------- #
@dataclass
class Timings:
    """Per-stage rolling stats, reported on exit. Honest measured numbers."""

    stages: Dict[str, List[float]] = field(
        default_factory=lambda: {s: [] for s in STAGE_NAMES + ("total",)}
    )
    n_frames: int = 0
    elapsed: float = 0.0

    def add(self, stage: str, dt: float) -> None:
        self.stages.setdefault(stage, []).append(dt)

    def mean_ms(self, stage: str) -> float:
        vals = self.stages.get(stage, [])
        return float(sum(vals) / len(vals)) * 1000.0 if vals else 0.0

    def report(self, ab_full: bool, ab_crops: bool = False) -> None:
        print("\nper-stage latency (per frame, mean ms):")
        total = self.mean_ms("total")
        for s in STAGE_NAMES:
            print(f"  {s:>20}: {self.mean_ms(s):6.1f}")
        if ab_full:
            print(f"  {'full-frame fwd (A/B)':>20}: {self.mean_ms('inference_ab_full'):6.1f}")
        if ab_crops:
            print(f"  {'symmetric-vs-forearm fwd':>20}: {self.mean_ms('inference_ab_crop'):6.1f}")
        print(f"  {'total pipeline':>20}: {total:6.1f}")
        if self.n_frames and total > 0:
            print(f"  {'rolling avg FPS':>20}: {1000.0 / total:6.1f}")
            print(f"  {'wall-clock FPS':>20}: {self.n_frames / self.elapsed:6.1f}")


# --------------------------------------------------------------------------- #
# Main loop
# --------------------------------------------------------------------------- #
def _terminate_handler(signum, frame):
    raise KeyboardInterrupt


def run(args: argparse.Namespace) -> None:
    signal.signal(signal.SIGTERM, _terminate_handler)
    speaker = Speaker(enabled=not args.no_tts)
    speller = WordSpeller(
        SpellerConfig(confidence_gate=args.confidence_threshold,
                      stable_seconds=args.stable_seconds),
        speaker,
    )
    clf = Classifier(args.checkpoint, device=args.device)
    cap = cv2.VideoCapture(args.camera)
    if args.width:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    if args.height:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    mode = pipeline_mode(args.compare_full_frame)
    forearm_margins = AsymmetricMargins(args.crop_left, args.crop_right,
                                        args.crop_top, args.crop_bottom)
    print(f"mode: {mode}   crop-mode: {args.crop_mode}   "
          f"crop-margin: {args.crop_margin}   "
          f"confidence-threshold: {args.confidence_threshold}")
    if args.crop_mode == "forearm":
        print(f"forearm margins L={args.crop_left} R={args.crop_right} "
              f"T={args.crop_top} B={args.crop_bottom} "
              f"(--crop-margin ignored in forearm mode)")
    save_dir = args.save_crops
    save_max = args.save_crops_max
    if save_dir:
        from pathlib import Path
        Path(save_dir).mkdir(parents=True, exist_ok=True)

    delete_btn = DeleteButton()
    cv2.namedWindow(WINDOW_NAME)
    cv2.setMouseCallback(WINDOW_NAME, on_mouse, delete_btn)

    t_start = time.time()
    timings = Timings()
    fps = 0.0
    last = time.time()
    saved = 0

    with HandCropper(min_detection_confidence=0.5) as cropper:
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                frame = cv2.flip(frame, 1)  # mirror for natural interaction

                out, tight = process_frame(frame, cropper, clf, args.crop_margin,
                                           args.confidence_threshold, mode,
                                           crop_mode=args.crop_mode,
                                           forearm_margins=forearm_margins,
                                           compare_crops=args.compare_crops)
                for stage, dt in out.timing.items():
                    timings.add(stage, dt)
                timings.n_frames += 1
                result, crop, margin_bbox, ab_line = (
                    out.result, out.crop, out.margin_bbox, out.ab_line)
                if ab_line:
                    print(ab_line)

                if result.status == "ok":
                    letter, conf = result.letter, result.confidence
                    committed = speller.update(letter, conf, time.time())
                    if committed:
                        print(f"committed: {committed!r}   word={speller.word!r}")
                else:
                    # no_hand / uncertain: idle tick resets the speller candidate so
                    # a stale letter cannot silently commit later.
                    speller.update("nothing", 1.0, time.time())

                # save sample webcam crops (crop + annotated frame)
                if save_dir and result.found_hand and saved < save_max:
                    crop_annot = frame.copy()
                    if margin_bbox is not None:
                        x0, y0, x1, y1 = margin_bbox
                        cv2.rectangle(crop_annot, (x0, y0), (x1, y1), (0, 255, 0), 2)
                    cv2.imwrite(f"{save_dir}/crop_{saved:04d}.jpg", crop)
                    cv2.imwrite(f"{save_dir}/frame_{saved:04d}.jpg", crop_annot)
                    saved += 1

                dt = time.time() - last
                timings.add("total", dt)
                fps = 0.9 * fps + 0.1 * (1.0 / dt if dt > 0 else 0.0)
                last = time.time()

                if args.debug:
                    if args.crop_mode == "forearm":
                        info = (f"forearm L{args.crop_left} R{args.crop_right} "
                                f"T{args.crop_top} B{args.crop_bottom}")
                    else:
                        info = f"symmetric margin {args.crop_margin:.2f}"
                    draw_debug(frame, tight, margin_bbox, crop, info)
                frame = draw_hud(frame, result, crop, speller.word, fps, ab_line)
                delete_btn.rect = draw_delete_button(frame)
                delete_btn.frame_h = frame.shape[0]

                cv2.imshow(WINDOW_NAME, frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("c"):
                    speller.word = ""
                button_hit = delete_btn.clicked
                delete_btn.clicked = False
                removed = None
                if button_hit:
                    removed = speller.delete_last()
                    if removed:
                        print(f"deleted {removed!r}   word={speller.word!r}")
                    else:
                        print("delete clicked, word empty")
                else:
                    removed = handle_delete_key(key, speller)
                    if removed:
                        print(f"deleted {removed!r}   word={speller.word!r}")
        except KeyboardInterrupt:
            print("interrupted; finishing session...")

    cap.release()
    cv2.destroyAllWindows()
    speaker.stop()
    if save_dir:
        print(f"saved {saved} webcam crops -> {save_dir}")
    timings.elapsed = time.time() - t_start
    timings.report(ab_full=(mode == "both"), ab_crops=args.compare_crops)
    if speller.spoken:
        print("spoken words this session:", speller.spoken)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Live ASL webcam bridge: MediaPipe localizes the hand, "
                    "the square+margin crop reproduces the training framing, "
                    "the targeted EfficientNet checkpoint classifies.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--device", default="cpu",
                   help="cpu, cuda, or mps (mps recommended on Apple Silicon; "
                        "measured ~16 FPS vs ~5 on CPU)")
    p.add_argument("--confidence-threshold", "--confidence", dest="confidence_threshold",
                   type=float, default=DEFAULT_CONFIDENCE_THRESHOLD,
                   help=f"min top-1 confidence to show a prediction "
                        f"(default: {DEFAULT_CONFIDENCE_THRESHOLD})")
    p.add_argument("--crop-margin", type=float, default=TRAINING_DERIVED_CROP_MARGIN,
                   help="fraction of the tight hand bbox side added around the "
                        "hand before cropping (default: training-derived "
                        f"{TRAINING_DERIVED_CROP_MARGIN}, see "
                        "scripts/measure_training_framing.py)")
    p.add_argument("--crop-mode", choices=["symmetric", "forearm"],
                   default="symmetric",
                   help="symmetric: hand-centred square margin (default). "
                        "forearm: asymmetric bottom-heavy crop that preserves the "
                        "wrist/forearm segment the training frames show "
                        "(experimental, see --crop-left/right/top/bottom)")
    p.add_argument("--crop-left", type=float, default=DEFAULT_CROP_LEFT,
                   help=f"forearm-mode context left of the hand, as a fraction "
                        f"of hand width (default: {DEFAULT_CROP_LEFT})")
    p.add_argument("--crop-right", type=float, default=DEFAULT_CROP_RIGHT,
                   help=f"forearm-mode context right of the hand, as a fraction "
                        f"of hand width (default: {DEFAULT_CROP_RIGHT})")
    p.add_argument("--crop-top", type=float, default=DEFAULT_CROP_TOP,
                   help=f"forearm-mode context above the hand, as a fraction of "
                        f"hand height (default: {DEFAULT_CROP_TOP})")
    p.add_argument("--crop-bottom", type=float, default=DEFAULT_CROP_BOTTOM,
                   help=f"forearm-mode context below the hand, as a fraction of "
                        f"hand height (default: {DEFAULT_CROP_BOTTOM})")
    p.add_argument("--width", type=int, default=0,
                   help="requested capture width (0 = camera default)")
    p.add_argument("--height", type=int, default=0,
                   help="requested capture height (0 = camera default)")
    p.add_argument("--debug", action="store_true",
                   help="draw tight + margin boxes and show the 224 crop window")
    p.add_argument("--compare-full-frame", action="store_true",
                   help="A/B diagnostic: run the full frame AND the localized "
                        "crop through the same net; log FULL/CROP lines")
    p.add_argument("--compare-crops", action="store_true",
                   help="A/B diagnostic: run the symmetric AND the forearm crop "
                        "through the same net per detected hand; log "
                        "SYMMETRIC/FOREARM lines")
    p.add_argument("--save-crops", default=None,
                   help="save sample webcam crops + annotated frames here")
    p.add_argument("--save-crops-max", type=int, default=50,
                   help="max webcam crops to save (default: 50)")
    p.add_argument("--stable-seconds", type=float, default=0.5,
                   help="hold time before a letter commits")
    p.add_argument("--no-tts", action="store_true", help="disable speech output")
    return p


def main() -> None:
    try:
        run(build_parser().parse_args())
    except KeyboardInterrupt:
        print("interrupted before the capture loop; exiting cleanly.")


if __name__ == "__main__":
    main()
