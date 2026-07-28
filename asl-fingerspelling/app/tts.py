"""Text-to-speech for the word-speller, on its own thread.

pyttsx3 is offline and cross-platform. Speaking blocks, so we run the engine in a
dedicated worker thread fed by a queue; the inference loop just calls ``say`` and
never stalls waiting for audio.
"""

from __future__ import annotations

import argparse
import queue
import threading
from typing import Optional


class Speaker:
    """Non-blocking TTS. Call ``say(text)``; a worker thread speaks it."""

    def __init__(self, rate: int = 165, volume: float = 1.0, enabled: bool = True) -> None:
        self.enabled = enabled
        self._q: "queue.Queue[Optional[str]]" = queue.Queue()
        self._thread: Optional[threading.Thread] = None
        self._rate = rate
        self._volume = volume
        if enabled:
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()

    def _run(self) -> None:
        import pyttsx3

        engine = pyttsx3.init()
        engine.setProperty("rate", self._rate)
        engine.setProperty("volume", self._volume)
        while True:
            text = self._q.get()
            if text is None:
                break
            try:
                engine.say(text)
                engine.runAndWait()
            except Exception as e:  # never let TTS crash the demo
                print(f"[tts] error: {e}")

    def say(self, text: str) -> None:
        if self.enabled and text.strip():
            self._q.put(text)

    def stop(self) -> None:
        if self._thread is not None:
            self._q.put(None)
            self._thread.join(timeout=2.0)


def main() -> None:
    p = argparse.ArgumentParser(description="Speak a phrase (TTS smoke test).")
    p.add_argument("text", nargs="+")
    p.add_argument("--rate", type=int, default=165)
    args = p.parse_args()
    sp = Speaker(rate=args.rate)
    sp.say(" ".join(args.text))
    sp.stop()


if __name__ == "__main__":
    main()
