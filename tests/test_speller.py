"""Unit tests for the debounced word-speller state machine.

Pure logic — no model, no webcam, no TTS engine (Speaker=None). Verifies the
stability hold, the confidence gate, and space/delete editing.
"""

from __future__ import annotations

import pytest

pytest.importorskip("cv2")  # app.webcam_speller imports src.crop -> cv2

from app.webcam_speller import SpellerConfig, WordSpeller  # noqa: E402


def _commit(speller, letter, t):
    """Feed a letter twice: once to set the candidate, once past the hold time."""
    speller.update(letter, 0.99, t)
    return speller.update(letter, 0.99, t + 1.0)  # 1s > 0.5s stable window


def test_low_confidence_is_ignored():
    s = WordSpeller(SpellerConfig(confidence_gate=0.6))
    assert s.update("A", 0.10, 0.0) is None
    assert s.update("A", 0.10, 1.0) is None
    assert s.word == ""


def test_letter_commits_only_after_stable_hold():
    s = WordSpeller(SpellerConfig(stable_seconds=0.5))
    assert s.update("A", 0.99, 0.0) is None       # just became candidate
    assert s.update("A", 0.99, 0.2) is None        # held 0.2s < 0.5s
    assert s.update("A", 0.99, 0.6) == "A"          # held 0.6s -> commit
    assert s.word == "A"


def test_flicker_does_not_commit():
    s = WordSpeller(SpellerConfig(stable_seconds=0.5))
    s.update("A", 0.99, 0.0)
    s.update("B", 0.99, 0.1)   # candidate resets
    s.update("A", 0.99, 0.2)   # resets again
    assert s.word == ""

def test_space_speaks_and_resets_word():
    s = WordSpeller(SpellerConfig())
    for i, ch in enumerate("HI"):
        _commit(s, ch, i * 3.0)
    assert s.word == "HI"
    _commit(s, "space", 100.0)
    assert s.word == ""
    assert s.spoken == ["HI"]


def test_delete_removes_last_letter():
    s = WordSpeller(SpellerConfig())
    _commit(s, "C", 0.0)
    _commit(s, "A", 3.0)
    _commit(s, "T", 6.0)
    assert s.word == "CAT"
    _commit(s, "delete", 9.0)
    assert s.word == "CA"


def test_nothing_is_idle():
    s = WordSpeller(SpellerConfig())
    assert s.update("nothing", 0.99, 0.0) is None
    assert s.update("nothing", 0.99, 1.0) is None
    assert s.word == ""
