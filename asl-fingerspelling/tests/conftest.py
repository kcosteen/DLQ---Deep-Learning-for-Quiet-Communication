"""Put the repo root on sys.path so ``import src`` / ``import app`` work in tests."""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
