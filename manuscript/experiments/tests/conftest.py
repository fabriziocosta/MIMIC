"""Test path setup for local streamlined experiment package."""

from __future__ import annotations

import sys
from pathlib import Path


EXPERIMENT_SRC = Path(__file__).resolve().parents[1] / "src"
if str(EXPERIMENT_SRC) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_SRC))
