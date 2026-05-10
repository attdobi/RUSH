"""Shared pytest fixtures — RUSH bulk-labeling tests.

Adds the repo root to ``sys.path`` so ``import pipeline`` works when pytest
is invoked from anywhere inside the repo.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
