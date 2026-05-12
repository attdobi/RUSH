"""Process-lifetime build identifier for cache-busting web assets."""
from __future__ import annotations

from datetime import datetime, timezone
import os


def _new_build_id() -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M")
    suffix = os.urandom(2).hex()
    return f"{timestamp}-{suffix}"


BUILD_ID: str = _new_build_id()


def get_build_id() -> str:
    """Return the process-stable build identifier."""
    return BUILD_ID
