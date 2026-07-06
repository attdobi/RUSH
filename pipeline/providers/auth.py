"""Secret loading for provider clients.

Hard rules (per docs/EXECUTION-PLAN-bulk-labeling-v1.md §5.2):

* Secrets come from a ``.env`` file or the process environment.
* We NEVER log secret values.
* We NEVER include secret values in error strings or exception payloads.
* Loading is idempotent and side-effect-light: existing env vars win over
  ``.env`` values (so CI / Pista can override at the shell).

Use :func:`get_secret` whenever a provider client needs an API key. It will
return the value (a non-empty string) or raise :class:`MissingSecretError`
with the env-var name only — never the value, never a substring of it.
"""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Final

try:  # python-dotenv is in requirements; tolerate absence so import-time errors
    # never blow up tests that don't touch real auth.
    from dotenv import load_dotenv as _load_dotenv  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - exercised only on bare envs
    _load_dotenv = None  # type: ignore[assignment]


# Override env var pointing at a custom .env path. Honoring this lets CI
# point at a fixture without polluting the repo's real .env.
DOTENV_PATH_ENV_VAR: Final[str] = "RUSH_DOTENV_PATH"

# Canonical env-var names per provider.
OPENAI_API_KEY_VAR: Final[str] = "OPENAI_API_KEY"
ANTHROPIC_API_KEY_VAR: Final[str] = "ANTHROPIC_API_KEY"
GEMINI_API_KEY_VAR: Final[str] = "GEMINI_API_KEY"
# Optional: Vertex/Google Cloud project for Gemini (informational; the genai
# client supports both API-key and ADC paths — we default to API key).
GOOGLE_CLOUD_PROJECT_VAR: Final[str] = "GOOGLE_CLOUD_PROJECT"


class MissingSecretError(RuntimeError):
    """Raised when a required secret is absent.

    The exception message intentionally contains ONLY the env-var name, never
    the value or any substring of it.
    """


_loaded: bool = False
_load_lock = threading.Lock()


def load_dotenv_once(*, dotenv_path: str | os.PathLike[str] | None = None) -> Path | None:
    """Load ``.env`` into ``os.environ`` exactly once per process.

    Existing env vars are NOT overwritten (``override=False``) so the shell
    always wins. Returns the path that was loaded, or ``None`` if no .env was
    found / dotenv is unavailable.

    Resolution order:

    1. Explicit ``dotenv_path`` argument.
    2. ``$RUSH_DOTENV_PATH`` environment variable.
    3. ``./.env`` next to the current working directory.

    Safe to call repeatedly AND concurrently; subsequent calls are no-ops.
    The lock matters: the flag must flip only AFTER the load completes, or a
    concurrent worker sees "loaded" while the env is still empty and raises a
    spurious MissingSecretError (observed as ~3 phantom failures at the start
    of concurrency-4 labeling runs).
    """
    global _loaded
    if _loaded:
        return None
    with _load_lock:
        if _loaded:
            return None
        try:
            if _load_dotenv is None:
                return None

            candidate: Path | None
            if dotenv_path is not None:
                candidate = Path(dotenv_path)
            elif os.environ.get(DOTENV_PATH_ENV_VAR):
                candidate = Path(os.environ[DOTENV_PATH_ENV_VAR])
            else:
                default = Path.cwd() / ".env"
                candidate = default if default.is_file() else None

            if candidate is None or not candidate.is_file():
                return None

            # override=False: existing env wins. Never log the path's contents.
            _load_dotenv(dotenv_path=str(candidate), override=False)
            return candidate
        finally:
            _loaded = True


def get_secret(
    var_name: str,
    *,
    required: bool = True,
    dotenv_path: str | os.PathLike[str] | None = None,
) -> str | None:
    """Return the value of ``var_name`` from env (loading .env first).

    Args:
        var_name: Env var name, e.g. ``"OPENAI_API_KEY"``.
        required: If ``True`` (default), raise :class:`MissingSecretError`
            when the variable is unset or empty. If ``False``, return
            ``None``.
        dotenv_path: Optional override path forwarded to
            :func:`load_dotenv_once`.

    Returns:
        The non-empty secret string, or ``None`` when ``required=False`` and
        the variable is unset.

    Raises:
        MissingSecretError: When ``required=True`` and no value is set.
            The exception message contains only ``var_name``.
    """
    load_dotenv_once(dotenv_path=dotenv_path)
    val = os.environ.get(var_name, "")
    if val:
        return val
    if required:
        # Critically: do NOT include any value, masked or otherwise.
        raise MissingSecretError(
            f"required environment variable not set: {var_name}"
        )
    return None


def reset_for_tests() -> None:
    """Reset the one-shot dotenv guard. For tests only."""
    global _loaded
    _loaded = False


__all__ = [
    "DOTENV_PATH_ENV_VAR",
    "OPENAI_API_KEY_VAR",
    "ANTHROPIC_API_KEY_VAR",
    "GEMINI_API_KEY_VAR",
    "GOOGLE_CLOUD_PROJECT_VAR",
    "MissingSecretError",
    "load_dotenv_once",
    "get_secret",
    "reset_for_tests",
]
