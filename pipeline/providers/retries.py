"""Exponential backoff with ``Retry-After`` honor for provider clients.

Hard rules (per docs/EXECUTION-PLAN-bulk-labeling-v1.md §5.5 & §5.7):

* Exponential backoff: base 1s, jitter, cap 60s, max 6 attempts.
* Honor ``Retry-After`` headers when the provider exposes them.
* No tight loops — every retry path goes through ``time.sleep``.

The :func:`retry_call` helper is intentionally tiny and dependency-free so
clients can wrap their SDK calls without pulling in tenacity / urllib3
behavior we don't control.
"""

from __future__ import annotations

import logging
import random
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass(frozen=True)
class RetryPolicy:
    """Backoff knobs for :func:`retry_call`."""

    max_attempts: int = 6
    base_delay_s: float = 1.0
    cap_delay_s: float = 60.0
    jitter_s: float = 0.5

    def delay_for_attempt(self, attempt_idx: int) -> float:
        """Compute delay before the *next* attempt (0-indexed)."""
        # 0-indexed attempt: 0 -> base, 1 -> 2*base, 2 -> 4*base ...
        exp = self.base_delay_s * (2 ** max(0, attempt_idx))
        jitter = random.uniform(0.0, self.jitter_s)
        return min(self.cap_delay_s, exp) + jitter


DEFAULT_POLICY: RetryPolicy = RetryPolicy()


class _RetryableError(Exception):
    """Internal sentinel — callers raise their own provider exceptions."""


def _coerce_retry_after(value: object) -> float | None:
    """Best-effort parse of ``Retry-After`` header values (seconds only).

    HTTP date forms are intentionally ignored — providers we use return
    integer/float seconds. Returns ``None`` if the value isn't a positive
    number.
    """
    if value is None:
        return None
    try:
        secs = float(value)
    except (TypeError, ValueError):
        return None
    if secs <= 0:
        return None
    return secs


def retry_call(
    fn: Callable[[], T],
    *,
    is_retryable: Callable[[BaseException], bool],
    extract_retry_after: Callable[[BaseException], float | None] | None = None,
    policy: RetryPolicy = DEFAULT_POLICY,
    sleep: Callable[[float], None] = time.sleep,
    label: str = "provider_call",
) -> T:
    """Run ``fn`` with exponential backoff for retryable failures.

    Args:
        fn: Zero-arg callable that performs the network call.
        is_retryable: Predicate over the raised exception. Return ``True``
            for transient errors (rate-limit, timeout, 5xx).
        extract_retry_after: Optional extractor that pulls a server-supplied
            wait-hint (in seconds) out of the exception. When this returns a
            positive float, it overrides the computed backoff for that step.
        policy: Backoff policy. Defaults to ``RetryPolicy()``.
        sleep: Injectable sleep — pass a no-op in tests to keep them fast.
        label: Free-form label for log lines (e.g. ``"openai.responses"``).

    Returns:
        Whatever ``fn`` returns on the first successful attempt.

    Raises:
        BaseException: The last exception raised by ``fn`` once attempts are
            exhausted, OR the first non-retryable exception.
    """
    last_exc: BaseException | None = None
    attempts = max(1, policy.max_attempts)
    for attempt in range(attempts):
        try:
            return fn()
        except BaseException as exc:  # noqa: BLE001 - we re-raise below
            last_exc = exc
            if not is_retryable(exc):
                raise
            if attempt + 1 >= attempts:
                # Out of attempts — surface the original exception.
                raise

            wait: float | None = None
            if extract_retry_after is not None:
                try:
                    wait = extract_retry_after(exc)
                except Exception:  # pragma: no cover - extractors must be cheap
                    wait = None
            if wait is None or wait <= 0:
                wait = policy.delay_for_attempt(attempt)

            wait = min(wait, policy.cap_delay_s + policy.jitter_s)
            # Log the *type* and label only — never the exception message,
            # which could carry headers or partial responses.
            logger.warning(
                "%s: retryable %s on attempt %d/%d; sleeping %.2fs",
                label,
                type(exc).__name__,
                attempt + 1,
                attempts,
                wait,
            )
            sleep(wait)

    # Defensive — loop should always return or raise.
    assert last_exc is not None  # pragma: no cover
    raise last_exc


__all__ = [
    "RetryPolicy",
    "DEFAULT_POLICY",
    "retry_call",
    "_coerce_retry_after",
]
