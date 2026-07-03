"""Scoring task registry.

A :class:`ScoringTask` describes the label space a scoring run operates over so
downstream code can dispatch between the cold-start **binary** GenAI path and a
generic **multiclass** path (e.g. the MNIST 0-9 demo) without hard-coding the
vocabulary.

Design notes
------------
* ``positive_class is None`` is the multiclass signal. Binary tasks name a
  single positive class (used by :func:`pipeline.scoring.decision_quality.compute_metrics`).
* ``classes`` is the ordered, closed label vocabulary. Predictions outside this
  set (other than ``abstain``) are treated as *wrong* by the multiclass metrics
  (see :mod:`pipeline.scoring.decision_quality_multiclass`).
* ``abstain`` is the non-decisive sentinel; it is excluded from metric
  numerators/denominators and counted separately, matching the binary
  convention.

The registry is intentionally tiny and stdlib-only. Callers that do not specify
a task default to :data:`GENAI_BINARY` for backward compatibility.
"""
from __future__ import annotations

from dataclasses import dataclass

from . import _common


@dataclass(frozen=True)
class ScoringTask:
    """Immutable description of a scoring label space.

    Attributes:
        name: Stable task identifier (e.g. ``"genai_binary"``).
        classes: Ordered, closed vocabulary of decidable class labels.
        positive_class: The positive label for binary DQ math, or ``None`` for
            a multiclass task.
        abstain: The non-decisive sentinel label (excluded from metrics).
    """

    name: str
    classes: tuple[str, ...]
    positive_class: str | None
    abstain: str = _common.ABSTAIN

    @property
    def is_binary(self) -> bool:
        """True when the task has a designated positive class (binary DQ)."""
        return self.positive_class is not None

    @property
    def is_multiclass(self) -> bool:
        """True when no positive class is set (multiclass DQ)."""
        return self.positive_class is None


# Cold-start GenAI: gen_ai vs not_gen_ai, positive = gen_ai. This is the
# default so existing callers keep their exact behavior.
GENAI_BINARY = ScoringTask(
    name="genai_binary",
    classes=_common.COLD_START_LABELS,
    positive_class=_common.POSITIVE_CLASS,
    abstain=_common.ABSTAIN,
)

# MNIST demo: digit labels "0".."9", no positive class → multiclass path.
MNIST_MULTICLASS = ScoringTask(
    name="mnist_multiclass",
    classes=tuple(str(d) for d in range(10)),
    positive_class=None,
    abstain=_common.ABSTAIN,
)


_REGISTRY: dict[str, ScoringTask] = {
    GENAI_BINARY.name: GENAI_BINARY,
    MNIST_MULTICLASS.name: MNIST_MULTICLASS,
}

# The task used when a caller does not specify one (backward compatibility).
DEFAULT_TASK = GENAI_BINARY


def get_task(name: str) -> ScoringTask:
    """Look up a registered task by name.

    Raises:
        KeyError: if ``name`` is not registered (message lists known tasks).
    """
    try:
        return _REGISTRY[name]
    except KeyError as exc:
        known = ", ".join(sorted(_REGISTRY)) or "<none>"
        raise KeyError(f"unknown scoring task '{name}'; known tasks: {known}") from exc


def register_task(task: ScoringTask) -> None:
    """Register (or replace) a task by its ``name``. Intended for tests/plugins."""
    _REGISTRY[task.name] = task


def available_tasks() -> tuple[str, ...]:
    """Return the sorted names of all registered tasks."""
    return tuple(sorted(_REGISTRY))
