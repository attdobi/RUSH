"""Flip-rate scoring across repeated labeling runs.

A flip-rate record summarizes whether a single ``image_id`` x ``model_id`` pair
changed labels across multiple run directories. The module intentionally keeps
all logic pure and stdlib-only; the only I/O is reading ``label_votes.jsonl``
from caller-provided run directories.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Sequence

from . import _common


@dataclass(frozen=True)
class FlipRateRecord:
    image_id: str
    model_id: str
    n_runs: int
    labels_observed: list[str]
    label_counts: dict[str, int]
    distinct_label_count: int
    flip_count: int
    flip_rate: float
    stable_label: str | None
    abstain_count: int
    confidence_min: float | None
    confidence_max: float | None
    confidence_mean: float | None
    first_seen_run_id: str
    last_seen_run_id: str


@dataclass(frozen=True)
class _Observation:
    run_id: str
    image_id: str
    model_id: str
    label: str
    confidence: float | None


def _numeric_confidence(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _run_id_for_vote(vote: dict[str, Any], run_dir: Path) -> str:
    run_id = vote.get("run_id")
    return str(run_id) if run_id else run_dir.name


def _observations_from_run_dir(run_dir: Path) -> list[_Observation]:
    votes_path = run_dir / "label_votes.jsonl"
    if not votes_path.exists():
        return []

    observations: list[_Observation] = []
    for vote in _common.load_label_votes(votes_path):
        image_id = vote.get("image_id")
        model_id = vote.get("model_id") or vote.get("labeler_id")
        if not image_id or not model_id:
            continue
        observations.append(
            _Observation(
                run_id=_run_id_for_vote(vote, run_dir),
                image_id=str(image_id),
                model_id=str(model_id),
                label=str(vote.get("label", _common.ABSTAIN)),
                confidence=_numeric_confidence(vote.get("confidence")),
            )
        )
    return observations


def _clamp_0_1(value: float) -> float:
    return max(0.0, min(1.0, value))


def build_flip_rate_records(run_dirs: Sequence[Path]) -> list[FlipRateRecord]:
    """Build flip-rate records from repeated run directories.

    Missing ``label_votes.jsonl`` files are skipped. Records are grouped by
    ``(image_id, model_id)`` across all observed runs and returned in stable
    ``image_id``/``model_id`` order. Within a group, observations are sorted by
    ``run_id`` alphabetically so first/last run IDs are deterministic.
    """
    grouped: dict[tuple[str, str], list[_Observation]] = defaultdict(list)
    for run_dir in run_dirs:
        for obs in _observations_from_run_dir(Path(run_dir)):
            grouped[(obs.image_id, obs.model_id)].append(obs)

    records: list[FlipRateRecord] = []
    for (image_id, model_id) in sorted(grouped.keys()):
        observations = sorted(grouped[(image_id, model_id)], key=lambda o: o.run_id)
        labels = [obs.label for obs in observations]
        label_counts = dict(sorted(Counter(labels).items()))
        labels_observed = list(label_counts.keys())
        distinct_label_count = len(labels_observed)
        n_runs = len(observations)
        flip_count = max(0, distinct_label_count - 1)
        flip_rate = _clamp_0_1(flip_count / max(1, n_runs - 1))
        confidences = [obs.confidence for obs in observations if obs.confidence is not None]
        records.append(
            FlipRateRecord(
                image_id=image_id,
                model_id=model_id,
                n_runs=n_runs,
                labels_observed=labels_observed,
                label_counts=label_counts,
                distinct_label_count=distinct_label_count,
                flip_count=flip_count,
                flip_rate=flip_rate,
                stable_label=labels_observed[0] if distinct_label_count == 1 else None,
                abstain_count=label_counts.get(_common.ABSTAIN, 0),
                confidence_min=min(confidences) if confidences else None,
                confidence_max=max(confidences) if confidences else None,
                confidence_mean=mean(confidences) if confidences else None,
                first_seen_run_id=observations[0].run_id,
                last_seen_run_id=observations[-1].run_id,
            )
        )
    return records


def _mean(values: Sequence[float]) -> float | None:
    return mean(values) if values else None


def cohort_rollups(records: Sequence[FlipRateRecord]) -> dict[str, Any]:
    """Compute cohort-level rollups over flip-rate records."""
    records_list = list(records)
    n_pairs_total = len(records_list)
    n_pairs_stable = sum(1 for r in records_list if r.flip_count == 0)
    n_pairs_flipped = sum(1 for r in records_list if r.flip_count > 0)

    per_model: dict[str, list[float]] = defaultdict(list)
    for record in records_list:
        if record.n_runs >= 2:
            per_model[record.model_id].append(record.flip_rate)
    per_model_flip_rate = {
        model_id: mean(values) for model_id, values in sorted(per_model.items())
    }

    by_image: dict[str, list[FlipRateRecord]] = defaultdict(list)
    for record in records_list:
        by_image[record.image_id].append(record)

    image_rollups: list[dict[str, Any]] = []
    for image_id, image_records in by_image.items():
        labels = sorted({label for r in image_records for label in r.labels_observed})
        models_that_flipped = sorted(r.model_id for r in image_records if r.flip_count > 0)
        rates = [r.flip_rate for r in image_records]
        image_rollups.append(
            {
                "image_id": image_id,
                "mean_flip_rate": mean(rates),
                "max_flip_rate": max(rates),
                "models_that_flipped": models_that_flipped,
                "labels_observed": labels,
                "_max_flip_count": max(r.flip_count for r in image_records),
            }
        )

    image_rollups.sort(
        key=lambda r: (
            -int(r["_max_flip_count"]),
            -float(r["mean_flip_rate"]),
            str(r["image_id"]),
        )
    )
    per_image_flip_rate = [
        {k: v for k, v in r.items() if k != "_max_flip_count"}
        for r in image_rollups[:20]
    ]

    return {
        "n_pairs_total": n_pairs_total,
        "n_pairs_stable": n_pairs_stable,
        "n_pairs_flipped": n_pairs_flipped,
        "mean_flip_rate": _mean([r.flip_rate for r in records_list]),
        "per_model_flip_rate": per_model_flip_rate,
        "per_image_flip_rate": per_image_flip_rate,
    }
