"""Per-image consensus / majority-vote aggregation.

Given a stream of ``LabelVote`` dicts (one per image x model), emit one
:class:`ConsensusRecord` per image describing how labelers voted, the majority
label (if any), unanimity, ties, and a small audit array of voters.

Pure function — no I/O. Persistence lives in
:mod:`pipeline.scoring.exporters` and ``scripts/score_labels.py``.

Conventions (cold-start / GenAI v0.1):
    * ``label == "abstain"`` is treated as **non-decisive**.
    * ``n_votes_total`` includes abstains; ``n_votes_decisive`` does not.
    * ``majority_label`` is the most-voted decisive label; ``None`` on tie or
      when every voter abstained.
    * ``majority_fraction`` is ``majority_count / n_votes_decisive`` (``None``
      when nothing was decided).
    * ``is_unanimous`` is true iff every voter cast the same decisive label
      and no voter abstained.
    * ``is_consensus`` is true iff every decisive vote agrees, even if some
      voters abstained (i.e. unanimity among non-abstainers).
    * ``is_split`` is true if there is more than one distinct decisive label
      **or** the leaders tied.
    * ``tie`` is true iff two or more labels share the top decisive count.

Output is JSON-serializable (no datetimes — ``computed_at`` is an ISO 8601
string in UTC).
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from typing import Any, Iterable

from . import _common


def _now_iso() -> str:
    # Seconds-resolution UTC timestamp with explicit Z suffix.
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


def _voter_audit(vote: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {
        "model_id": str(
            vote.get("model_id") or vote.get("labeler_id") or "unknown"
        ),
        "labeler_id": _common.labeler_id_for(vote),
        "label": vote.get("label", _common.ABSTAIN),
    }
    out["confidence"] = _common.optional_confidence(vote.get("confidence"))
    out["is_boundary"] = bool(vote.get("is_boundary", False))
    return out


def _consensus_for_image(
    image_id: str,
    image_votes: list[dict[str, Any]],
    *,
    run_id: str,
    computed_at: str,
) -> dict[str, Any]:
    voters = [_voter_audit(v) for v in image_votes]
    # Stable order: by labeler_id for deterministic output.
    voters.sort(key=lambda r: (r.get("labeler_id") or "", r.get("model_id") or ""))

    decisive_labels = [v["label"] for v in voters if v["label"] != _common.ABSTAIN]
    n_total = len(voters)
    n_decisive = len(decisive_labels)
    n_abstain = n_total - n_decisive

    distribution = dict(sorted(Counter(decisive_labels).items()))

    if n_decisive == 0:
        majority_label: str | None = None
        majority_count = 0
        majority_fraction: float | None = None
        tie = False
        is_consensus = False
        is_unanimous = False
        is_split = False  # nothing decided → not "split" in any meaningful way
    else:
        ranked = Counter(decisive_labels).most_common()
        top_count = ranked[0][1]
        leaders = [lbl for lbl, c in ranked if c == top_count]
        tie = len(leaders) > 1
        majority_count = top_count
        majority_fraction = round(top_count / n_decisive, 6)
        majority_label = None if tie else leaders[0]
        distinct = len(set(decisive_labels))
        is_consensus = distinct == 1
        is_unanimous = is_consensus and n_abstain == 0
        is_split = (distinct > 1) or tie

    any_boundary = any(v.get("is_boundary") for v in voters)
    boundary_count = sum(1 for v in voters if v.get("is_boundary"))

    return {
        "run_id": run_id,
        "image_id": image_id,
        "n_votes_total": n_total,
        "n_votes_decisive": n_decisive,
        "n_abstain": n_abstain,
        "vote_distribution": distribution,
        "majority_label": majority_label,
        "majority_count": majority_count,
        "majority_fraction": majority_fraction,
        "is_unanimous": is_unanimous,
        "is_consensus": is_consensus,
        "is_split": is_split,
        "tie": tie,
        "voters": voters,
        "any_boundary_flag": any_boundary,
        "boundary_voter_count": boundary_count,
        "computed_at": computed_at,
    }


def build_consensus_records(
    votes: Iterable[dict[str, Any]],
    *,
    run_id: str | None = None,
    computed_at: str | None = None,
) -> list[dict[str, Any]]:
    """Aggregate per-image consensus from an iterable of LabelVote dicts.

    Determinism: records are returned sorted by ``image_id``; the ``voters``
    array within each record is sorted by ``labeler_id`` then ``model_id``.

    ``run_id`` is taken from the first vote that carries one when not supplied
    explicitly. ``computed_at`` defaults to the current UTC timestamp; pass an
    explicit value (e.g. in tests) for stable output.
    """
    grouped: dict[str, list[dict[str, Any]]] = {}
    resolved_run_id = run_id or ""
    for v in votes:
        image_id = v.get("image_id")
        if not image_id:
            continue
        grouped.setdefault(image_id, []).append(v)
        if not resolved_run_id:
            rid = v.get("run_id")
            if rid:
                resolved_run_id = str(rid)

    ts = computed_at or _now_iso()
    out: list[dict[str, Any]] = []
    for image_id in sorted(grouped.keys()):
        out.append(
            _consensus_for_image(
                image_id,
                grouped[image_id],
                run_id=resolved_run_id,
                computed_at=ts,
            )
        )
    return out


def build_cohort_rollups(
    records: list[dict[str, Any]],
    *,
    ground_truth: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compute cohort-level rollups over a list of consensus records.

    ``ground_truth`` is the dict keyed by ``image_id`` produced by
    :func:`pipeline.scoring._common.load_ground_truth` (values expose ``.label``).
    When supplied, ``majority_vs_sme_accuracy`` is computed over the subset of
    images that (a) have an SME truth row and (b) have a non-null
    ``majority_label``.

    Per-model rollups: for each labeler observed, count how many records its
    decisive vote agreed with the record's ``majority_label``. Abstain rows do
    not count toward agreement or disagreement (they're tracked separately as
    ``n_abstain``).
    """
    n_total = len(records)
    n_unanimous = sum(1 for r in records if r.get("is_unanimous"))
    n_consensus = sum(1 for r in records if r.get("is_consensus"))
    n_split = sum(1 for r in records if r.get("is_split"))
    n_tie = sum(1 for r in records if r.get("tie"))
    n_boundary = sum(1 for r in records if r.get("any_boundary_flag"))

    rollup: dict[str, Any] = {
        "n_images_total": n_total,
        "n_images_unanimous": n_unanimous,
        "n_images_consensus": n_consensus,
        "n_images_split": n_split,
        "n_images_with_tie": n_tie,
        "n_images_with_boundary_flag": n_boundary,
    }

    # majority_vs_sme_accuracy
    if ground_truth:
        compared = 0
        correct = 0
        for r in records:
            image_id = r.get("image_id")
            gt = ground_truth.get(image_id) if image_id else None
            if not gt:
                continue
            majority = r.get("majority_label")
            if majority is None:
                continue
            compared += 1
            if majority == getattr(gt, "label", None):
                correct += 1
        rollup["majority_vs_sme_compared"] = compared
        rollup["majority_vs_sme_correct"] = correct
        rollup["majority_vs_sme_accuracy"] = (
            round(correct / compared, 6) if compared else None
        )

    # per-model agreement with majority
    per_model: dict[str, dict[str, int]] = {}
    for r in records:
        majority = r.get("majority_label")
        for voter in r.get("voters", []):
            mid = voter.get("labeler_id") or voter.get("model_id") or "unknown"
            slot = per_model.setdefault(
                mid,
                {"n_votes": 0, "n_decisive": 0, "n_agreed": 0, "n_abstain": 0},
            )
            slot["n_votes"] += 1
            if voter.get("label") == _common.ABSTAIN:
                slot["n_abstain"] += 1
                continue
            slot["n_decisive"] += 1
            if majority is not None and voter.get("label") == majority:
                slot["n_agreed"] += 1

    per_model_out: dict[str, dict[str, Any]] = {}
    for mid in sorted(per_model.keys()):
        slot = per_model[mid]
        agreement = (
            round(slot["n_agreed"] / slot["n_decisive"], 6)
            if slot["n_decisive"]
            else None
        )
        per_model_out[mid] = {
            **slot,
            "agreement_with_majority": agreement,
        }
    rollup["per_model_vs_majority_agreement"] = per_model_out

    return rollup
