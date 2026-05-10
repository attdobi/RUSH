"""Shared helpers for the scoring subpackage.

Stdlib-only; no network. The helpers cover:
    * SME ground-truth loading from the existing combined_labels manifest
    * LabelVote loading (schema-tolerant; X1 may add fields)
    * Optional jsonschema validation (best-effort)
    * prepared_image_* metadata extraction (audit pass-through)
    * deterministic sorting helpers
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import logging
from pathlib import Path
from typing import Any, Iterable

logger = logging.getLogger(__name__)

# Cold-start L0 vocabulary. The LabelVote schema also allows
# {positive, negative} for warm-start; we keep this binary for v1 GenAI.
COLD_START_LABELS = ("gen_ai", "not_gen_ai")
ABSTAIN = "abstain"
POSITIVE_CLASS = "gen_ai"  # binary positive for DQ math at cold-start

# Fields X1 attaches to label records for downsample audit pass-through.
# Any field listed here is preserved verbatim in web exports when present.
PREP_METADATA_FIELDS = (
    "prepared_image_sha256",
    "prepared_image_width",
    "prepared_image_height",
    "prepared_image_mime",
    "prepared_image_bytes",
)


@dataclass(frozen=True)
class GroundTruth:
    image_id: str
    label: str  # gen_ai | not_gen_ai
    truth_tier: str
    split: str
    repo_rel_path: str


def _coerce_truth_label(raw: str, label_int: int | None) -> str:
    """Map manifest label strings to the cold-start L0 vocabulary."""
    if raw == "ai_generated" or label_int == 1:
        return "gen_ai"
    if raw == "not_ai_generated" or label_int == 0:
        return "not_gen_ai"
    raise ValueError(f"Unrecognized SME label '{raw}' (label_int={label_int})")


def load_ground_truth(
    manifest_path: Path,
    *,
    truth_tiers: tuple[str, ...] = ("gold", "platinum", "gold_candidate"),
    splits: tuple[str, ...] | None = None,
) -> dict[str, GroundTruth]:
    """Load SME ground truth keyed by image_id (sample_id).

    The combined_labels manifest is JSONL; each line carries the SME label.
    `truth_tiers` filters which records count as ground truth; default keeps
    gold/platinum AND the cold-start gold_candidate so v1 has data to score.
    """
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    out: dict[str, GroundTruth] = {}
    for line_no, raw in enumerate(manifest_path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"{manifest_path.name}:{line_no} invalid JSON: {exc}"
            ) from exc
        sample_id = record.get("sample_id")
        if not sample_id:
            continue
        tier = record.get("truth_tier", "")
        if truth_tiers and tier not in truth_tiers:
            continue
        if splits and record.get("split") not in splits:
            continue
        out[sample_id] = GroundTruth(
            image_id=sample_id,
            label=_coerce_truth_label(record.get("label", ""), record.get("label_int")),
            truth_tier=tier,
            split=record.get("split", ""),
            repo_rel_path=record.get("repo_rel_path", ""),
        )
    return out


def load_label_votes(votes_path: Path) -> list[dict[str, Any]]:
    """Load LabelVote JSONL records as plain dicts (schema-tolerant)."""
    if not votes_path.exists():
        raise FileNotFoundError(f"label_votes not found: {votes_path}")
    rows: list[dict[str, Any]] = []
    for line_no, raw in enumerate(votes_path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"{votes_path.name}:{line_no} invalid JSON: {exc}"
            ) from exc
    return rows


def labeler_id_for(vote: dict[str, Any]) -> str:
    """Stable labeler identity. Prefer explicit labeler_id, else model_id, else 'unknown'."""
    return str(vote.get("labeler_id") or vote.get("model_id") or "unknown")


def optional_confidence(value: Any) -> float | None:
    """Return numeric confidence, or None when it is missing/malformed."""
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def extract_prep_metadata(vote: dict[str, Any]) -> dict[str, Any]:
    """Return a shallow dict of prepared_image_* fields present on the vote.

    This is the single point where downsample-audit metadata is pulled forward
    into web-ready exports. If X1 adds more PREP_METADATA_FIELDS, this helper
    auto-picks them up; misalignment/borderline exporters never reach into the
    raw vote object directly.
    """
    out: dict[str, Any] = {}
    for field in PREP_METADATA_FIELDS:
        if field in vote and vote[field] is not None:
            out[field] = vote[field]
    return out


def try_validate(instance: Any, schema_path: Path, *, label: str) -> list[str]:
    """Best-effort JSON Schema validation. Returns list of error strings (empty = ok)."""
    try:
        import jsonschema  # type: ignore
    except Exception:
        logger.debug("jsonschema not installed; skipping validation for %s", label)
        return []
    if not schema_path.exists():
        logger.warning("schema missing for %s: %s", label, schema_path)
        return []
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    validator = jsonschema.Draft202012Validator(schema)
    errs = sorted(validator.iter_errors(instance), key=lambda e: list(e.path))
    return [f"{label}: {'/'.join(str(p) for p in e.path) or '<root>'}: {e.message}" for e in errs]


def deterministic_image_ids(rows: Iterable[dict[str, Any]]) -> list[str]:
    return sorted({r.get("image_id", "") for r in rows if r.get("image_id")})


def deterministic_labelers(rows: Iterable[dict[str, Any]]) -> list[str]:
    return sorted({labeler_id_for(r) for r in rows})
