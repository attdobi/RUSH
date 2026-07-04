"""Atomic JSONL append + schema validation hooks for run artifacts (X2).

Every persisted record is validated against its JSON Schema before write.
Invalid records are rerouted to ``errors.jsonl`` with a structured reason so
the runner never silently drops bad data.

The append API is intentionally tiny:

    persistence.append_label_vote(paths, vote)
    persistence.append_llm_output(paths, output)
    persistence.append_error(paths, ...)
    persistence.write_run_manifest(paths, manifest)

`jsonschema` (4.x) is the only third-party dependency, and it's already used
by ``scripts/validate_foundation.py``.
"""
from __future__ import annotations

from datetime import datetime, timezone
from functools import lru_cache
import json
import os
from pathlib import Path
import tempfile
from typing import Any

from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError

from .io_paths import REPO_ROOT, RunPaths

SCHEMAS_DIR = REPO_ROOT / "schemas"

LABEL_VOTE_SCHEMA = "label-vote.schema.json"
LLM_OUTPUT_SCHEMA = "llm-output.schema.json"
RUN_MANIFEST_SCHEMA = "run-manifest.schema.json"


class PersistenceError(RuntimeError):
    """Raised when a record cannot be validated or persisted."""


@lru_cache(maxsize=None)
def _validator(schema_filename: str) -> Draft202012Validator:
    schema_path = SCHEMAS_DIR / schema_filename
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    return Draft202012Validator(schema)


def validate_record(record: dict[str, Any], schema_filename: str) -> list[str]:
    """Return human-readable validation error strings (empty list = valid)."""
    validator = _validator(schema_filename)
    return [
        f"{'.'.join(str(p) for p in err.absolute_path) or '<root>'}: {err.message}"
        for err in sorted(validator.iter_errors(record), key=lambda e: list(e.absolute_path))
    ]


def _utcnow_isoformat() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _atomic_write_text(path: Path, text: str) -> None:
    """Atomic write via tmpfile + os.replace (same directory)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_name, path)
    except Exception:
        # Best-effort cleanup of the tmpfile if rename failed.
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    """Append one JSON record + newline. Single short write keeps it atomic-enough."""
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, sort_keys=True, ensure_ascii=False, separators=(",", ":")) + "\n"
    with path.open("a", encoding="utf-8") as fh:
        fh.write(line)


def _strip_image_bytes(payload: Any) -> Any:
    """Defence-in-depth: scrub anything that looks like base64 image data.

    The runner already builds payloads without raw bytes, but persisted
    ``raw_provider_payload`` MUST never contain image bytes (§5.3).
    """
    if isinstance(payload, dict):
        cleaned: dict[str, Any] = {}
        for k, v in payload.items():
            lk = str(k).lower()
            if lk in {"image", "image_bytes", "image_b64", "image_base64", "data", "b64", "b64_json"}:
                cleaned[k] = "<image-bytes-omitted>"
            elif lk in {"image_url"} and isinstance(v, str) and v.startswith("data:"):
                cleaned[k] = "<image-bytes-omitted>"
            elif lk in {"inline_data"} and isinstance(v, dict):
                inner = dict(v)
                if "data" in inner:
                    inner["data"] = "<image-bytes-omitted>"
                cleaned[k] = inner
            else:
                cleaned[k] = _strip_image_bytes(v)
        return cleaned
    if isinstance(payload, list):
        return [_strip_image_bytes(x) for x in payload]
    return payload


# ---------------------------------------------------------------------------
# Public append helpers
# ---------------------------------------------------------------------------


def append_label_vote(paths: RunPaths, vote: dict[str, Any]) -> None:
    """Validate against label-vote.schema.json then append; reroute on failure."""
    errors = validate_record(vote, LABEL_VOTE_SCHEMA)
    if errors:
        append_error(
            paths,
            stage="label_vote_validation",
            image_id=vote.get("image_id", ""),
            model_id=vote.get("model_id", ""),
            reason="; ".join(errors),
            record=vote,
        )
        raise PersistenceError(
            f"label_vote failed schema validation: {'; '.join(errors)}"
        )
    _append_jsonl(paths.label_votes, vote)


def append_llm_output(paths: RunPaths, output: dict[str, Any], *, image_id: str, model_id: str) -> None:
    """Validate against llm-output.schema.json then append; reroute on failure.

    `image_id` and `model_id` are passed in separately because the schema
    object itself does not carry them (it mirrors the LLM's structured JSON).
    They are echoed into the appended row so X3 can join on them.
    """
    errors = validate_record(output, LLM_OUTPUT_SCHEMA)
    if errors:
        append_error(
            paths,
            stage="llm_output_validation",
            image_id=image_id,
            model_id=model_id,
            reason="; ".join(errors),
            record=output,
        )
        raise PersistenceError(
            f"llm_output failed schema validation: {'; '.join(errors)}"
        )
    envelope = {
        "image_id": image_id,
        "model_id": model_id,
        "recorded_at": _utcnow_isoformat(),
        "output": output,
    }
    _append_jsonl(paths.llm_outputs, envelope)


def append_cost_row(paths: RunPaths, row: dict[str, Any]) -> None:
    """Append one row to the durable per-image cost ledger (costs.jsonl).

    The ledger is analysis data (not a validated schema artifact); rows are
    self-describing (rates + pricing_version). Never raises on content — a
    ledger write must not break a labeling run.
    """
    _append_jsonl(paths.costs, row)


def append_error(
    paths: RunPaths,
    *,
    stage: str,
    image_id: str = "",
    model_id: str = "",
    reason: str,
    record: dict[str, Any] | None = None,
    attempts: int = 1,
) -> None:
    """Append a structured failure entry. Never persists secrets."""
    payload = {
        "recorded_at": _utcnow_isoformat(),
        "stage": stage,
        "image_id": image_id,
        "model_id": model_id,
        "attempts": attempts,
        "reason": reason,
    }
    if record is not None:
        payload["record"] = _strip_image_bytes(record)
    _append_jsonl(paths.errors, payload)


def write_run_manifest(paths: RunPaths, manifest: dict[str, Any]) -> None:
    """Validate then atomically (over)write the per-run manifest."""
    errors = validate_record(manifest, RUN_MANIFEST_SCHEMA)
    if errors:
        raise PersistenceError(
            f"run_manifest failed schema validation: {'; '.join(errors)}"
        )
    text = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    _atomic_write_text(paths.manifest, text)


__all__ = [
    "PersistenceError",
    "validate_record",
    "append_label_vote",
    "append_llm_output",
    "append_error",
    "write_run_manifest",
    "LABEL_VOTE_SCHEMA",
    "LLM_OUTPUT_SCHEMA",
    "RUN_MANIFEST_SCHEMA",
]
