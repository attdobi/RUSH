"""Policy iteration: turn misclassifications into proposed PolicyPatches.

This module assembles a structured prompt from:
    * the bundled policy-graph markdown
    * the run's misalignment.json (high/medium severity rows by default)
    * (optionally) borderline.json highlights

…and asks an LLM (typically GPT-5.5 reasoning=high) to produce a JSON list of
``PolicyPatch`` records validated against ``schemas/policy-patch.schema.json``.

Hard rules
----------
* **No image bytes are sent by default.** ``include_images=False`` is the
  only safe default; ``include_images=True`` is rejected unless a
  ``downsample_helper`` callable is supplied. The helper MUST be the shared
  X1 downsample utility — original image bytes are never read directly here.
* The LLM client is injected (`chat_callable`) so tests stay offline.
* All persisted patches are schema-validated; failures land in ``errors``.
* Engineers do not execute live calls — the CLI defaults to ``--dry-run``.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import logging
import re
from pathlib import Path
from typing import Any, Callable, Iterable, Protocol

from .scoring import _common

logger = logging.getLogger(__name__)


class DownsampleHelper(Protocol):
    """Shape contract for X1's shared downsample helper.

    The helper takes an absolute image path and returns prepared image
    metadata (sha256, width, height, mime, bytes_b64, etc.). The policy
    iterator only needs the metadata + bytes_b64; it never reads the raw
    file directly.
    """

    def __call__(self, image_path: Path) -> dict[str, Any]:  # pragma: no cover
        ...


ChatCallable = Callable[..., str]
"""LLM call signature: ``chat(messages, *, model_id, reasoning_effort) -> str``.

The string returned MUST be a JSON document (object with ``patches`` array,
or a bare array of patches). Provider-specific transports live in
:mod:`pipeline.providers` (X1).
"""


@dataclass(frozen=True)
class PolicyIterationInputs:
    misalignment: dict[str, Any]
    borderline: dict[str, Any] | None
    policy_markdown: str  # concatenated policy-graph MD bundle
    policy_graph_version: str


# ---------------------------------------------------------------------------
# Prompt assembly
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = (
    "You are RUSH's policy iterator. Given the bound policy graph and a list "
    "of model/SME misalignments, propose minimal PolicyPatch JSON records. "
    "Each patch MUST conform to schemas/policy-patch.schema.json: required "
    "fields are patch_id, status, suggestion_type, target_nodes (array), "
    "rationale, proposed_diff (array). Use status='proposed'. Be conservative: "
    "prefer clarification_with_examples over new_subcategory unless the "
    "evidence is overwhelming. NEVER invent image bytes or example URLs."
)


def _select_priority_rows(
    misalignment: dict[str, Any],
    *,
    severity: tuple[str, ...] = ("high", "medium"),
    max_rows: int = 40,
) -> list[dict[str, Any]]:
    rows = [
        r for r in misalignment.get("records", [])
        if r.get("severity") in severity
        and r.get("misalignment_type") != "all_agree"
    ]
    # high first, then medium, deterministic by image_id
    order = {s: i for i, s in enumerate(severity)}
    rows.sort(key=lambda r: (order.get(r.get("severity"), 99), r.get("image_id", "")))
    return rows[:max_rows]


def _vote_summary(vote: dict[str, Any]) -> dict[str, Any]:
    return {
        "labeler_id": _common.labeler_id_for(vote),
        "label": vote.get("label"),
        "l2_label": vote.get("l2_label"),
        "confidence": vote.get("confidence"),
        "is_boundary": bool(vote.get("is_boundary", False)),
        "difficulty": vote.get("difficulty"),
        "justification": vote.get("justification", "")[:600],
    }


def _row_summary(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "image_id": row["image_id"],
        "sme_truth": row["sme_truth"],
        "misalignment_type": row.get("misalignment_type"),
        "severity": row.get("severity"),
        "votes": [_vote_summary(v) for v in row.get("votes", [])],
    }


def build_user_prompt(
    inputs: PolicyIterationInputs,
    *,
    severity: tuple[str, ...] = ("high", "medium"),
    max_rows: int = 40,
    include_images: bool = False,
    downsample_helper: DownsampleHelper | None = None,
    image_root: Path | None = None,
) -> dict[str, Any]:
    """Build the structured user payload sent to the LLM.

    Returns a JSON-serializable dict with keys::

        policy_graph_version, policy_markdown, misclassifications,
        borderline_highlights (optional), images (optional, downsampled only)
    """
    if include_images and downsample_helper is None:
        raise ValueError(
            "include_images=True requires a downsample_helper (X1's shared utility)."
        )

    rows = _select_priority_rows(inputs.misalignment, severity=severity, max_rows=max_rows)
    payload: dict[str, Any] = {
        "policy_graph_version": inputs.policy_graph_version,
        "policy_markdown": inputs.policy_markdown,
        "misclassifications": [_row_summary(r) for r in rows],
    }
    if inputs.borderline:
        # send only counts + first-N examples per L0 group
        groups = inputs.borderline.get("groups", {})
        payload["borderline_highlights"] = {
            l0: [
                {"image_id": r.get("image_id"), "reasons": r.get("reasons", [])}
                for r in (recs[:10] if isinstance(recs, list) else [])
            ]
            for l0, recs in groups.items()
        }
    if include_images and downsample_helper is not None:
        if image_root is None:
            raise ValueError("include_images=True requires image_root.")
        image_blocks: list[dict[str, Any]] = []
        for r in rows:
            rel = (r.get("repo_rel_path") or "").lstrip("/")
            if not rel:
                continue
            abs_path = image_root / rel
            try:
                meta = downsample_helper(abs_path)
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("downsample failed for %s: %s", abs_path, exc)
                continue
            image_blocks.append({"image_id": r["image_id"], "prepared": meta})
        payload["images"] = image_blocks
    return payload


# ---------------------------------------------------------------------------
# LLM invocation + validation
# ---------------------------------------------------------------------------

def _parse_patches(raw: str) -> list[dict[str, Any]]:
    try:
        doc = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"LLM response was not valid JSON: {exc}") from exc
    if isinstance(doc, dict) and "patches" in doc:
        patches = doc["patches"]
    elif isinstance(doc, list):
        patches = doc
    else:
        raise ValueError("LLM response must be a JSON array or {'patches': [...]}.")
    if not isinstance(patches, list):
        raise ValueError("'patches' must be a list")
    return patches


def propose_policy_patches(
    *,
    inputs: PolicyIterationInputs,
    chat_callable: ChatCallable | None,
    model_id: str = "openai/gpt-5.5",
    reasoning_effort: str = "high",
    severity: tuple[str, ...] = ("high", "medium"),
    max_rows: int = 40,
    include_images: bool = False,
    downsample_helper: DownsampleHelper | None = None,
    image_root: Path | None = None,
    schemas_dir: Path | None = None,
) -> dict[str, Any]:
    """Run a policy-iteration step. Returns ``{patches: [...], errors: [...], prompt: {...}}``.

    Set ``chat_callable=None`` to dry-run (no network call). The function will
    still build the prompt and return ``patches=[]``. This is what the CLI uses
    by default while engineers are wiring the pipeline.
    """
    user_payload = build_user_prompt(
        inputs,
        severity=severity,
        max_rows=max_rows,
        include_images=include_images,
        downsample_helper=downsample_helper,
        image_root=image_root,
    )
    if chat_callable is None:
        return {"patches": [], "errors": [], "prompt": user_payload, "dry_run": True}

    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {
            "role": "user",
            "content": json.dumps(user_payload, sort_keys=False, indent=2),
        },
    ]
    raw = chat_callable(
        messages, model_id=model_id, reasoning_effort=reasoning_effort
    )
    patches = _parse_patches(raw)

    valid: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    schema_path = (
        schemas_dir / "policy-patch.schema.json" if schemas_dir else None
    )
    for idx, patch in enumerate(patches):
        if not isinstance(patch, dict):
            errors.append({"index": idx, "reason": "patch is not an object"})
            continue
        if schema_path is not None:
            errs = _common.try_validate(
                patch, schema_path, label=f"policy-patch[{idx}]"
            )
            if errs:
                errors.append({"index": idx, "reason": "; ".join(errs), "patch": patch})
                continue
        valid.append(patch)
    return {
        "patches": valid,
        "errors": errors,
        "prompt": user_payload,
        "dry_run": False,
    }


# ---------------------------------------------------------------------------
# Markdown bundling helper
# ---------------------------------------------------------------------------

def load_policy_markdown(policy_dir: Path) -> str:
    """Concatenate policy-graph MD files in deterministic order.

    Each file is prefixed with a ``<!-- name.md -->`` marker so the bundle stays
    legible. Draft LLMs sometimes echo that marker back into a proposed file, so
    every place that writes or parses a node runs the content through
    ``strip_leading_markers`` first — otherwise the echoed comment sits before the
    ``---`` frontmatter and the (start-of-file-anchored) parsers read the node as
    untyped/unparented. See ``strip_leading_markers``.
    """
    md_files = sorted(p for p in policy_dir.glob("*.md") if p.is_file())
    chunks: list[str] = []
    for p in md_files:
        chunks.append(f"\n\n<!-- {p.name} -->\n")
        chunks.append(strip_leading_markers(p.read_text(encoding="utf-8")))
    return "".join(chunks).strip() + "\n"


# Matches one-or-more HTML comments (with surrounding whitespace) at the very
# start of a string — e.g. the injected ``<!-- GA.root.md -->`` bundle markers.
_LEADING_MARKER_RE = re.compile(r"\A(?:\s*<!--.*?-->)+\s*", re.DOTALL)


def strip_leading_markers(text: str) -> str:
    """Drop any leading HTML comment markers so ``---`` frontmatter starts at BOF.

    A well-formed policy node begins with ``---``; this only removes comment
    cruft before it (single or doubled ``<!-- name.md -->`` markers an LLM echoed
    into a proposal) and is a no-op for clean files.
    """
    return _LEADING_MARKER_RE.sub("", text, count=1)


def write_patches_jsonl(path: Path, patches: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for patch in patches:
            fh.write(json.dumps(patch, sort_keys=True))
            fh.write("\n")
    tmp.replace(path)
