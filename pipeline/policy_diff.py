"""Policy proposal storage, diffs, and accept/reject lifecycle.

This module owns the server-side policy proposal workflow for the web API:
proposals are staged under ``data/policy_proposals/<proposal_id>/`` and only
``accept_proposal`` creates a new ``policy-graph/Generative_AI/vX.Y`` version.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
import difflib
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import time
import uuid
from typing import Any, Callable, Iterable

from pipeline.policy_iterator import (
    PolicyIterationInputs,
    build_user_prompt,
    load_policy_markdown,
)

DOMAIN = "Generative_AI"
DEFAULT_POLICY_MODEL = "openai/gpt-5.5"
ANTHROPIC_POLICY_MODEL = "anthropic/claude-opus-4-7"
ALLOWED_POLICY_MODELS = {DEFAULT_POLICY_MODEL, ANTHROPIC_POLICY_MODEL}
_VERSION_RE = re.compile(r"^v(\d+)\.(\d+)$")

ChatCallable = Callable[..., str]
ProgressCallable = Callable[[dict[str, Any]], None]

DEFAULT_LLM_TIMEOUT_S = 60.0
DEFAULT_LLM_RETRIES = 2
DEFAULT_LLM_BACKOFF_S = 1.0


@dataclass(frozen=True)
class ProposedFile:
    """Full proposed contents for one added/changed markdown file."""

    path: str
    content: str


def _repo_root(repo_root: Path | str) -> Path:
    return Path(repo_root).resolve()


def _policy_domain_dir(repo_root: Path | str, domain: str = DOMAIN) -> Path:
    return _repo_root(repo_root) / "policy-graph" / domain


def _version_dir(repo_root: Path | str, base_version: str, domain: str = DOMAIN) -> Path:
    version = _validate_version(base_version)
    path = _policy_domain_dir(repo_root, domain) / version
    if not path.is_dir():
        raise FileNotFoundError(f"unknown policy version: {version}")
    return path


def _proposal_root(repo_root: Path | str) -> Path:
    return _repo_root(repo_root) / "data" / "policy_proposals"


def _proposal_dir(repo_root: Path | str, proposal_id: str) -> Path:
    proposal_id = _validate_proposal_id(proposal_id)
    return _proposal_root(repo_root) / proposal_id


def _archive_dir(repo_root: Path | str, proposal_id: str) -> Path:
    proposal_id = _validate_proposal_id(proposal_id)
    return _proposal_root(repo_root) / "_archive" / proposal_id


def _validate_version(version: str) -> str:
    if not _VERSION_RE.match(version):
        raise ValueError(f"invalid policy version: {version!r}")
    return version


def _validate_proposal_id(proposal_id: str) -> str:
    if not proposal_id or "/" in proposal_id or ".." in proposal_id:
        raise ValueError(f"invalid proposal_id: {proposal_id!r}")
    return proposal_id


def _validate_md_filename(filename: str) -> str:
    path = PurePosixPath(filename)
    if (
        not filename
        or path.is_absolute()
        or len(path.parts) != 1
        or path.name in {".", ".."}
        or path.name != filename
        or not filename.endswith(".md")
    ):
        raise ValueError(f"invalid markdown filename: {filename!r}")
    return filename


def _atomic_write_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def _now_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")


def _new_proposal_id() -> str:
    return f"{_now_stamp()}-{uuid.uuid4().hex[:8]}"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


_PROPOSAL_STATUSES = {"pending", "accepted", "rejected", "parse_error"}


def _error_summary(exc: Exception) -> str:
    return f"{type(exc).__name__}: {exc}"


def _change_count(meta: dict[str, Any]) -> int:
    return sum(
        len(meta.get(key, []) or [])
        for key in ("files_changed", "files_added", "files_removed")
    )


def _proposal_summary(meta: dict[str, Any]) -> str:
    status = str(meta.get("status") or "pending")
    if status == "parse_error":
        err = meta.get("error") or meta.get("parse_error") or meta.get("raw_response")
        return str(err or "Proposal could not be parsed.")[:500]
    kind = meta.get("kind") or "propose_diff"
    changed = len(meta.get("files_changed", []) or [])
    added = len(meta.get("files_added", []) or [])
    removed = len(meta.get("files_removed", []) or [])
    return f"{kind}: {changed} changed, {added} added, {removed} removed"


def _normalize_proposal_meta(meta: dict[str, Any]) -> dict[str, Any]:
    """Return proposal metadata with stable list/detail fields for the UI."""
    out = dict(meta)
    status = str(out.get("status") or "pending")
    if status not in _PROPOSAL_STATUSES:
        status = "parse_error"
    out["status"] = status
    out.setdefault("kind", "propose_diff")
    out.setdefault("files_changed", [])
    out.setdefault("files_added", [])
    out.setdefault("files_removed", [])
    out["change_count"] = _change_count(out)
    out["summary"] = _proposal_summary(out)
    return out


def _run_has_score_inputs(run_dir: Path) -> bool:
    return (run_dir / "label_votes.jsonl").exists() and (run_dir / "llm_outputs.jsonl").exists()


def _load_run_inputs(repo_root: Path, run_id: str, base_version: str) -> PolicyIterationInputs:
    run_dir = repo_root / "data" / "runs" / run_id
    mis_path = run_dir / "scoring" / "misalignment.json"
    bord_path = run_dir / "scoring" / "borderline.json"
    if not mis_path.exists() and _run_has_score_inputs(run_dir):
        try:
            from pipeline.scoring.run_scoring import run_scoring  # noqa: PLC0415

            run_scoring(run_id, repo_root, runs_root=repo_root / "data" / "runs")
        except Exception as exc:  # noqa: BLE001 - preserve policy API failure path with context
            raise FileNotFoundError(
                f"missing scoring misalignment file and auto-scoring failed: {mis_path}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
    if not mis_path.exists():
        raise FileNotFoundError(f"missing scoring misalignment file: {mis_path}")
    misalignment = json.loads(mis_path.read_text(encoding="utf-8"))
    borderline = json.loads(bord_path.read_text(encoding="utf-8")) if bord_path.exists() else None
    policy_dir = _version_dir(repo_root, base_version)
    return PolicyIterationInputs(
        misalignment=misalignment,
        borderline=borderline,
        policy_markdown=load_policy_markdown(policy_dir),
        policy_graph_version=f"{DOMAIN}.{base_version}",
    )


def _coerce_content(value: Any, *, filename: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"proposed content for {filename} must be a string")
    return value


def _proposal_from_llm_json(raw: str) -> tuple[dict[str, str], list[str]]:
    """Parse and validate an LLM draft into proposed files + removals.

    Accepted JSON shape is intentionally simple and full-content based::

        {"files": [
          {"path": "GA.root.md", "change": "modified", "content": "..."},
          {"path": "GA.new.md", "change": "added", "content": "..."},
          {"path": "GA.old.md", "change": "removed"}
        ]}
    """
    try:
        doc = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"LLM response was not valid JSON: {exc}") from exc
    if not isinstance(doc, dict):
        raise ValueError("LLM response must be a JSON object")
    files = doc.get("files")
    if not isinstance(files, list):
        raise ValueError("LLM response must contain files: [...]")

    proposed: dict[str, str] = {}
    removed: list[str] = []
    for idx, item in enumerate(files):
        if not isinstance(item, dict):
            raise ValueError(f"files[{idx}] must be an object")
        filename = _validate_md_filename(str(item.get("path", "")))
        change = item.get("change") or item.get("status")
        if change not in {"modified", "added", "removed"}:
            raise ValueError(f"files[{idx}].change must be modified, added, or removed")
        if change == "removed":
            removed.append(filename)
            continue
        content = item.get("content", item.get("proposed_content"))
        proposed[filename] = _coerce_content(content, filename=filename)
    return proposed, removed


def _write_proposal_dir(
    *,
    repo_root: Path,
    proposal_id: str,
    metadata: dict[str, Any],
    prompt: dict[str, Any],
    raw_response: str,
    proposed_files: dict[str, str],
) -> dict[str, Any]:
    prop_dir = _proposal_dir(repo_root, proposal_id)
    if prop_dir.exists():
        raise FileExistsError(f"proposal already exists: {proposal_id}")
    proposed_dir = prop_dir / "proposed"
    proposed_dir.mkdir(parents=True, exist_ok=False)

    normalized = _normalize_proposal_meta(metadata)
    for filename, content in proposed_files.items():
        _atomic_write_text(proposed_dir / _validate_md_filename(filename), content)
    _atomic_write_json(prop_dir / "prompt.json", prompt)
    _atomic_write_text(prop_dir / "raw_response.txt", raw_response)
    _atomic_write_json(prop_dir / "proposal.json", normalized)
    return normalized


def _classify_changes_cold_start(
    proposed_files: dict[str, str],
) -> tuple[list[str], list[str], list[str]]:
    """Cold-start classification: no base dir, so every file is 'added'.

    Returns ``(changed=[], added=[...], removed=[])``.
    """
    added: list[str] = []
    for filename in sorted(proposed_files):
        added.append(_validate_md_filename(filename))
    return [], added, []


def _classify_changes(
    *,
    base_dir: Path,
    proposed_files: dict[str, str],
    files_removed: Iterable[str],
) -> tuple[list[str], list[str], list[str]]:
    changed: list[str] = []
    added: list[str] = []
    removed: list[str] = []
    for filename, content in sorted(proposed_files.items()):
        filename = _validate_md_filename(filename)
        base_file = base_dir / filename
        if base_file.exists():
            before = base_file.read_text(encoding="utf-8")
            if before != content:
                changed.append(filename)
        else:
            added.append(filename)
    for filename in sorted(files_removed):
        filename = _validate_md_filename(filename)
        if filename in proposed_files:
            raise ValueError(f"file cannot be both proposed and removed: {filename}")
        removed.append(filename)
    return changed, added, removed


def _is_retryable_llm_exception(exc: BaseException) -> bool:
    if isinstance(exc, (TimeoutError, FutureTimeoutError)):
        return True
    marker = f"{type(exc).__name__}: {exc}".lower()
    return "timeout" in marker or "timed out" in marker


def _call_chat_with_retries(
    chat_callable: ChatCallable,
    messages: list[dict[str, str]],
    *,
    model_id: str,
    reasoning_effort: str,
    timeout_s: float = DEFAULT_LLM_TIMEOUT_S,
    retries: int = DEFAULT_LLM_RETRIES,
    backoff_s: float = DEFAULT_LLM_BACKOFF_S,
    progress_callback: ProgressCallable | None = None,
) -> str:
    """Call the policy LLM with a per-attempt timeout and bounded retries."""
    attempts = max(1, int(retries) + 1)
    timeout_s = max(0.1, float(timeout_s))
    backoff_s = max(0.0, float(backoff_s))
    last_exc: BaseException | None = None

    for attempt in range(1, attempts + 1):
        if progress_callback:
            progress_callback(
                {
                    "status": "building",
                    "attempt": attempt,
                    "max_attempts": attempts,
                    "retry_count": attempt - 1,
                    "max_retries": attempts - 1,
                }
            )
        executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="policy-propose")
        future = executor.submit(
            chat_callable,
            messages,
            model_id=model_id,
            reasoning_effort=reasoning_effort,
        )
        retryable = False
        try:
            return future.result(timeout=timeout_s)
        except FutureTimeoutError as exc:
            future.cancel()
            last_exc = TimeoutError(f"LLM call timed out after {timeout_s:g}s")
            retryable = True
        except Exception as exc:  # noqa: BLE001 - transport-specific timeout classes vary
            last_exc = exc
            retryable = _is_retryable_llm_exception(exc)
        finally:
            executor.shutdown(wait=False, cancel_futures=True)

        if not retryable or attempt >= attempts:
            assert last_exc is not None
            raise last_exc

        if progress_callback:
            progress_callback(
                {
                    "status": "building_retry",
                    "attempt": attempt + 1,
                    "max_attempts": attempts,
                    "retry_count": attempt,
                    "max_retries": attempts - 1,
                    "reason": str(last_exc),
                }
            )
        time.sleep(backoff_s * (2 ** (attempt - 1)))

    assert last_exc is not None
    raise last_exc


def propose_diff(
    *,
    repo_root: Path | str,
    run_id: str,
    base_version: str = "v0.1",
    model_id: str | None = None,
    proposed_files: dict[str, str] | None = None,
    files_removed: Iterable[str] | None = None,
    chat_callable: ChatCallable | None = None,
    llm_timeout_s: float = DEFAULT_LLM_TIMEOUT_S,
    llm_retries: int = DEFAULT_LLM_RETRIES,
    llm_backoff_s: float = DEFAULT_LLM_BACKOFF_S,
    progress_callback: ProgressCallable | None = None,
) -> dict[str, Any]:
    """Create a policy proposal without modifying ``policy-graph``.

    Tests and internal callers may pass ``proposed_files`` directly to avoid an
    LLM call. Production callers omit it; the default model is GPT-5.5.
    Parse failures are persisted as ``status: parse_error`` with the raw text.
    """
    root = _repo_root(repo_root)
    base_version = _validate_version(base_version)
    base_dir = _version_dir(root, base_version)
    effective_model = model_id or DEFAULT_POLICY_MODEL
    if effective_model not in ALLOWED_POLICY_MODELS:
        raise ValueError(f"unsupported policy proposal model_id: {effective_model}")

    proposal_id = _new_proposal_id()
    created_at = datetime.now(timezone.utc).isoformat()

    if proposed_files is None:
        if chat_callable is None:
            if effective_model == DEFAULT_POLICY_MODEL:
                from pipeline.providers.openai_chat import policy_chat_callable
            elif effective_model == ANTHROPIC_POLICY_MODEL:
                from pipeline.providers.anthropic_chat import policy_chat_callable
            else:  # guarded by ALLOWED_POLICY_MODELS above
                raise ValueError(f"unsupported policy proposal model_id: {effective_model}")

            chat_callable = policy_chat_callable(effective_model)
        inputs = _load_run_inputs(root, run_id, base_version)
        user_payload = build_user_prompt(inputs)
        prompt = {
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are RUSH's policy diff writer. Return JSON only. "
                        "Draft minimal full-file markdown changes using this shape: "
                        "{\"files\":[{\"path\":\"name.md\",\"change\":\"modified|added|removed\","
                        "\"content\":\"full markdown for added/modified files\"}]}. "
                        "Never return unified diffs."
                    ),
                },
                {"role": "user", "content": json.dumps(user_payload, indent=2)},
            ],
            "user_payload": user_payload,
        }
        try:
            raw_response = _call_chat_with_retries(
                chat_callable,
                prompt["messages"],
                model_id=effective_model,
                reasoning_effort="high",
                timeout_s=llm_timeout_s,
                retries=llm_retries,
                backoff_s=llm_backoff_s,
                progress_callback=progress_callback,
            )
        except Exception as exc:  # noqa: BLE001 - persist provider/proxy failures for review
            error = _error_summary(exc)
            metadata = {
                "proposal_id": proposal_id,
                "kind": "propose_diff",
                "base_version": base_version,
                "model_id": effective_model,
                "run_id": run_id,
                "created_at": created_at,
                "status": "parse_error",
                "error": error,
                "error_type": "provider_error",
                "files_changed": [],
                "files_added": [],
                "files_removed": [],
                "raw_response": "",
            }
            return _write_proposal_dir(
                repo_root=root,
                proposal_id=proposal_id,
                metadata=metadata,
                prompt=prompt,
                raw_response="",
                proposed_files={},
            )
        try:
            proposed_files, removed = _proposal_from_llm_json(raw_response)
        except ValueError as exc:
            metadata = {
                "proposal_id": proposal_id,
                "kind": "propose_diff",
                "base_version": base_version,
                "model_id": effective_model,
                "run_id": run_id,
                "created_at": created_at,
                "status": "parse_error",
                "error": str(exc),
                "error_type": "parse_error",
                "files_changed": [],
                "files_added": [],
                "files_removed": [],
                "raw_response": raw_response,
                "raw_response_excerpt": raw_response[:1200],
            }
            return _write_proposal_dir(
                repo_root=root,
                proposal_id=proposal_id,
                metadata=metadata,
                prompt=prompt,
                raw_response=raw_response,
                proposed_files={},
            )
        files_removed = removed
    else:
        prompt = {
            "source": "direct_api",
            "run_id": run_id,
            "base_version": base_version,
            "model_id": effective_model,
            "files": sorted(proposed_files),
            "files_removed": sorted(files_removed or []),
        }
        proposed_files = {
            _validate_md_filename(filename): _coerce_content(content, filename=filename)
            for filename, content in proposed_files.items()
        }
        raw_response = json.dumps(
            {
                "files": [
                    {"path": filename, "change": "proposed", "content": content}
                    for filename, content in sorted(proposed_files.items())
                ],
                "files_removed": sorted(files_removed or []),
            },
            sort_keys=True,
        )

    removed_list = [_validate_md_filename(f) for f in (files_removed or [])]
    changed, added, removed = _classify_changes(
        base_dir=base_dir,
        proposed_files=proposed_files,
        files_removed=removed_list,
    )
    metadata = {
        "proposal_id": proposal_id,
        "kind": "propose_diff",
        "base_version": base_version,
        "model_id": effective_model,
        "run_id": run_id,
        "created_at": created_at,
        "status": "pending",
        "files_changed": changed,
        "files_added": added,
        "files_removed": removed,
    }
    return _write_proposal_dir(
        repo_root=root,
        proposal_id=proposal_id,
        metadata=metadata,
        prompt=prompt,
        raw_response=raw_response,
        proposed_files=proposed_files,
    )


_COLD_START_SYSTEM_PROMPT = (
    "You are seeding a policy graph from scratch. Return only JSON. "
    "Each markdown file MUST start with YAML frontmatter compatible with the "
    "existing schema (id, version, title, area, node_type, polarity, parent, "
    "status, edges). Produce a SKELETON graph: a root node plus 3-6 seed "
    "children that mix positive evidence, boundary cases, and provenance/negative "
    "signals. Include edges.json as one of the files if needed. "
    "Use this shape: {\"files\":[{\"path\":\"GA.root.md\",\"change\":\"added\",\"content\":\"...full markdown...\"}]}. "
    "Never return unified diffs. Never invent example image URLs."
)


_GROW_BATCH_SYSTEM_PROMPT = (
    "You are RUSH's policy diff writer. Return JSON only. "
    "Draft minimal full-file markdown changes using this shape: "
    "{\"files\":[{\"path\":\"name.md\",\"change\":\"modified|added|removed\","
    "\"content\":\"full markdown for added/modified files\"}]}. "
    "Never return unified diffs. You are growing an existing policy graph from a "
    "stratified batch of SME-labeled misclassifications (balanced positive vs. "
    "negative examples). Prefer small additive nodes / minor clarifications over "
    "large rewrites."
)


def seed_cold_start_proposal(
    *,
    repo_root: Path | str,
    task_description: str,
    model_id: str | None = None,
    domain: str = DOMAIN,
    chat_callable: ChatCallable | None = None,
    proposed_files: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Seed a brand-new policy graph from a free-form task description.

    Produces a ``cold_start`` proposal with no base_version. Tests / internal
    callers may pass ``proposed_files`` directly to skip the LLM call.
    Parse failures persist as ``status: parse_error`` with the raw text.
    """
    if not isinstance(task_description, str) or not task_description.strip():
        raise ValueError("task_description must be a non-empty string")
    if domain != DOMAIN:
        raise ValueError(f"unsupported domain: {domain!r}")

    root = _repo_root(repo_root)
    effective_model = model_id or DEFAULT_POLICY_MODEL
    if effective_model not in ALLOWED_POLICY_MODELS:
        raise ValueError(f"unsupported policy proposal model_id: {effective_model}")

    proposal_id = _new_proposal_id()
    created_at = datetime.now(timezone.utc).isoformat()
    task_description_trunc = task_description[:2000]

    if proposed_files is None:
        if chat_callable is None:
            if effective_model == DEFAULT_POLICY_MODEL:
                from pipeline.providers.openai_chat import policy_chat_callable
            elif effective_model == ANTHROPIC_POLICY_MODEL:
                from pipeline.providers.anthropic_chat import policy_chat_callable
            else:  # guarded above
                raise ValueError(
                    f"unsupported policy proposal model_id: {effective_model}"
                )
            chat_callable = policy_chat_callable(effective_model)

        user_payload = {
            "domain": domain,
            "task_description": task_description_trunc,
            "instructions": (
                "Return a skeleton policy graph for this classification task: "
                "a root node plus 3-6 seed nodes (mix positive evidence, boundary, "
                "provenance/negative). All markdown files must include YAML "
                "frontmatter (id, version, title, area, node_type, polarity, "
                "parent, status, edges)."
            ),
        }
        prompt = {
            "messages": [
                {"role": "system", "content": _COLD_START_SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(user_payload, indent=2)},
            ],
            "user_payload": user_payload,
        }
        try:
            raw_response = chat_callable(
                prompt["messages"], model_id=effective_model, reasoning_effort="high"
            )
        except Exception as exc:  # noqa: BLE001 - persist provider/proxy failures for review
            error = _error_summary(exc)
            metadata = {
                "proposal_id": proposal_id,
                "kind": "cold_start",
                "domain": domain,
                "base_version": None,
                "task_description": task_description_trunc,
                "model_id": effective_model,
                "created_at": created_at,
                "status": "parse_error",
                "files_changed": [],
                "files_added": [],
                "files_removed": [],
                "error": error,
                "raw_response": "",
            }
            return _write_proposal_dir(
                repo_root=root,
                proposal_id=proposal_id,
                metadata=metadata,
                prompt=prompt,
                raw_response="",
                proposed_files={},
            )
        try:
            proposed_files, removed = _proposal_from_llm_json(raw_response)
        except ValueError as exc:
            metadata = {
                "proposal_id": proposal_id,
                "kind": "cold_start",
                "domain": domain,
                "base_version": None,
                "task_description": task_description_trunc,
                "model_id": effective_model,
                "created_at": created_at,
                "status": "parse_error",
                "files_changed": [],
                "files_added": [],
                "files_removed": [],
                "error": str(exc),
                "raw_response": raw_response,
            }
            return _write_proposal_dir(
                repo_root=root,
                proposal_id=proposal_id,
                metadata=metadata,
                prompt=prompt,
                raw_response=raw_response,
                proposed_files={},
            )
        if removed:
            # Cold start has no base version to remove from; ignore silently.
            removed = []
    else:
        prompt = {
            "source": "direct_api",
            "kind": "cold_start",
            "domain": domain,
            "model_id": effective_model,
            "task_description": task_description_trunc,
            "files": sorted(proposed_files),
        }
        proposed_files = {
            _validate_md_filename(filename): _coerce_content(content, filename=filename)
            for filename, content in proposed_files.items()
        }
        raw_response = json.dumps(
            {
                "files": [
                    {"path": filename, "change": "added", "content": content}
                    for filename, content in sorted(proposed_files.items())
                ],
            },
            sort_keys=True,
        )

    if not proposed_files:
        raise ValueError("cold-start proposal produced no files")

    _, added, _ = _classify_changes_cold_start(proposed_files)
    metadata = {
        "proposal_id": proposal_id,
        "kind": "cold_start",
        "domain": domain,
        "base_version": None,
        "task_description": task_description_trunc,
        "model_id": effective_model,
        "created_at": created_at,
        "status": "pending",
        "files_changed": [],
        "files_added": added,
        "files_removed": [],
    }
    return _write_proposal_dir(
        repo_root=root,
        proposal_id=proposal_id,
        metadata=metadata,
        prompt=prompt,
        raw_response=raw_response,
        proposed_files=proposed_files,
    )


SME_TRUTH_POSITIVE_LABEL = "gen_ai"
SME_TRUTH_NEGATIVE_LABEL = "not_gen_ai"
DEFAULT_GROW_BATCH_SIZE = 20


def _stratified_batch_rows(
    records: list[dict[str, Any]],
    *,
    batch_index: int,
    batch_size: int,
) -> tuple[list[dict[str, Any]], int, int]:
    """Return (rows, n_positives, n_negatives) for a stratified 50/50 batch.

    Each class is sorted deterministically by image_id. If one class is
    exhausted, the remainder is filled from the other class.
    """
    positives = sorted(
        [r for r in records if r.get("sme_truth") == SME_TRUTH_POSITIVE_LABEL],
        key=lambda r: str(r.get("image_id", "")),
    )
    negatives = sorted(
        [r for r in records if r.get("sme_truth") == SME_TRUTH_NEGATIVE_LABEL],
        key=lambda r: str(r.get("image_id", "")),
    )

    half = batch_size // 2
    start = batch_index * half
    end = start + half
    pos_slice = list(positives[start:end])
    neg_slice = list(negatives[start:end])

    # Fallback: if one class came up short for this batch_index, fill the
    # remainder from the other class's leftover rows (beyond ``end``). Never
    # wrap or repeat rows.
    remaining = batch_size - len(pos_slice) - len(neg_slice)
    if remaining > 0:
        if len(neg_slice) < half:
            extra = positives[end : end + remaining]
            pos_slice.extend(extra)
            remaining -= len(extra)
        if remaining > 0 and len(pos_slice) < half:
            # positives slice was short; pull from negatives' leftover too
            extra = negatives[end : end + remaining]
            neg_slice.extend(extra)

    rows = pos_slice + neg_slice
    return rows, len(pos_slice), len(neg_slice)


def propose_growth_batch(
    *,
    repo_root: Path | str,
    run_id: str,
    base_version: str = "v0.1",
    batch_index: int = 0,
    batch_size: int = DEFAULT_GROW_BATCH_SIZE,
    model_id: str | None = None,
    chat_callable: ChatCallable | None = None,
    proposed_files: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Propose policy growth from a stratified 50/50 batch of misclassifications.

    Tests / internal callers may pass ``proposed_files`` directly to skip the
    LLM call. Parse failures persist as ``status: parse_error``.
    """
    if not isinstance(run_id, str) or not run_id:
        raise ValueError("run_id is required")
    if not isinstance(batch_index, int) or batch_index < 0:
        raise ValueError("batch_index must be a non-negative integer")
    if not isinstance(batch_size, int) or batch_size < 2:
        raise ValueError("batch_size must be an integer >= 2")
    base_version = _validate_version(base_version)

    root = _repo_root(repo_root)
    base_dir = _version_dir(root, base_version)
    effective_model = model_id or DEFAULT_POLICY_MODEL
    if effective_model not in ALLOWED_POLICY_MODELS:
        raise ValueError(f"unsupported policy proposal model_id: {effective_model}")

    run_dir = root / "data" / "runs" / run_id
    mis_path = run_dir / "scoring" / "misalignment.json"
    if not mis_path.exists() and _run_has_score_inputs(run_dir):
        try:
            from pipeline.scoring.run_scoring import run_scoring  # noqa: PLC0415

            run_scoring(run_id, root, runs_root=root / "data" / "runs")
        except Exception as exc:  # noqa: BLE001
            raise FileNotFoundError(
                f"missing scoring misalignment file and auto-scoring failed: "
                f"{mis_path}: {type(exc).__name__}: {exc}"
            ) from exc
    if not mis_path.exists():
        raise FileNotFoundError(f"missing scoring misalignment file: {mis_path}")

    misalignment = json.loads(mis_path.read_text(encoding="utf-8"))
    records = misalignment.get("records", []) or []
    batch_rows, n_positives, n_negatives = _stratified_batch_rows(
        records, batch_index=batch_index, batch_size=batch_size
    )
    batch_size_actual = n_positives + n_negatives

    bord_path = run_dir / "scoring" / "borderline.json"
    borderline = (
        json.loads(bord_path.read_text(encoding="utf-8"))
        if bord_path.exists()
        else None
    )
    inputs = PolicyIterationInputs(
        misalignment={"records": batch_rows},
        borderline=borderline,
        policy_markdown=load_policy_markdown(base_dir),
        policy_graph_version=f"{DOMAIN}.{base_version}",
    )

    proposal_id = _new_proposal_id()
    created_at = datetime.now(timezone.utc).isoformat()
    batch_meta = {
        "batch_size_requested": batch_size,
        "batch_size_actual": batch_size_actual,
        "n_positives": n_positives,
        "n_negatives": n_negatives,
        "sme_truth_positive_label": SME_TRUTH_POSITIVE_LABEL,
        "sme_truth_negative_label": SME_TRUTH_NEGATIVE_LABEL,
    }

    if proposed_files is None:
        if chat_callable is None:
            if effective_model == DEFAULT_POLICY_MODEL:
                from pipeline.providers.openai_chat import policy_chat_callable
            elif effective_model == ANTHROPIC_POLICY_MODEL:
                from pipeline.providers.anthropic_chat import policy_chat_callable
            else:
                raise ValueError(
                    f"unsupported policy proposal model_id: {effective_model}"
                )
            chat_callable = policy_chat_callable(effective_model)
        # Build the user payload via the iterator helper (uses the batch only),
        # then annotate it with batch_context so the LLM sees the stratification.
        user_payload = build_user_prompt(
            inputs, max_rows=max(batch_size, 1), severity=("high", "medium", "low")
        )
        user_payload["batch_context"] = {
            "batch_index": batch_index,
            "batch_size_requested": batch_size,
            "batch_size_actual": batch_size_actual,
            "n_positives": n_positives,
            "n_negatives": n_negatives,
            "base_version": base_version,
        }
        prompt = {
            "messages": [
                {"role": "system", "content": _GROW_BATCH_SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(user_payload, indent=2)},
            ],
            "user_payload": user_payload,
        }
        try:
            raw_response = chat_callable(
                prompt["messages"], model_id=effective_model, reasoning_effort="high"
            )
        except Exception as exc:  # noqa: BLE001 - persist provider/proxy failures for review
            error = _error_summary(exc)
            metadata = {
                "proposal_id": proposal_id,
                "kind": "grow_batch",
                "base_version": base_version,
                "batch_index": batch_index,
                "batch": batch_meta,
                "run_id": run_id,
                "model_id": effective_model,
                "created_at": created_at,
                "status": "parse_error",
                "files_changed": [],
                "files_added": [],
                "files_removed": [],
                "error": error,
                "raw_response": "",
            }
            return _write_proposal_dir(
                repo_root=root,
                proposal_id=proposal_id,
                metadata=metadata,
                prompt=prompt,
                raw_response="",
                proposed_files={},
            )
        try:
            proposed_files, removed = _proposal_from_llm_json(raw_response)
        except ValueError as exc:
            metadata = {
                "proposal_id": proposal_id,
                "kind": "grow_batch",
                "base_version": base_version,
                "batch_index": batch_index,
                "batch": batch_meta,
                "run_id": run_id,
                "model_id": effective_model,
                "created_at": created_at,
                "status": "parse_error",
                "files_changed": [],
                "files_added": [],
                "files_removed": [],
                "error": str(exc),
                "raw_response": raw_response,
            }
            return _write_proposal_dir(
                repo_root=root,
                proposal_id=proposal_id,
                metadata=metadata,
                prompt=prompt,
                raw_response=raw_response,
                proposed_files={},
            )
        files_removed = removed
    else:
        prompt = {
            "source": "direct_api",
            "kind": "grow_batch",
            "run_id": run_id,
            "base_version": base_version,
            "batch_index": batch_index,
            "batch": batch_meta,
            "model_id": effective_model,
            "files": sorted(proposed_files),
        }
        proposed_files = {
            _validate_md_filename(filename): _coerce_content(
                content, filename=filename
            )
            for filename, content in proposed_files.items()
        }
        raw_response = json.dumps(
            {
                "files": [
                    {"path": filename, "change": "proposed", "content": content}
                    for filename, content in sorted(proposed_files.items())
                ],
            },
            sort_keys=True,
        )
        files_removed = []

    removed_list = [_validate_md_filename(f) for f in (files_removed or [])]
    changed, added, removed = _classify_changes(
        base_dir=base_dir,
        proposed_files=proposed_files,
        files_removed=removed_list,
    )
    metadata = {
        "proposal_id": proposal_id,
        "kind": "grow_batch",
        "base_version": base_version,
        "batch_index": batch_index,
        "batch": batch_meta,
        "run_id": run_id,
        "model_id": effective_model,
        "created_at": created_at,
        "status": "pending",
        "files_changed": changed,
        "files_added": added,
        "files_removed": removed,
    }
    return _write_proposal_dir(
        repo_root=root,
        proposal_id=proposal_id,
        metadata=metadata,
        prompt=prompt,
        raw_response=raw_response,
        proposed_files=proposed_files,
    )


def list_policy_versions(*, repo_root: Path | str, domain: str = DOMAIN) -> dict[str, Any]:
    """Return available policy versions for the web API."""
    domain_dir = _policy_domain_dir(repo_root, domain)
    versions: list[dict[str, Any]] = []
    for path in sorted(domain_dir.iterdir() if domain_dir.exists() else []):
        if not path.is_dir() or not _VERSION_RE.match(path.name):
            continue
        file_count = len([p for p in path.glob("*.md") if p.is_file()])
        versions.append(
            {
                "version": path.name,
                "files": file_count,
                "complete": file_count > 0,
                "path": str(path.relative_to(_repo_root(repo_root))),
            }
        )
    versions.sort(key=lambda v: _version_key(v["version"]))
    complete_versions = [version for version in versions if version["complete"]]
    return {
        "domain": domain,
        "versions": versions,
        "current": complete_versions[-1]["version"] if complete_versions else None,
    }


def _version_key(version: str) -> tuple[int, int]:
    match = _VERSION_RE.match(version)
    if not match:
        raise ValueError(f"invalid policy version: {version!r}")
    return int(match.group(1)), int(match.group(2))


def _next_version(repo_root: Path | str, domain: str = DOMAIN) -> str:
    domain_dir = _policy_domain_dir(repo_root, domain)
    if not domain_dir.is_dir():
        return "v0.1"
    keys = [_version_key(p.name) for p in domain_dir.iterdir() if p.is_dir() and _VERSION_RE.match(p.name)]
    if not keys:
        return "v0.1"
    major, minor = max(keys)
    return f"v{major}.{minor + 1}"


def _find_proposal_json(repo_root: Path | str, proposal_id: str) -> Path:
    active = _proposal_dir(repo_root, proposal_id) / "proposal.json"
    if active.exists():
        return active
    archived = _archive_dir(repo_root, proposal_id) / "proposal.json"
    if archived.exists():
        return archived
    raise FileNotFoundError(f"unknown proposal_id: {proposal_id}")


def list_proposals(
    *,
    repo_root: Path | str,
    include_errors: bool = False,
) -> dict[str, Any]:
    root = _proposal_root(repo_root)
    proposals: list[dict[str, Any]] = []
    if root.exists():
        for path in sorted(root.glob("*/proposal.json")):
            if path.parent.name == "_archive":
                continue
            proposals.append(_read_json(path))
        archive = root / "_archive"
        if archive.exists():
            for path in sorted(archive.glob("*/proposal.json")):
                proposals.append(_read_json(path))
    status_priority = {"pending": 3, "accepted": 2, "rejected": 1, "parse_error": 0}
    proposals.sort(
        key=lambda p: (
            p.get("created_at", ""),
            status_priority.get(str(p.get("status", "")), -1),
        ),
        reverse=True,
    )
    hidden_error_count = sum(1 for p in proposals if p.get("status") == "parse_error")
    if not include_errors:
        proposals = [p for p in proposals if p.get("status") != "parse_error"]
    return {
        "proposals": proposals,
        "hidden_error_count": 0 if include_errors else hidden_error_count,
        "include_errors": include_errors,
    }


def _diff_for_file(*, filename: str, before: str, after: str) -> str:
    return "".join(
        difflib.unified_diff(
            before.splitlines(keepends=True),
            after.splitlines(keepends=True),
            fromfile=f"a/{filename}",
            tofile=f"b/{filename}",
        )
    )


def get_proposal(*, repo_root: Path | str, proposal_id: str) -> dict[str, Any]:
    proposal_json = _find_proposal_json(repo_root, proposal_id)
    prop_dir = proposal_json.parent
    meta = _read_json(proposal_json)
    base_version = meta.get("base_version")
    base_dir = _version_dir(repo_root, base_version) if base_version else None
    diffs: list[dict[str, Any]] = []

    for filename in meta.get("files_changed", []):
        filename = _validate_md_filename(filename)
        if base_dir is None:
            # cold-start has no base; should never appear in files_changed.
            continue
        before = (base_dir / filename).read_text(encoding="utf-8")
        after = (prop_dir / "proposed" / filename).read_text(encoding="utf-8")
        diffs.append(
            {
                "path": filename,
                "change": "modified",
                "unified_diff": _diff_for_file(filename=filename, before=before, after=after),
                "before": before,
                "after": after,
            }
        )
    for filename in meta.get("files_added", []):
        filename = _validate_md_filename(filename)
        after = (prop_dir / "proposed" / filename).read_text(encoding="utf-8")
        diffs.append(
            {
                "path": filename,
                "change": "added",
                "unified_diff": _diff_for_file(filename=filename, before="", after=after),
                "before": "",
                "after": after,
            }
        )
    for filename in meta.get("files_removed", []):
        filename = _validate_md_filename(filename)
        if base_dir is None:
            continue
        before = (base_dir / filename).read_text(encoding="utf-8")
        diffs.append(
            {
                "path": filename,
                "change": "removed",
                "unified_diff": _diff_for_file(filename=filename, before=before, after=""),
                "before": before,
                "after": "",
            }
        )
    out = dict(meta)
    out["diffs"] = diffs
    return out


def accept_proposal(*, repo_root: Path | str, proposal_id: str) -> dict[str, Any]:
    """Accept a pending proposal and create the next policy version."""
    root = _repo_root(repo_root)
    prop_dir = _proposal_dir(root, proposal_id)
    proposal_json = prop_dir / "proposal.json"
    if not proposal_json.exists():
        raise FileNotFoundError(f"unknown active proposal_id: {proposal_id}")
    meta = _read_json(proposal_json)
    if meta.get("status") != "pending":
        raise ValueError(f"proposal is not pending: {meta.get('status')}")

    base_version = meta.get("base_version")
    new_version = _next_version(root)
    new_dir = _policy_domain_dir(root) / new_version
    if new_dir.exists():
        raise FileExistsError(f"policy version already exists: {new_version}")

    if base_version is None:
        # Cold-start path: there is no base version to copy from. Start with
        # an empty version dir and overlay only the proposed (added) files.
        new_dir.mkdir(parents=True, exist_ok=False)
        try:
            for filename in meta.get("files_added", []):
                filename = _validate_md_filename(filename)
                shutil.copyfile(prop_dir / "proposed" / filename, new_dir / filename)
            # files_changed and files_removed are not meaningful for cold start.
        except Exception:
            shutil.rmtree(new_dir, ignore_errors=True)
            raise
    else:
        base_dir = _version_dir(root, base_version)
        shutil.copytree(base_dir, new_dir)
        try:
            for filename in meta.get("files_changed", []) + meta.get("files_added", []):
                filename = _validate_md_filename(filename)
                shutil.copyfile(prop_dir / "proposed" / filename, new_dir / filename)
            for filename in meta.get("files_removed", []):
                filename = _validate_md_filename(filename)
                target = new_dir / filename
                if target.exists():
                    target.unlink()
        except Exception:
            shutil.rmtree(new_dir, ignore_errors=True)
            raise

    meta["status"] = "accepted"
    meta["accepted_into_version"] = new_version
    _atomic_write_json(proposal_json, meta)
    return {"new_version": new_version, "path": str(new_dir.relative_to(root))}


def reject_proposal(*, repo_root: Path | str, proposal_id: str) -> dict[str, Any]:
    """Reject and archive an active proposal."""
    root = _repo_root(repo_root)
    prop_dir = _proposal_dir(root, proposal_id)
    if not prop_dir.exists():
        raise FileNotFoundError(f"unknown active proposal_id: {proposal_id}")
    proposal_json = prop_dir / "proposal.json"
    meta = _read_json(proposal_json)
    meta["status"] = "rejected"
    _atomic_write_json(proposal_json, meta)

    archive = _archive_dir(root, proposal_id)
    archive.parent.mkdir(parents=True, exist_ok=True)
    if archive.exists():
        raise FileExistsError(f"archived proposal already exists: {proposal_id}")
    shutil.move(str(prop_dir), str(archive))
    return {"proposal_id": proposal_id, "status": "rejected", "path": str(archive.relative_to(root))}


__all__ = [
    "ALLOWED_POLICY_MODELS",
    "DEFAULT_POLICY_MODEL",
    "ANTHROPIC_POLICY_MODEL",
    "SME_TRUTH_NEGATIVE_LABEL",
    "SME_TRUTH_POSITIVE_LABEL",
    "DEFAULT_GROW_BATCH_SIZE",
    "accept_proposal",
    "get_proposal",
    "list_policy_versions",
    "list_proposals",
    "propose_diff",
    "propose_growth_batch",
    "reject_proposal",
    "seed_cold_start_proposal",
]
