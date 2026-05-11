"""Policy proposal storage, diffs, and accept/reject lifecycle.

This module owns the server-side policy proposal workflow for the web API:
proposals are staged under ``data/policy_proposals/<proposal_id>/`` and only
``accept_proposal`` creates a new ``policy-graph/Generative_AI/vX.Y`` version.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import difflib
import json
from pathlib import Path, PurePosixPath
import re
import shutil
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


def _load_run_inputs(repo_root: Path, run_id: str, base_version: str) -> PolicyIterationInputs:
    run_dir = repo_root / "data" / "runs" / run_id
    mis_path = run_dir / "scoring" / "misalignment.json"
    bord_path = run_dir / "scoring" / "borderline.json"
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

    for filename, content in proposed_files.items():
        _atomic_write_text(proposed_dir / _validate_md_filename(filename), content)
    _atomic_write_json(prop_dir / "prompt.json", prompt)
    _atomic_write_text(prop_dir / "raw_response.txt", raw_response)
    _atomic_write_json(prop_dir / "proposal.json", metadata)
    return metadata


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


def propose_diff(
    *,
    repo_root: Path | str,
    run_id: str,
    base_version: str = "v0.1",
    model_id: str | None = None,
    proposed_files: dict[str, str] | None = None,
    files_removed: Iterable[str] | None = None,
    chat_callable: ChatCallable | None = None,
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
        raw_response = chat_callable(
            prompt["messages"], model_id=effective_model, reasoning_effort="high"
        )
        try:
            proposed_files, removed = _proposal_from_llm_json(raw_response)
        except ValueError:
            metadata = {
                "proposal_id": proposal_id,
                "base_version": base_version,
                "model_id": effective_model,
                "run_id": run_id,
                "created_at": created_at,
                "status": "parse_error",
                "files_changed": [],
                "files_added": [],
                "files_removed": [],
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


def list_policy_versions(*, repo_root: Path | str, domain: str = DOMAIN) -> dict[str, Any]:
    """Return available policy versions for the web API."""
    domain_dir = _policy_domain_dir(repo_root, domain)
    versions: list[dict[str, Any]] = []
    for path in sorted(domain_dir.iterdir() if domain_dir.exists() else []):
        if not path.is_dir() or not _VERSION_RE.match(path.name):
            continue
        versions.append(
            {
                "version": path.name,
                "files": len([p for p in path.glob("*.md") if p.is_file()]),
                "path": str(path.relative_to(_repo_root(repo_root))),
            }
        )
    versions.sort(key=lambda v: _version_key(v["version"]))
    return {
        "domain": domain,
        "versions": versions,
        "current": versions[-1]["version"] if versions else None,
    }


def _version_key(version: str) -> tuple[int, int]:
    match = _VERSION_RE.match(version)
    if not match:
        raise ValueError(f"invalid policy version: {version!r}")
    return int(match.group(1)), int(match.group(2))


def _next_version(repo_root: Path | str, domain: str = DOMAIN) -> str:
    domain_dir = _policy_domain_dir(repo_root, domain)
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


def list_proposals(*, repo_root: Path | str) -> dict[str, Any]:
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
    proposals.sort(key=lambda p: p.get("created_at", ""), reverse=True)
    return {"proposals": proposals}


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
    base_dir = _version_dir(repo_root, meta["base_version"])
    diffs: list[dict[str, Any]] = []

    for filename in meta.get("files_changed", []):
        filename = _validate_md_filename(filename)
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

    base_dir = _version_dir(root, meta["base_version"])
    new_version = _next_version(root)
    new_dir = _policy_domain_dir(root) / new_version
    if new_dir.exists():
        raise FileExistsError(f"policy version already exists: {new_version}")

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
    "accept_proposal",
    "get_proposal",
    "list_policy_versions",
    "list_proposals",
    "propose_diff",
    "reject_proposal",
]
