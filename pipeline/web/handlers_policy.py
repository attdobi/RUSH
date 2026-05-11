"""HTTP handler helpers for policy versions and proposals.

Each handler returns ``(status_code, body_dict)`` so the web server can stay a
thin routing layer.
"""
from __future__ import annotations

import json
from pathlib import Path
import re
import subprocess
import sys
from typing import Any

from pipeline.policy_diff import (
    DEFAULT_POLICY_MODEL,
    accept_proposal,
    get_proposal,
    list_policy_versions,
    list_proposals,
    propose_diff,
    reject_proposal,
)


def _root(repo_root: Path | str) -> Path:
    return Path(repo_root).resolve()


def _error(status: int, exc: Exception) -> tuple[int, dict[str, Any]]:
    return status, {"error": str(exc), "error_type": type(exc).__name__}


def _bad_request(exc: Exception) -> tuple[int, dict[str, Any]]:
    return _error(400, exc)


_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)


def _parse_frontmatter(text: str) -> tuple[dict[str, str], str]:
    """Extract a small YAML-frontmatter scalar map using stdlib only."""
    match = _FRONTMATTER_RE.match(text)
    if not match:
        return {}, text
    meta: dict[str, str] = {}
    for line in match.group(1).splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if line.startswith((" ", "\t")) or ":" not in line:
            continue
        key, _, value = line.partition(":")
        key = key.strip()
        value = value.strip()
        if not key:
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        meta[key] = value
    return meta, text[match.end() :]


def _nullish(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.strip()
    if not normalized or normalized.lower() in {"null", "none", "~"}:
        return None
    return normalized


def _heading_title(body: str, fallback: str) -> str:
    for line in body.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            return stripped.lstrip("#").strip() or fallback
    return fallback


def _policy_version_names(repo_root: Path | str) -> tuple[list[str], str | None]:
    payload = list_policy_versions(repo_root=repo_root)
    versions = [str(item["version"]) for item in payload.get("versions", [])]
    current = payload.get("current")
    return versions, str(current) if current else None


def _normalize_edge(edge: Any) -> dict[str, Any] | None:
    if not isinstance(edge, dict):
        return None
    source = edge.get("source") or edge.get("source_node_id")
    target = edge.get("target") or edge.get("target_node_id") or edge.get("to")
    if not source or not target:
        return None
    normalized = dict(edge)
    normalized["source"] = str(source)
    normalized["target"] = str(target)
    normalized["edge_type"] = str(
        edge.get("edge_type") or edge.get("type") or "related_to"
    )
    return normalized


def handle_policy_graph(
    repo_root: Path | str,
    version: str | None,
) -> tuple[int, dict[str, Any]]:
    """Return policy graph nodes and edges for the browser graph view."""
    try:
        versions, current = _policy_version_names(repo_root)
        if not versions or not current:
            return 404, {"error": "no policy versions found"}
        selected = (version or current).strip() or current
        if selected not in versions:
            return 404, {"error": f"unknown policy version: {selected}"}

        root = _root(repo_root)
        source = root / "policy-graph" / "Generative_AI" / selected
        nodes: list[dict[str, Any]] = []
        for path in sorted(
            source.glob("*.md"), key=lambda p: (p.name != "GA.root.md", p.name)
        ):
            meta, body = _parse_frontmatter(path.read_text(encoding="utf-8"))
            node_id = meta.get("id") or path.stem
            title = meta.get("title") or _heading_title(body, node_id)
            nodes.append(
                {
                    "id": node_id,
                    "node_type": meta.get("node_type") or "unknown",
                    "polarity": meta.get("polarity") or "mixed",
                    "title": title,
                    "parent": _nullish(meta.get("parent")),
                    "status": meta.get("status") or "unknown",
                }
            )

        edges_path = source / "edges.json"
        raw_edges = (
            json.loads(edges_path.read_text(encoding="utf-8"))
            if edges_path.exists()
            else []
        )
        if not isinstance(raw_edges, list):
            raw_edges = []
        edges = [edge for edge in (_normalize_edge(item) for item in raw_edges) if edge]
        return 200, {
            "version": selected,
            "title": f"Cold-start GenAI policy {selected}",
            "nodes": nodes,
            "edges": edges,
            "available_versions": versions,
        }
    except Exception as exc:  # noqa: BLE001 - surface to local web UI
        return _error(500, exc)


def handle_policy_versions(repo_root: Path | str) -> tuple[int, dict[str, Any]]:
    try:
        return 200, list_policy_versions(repo_root=repo_root)
    except Exception as exc:  # noqa: BLE001 - surface to local web UI
        return _error(500, exc)


def handle_propose_diff(
    repo_root: Path | str,
    body: dict[str, Any] | None,
) -> tuple[int, dict[str, Any]]:
    body = body or {}
    run_id = body.get("run_id")
    if not isinstance(run_id, str) or not run_id:
        return 400, {"error": "run_id is required"}
    base_version = body.get("base_version", "v0.1")
    model_id = body.get("model_id") or DEFAULT_POLICY_MODEL

    proposed = body.get("proposed_files")
    if proposed is not None and not isinstance(proposed, dict):
        return 400, {"error": "proposed_files must be an object when provided"}
    files_removed = body.get("files_removed") or []
    if not isinstance(files_removed, list):
        return 400, {"error": "files_removed must be a list when provided"}

    try:
        meta = propose_diff(
            repo_root=repo_root,
            run_id=run_id,
            base_version=base_version,
            model_id=model_id,
            proposed_files=proposed,
            files_removed=files_removed,
        )
        return (200 if meta.get("status") != "parse_error" else 422), meta
    except ValueError as exc:
        return _bad_request(exc)
    except FileNotFoundError as exc:
        return _error(404, exc)
    except Exception as exc:  # noqa: BLE001
        return _error(500, exc)


def handle_list_proposals(repo_root: Path | str) -> tuple[int, dict[str, Any]]:
    try:
        return 200, list_proposals(repo_root=repo_root)
    except Exception as exc:  # noqa: BLE001
        return _error(500, exc)


def handle_get_proposal(
    repo_root: Path | str,
    proposal_id: str,
) -> tuple[int, dict[str, Any]]:
    try:
        return 200, get_proposal(repo_root=repo_root, proposal_id=proposal_id)
    except FileNotFoundError as exc:
        return _error(404, exc)
    except ValueError as exc:
        return _bad_request(exc)
    except Exception as exc:  # noqa: BLE001
        return _error(500, exc)


def handle_accept_proposal(
    repo_root: Path | str,
    proposal_id: str,
) -> tuple[int, dict[str, Any]]:
    try:
        return 200, accept_proposal(repo_root=repo_root, proposal_id=proposal_id)
    except FileNotFoundError as exc:
        return _error(404, exc)
    except ValueError as exc:
        return _bad_request(exc)
    except Exception as exc:  # noqa: BLE001
        return _error(500, exc)


def handle_reject_proposal(
    repo_root: Path | str,
    proposal_id: str,
) -> tuple[int, dict[str, Any]]:
    try:
        return 200, reject_proposal(repo_root=repo_root, proposal_id=proposal_id)
    except FileNotFoundError as exc:
        return _error(404, exc)
    except ValueError as exc:
        return _bad_request(exc)
    except Exception as exc:  # noqa: BLE001
        return _error(500, exc)


def handle_build_pdf(
    repo_root: Path | str,
    body: dict[str, Any] | None,
) -> tuple[int, dict[str, Any]]:
    """Build a deterministic policy PDF via subprocess, never editing MD."""
    body = body or {}
    version = body.get("version", "v0.1")
    if not isinstance(version, str) or "/" in version or ".." in version:
        return 400, {"error": "invalid version"}

    root = _root(repo_root)
    source = root / "policy-graph" / "Generative_AI" / version
    if not source.is_dir():
        return 404, {"error": f"unknown policy version: {version}"}
    output = source / "policy.pdf"
    script = root / "scripts" / "build_policy_pdf.py"
    argv = [
        sys.executable,
        str(script),
        "--source",
        str(source),
        "--output",
        str(output),
        "--policy-graph-version",
        f"Generative_AI.{version}",
        "--json",
    ]
    try:
        proc = subprocess.run(  # noqa: S603 - explicit argv, shell=False by contract
            argv,
            cwd=str(root),
            shell=False,
            capture_output=True,
            text=True,
            check=False,
        )
    except Exception as exc:  # noqa: BLE001
        return _error(500, exc)
    if proc.returncode != 0:
        return 500, {"error": "build_policy_pdf.py failed", "stderr": proc.stderr}
    try:
        payload = json.loads(proc.stdout or "{}")
    except json.JSONDecodeError:
        payload = {"stdout": proc.stdout}
    payload.setdefault("output_path", str(output.relative_to(root)))
    return 200, payload


__all__ = [
    "handle_accept_proposal",
    "handle_build_pdf",
    "handle_get_proposal",
    "handle_list_proposals",
    "handle_policy_graph",
    "handle_policy_versions",
    "handle_propose_diff",
    "handle_reject_proposal",
]
