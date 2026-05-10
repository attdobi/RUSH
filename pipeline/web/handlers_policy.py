"""HTTP handler helpers for policy versions and proposals.

Each handler returns ``(status_code, body_dict)`` so the web server can stay a
thin routing layer.
"""
from __future__ import annotations

import json
from pathlib import Path
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
    "handle_policy_versions",
    "handle_propose_diff",
    "handle_reject_proposal",
]
