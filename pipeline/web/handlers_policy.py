"""HTTP handler helpers for policy versions and proposals.

Each handler returns ``(status_code, body_dict)`` so the web server can stay a
thin routing layer.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import re
import subprocess
import sys
import threading
import uuid
from typing import Any

from pipeline.policy_diff import (
    DEFAULT_POLICY_MODEL,
    DOMAIN,
    accept_proposal,
    get_proposal,
    list_policy_versions,
    list_proposals,
    propose_diff,
    propose_growth_batch,
    reject_proposal,
    seed_cold_start_proposal,
)
from pipeline.web.demo_area import (
    DEFAULT_POLICY_AREA,
    area_from_policy_version,
    normalize_policy_area,
    policy_version_matches_area,
)

_VERSION_RE = re.compile(r"^v\d+\.\d+$")


def _root(repo_root: Path | str) -> Path:
    return Path(repo_root).resolve()


def _domain_for_run(
    repo_root: Path | str,
    run_id: str,
    body: dict[str, Any] | None = None,
) -> str:
    """Resolve the policy domain/area for a proposal request.

    Prefers an explicit ``area``/``demo`` in the request body, otherwise derives
    it from the run manifest's ``policy_graph_version`` prefix. Falls back to the
    GenAI baseline so historical bare-version runs keep working.
    """
    body = body or {}
    explicit_area = body.get("area")
    explicit_demo = body.get("demo")
    if explicit_area or explicit_demo:
        try:
            return normalize_policy_area(explicit_area, demo=explicit_demo)
        except ValueError:
            pass
    try:
        manifest_path = _root(repo_root) / "data" / "runs" / str(run_id) / "run_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        return area_from_policy_version(manifest.get("policy_graph_version"))
    except Exception:  # noqa: BLE001 - best-effort; default to GenAI
        return DOMAIN


def _proposal_matches_domain(proposal: dict[str, Any], domain: str) -> bool:
    explicit = proposal.get("domain") or proposal.get("area")
    if explicit:
        try:
            return normalize_policy_area(str(explicit)) == domain
        except ValueError:
            return False
    base_version = proposal.get("base_version")
    if policy_version_matches_area(base_version, domain):
        return True
    proposal_id = str(proposal.get("proposal_id") or "")
    if proposal_id.startswith(f"{domain}."):
        return True
    # Historical proposal metadata predated the explicit domain field and used
    # bare versions only for GenAI.
    return domain == DEFAULT_POLICY_AREA and (
        base_version is None or str(base_version).startswith("v")
    )


def _error(status: int, exc: Exception) -> tuple[int, dict[str, Any]]:
    return status, {"error": str(exc), "error_type": type(exc).__name__}


def _bad_request(exc: Exception) -> tuple[int, dict[str, Any]]:
    return _error(400, exc)


_PROPOSAL_JOB_LOCK = threading.Lock()
_PROPOSAL_JOBS: dict[str, dict[str, Any]] = {}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _new_job_id() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    return f"{stamp}-{uuid.uuid4().hex[:8]}"


def _public_proposal_job(job: dict[str, Any]) -> dict[str, Any]:
    payload = {
        "job_id": job["job_id"],
        "status": job["status"],
        "status_url": job["status_url"],
        "run_id": job.get("run_id"),
        "base_version": job.get("base_version"),
        "model_id": job.get("model_id"),
        "created_at": job.get("created_at"),
        "updated_at": job.get("updated_at"),
        "attempt": job.get("attempt", 0),
        "max_attempts": job.get("max_attempts", 1),
        "retry_count": job.get("retry_count", 0),
        "max_retries": job.get("max_retries", 0),
        "message": job.get("message", ""),
    }
    if job.get("proposal_id"):
        payload["proposal_id"] = job["proposal_id"]
    if job.get("result") is not None:
        payload["result"] = job["result"]
    if job.get("error"):
        payload["error"] = job["error"]
        payload["error_type"] = job.get("error_type") or "error"
    return payload


def _proposal_job_path(repo_root: Path | str, job: dict[str, Any]) -> Path | None:
    run_id = str(job.get("run_id") or "")
    if not re.match(r"^[A-Za-z0-9_.-]+$", run_id):
        return None
    return _root(repo_root) / "data" / "runs" / run_id / "proposals" / "jobs" / f"{job['job_id']}.json"


def _persist_proposal_job(repo_root: Path | str, job: dict[str, Any]) -> None:
    path = _proposal_job_path(repo_root, job)
    if path is None:
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(json.dumps(_public_proposal_job(job), indent=2) + "\n", encoding="utf-8")
        tmp.replace(path)
    except Exception:
        # Persistence is best-effort; in-memory status remains authoritative.
        return


def _update_proposal_job(repo_root: Path | str, job_id: str, **updates: Any) -> dict[str, Any]:
    with _PROPOSAL_JOB_LOCK:
        job = _PROPOSAL_JOBS[job_id]
        job.update(updates)
        job["updated_at"] = _utc_now()
        snapshot = dict(job)
    _persist_proposal_job(repo_root, snapshot)
    return _public_proposal_job(snapshot)


def _start_propose_diff_job(
    repo_root: Path | str,
    *,
    run_id: str,
    base_version: str,
    model_id: str,
    proposed_files: dict[str, Any] | None,
    files_removed: list[Any],
    domain: str = DOMAIN,
) -> dict[str, Any]:
    job_id = _new_job_id()
    now = _utc_now()
    job = {
        "job_id": job_id,
        "status": "queued",
        "status_url": f"/api/policy/propose-diff/jobs/{job_id}",
        "run_id": run_id,
        "base_version": base_version,
        "model_id": model_id,
        "created_at": now,
        "updated_at": now,
        "attempt": 0,
        "max_attempts": 1,
        "retry_count": 0,
        "max_retries": 0,
        "message": "Queued proposal generation.",
    }
    with _PROPOSAL_JOB_LOCK:
        _PROPOSAL_JOBS[job_id] = job
    _persist_proposal_job(repo_root, job)

    def progress(event: dict[str, Any]) -> None:
        retry_count = int(event.get("retry_count") or 0)
        max_retries = int(event.get("max_retries") or 0)
        state = "building" if event.get("status") != "building_retry" else "building_retry"
        message = "Generating proposal… (still working, this can take a minute)"
        if retry_count:
            message = f"Generating proposal… retry {retry_count}/{max_retries} after timeout."
        _update_proposal_job(
            repo_root,
            job_id,
            status=state,
            attempt=event.get("attempt", 0),
            max_attempts=event.get("max_attempts", 1),
            retry_count=retry_count,
            max_retries=max_retries,
            message=message,
        )

    def worker() -> None:
        _update_proposal_job(
            repo_root,
            job_id,
            status="building",
            message="Generating proposal… (still working, this can take a minute)",
        )
        try:
            meta = propose_diff(
                repo_root=repo_root,
                run_id=run_id,
                base_version=base_version,
                domain=domain,
                model_id=model_id,
                proposed_files=proposed_files,
                files_removed=files_removed,
                progress_callback=progress,
            )
            final_status = "parse_error" if meta.get("status") == "parse_error" else "success"
            message = (
                "Proposal generated but the model returned malformed JSON."
                if final_status == "parse_error"
                else f"Created proposal {meta.get('proposal_id', 'unknown')}."
            )
            _update_proposal_job(
                repo_root,
                job_id,
                status=final_status,
                result=meta,
                proposal_id=meta.get("proposal_id"),
                message=message,
                error=meta.get("error") if final_status == "parse_error" else None,
                error_type=meta.get("error_type") if final_status == "parse_error" else None,
            )
        except Exception as exc:  # noqa: BLE001 - surface to local web UI
            _update_proposal_job(
                repo_root,
                job_id,
                status="failed",
                error=str(exc),
                error_type=type(exc).__name__,
                message=f"Proposal generation failed: {exc}",
            )

    thread = threading.Thread(target=worker, name=f"policy-propose-{job_id}", daemon=True)
    thread.start()
    return _public_proposal_job(job)


def handle_get_propose_diff_job(
    repo_root: Path | str,
    job_id: str,
) -> tuple[int, dict[str, Any]]:
    with _PROPOSAL_JOB_LOCK:
        job = _PROPOSAL_JOBS.get(job_id)
        snapshot = dict(job) if job else None
    if snapshot is None:
        return 404, {"error": f"unknown proposal job: {job_id}"}
    _persist_proposal_job(repo_root, snapshot)
    return 200, _public_proposal_job(snapshot)


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


def _policy_version_names(repo_root: Path | str, domain: str) -> tuple[list[str], str | None]:
    payload = list_policy_versions(repo_root=repo_root, domain=domain)
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


def _parse_inline_mapping(text: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for part in text.split(","):
        key, sep, value = part.partition(":")
        if not sep:
            continue
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        if key:
            out[key] = value
    return out


def _frontmatter_edges(text: str, source_node_id: str) -> list[dict[str, Any]]:
    match = _FRONTMATTER_RE.match(text)
    if not match:
        return []
    edges: list[dict[str, Any]] = []
    in_edges = False
    for line in match.group(1).splitlines():
        if line.startswith((" ", "\t")):
            stripped = line.strip()
            if in_edges and stripped.startswith("- {") and stripped.endswith("}"):
                mapping = _parse_inline_mapping(stripped[3:-1])
                target = mapping.get("to") or mapping.get("target")
                edge_type = mapping.get("type") or mapping.get("edge_type")
                if target and edge_type and edge_type != "subtype_of":
                    raw_edge = dict(mapping)
                    raw_edge["source"] = source_node_id
                    raw_edge["target"] = target
                    raw_edge["edge_type"] = edge_type
                    edges.append(raw_edge)
            continue
        in_edges = line.strip() == "edges:"
    return edges


def _dedupe_edges(edges: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for edge in edges:
        key = (
            str(edge.get("source") or ""),
            str(edge.get("target") or ""),
            str(edge.get("edge_type") or ""),
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(edge)
    return out


def handle_policy_graph(
    repo_root: Path | str,
    version: str | None,
    area: str | None = None,
) -> tuple[int, dict[str, Any]]:
    """Return policy graph nodes and edges for the browser graph view."""
    try:
        domain = normalize_policy_area(area)
        versions, current = _policy_version_names(repo_root, domain)
        if not versions:
            return 404, {"error": "no policy versions found"}
        selected = (version or current or "").strip()
        if not selected:
            return 404, {"error": "no complete policy versions found"}
        if selected not in versions:
            return 404, {"error": f"unknown policy version: {selected}"}

        root = _root(repo_root)
        source = root / "policy-graph" / domain / selected
        nodes: list[dict[str, Any]] = []
        md_items: list[tuple[Path, dict[str, str], str, str]] = []
        for path in source.glob("*.md") if source.is_dir() else []:
            if not path.is_file():
                continue
            text = path.read_text(encoding="utf-8")
            meta, body = _parse_frontmatter(text)
            node_id = meta.get("id") or path.stem
            md_items.append((path, meta, body, node_id))
        md_items.sort(key=lambda item: (_nullish(item[1].get("parent")) is not None, item[0].name))
        if source.is_dir() and not md_items:
            return 500, {
                "error": f"policy version {selected} is incomplete (no .md files)",
                "error_type": "IncompletePolicyVersion",
            }
        frontmatter_edges: list[dict[str, Any]] = []
        for path, meta, body, node_id in md_items:
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
            frontmatter_edges.extend(_frontmatter_edges(path.read_text(encoding="utf-8"), node_id))

        edges_path = source / "edges.json"
        raw_edges = (
            json.loads(edges_path.read_text(encoding="utf-8"))
            if edges_path.exists()
            else []
        )
        if not isinstance(raw_edges, list):
            raw_edges = []
        edges = [
            edge
            for edge in (_normalize_edge(item) for item in [*raw_edges, *frontmatter_edges])
            if edge
        ]
        return 200, {
            "version": selected,
            "area": domain,
            "title": f"Cold-start {'MNIST digit' if domain == 'MNIST_Digits' else 'GenAI'} policy {selected}",
            "nodes": nodes,
            "edges": _dedupe_edges(edges),
            "available_versions": versions,
        }
    except ValueError as exc:
        return _bad_request(exc)
    except Exception as exc:  # noqa: BLE001 - surface to local web UI
        return _error(500, exc)


def handle_policy_versions(repo_root: Path | str, area: str | None = None) -> tuple[int, dict[str, Any]]:
    try:
        domain = normalize_policy_area(area)
        return 200, list_policy_versions(repo_root=repo_root, domain=domain)
    except ValueError as exc:
        return _bad_request(exc)
    except Exception as exc:  # noqa: BLE001 - surface to local web UI
        return _error(500, exc)


def handle_propose_diff(
    repo_root: Path | str,
    body: dict[str, Any] | None,
    *,
    async_requested: bool = False,
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

    domain = _domain_for_run(repo_root, run_id, body)

    if async_requested:
        try:
            return 202, _start_propose_diff_job(
                repo_root,
                run_id=run_id,
                base_version=base_version,
                model_id=model_id,
                proposed_files=proposed,
                files_removed=files_removed,
                domain=domain,
            )
        except Exception as exc:  # noqa: BLE001
            return _error(500, exc)

    try:
        meta = propose_diff(
            repo_root=repo_root,
            run_id=run_id,
            base_version=base_version,
            domain=domain,
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


def handle_list_proposals(
    repo_root: Path | str,
    *,
    include_errors: bool = False,
    area: str | None = None,
) -> tuple[int, dict[str, Any]]:
    try:
        body = list_proposals(repo_root=repo_root, include_errors=include_errors)
        if area is not None:
            domain = normalize_policy_area(area)
            body["proposals"] = [
                proposal
                for proposal in body.get("proposals", [])
                if _proposal_matches_domain(proposal, domain)
            ]
        return 200, body
    except ValueError as exc:
        return _bad_request(exc)
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


def handle_cold_start(
    repo_root: Path | str,
    body: dict[str, Any] | None,
) -> tuple[int, dict[str, Any]]:
    body = body or {}
    task_description = body.get("task_description")
    if not isinstance(task_description, str) or not task_description.strip():
        return 400, {"error": "task_description is required"}
    raw_domain = body.get("domain") or body.get("area")
    try:
        domain = normalize_policy_area(raw_domain, demo=body.get("demo"))
    except ValueError:
        return 400, {"error": f"unsupported domain: {raw_domain!r}"}
    model_id = body.get("model_id") or DEFAULT_POLICY_MODEL
    if not isinstance(model_id, str):
        return 400, {"error": "model_id must be a string"}

    try:
        meta = seed_cold_start_proposal(
            repo_root=repo_root,
            task_description=task_description,
            model_id=model_id,
            domain=domain,
        )
        return (200 if meta.get("status") != "parse_error" else 422), meta
    except ValueError as exc:
        return _bad_request(exc)
    except FileNotFoundError as exc:
        return _error(404, exc)
    except Exception as exc:  # noqa: BLE001
        return _error(500, exc)


def handle_grow_batch(
    repo_root: Path | str,
    body: dict[str, Any] | None,
) -> tuple[int, dict[str, Any]]:
    body = body or {}
    run_id = body.get("run_id")
    if not isinstance(run_id, str) or not run_id:
        return 400, {"error": "run_id is required"}
    base_version = body.get("base_version")
    if not isinstance(base_version, str) or not _VERSION_RE.match(base_version):
        return 400, {"error": "base_version is required and must match ^v\\d+\\.\\d+$"}
    batch_index = body.get("batch_index")
    if not isinstance(batch_index, int) or isinstance(batch_index, bool) or batch_index < 0:
        return 400, {"error": "batch_index must be an integer >= 0"}
    batch_size = body.get("batch_size")
    if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < 2:
        return 400, {"error": "batch_size must be an integer >= 2"}
    model_id = body.get("model_id") or DEFAULT_POLICY_MODEL
    if not isinstance(model_id, str):
        return 400, {"error": "model_id must be a string"}

    try:
        meta = propose_growth_batch(
            repo_root=repo_root,
            run_id=run_id,
            base_version=base_version,
            domain=_domain_for_run(repo_root, run_id, body),
            batch_index=batch_index,
            batch_size=batch_size,
            model_id=model_id,
        )
        return (200 if meta.get("status") != "parse_error" else 422), meta
    except ValueError as exc:
        return _bad_request(exc)
    except FileNotFoundError as exc:
        return _error(404, exc)
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
    try:
        area = normalize_policy_area(body.get("area") if isinstance(body.get("area"), str) else None)
    except ValueError as exc:
        return _bad_request(exc)

    root = _root(repo_root)
    source = root / "policy-graph" / area / version
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
        f"{area}.{version}",
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
    "handle_cold_start",
    "handle_get_proposal",
    "handle_grow_batch",
    "handle_list_proposals",
    "handle_policy_graph",
    "handle_policy_versions",
    "handle_propose_diff",
    "handle_reject_proposal",
]
