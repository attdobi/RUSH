#!/usr/bin/env python3
"""Finalize stale run manifests whose owning process is gone.

This is a one-shot cleanup for web/CLI runs that wrote
``data/runs/<run_id>/run_manifest.json`` with ``finished_at=null`` and then
lost their subprocess before the manifest could be finalized.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse_iso(value: object) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise


def _process_is_alive(pid: object) -> bool:
    try:
        pid_int = int(pid)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return False
    if pid_int <= 0:
        return False
    try:
        os.kill(pid_int, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _job_states(runs_root: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for path in sorted((runs_root / "_jobs").glob("*.json")):
        state = _read_json(path)
        run_id = state.get("run_id")
        if isinstance(run_id, str) and run_id:
            out[run_id] = state
    return out


def _manifest_owner_pid(manifest: dict[str, Any], state: dict[str, Any] | None) -> object:
    if state and state.get("pid") is not None:
        return state.get("pid")
    for key in ("pid", "owner_pid", "process_pid"):
        if manifest.get(key) is not None:
            return manifest.get(key)
    process = manifest.get("process")
    if isinstance(process, dict) and process.get("pid") is not None:
        return process.get("pid")
    return None


def finalize_orphans(
    runs_root: Path,
    *,
    dry_run: bool = False,
    unowned_grace_minutes: int = 10,
) -> list[str]:
    fixed: list[str] = []
    if not runs_root.is_dir():
        return fixed

    now = datetime.now(timezone.utc)
    finished_at = _utcnow_iso()
    jobs_by_run = _job_states(runs_root)
    for manifest_path in sorted(runs_root.glob("*/run_manifest.json")):
        if manifest_path.parent.name.startswith("_"):
            continue
        manifest = _read_json(manifest_path)
        if not manifest or manifest.get("finished_at"):
            continue
        status = str(manifest.get("status") or "").lower()
        if status == "completed":
            continue
        run_id = str(manifest.get("run_id") or manifest_path.parent.name)
        state = jobs_by_run.get(run_id)
        pid = _manifest_owner_pid(manifest, state)
        reason: str | None = None
        if pid is not None:
            if _process_is_alive(pid):
                continue
            reason = f"owning process pid {pid} is not alive"
        else:
            started = _parse_iso(manifest.get("started_at"))
            if started is None or now - started < timedelta(minutes=unowned_grace_minutes):
                continue
            reason = "unfinished run has no live owner metadata"

        fixed.append(run_id)
        if dry_run:
            continue
        manifest["finished_at"] = finished_at
        manifest["status"] = "aborted"
        manifest["abort_reason"] = reason
        _atomic_write_json(manifest_path, manifest)
        if state and not state.get("finished_at"):
            state["finished_at"] = finished_at
            state["returncode"] = -9
            state["status"] = "aborted"
            state["abort_reason"] = reason
            job_id = state.get("job_id")
            if isinstance(job_id, str):
                _atomic_write_json(runs_root / "_jobs" / f"{job_id}.json", state)
    return fixed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", default=str(_REPO_ROOT / "data" / "runs"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--unowned-grace-minutes", type=int, default=10)
    args = parser.parse_args(argv)

    fixed = finalize_orphans(
        Path(args.runs_root),
        dry_run=args.dry_run,
        unowned_grace_minutes=args.unowned_grace_minutes,
    )
    verb = "would finalize" if args.dry_run else "finalized"
    print(f"{verb} {len(fixed)} orphaned run(s)")
    for run_id in fixed:
        print(run_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
