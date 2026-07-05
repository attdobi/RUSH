"""Run discovery and job lifecycle tracking for the local web API."""
from __future__ import annotations

from collections import deque
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import secrets
import subprocess
import threading
import traceback
from typing import Any

from pipeline.io_paths import RUN_ID_PATTERN
from pipeline.scoring import run_scoring

from ._safety import APIError, utcnow_iso

_ARCHIVE_PREFIX = "_"


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except json.JSONDecodeError:
        return {}


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def _count_jsonl_lines(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("rb") as fh:
        return sum(1 for line in fh if line.strip())


def _sum_jsonl_cost(path: Path) -> float:
    if not path.exists():
        return 0.0
    total = 0.0
    with path.open("r", encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            value = row.get("cost_usd")
            if value is None or isinstance(value, bool):
                continue
            try:
                total += float(value)
            except (TypeError, ValueError):
                continue
    return total


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


def _finalize_manifest_aborted(
    manifest_path: Path,
    *,
    finished_at: str,
    reason: str,
) -> bool:
    manifest = _read_json(manifest_path)
    if not manifest or manifest.get("finished_at"):
        return False
    status = str(manifest.get("status") or "").lower()
    if status == "completed":
        return False
    manifest["finished_at"] = finished_at
    manifest["status"] = "aborted"
    manifest["abort_reason"] = reason
    _atomic_write_json(manifest_path, manifest)
    return True


def _read_model_speed_summary(run_dir: Path) -> dict[str, Any] | None:
    payload = _read_json(run_dir / "model_speed_summary.json")
    return payload if payload else None


def _parse_iso(ts: str | None) -> datetime | None:
    if not ts or not isinstance(ts, str):
        return None
    text = ts.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _elapsed_seconds(started_at: str | None, finished_at: str | None) -> float:
    """Wall-clock seconds from started_at to finished_at (or now if running)."""
    start = _parse_iso(started_at)
    if start is None:
        return 0.0
    end = _parse_iso(finished_at) or datetime.now(timezone.utc)
    return max(0.0, (end - start).total_seconds())


def _per_model_rollup(
    votes_path: Path | None,
    model_ids: list[str],
    elapsed_seconds: float,
    expected_calls: int,
) -> list[dict[str, Any]]:
    """Per-model speed telemetry from label_votes.jsonl (carries model_id +
    latency_ms). calls_total is derived from the manifest's expected_calls split
    evenly across the configured models (uniform image set per model)."""
    done: dict[str, int] = {}
    latency_sum: dict[str, int] = {}
    latency_n: dict[str, int] = {}
    for mid in model_ids:
        done[mid] = 0
        latency_sum[mid] = 0
        latency_n[mid] = 0
    if votes_path and votes_path.exists():
        with votes_path.open("r", encoding="utf-8") as fh:
            for raw in fh:
                line = raw.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                mid = row.get("model_id")
                if not isinstance(mid, str):
                    continue
                done[mid] = done.get(mid, 0) + 1
                lat = row.get("latency_ms")
                if isinstance(lat, (int, float)) and not isinstance(lat, bool) and lat >= 0:
                    latency_sum[mid] = latency_sum.get(mid, 0) + int(lat)
                    latency_n[mid] = latency_n.get(mid, 0) + 1
    n_models = len(model_ids)
    per_model_total = (expected_calls // n_models) if (n_models and expected_calls) else 0
    rollup: list[dict[str, Any]] = []
    all_ids = list(dict.fromkeys([*model_ids, *[m for m in done if m not in model_ids]]))
    for mid in all_ids:
        calls_done = done.get(mid, 0)
        calls_total = per_model_total if mid in model_ids else calls_done
        n = latency_n.get(mid, 0)
        avg_latency_ms = (latency_sum.get(mid, 0) / n) if n else None
        throughput = (calls_done / (elapsed_seconds / 60.0)) if elapsed_seconds > 0 else None
        rollup.append(
            {
                "model_id": mid,
                "calls_done": calls_done,
                "calls_total": calls_total,
                "avg_latency_ms": avg_latency_ms,
                "throughput_imgs_per_min": throughput,
                "done": bool(calls_total) and calls_done >= calls_total,
            }
        )
    # Slowest-first: models still with work / lowest throughput bubble up; done last.
    rollup.sort(
        key=lambda r: (
            r["done"],
            r["throughput_imgs_per_min"] if r["throughput_imgs_per_min"] is not None else float("inf"),
        )
    )
    return rollup


def _job_id() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    return f"job-{stamp}-{secrets.token_hex(4)}"


def _last_json_object(text: str) -> dict[str, Any] | None:
    decoder = json.JSONDecoder()
    last: dict[str, Any] | None = None
    for idx, char in enumerate(text):
        if char != "{":
            continue
        try:
            obj, _end = decoder.raw_decode(text[idx:])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            last = obj
    return last


def _manifest_models(manifest: dict[str, Any]) -> list[str]:
    models: list[str] = []
    for item in manifest.get("models", []):
        if isinstance(item, str):
            models.append(item)
        elif isinstance(item, dict) and isinstance(item.get("model_id"), str):
            models.append(item["model_id"])
    return models


def _manifest_is_completed(manifest: dict[str, Any]) -> bool:
    status = str(manifest.get("status") or "").lower()
    if status in {"aborted", "failed", "cancelled", "canceled"}:
        return False
    if status == "completed":
        return True
    totals = manifest.get("totals", {}) if isinstance(manifest.get("totals"), dict) else {}
    expected = int(totals.get("expected_calls") or 0)
    completed = int(totals.get("completed_calls") or 0)
    errored = int(totals.get("errored_calls") or 0)
    return bool(manifest.get("finished_at")) and errored == 0 and (expected == 0 or completed >= expected)


class RunRegistry:
    """Discovers completed runs and tracks subprocess-backed live jobs."""

    def __init__(self, repo_root: Path, *, runs_root: Path | None = None) -> None:
        self.repo_root = repo_root.resolve()
        self.runs_root = (runs_root or self.repo_root / "data" / "runs").resolve()
        self.jobs_root = self.runs_root / "_jobs"
        self.jobs_root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._processes: dict[str, subprocess.Popen] = {}
        self._tails: dict[str, deque[str]] = {}

    def _state_path(self, job_id: str) -> Path:
        return self.jobs_root / f"{job_id}.json"

    def _log_path(self, job_id: str) -> Path:
        return self.jobs_root / f"{job_id}.log"

    def _write_state(self, state: dict[str, Any]) -> None:
        _atomic_write_json(self._state_path(state["job_id"]), state)

    def _read_state(self, job_id: str) -> dict[str, Any]:
        return _read_json(self._state_path(job_id))

    def start_job(self, request: dict[str, Any]) -> dict[str, Any]:
        job_id = _job_id()
        argv = [
            ".venv/bin/python",
            "-u",
            "scripts/run_bulk_labeling.py",
            "--models",
            ",".join(request["models"]),
            "--split",
            request["split"],
            "--live",
            "--allow-spend",
            "--concurrency",
            str(request["concurrency"]),
            "--policy-version",
            request["policy_version"],
            "--batch-size",
            str(request.get("batch_size") or 20),
        ]
        if request.get("reasoning_effort") is not None:
            argv.extend(["--reasoning-effort", request["reasoning_effort"]])
        if request.get("limit") is not None:
            argv.extend(["--limit", str(request["limit"])])
        if request.get("sample_ids"):
            argv.extend(["--sample-ids", request["sample_ids"]])
        if request.get("split") == "holdout" or request.get("allow_holdout"):
            argv.append("--allow-holdout")

        state: dict[str, Any] = {
            "job_id": job_id,
            "run_id": None,
            "pid": None,
            "argv": argv,
            "started_at": utcnow_iso(),
            "finished_at": None,
            "returncode": None,
            "models": list(request["models"]),
            "split": request["split"],
            "mode": request["mode"],
            # New picker flow encodes reasoning in model ids; keep this nullable
            # and let per-model ids/runtime config be the source of truth.
            "reasoning_effort": request.get("reasoning_effort"),
            "policy_version": request["policy_version"],
            "allow_spend": bool(request["allow_spend"]),
            "allow_holdout": bool(request.get("allow_holdout")),
            "limit": request.get("limit"),
            "sample_ids": request.get("sample_ids"),
            "batch_size": request.get("batch_size"),
        }
        self._write_state(state)

        env = os.environ.copy()
        log_path = self._log_path(job_id)
        tail: deque[str] = deque(maxlen=200)
        proc = subprocess.Popen(
            argv,
            cwd=str(self.repo_root),
            env=env,
            shell=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        state["pid"] = proc.pid
        self._write_state(state)

        with self._lock:
            self._processes[job_id] = proc
            self._tails[job_id] = tail

        thread = threading.Thread(
            target=self._monitor_process,
            args=(job_id, proc, log_path, tail),
            name=f"rush-web-job-{job_id}",
            daemon=True,
        )
        thread.start()
        return state

    def _monitor_process(
        self,
        job_id: str,
        proc: subprocess.Popen,
        log_path: Path,
        tail: deque[str],
    ) -> None:
        output_parts: list[str] = []
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as log:
            stdout = proc.stdout
            if stdout is not None:
                for line in stdout:
                    log.write(line)
                    log.flush()
                    output_parts.append(line)
                    tail.append(line.rstrip("\n"))
            returncode = proc.wait()

        parsed = _last_json_object("".join(output_parts))
        run_id = parsed.get("run_id") if isinstance(parsed, dict) else None
        state = self._read_state(job_id)
        if isinstance(run_id, str):
            state["run_id"] = run_id
        state["finished_at"] = utcnow_iso()
        state["returncode"] = returncode
        state["status"] = "finished" if returncode == 0 else "aborted"
        if returncode != 0:
            state["abort_reason"] = f"run subprocess exited with returncode {returncode}"
        self._write_state(state)

        if returncode != 0:
            resolved_run_id = run_id if isinstance(run_id, str) else self._infer_run_id_for_job(state)
            if isinstance(resolved_run_id, str):
                _finalize_manifest_aborted(
                    self.runs_root / resolved_run_id / "run_manifest.json",
                    finished_at=state["finished_at"],
                    reason=state["abort_reason"],
                )

        if returncode == 0 and isinstance(run_id, str):
            run_dir = self.runs_root / run_id
            manifest = _read_json(run_dir / "run_manifest.json")
            if _manifest_is_completed(manifest):
                state["status"] = "scoring"
                state["scoring_started_at"] = utcnow_iso()
                self._write_state(state)
                with log_path.open("a", encoding="utf-8") as log:
                    log.write(f"\n[web] auto-scoring run {run_id}\n")
                    log.flush()
                    try:
                        score_result = self.compute_now(run_id)
                        state["status"] = "completed"
                        state["scoring_finished_at"] = utcnow_iso()
                        state["scoring_result"] = score_result
                        log.write("[web] auto-scoring completed\n")
                    except Exception as exc:  # pragma: no cover - defensive job monitor path
                        state["status"] = "scoring_failed"
                        state["scoring_finished_at"] = utcnow_iso()
                        state["scoring_error"] = str(exc)
                        log.write("[web] auto-scoring failed:\n")
                        log.write(traceback.format_exc())
                    log.flush()
                self._write_state(state)
        with self._lock:
            self._processes.pop(job_id, None)

    def _job_states(self) -> list[dict[str, Any]]:
        states: list[dict[str, Any]] = []
        for path in sorted(self.jobs_root.glob("*.json")):
            state = _read_json(path)
            if state.get("job_id"):
                states.append(state)
        return states

    def _infer_run_id_for_job(self, state: dict[str, Any]) -> str | None:
        """Best-effort early run_id discovery while the runner is still live."""
        existing = state.get("run_id")
        if isinstance(existing, str):
            return existing
        if not self.runs_root.exists():
            return None

        requested_models = sorted(state.get("models") or [])
        requested_split = state.get("split")
        job_started = state.get("started_at") or ""
        candidates: list[tuple[float, str]] = []
        for run_dir in self.runs_root.iterdir():
            if not run_dir.is_dir() or run_dir.name.startswith(_ARCHIVE_PREFIX):
                continue
            manifest_path = run_dir / "run_manifest.json"
            if not manifest_path.exists():
                continue
            manifest = _read_json(manifest_path)
            run_id = manifest.get("run_id") or run_dir.name
            if not isinstance(run_id, str) or not RUN_ID_PATTERN.match(run_id):
                continue
            if requested_split and manifest.get("split") != requested_split:
                continue
            if requested_models and sorted(_manifest_models(manifest)) != requested_models:
                continue
            started_at = manifest.get("started_at") or ""
            if job_started and started_at and started_at < job_started:
                continue
            candidates.append((manifest_path.stat().st_mtime, run_id))

        if not candidates:
            return None
        candidates.sort(reverse=True)
        run_id = candidates[0][1]
        updated = dict(state)
        updated["run_id"] = run_id
        self._write_state(updated)
        return run_id

    def find_job(self, token: str) -> dict[str, Any] | None:
        if token.startswith("job-"):
            state = self._read_state(token)
            if state:
                self._infer_run_id_for_job(state)
                state = self._read_state(token)
            return state or None
        for state in self._job_states():
            if state.get("run_id") == token:
                return state
        return None

    def resolve_run_id(self, token: str) -> str | None:
        if RUN_ID_PATTERN.match(token):
            return token
        state = self.find_job(token)
        if not state:
            return None
        run_id = state.get("run_id")
        if isinstance(run_id, str):
            return run_id
        return self._infer_run_id_for_job(state)

    def _finalize_dead_job(
        self,
        job_id: str,
        state: dict[str, Any],
        *,
        returncode: int | None = None,
        reason: str = "run subprocess is no longer alive",
    ) -> None:
        if state.get("finished_at") or state.get("returncode") is not None:
            return
        finished_at = utcnow_iso()
        updated = dict(state)
        updated["finished_at"] = finished_at
        updated["returncode"] = -9 if returncode is None else returncode
        updated["status"] = "aborted"
        updated["abort_reason"] = reason
        run_id = updated.get("run_id")
        if not isinstance(run_id, str):
            run_id = self._infer_run_id_for_job(updated)
            if isinstance(run_id, str):
                updated["run_id"] = run_id
        self._write_state(updated)
        if isinstance(run_id, str):
            _finalize_manifest_aborted(
                self.runs_root / run_id / "run_manifest.json",
                finished_at=finished_at,
                reason=reason,
            )

    def is_job_running(self, token: str) -> bool:
        state = self.find_job(token)
        if not state:
            return False
        job_id = state["job_id"]
        with self._lock:
            proc = self._processes.get(job_id)
        if proc is not None:
            returncode = proc.poll()
            if returncode is None:
                return True
            if returncode != 0:
                self._finalize_dead_job(
                    job_id,
                    state,
                    returncode=returncode,
                    reason=f"run subprocess exited with returncode {returncode}",
                )
            return False
        refreshed = self._read_state(job_id)
        if refreshed.get("returncode") is not None or refreshed.get("finished_at"):
            return False
        if _process_is_alive(refreshed.get("pid")):
            return True
        self._finalize_dead_job(job_id, refreshed)
        return False

    def log_tail(self, token: str, *, n: int = 40) -> list[str]:
        state = self.find_job(token)
        if not state:
            return []
        job_id = state["job_id"]
        with self._lock:
            tail = self._tails.get(job_id)
            if tail:
                return list(tail)[-n:]
        path = self._log_path(job_id)
        if not path.exists():
            return []
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        return lines[-n:]

    def list_runs(self) -> list[dict[str, Any]]:
        runs: list[dict[str, Any]] = []
        if not self.runs_root.exists():
            return runs
        for run_dir in self.runs_root.iterdir():
            if not run_dir.is_dir() or run_dir.name.startswith(_ARCHIVE_PREFIX):
                continue
            manifest_path = run_dir / "run_manifest.json"
            if not manifest_path.exists():
                continue
            manifest = _read_json(manifest_path)
            run_id = manifest.get("run_id") or run_dir.name
            if not isinstance(run_id, str):
                continue
            runs.append(
                {
                    "run_id": run_id,
                    "started_at": manifest.get("started_at"),
                    "finished_at": manifest.get("finished_at"),
                    "split": manifest.get("split"),
                    "policy_graph_version": manifest.get("policy_graph_version"),
                    "models": _manifest_models(manifest),
                    "totals": manifest.get("totals", {}),
                    "scoring_done": (run_dir / "scoring" / "decision_quality.json").exists(),
                    "running": self.is_job_running(run_id),
                    "status": manifest.get("status"),
                    "model_speed_summary": _read_model_speed_summary(run_dir),
                }
            )
        return sorted(runs, key=lambda row: row.get("started_at") or "", reverse=True)

    def status(self, token: str) -> dict[str, Any]:
        state = self.find_job(token)
        run_id = self.resolve_run_id(token)
        manifest: dict[str, Any] = {}
        run_dir: Path | None = None
        if run_id:
            run_dir = self.runs_root / run_id
            manifest = _read_json(run_dir / "run_manifest.json")
        elif state is None:
            raise APIError(404, "run_not_found", f"unknown run or job id: {token}")

        totals = manifest.get("totals", {}) if manifest else {}
        expected = int(totals.get("expected_calls") or 0)
        votes_path = run_dir / "label_votes.jsonl" if run_dir else None
        completed = _count_jsonl_lines(votes_path) if votes_path else 0
        running_cost_usd_estimate = _sum_jsonl_cost(votes_path) if votes_path else 0.0
        errored = int(totals.get("errored_calls") or 0)
        running = self.is_job_running(token) or (bool(run_id) and self.is_job_running(run_id))
        finished_at = manifest.get("finished_at") or (state or {}).get("finished_at")
        started_at = manifest.get("started_at") or (state or {}).get("started_at")
        progress = (completed / expected) if expected else 0.0
        elapsed_seconds = _elapsed_seconds(started_at, finished_at if not running else None)
        model_ids = _manifest_models(manifest) or list((state or {}).get("models") or [])
        per_model = _per_model_rollup(votes_path, model_ids, elapsed_seconds, expected)
        manifest_cost = manifest.get("cost") if isinstance(manifest.get("cost"), dict) else None
        recorded_cost = (
            manifest_cost.get("total_cost_usd") if manifest_cost else None
        )

        return {
            "run_id": run_id or token,
            "job_id": (state or {}).get("job_id"),
            "running": running,
            "started_at": started_at,
            "finished_at": finished_at,
            "expected_calls": expected,
            "completed_calls": completed,
            "errored_calls": errored,
            "progress": progress,
            "elapsed_seconds": elapsed_seconds,
            "per_model": per_model,
            "running_cost_usd_estimate": running_cost_usd_estimate,
            "cost": manifest_cost,
            "recorded_cost_usd": recorded_cost,
            "status": manifest.get("status") or (state or {}).get("status"),
            "model_speed_summary": _read_model_speed_summary(run_dir) if run_dir else None,
            "scoring_done": bool(run_dir and (run_dir / "scoring" / "decision_quality.json").exists()),
            "returncode": (state or {}).get("returncode"),
            "log_tail": self.log_tail(token),
        }

    def compute_now(self, token: str) -> dict[str, Any]:
        run_id = self.resolve_run_id(token)
        if not run_id:
            raise APIError(404, "run_not_found", f"unknown run or unresolved job id: {token}")
        run_dir = self.runs_root / run_id
        if not (run_dir / "run_manifest.json").exists():
            raise APIError(404, "run_not_found", f"run not found: {run_id}")
        try:
            return run_scoring(run_id, self.repo_root, runs_root=self.runs_root)
        except FileNotFoundError as exc:
            raise APIError(404, "run_not_found", str(exc)) from exc
        except Exception as exc:
            raise APIError(500, "score_failed", str(exc)) from exc

    def score(self, token: str) -> dict[str, Any]:
        """Backward-compatible alias for older Score now buttons."""
        return self.compute_now(token)
