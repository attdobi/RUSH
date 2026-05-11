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
from typing import Any

from pipeline.io_paths import RUN_ID_PATTERN

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
        ]
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
            "policy_version": request["policy_version"],
            "allow_spend": bool(request["allow_spend"]),
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

    def is_job_running(self, token: str) -> bool:
        state = self.find_job(token)
        if not state:
            return False
        job_id = state["job_id"]
        with self._lock:
            proc = self._processes.get(job_id)
        if proc is not None:
            return proc.poll() is None
        refreshed = self._read_state(job_id)
        return refreshed.get("returncode") is None and refreshed.get("finished_at") is None

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
            "running_cost_usd_estimate": running_cost_usd_estimate,
            "scoring_done": bool(run_dir and (run_dir / "scoring" / "decision_quality.json").exists()),
            "returncode": (state or {}).get("returncode"),
            "log_tail": self.log_tail(token),
        }

    def score(self, token: str) -> dict[str, Any]:
        run_id = self.resolve_run_id(token)
        if not run_id:
            raise APIError(404, "run_not_found", f"unknown run or unresolved job id: {token}")
        run_dir = self.runs_root / run_id
        if not (run_dir / "run_manifest.json").exists():
            raise APIError(404, "run_not_found", f"run not found: {run_id}")
        scoring_path = run_dir / "scoring" / "decision_quality.json"
        if scoring_path.exists():
            raise APIError(409, "already_scored", f"run already scored: {run_id}")
        result = subprocess.run(
            [".venv/bin/python", "scripts/score_labels.py", "--run-id", run_id],
            cwd=str(self.repo_root),
            env=os.environ.copy(),
            shell=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        if result.returncode != 0:
            raise APIError(
                500,
                "score_failed",
                f"score_labels.py exited with {result.returncode}",
                details={"output": result.stdout[-4000:] if result.stdout else ""},
            )
        return {"ok": True, "scoring_done": True}
