from __future__ import annotations

import json
import threading
import time
from pathlib import Path

from pipeline.web import run_registry as run_registry_mod
from pipeline.web.run_registry import RunRegistry, _manifest_is_completed


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _make_run(runs_root: Path, run_id: str, *, started_at: str) -> None:
    run_dir = runs_root / run_id
    _write_json(
        run_dir / "run_manifest.json",
        {
            "run_id": run_id,
            "started_at": started_at,
            "finished_at": "2026-05-10T23:01:00Z",
            "split": "dev_golden",
            "policy_graph_version": "v0.1",
            "models": [{"model_id": "openai/gpt-5.5"}],
            "totals": {"expected_calls": 1, "completed_calls": 1, "errored_calls": 0},
        },
    )


def test_list_runs_ignores_internal_dirs(tmp_path: Path) -> None:
    runs_root = tmp_path / "data" / "runs"
    _make_run(runs_root, "20260510T230000-aaaaaaaa", started_at="2026-05-10T23:00:00Z")
    _make_run(runs_root, "20260510T231000-bbbbbbbb", started_at="2026-05-10T23:10:00Z")
    _write_json(runs_root / "_jobs" / "run_manifest.json", {"run_id": "bad"})
    _write_json(runs_root / "_junk" / "run_manifest.json", {"run_id": "bad"})
    (runs_root / "20260510T231000-bbbbbbbb" / "scoring").mkdir(parents=True)
    (runs_root / "20260510T231000-bbbbbbbb" / "scoring" / "decision_quality.json").write_text("{}")

    registry = RunRegistry(tmp_path)
    runs = registry.list_runs()

    assert [r["run_id"] for r in runs] == [
        "20260510T231000-bbbbbbbb",
        "20260510T230000-aaaaaaaa",
    ]
    assert runs[0]["scoring_done"] is True
    assert all(not r["run_id"].startswith("_") for r in runs)


def test_status_includes_running_cost_estimate(tmp_path: Path) -> None:
    runs_root = tmp_path / "data" / "runs"
    run_id = "20260510T230000-aaaaaaaa"
    _make_run(runs_root, run_id, started_at="2026-05-10T23:00:00Z")
    run_dir = runs_root / run_id
    (run_dir / "label_votes.jsonl").write_text(
        '{"cost_usd": 0.10}\n{"cost_usd": null}\n{"cost_usd": 0.25}\n',
        encoding="utf-8",
    )

    status = RunRegistry(tmp_path).status(run_id)

    assert status["completed_calls"] == 3
    assert status["running_cost_usd_estimate"] == 0.35


def test_aborted_manifest_is_not_completed() -> None:
    assert (
        _manifest_is_completed(
            {
                "status": "aborted",
                "finished_at": "2026-05-10T23:01:00Z",
                "totals": {
                    "expected_calls": 1,
                    "completed_calls": 1,
                    "errored_calls": 0,
                },
            }
        )
        is False
    )


def test_status_includes_per_model_speed_rollup(tmp_path: Path) -> None:
    runs_root = tmp_path / "data" / "runs"
    run_id = "20260510T230000-dddddddd"
    run_dir = runs_root / run_id
    _write_json(
        run_dir / "run_manifest.json",
        {
            "run_id": run_id,
            "started_at": "2026-05-10T23:00:00Z",
            "finished_at": "2026-05-10T23:02:00Z",
            "split": "dev_golden",
            "policy_graph_version": "v0.1",
            "models": [{"model_id": "openai/gpt-5.5"}, {"model_id": "local/qwen3.6-27b"}],
            "totals": {"expected_calls": 4, "completed_calls": 3, "errored_calls": 0},
        },
    )
    (run_dir / "label_votes.jsonl").write_text(
        '{"model_id": "openai/gpt-5.5", "latency_ms": 1000}\n'
        '{"model_id": "openai/gpt-5.5", "latency_ms": 3000}\n'
        '{"model_id": "local/qwen3.6-27b", "latency_ms": 60000}\n',
        encoding="utf-8",
    )

    status = RunRegistry(tmp_path).status(run_id)

    assert status["elapsed_seconds"] == 120.0
    per_model = {r["model_id"]: r for r in status["per_model"]}
    assert set(per_model) == {"openai/gpt-5.5", "local/qwen3.6-27b"}
    gpt = per_model["openai/gpt-5.5"]
    assert gpt["calls_done"] == 2
    assert gpt["calls_total"] == 2
    assert gpt["avg_latency_ms"] == 2000.0
    assert gpt["done"] is True
    qwen = per_model["local/qwen3.6-27b"]
    assert qwen["calls_done"] == 1
    assert qwen["calls_total"] == 2
    assert qwen["avg_latency_ms"] == 60000.0
    assert qwen["done"] is False
    # Slowest/incomplete first: qwen (still running) ahead of finished gpt.
    assert status["per_model"][0]["model_id"] == "local/qwen3.6-27b"


def test_dead_job_state_is_not_running_and_aborts_manifest(monkeypatch, tmp_path: Path) -> None:
    runs_root = tmp_path / "data" / "runs"
    run_id = "20260510T233000-deadbeef"
    job_id = "job-20260510T233001-deadbeef"
    run_dir = runs_root / run_id
    _write_json(
        run_dir / "run_manifest.json",
        {
            "run_id": run_id,
            "started_at": "2026-05-10T23:30:00Z",
            "finished_at": None,
            "split": "dev_golden",
            "policy_graph_version": "v0.1",
            "models": [{"model_id": "openai/gpt-5.5"}],
            "totals": {"expected_calls": 3, "completed_calls": 1, "errored_calls": 0},
        },
    )
    _write_json(
        runs_root / "_jobs" / f"{job_id}.json",
        {
            "job_id": job_id,
            "run_id": run_id,
            "pid": 99999999,
            "started_at": "2026-05-10T23:30:00Z",
            "finished_at": None,
            "returncode": None,
        },
    )
    monkeypatch.setattr(run_registry_mod, "_process_is_alive", lambda pid: False)

    registry = RunRegistry(tmp_path)
    assert registry.is_job_running(job_id) is False

    manifest = json.loads((run_dir / "run_manifest.json").read_text())
    state = json.loads((runs_root / "_jobs" / f"{job_id}.json").read_text())
    assert manifest["status"] == "aborted"
    assert manifest["finished_at"]
    assert state["status"] == "aborted"
    assert state["returncode"] == -9


def test_status_transitions_from_running_job_to_resolved_run(monkeypatch, tmp_path: Path) -> None:
    run_id = "20260510T232000-cccccccc"
    done = threading.Event()
    created: list["FakePopen"] = []

    class BlockingStdout:
        def __iter__(self):
            done.wait(timeout=2)
            yield json.dumps({"run_id": run_id}) + "\n"

    class FakePopen:
        def __init__(self, argv, **kwargs):
            assert kwargs["shell"] is False
            assert kwargs["stderr"] is run_registry_mod.subprocess.STDOUT
            self.argv = argv
            self.pid = 12345
            self.stdout = BlockingStdout()
            created.append(self)

        def poll(self):
            return 0 if done.is_set() else None

        def wait(self):
            done.wait(timeout=2)
            return 0

    monkeypatch.setattr(run_registry_mod.subprocess, "Popen", FakePopen)
    registry = RunRegistry(tmp_path)
    state = registry.start_job(
        {
            "models": ["openai/gpt-5.5"],
            "split": "dev_golden",
            "limit": 3,
            "sample_ids": None,
            "policy_version": "v0.1",
            "mode": "cold_start",
            "reasoning_effort": "high",
            "allow_spend": True,
            "allow_holdout": False,
            "concurrency": 1,
        }
    )

    assert created
    assert created[0].argv[:4] == ["/Users/sacsimoto/GitHub/RUSH/.venv/bin/python", "-u", "scripts/run_bulk_labeling.py", "--models"]
    assert "--reasoning-effort" in created[0].argv
    assert created[0].argv[created[0].argv.index("--reasoning-effort") + 1] == "high"
    assert state["reasoning_effort"] == "high"
    running = registry.status(state["job_id"])
    assert running["running"] is True
    assert running["run_id"] == state["job_id"]

    run_dir = tmp_path / "data" / "runs" / run_id
    _write_json(
        run_dir / "run_manifest.json",
        {
            "run_id": run_id,
            "started_at": "2026-05-10T23:20:00Z",
            "finished_at": "2026-05-10T23:20:05Z",
            "split": "dev_golden",
            "policy_graph_version": "v0.1",
            "models": [{"model_id": "openai/gpt-5.5"}],
            "totals": {"expected_calls": 3, "completed_calls": 3, "errored_calls": 0},
        },
    )
    (run_dir / "label_votes.jsonl").write_text('{"a": 1}\n{"a": 2}\n{"a": 3}\n', encoding="utf-8")

    done.set()
    for _ in range(100):
        status = registry.status(state["job_id"])
        if status["running"] is False and status["run_id"] == run_id:
            break
        time.sleep(0.01)

    assert status["running"] is False
    assert status["run_id"] == run_id
    assert status["expected_calls"] == 3
    assert status["completed_calls"] == 3
    assert status["progress"] == 1.0
    assert any(run_id in line for line in status["log_tail"])


def test_start_job_omits_reasoning_arg_when_variant_carries_effort(monkeypatch, tmp_path: Path) -> None:
    created: list["FakePopen"] = []

    class EmptyStdout:
        def __iter__(self):
            return iter(())

    class FakePopen:
        def __init__(self, argv, **kwargs):
            self.argv = argv
            self.pid = 23456
            self.stdout = EmptyStdout()
            created.append(self)

        def poll(self):
            return 0

        def wait(self):
            return 0

    monkeypatch.setattr(run_registry_mod.subprocess, "Popen", FakePopen)
    registry = RunRegistry(tmp_path)
    state = registry.start_job(
        {
            "models": ["openai/gpt-5.5-xhigh", "openai/gpt-5.5-high"],
            "split": "dev_golden",
            "limit": 3,
            "sample_ids": None,
            "policy_version": "v0.1",
            "mode": "cold_start",
            "reasoning_effort": None,
            "allow_spend": True,
            "allow_holdout": False,
            "concurrency": 1,
        }
    )

    assert created
    assert "--reasoning-effort" not in created[0].argv
    assert state["reasoning_effort"] is None


def test_start_job_threads_mnist_area_manifest_and_policy(monkeypatch, tmp_path: Path) -> None:
    created: list["FakePopen"] = []

    class EmptyStdout:
        def __iter__(self):
            return iter(())

    class FakePopen:
        def __init__(self, argv, **kwargs):
            self.argv = argv
            self.pid = 34567
            self.stdout = EmptyStdout()
            created.append(self)

        def poll(self):
            return 0

        def wait(self):
            return 0

    monkeypatch.setattr(run_registry_mod.subprocess, "Popen", FakePopen)
    registry = RunRegistry(tmp_path)
    state = registry.start_job(
        {
            "models": ["local/qwen3.6-27b"],
            "split": "all",
            "limit": 2,
            "sample_ids": None,
            "policy_version": "v0.1",
            "mode": "cold_start",
            "reasoning_effort": None,
            "allow_spend": True,
            "allow_holdout": True,
            "concurrency": 1,
            "area": "MNIST_Digits",
            "demo": "mnist",
        }
    )

    argv = created[0].argv
    assert argv[argv.index("--area") + 1] == "MNIST_Digits"
    assert argv[argv.index("--policy-version") + 1] == "v0.1"
    assert argv[argv.index("--manifest") + 1].endswith(
        "data/images/mnist-classification/manifests/combined_labels.jsonl"
    )
    assert state["area"] == "MNIST_Digits"
    assert state["policy_graph_version"] == "MNIST_Digits.v0.1"
