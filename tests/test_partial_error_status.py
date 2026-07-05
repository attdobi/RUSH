from __future__ import annotations

import io
import importlib
import json
from collections import deque
from contextlib import redirect_stdout
from pathlib import Path

from pipeline.manifest import SampleRecord
from pipeline.providers.base import LabelRequest, LabelResponse
from pipeline.runner import DeterministicFakeClient, run_labeling
from pipeline.web.run_registry import RunRegistry


SAMPLES = [
    SampleRecord(
        sample_id=f"dev_golden_{idx:04d}",
        repo_rel_path=f"data/images/test/dev_golden_{idx:04d}.jpg",
        split="dev_golden",
        sme_label_raw="ai_generated" if idx % 2 else "not_ai_generated",
        sme_label="gen_ai" if idx % 2 else "not_gen_ai",
        dataset="test",
        sha256=f"{idx:064x}"[-64:],
        sampling_version="test-sampling-v1",
    )
    for idx in range(1, 5)
]


def _read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _parse_failed(request: LabelRequest, attempts: int = 1) -> LabelResponse:
    return LabelResponse(
        image_id=request.image_id,
        model_id=request.model_id,
        label="abstain",
        l2_label="",
        justification="non-JSON response",
        confidence=None,
        difficulty="medium",
        is_boundary=False,
        raw_provider_payload={"text": "not json"},
        error="parse_failed",
        latency_ms=1,
        attempts=attempts,
    )


class RetryThenSuccessClient(DeterministicFakeClient):
    def __init__(self, model_id: str) -> None:
        super().__init__(model_id)
        self.calls = 0

    def label(self, request: LabelRequest) -> LabelResponse:
        self.calls += 1
        if self.calls == 1:
            return _parse_failed(request)
        return super().label(request)


class OneImageAlwaysParseFailsClient(DeterministicFakeClient):
    def label(self, request: LabelRequest) -> LabelResponse:
        if request.image_id == "dev_golden_0001":
            return _parse_failed(request)
        return super().label(request)


class AlwaysParseFailsClient(DeterministicFakeClient):
    def label(self, request: LabelRequest) -> LabelResponse:
        return _parse_failed(request)


def test_parse_failed_retry_recovers_on_second_attempt(tmp_path: Path) -> None:
    clients: dict[str, RetryThenSuccessClient] = {}

    def factory(spec):
        client = RetryThenSuccessClient(spec.model_id)
        clients[spec.model_id] = client
        return client

    summary = run_labeling(
        models=["local/qwen3.6-27b"],
        samples=SAMPLES,
        split="dev_golden",
        limit=1,
        runs_root=tmp_path,
        client_factory=factory,
        concurrency=1,
    )

    assert summary.completed_calls == 1
    assert summary.errored_calls == 0
    assert clients["local/qwen3.6-27b"].calls == 2
    votes = _read_jsonl(summary.paths.label_votes)
    assert votes[0]["attempts"] == 2
    manifest = json.loads(summary.paths.manifest.read_text(encoding="utf-8"))
    assert manifest["status"] == "completed"
    assert manifest["completed_with_errors"] is False


def test_cli_partial_parse_failure_exits_zero_and_scores(monkeypatch, tmp_path: Path) -> None:
    from scripts import run_bulk_labeling
    scoring_mod = importlib.import_module("pipeline.scoring.run_scoring")

    monkeypatch.setattr(
        run_bulk_labeling,
        "_resolve_factory",
        lambda **_kwargs: (lambda spec: OneImageAlwaysParseFailsClient(spec.model_id)),
    )
    scoring_calls: list[str] = []

    def fake_run_scoring(run_id: str, repo_root: Path, *, runs_root: Path) -> dict:
        scoring_calls.append(run_id)
        return {"run_id": run_id, "scored": True}

    monkeypatch.setattr(scoring_mod, "run_scoring", fake_run_scoring)

    out = io.StringIO()
    with redirect_stdout(out):
        rc = run_bulk_labeling.main(
            [
                "--models",
                "local/qwen3.6-27b",
                "--split",
                "dev_golden",
                "--limit",
                "2",
                "--runs-root",
                str(tmp_path),
            ]
        )

    payload = json.loads(out.getvalue())
    assert rc == 0
    assert payload["completed_calls"] == 1
    assert payload["errored_calls"] == 1
    assert payload["completed_with_errors"] is True
    assert payload["fatal_error"] is None
    assert scoring_calls == [payload["run_id"]]
    assert payload["scoring"] == {"run_id": payload["run_id"], "scored": True}

    manifest = json.loads((tmp_path / payload["run_id"] / "run_manifest.json").read_text())
    assert manifest["status"] == "completed"
    assert manifest["completed_with_errors"] is True

    status = RunRegistry(Path.cwd(), runs_root=tmp_path).status(payload["run_id"])
    assert status["status"] == "completed"
    assert status["completed_with_errors"] is True
    assert status["returncode"] is None


def test_cli_all_calls_fail_exits_nonzero_and_finalizes_failed(monkeypatch, tmp_path: Path) -> None:
    from scripts import run_bulk_labeling
    scoring_mod = importlib.import_module("pipeline.scoring.run_scoring")

    monkeypatch.setattr(
        run_bulk_labeling,
        "_resolve_factory",
        lambda **_kwargs: (lambda spec: AlwaysParseFailsClient(spec.model_id)),
    )
    scoring_calls: list[str] = []
    monkeypatch.setattr(
        scoring_mod,
        "run_scoring",
        lambda run_id, repo_root, *, runs_root: scoring_calls.append(run_id),
    )

    out = io.StringIO()
    with redirect_stdout(out):
        rc = run_bulk_labeling.main(
            [
                "--models",
                "local/qwen3.6-27b",
                "--split",
                "dev_golden",
                "--limit",
                "2",
                "--runs-root",
                str(tmp_path),
            ]
        )

    payload = json.loads(out.getvalue())
    assert rc == 1
    assert payload["completed_calls"] == 0
    assert payload["errored_calls"] == 2
    assert payload["completed_with_errors"] is False
    assert payload["fatal_error"] == "all calls failed"
    assert scoring_calls == []

    manifest = json.loads((tmp_path / payload["run_id"] / "run_manifest.json").read_text())
    assert manifest["status"] == "failed"
    assert manifest["abort_reason"] == "all calls failed"
    assert manifest["completed_with_errors"] is False


def test_run_registry_autoscores_completed_with_errors(monkeypatch, tmp_path: Path) -> None:
    runs_root = tmp_path / "data" / "runs"
    run_id = "20260510T232000-abc12345"
    job_id = "job-20260510T232001-abc12345"
    run_dir = runs_root / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "run_manifest.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "started_at": "2026-05-10T23:20:00Z",
                "finished_at": "2026-05-10T23:20:05Z",
                "status": "completed",
                "completed_with_errors": True,
                "split": "dev_golden",
                "policy_graph_version": "v0.1",
                "models": [{"model_id": "local/qwen3.6-27b"}],
                "totals": {
                    "expected_calls": 2,
                    "completed_calls": 1,
                    "errored_calls": 1,
                },
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "label_votes.jsonl").write_text('{"model_id": "local/qwen3.6-27b"}\n')
    jobs_root = runs_root / "_jobs"
    jobs_root.mkdir(parents=True)
    (jobs_root / f"{job_id}.json").write_text(
        json.dumps({"job_id": job_id, "started_at": "2026-05-10T23:20:00Z"}),
        encoding="utf-8",
    )

    class FakeProc:
        stdout = [json.dumps({"run_id": run_id}) + "\n"]

        def wait(self) -> int:
            return 0

    scoring_calls: list[str] = []

    def fake_compute_now(self: RunRegistry, token: str) -> dict:
        scoring_calls.append(token)
        return {"run_id": token, "scored": True}

    monkeypatch.setattr(RunRegistry, "compute_now", fake_compute_now)
    registry = RunRegistry(tmp_path, runs_root=runs_root)

    registry._monitor_process(job_id, FakeProc(), registry._log_path(job_id), deque(maxlen=20))

    state = json.loads((jobs_root / f"{job_id}.json").read_text(encoding="utf-8"))
    assert state["status"] == "completed"
    assert state["returncode"] == 0
    assert state["completed_with_errors"] is True
    assert state["errored_calls"] == 1
    assert state["scoring_result"] == {"run_id": run_id, "scored": True}
    assert scoring_calls == [run_id]


def test_every_call_for_one_model_failed_is_fatal(tmp_path: Path) -> None:
    def factory(spec):
        if spec.model_id == "local/qwen3.6-27b":
            return AlwaysParseFailsClient(spec.model_id)
        return DeterministicFakeClient(spec.model_id)

    summary = run_labeling(
        models=["local/gemma-4-26b-a4b-qat", "local/qwen3.6-27b"],
        samples=SAMPLES,
        split="dev_golden",
        limit=2,
        runs_root=tmp_path,
        client_factory=factory,
        concurrency=1,
    )

    assert summary.completed_calls == 2
    assert summary.errored_calls == 2
    assert summary.fatal_error == "all calls failed for model(s): local/qwen3.6-27b"
    manifest = json.loads(summary.paths.manifest.read_text(encoding="utf-8"))
    assert manifest["status"] == "failed"
    assert manifest["abort_reason"] == summary.fatal_error
