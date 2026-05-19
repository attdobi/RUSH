from __future__ import annotations

import json
import time
from pathlib import Path

from pipeline.policy_diff import propose_diff
from pipeline.web import handlers_policy


def _seed_repo(tmp_path: Path) -> Path:
    base = tmp_path / "policy-graph" / "Generative_AI" / "v0.1"
    base.mkdir(parents=True)
    (base / "GA.root.md").write_text("# Root\n\nOld root text.\n", encoding="utf-8")
    scoring = tmp_path / "data" / "runs" / "run-robust" / "scoring"
    scoring.mkdir(parents=True)
    (scoring / "misalignment.json").write_text(
        json.dumps(
            {
                "records": [
                    {
                        "image_id": "img-1",
                        "sme_truth": "gen_ai",
                        "severity": "high",
                        "misalignment_type": "false_negative",
                        "votes": [],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    return tmp_path


def test_propose_diff_retries_timeout_once_then_succeeds(tmp_path: Path) -> None:
    root = _seed_repo(tmp_path)
    calls = {"count": 0}

    def fake_chat(*args, **kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            raise TimeoutError("temporary provider timeout")
        return json.dumps(
            {
                "files": [
                    {
                        "path": "GA.root.md",
                        "change": "modified",
                        "content": "# Root\n\nNew robust text.\n",
                    }
                ]
            }
        )

    proposal = propose_diff(
        repo_root=root,
        run_id="run-robust",
        base_version="v0.1",
        chat_callable=fake_chat,
        llm_timeout_s=1,
        llm_retries=2,
        llm_backoff_s=0,
    )

    assert calls["count"] == 2
    assert proposal["status"] == "pending"
    assert proposal["files_changed"] == ["GA.root.md"]


def test_propose_diff_parse_error_includes_excerpt(tmp_path: Path) -> None:
    root = _seed_repo(tmp_path)

    proposal = propose_diff(
        repo_root=root,
        run_id="run-robust",
        base_version="v0.1",
        chat_callable=lambda *args, **kwargs: "not json at all",
        llm_backoff_s=0,
    )

    assert proposal["status"] == "parse_error"
    assert proposal["error_type"] == "parse_error"
    assert "not valid JSON" in proposal["error"]
    assert proposal["raw_response_excerpt"] == "not json at all"


def test_async_propose_diff_job_completes(tmp_path: Path, monkeypatch) -> None:
    calls = {"count": 0}

    def fake_propose_diff(**kwargs):
        calls["count"] += 1
        kwargs["progress_callback"](
            {
                "status": "building_retry",
                "attempt": 2,
                "max_attempts": 3,
                "retry_count": 1,
                "max_retries": 2,
                "reason": "temporary provider timeout",
            }
        )
        return {
            "proposal_id": "p-job",
            "base_version": kwargs["base_version"],
            "model_id": kwargs["model_id"],
            "run_id": kwargs["run_id"],
            "status": "pending",
            "files_changed": ["GA.root.md"],
            "files_added": [],
            "files_removed": [],
        }

    monkeypatch.setattr(handlers_policy, "propose_diff", fake_propose_diff)
    status, body = handlers_policy.handle_propose_diff(
        tmp_path,
        {"run_id": "run-robust", "base_version": "v0.1"},
        async_requested=True,
    )

    assert status == 202
    assert body["job_id"]
    assert body["status"] in {"queued", "building", "building_retry", "success"}

    for _ in range(50):
        status, job = handlers_policy.handle_get_propose_diff_job(tmp_path, body["job_id"])
        assert status == 200
        if job["status"] == "success":
            break
        time.sleep(0.02)
    else:
        raise AssertionError(f"job did not finish: {job}")

    assert calls["count"] == 1
    assert job["proposal_id"] == "p-job"
    assert job["result"]["status"] == "pending"
    assert job["retry_count"] == 1
