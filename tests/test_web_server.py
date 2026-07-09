from __future__ import annotations

import http.client
import json
import threading
from pathlib import Path

import pytest

from pipeline.web._safety import (
    APIError,
    safe_static_path,
    validate_cascade_payload,
    validate_start_payload,
)
from pipeline.web.server import create_server


_VALID = {
    "models": ["openai/gpt-5.5"],
    "split": "dev_golden",
    "limit": 1,
    "policy_version": "v0.1",
    "mode": "cold_start",
    "allow_spend": True,
    "concurrency": 1,
}

_VALID_CASCADE = {
    "models": ["openai/gpt-5.4-mini-low", "anthropic/claude-haiku-4-5-low"],
    "escalate_models": ["anthropic/claude-sonnet-5-medium"],
    "split": "dev_golden",
    "limit": 20,
    "policy_version": "v0.1",
    "mode": "cold_start",
    "allow_spend": True,
    "concurrency": 2,
}


def test_validate_cascade_payload_accepts_two_tier_request() -> None:
    normalized = validate_cascade_payload(dict(_VALID_CASCADE))

    assert normalized["models"] == [
        "openai/gpt-5.4-mini-low",
        "anthropic/claude-haiku-4-5-low",
    ]
    assert normalized["escalate_models"] == ["anthropic/claude-sonnet-5-medium"]
    assert normalized["limit"] == 20


def test_validate_cascade_payload_requires_escalate_models() -> None:
    payload = dict(_VALID_CASCADE)
    payload.pop("escalate_models")

    with pytest.raises(APIError) as excinfo:
        validate_cascade_payload(payload)
    assert excinfo.value.status == 400


def test_validate_cascade_payload_rejects_unknown_escalate_model() -> None:
    payload = dict(_VALID_CASCADE)
    payload["escalate_models"] = ["nope/not-a-model"]

    with pytest.raises(APIError) as excinfo:
        validate_cascade_payload(payload)
    assert excinfo.value.code == "unknown_model_id"


def test_validate_cascade_payload_rejects_single_cheap_model() -> None:
    payload = dict(_VALID_CASCADE)
    payload["models"] = ["openai/gpt-5.4-mini-low"]

    with pytest.raises(APIError) as excinfo:
        validate_cascade_payload(payload)
    assert excinfo.value.status == 400
    assert "consensus" in excinfo.value.message


def test_validate_cascade_payload_rejects_sample_ids() -> None:
    payload = dict(_VALID_CASCADE)
    payload["limit"] = None
    payload["sample_ids"] = "train_00001,train_00002"

    with pytest.raises(APIError) as excinfo:
        validate_cascade_payload(payload)
    assert excinfo.value.status == 400
    assert "sample_ids" in excinfo.value.message


@pytest.fixture
def web_server(tmp_path: Path):
    (tmp_path / "data" / "runs").mkdir(parents=True)
    server = create_server(host="127.0.0.1", port=0, repo_root=tmp_path)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        thread.join(timeout=2)
        server.server_close()


def _post_json(server, path: str, payload: dict) -> tuple[int, dict]:
    host, port = server.server_address
    conn = http.client.HTTPConnection(host, port, timeout=2)
    body = json.dumps(payload).encode("utf-8")
    conn.request(
        "POST",
        path,
        body=body,
        headers={"Content-Type": "application/json", "Content-Length": str(len(body))},
    )
    response = conn.getresponse()
    raw = response.read()
    conn.close()
    return response.status, json.loads(raw.decode("utf-8"))


@pytest.mark.parametrize(
    ("mutate", "code"),
    [
        (lambda p: p.update({"models": ["unknown/model"]}), "unknown_model_id"),
        (lambda p: p.pop("allow_spend"), "validation_error"),
        (lambda p: p.update({"split": "holdout"}), "validation_error"),
        (lambda p: p.update({"policy_version": "version-1"}), "validation_error"),
        (lambda p: p.update({"concurrency": 5}), "validation_error"),
        (lambda p: p.pop("limit"), "validation_error"),
    ],
)
def test_post_start_rejects_invalid_payloads(web_server, mutate, code: str) -> None:
    payload = dict(_VALID)
    mutate(payload)
    status, data = _post_json(web_server, "/api/runs/start", payload)
    assert status == 400
    assert data["error"]["code"] == code


def test_validate_start_payload_allows_missing_reasoning_effort_with_variants() -> None:
    payload = dict(_VALID)
    payload["models"] = ["openai/gpt-5.5-xhigh", "openai/gpt-5.4-mini-high"]

    normalized = validate_start_payload(payload)

    assert normalized["models"] == ["openai/gpt-5.5-xhigh", "openai/gpt-5.4-mini-high"]
    assert normalized["reasoning_effort"] is None


def test_validate_start_payload_accepts_high_reasoning_effort() -> None:
    payload = dict(_VALID)
    payload["reasoning_effort"] = "high"

    normalized = validate_start_payload(payload)

    assert normalized["reasoning_effort"] == "high"


def test_validate_start_payload_accepts_local_reasoning_map() -> None:
    payload = dict(_VALID)
    payload["local_reasoning"] = {
        "local/qwen3.6-27b": True,
        "local/gemma-4-26b-a4b-qat": False,
    }

    normalized = validate_start_payload(payload)

    assert normalized["local_reasoning"] == {
        "local/qwen3.6-27b": True,
        "local/gemma-4-26b-a4b-qat": False,
    }


@pytest.mark.parametrize(
    "local_reasoning",
    [
        ["local/qwen3.6-27b=on"],
        {"openai/gpt-5.5": True},
        {"local/nope": True},
        {"local/qwen3.6-27b": "on"},
    ],
)
def test_validate_start_payload_rejects_invalid_local_reasoning(local_reasoning) -> None:
    payload = dict(_VALID)
    payload["local_reasoning"] = local_reasoning

    with pytest.raises(APIError) as excinfo:
        validate_start_payload(payload)

    assert excinfo.value.status == 400
    assert excinfo.value.details["field"] == "local_reasoning"


def test_validate_start_payload_accepts_mnist_demo_area() -> None:
    payload = dict(_VALID)
    payload.update({"demo": "mnist", "area": "MNIST_Digits"})

    normalized = validate_start_payload(payload)

    assert normalized["demo"] == "mnist"
    assert normalized["area"] == "MNIST_Digits"


def test_validate_start_payload_rejects_unknown_area() -> None:
    payload = dict(_VALID)
    payload["area"] = "Other_Area"

    with pytest.raises(APIError) as excinfo:
        validate_start_payload(payload)

    assert excinfo.value.status == 400
    assert excinfo.value.code == "validation_error"
    assert excinfo.value.details == {"field": "area"}


def test_validate_start_payload_rejects_invalid_reasoning_effort() -> None:
    payload = dict(_VALID)
    payload["reasoning_effort"] = "medium"

    with pytest.raises(APIError) as excinfo:
        validate_start_payload(payload)

    assert excinfo.value.status == 400
    assert excinfo.value.code == "validation_error"
    assert excinfo.value.details == {"field": "reasoning_effort"}


def test_static_path_validation_rejects_dotdot(tmp_path: Path) -> None:
    with pytest.raises(APIError):
        safe_static_path(tmp_path, "/web/%2e%2e/secret.txt")


def test_server_refuses_non_localhost_bind(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="127.0.0.1"):
        create_server(host="0.0.0.0", port=0, repo_root=tmp_path)


# ---- experiment crank payload validation ----------------------------------


def _experiment_payload(**overrides):
    payload = {
        "area": "MNIST_Digits",
        "models": ["openai/gpt-5.4-mini-low", "google/gemini-3.1-flash-lite"],
        "allow_spend": True,
        "live": True,
    }
    payload.update(overrides)
    return payload


def test_validate_experiment_payload_defaults() -> None:
    from pipeline.web._safety import validate_experiment_payload

    request = validate_experiment_payload(_experiment_payload())
    assert request["k_max"] == 5
    assert request["batch_n"] == 20
    assert request["test_n"] == 100
    assert request["max_changes"] == 5
    assert request["max_anchors"] == 10
    assert request["max_aligned_anchors"] == 10  # aligned-anchor split default
    assert request["gate_model"] == "openai/gpt-5.5"
    assert request["gate_mode"] == "agent"
    assert request["seed"] is None
    assert request["epsilon"] == 0.0


def test_validate_experiment_payload_aligned_anchor_split() -> None:
    from pipeline.web._safety import validate_experiment_payload

    # 0 disables the aligned side; explicit values pass through and clamp.
    off = validate_experiment_payload(_experiment_payload(max_aligned_anchors=0))
    assert off["max_aligned_anchors"] == 0
    explicit = validate_experiment_payload(
        _experiment_payload(max_anchors=6, max_aligned_anchors=4)
    )
    assert explicit["max_anchors"] == 6 and explicit["max_aligned_anchors"] == 4


def test_validate_experiment_payload_panel_bounds() -> None:
    from pipeline.web._safety import APIError, validate_experiment_payload

    with pytest.raises(APIError):
        validate_experiment_payload(_experiment_payload(models=["openai/gpt-5.4-mini-low"]))
    six = [
        "openai/gpt-5.4-mini-low", "google/gemini-3.1-flash-lite",
        "anthropic/claude-haiku-4-5-low", "openai/gpt-5.5-low",
        "google/gemini-3.5-flash", "anthropic/claude-sonnet-4-6",
    ]
    with pytest.raises(APIError):
        validate_experiment_payload(_experiment_payload(models=six))


def test_validate_experiment_payload_rejects_bad_fields() -> None:
    from pipeline.web._safety import APIError, validate_experiment_payload

    with pytest.raises(APIError):
        validate_experiment_payload(_experiment_payload(max_changes=6))
    with pytest.raises(APIError):
        validate_experiment_payload(_experiment_payload(gate_model="local/qwen2.5-vl-7b"))
    with pytest.raises(APIError):
        validate_experiment_payload(_experiment_payload(gate_mode="vibes"))
    with pytest.raises(APIError):
        # Gate off on a dry run would mint real versions from fake no-op edits.
        validate_experiment_payload(
            _experiment_payload(gate_mode="off", live=False, allow_spend=False)
        )
    with pytest.raises(APIError):
        validate_experiment_payload(_experiment_payload(policy_version="../etc"))
    with pytest.raises(APIError):
        # Live without allow_spend refuses (402).
        validate_experiment_payload(_experiment_payload(allow_spend=False))


def test_validate_experiment_payload_gate_off_live() -> None:
    from pipeline.web._safety import validate_experiment_payload

    request = validate_experiment_payload(_experiment_payload(gate_mode="off"))
    assert request["gate_mode"] == "off"
    # The agent model field stays valid (unused when the gate is off).
    assert request["gate_model"] == "openai/gpt-5.5"


def test_validate_experiment_payload_gate_agent_only() -> None:
    from pipeline.web._safety import validate_experiment_payload

    request = validate_experiment_payload(
        _experiment_payload(gate_mode="agent_only", gate_model="openai/gpt-5.5-low")
    )
    assert request["gate_mode"] == "agent_only"
    assert request["gate_model"] == "openai/gpt-5.5-low"
    # Unlike gate-off, the critic gate still gates — dry runs are allowed
    # (the fake gate defers to the metric rule offline).
    request = validate_experiment_payload(
        _experiment_payload(gate_mode="agent_only", live=False, allow_spend=False)
    )
    assert request["gate_mode"] == "agent_only"


def test_experiment_endpoints_list_detail_and_review(tmp_path: Path, monkeypatch) -> None:
    from pipeline import experiment as exp
    from pipeline.experiment import store as exp_store
    from pipeline.web import handlers_experiment
    from pipeline.web._safety import APIError

    monkeypatch.setattr(exp_store, "try_sync_gate_review", lambda **_: False)
    exp_id = exp.mint_experiment_id()
    exp.write_state(tmp_path, {
        "experiment_id": exp_id,
        "area": "MNIST_Digits",
        "seed": 7,
        "status": "completed",
        "started_at": exp.utcnow_iso(),
        "cycles": [{"k": 0}, {"k": 1, "status": "skipped", "gate": {"decision": "skip"}}],
    })

    status, body = handlers_experiment.handle_list_experiments(tmp_path)
    assert status == 200 and body["experiments"][0]["experiment_id"] == exp_id

    status, body = handlers_experiment.handle_get_experiment(tmp_path, exp_id)
    assert status == 200 and body["seed"] == 7

    with pytest.raises(APIError) as excinfo:
        handlers_experiment.handle_get_experiment(tmp_path, "exp-19990101T000000-abc123")
    assert excinfo.value.status == 404
    with pytest.raises(APIError):
        handlers_experiment.handle_get_experiment(tmp_path, "../sneaky")

    status, body = handlers_experiment.handle_gate_review(
        tmp_path, exp_id, {"k": 1, "verdict": "incorrect", "comment": "bad skip"}
    )
    assert status == 200 and body["review"]["verdict"] == "incorrect"
    with pytest.raises(APIError):
        handlers_experiment.handle_gate_review(tmp_path, exp_id, {"k": 0, "verdict": "correct"})
    with pytest.raises(APIError):
        handlers_experiment.handle_gate_review(tmp_path, exp_id, {"k": 1, "verdict": "nope"})


def test_handle_adjudication_queue_aggregates(tmp_path) -> None:
    from pipeline import experiment as exp
    from pipeline.web.handlers_experiment import handle_adjudication_queue

    state = {
        "experiment_id": exp.mint_experiment_id(),
        "run_number": 7,
        "area": "MNIST_Digits",
        "seed": 7,
        "dry_run": False,
        "status": "completed",
        "started_at": exp.utcnow_iso(),
        "cycles": [],
        "readjudication": {"items": [{
            "image_id": "img_q", "sha256": "sha-q", "repo_rel_path": "p.png",
            "split": "dev_golden", "sme_truth": "7",
            "misalignment_type": "consensus_wrong", "severity": "high",
            "source": {"kind": "test", "k": 1, "run_id": "r7",
                       "policy": "MNIST_Digits.v0.2"},
            "n_judges": 2, "majority_label": "1",
            "consensus": {"decisive": 2, "majority_count": 2,
                          "fraction": 1.0, "tie": False},
            "avg_confidence": 0.9, "difficulty_score": 1.0,
            "gradient": {"n": 2, "avg_magnitude": 0.9, "max_magnitude": 0.9,
                         "avg_hessian": 0.09, "avg_loss": 2.3},
            "any_boundary": True, "boundary_pairs": ["1↔7"], "votes": [],
        }]},
    }
    exp.write_state(tmp_path, state)

    status, body = handle_adjudication_queue(tmp_path, {"area": ["MNIST_Digits"]})
    assert status == 200
    assert body["n_items"] == 1
    assert body["items"][0]["run_numbers"] == [7]

    status, body = handle_adjudication_queue(tmp_path, {"area": ["Generative_AI"]})
    assert status == 200 and body["n_items"] == 0

    # no query -> all areas
    status, body = handle_adjudication_queue(tmp_path, None)
    assert status == 200 and body["n_items"] == 1


def test_validate_experiment_payload_strategy_and_drafter() -> None:
    from pipeline.web._safety import APIError, validate_experiment_payload

    request = validate_experiment_payload(_experiment_payload())
    assert request["strategy"] == "random_misalignment"
    assert request["max_anchors"] == 10

    request = validate_experiment_payload(
        _experiment_payload(strategy="top_gradient",
                            drafter_model="openai/gpt-5.4-mini-low")
    )
    assert request["strategy"] == "top_gradient"
    assert request["drafter_model"] == "openai/gpt-5.4-mini-low"

    # Every strategy the backend/UI offers must validate here — the web layer
    # can't be stricter than the CLI (top_importance regression).
    from pipeline.experiment import STRATEGIES
    for strategy in STRATEGIES:
        assert validate_experiment_payload(
            _experiment_payload(strategy=strategy))["strategy"] == strategy

    with pytest.raises(APIError):
        validate_experiment_payload(_experiment_payload(strategy="s9_hunches"))


def test_handle_adjudication_review_records_and_returns_queue(tmp_path) -> None:
    from pipeline import experiment as exp
    from pipeline.web.handlers_experiment import (
        handle_adjudication_review, handle_adjudication_queue,
    )
    from pipeline.web._safety import APIError

    # Seed one queued item.
    exp_dir = tmp_path / "data" / "experiments" / "exp-20260101T000000-bbbbbb"
    exp_dir.mkdir(parents=True)
    votes = [{"model": f"m{i}", "label": "1", "confidence": 0.9, "difficulty": "high",
              "is_boundary": False} for i in range(3)]
    sig = exp.panel_signal({"sme_truth": "7", "votes": votes})
    item = {"image_id": "img1", "sha256": "sha-9", "repo_rel_path": "p.png",
            "split": "dev_golden", "sme_truth": "7", "misalignment_type": "consensus_wrong",
            "severity": "high", "source": {"kind": "test", "k": 0, "run_id": "r0"},
            **sig, "votes": votes}
    state = {"experiment_id": "exp-20260101T000000-bbbbbb", "run_number": 1,
             "area": "MNIST_Digits", "seed": 1, "status": "completed",
             "started_at": "2026-01-01T00:00:00Z", "dry_run": False,
             "cycles": [], "readjudication": {"items": [item]}}
    exp.write_state(tmp_path, state)

    status, body = handle_adjudication_review(tmp_path, {
        "area": "MNIST_Digits", "key": "sha-9", "image_id": "img1",
        "verdict": "confirm", "prior_truth": "7"})
    assert status == 200
    assert body["recorded"]["verdict"] == "confirm"
    assert body["queue"]["items"][0]["review"]["resolution"] == "confirmed"

    for bad in ({"area": "MNIST_Digits", "key": "", "verdict": "confirm"},
                {"area": "MNIST_Digits", "key": "sha-9", "verdict": "nope"},
                {"area": "MNIST_Digits", "key": "sha-9", "verdict": "overturn"}):
        try:
            handle_adjudication_review(tmp_path, bad)
            assert False, f"expected APIError for {bad}"
        except APIError:
            pass
