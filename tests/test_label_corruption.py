"""Protocol A label corruption — the run knob for controlled noise injection.

The sim (sim/label-noise) predicts what wrong human labels do to policy
descent; the run option lets the real crank reproduce it. These tests pin the
contracts the knob lives or dies by:

  * web validation bounds — rho in [0, 0.6], mode random|anchors, both in the
    sanitized payload;
  * argv threading from the web launcher to the driver (only when nonzero);
  * determinism — the same experiment seed plans the same flips;
  * isolation — corruption is an in-memory VIEW: the plan writes nothing, the
    record patch copies, on-disk artifacts (what the label store ingests and
    the SME queue rebuilds from) stay clean-truth;
  * anchors mode restricts the flip pool to the k=0-misaligned ids.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from pipeline import experiment as exp
from pipeline.scoring.tasks import GENAI_BINARY, MNIST_MULTICLASS


# ---- web validation bounds -------------------------------------------------


def _experiment_payload(**overrides):
    payload = {
        "area": "MNIST_Digits",
        "models": ["openai/gpt-5.4-mini-low", "google/gemini-3.1-flash-lite"],
        "allow_spend": True,
        "launch_pin": "4850",
        "live": True,
    }
    payload.update(overrides)
    return payload


def test_validate_corrupt_labels_defaults_and_bounds() -> None:
    from pipeline.web._safety import APIError, validate_experiment_payload

    clean = validate_experiment_payload(_experiment_payload())
    assert clean["corrupt_labels"] == 0.0
    assert clean["corrupt_mode"] == "random"

    corrupted = validate_experiment_payload(
        _experiment_payload(corrupt_labels=0.6, corrupt_mode="anchors")
    )
    assert corrupted["corrupt_labels"] == 0.6
    assert corrupted["corrupt_mode"] == "anchors"

    for bad in (-0.1, 0.61, float("nan"), float("inf"), "0.2", True):
        with pytest.raises(APIError) as excinfo:
            validate_experiment_payload(_experiment_payload(corrupt_labels=bad))
        assert excinfo.value.status == 400
        assert excinfo.value.details == {"field": "corrupt_labels"}

    with pytest.raises(APIError) as excinfo:
        validate_experiment_payload(
            _experiment_payload(corrupt_labels=0.2, corrupt_mode="sabotage")
        )
    assert excinfo.value.status == 400
    assert excinfo.value.details == {"field": "corrupt_mode"}


# ---- argv threading (web launcher -> driver) -------------------------------


def _fake_popen(created: list):
    class FakePopen:
        def __init__(self, argv, **kwargs):
            self.argv = argv
            self.pid = 4242
            self.stdout = iter(())
            created.append(self)

        def poll(self):
            return 0

        def wait(self, timeout=None):
            return 0

    return FakePopen


def _experiment_request(**overrides):
    request = {
        "area": "MNIST_Digits",
        "models": ["openai/gpt-5.4-mini-low", "google/gemini-3.1-flash-lite"],
        "seed": 42,
        "k_max": 1,
        "batch_n": 4,
        "test_n": 10,
        "max_changes": 3,
        "max_anchors": 15,
        "max_aligned_anchors": 5,
        "epsilon": 0,
        "gate_model": "openai/gpt-5.5",
        "gate_mode": "metric_only",
        "gate_persona": "lenient",
        "drafter_model": "openai/gpt-5.5",
        "drafter_context": "text_only",
        "strategy": "random_misalignment",
        "concurrency": 2,
        "live": False,
        "allow_spend": False,
    }
    request.update(overrides)
    return request


def test_start_experiment_job_threads_corrupt_argv(monkeypatch, tmp_path: Path) -> None:
    from pipeline.web import run_registry as run_registry_mod
    from pipeline.web.run_registry import RunRegistry

    created: list = []
    monkeypatch.setattr(run_registry_mod.subprocess, "Popen", _fake_popen(created))
    registry = RunRegistry(tmp_path)
    state = registry.start_experiment_job(
        _experiment_request(corrupt_labels=0.25, corrupt_mode="anchors")
    )

    assert created
    argv = created[0].argv
    assert argv[argv.index("--corrupt-labels") + 1] == "0.25"
    assert argv[argv.index("--corrupt-mode") + 1] == "anchors"
    # The knob lands in the job's experiment_config for the UI/analysis layer.
    assert state["experiment_config"]["corrupt_labels"] == 0.25
    assert state["experiment_config"]["corrupt_mode"] == "anchors"


def test_start_experiment_job_clean_run_omits_corrupt_argv(monkeypatch, tmp_path: Path) -> None:
    from pipeline.web import run_registry as run_registry_mod
    from pipeline.web.run_registry import RunRegistry

    created: list = []
    monkeypatch.setattr(run_registry_mod.subprocess, "Popen", _fake_popen(created))
    registry = RunRegistry(tmp_path)
    registry.start_experiment_job(_experiment_request(corrupt_labels=0))

    assert created
    assert "--corrupt-labels" not in created[0].argv
    assert "--corrupt-mode" not in created[0].argv


# ---- determinism -----------------------------------------------------------


def test_corruption_plan_is_deterministic_per_seed() -> None:
    truth = {f"img_{i:03d}": str(i % 10) for i in range(100)}

    plan = exp.plan_label_corruption(
        truth, seed=42, fraction=0.2, task=MNIST_MULTICLASS
    )
    again = exp.plan_label_corruption(
        truth, seed=42, fraction=0.2, task=MNIST_MULTICLASS
    )
    assert plan == again  # same seed -> same flips, same targets

    assert plan["n_flipped"] == 20
    assert plan["flipped_ids"] == sorted(plan["overrides"])
    for image_id, new_label in plan["overrides"].items():
        # MNIST flips to a RANDOM OTHER class, still in the task vocabulary.
        assert new_label != truth[image_id]
        assert new_label in MNIST_MULTICLASS.classes

    other = exp.plan_label_corruption(
        truth, seed=43, fraction=0.2, task=MNIST_MULTICLASS
    )
    assert other["corrupt_seed"] != plan["corrupt_seed"]
    assert other["flipped_ids"] != plan["flipped_ids"]


def test_binary_corruption_flips_to_the_other_class() -> None:
    truth = {f"g_{i:02d}": ("gen_ai" if i % 2 else "not_gen_ai") for i in range(40)}

    plan = exp.plan_label_corruption(truth, seed=7, fraction=0.5, task=GENAI_BINARY)

    assert plan["n_flipped"] == 20
    for image_id, new_label in plan["overrides"].items():
        expected = "not_gen_ai" if truth[image_id] == "gen_ai" else "gen_ai"
        assert new_label == expected


def test_corruption_plan_rejects_bad_inputs() -> None:
    truth = {"a": "3", "b": "5"}
    with pytest.raises(ValueError):
        exp.plan_label_corruption(truth, seed=1, fraction=0.7, task=MNIST_MULTICLASS)
    with pytest.raises(ValueError):
        exp.plan_label_corruption(truth, seed=1, fraction=-0.1, task=MNIST_MULTICLASS)
    with pytest.raises(ValueError):
        exp.plan_label_corruption(
            truth, seed=1, fraction=0.2, task=MNIST_MULTICLASS, mode="sabotage"
        )


# ---- anchors mode: pool restriction ----------------------------------------


def test_anchors_mode_restricts_flips_to_the_supplied_pool() -> None:
    truth = {f"img_{i:03d}": str(i % 10) for i in range(60)}
    # The driver passes the k=0-misaligned ids; one id has no golden label
    # (e.g. an errored call) and must silently drop from the pool.
    pool = [f"img_{i:03d}" for i in range(10)] + ["img_999"]

    plan = exp.plan_label_corruption(
        truth, seed=11, fraction=0.5, task=MNIST_MULTICLASS, pool=pool,
        mode="anchors",
    )

    assert plan["corrupt_mode"] == "anchors"
    assert plan["n_flipped"] == 5  # round(0.5 * the 10 eligible pool ids)
    assert set(plan["flipped_ids"]) <= set(pool) - {"img_999"}


# ---- isolation: an in-memory view, never a persisted write ------------------


def test_corruption_is_an_in_memory_view_only(tmp_path: Path) -> None:
    """The record patch copies; the artifact on disk keeps clean truth.

    The label store ingests run artifacts from disk and the SME
    re-adjudication queue rebuilds from the same files — as long as those
    bytes never change, corruption cannot be persisted anywhere.
    """
    records = [
        {
            "image_id": "img_1", "sme_truth": "3", "split": "dev_golden",
            "misalignment_type": "all_agree", "severity": "low",
            "votes": [
                {"labeler_id": "m1", "label": "3"},
                {"labeler_id": "m2", "label": "3"},
            ],
        },
        {
            "image_id": "img_2", "sme_truth": "5", "split": "dev_golden",
            "misalignment_type": "all_agree", "severity": "low",
            "votes": [
                {"labeler_id": "m1", "label": "5"},
                {"labeler_id": "m2", "label": "5"},
            ],
        },
    ]
    artifact = tmp_path / "misalignment.json"
    artifact.write_text(json.dumps({"records": records}), encoding="utf-8")
    before = artifact.read_bytes()

    loaded = json.loads(artifact.read_text(encoding="utf-8"))["records"]
    patched = exp.corrupt_misalignment_records(loaded, {"img_1": "7"})

    view = {r["image_id"]: r for r in patched}
    # Flipped truth + misalignment_type/severity re-derived under it: both
    # judges now "disagree" with the planted label -> consensus_wrong/high.
    assert view["img_1"]["sme_truth"] == "7"
    assert view["img_1"]["misalignment_type"] == "consensus_wrong"
    assert view["img_1"]["severity"] == "high"
    # Un-flipped records pass through untouched.
    assert view["img_2"]["sme_truth"] == "5"
    assert view["img_2"]["misalignment_type"] == "all_agree"

    # The loaded originals and the artifact bytes are unchanged.
    assert loaded[0]["sme_truth"] == "3"
    assert loaded[0]["misalignment_type"] == "all_agree"
    assert artifact.read_bytes() == before


def test_corruption_plan_mutates_nothing() -> None:
    truth = {f"img_{i:03d}": str(i % 10) for i in range(30)}
    snapshot = dict(truth)

    exp.plan_label_corruption(truth, seed=1, fraction=0.3, task=MNIST_MULTICLASS)

    assert truth == snapshot


def test_empty_overrides_is_a_no_op_passthrough() -> None:
    records = [{"image_id": "x", "sme_truth": "1", "votes": []}]
    assert exp.corrupt_misalignment_records(records, {}) is records


def test_list_experiments_carries_corruption_fields(tmp_path: Path) -> None:
    state = {
        "experiment_id": "exp-20260712T000000-abc123",
        "area": "MNIST_Digits",
        "seed": 42,
        "status": "completed",
        "corrupt_labels": 0.2,
        "corrupt_mode": "anchors",
        "started_at": "2026-07-12T00:00:00Z",
        "cycles": [],
    }
    exp.write_state(tmp_path, state)

    rows = exp.list_experiments(tmp_path)

    assert rows and rows[0]["corrupt_labels"] == 0.2
    assert rows[0]["corrupt_mode"] == "anchors"
