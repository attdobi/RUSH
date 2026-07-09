"""Unit tests for the experiment crank core (pipeline/experiment).

The driver's subprocess loop is exercised by dry-running
``scripts/run_experiment.py`` manually; these tests pin the pure logic every
cycle depends on: seeded partitions/batches/anchors, the 1..5 change clip,
panel metrics (per judge + system), and the PPO gate truth table.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline import experiment as exp  # noqa: E402
from pipeline.scoring.decision_quality_multiclass import (  # noqa: E402
    compute_multiclass_metrics,
)
from pipeline.scoring.tasks import MNIST_MULTICLASS  # noqa: E402


@dataclass(frozen=True)
class FakeRecord:
    sample_id: str
    split: str
    sme_label: str


def _pool(n_per_class: int = 30, split: str = "dev_golden") -> list[FakeRecord]:
    return [
        FakeRecord(f"img_{digit}_{i:03d}", split, str(digit))
        for digit in range(10)
        for i in range(n_per_class)
    ]


# ---------------------------------------------------------------------------
# Seeded partition + sampling


def test_partition_is_deterministic_disjoint_and_stratified():
    records = _pool() + [FakeRecord("hold_1", "holdout", "3")]
    test_a, pool_a = exp.partition_test_train(records, seed=42, test_n=50)
    test_b, pool_b = exp.partition_test_train(records, seed=42, test_n=50)
    assert test_a == test_b and pool_a == pool_b  # same seed -> same partition
    assert len(test_a) == 50
    assert not set(test_a) & set(pool_a)
    assert len(test_a) + len(pool_a) == 300  # holdout excluded
    assert "hold_1" not in test_a and "hold_1" not in pool_a
    # Stratified: 50 / 10 classes = 5 per digit.
    per_class = {d: sum(1 for s in test_a if s.startswith(f"img_{d}_")) for d in range(10)}
    assert all(v == 5 for v in per_class.values())


def test_partition_different_seed_differs():
    records = _pool()
    test_a, _ = exp.partition_test_train(records, seed=1, test_n=50)
    test_b, _ = exp.partition_test_train(records, seed=2, test_n=50)
    assert test_a != test_b


def test_partition_rejects_degenerate_sizes():
    records = _pool(n_per_class=2)  # pool of 20
    with pytest.raises(ValueError):
        exp.partition_test_train(records, seed=1, test_n=20)
    with pytest.raises(ValueError):
        exp.partition_test_train(records, seed=1, test_n=0)


def test_train_batch_without_replacement_then_top_up():
    pool = [f"s{i:02d}" for i in range(10)]
    used: set[str] = set()
    batch1 = exp.sample_train_batch(pool, used, seed=7, k=1, batch_n=6)
    used.update(batch1)
    batch2 = exp.sample_train_batch(pool, used, seed=7, k=2, batch_n=6)
    assert len(batch1) == 6 and len(batch2) == 6
    assert not set(batch1) & (set(batch2) - set(pool[:0]) - (set(batch2) & set(batch1)))
    # batch2 must contain the 4 remaining fresh ids, topped up with 2 reused.
    fresh_remaining = set(pool) - set(batch1)
    assert fresh_remaining <= set(batch2)
    assert len(set(batch2) & set(batch1)) == 2
    # Determinism.
    assert batch1 == exp.sample_train_batch(pool, set(), seed=7, k=1, batch_n=6)


def test_cycle_seed_stable():
    assert exp.cycle_seed(42, 3) == exp.cycle_seed(42, 3)
    assert exp.cycle_seed(42, 3) != exp.cycle_seed(42, 4)


# ---------------------------------------------------------------------------
# S1 anchors


def _mis(image_id: str, mtype: str = "consensus_wrong") -> dict:
    return {"image_id": image_id, "misalignment_type": mtype, "severity": "high"}


def test_anchors_filter_all_agree_and_respect_cap():
    records = [_mis(f"a{i}") for i in range(20)] + [_mis("ok1", "all_agree")]
    anchors = exp.select_anchors(records, seed=5, k=1, max_anchors=8)
    assert len(anchors) == 8
    assert all(a["misalignment_type"] != "all_agree" for a in anchors)
    assert anchors == exp.select_anchors(records, seed=5, k=1, max_anchors=8)
    assert anchors != exp.select_anchors(records, seed=5, k=2, max_anchors=8)


def test_anchors_restricted_to_train_ids():
    records = [_mis("in1"), _mis("out1")]
    anchors = exp.select_anchors(records, seed=1, k=1, max_anchors=8, train_ids=["in1"])
    assert [a["image_id"] for a in anchors] == ["in1"]


# ---------------------------------------------------------------------------
# The 1..5 change clip


def test_clip_drops_noops_and_enforces_cap(tmp_path):
    base = tmp_path / "v0.1"
    base.mkdir()
    for name in ("a.md", "b.md", "c.md"):
        (base / name).write_text(f"# {name}\n", encoding="utf-8")
    proposed = {
        "a.md": "# a.md\n",          # no-op -> dropped, not counted
        "b.md": "# b.md CHANGED\n",  # 1: modified
        "d.md": "# new\n",           # 2: added
        "e.md": "# new2\n",          # 3: added
    }
    removed = ["c.md", "zz.md"]      # 4: removed; zz.md doesn't exist -> dropped
    clip = exp.clip_changes(proposed, removed, base_dir=base, max_changes=2)
    assert clip["n_proposed"] == 4
    assert clip["n_applied"] == 2
    assert clip["clipped"] is True
    # Emission order preserved: b.md then d.md survive the clip.
    assert [c["path"] for c in clip["changes"]] == ["b.md", "d.md"]
    assert clip["removed"] == []
    assert [d["path"] for d in clip["dropped"]] == ["e.md", "c.md"]


def test_clip_hard_cap_bounds():
    with pytest.raises(ValueError):
        exp.clip_changes({}, [], base_dir=Path("."), max_changes=0)
    with pytest.raises(ValueError):
        exp.clip_changes({}, [], base_dir=Path("."), max_changes=6)


def test_materialize_candidate_overlays_and_removes(tmp_path):
    base = tmp_path / "base"
    base.mkdir()
    (base / "keep.md").write_text("keep\n", encoding="utf-8")
    (base / "mod.md").write_text("old\n", encoding="utf-8")
    (base / "gone.md").write_text("bye\n", encoding="utf-8")
    out = exp.materialize_candidate(
        base_dir=base,
        out_dir=tmp_path / "cand",
        files={"mod.md": "new\n", "added.md": "hello\n"},
        removed=["gone.md"],
    )
    assert (out / "keep.md").read_text(encoding="utf-8") == "keep\n"
    assert (out / "mod.md").read_text(encoding="utf-8") == "new\n"
    assert (out / "added.md").exists()
    assert not (out / "gone.md").exists()
    # Base untouched.
    assert (base / "gone.md").exists()
    assert (base / "mod.md").read_text(encoding="utf-8") == "old\n"


# ---------------------------------------------------------------------------
# Panel metrics: per judge + the system of judges


def _vote(image_id: str, model: str, label: str) -> dict:
    return {"image_id": image_id, "model_id": model, "label": label}


def test_panel_metrics_per_judge_and_system_majority():
    truth = {"i1": "1", "i2": "2", "i3": "3", "i4": "4"}
    votes = []
    # m1 always right; m2 right except i2; m3 right except i2/i3.
    answers = {
        "m1": {"i1": "1", "i2": "2", "i3": "3", "i4": "4"},
        "m2": {"i1": "1", "i2": "7", "i3": "3", "i4": "4"},
        "m3": {"i1": "1", "i2": "7", "i3": "9", "i4": "4"},
    }
    for model, by_img in answers.items():
        votes.extend(_vote(i, model, lab) for i, lab in by_img.items())
    out = exp.panel_metrics(votes, truth, task=MNIST_MULTICLASS)
    assert set(out) == {"m1", "m2", "m3", exp.SYSTEM_SCORER}
    assert out["m1"]["accuracy"] == 1.0
    assert out["m2"]["accuracy"] == 0.75
    # System majority: i2 -> 7 (2 of 3, wrong), rest correct -> 3/4.
    assert out[exp.SYSTEM_SCORER]["accuracy"] == 0.75
    assert out[exp.SYSTEM_SCORER]["n"] == 4
    # All six tracked metric families present, macro + micro.
    for key in (
        "accuracy", "macro_f1", "macro_precision", "macro_recall", "macro_fpr",
        "macro_fnr", "micro_f1", "micro_precision", "micro_recall", "micro_fpr",
        "micro_fnr",
    ):
        assert key in out["m1"], key
    # Confusion matrix stripped from the cycle records.
    assert "confusion_matrix" not in out["m1"]


def test_panel_metrics_tie_excluded_and_abstain_counted():
    truth = {"i1": "1"}
    votes = [
        _vote("i1", "m1", "1"),
        _vote("i1", "m2", "2"),  # 1-1 tie -> system excluded
        _vote("i1", "m3", "abstain"),
    ]
    out = exp.panel_metrics(votes, truth, task=MNIST_MULTICLASS)
    assert exp.SYSTEM_SCORER not in out  # no decided majority anywhere
    assert out["m3"]["n"] == 0 and out["m3"]["n_abstained"] == 1


def test_fnr_in_multiclass_metrics():
    metrics = compute_multiclass_metrics(
        ["1", "1", "2", "2"], ["1", "2", "2", "2"], classes=("1", "2")
    )
    # class 2: tp=2 fn=1 -> fnr = 1/3 = 1 - recall.
    assert metrics["per_class"]["2"]["fnr"] == pytest.approx(1 / 3, abs=1e-6)
    assert metrics["per_class"]["2"]["fnr"] == pytest.approx(
        1 - metrics["per_class"]["2"]["recall"], abs=1e-6
    )
    assert metrics["micro_fnr"] == pytest.approx(1 - metrics["micro_recall"], abs=1e-6)
    assert metrics["macro_fnr"] is not None
    assert metrics["micro_f1"] is not None


# ---------------------------------------------------------------------------
# The PPO gate


def test_metric_passes_rule():
    assert exp.metric_passes(0.80, 0.81)
    assert not exp.metric_passes(0.80, 0.80)  # strict improvement
    assert not exp.metric_passes(0.80, 0.79)
    assert not exp.metric_passes(None, 0.9)  # undefined baseline never passes
    assert not exp.metric_passes(0.9, None)
    assert not exp.metric_passes(0.80, 0.81, epsilon=0.02)
    assert exp.metric_passes(0.80, 0.83, epsilon=0.02)


def test_gate_truth_table():
    accept = {"decision": "accept", "rationale": "r", "risk_flags": []}
    veto = {"decision": "skip", "rationale": "leaky", "risk_flags": ["leak"]}
    # Rule fail: agent can never force an accept.
    out = exp.resolve_gate_decision(metric_pass=False, agent=accept)
    assert out["decision"] == "skip" and out["decided_by"] == "override_guard"
    out = exp.resolve_gate_decision(metric_pass=False, agent=None)
    assert out["decision"] == "skip" and out["decided_by"] == "metric_rule"
    # Rule pass.
    out = exp.resolve_gate_decision(metric_pass=True, agent=None)
    assert out["decision"] == "accept" and out["decided_by"] == "metric_rule"
    out = exp.resolve_gate_decision(metric_pass=True, agent=accept)
    assert out["decision"] == "accept" and out["decided_by"] == "gate_agent"
    out = exp.resolve_gate_decision(metric_pass=True, agent=veto)
    assert out["decision"] == "skip" and out["decided_by"] == "gate_agent_veto"
    assert out["risk_flags"] == ["leak"]


def test_gate_agent_only_verdict_decides_regardless_of_metric():
    accept = {"decision": "accept", "rationale": "sound boundary node", "risk_flags": []}
    veto = {"decision": "skip", "rationale": "overfits", "risk_flags": ["overfit"]}
    # The critic may accept a metric-FAILING edit (the whole point of the mode).
    out = exp.resolve_gate_decision(metric_pass=False, agent=accept, agent_only=True)
    assert out["decision"] == "accept" and out["decided_by"] == "gate_agent"
    # And may skip a metric-passing one.
    out = exp.resolve_gate_decision(metric_pass=True, agent=veto, agent_only=True)
    assert out["decision"] == "skip" and out["decided_by"] == "gate_agent"
    assert out["risk_flags"] == ["overfit"]
    # No verdict (agent error): fall back to the metric rule — the gate never
    # silently degrades to gate-off.
    out = exp.resolve_gate_decision(metric_pass=True, agent=None, agent_only=True)
    assert out["decision"] == "accept" and out["decided_by"] == "metric_rule"
    assert "gate_agent_unavailable" in out["risk_flags"]
    out = exp.resolve_gate_decision(metric_pass=False, agent=None, agent_only=True)
    assert out["decision"] == "skip" and out["decided_by"] == "metric_rule"


def test_gate_agent_only_prompt_selection():
    kwargs = dict(
        metric=exp.GATE_METRIC, value_before=0.9, value_after=0.9, epsilon=0.0,
        metric_pass=False, metrics_before={}, metrics_after={}, diffs=[],
        anchors=[], k=1,
    )
    default_messages = exp.build_gate_messages(**kwargs)
    critic_messages = exp.build_gate_messages(**kwargs, agent_is_sole_gate=True)
    assert default_messages[0]["content"].startswith(exp.GATE_SYSTEM_PROMPT)
    assert critic_messages[0]["content"].startswith(exp.GATE_AGENT_ONLY_SYSTEM_PROMPT)
    assert "verdict alone decides" in critic_messages[0]["content"]


def test_gate_persona_stance_appended_and_validated():
    kwargs = dict(
        metric=exp.GATE_METRIC, value_before=0.9, value_after=0.9, epsilon=0.0,
        metric_pass=False, metrics_before={}, metrics_after={}, diffs=[],
        anchors=[], k=1,
    )
    # Default = lenient (Attila 2026-07-09: the old implicit stance read as
    # too strict — metric-flat-but-sound edits were skipped).
    assert exp.DEFAULT_GATE_PERSONA == "lenient"
    default = exp.build_gate_messages(**kwargs)[0]["content"]
    assert "STANCE — LENIENT" in default
    for persona, marker in (
        ("lenient", "STANCE — LENIENT"),
        ("moderate", "STANCE — MODERATE"),
        ("strict", "STANCE — STRICT"),
    ):
        for sole in (False, True):
            content = exp.build_gate_messages(
                **kwargs, agent_is_sole_gate=sole, persona=persona
            )[0]["content"]
            assert marker in content
    with pytest.raises(ValueError):
        exp.build_gate_messages(**kwargs, persona="ruthless")


def test_build_run_summary_deltas_and_metadata():
    state = {
        "experiment_id": "exp-x", "run_number": 7, "status": "completed",
        "area": "MNIST_Digits", "seed": 42, "k_max": 2, "batch_n": 10,
        "test_n": 50, "max_changes": 2, "max_anchors": 8, "epsilon": 0.0,
        "strategy": "random_misalignment", "gate_mode": "agent",
        "gate_model": "openai/gpt-5.5", "drafter_model": "openai/gpt-5.5",
        "judge_models": ["m/a", "m/b"], "concurrency": 4, "dry_run": False,
        "base_version": "v0.1", "current_version": "v0.2",
        "cost_usd_total": 1.5, "started_at": "t0", "finished_at": "t1",
        "cycles": [
            {"k": 0, "status": "baseline", "metrics": {"test": {
                "m/a": {"accuracy": 0.90, "macro_f1": 0.88, "n": 50},
                "system": {"accuracy": 0.95, "macro_f1": 0.94, "n": 50},
            }}},
            {"k": 1, "status": "accepted", "new_version": "v0.2", "metrics": {"test": {
                "m/a": {"accuracy": 0.92, "macro_f1": 0.91, "n": 50},
                "system": {"accuracy": 0.97, "macro_f1": None, "n": 50},
            }}},
        ],
        "benchmark": {"n": 10, "start": {"metrics": {"system": {"macro_f1": 0.9}}},
                      "final": {"metrics": {"system": {"macro_f1": 0.93}}}},
    }
    summary = exp.build_run_summary(state)
    assert summary["run_number"] == 7
    assert summary["policy"] == {
        "base_version": "v0.1", "final_version": "v0.2",
        "accepted_cycles": [1], "n_cycles": 1,
    }
    delta_a = summary["test_metrics"]["m/a"]["delta"]
    assert delta_a["accuracy"] == 0.02 and delta_a["macro_f1"] == 0.03
    # None on either side -> delta None, never a crash.
    assert summary["test_metrics"]["system"]["delta"]["macro_f1"] is None
    assert summary["config"]["gate_mode"] == "agent"
    assert summary["benchmark_system"]["final"]["macro_f1"] == 0.93
    # No benchmark block -> key present but None (holdout likewise).
    state.pop("benchmark")
    assert exp.build_run_summary(state)["benchmark_system"] is None


def test_drafter_prompt_formats_and_targets_subnodes():
    # The {max_changes} placeholder must survive the JSON braces, and the
    # node-targeting directives (Attila 2026-07-06) must be present.
    rendered = exp.drafter_system_prompt(area="MNIST_Digits", max_changes=3)
    assert "at most 3 file changes" in rendered
    assert "MOST SPECIFIC NODE" in rendered
    assert "boundary node" in rendered
    assert "frozen" in rendered  # the root is not a dumping ground
    assert "abstain" in rendered  # decisive-label rule retained
    # Two anchor sets: fix the misaligned without regressing the aligned.
    assert "MISALIGNED" in rendered and "ALIGNED" in rendered
    assert "regress" in rendered


def test_drafter_prompt_area_guidance():
    # MNIST grows confusion-pair boundary nodes; the binary GenAI area grows
    # KG sub-category cue nodes (subtype_of edges) — the drafter must be told
    # the right graph idiom per area (Attila 2026-07-09).
    mnist = exp.drafter_system_prompt(area="MNIST_Digits", max_changes=3)
    assert "MD.boundary.4_vs_9.md" in mnist
    assert "confused_with" in mnist
    genai = exp.drafter_system_prompt(area="Generative_AI", max_changes=3)
    assert "sub-category" in genai
    assert "subtype_of" in genai
    assert "GA." in genai and "MD." not in genai
    # Shared spine either way: root frozen, decisive labels, JSON contract.
    for rendered in (mnist, genai):
        assert "frozen" in rendered and "abstain" in rendered
        assert "at most 3 file changes" in rendered


def test_gate_off_accepts_regardless_of_metric_and_agent():
    # --gate-mode off: the metric is recorded but never enforced; even a
    # metric-failing candidate (and any stray agent verdict) lands.
    veto = {"decision": "skip", "rationale": "leaky", "risk_flags": ["leak"]}
    out = exp.resolve_gate_decision(metric_pass=False, agent=None, gate_off=True)
    assert out["decision"] == "accept" and out["decided_by"] == "gate_off"
    out = exp.resolve_gate_decision(metric_pass=True, agent=veto, gate_off=True)
    assert out["decision"] == "accept" and out["decided_by"] == "gate_off"


def test_parse_gate_response_tolerant_and_strict():
    parsed = exp.parse_gate_response('{"decision": "Accept", "rationale": "ok"}')
    assert parsed["decision"] == "accept"
    fenced = "```json\n{\"decision\": \"skip\", \"rationale\": \"no\"}\n```"
    assert exp.parse_gate_response(fenced)["decision"] == "skip"
    with pytest.raises(ValueError):
        exp.parse_gate_response('{"verdict": "yes"}')


def test_fake_gate_defers_to_metric_rule():
    gate = exp.fake_gate_callable()
    messages = [
        {"role": "system", "content": "x"},
        {"role": "user", "content": json.dumps({"metric_pass": True})},
    ]
    assert exp.parse_gate_response(gate(messages))["decision"] == "accept"
    messages[1]["content"] = json.dumps({"metric_pass": False})
    assert exp.parse_gate_response(gate(messages))["decision"] == "skip"


def test_fake_drafter_emits_one_countable_change(tmp_path):
    base = tmp_path / "v0.1"
    base.mkdir()
    (base / "MD.root.md").write_text("# root\n", encoding="utf-8")
    drafter = exp.fake_drafter_callable(base)
    raw = drafter([{"role": "user", "content": json.dumps({"cycle_k": 3})}])
    files, removed = json.loads(raw)["files"], []
    assert len(files) == 1 and files[0]["change"] == "modified"
    clip = exp.clip_changes(
        {files[0]["path"]: files[0]["content"]}, removed, base_dir=base, max_changes=5
    )
    assert clip["n_applied"] == 1 and not clip["clipped"]


# ---------------------------------------------------------------------------
# State I/O + review recording


def test_state_roundtrip_and_listing(tmp_path):
    state = {
        "experiment_id": exp.mint_experiment_id(),
        "run_number": 1,
        "area": "MNIST_Digits",
        "seed": 42,
        "k_max": 3,
        "status": "running",
        "started_at": exp.utcnow_iso(),
        "gate_mode": "agent_only",
        "gate_persona": "lenient",
        "drafter_model": "openai/gpt-5.5",
        "drafter_context": "text_only",
        "strategy": "random_misalignment",
        "cycles": [{"k": 0}, {"k": 1, "status": "accepted"}],
        # The compact shape the Benchmarks tab consumes.
        "benchmark": {
            "n": 1000,
            "split": "validation",
            "start": {"version": "v0.1",
                      "metrics": {"system": {"macro_f1": 0.9710, "accuracy": 0.9717}}},
            "final": {"version": "v1.1",
                      "metrics": {"system": {"macro_f1": 0.9800, "accuracy": 0.9805}}},
        },
    }
    exp.write_state(tmp_path, state)
    loaded = exp.load_state(tmp_path, state["experiment_id"])
    assert loaded["seed"] == 42 and loaded["updated_at"]
    listing = exp.list_experiments(tmp_path)
    assert len(listing) == 1
    assert listing[0]["accepted"] == 1
    assert listing[0]["cycles_done"] == 1  # k=0 baseline excluded
    # Knobs + compact benchmark readout ride the list payload (Benchmarks tab).
    assert listing[0]["gate_mode"] == "agent_only"
    assert listing[0]["gate_persona"] == "lenient"
    assert listing[0]["drafter_context"] == "text_only"
    assert listing[0]["benchmark"] == {
        "n": 1000,
        "start_version": "v0.1",
        "final_version": "v1.1",
        "start_macro_f1": 0.9710,
        "final_macro_f1": 0.9800,
        "start_accuracy": 0.9717,
        "final_accuracy": 0.9805,
    }
    assert listing[0]["holdout"] is None  # no holdout readout on this run
    assert exp.next_run_number(tmp_path, "MNIST_Digits") == 2
    assert exp.next_run_number(tmp_path, "Generative_AI") == 1


def test_experiment_id_validation():
    good = exp.mint_experiment_id()
    assert exp.validate_experiment_id(good) == good
    for bad in ("", "exp-x", "../etc", "exp-20260101T000000-zzzzzz!"):
        with pytest.raises(ValueError):
            exp.validate_experiment_id(bad)


def test_record_gate_review(tmp_path, monkeypatch):
    from pipeline.experiment import store as exp_store

    monkeypatch.setattr(exp_store, "try_sync_gate_review", lambda **_: False)
    exp_id = exp.mint_experiment_id()
    state = {
        "experiment_id": exp_id,
        "area": "MNIST_Digits",
        "status": "completed",
        "started_at": exp.utcnow_iso(),
        "cycles": [
            {"k": 0},
            {"k": 1, "status": "accepted", "gate": {"decision": "accept"}},
        ],
    }
    exp.write_state(tmp_path, state)
    review = exp.record_gate_review(
        tmp_path, exp_id, 1, verdict="correct", reviewer="attila", comment="good gate"
    )
    assert review["verdict"] == "correct"
    loaded = exp.load_state(tmp_path, exp_id)
    assert loaded["cycles"][1]["review"]["reviewer"] == "attila"
    with pytest.raises(ValueError):
        exp.record_gate_review(tmp_path, exp_id, 1, verdict="meh", reviewer="x")
    with pytest.raises(ValueError):
        # k=0 baseline has no gate to review.
        exp.record_gate_review(tmp_path, exp_id, 0, verdict="correct", reviewer="x")
    with pytest.raises(KeyError):
        exp.record_gate_review(tmp_path, exp_id, 9, verdict="correct", reviewer="x")


# ---------------------------------------------------------------------------
# Review-driven fixes (2026-07-07 adversarial pass)


def test_clip_drops_contradictory_modify_plus_remove(tmp_path):
    base = tmp_path / "v0.1"
    base.mkdir()
    (base / "a.md").write_text("old\n", encoding="utf-8")
    clip = exp.clip_changes({"a.md": "new\n"}, ["a.md"], base_dir=base, max_changes=5)
    # The content change wins; the contradictory removal is dropped, so the
    # result can never trip propose_diff's both-proposed-and-removed error.
    assert clip["n_applied"] == 1
    assert clip["removed"] == []
    assert "a.md" in clip["files"]


def _votes_for(decisions: dict[str, dict[str, str]]) -> list[dict]:
    return [
        {"image_id": image_id, "model_id": model, "label": label}
        for image_id, by_model in decisions.items()
        for model, label in by_model.items()
    ]


def test_gate_comparison_scores_common_subset_only():
    truth = {f"i{n}": "1" for n in range(4)}
    truth.update({f"j{n}": "2" for n in range(4)})
    # Baseline: verdicts on all 8 images, 6 correct.
    before = {i: {"m1": "1", "m2": "1", "m3": "1"} for i in ("i0", "i1", "i2", "i3")}
    before.update({j: {"m1": "2", "m2": "2", "m3": "2"} for j in ("j0", "j1")})
    before.update({j: {"m1": "1", "m2": "1", "m3": "1"} for j in ("j2", "j3")})  # wrong
    # Candidate: the two hard images errored out (no votes at all) and the
    # rest is identical — full-run F1 would LOOK better purely on coverage.
    after = {k: v for k, v in before.items() if k not in ("j2", "j3")}
    comparison = exp.gate_comparison(
        _votes_for(before), _votes_for(after), truth, task=MNIST_MULTICLASS
    )
    assert comparison["n_before"] == 8
    assert comparison["n_after"] == 6
    assert comparison["n_common"] == 6
    # Over the SAME 6 images both sides are identical: no improvement.
    assert comparison["value_before"] == comparison["value_after"]
    assert not exp.metric_passes(comparison["value_before"], comparison["value_after"])


def test_gate_comparison_detects_real_improvement():
    truth = {"i1": "1", "i2": "2", "i3": "1", "i4": "2"}
    before = {
        "i1": {"m1": "1", "m2": "1"}, "i2": {"m1": "1", "m2": "1"},  # i2 wrong
        "i3": {"m1": "1", "m2": "1"}, "i4": {"m1": "2", "m2": "2"},
    }
    after = dict(before)
    after["i2"] = {"m1": "2", "m2": "2"}  # candidate fixes i2
    comparison = exp.gate_comparison(
        _votes_for(before), _votes_for(after), truth, task=MNIST_MULTICLASS
    )
    assert comparison["n_common"] == 4
    assert comparison["value_after"] > comparison["value_before"]
    assert exp.metric_passes(comparison["value_before"], comparison["value_after"])


def test_gate_comparison_empty_common_is_none():
    truth = {"i1": "1"}
    comparison = exp.gate_comparison(
        _votes_for({"i1": {"m1": "1", "m2": "2"}}),  # tie -> no system verdict
        _votes_for({"i1": {"m1": "1", "m2": "1"}}),
        truth, task=MNIST_MULTICLASS,
    )
    assert comparison["n_common"] == 0
    assert comparison["value_before"] is None
    assert not exp.metric_passes(comparison["value_before"], comparison["value_after"])


def test_merge_disk_reviews_preserves_web_review(tmp_path):
    exp_id = exp.mint_experiment_id()
    driver_state = {
        "experiment_id": exp_id,
        "area": "MNIST_Digits",
        "status": "running",
        "started_at": exp.utcnow_iso(),
        "cycles": [{"k": 0}, {"k": 1, "status": "skipped", "gate": {"decision": "skip"}}],
    }
    exp.write_state(tmp_path, driver_state)
    # The web endpoint writes a review into the same file from another process.
    disk = exp.load_state(tmp_path, exp_id)
    disk["cycles"][1]["review"] = {"verdict": "correct", "reviewer": "attila"}
    exp.write_state(tmp_path, disk)
    # The driver's next rewrite must fold it in, not clobber it.
    exp.merge_disk_reviews(tmp_path, driver_state)
    exp.write_state(tmp_path, driver_state)
    final = exp.load_state(tmp_path, exp_id)
    assert final["cycles"][1]["review"]["verdict"] == "correct"


def test_effort_suffixed_gpt_variants_apply_their_effort():
    from pipeline.providers.openai_chat import _effective_effort

    assert _effective_effort("openai/gpt-5.5-low", "high") == "low"
    assert _effective_effort("openai/gpt-5.5-medium", "high") == "medium"
    assert _effective_effort("openai/gpt-5.5", "high") == "high"  # caller wins


def test_multiclass_snapshot_passes_schema_validation(tmp_path):
    # The FNR/micro additions must stay inside the schema contract
    # (additionalProperties: false) — regression guard for --validate-schemas.
    import json

    from pipeline.scoring import _common as scoring_common

    metrics = compute_multiclass_metrics(
        ["1", "2", "1"], ["1", "2", "2"], classes=("1", "2")
    )
    snapshot = {
        "policy_graph_version": "MNIST_Digits.v0.1",
        "task": "mnist_multiclass",
        "classes": ["1", "2"],
        "ground_truth_tier": ["gold"],
        "labelers": [{"labeler_id": "m1", "labeler_type": "llm", "metrics": metrics}],
    }
    schema_path = ROOT / "schemas" / "decision-quality-multiclass.schema.json"
    errors = scoring_common.try_validate(
        snapshot, schema_path, label="decision-quality-multiclass"
    )
    assert errors == [], errors
    assert "fnr" in json.loads(schema_path.read_text())["$defs"]["per_class_metrics"]["properties"]


def test_accept_proposal_allow_branch_from_fixed_baseline(tmp_path):
    # Fixed-k=0 runs: run #2 accepts from v0.1 even after run #1 minted v0.2.
    from pipeline.policy_diff import accept_proposal, propose_diff

    graph = tmp_path / "policy-graph" / "MNIST_Digits"
    (graph / "v0.1").mkdir(parents=True)
    (graph / "v0.1" / "MD.root.md").write_text("# root v0.1\n", encoding="utf-8")
    (graph / "v0.2").mkdir()  # run #1's accepted version — v0.1 is now stale
    (graph / "v0.2" / "MD.root.md").write_text("# root v0.2\n", encoding="utf-8")

    proposal = propose_diff(
        repo_root=tmp_path, run_id="r1", base_version="v0.1", domain="MNIST_Digits",
        proposed_files={"MD.root.md": "# root improved from v0.1\n"},
    )
    # The manual path still refuses stale bases…
    with pytest.raises(ValueError, match="stale"):
        accept_proposal(repo_root=tmp_path, proposal_id=proposal["proposal_id"])
    # …the crank branches from the fixed baseline.
    accepted = accept_proposal(
        repo_root=tmp_path, proposal_id=proposal["proposal_id"], allow_branch=True
    )
    assert accepted["new_version"] == "v0.3"
    new_root = graph / "v0.3" / "MD.root.md"
    assert new_root.read_text(encoding="utf-8") == "# root improved from v0.1\n"


# ---------------------------------------------------------------------------
# SME re-adjudication queue (wave 5): gradient formalism, panel signal,
# end-of-run flagging, cross-run aggregation


def _mis_record(image_id, truth, votes, mis_type="consensus_wrong",
                severity="high", split="dev_golden"):
    return {
        "image_id": image_id,
        "repo_rel_path": f"data/{image_id}.png",
        "sme_truth": truth,
        "split": split,
        "misalignment_type": mis_type,
        "severity": severity,
        "votes": votes,
    }


def test_vote_gradient_formalism():
    import math

    # confident-correct: p = c, |g| = 1 - c (ignorable)
    g = exp.vote_gradient({"label": "7", "confidence": 0.9}, "7")
    assert g["p_true"] == pytest.approx(0.9)
    assert g["magnitude"] == pytest.approx(0.1)
    # confident-wrong: p = 1 - c -> the most informative error
    g = exp.vote_gradient({"label": "1", "confidence": 0.9}, "7")
    assert g["p_true"] == pytest.approx(0.1)
    assert g["magnitude"] == pytest.approx(0.9)
    assert g["loss"] == pytest.approx(-math.log(0.1))
    assert g["hessian"] == pytest.approx(0.09)
    # abstains and missing confidences are excluded (rush.sample_gradient)
    assert exp.vote_gradient({"label": "abstain", "confidence": 0.9}, "7") is None
    assert exp.vote_gradient({"label": "7", "confidence": None}, "7") is None


def test_panel_signal_consensus_confidence_difficulty():
    record = _mis_record("img1", "7", [
        {"label": "1", "confidence": 0.9, "difficulty": "high",
         "is_boundary": True, "is_boundary_between": ["1", "7"]},
        {"label": "1", "confidence": 0.7, "difficulty": "medium"},
        {"label": "7", "confidence": 0.5, "difficulty": "low"},
        {"label": "abstain", "confidence": 0.1, "difficulty": "high"},
    ])
    signal = exp.panel_signal(record)
    assert signal["n_judges"] == 4
    assert signal["majority_label"] == "1"
    assert signal["consensus"]["decisive"] == 3
    assert signal["consensus"]["majority_count"] == 2
    assert signal["consensus"]["fraction"] == pytest.approx(2 / 3, abs=1e-6)
    assert signal["consensus"]["tie"] is False
    # confidence and difficulty averaged across judges (abstain's difficulty
    # still counts toward difficulty; its confidence is not decisive)
    assert signal["avg_confidence"] == pytest.approx((0.9 + 0.7 + 0.5) / 3)
    assert signal["difficulty_score"] == pytest.approx((1 + 0.5 + 0 + 1) / 4)
    # gradient over decisive votes: wrong@0.9 -> .9, wrong@0.7 -> .7, right@0.5 -> .5
    assert signal["gradient"]["n"] == 3
    assert signal["gradient"]["avg_magnitude"] == pytest.approx((0.9 + 0.7 + 0.5) / 3)
    assert signal["gradient"]["max_magnitude"] == pytest.approx(0.9)
    assert signal["any_boundary"] is True
    assert signal["boundary_pairs"] == ["1↔7"]


def test_panel_signal_tie_has_no_majority():
    record = _mis_record("img2", "3", [
        {"label": "3", "confidence": 0.6, "difficulty": "low"},
        {"label": "5", "confidence": 0.6, "difficulty": "low"},
    ])
    signal = exp.panel_signal(record)
    assert signal["majority_label"] is None
    assert signal["consensus"]["tie"] is True


def test_build_readjudication_sources_and_ranking():
    state = {
        "area": "MNIST_Digits",
        "current_version": "v0.2",
        "holdout": None,
        "benchmark": {"n": 2, "final": {"run_id": "run-bench", "version": "v0.2"}},
        "cycles": [
            {"k": 0, "test_run_id": "run-test0",
             "generator_before": "MNIST_Digits.v0.1", "status": "baseline"},
            {"k": 1, "train_run_id": "run-train1",
             "generator_before": "MNIST_Digits.v0.1", "status": "skipped"},
            {"k": 2, "train_run_id": "run-train2", "candidate_run_id": "run-cand2",
             "generator_before": "MNIST_Digits.v0.1",
             "generator_after": "MNIST_Digits.v0.2", "status": "accepted"},
        ],
    }
    by_run = {
        "run-train1": [
            _mis_record("train_a", "7",
                        [{"label": "1", "confidence": 0.9, "difficulty": "high"}]),
            _mis_record("train_ok", "3",
                        [{"label": "3", "confidence": 0.9, "difficulty": "low"}],
                        mis_type="all_agree", severity="low"),
        ],
        "run-train2": [
            _mis_record("train_b", "8", [
                {"label": "6", "confidence": 0.4, "difficulty": "medium"},
                {"label": "8", "confidence": 0.4, "difficulty": "medium"},
            ], mis_type="model_vs_sme", severity="medium"),
        ],
        "run-cand2": [
            _mis_record("test_c", "4",
                        [{"label": "9", "confidence": 0.95, "difficulty": "high"}]),
        ],
        "run-bench": [
            _mis_record("bench_d", "2",
                        [{"label": "abstain", "confidence": None, "difficulty": "high"}],
                        mis_type="model_vs_sme", severity="medium", split="validation"),
        ],
    }
    block = exp.build_readjudication(
        state,
        load_misalignment=lambda run_id: by_run.get(run_id, []),
        sha_by_image={"train_a": "sha-a"},
    )
    assert block["n_flagged"] == 4  # the all_agree row is not queued
    kinds = {(s["kind"], s["run_id"]) for s in block["sources"]}
    # test evidence = the ACCEPTED candidate eval, not the stale k=0 baseline
    assert ("test", "run-cand2") in kinds
    assert ("test", "run-test0") not in kinds
    assert ("benchmark", "run-bench") in kinds
    items = {i["image_id"]: i for i in block["items"]}
    assert items["train_a"]["sha256"] == "sha-a"
    assert items["train_a"]["source"]["k"] == 1
    assert items["test_c"]["source"]["policy"] == "MNIST_Digits.v0.2"
    assert items["bench_d"]["source"]["policy"] == "MNIST_Digits.v0.2"
    # Four-tier importance rank: the confident-unanimous-wrong panels (T1)
    # lead, most-confident first (test_c @0.95 > train_a @0.9); the split
    # panel (train_b) next; the no-signal all-abstain panel (bench_d) last.
    assert [i["image_id"] for i in block["items"]] == [
        "test_c", "train_a", "train_b", "bench_d",
    ]
    # T1 = misaligned (majority disagrees with SME) + high LLM-consensus.
    assert items["test_c"]["importance"]["tier"] == 1
    assert items["test_c"]["sme_agreement"]["fraction"] == 0.0
    assert items["test_c"]["consensus"]["fraction"] == 1.0


def test_build_readjudication_falls_back_to_baseline_test():
    state = {
        "area": "MNIST_Digits", "current_version": "v0.1",
        "holdout": None, "benchmark": None,
        "cycles": [{"k": 0, "test_run_id": "run-test0",
                    "generator_before": "MNIST_Digits.v0.1", "status": "baseline"}],
    }
    block = exp.build_readjudication(
        state,
        load_misalignment=lambda run_id: [
            _mis_record("t", "1", [{"label": "2", "confidence": 0.8, "difficulty": "low"}]),
        ],
    )
    assert block["sources"][0]["kind"] == "test"
    assert block["sources"][0]["run_id"] == "run-test0"
    assert block["n_flagged"] == 1
    assert block["items"][0]["sha256"] is None


def _write_flagging_state(tmp_path, run_number, *, dry, conf,
                          image_id="img_x", sha="sha-x", area="MNIST_Digits"):
    state = {
        "experiment_id": exp.mint_experiment_id(),
        "run_number": run_number,
        "area": area,
        "seed": run_number,
        "dry_run": dry,
        "status": "completed",
        "started_at": exp.utcnow_iso(),
        "cycles": [],
        "readjudication": {"items": [{
            "image_id": image_id, "sha256": sha, "repo_rel_path": "p.png",
            "split": "dev_golden", "sme_truth": "7",
            "misalignment_type": "consensus_wrong", "severity": "high",
            "source": {"kind": "test", "k": 2, "run_id": f"r{run_number}",
                       "policy": "MNIST_Digits.v0.2"},
            "n_judges": 1, "majority_label": "1",
            "consensus": {"decisive": 1, "majority_count": 1,
                          "fraction": 1.0, "tie": False},
            "avg_confidence": conf, "difficulty_score": 0.5,
            "gradient": {"n": 1, "avg_magnitude": conf, "max_magnitude": conf,
                         "avg_hessian": 0.2, "avg_loss": 1.0},
            "any_boundary": False, "boundary_pairs": [], "votes": [],
        }]},
    }
    exp.write_state(tmp_path, state)


def test_aggregate_readjudication_dedupes_across_runs_and_skips_dry(tmp_path):
    _write_flagging_state(tmp_path, 1, dry=False, conf=0.6)
    _write_flagging_state(tmp_path, 2, dry=False, conf=0.8)
    _write_flagging_state(tmp_path, 3, dry=True, conf=0.9)  # dry -> excluded
    _write_flagging_state(tmp_path, 4, dry=False, conf=0.9,
                          image_id="other", sha="sha-y", area="Generative_AI")
    queue = exp.aggregate_readjudication(tmp_path, area="MNIST_Digits")
    assert queue["n_items"] == 1
    item = queue["items"][0]
    assert item["run_numbers"] == [1, 2]
    assert item["n_runs"] == 2
    assert item["agg"]["avg_confidence"] == pytest.approx(0.7)
    assert item["latest"]["run_number"] == 2
    with_dry = exp.aggregate_readjudication(
        tmp_path, area="MNIST_Digits", include_dry=True
    )
    assert with_dry["items"][0]["n_runs"] == 3
    everything = exp.aggregate_readjudication(tmp_path)
    assert everything["n_items"] == 2


def test_mnist_prompts_never_recommend_abstain():
    # Attila's standing rule: judges must always return a label; confidence
    # [0,1] + difficulty carry the uncertainty. The schema stays tolerant
    # (parser falls back to abstain on malformed replies) but the PROMPT
    # must never suggest it.
    from pipeline.providers.ontology import (
        MNIST_SYSTEM_PROMPT, MNIST_USER_INSTRUCTIONS,
    )
    lowered = MNIST_SYSTEM_PROMPT.lower()
    assert "abstain is legitimate" not in lowered
    assert "never abstain" in lowered
    assert "always return exactly one digit" in lowered
    assert "or \"abstain\"" not in MNIST_USER_INSTRUCTIONS.lower()


def test_build_readjudication_clears_stale_flag_on_later_all_agree():
    # sample_train_batch re-uses train ids once the pool runs short: an image
    # misaligned at k=1 but re-judged all_agree at k=3 must NOT stay queued.
    state = {
        "area": "MNIST_Digits", "current_version": "v0.1",
        "holdout": None, "benchmark": None,
        "cycles": [
            {"k": 1, "train_run_id": "run-t1",
             "generator_before": "MNIST_Digits.v0.1", "status": "skipped"},
            {"k": 3, "train_run_id": "run-t3",
             "generator_before": "MNIST_Digits.v0.1", "status": "skipped"},
        ],
    }
    by_run = {
        "run-t1": [_mis_record("img_reused", "7",
                               [{"label": "1", "confidence": 0.9, "difficulty": "high"}])],
        "run-t3": [_mis_record("img_reused", "7",
                               [{"label": "7", "confidence": 0.9, "difficulty": "low"}],
                               mis_type="all_agree", severity="low")],
    }
    block = exp.build_readjudication(
        state, load_misalignment=lambda run_id: by_run.get(run_id, []),
    )
    assert block["n_flagged"] == 0
    assert block["items"] == []


def test_build_readjudication_holdout_start_leg_fallback():
    # A run stopped between the paid start leg and the final leg still
    # surfaces the start leg's misalignments (latest available evidence).
    state = {
        "area": "MNIST_Digits", "current_version": "v0.1",
        "holdout": {"n": 1, "start": {"run_id": "run-hold-start", "version": "v0.1"}},
        "benchmark": None,
        "cycles": [],
    }
    block = exp.build_readjudication(
        state,
        load_misalignment=lambda run_id: [
            _mis_record("hold_x", "5",
                        [{"label": "6", "confidence": 0.8, "difficulty": "medium"}],
                        split="holdout"),
        ] if run_id == "run-hold-start" else [],
    )
    assert [(s["kind"], s["run_id"]) for s in block["sources"]] == [
        ("holdout", "run-hold-start"),
    ]
    assert block["items"][0]["source"]["policy"] == "MNIST_Digits.v0.1"


def test_policy_quotes_hard_capped_at_schema_limit():
    # A model ignoring the soft <=240-char instruction must not produce a
    # vote that fails llm-output schema validation (maxLength 600).
    from pipeline.providers.base import coerce_label_fields

    fields = coerce_label_fields({
        "label": "7", "l2_label": "MD.digit.7",
        "justification": "long enough to pass the sanity floor",
        "confidence": 0.9, "difficulty": "low", "is_boundary": False,
        "policy_citations": ["MD.digit.7"],
        "policy_quotes": ["q" * 700, "short"],
    })
    assert len(fields["policy_quotes"][0]) == 600
    assert fields["policy_quotes"][1] == "short"


# ---------------------------------------------------------------------------
# Wave 6: named versions (vRUN.k), top-gradient anchors, multimodal drafter


def test_accept_proposal_explicit_version_name(tmp_path):
    from pipeline.policy_diff import accept_proposal, propose_diff

    graph = tmp_path / "policy-graph" / "MNIST_Digits"
    (graph / "v0.1").mkdir(parents=True)
    (graph / "v0.1" / "MD.root.md").write_text("# root v0.1\n", encoding="utf-8")

    proposal = propose_diff(
        repo_root=tmp_path, run_id="r1", base_version="v0.1", domain="MNIST_Digits",
        proposed_files={"MD.root.md": "# improved\n"},
    )
    accepted = accept_proposal(
        repo_root=tmp_path, proposal_id=proposal["proposal_id"],
        allow_branch=True, new_version="v5.3",
    )
    assert accepted["new_version"] == "v5.3"
    assert (graph / "v5.3" / "MD.root.md").exists()

    # Name collision falls back to the global mint instead of failing the
    # paid accept; bad formats are rejected outright.
    proposal2 = propose_diff(
        repo_root=tmp_path, run_id="r2", base_version="v0.1", domain="MNIST_Digits",
        proposed_files={"MD.root.md": "# improved again\n"},
    )
    with pytest.raises(ValueError, match="vMAJOR.MINOR"):
        accept_proposal(repo_root=tmp_path, proposal_id=proposal2["proposal_id"],
                        allow_branch=True, new_version="run5-k3")
    accepted2 = accept_proposal(
        repo_root=tmp_path, proposal_id=proposal2["proposal_id"],
        allow_branch=True, new_version="v5.3",
    )
    assert accepted2["new_version"] == "v5.4"  # max(v0.1, v5.3) minor + 1


def test_select_anchors_top_gradient_ranks_confident_wrong_first():
    records = [
        # right at 0.9 -> |g| = 0.1
        _mis_record("img_low", "7",
                    [{"label": "7", "confidence": 0.9, "difficulty": "low"}],
                    mis_type="model_vs_sme", severity="medium"),
        # wrong at 0.95 -> |g| = 0.95
        _mis_record("img_high", "7",
                    [{"label": "1", "confidence": 0.95, "difficulty": "high"}]),
        # wrong at 0.6 -> |g| = 0.6
        _mis_record("img_mid", "7",
                    [{"label": "1", "confidence": 0.6, "difficulty": "medium"}]),
        # all-abstain -> no gradient signal at all -> ranked first
        _mis_record("img_nosignal", "7",
                    [{"label": "abstain", "confidence": None, "difficulty": "high"}],
                    mis_type="model_vs_sme", severity="medium"),
        _mis_record("img_agree", "7",
                    [{"label": "7", "confidence": 0.9, "difficulty": "low"}],
                    mis_type="all_agree", severity="low"),
    ]
    anchors = exp.select_anchors(records, seed=1, k=1, max_anchors=3,
                                 strategy="top_gradient")
    assert [a["image_id"] for a in anchors] == ["img_nosignal", "img_high", "img_mid"]
    with pytest.raises(ValueError, match="unknown anchor strategy"):
        exp.select_anchors(records, seed=1, k=1, max_anchors=3, strategy="s9")


def test_build_drafter_messages_attaches_images_per_provider():
    anchors = [{"image_id": "img1", "sme_truth": "7", "misalignment_type": "consensus_wrong",
                "severity": "high", "votes": []}]
    images = [{"image_id": "img1", "sme_truth": "7",
               "mime_type": "image/jpeg", "b64": "QUJD"}]
    base = dict(policy_markdown="# policy", base_version="v0.1",
                area="MNIST_Digits", anchors=anchors, max_changes=1, k=2)

    plain = exp.build_drafter_messages(**base)
    assert isinstance(plain[1]["content"], str)

    openai_msgs = exp.build_drafter_messages(**base, anchor_images=images,
                                             provider="openai")
    parts = openai_msgs[1]["content"]
    assert isinstance(parts, list)
    image_parts = [p for p in parts if p.get("type") == "image_url"]
    assert image_parts[0]["image_url"]["url"] == "data:image/jpeg;base64,QUJD"
    assert any(p.get("type") == "text" and "img1" in p.get("text", "")
               for p in parts)

    anthropic_msgs = exp.build_drafter_messages(**base, anchor_images=images,
                                                provider="anthropic")
    blocks = anthropic_msgs[1]["content"]
    image_blocks = [p for p in blocks if p.get("type") == "image"]
    assert image_blocks[0]["source"] == {
        "type": "base64", "media_type": "image/jpeg", "data": "QUJD",
    }
    # System stays a plain string either way (anthropic flattens it to text).
    assert isinstance(openai_msgs[0]["content"], str)


# ---------------------------------------------------------------------------
# Aligned anchors (Attila 2026-07-07): learn from correct classifications too


def test_select_aligned_anchors_picks_only_correct_and_respects_cap():
    records = ([_mis(f"ok{i}", "all_agree") for i in range(15)]
               + [_mis(f"bad{i}") for i in range(5)])
    aligned = exp.select_aligned_anchors(records, seed=5, k=1, max_aligned=8)
    assert len(aligned) == 8
    assert all(a["misalignment_type"] == "all_agree" for a in aligned)
    # seeded + reproducible; varies by cycle k
    assert aligned == exp.select_aligned_anchors(records, seed=5, k=1, max_aligned=8)
    assert aligned != exp.select_aligned_anchors(records, seed=5, k=2, max_aligned=8)


def test_select_aligned_anchors_respects_train_ids_and_disable():
    records = [_mis("in1", "all_agree"), _mis("out1", "all_agree"),
               _mis("in2", "consensus_wrong")]
    aligned = exp.select_aligned_anchors(records, seed=1, k=1, max_aligned=5,
                                         train_ids=["in1", "in2"])
    assert [a["image_id"] for a in aligned] == ["in1"]  # out1 excluded, in2 not all_agree
    # max_aligned == 0 disables the aligned side entirely
    assert exp.select_aligned_anchors(records, seed=1, k=1, max_aligned=0) == []


def test_build_drafter_messages_carries_aligned_anchors_and_images():
    mis = [{"image_id": "wrong1", "sme_truth": "4", "misalignment_type": "consensus_wrong",
            "severity": "high", "votes": []}]
    aligned = [{"image_id": "right1", "sme_truth": "8", "misalignment_type": "all_agree",
                "severity": "low", "votes": []}]
    mis_imgs = [{"image_id": "wrong1", "sme_truth": "4", "mime_type": "image/jpeg", "b64": "QUJD"}]
    aligned_imgs = [{"image_id": "right1", "sme_truth": "8", "mime_type": "image/jpeg", "b64": "WFla"}]

    # Text-only path still carries aligned_samples in the JSON payload.
    plain = exp.build_drafter_messages(
        policy_markdown="# p", base_version="v0.1", area="MNIST_Digits",
        anchors=mis, aligned_anchors=aligned, max_changes=1, k=2)
    payload = json.loads(plain[1]["content"])
    assert [s["image_id"] for s in payload["misaligned_samples"]] == ["wrong1"]
    assert [s["image_id"] for s in payload["aligned_samples"]] == ["right1"]

    # Multimodal: BOTH image groups attach, each labeled, aligned marked correct.
    msgs = exp.build_drafter_messages(
        policy_markdown="# p", base_version="v0.1", area="MNIST_Digits",
        anchors=mis, aligned_anchors=aligned, max_changes=1, k=2,
        anchor_images=mis_imgs, aligned_images=aligned_imgs, provider="openai")
    parts = msgs[1]["content"]
    urls = [p["image_url"]["url"] for p in parts if p.get("type") == "image_url"]
    assert urls == ["data:image/jpeg;base64,QUJD", "data:image/jpeg;base64,WFla"]
    texts = " ".join(p.get("text", "") for p in parts if p.get("type") == "text")
    assert "MISALIGNED" in texts and "ALIGNED" in texts and "CORRECTLY" in texts

    # Aligned images alone (misaligned images suppressed) still trigger multimodal.
    only_aligned = exp.build_drafter_messages(
        policy_markdown="# p", base_version="v0.1", area="MNIST_Digits",
        anchors=mis, aligned_anchors=aligned, max_changes=1, k=2,
        aligned_images=aligned_imgs, provider="openai")
    assert isinstance(only_aligned[1]["content"], list)


# ---------------------------------------------------------------------------
# Wave 7: the importance formalism — SME alignment vs LLM consensus, 4 tiers


def _panel(truth, votes):
    return _mis_record("x", truth, votes, mis_type="model_vs_sme")


def test_panel_signal_splits_sme_alignment_from_llm_consensus():
    # 4 judges, SME truth = 7. Three say "1" (wrong, agree with each other),
    # one says "7" (right). LLM consensus is HIGH (3/4 on "1"); SME agreement
    # is LOW (1/4 match the human). The two signals must not be conflated.
    sig = exp.panel_signal(_panel("7", [
        {"label": "1", "confidence": 0.9, "difficulty": "high"},
        {"label": "1", "confidence": 0.9, "difficulty": "high"},
        {"label": "1", "confidence": 0.8, "difficulty": "medium"},
        {"label": "7", "confidence": 0.6, "difficulty": "low"},
    ]))
    assert sig["consensus"]["fraction"] == 0.75          # LLM<->LLM
    assert sig["sme_agreement"]["fraction"] == 0.25      # LLM<->SME
    assert sig["sme_agreement"]["n_agree"] == 1
    assert sig["majority_aligned"] is False              # plurality "1" != 7
    assert sig["importance"]["tier"] == 1                # misaligned + high consensus


def test_importance_four_tier_ordering():
    def score(sme_frac, consensus, aligned):
        return exp.importance_scores(
            sme_fraction=sme_frac, consensus_fraction=consensus,
            majority_aligned=aligned, mean_grad=0.5, boundary_rate=0.0,
        )
    t1 = score(0.0, 1.0, False)   # misaligned + high consensus
    t2 = score(0.0, 0.34, False)  # misaligned + low consensus
    t3 = score(1.0, 0.34, True)   # aligned + low consensus
    t4 = score(1.0, 1.0, True)    # aligned + high consensus (ideal)
    assert (t1["tier"], t2["tier"], t3["tier"], t4["tier"]) == (1, 2, 3, 4)
    # The continuous base score respects the same ordering.
    assert t1["base"] > t2["base"] > t3["base"] > t4["base"]


def test_importance_boundary_and_confidence_amplify():
    plain = exp.importance_scores(sme_fraction=0.0, consensus_fraction=1.0,
                                  majority_aligned=False, mean_grad=0.0,
                                  boundary_rate=0.0)
    boundaried = exp.importance_scores(sme_fraction=0.0, consensus_fraction=1.0,
                                       majority_aligned=False, mean_grad=0.9,
                                       boundary_rate=1.0)
    assert boundaried["anchor"] > plain["anchor"]        # both amplifiers raise it
    assert boundaried["tier"] == plain["tier"] == 1      # tier is amplifier-blind


def test_human_confidence_fades_readjudication():
    assert exp.human_confidence(0) == 0.0
    assert exp.human_confidence(1) == pytest.approx(0.166667, abs=1e-5)
    assert exp.human_confidence(2) == pytest.approx(0.545455, abs=1e-5)
    assert exp.human_confidence(3) > exp.human_confidence(2)
    # A re-confirmed label (m=3) has far lower re-adjudication priority than a
    # default single-confirmation one, with identical panel evidence.
    common = dict(sme_fraction=0.0, consensus_fraction=1.0, majority_aligned=False,
                  mean_grad=0.9, boundary_rate=0.0)
    fresh = exp.importance_scores(**common, sme_confirmations=1)
    confirmed = exp.importance_scores(**common, sme_confirmations=3)
    assert confirmed["readjudication"] < fresh["readjudication"]
    assert confirmed["anchor"] == fresh["anchor"]        # policy value is unchanged


def test_select_anchors_top_importance_leads_with_confident_unanimous_wrong():
    records = [
        # aligned, unanimous (T4) — least important
        _mis_record("aligned", "7", [
            {"label": "7", "confidence": 0.9, "difficulty": "low"},
            {"label": "7", "confidence": 0.9, "difficulty": "low"},
        ], mis_type="model_vs_sme"),
        # misaligned, split (T2)
        _mis_record("split", "7", [
            {"label": "1", "confidence": 0.6, "difficulty": "medium"},
            {"label": "3", "confidence": 0.6, "difficulty": "medium"},
        ]),
        # misaligned, unanimous & confident (T1) — most important
        _mis_record("systematic", "7", [
            {"label": "1", "confidence": 0.95, "difficulty": "high"},
            {"label": "1", "confidence": 0.95, "difficulty": "high"},
        ]),
    ]
    anchors = exp.select_anchors(records, seed=1, k=1, max_anchors=3,
                                 strategy="top_importance")
    assert anchors[0]["image_id"] == "systematic"


def test_importance_tie_is_low_consensus_not_high():
    # An even split (1-1, 2-2) has consensus fraction exactly 0.5 — a TIE, not
    # consensus. It must land in the low-consensus tier (T2 misaligned / T3
    # aligned), never T1/T4. The continuous score is unaffected (only the badge).
    tie_misaligned = exp.importance_scores(
        sme_fraction=0.0, consensus_fraction=0.5, majority_aligned=False,
        mean_grad=0.5, boundary_rate=0.0)
    assert tie_misaligned["tier"] == 2
    real_consensus = exp.importance_scores(
        sme_fraction=0.0, consensus_fraction=0.75, majority_aligned=False,
        mean_grad=0.5, boundary_rate=0.0)
    assert real_consensus["tier"] == 1
    # A 4-judge 2-2 split, both pairs wrong different ways, is tier 2 via panel_signal.
    sig = exp.panel_signal({"sme_truth": "7", "votes": [
        {"label": "1", "confidence": 0.9}, {"label": "1", "confidence": 0.9},
        {"label": "3", "confidence": 0.9}, {"label": "3", "confidence": 0.9}]})
    assert sig["consensus"]["fraction"] == 0.5 and sig["consensus"]["tie"] is True
    assert sig["importance"]["tier"] == 2


# ---------------------------------------------------------------------------
# SME re-adjudication actions (confirm fades, overturn re-scores)


def _seed_queue_experiment(tmp_path, votes, truth="7"):
    """Write one completed experiment.json whose queue holds a single item."""
    exp_dir = tmp_path / "data" / "experiments" / "exp-20260101T000000-aaaaaa"
    exp_dir.mkdir(parents=True)
    sig = exp.panel_signal({"sme_truth": truth, "votes": votes})
    item = {"image_id": "img1", "sha256": "sha-1", "repo_rel_path": "p.png",
            "split": "dev_golden", "sme_truth": truth,
            "misalignment_type": "consensus_wrong", "severity": "high",
            "source": {"kind": "test", "k": 0, "run_id": "r0", "policy": "MNIST_Digits.v0.1"},
            **sig, "votes": votes}
    state = {"experiment_id": "exp-20260101T000000-aaaaaa", "run_number": 1,
             "area": "MNIST_Digits", "seed": 1, "status": "completed",
             "started_at": "2026-01-01T00:00:00Z", "dry_run": False,
             "cycles": [], "readjudication": {"items": [item]}}
    (exp_dir / "experiment.json").write_text(json.dumps(state), encoding="utf-8")


def test_record_and_fold_adjudications(tmp_path):
    exp.record_adjudication(tmp_path, area="MNIST_Digits", key="k1", image_id="i1",
                            verdict="confirm", prior_truth="7")
    exp.record_adjudication(tmp_path, area="MNIST_Digits", key="k1", image_id="i1",
                            verdict="confirm")
    by_key = exp.load_adjudications(tmp_path, area="MNIST_Digits")
    eff, conf, res = exp._fold_reviews(by_key["k1"], "7")
    assert (eff, conf, res) == ("7", 3, "confirmed")  # seed + 2 confirms
    # An overturn resets to the new label with the overturning SME as sole confirmer.
    exp.record_adjudication(tmp_path, area="MNIST_Digits", key="k1", image_id="i1",
                            verdict="overturn", new_label="1")
    by_key = exp.load_adjudications(tmp_path, area="MNIST_Digits")
    eff, conf, res = exp._fold_reviews(by_key["k1"], "7")
    assert (eff, conf, res) == ("1", 1, "overturned")
    with pytest.raises(ValueError):
        exp.record_adjudication(tmp_path, area="MNIST_Digits", key="k1", image_id="i1",
                                verdict="overturn")  # missing new_label
    with pytest.raises(ValueError):
        exp.record_adjudication(tmp_path, area="MNIST_Digits", key="k1", image_id="i1",
                                verdict="bogus")


def test_confirm_fades_and_overturn_reclassifies(tmp_path):
    # 4 judges all say "1"; SME truth "7" -> T1 (unanimous & wrong).
    votes = [{"model": f"m{i}", "label": "1", "confidence": 0.9, "difficulty": "high",
              "is_boundary": False} for i in range(4)]
    _seed_queue_experiment(tmp_path, votes, truth="7")

    q0 = exp.aggregate_readjudication(tmp_path, area="MNIST_Digits")
    it0 = q0["items"][0]
    assert q0["n_open"] == 1
    assert it0["review"]["resolution"] == "open"
    base_imp = it0["agg"]["effective_importance"]
    assert it0["agg"]["worst_tier"] == 1

    # CONFIRM: the golden label holds, human confidence rises -> importance fades.
    exp.record_adjudication(tmp_path, area="MNIST_Digits", key="sha-1", image_id="img1",
                            verdict="confirm", prior_truth="7")
    q1 = exp.aggregate_readjudication(tmp_path, area="MNIST_Digits")
    it1 = q1["items"][0]
    assert it1["review"]["resolution"] == "confirmed"
    assert it1["review"]["resolved"] is True      # m=2 (seed + confirm) -> ≥2 SMEs agree
    assert q1["n_open"] == 0
    assert it1["agg"]["effective_importance"] < base_imp
    assert it1["agg"]["human_confidence"] > 0.5   # m=2 -> 0.545

    # OVERTURN to "1" (what the panel confidently said): the item is re-scored
    # against the new truth and becomes T4 (aligned & unanimous) — it drops.
    exp.record_adjudication(tmp_path, area="MNIST_Digits", key="sha-1", image_id="img1",
                            verdict="overturn", prior_truth="7", new_label="1")
    q2 = exp.aggregate_readjudication(tmp_path, area="MNIST_Digits")
    it2 = q2["items"][0]
    assert it2["review"]["resolution"] == "overturned"
    assert it2["review"]["effective_label"] == "1"
    assert it2["review"]["overturned_from"] == "7"
    # A lone overturn is one SME's new opinion (m=1) — it stays OPEN for a 2nd.
    assert it2["review"]["resolved"] is False
    assert q2["n_open"] == 1
    assert it2["agg"]["recomputed"]["tier"] == 4          # aligned + high consensus
    assert it2["agg"]["recomputed"]["sme_fraction"] == 1.0
    assert it2["agg"]["effective_importance"] < base_imp  # dropped hard


def test_still_misaligned_overturn_stays_open_and_top_priority(tmp_path):
    # The review's finding: overturning to a label NO judge gave leaves the
    # panel still misaligned (T1). It must stay OPEN and high-priority — a
    # lone overturn is not a resolution, and the item still needs a 2nd SME.
    votes = [{"model": f"m{i}", "label": "9", "confidence": 0.9, "difficulty": "high",
              "is_boundary": False} for i in range(4)]
    _seed_queue_experiment(tmp_path, votes, truth="8")
    exp.record_adjudication(tmp_path, area="MNIST_Digits", key="sha-1", image_id="img1",
                            verdict="overturn", prior_truth="8", new_label="3")
    q = exp.aggregate_readjudication(tmp_path, area="MNIST_Digits")
    it = q["items"][0]
    assert it["review"]["resolution"] == "overturned"
    assert it["review"]["effective_label"] == "3"
    assert it["review"]["resolved"] is False       # only 1 SME so far
    assert q["n_open"] == 1                          # still needs a human
    assert it["agg"]["recomputed"]["tier"] == 1     # panel still confidently wrong vs "3"
    assert it["agg"]["recomputed"]["sme_fraction"] == 0.0
    # A second SME confirming the overturn (m=2) finally resolves it.
    exp.record_adjudication(tmp_path, area="MNIST_Digits", key="sha-1", image_id="img1",
                            verdict="confirm", prior_truth="3")
    q2 = exp.aggregate_readjudication(tmp_path, area="MNIST_Digits")
    assert q2["items"][0]["review"]["resolved"] is True
    assert q2["n_open"] == 0
