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
    rendered = exp.DRAFTER_SYSTEM_PROMPT.format(max_changes=3)
    assert "at most 3 file changes" in rendered
    assert "MOST SPECIFIC NODE" in rendered
    assert "boundary node" in rendered
    assert "frozen" in rendered  # the root is not a dumping ground
    assert "abstain" in rendered  # decisive-label rule retained


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
        "cycles": [{"k": 0}, {"k": 1, "status": "accepted"}],
    }
    exp.write_state(tmp_path, state)
    loaded = exp.load_state(tmp_path, state["experiment_id"])
    assert loaded["seed"] == 42 and loaded["updated_at"]
    listing = exp.list_experiments(tmp_path)
    assert len(listing) == 1
    assert listing[0]["accepted"] == 1
    assert listing[0]["cycles_done"] == 1  # k=0 baseline excluded
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
