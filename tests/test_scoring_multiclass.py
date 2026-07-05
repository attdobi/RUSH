"""Offline tests for multiclass scoring (MNIST 0-9 generalization).

No network. No real LLM. Small synthetic label/manifest data.
Run with: ``pytest tests/test_scoring_multiclass.py -v``

Covers the multiclass metrics contract documented in
:mod:`pipeline.scoring.decision_quality_multiclass`:
    * perfect / all-wrong extremes
    * a hand-computed realistic 10-class mix (one class + macro-F1)
    * abstain exclusion + counting
    * unknown-label handling (treated as wrong, still in confusion matrix)
    * confusion-matrix shape + totals
    * task registry + end-to-end multiclass run_scoring dispatch
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline.scoring import decision_quality_multiclass as dqmc  # noqa: E402
from pipeline.scoring import tasks as tasks_mod  # noqa: E402
from pipeline.scoring import run_scoring  # noqa: E402  (the callable)
from pipeline.scoring._common import ABSTAIN  # noqa: E402


CLASSES = tuple(str(d) for d in range(10))


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r, sort_keys=True) for r in rows) + "\n")


# ---------------------------------------------------------------------------
# task registry
# ---------------------------------------------------------------------------

def test_task_registry_binary_and_multiclass():
    assert tasks_mod.GENAI_BINARY.is_binary
    assert not tasks_mod.GENAI_BINARY.is_multiclass
    assert tasks_mod.GENAI_BINARY.positive_class == "gen_ai"

    assert tasks_mod.MNIST_MULTICLASS.is_multiclass
    assert tasks_mod.MNIST_MULTICLASS.positive_class is None
    assert tasks_mod.MNIST_MULTICLASS.classes == CLASSES
    assert tasks_mod.DEFAULT_TASK is tasks_mod.GENAI_BINARY

    assert tasks_mod.get_task("mnist_multiclass") is tasks_mod.MNIST_MULTICLASS
    with pytest.raises(KeyError):
        tasks_mod.get_task("does_not_exist")


# ---------------------------------------------------------------------------
# compute_multiclass_metrics: extremes
# ---------------------------------------------------------------------------

def test_perfect_prediction():
    truths = list(CLASSES)
    preds = list(CLASSES)
    m = dqmc.compute_multiclass_metrics(preds, truths, classes=CLASSES)
    assert m["n"] == 10
    assert m["n_abstained"] == 0
    assert m["accuracy"] == 1.0
    assert m["macro_precision"] == 1.0
    assert m["macro_recall"] == 1.0
    assert m["macro_f1"] == 1.0
    for cls in CLASSES:
        pc = m["per_class"][cls]
        assert pc["precision"] == 1.0
        assert pc["recall"] == 1.0
        assert pc["f1"] == 1.0
        assert pc["fpr"] == 0.0
        assert pc["support"] == 1


def test_all_wrong_prediction():
    truths = list(CLASSES)
    # shift every prediction by one → nothing is correct
    preds = [CLASSES[(i + 1) % 10] for i in range(10)]
    m = dqmc.compute_multiclass_metrics(preds, truths, classes=CLASSES)
    assert m["n"] == 10
    assert m["accuracy"] == 0.0
    # every class: tp=0 → precision/recall/f1 defined as 0.0 (tp+fp>0, tp+fn>0)
    for cls in CLASSES:
        pc = m["per_class"][cls]
        assert pc["precision"] == 0.0
        assert pc["recall"] == 0.0
        assert pc["f1"] is None  # precision==0 and recall==0 → f1 None
        assert pc["support"] == 1
    assert m["macro_precision"] == 0.0
    assert m["macro_recall"] == 0.0
    assert m["macro_f1"] is None  # all per-class f1 undefined


# ---------------------------------------------------------------------------
# compute_multiclass_metrics: realistic hand-computed mix
# ---------------------------------------------------------------------------

def test_realistic_mix_hand_computed():
    # Confusion only between classes '0' and '1'; classes '2'..'9' perfect.
    #   truth '0' x3 -> pred 0,0,1   (2 correct, 1 leaked to class 1)
    #   truth '1' x2 -> pred 1,0     (1 correct, 1 leaked to class 0)
    truths = ["0", "0", "0", "1", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
    preds = ["0", "0", "1", "1", "0", "2", "3", "4", "5", "6", "7", "8", "9"]
    m = dqmc.compute_multiclass_metrics(preds, truths, classes=CLASSES)

    assert m["n"] == 13
    assert m["n_abstained"] == 0
    # 11 of 13 correct
    assert m["accuracy"] == round(11 / 13, 6)

    # Class '0': tp=2, fp=1 (truth1->pred0), fn=1 (truth0->pred1)
    c0 = m["per_class"]["0"]
    assert c0["precision"] == round(2 / 3, 6)
    assert c0["recall"] == round(2 / 3, 6)
    assert c0["f1"] == round(2 / 3, 6)
    assert c0["support"] == 3

    # Class '1': tp=1, fp=1, fn=1
    c1 = m["per_class"]["1"]
    assert c1["precision"] == 0.5
    assert c1["recall"] == 0.5
    assert c1["f1"] == 0.5
    assert c1["support"] == 2

    # Classes 2..9 are perfect
    for cls in CLASSES[2:]:
        assert m["per_class"][cls]["f1"] == 1.0

    # macro-F1 = (f1_0 + f1_1 + 8*1) / 10 = (2/3 + 1/2 + 8) / 10
    expected_macro_f1 = round((2 / 3 + 1 / 2 + 8) / 10, 6)
    assert m["macro_f1"] == expected_macro_f1
    # precision==recall for classes 0/1 here → macro precision/recall match
    assert m["macro_precision"] == expected_macro_f1
    assert m["macro_recall"] == expected_macro_f1


# ---------------------------------------------------------------------------
# abstain handling
# ---------------------------------------------------------------------------

def test_abstain_excluded_and_counted():
    truths = ["0", "1", "2", "3"]
    preds = ["0", ABSTAIN, "2", ABSTAIN]
    m = dqmc.compute_multiclass_metrics(preds, truths, classes=CLASSES)
    assert m["n"] == 2          # only the two decided rows
    assert m["n_abstained"] == 2
    assert m["accuracy"] == 1.0  # both decided rows correct
    # abstained rows must not appear anywhere in the confusion matrix
    total = sum(c for row in m["confusion_matrix"].values() for c in row.values())
    assert total == 2
    # class '1' and '3' had their only sample abstained → support 0, recall None
    assert m["per_class"]["1"]["support"] == 0
    assert m["per_class"]["1"]["recall"] is None


# ---------------------------------------------------------------------------
# unknown label handling (documented: treated as wrong, kept in confusion)
# ---------------------------------------------------------------------------

def test_unknown_predicted_label_treated_as_wrong():
    truths = ["0", "1"]
    preds = ["X", "1"]  # 'X' is not in CLASSES
    m = dqmc.compute_multiclass_metrics(preds, truths, classes=CLASSES)
    assert m["n"] == 2
    assert m["n_abstained"] == 0
    assert m["accuracy"] == 0.5  # only the '1' row is correct
    # class '0' got no correct prediction: tp=0, fp=0 (X!=0), fn=1
    c0 = m["per_class"]["0"]
    assert c0["recall"] == 0.0
    assert c0["precision"] is None  # tp+fp == 0 → undefined
    assert c0["f1"] is None
    # unknown label recorded in confusion matrix under truth '0'
    assert m["confusion_matrix"]["0"]["X"] == 1


# ---------------------------------------------------------------------------
# confusion matrix shape + totals
# ---------------------------------------------------------------------------

def test_confusion_matrix_shape_and_totals():
    truths = ["0", "0", "0", "1", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
    preds = ["0", "0", "1", "1", "0", "2", "3", "4", "5", "6", "7", "8", "9"]
    m = dqmc.compute_multiclass_metrics(preds, truths, classes=CLASSES)
    cm = m["confusion_matrix"]
    # every class is a row (pre-seeded grid)
    assert set(cm.keys()) == set(CLASSES)
    # every row has a full column set (no unknown labels here)
    for row in cm.values():
        assert set(row.keys()) == set(CLASSES)
    # off-diagonal confusions
    assert cm["0"]["1"] == 1
    assert cm["1"]["0"] == 1
    assert cm["0"]["0"] == 2
    assert cm["1"]["1"] == 1
    # totals equal number of decided predictions
    total = sum(c for row in cm.values() for c in row.values())
    assert total == 13


def test_length_mismatch_raises():
    with pytest.raises(ValueError, match="length mismatch"):
        dqmc.compute_multiclass_metrics(["0"], ["0", "1"], classes=CLASSES)


# ---------------------------------------------------------------------------
# ground-truth coercer
# ---------------------------------------------------------------------------

def test_make_label_coercer_accepts_string_and_int():
    coerce = dqmc.make_label_coercer(CLASSES)
    assert coerce("7", None) == "7"
    assert coerce("", 3) == "3"
    with pytest.raises(ValueError, match="Unrecognized multiclass label"):
        coerce("cat", None)


# ---------------------------------------------------------------------------
# end-to-end: compute_decision_quality_multiclass + run_scoring dispatch
# ---------------------------------------------------------------------------

@pytest.fixture
def mnist_manifest(tmp_path: Path) -> Path:
    rows = [
        {"sample_id": f"img_{d}", "label": str(d), "label_int": d,
         "truth_tier": "gold", "split": "dev_golden",
         "repo_rel_path": f"data/images/mnist/{d}.png"}
        for d in range(4)
    ]
    p = tmp_path / "manifest.jsonl"
    _write_jsonl(p, rows)
    return p


@pytest.fixture
def mnist_votes(tmp_path: Path) -> Path:
    # one labeler 'cnn': correct on 0,1,2; wrong on 3 (predicts 2)
    rows = [
        {"run_id": "r1", "image_id": "img_0", "labeler_id": "cnn",
         "model_id": "demo/cnn", "label": "0"},
        {"run_id": "r1", "image_id": "img_1", "labeler_id": "cnn",
         "model_id": "demo/cnn", "label": "1"},
        {"run_id": "r1", "image_id": "img_2", "labeler_id": "cnn",
         "model_id": "demo/cnn", "label": "2"},
        {"run_id": "r1", "image_id": "img_3", "labeler_id": "cnn",
         "model_id": "demo/cnn", "label": "2"},
    ]
    p = tmp_path / "label_votes.jsonl"
    _write_jsonl(p, rows)
    return p


def test_compute_decision_quality_multiclass(mnist_votes, mnist_manifest):
    snap = dqmc.compute_decision_quality_multiclass(
        mnist_votes, mnist_manifest,
        task=tasks_mod.MNIST_MULTICLASS,
        policy_graph_version="MNIST.v0.1",
        ground_truth_tier=("gold",),
    )
    assert snap["task"] == "mnist_multiclass"
    assert snap["classes"] == list(CLASSES)
    by_id = {row["labeler_id"]: row for row in snap["labelers"]}
    assert "cnn" in by_id
    cnn = by_id["cnn"]["metrics"]
    assert cnn["n"] == 4
    assert cnn["accuracy"] == 0.75  # 3/4 correct
    # class '3': never predicted → precision undefined, recall 0
    assert cnn["per_class"]["3"]["recall"] == 0.0
    assert cnn["per_class"]["3"]["precision"] is None


def test_binary_task_rejected_by_multiclass_dq(mnist_votes, mnist_manifest):
    with pytest.raises(ValueError, match="is binary"):
        dqmc.compute_decision_quality_multiclass(
            mnist_votes, mnist_manifest,
            task=tasks_mod.GENAI_BINARY,
            policy_graph_version="x",
        )


def test_run_scoring_dispatches_multiclass(tmp_path, mnist_votes, mnist_manifest):
    runs_root = tmp_path / "runs"
    run_dir = runs_root / "run42"
    run_dir.mkdir(parents=True)
    (run_dir / "label_votes.jsonl").write_text(mnist_votes.read_text())

    result = run_scoring(
        "run42",
        tmp_path,  # repo_root (unused for artifact math here)
        runs_root=runs_root,
        manifest=mnist_manifest,
        policy_graph_version="MNIST.v0.1",
        ground_truth_tier=("gold",),
        task=tasks_mod.MNIST_MULTICLASS,
        validate_schemas=False,
    )
    assert result["ok"] is True
    assert result["multiclass"] is True
    assert result["task"] == "mnist_multiclass"
    # canonical multiclass artifact written
    dq_path = run_dir / "scoring" / "decision_quality_multiclass.json"
    assert dq_path.exists()
    dq = json.loads(dq_path.read_text())
    assert dq["task"] == "mnist_multiclass"
    assert any(row["labeler_id"] == "cnn" for row in dq["labelers"])


def test_run_scoring_auto_detects_mnist_manifest_context(tmp_path, mnist_votes, mnist_manifest):
    runs_root = tmp_path / "runs"
    run_dir = runs_root / "run-auto"
    run_dir.mkdir(parents=True)
    (run_dir / "label_votes.jsonl").write_text(mnist_votes.read_text())
    (run_dir / "run_manifest.json").write_text(json.dumps({
        "run_id": "run-auto",
        "area": "MNIST_Digits",
        "policy_version": "v0.1",
        "policy_graph_version": "MNIST_Digits.v0.1",
        "sample_manifest_path": str(mnist_manifest),
    }))

    result = run_scoring(
        "run-auto",
        tmp_path,
        runs_root=runs_root,
        ground_truth_tier=("gold",),
        validate_schemas=False,
    )

    assert result["multiclass"] is True
    assert (run_dir / "scoring" / "decision_quality_multiclass.json").exists()
    web_dq = json.loads((run_dir / "scoring" / "decision_quality.json").read_text())
    metrics = web_dq["labelers"][0]["metrics"]
    assert metrics["f1"] == metrics["macro_f1"]


def test_run_scoring_default_is_binary_backcompat(tmp_path):
    # No task passed → binary path; missing votes raises the binary error path,
    # proving the default dispatch did not divert to multiclass.
    runs_root = tmp_path / "runs"
    (runs_root / "runX").mkdir(parents=True)
    with pytest.raises(FileNotFoundError, match="label_votes.jsonl"):
        run_scoring(
            "runX", tmp_path, runs_root=runs_root,
        )
