"""RUSH experiment crank — the seeded PPO-style policy-iteration loop.

One **experiment** is a numbered, seeded demo run (Attila 2026-07-06): from a
base policy version (the generator at k=0), turn the crank ``k_max`` times.
Each cycle labels a fresh seeded mini-batch of N train images with the fixed
judge panel, proposes ONE clipped policy edit (1..max_changes node files) from
S1 randomly-sampled misalignment anchors, evaluates the candidate on the
experiment's fixed seeded TEST partition, and gates acceptance PPO-style:

    accept  iff  test system macro-F1 (majority vote) improves, i.e.
            f1_after > f1_before + epsilon        (the trust region hard wall)

The gate agent (gpt-5.5 by default) reviews every candidate and may VETO a
metric-passed edit (e.g. answer leakage, judge-specific hacks) but can never
force-accept a metric-failed one. Human SME review of gate decisions is
deferred to the end of the iteration cycle and recorded (``gate_review``) for
future RLHF of the critic agent.

Split discipline (the DS-honest reading of "gate on the test set"): the test
partition is carved once per experiment out of ``dev_golden`` with the master
seed and reused by every gate — formally a validation set, since the loop
adapts to it. The 500-image ``holdout`` split stays LOCKED: it is only scored
under the start and final versions (``--holdout-final``) for the untouched
before/after readout.

State lives in two places, deliberately:
  * ``data/experiments/<experiment_id>/experiment.json`` — the portable
    per-experiment truth (atomic rewrites; the web UI polls it; a fresh clone
    demos with no database).
  * the Postgres ``rush`` schema (``experiment`` / ``experiment_cycle`` /
    ``experiment_metric`` / ``gate_decision`` / ``gate_review``) — the
    cross-experiment analysis layer for the paper. All store writes are
    best-effort (:mod:`pipeline.experiment.store`): a dead database never
    stops the crank.

Metrics tracked per cycle, per judge AND the system of judges (majority
vote), on both train and test: accuracy, F1, precision, recall, FPR, FNR
(macro + micro + per-class) — via
:func:`pipeline.scoring.decision_quality_multiclass.compute_multiclass_metrics`.
"""
from __future__ import annotations

import json
import os
import random
import re
import secrets
import tempfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

from pipeline.scoring import _common
from pipeline.scoring.decision_quality import _majority_vote
from pipeline.scoring.decision_quality_multiclass import (
    compute_multiclass_metrics,
    make_label_coercer,
)
from pipeline.scoring.tasks import ScoringTask

SYSTEM_SCORER = "system"  # the majority-vote "system of judges" row
DEFAULT_GATE_MODEL = "openai/gpt-5.5"
DEFAULT_DRAFTER_MODEL = "openai/gpt-5.5"
DEFAULT_MAX_CHANGES = 5
MAX_CHANGES_HARD_CAP = 5  # Attila: "limit to 1, max 5 changes at a time"
DEFAULT_STRATEGY = "random_misalignment"  # S1; S2-S5 are the next experiments
GATE_METRIC = "test_system_macro_f1"

_EXPERIMENT_ID_RE = re.compile(r"^exp-[0-9]{8}T[0-9]{6}-[a-f0-9]{6}$")


# ---------------------------------------------------------------------------
# Identity + paths


def mint_experiment_id(now: datetime | None = None) -> str:
    stamp = (now or datetime.now(timezone.utc)).strftime("%Y%m%dT%H%M%S")
    return f"exp-{stamp}-{secrets.token_hex(3)}"


def validate_experiment_id(experiment_id: str) -> str:
    if not _EXPERIMENT_ID_RE.match(experiment_id or ""):
        raise ValueError(f"invalid experiment_id: {experiment_id!r}")
    return experiment_id


def experiments_root(repo_root: Path | str) -> Path:
    return Path(repo_root) / "data" / "experiments"


def experiment_dir(repo_root: Path | str, experiment_id: str) -> Path:
    return experiments_root(repo_root) / validate_experiment_id(experiment_id)


def state_path(repo_root: Path | str, experiment_id: str) -> Path:
    return experiment_dir(repo_root, experiment_id) / "experiment.json"


# ---------------------------------------------------------------------------
# Seeded sampling: test partition + per-cycle train batches + S1 anchors
#
# Every random draw derives its own ``random.Random`` from (master seed, role,
# k) over sample_id-sorted pools, so any single piece is reproducible from the
# experiment record alone — no shared RNG state to keep in sync.


def partition_test_train(
    records: Iterable[Any],
    *,
    seed: int,
    test_n: int,
    pool_split: str = "dev_golden",
) -> tuple[list[str], list[str]]:
    """Carve the experiment's fixed test partition out of the golden pool.

    Stratified by the human/gold label so the gate metric (macro-F1) is
    stable: each class contributes ``test_n / n_classes`` (lexically-first
    classes absorb the remainder), drawn seeded from the sample_id-sorted
    class pools. Returns ``(test_ids, train_pool_ids)`` — disjoint, together
    covering the whole pool split.
    """
    pool = sorted(
        (r for r in records if r.split == pool_split), key=lambda r: r.sample_id
    )
    if test_n < 1 or test_n >= len(pool):
        raise ValueError(
            f"test_n must be in [1, {len(pool) - 1}] for a {len(pool)}-image pool; got {test_n}"
        )
    by_label: dict[str, list[str]] = defaultdict(list)
    for rec in pool:
        by_label[str(rec.sme_label)].append(rec.sample_id)
    labels = sorted(by_label)
    base, remainder = divmod(test_n, len(labels))
    test_ids: list[str] = []
    for idx, label in enumerate(labels):
        quota = min(base + (1 if idx < remainder else 0), len(by_label[label]))
        rng = random.Random(f"{seed}:test:{label}")
        test_ids.extend(rng.sample(by_label[label], quota))
    # Back-fill any shortfall from the remaining pool (lopsided class pools).
    if len(test_ids) < test_n:
        taken = set(test_ids)
        leftover = [r.sample_id for r in pool if r.sample_id not in taken]
        rng = random.Random(f"{seed}:test:backfill")
        test_ids.extend(rng.sample(leftover, test_n - len(test_ids)))
    test_set = set(test_ids)
    train_pool = [r.sample_id for r in pool if r.sample_id not in test_set]
    return sorted(test_ids), sorted(train_pool)


def cycle_seed(seed: int, k: int) -> int:
    """Stable per-cycle seed recorded in the cycle row (audit convenience)."""
    return random.Random(f"{seed}:cycle:{k}").getrandbits(48)


def sample_train_batch(
    train_pool: list[str],
    used_ids: set[str],
    *,
    seed: int,
    k: int,
    batch_n: int,
) -> list[str]:
    """Seeded draw of this cycle's mini-batch from the not-yet-used train pool.

    Without replacement across cycles while the pool lasts; once fewer than
    ``batch_n`` unused ids remain, the batch tops up from already-used ids
    (recorded by the caller via the shrinking ``used_ids`` invariant — reuse
    means the image is re-judged under a NEWER generator, which the llm_label
    dedup contract stores as a distinct verdict).
    """
    fresh = sorted(set(train_pool) - used_ids)
    rng = random.Random(f"{seed}:train:{k}")
    if len(fresh) >= batch_n:
        return sorted(rng.sample(fresh, batch_n))
    reused_pool = sorted(set(train_pool) - set(fresh))
    top_up = rng.sample(reused_pool, min(batch_n - len(fresh), len(reused_pool)))
    return sorted(fresh + top_up)


def select_anchors(
    misalignment_records: list[dict[str, Any]],
    *,
    seed: int,
    k: int,
    max_anchors: int,
    train_ids: Iterable[str] | None = None,
) -> list[dict[str, Any]]:
    """S1 — random misalignment anchors (the SVM-flavored baseline).

    Eligible rows: this cycle's train images whose panel verdict misaligned
    with the golden label (``misalignment_type != 'all_agree'``). A uniform
    seeded sample of up to ``max_anchors`` of them drives the policy edit —
    unbiased coverage of the error surface, no ranking heuristics (those are
    strategies S2-S5, tested separately).
    """
    allowed = set(train_ids) if train_ids is not None else None
    eligible = [
        r
        for r in misalignment_records
        if r.get("misalignment_type") != "all_agree"
        and (allowed is None or str(r.get("image_id")) in allowed)
    ]
    eligible.sort(key=lambda r: str(r.get("image_id")))
    rng = random.Random(f"{seed}:anchors:{k}")
    if len(eligible) <= max_anchors:
        return eligible
    return sorted(
        rng.sample(eligible, max_anchors), key=lambda r: str(r.get("image_id"))
    )


# ---------------------------------------------------------------------------
# Edit clipping: 1..max_changes discrete node-file changes per version bump


def clip_changes(
    proposed_files: dict[str, str],
    files_removed: list[str],
    *,
    base_dir: Path,
    max_changes: int,
) -> dict[str, Any]:
    """Enforce the 1..max_changes clip on a drafted edit, in emission order.

    The countable unit of change is the policy NODE FILE (one ``.md`` in the
    version bundle): a modification, addition, or removal each count 1. The
    drafter is instructed to stay within the budget; this is the hard
    backstop for when it does not. Entries that are no-ops against the base
    bundle (identical content; removing a file that does not exist) are
    dropped, not counted.

    Returns ``{"files": ..., "removed": ..., "n_proposed": int,
    "n_applied": int, "clipped": bool, "dropped": [...]}``.

    Ordering note: the parse layer splits the drafter's array into
    modified/added (order preserved) and a separate removals list, so
    removals always sort after content changes here — under a tight clip,
    removals are the first thing dropped. Deliberate: destructive changes
    get the lowest survival priority.
    """
    if not 1 <= max_changes <= MAX_CHANGES_HARD_CAP:
        raise ValueError(
            f"max_changes must be 1..{MAX_CHANGES_HARD_CAP}, got {max_changes}"
        )
    real: list[tuple[str, str, str | None]] = []  # (kind, filename, content)
    for filename, content in proposed_files.items():
        base_file = base_dir / filename
        if base_file.exists():
            if base_file.read_text(encoding="utf-8") == content:
                continue  # no-op rewrite
            real.append(("modified", filename, content))
        else:
            real.append(("added", filename, content))
    kept_names = {name for _, name, _ in real}
    for filename in files_removed:
        # A drafter can emit the same path as modified AND removed (a merge
        # gone sideways); downstream _classify_changes hard-errors on that.
        # The content change wins — the contradictory removal is dropped.
        if filename in kept_names:
            continue
        if (base_dir / filename).exists():
            real.append(("removed", filename, None))

    n_proposed = len(real)
    kept = real[:max_changes]  # emission order = the drafter's own priority
    dropped = [
        {"change": kind, "path": name} for kind, name, _ in real[max_changes:]
    ]
    return {
        "files": {name: content for kind, name, content in kept if kind != "removed"},
        "removed": [name for kind, name, _ in kept if kind == "removed"],
        "n_proposed": n_proposed,
        "n_applied": len(kept),
        "clipped": n_proposed > len(kept),
        "dropped": dropped,
        "changes": [{"change": kind, "path": name} for kind, name, _ in kept],
    }


def materialize_candidate(
    *,
    base_dir: Path,
    out_dir: Path,
    files: dict[str, str],
    removed: list[str],
) -> Path:
    """Write the candidate policy bundle: base version + clipped overlay.

    The candidate lives under the experiment dir, NOT ``policy-graph/`` —
    only the gate's accept turns it into a real version (via
    ``policy_diff.accept_proposal``), preserving the invariant that
    ``policy-graph/`` holds accepted versions only.
    """
    import shutil

    if out_dir.exists():
        shutil.rmtree(out_dir)
    shutil.copytree(base_dir, out_dir)
    for filename, content in files.items():
        (out_dir / filename).write_text(content, encoding="utf-8")
    for filename in removed:
        target = out_dir / filename
        if target.exists():
            target.unlink()
    return out_dir


# ---------------------------------------------------------------------------
# Per-cycle metrics: every judge + the system of judges, all six metrics


def panel_metrics(
    votes: list[dict[str, Any]],
    truth_labels: dict[str, str],
    *,
    task: ScoringTask,
    include_confusion: bool = False,
) -> dict[str, dict[str, Any]]:
    """Compute per-judge + system metrics for one run's votes.

    ``truth_labels`` maps image_id -> golden label (already coerced to the
    task vocabulary). Returns ``{scorer: metrics}`` where scorer is each
    judge's model id plus :data:`SYSTEM_SCORER` (majority vote across decided
    judges, ties excluded — the same ensemble rule as the scoring artifacts).
    The heavyweight confusion matrix is stripped unless requested; the full
    artifact remains in the child run's scoring dir.
    """
    class_set = set(task.classes)
    by_labeler: dict[str, list[tuple[str, str]]] = defaultdict(list)
    abstains: dict[str, int] = defaultdict(int)
    per_image: dict[str, dict[str, str]] = defaultdict(dict)

    for vote in votes:
        image_id = str(vote.get("image_id") or "")
        gt = truth_labels.get(image_id)
        if gt is None:
            continue
        labeler = _common.labeler_id_for(vote)
        label = vote.get("label", task.abstain)
        if label not in class_set and label != task.abstain:
            match = re.fullmatch(r"MD\.digit\.(\d+)", str(vote.get("l2_label", "")))
            if match and match.group(1) in class_set:
                label = match.group(1)
        per_image[image_id][labeler] = label
        if label == task.abstain:
            abstains[labeler] += 1
            continue
        by_labeler[labeler].append((label, gt))

    out: dict[str, dict[str, Any]] = {}
    for labeler in sorted(by_labeler.keys() | abstains.keys()):
        pairs = by_labeler.get(labeler, [])
        preds = [p for p, _ in pairs] + [task.abstain] * abstains.get(labeler, 0)
        truths = [t for _, t in pairs] + [""] * abstains.get(labeler, 0)
        out[labeler] = compute_multiclass_metrics(
            preds, truths, classes=task.classes, abstain=task.abstain
        )

    system_preds: list[str] = []
    system_truths: list[str] = []
    for image_id in sorted(per_image):
        winner = _majority_vote(per_image[image_id])
        if winner is None:
            continue
        system_preds.append(winner)
        system_truths.append(truth_labels[image_id])
    if system_preds:
        out[SYSTEM_SCORER] = compute_multiclass_metrics(
            system_preds, system_truths, classes=task.classes, abstain=task.abstain
        )

    if not include_confusion:
        for metrics in out.values():
            metrics.pop("confusion_matrix", None)
    return out


def load_truth_labels(
    manifest_path: Path,
    *,
    task: ScoringTask,
    restrict_ids: Iterable[str] | None = None,
    ground_truth_tier: tuple[str, ...] = ("gold", "platinum", "gold_candidate"),
) -> dict[str, str]:
    """Golden labels (coerced to the task vocabulary) keyed by image_id."""
    truth = _common.load_ground_truth(
        manifest_path,
        truth_tiers=ground_truth_tier,
        label_coercer=make_label_coercer(task.classes),
    )
    allowed = set(restrict_ids) if restrict_ids is not None else None
    return {
        image_id: gt.label
        for image_id, gt in truth.items()
        if allowed is None or image_id in allowed
    }


def load_votes(run_dir: Path) -> list[dict[str, Any]]:
    """A run's label votes (schema-tolerant JSONL)."""
    return _common.load_label_votes(Path(run_dir) / "label_votes.jsonl")


def load_run_panel_metrics(
    run_dir: Path,
    manifest_path: Path,
    *,
    task: ScoringTask,
    restrict_ids: Iterable[str] | None = None,
    ground_truth_tier: tuple[str, ...] = ("gold", "platinum", "gold_candidate"),
) -> dict[str, dict[str, Any]]:
    """Panel metrics for a completed child run, straight from its artifacts."""
    truth = _common.load_ground_truth(
        manifest_path,
        truth_tiers=ground_truth_tier,
        label_coercer=make_label_coercer(task.classes),
    )
    allowed = set(restrict_ids) if restrict_ids is not None else None
    truth_labels = {
        image_id: gt.label
        for image_id, gt in truth.items()
        if allowed is None or image_id in allowed
    }
    votes = _common.load_label_votes(run_dir / "label_votes.jsonl")
    return panel_metrics(votes, truth_labels, task=task)


def system_decisions(
    votes: list[dict[str, Any]],
    truth_labels: dict[str, str],
    *,
    task: ScoringTask,
) -> dict[str, str]:
    """Per-image majority-vote decision (ties/all-abstain excluded)."""
    class_set = set(task.classes)
    per_image: dict[str, dict[str, str]] = defaultdict(dict)
    for vote in votes:
        image_id = str(vote.get("image_id") or "")
        if image_id not in truth_labels:
            continue
        label = vote.get("label", task.abstain)
        if label not in class_set and label != task.abstain:
            match = re.fullmatch(r"MD\.digit\.(\d+)", str(vote.get("l2_label", "")))
            if match and match.group(1) in class_set:
                label = match.group(1)
        per_image[image_id][_common.labeler_id_for(vote)] = label
    out: dict[str, str] = {}
    for image_id, panel in per_image.items():
        winner = _majority_vote(panel)
        if winner is not None:
            out[image_id] = winner
    return out


def gate_comparison(
    votes_before: list[dict[str, Any]],
    votes_after: list[dict[str, Any]],
    truth_labels: dict[str, str],
    *,
    task: ScoringTask,
) -> dict[str, Any]:
    """The gate's before/after values, computed over the SAME images.

    Errored calls (partial-error child runs) and majority-vote ties both
    remove images from one side's system verdicts — comparing full-run
    macro-F1 would then compare different subsets of the "fixed" test
    partition and can flip the gate on coverage alone. The rule therefore
    scores both policies over the intersection of images where BOTH sides
    produced a decided system verdict, and reports coverage so the gate
    record (and the gate agent) can see when it degraded.
    """
    before = system_decisions(votes_before, truth_labels, task=task)
    after = system_decisions(votes_after, truth_labels, task=task)
    common = sorted(set(before) & set(after))
    result: dict[str, Any] = {
        "n_common": len(common),
        "n_before": len(before),
        "n_after": len(after),
        "n_expected": len(truth_labels),
    }
    if not common:
        result.update({"value_before": None, "value_after": None})
        return result
    truths = [truth_labels[i] for i in common]
    metrics_before = compute_multiclass_metrics(
        [before[i] for i in common], truths, classes=task.classes, abstain=task.abstain
    )
    metrics_after = compute_multiclass_metrics(
        [after[i] for i in common], truths, classes=task.classes, abstain=task.abstain
    )
    result.update(
        {
            "value_before": metrics_before.get("macro_f1"),
            "value_after": metrics_after.get("macro_f1"),
        }
    )
    return result


# ---------------------------------------------------------------------------
# The PPO gate: deterministic rule + gate-agent review


def metric_passes(
    value_before: float | None, value_after: float | None, *, epsilon: float = 0.0
) -> bool:
    """The trust-region rule: candidate must strictly improve test system F1."""
    if value_before is None or value_after is None:
        return False
    return value_after > value_before + epsilon


GATE_SYSTEM_PROMPT = (
    "You are RUSH's PPO acceptance gate for policy-prompt updates. A candidate "
    "policy edit was evaluated on the held-out test partition. The hard rule: "
    "ACCEPT only if the system-of-judges macro-F1 on test improved "
    "(value_after > value_before + epsilon). You may SKIP a metric-passing "
    "candidate when the edit itself is unsound — e.g. it leaks ground-truth "
    "answers, overfits to named examples instead of stating a general "
    "guideline, targets one judge model's quirks, tells judges to abstain or "
    "defer instead of committing to a label, piles class- or pair-specific "
    "rules into the root file instead of the owning class/boundary node, or "
    "is incoherent with the "
    "policy's structure. You can NEVER accept a metric-failing candidate. "
    "Respond with JSON only: {\"decision\": \"accept\"|\"skip\", "
    "\"rationale\": \"<=80 words\", \"risk_flags\": [\"...\"]}."
)


def build_gate_messages(
    *,
    metric: str,
    value_before: float | None,
    value_after: float | None,
    epsilon: float,
    metric_pass: bool,
    metrics_before: dict[str, Any],
    metrics_after: dict[str, Any],
    diffs: list[dict[str, Any]],
    anchors: list[dict[str, Any]],
    k: int,
    comparison: dict[str, Any] | None = None,
) -> list[dict[str, str]]:
    """Assemble the gate agent's review packet (metric table + diff + anchors)."""

    def _summary(metrics: dict[str, Any]) -> dict[str, Any]:
        return {
            scorer: {
                key: m.get(key)
                for key in (
                    "accuracy",
                    "macro_f1",
                    "macro_precision",
                    "macro_recall",
                    "macro_fpr",
                    "macro_fnr",
                    "n",
                    "n_abstained",
                )
            }
            for scorer, m in metrics.items()
        }

    payload = {
        "cycle_k": k,
        "gate_metric": metric,
        "value_before": value_before,
        "value_after": value_after,
        "epsilon": epsilon,
        "metric_pass": metric_pass,
        # Coverage of the common-subset comparison: flag degraded coverage
        # (errored calls / ties shrinking the compared set) as a risk.
        "comparison_coverage": comparison or {},
        "test_metrics_baseline": _summary(metrics_before),
        "test_metrics_candidate": _summary(metrics_after),
        "proposed_edit_diffs": [
            {
                "path": d.get("path"),
                "change": d.get("change"),
                "unified_diff": str(d.get("unified_diff", ""))[:4000],
            }
            for d in diffs
        ],
        "anchor_misalignments": [
            {
                "image_id": a.get("image_id"),
                "sme_truth": a.get("sme_truth"),
                "misalignment_type": a.get("misalignment_type"),
                "severity": a.get("severity"),
            }
            for a in anchors
        ],
    }
    return [
        {"role": "system", "content": GATE_SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(payload, indent=2)},
    ]


def parse_gate_response(raw: str) -> dict[str, Any]:
    """Parse the gate agent's JSON (tolerant: fenced/embedded JSON accepted)."""
    from pipeline.providers.base import parse_label_json

    obj = parse_label_json(raw)
    decision = str(obj.get("decision", "")).strip().lower()
    if decision not in {"accept", "skip"}:
        raise ValueError(f"gate response has no accept/skip decision: {raw[:200]!r}")
    return {
        "decision": decision,
        "rationale": str(obj.get("rationale", "")).strip(),
        "risk_flags": [str(f) for f in obj.get("risk_flags", []) if f],
    }


def resolve_gate_decision(
    *, metric_pass: bool, agent: dict[str, Any] | None, gate_off: bool = False
) -> dict[str, Any]:
    """Compose the final gate outcome from the rule and the agent's review.

    Truth table (the rule is the hard wall, the agent is a one-way valve):
      gate_off                   -> accept (gate_off; --gate-mode off — the metric
                                   is recorded for the curve, never enforced)
      rule fail + any agent      -> skip  (override_guard if agent said accept)
      rule pass + no agent       -> accept (metric_rule; --gate-mode metric_only)
      rule pass + agent accept   -> accept (gate_agent)
      rule pass + agent skip     -> skip  (gate_agent_veto)
    """
    if gate_off:
        return {
            "decision": "accept",
            "decided_by": "gate_off",
            "rationale": (
                "gate disabled for this run: every clipped edit is applied so the "
                "learning curve shows unfiltered policy drift (metric recorded, "
                "not enforced)"
            ),
            "risk_flags": [],
        }
    if not metric_pass:
        decided_by = (
            "override_guard" if agent and agent.get("decision") == "accept" else "metric_rule"
        )
        return {
            "decision": "skip",
            "decided_by": decided_by,
            "rationale": (agent or {}).get(
                "rationale", "test system macro-F1 did not improve"
            ),
            "risk_flags": (agent or {}).get("risk_flags", []),
        }
    if agent is None:
        return {
            "decision": "accept",
            "decided_by": "metric_rule",
            "rationale": "test system macro-F1 improved (metric_only gate)",
            "risk_flags": [],
        }
    if agent["decision"] == "accept":
        return {
            "decision": "accept",
            "decided_by": "gate_agent",
            "rationale": agent.get("rationale", ""),
            "risk_flags": agent.get("risk_flags", []),
        }
    return {
        "decision": "skip",
        "decided_by": "gate_agent_veto",
        "rationale": agent.get("rationale", ""),
        "risk_flags": agent.get("risk_flags", []),
    }


# ---------------------------------------------------------------------------
# Drafter prompt (S1 anchors -> one clipped edit)

DRAFTER_SYSTEM_PROMPT = (
    "You are RUSH's policy diff writer inside a PPO-style iteration loop. "
    "From the misaligned samples, draft the SINGLE most impactful policy "
    "improvement as minimal full-file markdown changes. HARD BUDGET: at most "
    "{max_changes} file changes total (modified + added + removed combined); "
    "fewer is better — one focused, human-reviewable change is ideal. "
    "TARGET THE MOST SPECIFIC NODE THAT OWNS THE ERROR: class-specific "
    "guidance belongs in that class's own node file (e.g. MD.digit.4.md), "
    "and guidance about ONE confusion pair belongs in a dedicated boundary "
    "node. The root file is the decision procedure — treat it as effectively "
    "frozen and touch it only when no more specific node can carry the rule; "
    "piling rules into the root is a defect, not an improvement. When "
    "several anchors share one confusion pair (truth A misread as B), PREFER "
    "ADDING a new boundary node — a file named like MD.boundary.4_vs_9.md "
    "whose frontmatter mirrors a sibling node's shape with node_type "
    "'boundary', parent set to the true class's node id, polarity 'mixed', "
    "status 'draft', and frontmatter edges of type 'confused_with' pointing "
    "at BOTH class nodes — so the graph grows a visible boundary case "
    "instead of a longer root. State "
    "general, model-agnostic guidance (clear definitions, boundary rules, "
    "canonical examples); NEVER encode per-image answers or ground-truth "
    "labels. The policy must always demand a decisive label: NEVER add "
    "guidance telling judges to abstain, defer, or decline — uncertainty "
    "belongs in the confidence score [0,1] and the difficulty rating, not in "
    "refusing to answer. "
    "Keep each file's YAML frontmatter (id, version, title, area, "
    "node_type, polarity, parent, status, edges) intact and consistent. "
    "Return JSON only: {{\"files\":[{{\"path\":\"name.md\",\"change\":"
    "\"modified|added|removed\",\"content\":\"full markdown for "
    "added/modified files\"}}]}}. Never return unified diffs."
)


def build_drafter_messages(
    *,
    policy_markdown: str,
    base_version: str,
    area: str,
    anchors: list[dict[str, Any]],
    max_changes: int,
    k: int,
) -> list[dict[str, str]]:
    """Assemble the drafter packet: current bundle + S1 anchors + the budget."""
    payload = {
        "task": (
            "Improve the policy so ALL judges (and human raters) decide these "
            "misaligned samples correctly — without regressing the rest."
        ),
        "area": area,
        "base_version": base_version,
        "cycle_k": k,
        "max_changes": max_changes,
        "policy_markdown": policy_markdown,
        "misaligned_samples": [
            {
                "image_id": a.get("image_id"),
                "sme_truth": a.get("sme_truth"),
                "misalignment_type": a.get("misalignment_type"),
                "severity": a.get("severity"),
                "votes": [
                    {
                        "model": _common.labeler_id_for(v),
                        "label": v.get("label"),
                        "confidence": v.get("confidence"),
                        "is_boundary": v.get("is_boundary"),
                        "difficulty": v.get("difficulty"),
                        "justification": str(v.get("justification", ""))[:400],
                    }
                    for v in a.get("votes", [])
                ],
            }
            for a in anchors
        ],
    }
    return [
        {
            "role": "system",
            "content": DRAFTER_SYSTEM_PROMPT.format(max_changes=max_changes),
        },
        {"role": "user", "content": json.dumps(payload, indent=2)},
    ]


# ---------------------------------------------------------------------------
# Dry-run fakes (no network): exercise the full loop shape offline


def fake_drafter_callable(base_dir: Path) -> Callable[..., str]:
    """Deterministic offline drafter: one benign modification to one node.

    Appends an audit comment to the lexically-first ``.md`` node — a real,
    countable, acceptable-shape change with zero semantic effect, so dry runs
    exercise staging/clipping/candidate-eval/gating end to end.
    """

    def _chat(messages: list[dict[str, Any]], **_: Any) -> str:
        payload = json.loads(messages[-1]["content"])
        k = payload.get("cycle_k", 0)
        nodes = sorted(p.name for p in base_dir.glob("*.md"))
        if not nodes:
            return json.dumps({"files": []})
        target = nodes[0]
        content = (base_dir / target).read_text(encoding="utf-8")
        content += f"\n<!-- dry-run experiment edit, cycle k={k} -->\n"
        return json.dumps(
            {"files": [{"path": target, "change": "modified", "content": content}]}
        )

    return _chat


def fake_gate_callable() -> Callable[..., str]:
    """Offline gate agent: defer to the metric rule, flag nothing."""

    def _chat(messages: list[dict[str, Any]], **_: Any) -> str:
        payload = json.loads(messages[-1]["content"])
        decision = "accept" if payload.get("metric_pass") else "skip"
        return json.dumps(
            {
                "decision": decision,
                "rationale": "dry-run gate: deferring to the metric rule",
                "risk_flags": [],
            }
        )

    return _chat


# ---------------------------------------------------------------------------
# State I/O: experiment.json is the portable truth the web UI polls


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


SUMMARY_METRIC_KEYS = (
    "accuracy", "macro_f1", "macro_precision", "macro_recall",
    "macro_fpr", "macro_fnr", "micro_f1", "n", "n_abstained",
)


def build_run_summary(state: dict[str, Any]) -> dict[str, Any]:
    """End-of-run analysis record (Attila 2026-07-06).

    Per-scorer test metrics at k=0 and at the final cycle plus their deltas,
    with the run metadata needed to write a cross-run analysis later without
    re-opening cycle records. Written into experiment.json and mirrored to
    ``rush.experiment.summary``.
    """
    cycles = [c for c in state.get("cycles", []) if isinstance(c.get("k"), int)]
    base_cycle = next((c for c in cycles if c["k"] == 0), None)
    final_cycle = cycles[-1] if cycles else None
    base_metrics = ((base_cycle or {}).get("metrics") or {}).get("test") or {}
    final_metrics = ((final_cycle or {}).get("metrics") or {}).get("test") or {}

    def _row(metrics: dict[str, Any], scorer: str) -> dict[str, Any]:
        m = metrics.get(scorer) or {}
        return {key: m.get(key) for key in SUMMARY_METRIC_KEYS}

    per_scorer: dict[str, Any] = {}
    for scorer in sorted(set(base_metrics) | set(final_metrics)):
        baseline = _row(base_metrics, scorer)
        final = _row(final_metrics, scorer)
        delta: dict[str, Any] = {}
        for key in SUMMARY_METRIC_KEYS:
            b, f = baseline.get(key), final.get(key)
            numeric = (
                isinstance(b, (int, float)) and not isinstance(b, bool)
                and isinstance(f, (int, float)) and not isinstance(f, bool)
            )
            delta[key] = round(f - b, 6) if numeric else None
        per_scorer[scorer] = {"baseline": baseline, "final": final, "delta": delta}

    holdout = state.get("holdout") or {}
    benchmark = state.get("benchmark") or {}

    def _system_row(block: dict[str, Any], tag: str) -> dict[str, Any] | None:
        metrics = ((block.get(tag) or {}).get("metrics") or {}).get(SYSTEM_SCORER)
        if not metrics:
            return None
        return {key: metrics.get(key) for key in SUMMARY_METRIC_KEYS}

    return {
        "recorded_at": utcnow_iso(),
        "experiment_id": state.get("experiment_id"),
        "run_number": state.get("run_number"),
        "status": state.get("status"),
        "area": state.get("area"),
        "seed": state.get("seed"),
        "config": {
            key: state.get(key)
            for key in (
                "k_max", "batch_n", "test_n", "max_changes", "max_anchors",
                "epsilon", "strategy", "gate_mode", "gate_model",
                "drafter_model", "judge_models", "concurrency", "dry_run",
            )
        },
        "policy": {
            "base_version": state.get("base_version"),
            "final_version": state.get("current_version"),
            "accepted_cycles": [c["k"] for c in cycles if c.get("status") == "accepted"],
            "n_cycles": max(0, len(cycles) - 1),
        },
        "cost_usd_total": state.get("cost_usd_total"),
        "started_at": state.get("started_at"),
        "finished_at": state.get("finished_at"),
        "test_metrics": per_scorer,
        "holdout_system": (
            {"start": _system_row(holdout, "start"), "final": _system_row(holdout, "final")}
            if holdout else None
        ),
        # The fixed cross-run validation split — the benchmark numbers that
        # compare run numbers / strategies on identical images.
        "benchmark_system": (
            {"start": _system_row(benchmark, "start"), "final": _system_row(benchmark, "final")}
            if benchmark else None
        ),
    }


def write_state(repo_root: Path | str, state: dict[str, Any]) -> Path:
    """Atomic rewrite of experiment.json (tmpfile + rename, crash-safe)."""
    path = state_path(repo_root, state["experiment_id"])
    path.parent.mkdir(parents=True, exist_ok=True)
    state["updated_at"] = utcnow_iso()
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(state, fh, indent=1, sort_keys=True)
            fh.write("\n")
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    return path


def merge_disk_reviews(repo_root: Path | str, state: dict[str, Any]) -> None:
    """Fold SME reviews written to disk into the driver's in-memory state.

    The driver rewrites experiment.json wholesale; the web review endpoint
    writes into the same file from another process. Without this merge, a
    review posted mid-run is clobbered by the driver's next rewrite. Called
    before every driver write — the race window shrinks from a whole cycle
    to the read-modify-write itself (fine for a single-operator local tool).
    """
    try:
        disk = load_state(repo_root, state["experiment_id"])
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return
    disk_reviews = {
        cycle.get("k"): cycle.get("review")
        for cycle in disk.get("cycles", [])
        if cycle.get("review")
    }
    if not disk_reviews:
        return
    for cycle in state.get("cycles", []):
        if not cycle.get("review") and disk_reviews.get(cycle.get("k")):
            cycle["review"] = disk_reviews[cycle["k"]]


def load_state(repo_root: Path | str, experiment_id: str) -> dict[str, Any]:
    path = state_path(repo_root, experiment_id)
    if not path.exists():
        raise FileNotFoundError(f"no such experiment: {experiment_id}")
    return json.loads(path.read_text(encoding="utf-8"))


def list_experiments(repo_root: Path | str) -> list[dict[str, Any]]:
    """Newest-first experiment summaries for the web list endpoint."""
    root = experiments_root(repo_root)
    out: list[dict[str, Any]] = []
    if not root.is_dir():
        return out
    for entry in root.iterdir():
        if not entry.is_dir() or entry.name.startswith("_"):
            continue
        path = entry / "experiment.json"
        if not path.exists():
            continue
        try:
            state = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        cycles = state.get("cycles", [])
        accepted = sum(1 for c in cycles if c.get("status") == "accepted")
        out.append(
            {
                "experiment_id": state.get("experiment_id", entry.name),
                "run_number": state.get("run_number"),
                "area": state.get("area"),
                "seed": state.get("seed"),
                "status": state.get("status"),
                "phase": state.get("phase"),
                "k_max": state.get("k_max"),
                "cycles_done": max(0, len(cycles) - 1),  # k=0 is the baseline
                "accepted": accepted,
                "judge_models": state.get("judge_models", []),
                "gate_model": state.get("gate_model"),
                "base_version": state.get("base_version"),
                "current_version": state.get("current_version"),
                "dry_run": state.get("dry_run", False),
                "cost_usd_total": state.get("cost_usd_total"),
                "started_at": state.get("started_at"),
                "finished_at": state.get("finished_at"),
            }
        )
    out.sort(key=lambda e: str(e.get("started_at") or ""), reverse=True)
    return out


def next_run_number(repo_root: Path | str, area: str) -> int:
    """Human-friendly sequential run number per area (Attila's 'run numbers')."""
    numbers = [
        int(e["run_number"])
        for e in list_experiments(repo_root)
        if e.get("area") == area and isinstance(e.get("run_number"), int)
    ]
    return (max(numbers) + 1) if numbers else 1


def record_gate_review(
    repo_root: Path | str,
    experiment_id: str,
    k: int,
    *,
    verdict: str,
    reviewer: str,
    comment: str = "",
) -> dict[str, Any]:
    """SME review of one gate decision — the recorded critic-of-the-critic.

    Updates the portable state and best-effort mirrors to ``rush.gate_review``
    (future RLHF training data for the gate agent).
    """
    if verdict not in {"correct", "incorrect", "unsure"}:
        raise ValueError(f"verdict must be correct|incorrect|unsure, got {verdict!r}")
    state = load_state(repo_root, experiment_id)
    target = None
    for cycle in state.get("cycles", []):
        if cycle.get("k") == k:
            target = cycle
            break
    if target is None:
        raise KeyError(f"experiment {experiment_id} has no cycle k={k}")
    if not target.get("gate"):
        raise ValueError(f"cycle k={k} has no gate decision to review")
    review = {
        "verdict": verdict,
        "reviewer": reviewer or "sme",
        "comment": comment or "",
        "created_at": utcnow_iso(),
    }
    target["review"] = review
    write_state(repo_root, state)

    from pipeline.experiment import store

    review["store_synced"] = store.try_sync_gate_review(
        experiment_id=experiment_id, k=k, review=review
    )
    return review
