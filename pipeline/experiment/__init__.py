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
import math
import os
import random
import re
import secrets
import tempfile
from collections import Counter, defaultdict
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
DEFAULT_STRATEGY = "random_misalignment"  # S1; S2-S4 are the next experiments
# Anchor-selection strategies the driver accepts. top_gradient ranks by avg
# |g| (confidence x error) descending; top_importance ranks by the full
# four-tier importance score (misalignment x LLM-consensus x confidence x
# boundary) — "leverage all information to stack-rank the anchors" (Attila).
STRATEGIES = ("random_misalignment", "top_gradient", "top_importance")
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
    strategy: str = "random_misalignment",
) -> list[dict[str, Any]]:
    """Pick the misalignments that drive this cycle's policy edit.

    Eligible rows: this cycle's train images whose panel verdict misaligned
    with the golden label (``misalignment_type != 'all_agree'``).

    Strategies:
      * ``random_misalignment`` (S1) — uniform seeded sample of up to
        ``max_anchors``: unbiased coverage of the error surface.
      * ``top_gradient`` (S5-flavored, Attila 2026-07-07: "the most important
        misalignments for the gradient descent") — deterministic top
        ``max_anchors`` by panel avg gradient magnitude |g| = 1-p
        (confident-wrong panels first; no-signal panels rank above
        confident-correct-adjacent noise via the None-first tie).
    """
    if strategy not in STRATEGIES:
        raise ValueError(f"unknown anchor strategy: {strategy!r}")
    allowed = set(train_ids) if train_ids is not None else None
    eligible = [
        r
        for r in misalignment_records
        if r.get("misalignment_type") != "all_agree"
        and (allowed is None or str(r.get("image_id")) in allowed)
    ]
    eligible.sort(key=lambda r: str(r.get("image_id")))
    if strategy == "top_gradient":
        def _grad_key(record: dict[str, Any]) -> tuple[float, str]:
            magnitude = (panel_signal(record).get("gradient") or {}).get("avg_magnitude")
            # No decisive/confident votes = no machine signal at all — rank
            # those first (they need the policy's attention most), then by
            # descending |g|.
            return (-(magnitude if magnitude is not None else 2.0),
                    str(record.get("image_id")))
        return sorted(eligible, key=_grad_key)[:max_anchors]
    if strategy == "top_importance":
        def _imp_key(record: dict[str, Any]) -> tuple[float, str]:
            # Full four-tier anchor value: misalignment x LLM-consensus x
            # confidence x boundary. T1 (unanimous & wrong) leads.
            score = (panel_signal(record).get("importance") or {}).get("anchor")
            return (-(score if score is not None else 0.0),
                    str(record.get("image_id")))
        return sorted(eligible, key=_imp_key)[:max_anchors]
    rng = random.Random(f"{seed}:anchors:{k}")
    if len(eligible) <= max_anchors:
        return eligible
    return sorted(
        rng.sample(eligible, max_anchors), key=lambda r: str(r.get("image_id"))
    )


def select_aligned_anchors(
    misalignment_records: list[dict[str, Any]],
    *,
    seed: int,
    k: int,
    max_aligned: int,
    train_ids: Iterable[str] | None = None,
) -> list[dict[str, Any]]:
    """Pick correctly-classified anchors to accompany the misaligned ones.

    Attila 2026-07-07: "not only do an SVM [the misaligned/boundary errors]
    but the system is also learning from correct classifications." Where
    :func:`select_anchors` returns the panel's *errors*, this returns a
    seeded sample of up to ``max_aligned`` of its *successes* — this cycle's
    train images the whole panel labeled in agreement with the golden label
    (``misalignment_type == 'all_agree'``). Feeding both lets the drafter
    sharpen the boundary without regressing what the policy already gets
    right. ``max_aligned == 0`` disables aligned anchors (empty list).

    Unbiased seeded sample (not ranked): correct classifications are the
    positive reference class, so representative coverage beats picking the
    "most confident" easy wins. A ranked/near-boundary variant can be added
    later the way ``top_gradient`` was added for the misaligned side.
    """
    if max_aligned <= 0:
        return []
    allowed = set(train_ids) if train_ids is not None else None
    eligible = [
        r
        for r in misalignment_records
        if r.get("misalignment_type") == "all_agree"
        and (allowed is None or str(r.get("image_id")) in allowed)
    ]
    eligible.sort(key=lambda r: str(r.get("image_id")))
    if len(eligible) <= max_aligned:
        return eligible
    rng = random.Random(f"{seed}:aligned:{k}")
    return sorted(
        rng.sample(eligible, max_aligned), key=lambda r: str(r.get("image_id"))
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
    "rules into the root file instead of the owning class/boundary node, "
    "is incoherent with the "
    "policy's structure, is mere rewording — restating existing sentences "
    "with the same meaning — or needlessly balloons the policy's prompt "
    "mass (see policy_bundle_size: the bundle is every judge's context, and "
    "small judges measurably collapse under a bloated one — weigh the size "
    "delta against the value delivered). Diff churn that changes no "
    "semantic content, "
    "decision boundary, or objective fact is not an improvement; SKIP it even "
    "if the metric ticked up (small-partition noise can pass a no-op). "
    "You can NEVER accept a metric-failing candidate. "
    "Respond with JSON only: {\"decision\": \"accept\"|\"skip\", "
    "\"rationale\": \"<=80 words\", \"risk_flags\": [\"...\"]}."
)

# --gate-mode agent_only: the critic agent IS the gate. The metric comparison
# is advisory context (recorded for the learning curve, never enforced), so
# the agent may accept a metric-flat edit it judges sound — the arm that
# tests LLM judgment against the metric rule.
GATE_AGENT_ONLY_SYSTEM_PROMPT = (
    "You are RUSH's acceptance CRITIC for policy-prompt updates — in this run "
    "your verdict alone decides; there is no hard metric rule. A candidate "
    "policy edit was evaluated on the held-out test partition and the metric "
    "comparison (value_before/value_after, per-judge tables, comparison "
    "coverage) is supplied as ADVISORY evidence: weigh it, but you may accept "
    "an edit whose metric did not improve if the edit is sound and clearly "
    "valuable (e.g. a well-formed boundary node the small test partition "
    "cannot yet measure), and you should SKIP any edit that is unsound — it "
    "leaks ground-truth answers, overfits to named examples instead of "
    "stating a general guideline, targets one judge model's quirks, tells "
    "judges to abstain or defer instead of committing to a label, piles "
    "class- or pair-specific rules into the root file instead of the owning "
    "class/boundary node, is incoherent with the policy's structure, is "
    "mere rewording — restating existing sentences with the same meaning — "
    "or needlessly balloons the policy's prompt mass (see "
    "policy_bundle_size: the bundle is every judge's context, and small "
    "judges measurably collapse under a bloated one — weigh the size delta "
    "against the value delivered). "
    "Diff churn that changes no semantic content, decision boundary, or "
    "objective fact is not an improvement and must be skipped regardless of "
    "structural tidiness. "
    "Respond with JSON only: {\"decision\": \"accept\"|\"skip\", "
    "\"rationale\": \"<=80 words\", \"risk_flags\": [\"...\"]}."
)

# Gate-agent persona: how strict the critic is about shipping (Attila
# 2026-07-09: the default gate read as too strict — skipping metric-flat
# edits with sound structure). The stance paragraph is appended to whichever
# gate system prompt is in force. In `agent` (veto) mode it shifts how
# eagerly the agent vetoes a metric-passing edit; in `agent_only` mode it
# shifts the accept/skip threshold itself.
DEFAULT_GATE_PERSONA = "lenient"
GATE_PERSONAS: dict[str, str] = {
    "lenient": (
        "STANCE — LENIENT: favor progress. A flat or mildly negative metric "
        "on a small test partition is sampling noise, not a defect — do not "
        "treat it alone as a reason to skip. Skip ONLY on a clear defect "
        "(ground-truth leakage, per-image answers, abstain guidance, "
        "root-dumping, incoherent structure, meaning-preserving rewording, "
        "gross prompt bloat) "
        "or a large unambiguous regression across multiple judges. Lenient "
        "means generous about unmeasured VALUE, not about no-op edits: a "
        "diff that only re-phrases existing sentences is still a skip. A "
        "well-formed node addition with plausible generalization value gets "
        "the benefit of the doubt."
    ),
    "moderate": (
        "STANCE — MODERATE: balance progress against risk. Weigh measured "
        "movement and structural value together: accept sound edits whose "
        "overall evidence leans positive; skip when regressions or risky "
        "over-broad wording outweigh the structural gain."
    ),
    "strict": (
        "STANCE — STRICT: be conservative — an accepted edit ships as the "
        "next policy version. Any measured regression, over-broad wording, "
        "or claimed-but-unmeasured value is sufficient reason to skip."
    ),
}


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
    agent_is_sole_gate: bool = False,
    persona: str = DEFAULT_GATE_PERSONA,
    bundle_chars_before: int | None = None,
    bundle_chars_after: int | None = None,
) -> list[dict[str, str]]:
    """Assemble the gate agent's review packet (metric table + diff + anchors).

    ``agent_is_sole_gate`` selects the agent_only system prompt: the critic
    decides, the metric is advisory (--gate-mode agent_only). ``persona``
    appends a leniency stance (lenient/moderate/strict) to the system prompt.
    """
    if persona not in GATE_PERSONAS:
        raise ValueError(
            f"unknown gate persona: {persona!r} (choose from {sorted(GATE_PERSONAS)})"
        )

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

    payload: dict[str, Any] = {
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
    if bundle_chars_before is not None and bundle_chars_after is not None:
        # Hard evidence for the prompt-mass watch: the policy bundle is every
        # judge's context window, and a measured failure mode (2026-07-09) is
        # a 7B judge collapsing to the policy's default branch under a bloated
        # bundle while a 26B still discriminates.
        payload["policy_bundle_size"] = {
            "chars_before": bundle_chars_before,
            "chars_after": bundle_chars_after,
            "delta_chars": bundle_chars_after - bundle_chars_before,
            "approx_tokens_after": bundle_chars_after // 4,
            "note": ("the full bundle is prepended to every judge call; "
                     "unnecessary growth dilutes small judges"),
        }
    base_prompt = GATE_AGENT_ONLY_SYSTEM_PROMPT if agent_is_sole_gate else GATE_SYSTEM_PROMPT
    return [
        {
            "role": "system",
            "content": f"{base_prompt}\n\n{GATE_PERSONAS[persona]}",
        },
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
    *,
    metric_pass: bool,
    agent: dict[str, Any] | None,
    gate_off: bool = False,
    agent_only: bool = False,
) -> dict[str, Any]:
    """Compose the final gate outcome from the rule and the agent's review.

    Truth table (the rule is the hard wall, the agent is a one-way valve):
      gate_off                   -> accept (gate_off; --gate-mode off — the metric
                                   is recorded for the curve, never enforced)
      rule fail + any agent      -> skip  (override_guard if agent said accept)
      rule pass + no agent       -> accept (metric_rule; --gate-mode metric_only)
      rule pass + agent accept   -> accept (gate_agent)
      rule pass + agent skip     -> skip  (gate_agent_veto)

    ``agent_only`` (--gate-mode agent_only) replaces the table: the critic's
    verdict decides regardless of the metric (decided_by gate_agent either
    way); a missing verdict (agent error / unparseable reply) falls back to
    the metric rule so the gate never silently degrades to gate-off.
    """
    if agent_only and not gate_off:
        if agent is not None:
            return {
                "decision": agent["decision"],
                "decided_by": "gate_agent",
                "rationale": agent.get("rationale", ""),
                "risk_flags": agent.get("risk_flags", []),
            }
        return {
            "decision": "accept" if metric_pass else "skip",
            "decided_by": "metric_rule",
            "rationale": (
                "agent_only gate produced no verdict (agent error); fell back "
                "to the metric rule"
            ),
            "risk_flags": ["gate_agent_unavailable"],
        }
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
    "You are given two sets of anchors from this cycle's train batch — as "
    "each judge's full text output plus the SME ground truth, and sometimes "
    "also the anchor images themselves: MISALIGNED anchors (the panel "
    "disagreed with the SME ground truth — the errors to fix) and ALIGNED "
    "anchors (the panel "
    "classified these CORRECTLY — the successes to protect). Draft the "
    "SINGLE most impactful policy improvement that would fix the misaligned "
    "cases WITHOUT regressing the aligned ones — use the aligned anchors as "
    "positive references for what the current policy already gets right, so "
    "your edit sharpens the boundary instead of over-correcting past it. "
    "Write minimal full-file markdown changes. HARD BUDGET: at most "
    "{max_changes} file changes total (modified + added + removed combined); "
    "fewer is better — one focused, human-reviewable change is ideal. "
    "{area_guidance} State "
    "general, model-agnostic guidance (clear definitions, boundary rules, "
    "canonical examples); NEVER encode per-image answers or ground-truth "
    "labels. DO NOT REWORD: when you modify an existing file, leave every "
    "sentence you are not semantically changing byte-for-byte intact — never "
    "re-phrase, reorder, or polish text whose meaning stays the same. Touch "
    "a sentence only to change its semantic meaning, tighten a decision "
    "boundary, or clarify an objective fact; paraphrase-only churn wastes "
    "the review diff and the gate will skip it. "
    "PROMPT BUDGET: the policy bundle is prepended to EVERY judge call — it "
    "is the judges' entire context, and small judges measurably collapse "
    "under a bloated bundle. Keep new text tight (a new node's body well "
    "under ~250 words), never duplicate guidance that already lives in "
    "another node (cite that node instead), and DELETING redundant or "
    "duplicated guidance is a legitimate edit when you state that as its "
    "purpose — deletion trims prompt mass; re-phrasing does not. "
    "The policy must always demand a decisive label: NEVER add "
    "guidance telling judges to abstain, defer, or decline, and NEVER add "
    "or reintroduce an 'abstain'/'unknown'/'uncertain' label, node, or "
    "routing rule — uncertainty "
    "belongs in the confidence score [0,1] and the difficulty rating, not in "
    "refusing to answer. "
    "Keep each file's YAML frontmatter (id, version, title, area, "
    "node_type, polarity, parent, status, edges) intact and consistent. "
    "Return JSON only: {{\"files\":[{{\"path\":\"name.md\",\"change\":"
    "\"modified|added|removed\",\"content\":\"full markdown for "
    "added/modified files\"}}]}}. Never return unified diffs."
)

# Node-targeting guidance is area-specific: MNIST grows per-digit and
# confusion-pair boundary nodes; the binary GenAI area grows a knowledge
# graph of evidence sub-categories (cue nodes) under their parent category.
MNIST_DRAFTER_GUIDANCE = (
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
    "instead of a longer root."
)
GENAI_DRAFTER_GUIDANCE = (
    "TARGET THE MOST SPECIFIC NODE THAT OWNS THE ERROR: cue-specific "
    "guidance belongs in that cue's own node file (e.g. "
    "GA.visual_artifacts.anatomy.hands.md), and guidance about one recurring "
    "evidence pattern belongs in a dedicated sub-category node under its "
    "parent category. The root file is the decision procedure — treat it as "
    "effectively frozen and touch it only when no more specific node can "
    "carry the rule; piling rules into the root is a defect, not an "
    "improvement. When several anchors share one recurring cue (a texture "
    "signature, an anatomy artifact, a lighting/geometry inconsistency, a "
    "provenance tell, or an authentic-photo pattern the panel misreads as "
    "generated), PREFER ADDING a new sub-category node — a file named like "
    "GA.<category>.<specific_cue>.md whose frontmatter mirrors a sibling "
    "node's shape with node_type 'category', parent set to the owning "
    "category node id (or GA.root only for a genuinely new category), "
    "polarity 'positive' for generated-image cues or 'negative' for "
    "authenticity cues, status 'draft', and a frontmatter edge of type "
    "'subtype_of' pointing at the parent — so the knowledge graph grows "
    "deeper, reviewable sub-categories instead of a longer root."
)


def drafter_system_prompt(*, area: str, max_changes: int) -> str:
    """Render the drafter system prompt with area-appropriate node guidance."""
    guidance = (
        MNIST_DRAFTER_GUIDANCE if area == "MNIST_Digits" else GENAI_DRAFTER_GUIDANCE
    )
    return DRAFTER_SYSTEM_PROMPT.format(
        max_changes=max_changes, area_guidance=guidance
    )


def _anchor_sample_block(anchor: dict[str, Any]) -> dict[str, Any]:
    """Compact per-anchor record for the drafter text payload."""
    return {
        "image_id": anchor.get("image_id"),
        "sme_truth": anchor.get("sme_truth"),
        "misalignment_type": anchor.get("misalignment_type"),
        "severity": anchor.get("severity"),
        "votes": [
            {
                "model": _common.labeler_id_for(v),
                "label": v.get("label"),
                "confidence": v.get("confidence"),
                "is_boundary": v.get("is_boundary"),
                "difficulty": v.get("difficulty"),
                "justification": str(v.get("justification", ""))[:400],
            }
            for v in anchor.get("votes", [])
        ],
    }


def _append_image_parts(
    parts: list[dict[str, Any]], images: list[dict[str, Any]], *,
    provider: str, lead: str,
) -> None:
    """Append a labeled group of image blocks (provider-shaped) to ``parts``."""
    if not images:
        return
    parts.append({"type": "text", "text": lead})
    for image in images:
        parts.append({"type": "text", "text": (
            f"anchor {image.get('image_id')} — SME truth: {image.get('sme_truth')}"
        )})
        mime = image.get("mime_type") or "image/jpeg"
        if provider == "anthropic":
            parts.append({"type": "image", "source": {
                "type": "base64", "media_type": mime, "data": image["b64"],
            }})
        else:
            parts.append({"type": "image_url", "image_url": {
                "url": f"data:{mime};base64,{image['b64']}",
            }})


def build_drafter_messages(
    *,
    policy_markdown: str,
    base_version: str,
    area: str,
    anchors: list[dict[str, Any]],
    max_changes: int,
    k: int,
    aligned_anchors: list[dict[str, Any]] | None = None,
    anchor_images: list[dict[str, Any]] | None = None,
    aligned_images: list[dict[str, Any]] | None = None,
    provider: str = "openai",
) -> list[dict[str, Any]]:
    """Assemble the drafter packet: current bundle + both anchor sets + budget.

    Two anchor sets (Attila 2026-07-07: "aligned and misaligned anchors"):
    ``anchors`` are the panel's misalignments (the SVM/boundary errors to
    fix); ``aligned_anchors`` are a sample of correctly-classified images
    (the successes to protect). Both go to the drafter as text (compact vote
    records) and — when the corresponding ``*_images`` are supplied — as the
    actual images, so the drafter SEES what the panel misread AND what it got
    right instead of reasoning from justifications alone. Each image entry:
    ``{"image_id", "sme_truth", "mime_type", "b64"}``.

    When either image set is provided the user message content becomes a
    provider-shaped parts array (OpenAI ``image_url`` data URLs / Anthropic
    base64 ``image`` blocks — both chat transports forward user content
    verbatim). Without any images the message stays a plain string (dry runs,
    old tests).
    """
    aligned_anchors = aligned_anchors or []
    # Prompt-caching prefix discipline: the stable content (task + policy
    # bundle, unchanged between cycles unless an edit was accepted) leads;
    # the volatile content (cycle number, anchors) follows. Providers cache
    # by byte prefix, so this ordering lets repeat drafter calls re-read the
    # multi-thousand-token policy instead of re-paying for it.
    stable_payload: dict[str, Any] = {
        "task": (
            "Improve the policy so ALL judges (and human raters) decide the "
            "MISALIGNED samples correctly, WITHOUT regressing the ALIGNED "
            "samples (already correct) or the rest of the set."
        ),
        "area": area,
        "policy_markdown": policy_markdown,
    }
    volatile_payload: dict[str, Any] = {
        "base_version": base_version,
        "cycle_k": k,
        "max_changes": max_changes,
        "misaligned_samples": [_anchor_sample_block(a) for a in anchors],
        "aligned_samples": [_anchor_sample_block(a) for a in aligned_anchors],
    }
    system = {
        "role": "system",
        "content": drafter_system_prompt(area=area, max_changes=max_changes),
    }
    if not anchor_images and not aligned_images:
        # Single JSON string (dry runs, old tests): key order is
        # stable-first, and json.loads consumers are order-agnostic.
        return [
            system,
            {"role": "user", "content": json.dumps({**stable_payload, **volatile_payload}, indent=2)},
        ]

    stable_part: dict[str, Any] = {
        "type": "text",
        "text": json.dumps(stable_payload, indent=2),
    }
    if provider == "anthropic":
        # Explicit breakpoint: system + this block (the policy bundle) is the
        # reusable prefix across retries and unchanged-policy cycles.
        stable_part["cache_control"] = {"type": "ephemeral"}
    parts: list[dict[str, Any]] = [
        stable_part,
        {"type": "text", "text": json.dumps(volatile_payload, indent=2)},
    ]
    _append_image_parts(
        parts, anchor_images or [], provider=provider,
        lead=("The MISALIGNED anchor images follow, in order — the panel got "
              "these WRONG. Look at each: the visual evidence outranks any "
              "judge's justification."),
    )
    _append_image_parts(
        parts, aligned_images or [], provider=provider,
        lead=("The ALIGNED anchor images follow — the panel classified these "
              "CORRECTLY. Use them as positive references; your edit must not "
              "regress them."),
    )
    return [system, {"role": "user", "content": parts}]


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
                "max_aligned_anchors", "epsilon", "strategy", "gate_mode",
                "gate_model", "drafter_model", "judge_models", "concurrency",
                "dry_run",
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


# ---------------------------------------------------------------------------
# SME re-adjudication queue (Attila 2026-07-06): "the remaining training and
# testing (+ validation) images that still have misalignments should be
# flagged for re-adjudication by a human SME", stack-ranked by consensus /
# confidence / difficulty averaged across judges — and by the per-sample
# gradient formalism (rush.sample_gradient: p = c if correct else 1-c,
# |g| = 1-p, loss = -ln p; confident-wrong panels are the strongest signal
# that either the policy or the golden label itself needs a human look).


DIFFICULTY_SCORE = {"low": 0.0, "medium": 0.5, "high": 1.0}


def vote_gradient(vote: dict[str, Any], truth: str) -> dict[str, float] | None:
    """Per-vote gradient signals; None for abstains / missing confidence
    (excluded by definition, mirroring the rush.sample_gradient view)."""
    label = str(vote.get("label") or "")
    confidence = vote.get("confidence")
    if not label or label == _common.ABSTAIN or confidence is None:
        return None
    c = min(1.0, max(0.0, float(confidence)))
    p = c if label == str(truth) else 1.0 - c
    return {
        "p_true": p,
        "magnitude": 1.0 - p,
        "hessian": c * (1.0 - c),
        "loss": -math.log(max(p, 1e-6)),
    }


def _avg(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 6) if values else None


# --- The importance formalism (Attila 2026-07-07) --------------------------
# A multi-LLM judge over item i produces TWO alignment signals the single-vote
# gradient (p, |g|) cannot express:
#
#   SME agreement  a = (#judges whose label == SME truth) / N_decisive
#                      — how aligned the panel is with the HUMAN label.
#                      Misalignment m = 1 - a.
#   LLM consensus  k = (#judges on the modal label) / N_decisive
#                      — how much the LLMs agree with EACH OTHER (SME-blind).
#
# Consensus flips meaning with alignment, giving the four-tier hierarchy:
#   T1 misaligned + high k  (unanimous & wrong — systematic; the WORST)
#   T2 misaligned + low  k  (split & wrong)
#   T3 aligned    + low  k  (right but the panel argued — still instructive)
#   T4 aligned    + high k  (unanimous & right — the ideal state)
# Base importance reproduces that ordering continuously:
#   I_base = m + k*(2m - 1)   in [-1, 2]      (normalized -> [0,1])
# Two derived scores then amplify by the panel's confidence (mean |g|) and its
# boundary rate; re-adjudication additionally fades with the human-label
# confidence, because a re-confirmed golden label barely needs another look.
# k threshold for the discrete tier badge. STRICT (k > 0.5): a plurality that
# is only half the panel is a TIE, not consensus — an even split (1-1, 2-2)
# lands in the low-consensus tier, not the high one.
CONSENSUS_HIGH = 0.5
GRAD_WEIGHT = 1.0        # confidence amplifier: x (1 + w * mean|g|)
BOUNDARY_WEIGHT = 0.5    # boundary amplifier:   x (1 + w * boundary_rate)


def human_confidence(sme_confirmations: int = 1) -> float:
    """p_human = 1 - 1/(m + 0.2), m = # SME confirmations of the golden label.

    HIS formula: m=1 (default) -> 0.167, m=2 -> 0.545, m=3 -> 0.688; m=0 -> 0.
    Re-adjudication priority is multiplied by (1 - p_human), so a re-confirmed
    label drops toward zero priority.
    """
    m = max(0, int(sme_confirmations))
    return 0.0 if m == 0 else round(1.0 - 1.0 / (m + 0.2), 6)


def importance_scores(*, sme_fraction, consensus_fraction, majority_aligned,
                      mean_grad, boundary_rate, sme_confirmations=1):
    """The four-tier hierarchy as a continuous score plus a discrete tier."""
    a = sme_fraction if sme_fraction is not None else 0.0   # all-abstain -> 0
    k = consensus_fraction if consensus_fraction is not None else 0.0
    m = 1.0 - a
    base = (m + k * (2.0 * m - 1.0) + 1.0) / 3.0            # [-1,2] -> [0,1]
    amp = (1.0 + GRAD_WEIGHT * (mean_grad or 0.0)) * (1.0 + BOUNDARY_WEIGHT * (boundary_rate or 0.0))
    anchor = base * amp
    h = human_confidence(sme_confirmations)
    high_consensus = k > CONSENSUS_HIGH   # strict: an even-split tie is NOT consensus
    if majority_aligned:
        tier = 4 if high_consensus else 3
    else:
        tier = 1 if high_consensus else 2
    return {
        "base": round(base, 6),
        "tier": tier,
        "anchor": round(anchor, 6),                        # policy-learning value
        "readjudication": round(anchor * (1.0 - h), 6),    # human-review priority
        "human_confidence": h,
    }


def panel_signal(record: dict[str, Any]) -> dict[str, Any]:
    """Per-image rollup across the judge panel — the stack-rank substrate.

    Mirrors rush.panel_signal and adds the multi-LLM alignment layer. Two
    distinct agreement signals are reported: ``consensus`` is strictly
    LLM<->LLM agreement (SME-blind — the fraction on the modal label);
    ``sme_agreement`` is alignment with the human label (graded m/N, its sign
    the majority-vote collapse used for accounting). Plus a boundary rate and
    the four-tier importance scores (see ``importance_scores``).
    """
    truth = str(record.get("sme_truth"))
    votes = record.get("votes") or []
    n_judges = len(votes)
    decisive = [v for v in votes
                if v.get("label") and str(v.get("label")) != _common.ABSTAIN]
    n_dec = len(decisive)
    counts = Counter(str(v["label"]) for v in decisive)
    top = counts.most_common()
    tie = len(top) > 1 and top[0][1] == top[1][1]
    majority_count = top[0][1] if top else 0
    majority_label = top[0][0] if top and not tie else None
    majority_aligned = majority_label is not None and majority_label == truth

    n_agree = sum(1 for v in decisive if str(v.get("label")) == truth)
    sme_fraction = round(n_agree / n_dec, 6) if n_dec else None
    consensus_fraction = round(majority_count / n_dec, 6) if n_dec else None
    boundary_votes = sum(1 for v in votes if v.get("is_boundary"))
    boundary_rate = round(boundary_votes / n_judges, 6) if n_judges else 0.0

    grads = [g for g in (vote_gradient(v, truth) for v in decisive) if g]
    mean_grad = _avg([g["magnitude"] for g in grads])
    importance = importance_scores(
        sme_fraction=sme_fraction, consensus_fraction=consensus_fraction,
        majority_aligned=majority_aligned, mean_grad=mean_grad,
        boundary_rate=boundary_rate,
        sme_confirmations=int(record.get("sme_confirmations") or 1),
    )
    return {
        "n_judges": n_judges,
        "majority_label": majority_label,
        "majority_aligned": majority_aligned,
        # LLM<->LLM agreement (SME-blind).
        "consensus": {
            "decisive": n_dec,
            "majority_count": majority_count,
            "fraction": consensus_fraction,
            "tie": tie,
        },
        # LLM<->SME agreement (graded m/N; the majority collapse is the sign).
        "sme_agreement": {
            "n_agree": n_agree,
            "decisive": n_dec,
            "fraction": sme_fraction,
            "majority_aligned": majority_aligned,
        },
        "avg_confidence": _avg([float(v["confidence"]) for v in decisive
                                if v.get("confidence") is not None]),
        "difficulty_score": _avg([DIFFICULTY_SCORE[v["difficulty"]] for v in votes
                                  if v.get("difficulty") in DIFFICULTY_SCORE]),
        "gradient": {
            "n": len(grads),
            "avg_magnitude": mean_grad,
            "max_magnitude": (round(max(g["magnitude"] for g in grads), 6)
                              if grads else None),
            "avg_hessian": _avg([g["hessian"] for g in grads]),
            "avg_loss": _avg([g["loss"] for g in grads]),
        },
        "boundary_rate": boundary_rate,
        "any_boundary": boundary_votes > 0,
        "boundary_pairs": sorted({
            "↔".join(str(d) for d in (v.get("is_boundary_between") or []))
            for v in votes if v.get("is_boundary") and v.get("is_boundary_between")
        }),
        "importance": importance,
    }


def readjudication_sort_key(item: dict[str, Any]) -> tuple[float, str]:
    """Default stack rank: the four-tier importance, most valuable first.

    Misaligned-with-high-LLM-consensus (T1) leads — the panel is confidently,
    unanimously wrong about the human label: the single most valuable case for
    both re-adjudication and policy learning.
    """
    score = (item.get("importance") or {}).get("readjudication")
    return (-(score if score is not None else 0.0), str(item.get("image_id") or ""))


def build_readjudication(
    state: dict[str, Any],
    *,
    load_misalignment: Callable[[str], list[dict[str, Any]]],
    sha_by_image: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Flag every image whose LATEST evaluation in this run is still
    misaligned (anything but all_agree) for human SME re-adjudication.

    Latest evidence per image: train batches under the policy in force at
    their cycle k (sample_train_batch may RE-USE a train image once the pool
    runs short — later cycles overwrite, and an all_agree re-judgment CLEARS
    a stale flag); the test partition under the final policy (last accepted
    candidate eval, else the k=0 baseline); holdout and benchmark under
    their final leg, falling back to the paid start leg when a stop/failure
    interrupted the run before the final pass.
    """
    area = state.get("area") or ""
    cycles = [c for c in state.get("cycles", []) if isinstance(c.get("k"), int)]
    sources: list[dict[str, Any]] = []
    for cycle in cycles:
        if cycle.get("train_run_id"):
            sources.append({"kind": "train", "k": cycle["k"],
                            "run_id": cycle["train_run_id"],
                            "policy": cycle.get("generator_before")})
    accepted = [c for c in cycles
                if c.get("status") == "accepted" and c.get("candidate_run_id")]
    if accepted:
        last = accepted[-1]
        sources.append({"kind": "test", "k": last["k"],
                        "run_id": last["candidate_run_id"],
                        "policy": last.get("generator_after")})
    else:
        base = next((c for c in cycles if c["k"] == 0 and c.get("test_run_id")), None)
        if base:
            sources.append({"kind": "test", "k": 0, "run_id": base["test_run_id"],
                            "policy": base.get("generator_before")})
    for kind in ("holdout", "benchmark"):
        block = state.get(kind) or {}
        # 'final' is aliased to 'start' on completed runs; a stopped/failed
        # run may hold only the paid start leg — still its latest evidence.
        leg = block.get("final") or block.get("start") or {}
        if leg.get("run_id"):
            version = leg.get("version")
            sources.append({"kind": kind, "k": None, "run_id": leg["run_id"],
                            "policy": f"{area}.{version}" if version else None})

    items_by_image: dict[str, dict[str, Any]] = {}
    scanned = 0
    for source in sources:
        records = load_misalignment(source["run_id"])
        source["n_scanned"] = len(records)
        source["n_flagged"] = 0
        for record in records:
            scanned += 1
            mis_type = record.get("misalignment_type")
            if not mis_type or mis_type == "all_agree":
                # A later re-judgment that fully agrees CLEARS a stale flag
                # (train ids can be re-used across cycles once the pool runs
                # short — only the LATEST evaluation decides).
                items_by_image.pop(record.get("image_id"), None)
                continue
            source["n_flagged"] += 1
            image_id = record.get("image_id")
            items_by_image[image_id] = {
                "image_id": image_id,
                "sha256": (sha_by_image or {}).get(image_id),
                "repo_rel_path": record.get("repo_rel_path"),
                "split": record.get("split"),
                "sme_truth": record.get("sme_truth"),
                "misalignment_type": mis_type,
                "severity": record.get("severity"),
                "source": {key: source.get(key)
                           for key in ("kind", "k", "run_id", "policy")},
                **panel_signal(record),
                "votes": [{
                    "model": v.get("model_id") or v.get("labeler_id"),
                    "label": v.get("label"),
                    "confidence": v.get("confidence"),
                    "difficulty": v.get("difficulty"),
                    "is_boundary": bool(v.get("is_boundary")),
                } for v in record.get("votes") or []],
            }

    items = sorted(items_by_image.values(), key=readjudication_sort_key)
    return {
        "generated_at": utcnow_iso(),
        "final_version": state.get("current_version"),
        "n_scanned": scanned,
        "n_flagged": len(items),
        "sources": [{k: s.get(k) for k in
                     ("kind", "k", "run_id", "policy", "n_scanned", "n_flagged")}
                    for s in sources],
        "items": items,
    }


# --- SME re-adjudication actions (Attila 2026-07-07) -----------------------
# The queue is not read-only: an SME can CONFIRM the golden label (the LLMs
# were wrong — raises human confidence, fades the item), OVERTURN it to a new
# label (the golden set was wrong — the item is re-scored against the new
# truth, and if the panel now agrees it drops out of the misaligned queue), or
# mark it UNCERTAIN (needs another look). Actions append to a portable
# JSONL log — file = truth, same as everything else — read back at query time.
VERDICTS = ("confirm", "overturn", "uncertain")


def adjudication_log_path(repo_root: Path | str) -> Path:
    return Path(repo_root) / "data" / "adjudication_reviews.jsonl"


def record_adjudication(
    repo_root: Path | str, *, area: str, key: str, image_id: str | None,
    verdict: str, prior_truth: str | None = None, new_label: str | None = None,
    reviewer: str = "sme", comment: str = "",
) -> dict[str, Any]:
    """Append one SME re-adjudication event to the portable review log."""
    if verdict not in VERDICTS:
        raise ValueError(f"verdict must be one of {VERDICTS}: {verdict!r}")
    if verdict == "overturn" and not new_label:
        raise ValueError("overturn requires a new_label")
    record = {
        "recorded_at": utcnow_iso(),
        "area": area,
        "key": key,
        "image_id": image_id,
        "verdict": verdict,
        "prior_truth": prior_truth,
        "new_label": new_label if verdict == "overturn" else None,
        "reviewer": (reviewer or "sme")[:80],
        "comment": (comment or "")[:2000],
    }
    path = adjudication_log_path(repo_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, sort_keys=True) + "\n")
    return record


def load_adjudications(
    repo_root: Path | str, *, area: str | None = None
) -> dict[str, list[dict[str, Any]]]:
    """Review events grouped by item key, time-sorted (oldest first)."""
    path = adjudication_log_path(repo_root)
    by_key: dict[str, list[dict[str, Any]]] = defaultdict(list)
    if not path.exists():
        return by_key
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if area and rec.get("area") != area:
            continue
        if rec.get("key"):
            by_key[rec["key"]].append(rec)
    for events in by_key.values():
        events.sort(key=lambda r: str(r.get("recorded_at") or ""))
    return by_key


def _fold_reviews(reviews: list[dict[str, Any]], seed_label: str | None):
    """Collapse an item's review history into (effective_label,
    sme_confirmations, resolution). The seed golden label counts as one
    human's assertion; each confirm adds one; an overturn resets to the new
    label with the overturning SME as its sole confirmer."""
    effective = seed_label
    confirmations = 1
    resolution = "open"
    for r in reviews:
        verdict = r.get("verdict")
        if verdict == "confirm":
            confirmations += 1
            resolution = "confirmed"
        elif verdict == "overturn":
            effective = r.get("new_label") or effective
            confirmations = 1
            resolution = "overturned"
        elif verdict == "uncertain":
            resolution = "uncertain"
    return effective, confirmations, resolution


def _recompute_importance(votes, effective_label, sme_confirmations):
    """Re-score the panel against a new golden label (after an overturn)."""
    record = {
        "sme_truth": effective_label,
        "sme_confirmations": sme_confirmations,
        "votes": [{
            "label": v.get("label"), "confidence": v.get("confidence"),
            "difficulty": v.get("difficulty"), "is_boundary": v.get("is_boundary"),
        } for v in (votes or [])],
    }
    return panel_signal(record)


def _read_time_misalignment_loader(
    repo_root: Path | str,
) -> Callable[[str], list[dict[str, Any]]]:
    """Read-time twin of the driver's ``_load_misalignment``: pull the scored
    misalignment records straight from ``data/runs/<run_id>/scoring/``."""
    runs_root = Path(repo_root) / "data" / "runs"

    def _load(run_id: str) -> list[dict[str, Any]]:
        path = runs_root / str(run_id) / "scoring" / "misalignment.json"
        if not path.exists():
            return []
        try:
            return json.loads(path.read_text(encoding="utf-8")).get("records", [])
        except (json.JSONDecodeError, OSError):
            return []

    return _load


def _sha_by_image(repo_root: Path | str, area: str | None) -> dict[str, str]:
    """sample_id -> sha256 from the area's manifest (the dedupe identity the
    driver stamps on flagged items). Empty on any load failure — the queue
    then dedupes by area:image_id, which is still correct within one area."""
    try:
        from pipeline.io_paths import MNIST_SAMPLE_MANIFEST, genai_manifest_default
        from pipeline.manifest import load_records
        from pipeline.web.demo_area import MNIST_POLICY_AREA

        manifest = (
            MNIST_SAMPLE_MANIFEST if area == MNIST_POLICY_AREA
            else genai_manifest_default()
        )
        return {r.sample_id: r.sha256 for r in load_records(manifest)}
    except Exception:  # noqa: BLE001 - enrichment only, never block the queue
        return {}


def aggregate_readjudication(
    repo_root: Path | str,
    *,
    area: str | None = None,
    include_dry: bool = False,
) -> dict[str, Any]:
    """The cross-run running list of items awaiting SME re-adjudication.

    Derived at read time from every experiment.json readjudication block
    (file = truth; no merge state to corrupt). A run whose driver died before
    end-of-run flagging (killed, stuck "running", benchmark crash) has no
    block — for those, the queue is REBUILT here from the run's misalignment
    records on disk, so any live run with recorded labels contributes its
    residuals. Items are deduped by sha256
    (the label-store entity identity; sample_id fallback) with the run
    numbers that flagged them unioned, per-run evidence kept, and the rank
    signals averaged across runs. Dry runs are excluded by default — their
    deterministic fake votes would pollute a queue meant for humans.
    """
    root = experiments_root(repo_root)
    reviews_by_key = load_adjudications(repo_root, area=area)
    load_misalignment = _read_time_misalignment_loader(repo_root)
    sha_maps: dict[str, dict[str, str]] = {}
    grouped: dict[str, dict[str, Any]] = {}
    if root.is_dir():
        for entry in sorted(root.iterdir()):
            if not entry.is_dir() or entry.name.startswith("_"):
                continue
            path = entry / "experiment.json"
            if not path.exists():
                continue
            try:
                state = json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
            if area and state.get("area") != area:
                continue
            if state.get("dry_run") and not include_dry:
                continue
            block = state.get("readjudication") or {}
            if not block.get("items"):
                run_area = str(state.get("area") or "")
                if run_area not in sha_maps:
                    sha_maps[run_area] = _sha_by_image(repo_root, run_area)
                try:
                    block = build_readjudication(
                        state,
                        load_misalignment=load_misalignment,
                        sha_by_image=sha_maps[run_area],
                    )
                except Exception:  # noqa: BLE001 - one bad run must not empty the queue
                    block = {}
            for item in block.get("items") or []:
                key = item.get("sha256") or f"{state.get('area')}:{item.get('image_id')}"
                group = grouped.setdefault(key, {
                    "key": key,
                    "image_id": item.get("image_id"),
                    "sha256": item.get("sha256"),
                    "repo_rel_path": item.get("repo_rel_path"),
                    "split": item.get("split"),
                    "sme_truth": item.get("sme_truth"),
                    "area": state.get("area"),
                    "runs": [],
                })
                source = item.get("source") or {}
                group["runs"].append({
                    "run_number": state.get("run_number"),
                    "experiment_id": state.get("experiment_id"),
                    "started_at": state.get("started_at"),
                    "kind": source.get("kind"),
                    "k": source.get("k"),
                    "run_id": source.get("run_id"),
                    "policy": source.get("policy"),
                    "misalignment_type": item.get("misalignment_type"),
                    "severity": item.get("severity"),
                    "majority_label": item.get("majority_label"),
                    "majority_aligned": item.get("majority_aligned"),
                    "n_judges": item.get("n_judges"),
                    "consensus": item.get("consensus"),
                    "sme_agreement": item.get("sme_agreement"),
                    "avg_confidence": item.get("avg_confidence"),
                    "difficulty_score": item.get("difficulty_score"),
                    "gradient": item.get("gradient"),
                    "boundary_rate": item.get("boundary_rate"),
                    "any_boundary": item.get("any_boundary"),
                    "boundary_pairs": item.get("boundary_pairs"),
                    "importance": item.get("importance"),
                    "votes": item.get("votes"),
                })

    def _mean(runs: list[dict[str, Any]], pick: Callable[[dict], Any]) -> float | None:
        values = [pick(r) for r in runs]
        values = [v for v in values
                  if isinstance(v, (int, float)) and not isinstance(v, bool)]
        return round(sum(values) / len(values), 6) if values else None

    # Full-evidence enrichment: queue blocks store votes stripped to five
    # fields (experiment.json stays lean), but the SME drawer needs the whole
    # LLM response — justification, l2_label, citations, quotes, boundary
    # pair, tokens, cost. Re-attach it to each item's LATEST run from that
    # run's misalignment.json, one read per distinct run id.
    full_votes_cache: dict[str, dict[str, list[dict[str, Any]]]] = {}

    def _full_votes(run_id: str | None, image_id: str | None) -> list[dict[str, Any]] | None:
        if not run_id or not image_id:
            return None
        if run_id not in full_votes_cache:
            full_votes_cache[run_id] = {
                str(rec.get("image_id")): rec.get("votes") or []
                for rec in load_misalignment(str(run_id))
            }
        votes = full_votes_cache[run_id].get(str(image_id))
        if not votes:
            return None
        return [{**v, "model": v.get("labeler_id") or v.get("model_id")}
                for v in votes]

    items: list[dict[str, Any]] = []
    for group in grouped.values():
        runs = sorted(group["runs"],
                      key=lambda r: (r.get("run_number") or 0,
                                     str(r.get("started_at") or "")))
        group["runs"] = runs
        group["n_runs"] = len(runs)
        group["run_numbers"] = sorted({r.get("run_number") for r in runs
                                       if r.get("run_number") is not None})
        group["latest"] = runs[-1]
        enriched = _full_votes(runs[-1].get("run_id"), group.get("image_id"))
        if enriched:
            runs[-1]["votes"] = enriched
        group["agg"] = {
            "importance": _mean(runs, lambda r: (r.get("importance") or {}).get("readjudication")),
            "anchor": _mean(runs, lambda r: (r.get("importance") or {}).get("anchor")),
            "sme_fraction": _mean(runs, lambda r: (r.get("sme_agreement") or {}).get("fraction")),
            "consensus_fraction": _mean(runs, lambda r: (r.get("consensus") or {}).get("fraction")),
            "avg_confidence": _mean(runs, lambda r: r.get("avg_confidence")),
            "difficulty_score": _mean(runs, lambda r: r.get("difficulty_score")),
            "grad_magnitude": _mean(runs, lambda r: (r.get("gradient") or {}).get("avg_magnitude")),
            "loss": _mean(runs, lambda r: (r.get("gradient") or {}).get("avg_loss")),
            "boundary_rate": _mean(runs, lambda r: r.get("boundary_rate")),
            "any_boundary": any(r.get("any_boundary") for r in runs),
            # Worst (lowest-numbered) tier the item ever hit — the reason to look.
            "worst_tier": min((r.get("importance") or {}).get("tier") or 4 for r in runs),
        }

        # Fold in SME re-adjudication: confirm fades via human confidence,
        # overturn re-scores the panel against the new golden label.
        reviews = reviews_by_key.get(group["key"], [])
        effective_label, sme_conf, resolution = _fold_reviews(reviews, group["sme_truth"])
        h = human_confidence(sme_conf)
        agg = group["agg"]
        agg["human_confidence"] = h
        overturned = resolution == "overturned" and effective_label != group["sme_truth"]
        if overturned:
            recomputed = _recompute_importance(group["latest"].get("votes"),
                                               effective_label, sme_conf)
            imp = recomputed.get("importance") or {}
            agg["effective_importance"] = imp.get("readjudication")
            agg["recomputed"] = {
                "sme_fraction": (recomputed.get("sme_agreement") or {}).get("fraction"),
                "consensus_fraction": (recomputed.get("consensus") or {}).get("fraction"),
                "tier": imp.get("tier"),
                "majority_aligned": recomputed.get("majority_aligned"),
            }
        else:
            anchor = agg.get("anchor")
            agg["effective_importance"] = (round(anchor * (1.0 - h), 6)
                                           if isinstance(anchor, (int, float)) else agg.get("importance"))
        group["review"] = {
            "resolution": resolution,
            # Resolved = TWO+ humans agree on the current effective label
            # (HIS confidence tiers: 1 SME = default/open, 2 = re-confirmed).
            # A lone overturn (m=1) is one SME's new opinion — it stays OPEN
            # for a second SME to confirm, even if the panel is still wrong.
            "resolved": sme_conf >= 2,
            "effective_label": effective_label,
            "overturned_from": group["sme_truth"] if overturned else None,
            "sme_confirmations": sme_conf,
            "count": len(reviews),
            "events": reviews[-6:],
        }
        items.append(group)

    # Default order = the effective four-tier importance after SME actions
    # (re-confirmed items fade via human confidence); re-sortable client-side.
    def _eff(g):
        v = g["agg"].get("effective_importance")
        return v if isinstance(v, (int, float)) else -1.0
    items.sort(key=lambda g: (-_eff(g), str(g.get("image_id") or "")))
    return {
        "generated_at": utcnow_iso(),
        "area": area,
        "n_items": len(items),
        "n_open": sum(1 for g in items if not g["review"]["resolved"]),
        "items": items,
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


def _system_leg_readout(block: dict[str, Any] | None) -> dict[str, Any] | None:
    """Compact start→final system readout of a holdout/benchmark block.

    Feeds the cross-run Benchmarks tab: one small dict per run instead of the
    full per-judge metric tree.
    """
    if not isinstance(block, dict) or not isinstance(block.get("start"), dict):
        return None

    def _system(leg: dict[str, Any] | None) -> dict[str, Any]:
        return ((leg or {}).get("metrics") or {}).get("system") or {}

    start, final = block["start"], block.get("final")
    return {
        "n": block.get("n"),
        "start_version": start.get("version"),
        "final_version": (final or {}).get("version"),
        "start_macro_f1": _system(start).get("macro_f1"),
        "final_macro_f1": _system(final).get("macro_f1"),
        "start_accuracy": _system(start).get("accuracy"),
        "final_accuracy": _system(final).get("accuracy"),
    }


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
                "gate_mode": state.get("gate_mode"),
                "gate_persona": state.get("gate_persona"),
                "drafter_model": state.get("drafter_model"),
                "drafter_context": state.get("drafter_context"),
                "strategy": state.get("strategy"),
                "batch_n": state.get("batch_n"),
                "test_n": state.get("test_n"),
                "max_anchors": state.get("max_anchors"),
                "max_aligned_anchors": state.get("max_aligned_anchors"),
                "max_changes": state.get("max_changes"),
                "base_version": state.get("base_version"),
                "current_version": state.get("current_version"),
                "dry_run": state.get("dry_run", False),
                "cost_usd_total": state.get("cost_usd_total"),
                "started_at": state.get("started_at"),
                "finished_at": state.get("finished_at"),
                # Compact start→final system readouts for the Benchmarks tab.
                "benchmark": _system_leg_readout(state.get("benchmark")),
                "holdout": _system_leg_readout(state.get("holdout")),
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
