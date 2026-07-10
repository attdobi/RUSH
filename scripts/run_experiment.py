#!/usr/bin/env python3
"""Turn the RUSH crank: one seeded, numbered, PPO-gated policy-iteration run.

The experiment is the demo-as-instrument (Attila 2026-07-06): pick the judge
panel, initialize the generator (the policy prompt at k=0), then run k_max
cycles of

    label N seeded train images  ->  S1 random misalignment anchors
    ->  ONE clipped policy edit (1..max_changes node files)
    ->  evaluate candidate on the FIXED seeded test partition
    ->  PPO gate: auto-accept iff test system macro-F1 improves
        (gate agent gpt-5.5 reviews and may veto; it can never force)
    ->  accept: the candidate becomes the next policy-graph version
        skip:   the proposal is archived, the version stays

while recording accuracy / F1 / precision / recall / FPR / FNR — per judge
AND for the system of judges, on train and test — at every cycle, plus every
gate decision (for later SME review = RLHF data for the critic agent).

Split discipline: the test partition is carved once per experiment from
dev_golden with the master seed (formally a validation set — the gate adapts
to it); the 500-image holdout stays locked and is only scored under the start
and final versions with --holdout-final.

Everything a fresh clone needs lives in data/experiments/<id>/; the Postgres
rush schema mirrors it best-effort for cross-experiment analysis.

Examples
--------

Offline dry run (no network, deterministic fake judges — exercises the whole
loop shape; fake labels are policy-independent so every gate skips)::

    python scripts/run_experiment.py --area MNIST_Digits \
        --models openai/gpt-5.4-mini-low,google/gemini-3.1-flash-lite \
        --seed 42 --k-max 2 --batch-n 8 --test-n 20

Live crank (spends money)::

    python scripts/run_experiment.py --area MNIST_Digits \
        --models openai/gpt-5.4-mini-low,google/gemini-3.1-flash-lite \
        --seed 42 --k-max 5 --batch-n 20 --test-n 100 \
        --live --allow-spend --concurrency 4
"""
from __future__ import annotations

import argparse
import json
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline import experiment as exp  # noqa: E402
from pipeline.experiment import store as exp_store  # noqa: E402
from pipeline.io_paths import (  # noqa: E402
    MNIST_SAMPLE_MANIFEST,
    genai_manifest_default,
    mint_run_id,
)
from pipeline.labeling.image_prep import prepare_image  # noqa: E402
from pipeline.manifest import load_records  # noqa: E402
from pipeline.policy_diff import (  # noqa: E402
    _call_chat_with_retries,
    _proposal_from_llm_json,
    call_and_parse_with_reask,
    _version_dir,
    accept_proposal,
    get_proposal,
    propose_diff,
    reject_proposal,
)
from pipeline.scoring.tasks import GENAI_BINARY, MNIST_MULTICLASS  # noqa: E402
from pipeline.web.demo_area import DEFAULT_POLICY_AREA, MNIST_POLICY_AREA  # noqa: E402

CHILD = "scripts/run_bulk_labeling.py"

_CURRENT_CHILD: subprocess.Popen | None = None
_STOP_REQUESTED = False
# Comma-separated judge ids labeling under the compressed policy render;
# set once in main() and appended to every child argv (see _run_child).
_COMPRESSED_MODELS_ARG: str = ""
# Cross-run label cache flag; set once in main() and appended to every child
# argv. In practice only the fixed benchmark/baseline legs ever hit — every
# candidate policy has fresh bytes, so its fingerprint never matches.
_LABEL_CACHE_ARG: bool = False


def _forward_signal(signum: int, _frame: object) -> None:  # pragma: no cover - signal path
    """Stop the in-flight child and let the loop finalize state as 'stopped'."""
    global _STOP_REQUESTED
    _STOP_REQUESTED = True
    child = _CURRENT_CHILD
    if child is not None and child.poll() is None:
        child.terminate()
        try:
            child.wait(timeout=10)
        except subprocess.TimeoutExpired:
            child.kill()


def _python() -> str:
    venv = ROOT / ".venv" / "bin" / "python"
    return str(venv) if venv.exists() else sys.executable


def _last_json_object(text: str) -> dict | None:
    depth = 0
    start = None
    last: dict | None = None
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                try:
                    last = json.loads(text[start : i + 1])
                except json.JSONDecodeError:
                    pass
    return last


def _run_dir(run_id: str) -> Path:
    return ROOT / "data" / "runs" / run_id


def _run_cost(run_id: str) -> float:
    manifest = _run_dir(run_id) / "run_manifest.json"
    if not manifest.exists():
        return 0.0
    total = (
        json.loads(manifest.read_text(encoding="utf-8"))
        .get("per_model_timing", {})
        .get("total", {})
        or {}
    )
    return round(float(total.get("total_cost_usd") or 0.0), 6)


class ExperimentStopped(Exception):
    """Operator stop. ``run_id`` carries a terminated child's run for cost capture."""

    def __init__(self, label: str, run_id: str | None = None):
        super().__init__(label)
        self.run_id = run_id


def _count_jsonl_lines(path: Path) -> int:
    try:
        with path.open("rb") as fh:
            return sum(1 for _ in fh)
    except OSError:
        return 0


def _sum_jsonl_cost(path: Path) -> float:
    total = 0.0
    try:
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                try:
                    total += float(json.loads(line).get("cost_usd") or 0.0)
                except (json.JSONDecodeError, TypeError, ValueError):
                    continue
    except OSError:
        return 0.0
    return total


def _run_child(
    *,
    models: list[str],
    area: str,
    sample_ids: list[str],
    manifest: Path,
    concurrency: int,
    batch_size: int,
    live: bool,
    policy_version: str,
    policy_dir: Path | None = None,
    policy_label: str | None = None,
    allow_holdout: bool = False,
    label: str,
    on_progress: Callable[[int, int, float], None] | None = None,
) -> dict:
    """One scoped labeling run (a mini-batch or a test/holdout eval).

    The child's run id is pre-minted here so its ``label_votes.jsonl`` can be
    watched WHILE it labels; ``on_progress(done, expected, cost_usd)`` fires
    whenever the completed-call count moves (the web status line shows it).
    """
    child_run_id = mint_run_id()
    argv = [
        _python(), "-u", CHILD,
        "--area", area,
        "--models", ",".join(models),
        "--policy-version", policy_version,
        "--manifest", str(manifest),
        "--sample-ids", ",".join(sample_ids),
        "--concurrency", str(concurrency),
        "--batch-size", str(batch_size),
        "--run-id", child_run_id,
    ]
    if policy_dir is not None:
        argv += [
            "--policy-dir", str(policy_dir),
            "--policy-graph-version-label", str(policy_label),
        ]
    if live:
        argv += ["--live", "--allow-spend"]
    if allow_holdout:
        argv += ["--allow-holdout"]
    if _COMPRESSED_MODELS_ARG:
        # Per-judge policy render (policy-rendering × judge-scale axis): the
        # flagged judges label under the deterministic digest in EVERY child
        # pass — train, candidate eval, holdout, benchmark — so a run's
        # render assignment is constant end to end.
        argv += ["--compressed-models", _COMPRESSED_MODELS_ARG]
    if _LABEL_CACHE_ARG and live:
        # Live-only by contract: dry children must never touch Postgres.
        argv += ["--label-cache"]

    # A stop requested during a non-child phase (drafter/gate/scoring) must
    # not launch — and pay for — the next labeling run.
    if _STOP_REQUESTED:
        raise ExperimentStopped(label)

    print(f"[experiment] {label}: {len(sample_ids)} images x {len(models)} judges", flush=True)
    global _CURRENT_CHILD
    t0 = time.monotonic()
    proc = subprocess.Popen(
        argv, cwd=str(ROOT), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    _CURRENT_CHILD = proc
    # Reader threads keep the pipes drained (a chatty child must never block
    # on a full pipe buffer) while the main thread polls vote-file growth.
    captured: dict[str, str] = {"out": "", "err": ""}

    def _drain(stream: Any, key: str) -> None:
        try:
            captured[key] = stream.read() or ""
        except Exception:  # noqa: BLE001 - a broken pipe just ends capture
            pass

    readers = [
        threading.Thread(target=_drain, args=(proc.stdout, "out"), daemon=True),
        threading.Thread(target=_drain, args=(proc.stderr, "err"), daemon=True),
    ]
    for reader in readers:
        reader.start()
    expected_calls = len(sample_ids) * len(models)
    votes_path = _run_dir(child_run_id) / "label_votes.jsonl"
    last_done = -1
    try:
        while proc.poll() is None:
            time.sleep(2.0)
            if on_progress is not None:
                done = _count_jsonl_lines(votes_path)
                if done != last_done:
                    last_done = done
                    try:
                        on_progress(done, expected_calls, _sum_jsonl_cost(votes_path))
                    except Exception:  # noqa: BLE001 - progress is cosmetic
                        pass
        for reader in readers:
            reader.join(timeout=10)
    finally:
        _CURRENT_CHILD = None
    stdout, stderr = captured["out"], captured["err"]
    if _STOP_REQUESTED:
        # Surface the terminated child's run id so its partial spend still
        # lands in the experiment's cost total.
        partial = _last_json_object(stdout) or {}
        raise ExperimentStopped(label, run_id=partial.get("run_id"))
    payload = _last_json_object(stdout)
    if payload is None or not payload.get("run_id"):
        sys.stderr.write(stdout[-1500:] + "\n" + stderr[-1500:] + "\n")
        raise RuntimeError(
            f"[experiment] child '{label}' produced no run payload (exit {proc.returncode})"
        )
    if payload.get("fatal_error"):
        raise RuntimeError(
            f"[experiment] child '{label}' failed fatally: {payload['fatal_error']}"
        )
    payload["_wall_s"] = round(time.monotonic() - t0, 1)
    return payload


def _gen_id(area: str, version: str) -> str:
    """The policy_graph_version convention used across the repo/store."""
    return f"{area}.{version}" if area != DEFAULT_POLICY_AREA else version


def _text_call_cost(model_id: str, usage_rows: list[dict[str, Any]]) -> float:
    from pipeline.providers.pricing import compute_call_cost

    total = 0.0
    for row in usage_rows:
        cost = compute_call_cost(
            row.get("model_id") or model_id,
            row.get("input_tokens"),
            row.get("output_tokens"),
            image_count=0,
            cached_input_tokens=row.get("cached_input_tokens"),
            cache_creation_input_tokens=row.get("cache_creation_input_tokens"),
        )
        total += cost or 0.0
    return round(total, 6)


def _sum_usage_field(usage_rows: list[dict[str, Any]], field: str) -> int | None:
    """Sum a token field across agent usage rows; None when never reported."""
    values = [row.get(field) for row in usage_rows if row.get(field) is not None]
    if not values:
        return None
    return int(sum(int(v) for v in values))


def _append_agent_cost(
    exp_dir: Path, *, experiment_id: str, k: int, role: str, model_id: str,
    usage_rows: list[dict[str, Any]],
) -> None:
    """Ledger drafter/gate text-call spend (the chat stack only logs it)."""
    from pipeline.scoring.cost_ledger import build_cost_row

    path = exp_dir / "costs.jsonl"
    with path.open("a", encoding="utf-8") as fh:
        for idx, row in enumerate(usage_rows):
            ledger = build_cost_row(
                run_id=experiment_id,
                batch_index=k,
                image_id=f"k{k}-{role}-{idx}",
                model_id=row.get("model_id") or model_id,
                input_tokens=row.get("input_tokens"),
                output_tokens=row.get("output_tokens"),
                recorded_at=exp.utcnow_iso(),
                image_count=0,
                cached_input_tokens=row.get("cached_input_tokens"),
                cache_creation_input_tokens=row.get("cache_creation_input_tokens"),
            )
            fh.write(json.dumps(ledger, sort_keys=True) + "\n")


def _load_misalignment(run_id: str) -> list[dict[str, Any]]:
    path = _run_dir(run_id) / "scoring" / "misalignment.json"
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8")).get("records", [])


def _persist_agent_exchange(
    exp_dir: Path, *, k: int, role: str, messages: list[dict], raw: str
) -> None:
    """Keep the full agent packet — training data for future RLHF of the critic."""
    out = exp_dir / "agents"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"k{k}-{role}.json").write_text(
        json.dumps({"k": k, "role": role, "messages": messages, "raw_response": raw},
                   indent=1) + "\n",
        encoding="utf-8",
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--area", default=MNIST_POLICY_AREA)
    ap.add_argument("--models", required=True,
                    help="Judge panel, comma-separated registry ids (2-5 recommended).")
    ap.add_argument("--seed", type=int, default=None,
                    help="Master seed for the whole run (default: random, recorded).")
    ap.add_argument("--k-max", type=int, default=5, help="Crank turns (cycles).")
    ap.add_argument("--batch-n", type=int, default=20, help="Train images per cycle.")
    ap.add_argument("--test-n", type=int, default=100,
                    help="Fixed seeded test partition size (the gate set).")
    ap.add_argument("--test-mode", choices=["fixed", "resample"], default="fixed",
                    help="fixed (default): one seeded test partition for the whole "
                         "run — the stable gate yardstick. resample: re-draw the "
                         "gate partition every cycle from the not-yet-trained-on "
                         "pool and RE-EVALUATE the incumbent on it before the "
                         "candidate (paired eval on the same fresh images) — "
                         "removes fixed-partition overfitting at ~+1 test eval "
                         "per cycle; the cross-run benchmark stays the honest "
                         "comparison.")
    ap.add_argument("--compliance-deweight", choices=["on", "off"], default="on",
                    help="on (default): a judge flagged non-compliant (near-"
                         "constant output — it is not conditioning on the policy) "
                         "is DEWEIGHTED: its votes leave the system majority vote "
                         "and the drafter's anchor/blame signal (per-judge rows "
                         "still reported). off: votes count everywhere regardless.")
    ap.add_argument("--max-anchors", type=int, default=15,
                    help="Max MISALIGNED anchors per edit (the panel's errors / SVM side).")
    ap.add_argument("--max-aligned-anchors", type=int, default=5,
                    help="Max ALIGNED anchors per edit — a sample of correctly-classified "
                         "images so the drafter also learns from correct calls (0 disables). "
                         "The mis/aligned split is a hyperparameter.")
    ap.add_argument("--max-changes", type=int, default=3,
                    help=f"Clip: max node-file changes per policy update "
                         f"(1-{exp.MAX_CHANGES_HARD_CAP}).")
    ap.add_argument("--epsilon", type=float, default=0.0,
                    help="Accept iff test system macro-F1 > before + epsilon.")
    ap.add_argument("--policy-version", default="v0.1",
                    help="Starting generator (k=0). FIXED at v0.1 by default so every "
                         "run number starts from the same baseline; accepted versions "
                         "branch from it (lineage recorded per run).")
    ap.add_argument("--gate-model", default=exp.DEFAULT_GATE_MODEL)
    ap.add_argument("--gate-persona", choices=sorted(exp.GATE_PERSONAS),
                    default=exp.DEFAULT_GATE_PERSONA,
                    help="Gate-agent stance appended to its system prompt: lenient "
                         "(default — a flat metric on a small test set is noise, skip "
                         "only clear defects), moderate, or strict (any regression or "
                         "unmeasured value skips).")
    ap.add_argument("--gate-mode", choices=["agent", "agent_only", "metric_only", "off"],
                    default="agent",
                    help="agent: metric rule + gate-agent veto | agent_only: the critic "
                         "agent's verdict alone decides (metric recorded as advisory, "
                         "never enforced; agent failure falls back to the metric rule) | "
                         "metric_only: rule alone | "
                         "off: accept EVERY clipped edit (metric recorded, never enforced "
                         "— shows unfiltered policy drift; requires --live).")
    ap.add_argument("--compressed-models", default=None,
                    help="Comma-separated judge ids that label under the "
                         "deterministic structural digest of the policy "
                         "bundle instead of the full render — the "
                         "policy-rendering × judge-scale knob (small judges "
                         "collapse under the full bundle; see "
                         "pipeline/policy_render.py). Subset of --models.")
    ap.add_argument("--label-cache", action="store_true",
                    help="Serve already-sampled (image, prompt, model) keys "
                         "from the Postgres label cache in every live child "
                         "pass (see pipeline/label_cache.py). Only "
                         "byte-identical prompts hit — in practice the v0 "
                         "baseline/benchmark legs; candidates always run live.")
    ap.add_argument("--drafter-model", default=exp.DEFAULT_DRAFTER_MODEL)
    ap.add_argument("--drafter-context", choices=["text_and_images", "text_only"],
                    default="text_only",
                    help="What the drafter (optimizer) receives per anchor: "
                         "text_only (default) = the judges' full text output "
                         "(label/confidence/difficulty/boundary/justification) "
                         "— cheap, and the justifications usually carry the "
                         "signal | text_and_images = additionally attach the "
                         "anchor image bytes so the drafter sees the pixels "
                         "(visual boundary cases, extra input tokens).")
    ap.add_argument("--strategy", choices=list(exp.STRATEGIES), default=exp.DEFAULT_STRATEGY,
                    help="Anchor selection: random_misalignment (S1, unbiased) or "
                         "top_gradient (most-important-first: panel avg |g| desc).")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=10,
                    help="Images per provider batch inside child runs.")
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--allow-spend", action="store_true")
    ap.add_argument("--holdout-final", action="store_true",
                    help="Score the LOCKED holdout under start + final versions at the end.")
    ap.add_argument("--validation-final", action="store_true",
                    help="Score the FIXED cross-run validation split (split='validation', "
                         "same images every run) under start + final versions — the "
                         "benchmark numbers for comparing run numbers/strategies.")
    ap.add_argument("--manifest", type=Path, default=None)
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    signal.signal(signal.SIGTERM, _forward_signal)
    signal.signal(signal.SIGINT, _forward_signal)

    live = bool(args.live and args.allow_spend)
    if args.live and not args.allow_spend:
        print("[experiment] refusing --live without --allow-spend", file=sys.stderr)
        return 2
    if args.gate_mode == "off" and not live:
        # A dry run with the gate off would mint REAL policy-graph versions
        # out of fake-label no-op edits every single cycle.
        print("[experiment] --gate-mode off requires --live --allow-spend", file=sys.stderr)
        return 2
    if not 1 <= args.max_changes <= exp.MAX_CHANGES_HARD_CAP:
        print(f"[experiment] --max-changes must be 1..{exp.MAX_CHANGES_HARD_CAP}", file=sys.stderr)
        return 2
    if args.k_max < 1 or args.batch_n < 1:
        print("[experiment] --k-max and --batch-n must be >= 1", file=sys.stderr)
        return 2
    import math

    if not math.isfinite(args.epsilon) or args.epsilon < 0:
        # Same contract as the web validator: a negative epsilon would accept
        # equal-or-worse candidates — including dry-run no-op edits.
        print("[experiment] --epsilon must be a finite non-negative number", file=sys.stderr)
        return 2

    area = args.area
    task = MNIST_MULTICLASS if area == MNIST_POLICY_AREA else GENAI_BINARY
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    if not 2 <= len(models) <= 5:
        print("[experiment] judge panel should be 2-5 models", file=sys.stderr)
        if not models:
            return 2
    compressed_models = sorted(
        m.strip() for m in (args.compressed_models or "").split(",") if m.strip()
    )
    unknown_compressed = [m for m in compressed_models if m not in models]
    if unknown_compressed:
        print("[experiment] --compressed-models must be a subset of --models; "
              f"unknown: {', '.join(unknown_compressed)}", file=sys.stderr)
        return 2
    global _COMPRESSED_MODELS_ARG
    _COMPRESSED_MODELS_ARG = ",".join(compressed_models)
    global _LABEL_CACHE_ARG
    _LABEL_CACHE_ARG = bool(args.label_cache)
    if (args.drafter_model.startswith("google/")
            and args.drafter_context == "text_and_images"):
        # Mirror of the web validator: the gemini text transport would
        # silently drop the anchor images.
        print("[experiment] gemini drafters support --drafter-context "
              "text_only (anchor images are not wired for the gemini "
              "transport yet)", file=sys.stderr)
        return 2
    manifest = args.manifest or (
        MNIST_SAMPLE_MANIFEST if area == MNIST_POLICY_AREA else genai_manifest_default()
    )

    base_version = args.policy_version or "v0.1"  # fixed k=0 baseline across runs
    if not _version_dir(ROOT, base_version, area).is_dir():
        print(f"[experiment] no such policy version: {area}/{base_version}", file=sys.stderr)
        return 2

    import secrets as _secrets

    seed = args.seed if args.seed is not None else _secrets.randbits(32)
    experiment_id = exp.mint_experiment_id()
    exp_dir = exp.experiment_dir(ROOT, experiment_id)
    exp_dir.mkdir(parents=True, exist_ok=True)

    records = load_records(manifest)
    try:
        test_ids, train_pool = exp.partition_test_train(
            records, seed=seed, test_n=args.test_n
        )
    except ValueError as exc:
        # E.g. --test-n 100 against the 40-image GenAI dev_golden pool. Exit
        # cleanly BEFORE any state/spend so a UI launch surfaces a real error
        # instead of a driver traceback with no experiment record.
        print(f"[experiment] invalid --test-n for area {area}: {exc}", file=sys.stderr)
        return 2
    holdout_ids = sorted(r.sample_id for r in records if r.split == "holdout")
    validation_records = [r for r in records if r.split == "validation"]
    validation_ids = sorted(r.sample_id for r in validation_records)
    if args.validation_final:
        if not validation_ids:
            print("[experiment] --validation-final: manifest has no validation split; "
                  "mint it once with scripts/build_mnist_validation_split.py", file=sys.stderr)
            return 2
        # Fail fast BEFORE any spend if the benchmark images aren't on disk
        # (e.g. a clone whose manifest was committed without the payloads).
        missing = [r.sample_id for r in validation_records
                   if not (ROOT / r.repo_rel_path).exists()]
        if missing:
            print(f"[experiment] --validation-final: {len(missing)} validation images "
                  f"missing on disk (first: {missing[0]}); re-run "
                  "scripts/build_mnist_validation_split.py or unpack_mnist.py",
                  file=sys.stderr)
            return 2

    state: dict[str, Any] = {
        "experiment_id": experiment_id,
        "run_number": exp.next_run_number(ROOT, area),
        "area": area,
        "seed": seed,
        "k_max": args.k_max,
        "batch_n": args.batch_n,
        "test_n": args.test_n,
        "max_anchors": args.max_anchors,
        "max_aligned_anchors": args.max_aligned_anchors,
        "max_changes": args.max_changes,
        "epsilon": args.epsilon,
        "strategy": args.strategy,
        "judge_models": models,
        "gate_model": args.gate_model,
        "gate_mode": args.gate_mode,
        "gate_persona": args.gate_persona,
        "drafter_model": args.drafter_model,
        "drafter_context": args.drafter_context,
        "compressed_models": compressed_models,
        "label_cache": bool(args.label_cache),
        "test_mode": args.test_mode,
        "compliance_deweight": args.compliance_deweight == "on",
        "base_version": base_version,
        "base_generator": _gen_id(area, base_version),
        "current_version": base_version,
        "concurrency": args.concurrency,
        "dry_run": not live,
        "splits": {
            "pool_split": "dev_golden",
            "test_ids": test_ids,
            "train_pool_n": len(train_pool),
            "holdout_n": len(holdout_ids),
            "holdout_locked": True,
        },
        "status": "running",
        "phase": "initializing",
        "started_at": exp.utcnow_iso(),
        "finished_at": None,
        "cost_usd_total": 0.0,
        "cycles": [],
        "holdout": None,
        "benchmark": None,
        "readjudication": None,
    }
    exp.write_state(ROOT, state)
    print(
        f"[experiment] {experiment_id} (run #{state['run_number']}, seed {seed}): "
        f"panel {', '.join(models)} | {base_version} -> k_max {args.k_max} | "
        f"test {len(test_ids)} / train pool {len(train_pool)} | "
        f"{'LIVE' if live else 'dry-run'}",
        flush=True,
    )

    def _phase(text: str) -> None:
        state["phase"] = text
        # The web review endpoint writes into the same file from another
        # process; fold its reviews in before every wholesale rewrite.
        exp.merge_disk_reviews(ROOT, state)
        exp.write_state(ROOT, state)
        print(f"[experiment] {text}", flush=True)

    def _add_cost(amount: float) -> None:
        state["cost_usd_total"] = round(state["cost_usd_total"] + (amount or 0.0), 6)

    def _progress_phase(base: str) -> Callable[[int, int, float], None]:
        # Live sub-phase while a child labels: "<base> · 137/500 calls · $0.42".
        def _cb(done: int, total: int, cost: float) -> None:
            note = f"{base} · {done}/{total} calls"
            if live and cost > 0:
                note += f" · ${cost:.2f}"
            _phase(note)
        return _cb

    def _sync() -> None:
        # Dry runs never touch the store: fake verdicts and placeholder
        # experiments would pollute the real cross-run analysis layer.
        exp.merge_disk_reviews(ROOT, state)
        state["store_synced"] = exp_store.try_sync_state(state) if live else False
        exp.write_state(ROOT, state)

    def _ingest(run_id: str) -> None:
        if live:
            exp_store.try_ingest_run(ROOT, run_id)

    usage_drafter: list[dict[str, Any]] = []
    usage_gate: list[dict[str, Any]] = []
    if live:
        from pipeline.providers.registry import get_chat_callable

        drafter_chat = get_chat_callable(args.drafter_model, usage_sink=usage_drafter)
        gate_chat = (
            get_chat_callable(args.gate_model, usage_sink=usage_gate)
            if args.gate_mode in {"agent", "agent_only"}
            else None
        )
    else:
        drafter_chat = None  # rebuilt per cycle against the current version dir
        gate_chat = (
            exp.fake_gate_callable() if args.gate_mode in {"agent", "agent_only"} else None
        )

    used_train_ids: set[str] = set()
    last_run_id: str | None = None
    exit_status = 0
    # Track the staged-but-undecided proposal so failure/stop paths can
    # archive it instead of leaving a human-acceptable orphan pending.
    inflight_proposal: dict[str, str | None] = {"id": None}

    def _archive_inflight_proposal(reason: str) -> None:
        proposal_id = inflight_proposal["id"]
        if not proposal_id:
            return
        inflight_proposal["id"] = None
        try:
            reject_proposal(repo_root=ROOT, proposal_id=proposal_id)
            print(f"[experiment] archived pending proposal {proposal_id} ({reason})",
                  file=sys.stderr, flush=True)
        except Exception as exc:  # noqa: BLE001 - best effort on teardown
            print(f"[experiment] warning: could not archive proposal {proposal_id}: {exc}",
                  file=sys.stderr, flush=True)

    # Golden truth for the test partition — the gate's fixed ruler.
    test_truth = exp.load_truth_labels(manifest, task=task, restrict_ids=test_ids)

    try:
        # ---- k=0 baseline: score the starting generator on the test partition
        _phase(f"cycle 0/{args.k_max}: baseline eval of {base_version} on test ({len(test_ids)})")
        baseline = _run_child(
            models=models, area=area, sample_ids=test_ids, manifest=manifest,
            concurrency=args.concurrency, batch_size=args.batch_size, live=live,
            policy_version=base_version, label="baseline test eval",
            on_progress=_progress_phase(
                f"cycle 0/{args.k_max}: baseline eval of {base_version} on test"
            ),
        )
        last_run_id = baseline["run_id"]
        _add_cost(_run_cost(baseline["run_id"]))
        # Compliance flag from the k=0 eval itself: a non-compliant judge is
        # known BEFORE the first drafter call, and (when deweighting is on)
        # its votes leave the system majority + the optimization signal from
        # the very first measurement.
        baseline_health = exp.judge_health(_load_misalignment(baseline["run_id"]))
        noncompliant: set[str] = set(exp.noncompliant_judges(baseline_health))
        deweight = args.compliance_deweight == "on"

        def _sys_excl() -> frozenset[str]:
            return frozenset(noncompliant) if deweight else frozenset()

        for h in baseline_health:
            if not h["compliant"]:
                _phase(
                    f"cycle 0/{args.k_max}: WARNING judge {h['model']} "
                    f"non-compliant — {h['top_share']:.0%} of votes are "
                    f"'{h['top_label']}'"
                    + ("; deweighted from the system vote" if deweight else "")
                )
        state["noncompliant_judges"] = sorted(noncompliant)
        current_test_metrics = exp.load_run_panel_metrics(
            _run_dir(baseline["run_id"]), manifest, task=task, restrict_ids=test_ids,
            system_exclude=_sys_excl(),
        )
        current_test_run_id = baseline["run_id"]
        _ingest(baseline["run_id"])
        state["cycles"].append(
            {
                "k": 0,
                "kind": "baseline",
                "cycle_seed": exp.cycle_seed(seed, 0),
                "generator_before": _gen_id(area, base_version),
                "generator_after": _gen_id(area, base_version),
                "test_run_id": baseline["run_id"],
                "errored_calls": baseline.get("errored_calls") or 0,
                "judge_health": baseline_health,
                "metrics": {"test": current_test_metrics},
                "status": "baseline",
                "started_at": state["started_at"],
                "closed_at": exp.utcnow_iso(),
            }
        )
        _sync()

        # ---- the crank
        for k in range(1, args.k_max + 1):
            if _STOP_REQUESTED:
                raise ExperimentStopped(f"before cycle {k}")
            cycle: dict[str, Any] = {
                "k": k,
                "cycle_seed": exp.cycle_seed(seed, k),
                "generator_before": _gen_id(area, state["current_version"]),
                "status": "open",
                "started_at": exp.utcnow_iso(),
                "metrics": {},
            }
            state["cycles"].append(cycle)
            base_dir = _version_dir(ROOT, state["current_version"], area)

            # 1. seeded train mini-batch under the current policy
            train_ids = exp.sample_train_batch(
                train_pool, used_train_ids, seed=seed, k=k, batch_n=args.batch_n
            )
            used_train_ids.update(train_ids)
            cycle["train_ids"] = train_ids
            _phase(f"cycle {k}/{args.k_max}: labeling train batch ({len(train_ids)}) "
                   f"under {state['current_version']}")
            train_run = _run_child(
                models=models, area=area, sample_ids=train_ids, manifest=manifest,
                concurrency=args.concurrency, batch_size=args.batch_size, live=live,
                policy_version=state["current_version"], label=f"k{k} train batch",
                on_progress=_progress_phase(
                    f"cycle {k}/{args.k_max}: labeling train batch "
                    f"under {state['current_version']}"
                ),
            )
            last_run_id = cycle["train_run_id"] = train_run["run_id"]
            _add_cost(_run_cost(train_run["run_id"]))
            _ingest(train_run["run_id"])
            if train_run.get("errored_calls"):
                cycle["train_errored_calls"] = train_run["errored_calls"]
            cycle["metrics"]["train"] = exp.load_run_panel_metrics(
                _run_dir(train_run["run_id"]), manifest, task=task,
                restrict_ids=train_ids, system_exclude=_sys_excl(),
            )

            # 2. S1: random misalignment anchors. A missing misalignment
            # artifact means the child's auto-scoring failed — that is a
            # skipped cycle with an error, NOT "panel fully aligned".
            mis_path = _run_dir(train_run["run_id"]) / "scoring" / "misalignment.json"
            if not mis_path.exists():
                cycle["status"] = "skipped"
                cycle["error"] = "train run scoring failed (no misalignment artifact)"
                cycle["metrics"]["test"] = current_test_metrics
                cycle["generator_after"] = cycle["generator_before"]
                cycle["closed_at"] = exp.utcnow_iso()
                _phase(f"cycle {k}/{args.k_max}: train scoring failed — cycle skipped")
                _sync()
                continue
            mis_records = _load_misalignment(train_run["run_id"])
            # Per-node citation stats: which nodes votes cited, wrong vs
            # right, with wrong_share + an advisory edit-type hint (split /
            # remove / clarify). The FULL table (every cited node, incl.
            # single-model rows) is recorded on the cycle for audit and node
            # health-tracking; the AGENTS get only the cross-model (>=2
            # judges) top rows — model-agnostic by construction: one judge's
            # quirks never steer the policy, a clause that misleads several
            # gets fixed once and helps them all.
            # Judge self-health for this batch: a near-constant judge has no
            # policy-text gradient — record it, tell the drafter, warn the
            # operator (measured: qwen byte-flat across three accepted edits
            # in GenAI run 5 while every reading judge improved). The flag is
            # STICKY for the run: once non-compliant, deweighted until done.
            cycle["judge_health"] = exp.judge_health(mis_records)
            newly = exp.noncompliant_judges(cycle["judge_health"]) - noncompliant
            noncompliant.update(newly)
            state["noncompliant_judges"] = sorted(noncompliant)
            for h in cycle["judge_health"]:
                if not h["compliant"]:
                    _phase(
                        f"cycle {k}/{args.k_max}: WARNING judge {h['model']} "
                        f"non-compliant — {h['top_share']:.0%} of votes are "
                        f"'{h['top_label']}'; policy edits cannot fix this judge"
                        + ("; deweighted from the system vote" if deweight else "")
                    )
            # Deweight (weight 0) non-compliant votes in the OPTIMIZATION
            # signal: blame, anchors, and eligibility are computed as if the
            # non-compliant judge never voted (run artifacts untouched).
            opt_records = exp.strip_noncompliant_votes(mis_records, _sys_excl())
            cycle["policy_blame"] = exp.policy_blame(
                opt_records, min_models=1, top_n=None
            )
            blame = [r for r in cycle["policy_blame"]
                     if r["n_models_wrong"] >= 2][:8]
            blamed_node_ids = frozenset(r["node"] for r in blame)
            anchors = exp.select_anchors(
                opt_records, seed=seed, k=k, max_anchors=args.max_anchors,
                train_ids=train_ids, strategy=args.strategy,
                blamed_nodes=blamed_node_ids or None,
            )
            # Aligned anchors: a sample of the panel's CORRECT calls, fed
            # alongside the errors so the drafter learns from correct
            # classifications and does not regress them (Attila 2026-07-07).
            aligned_anchors = exp.select_aligned_anchors(
                opt_records, seed=seed, k=k, max_aligned=args.max_aligned_anchors,
                train_ids=train_ids,
            )
            eligible = [
                r for r in opt_records
                if r.get("misalignment_type") != "all_agree"
                and str(r.get("image_id")) in set(train_ids)
            ]
            cycle["n_misaligned"] = len(eligible)
            cycle["anchor_ids"] = [str(a.get("image_id")) for a in anchors]
            cycle["aligned_anchor_ids"] = [str(a.get("image_id")) for a in aligned_anchors]

            # Compact anchor records so the ledger can show WHICH images drove
            # the edit (thumbnail path, truth, each judge's wrong/right vote).
            def _compact(anchor: dict[str, Any]) -> dict[str, Any]:
                return {
                    "image_id": anchor.get("image_id"),
                    "repo_rel_path": anchor.get("repo_rel_path"),
                    "sme_truth": anchor.get("sme_truth"),
                    "misalignment_type": anchor.get("misalignment_type"),
                    "severity": anchor.get("severity"),
                    "votes": [
                        {
                            "model": v.get("labeler_id") or v.get("model_id"),
                            "label": v.get("label"),
                            "confidence": v.get("confidence"),
                        }
                        for v in (anchor.get("votes") or [])
                    ],
                }
            cycle["anchors"] = [_compact(a) for a in anchors]
            cycle["aligned_anchors"] = [_compact(a) for a in aligned_anchors]
            if not anchors:
                cycle["status"] = "no_misalignments"
                cycle["metrics"]["test"] = current_test_metrics
                cycle["generator_after"] = cycle["generator_before"]
                cycle["closed_at"] = exp.utcnow_iso()
                _phase(f"cycle {k}/{args.k_max}: panel fully aligned on this batch — no edit")
                _sync()
                continue

            # 3. drafter: one clipped edit from the anchors
            _phase(f"cycle {k}/{args.k_max}: drafting edit from {len(anchors)} misaligned "
                   f"+ {len(aligned_anchors)} aligned anchors (max {args.max_changes} changes)")
            from pipeline.policy_iterator import load_policy_markdown

            # Attach the anchor images themselves (downsampled like judge
            # calls) so the drafter sees what the panel misread AND what it got
            # right. A missing or unreadable file drops that image, never the
            # draft. Only prepared for live runs — dry runs send no images
            # (fake drafter, no spend), so we don't pay the LANCZOS/JPEG/base64
            # cost for nothing.
            def _load_anchor_images(anchor_list: list[dict[str, Any]],
                                    kind: str) -> list[dict[str, Any]]:
                out: list[dict[str, Any]] = []
                for a in anchor_list:
                    rel = a.get("repo_rel_path")
                    if not rel:
                        continue
                    try:
                        prepared = prepare_image(ROOT / rel)
                    except Exception as prep_exc:  # noqa: BLE001 - optional evidence
                        print(f"[experiment] k{k}: {kind} anchor image skipped "
                              f"({a.get('image_id')}): {prep_exc}", file=sys.stderr)
                        continue
                    out.append({
                        "image_id": a.get("image_id"),
                        "sme_truth": a.get("sme_truth"),
                        "mime_type": prepared.mime_type,
                        "b64": prepared.to_base64(),
                    })
                return out

            with_images = args.drafter_context == "text_and_images"
            anchor_images = (
                _load_anchor_images(anchors, "misaligned") if live and with_images else []
            )
            aligned_images = (
                _load_anchor_images(aligned_anchors, "aligned") if live and with_images else []
            )
            messages = exp.build_drafter_messages(
                policy_markdown=load_policy_markdown(base_dir),
                base_version=state["current_version"],
                area=area,
                anchors=anchors,
                aligned_anchors=aligned_anchors,
                max_changes=args.max_changes,
                k=k,
                anchor_images=anchor_images or None,
                aligned_images=aligned_images or None,
                provider="anthropic" if args.drafter_model.startswith("anthropic/")
                else "openai",
                policy_blame=blame or None,
                judge_health=cycle["judge_health"] or None,
            )
            chat = drafter_chat or exp.fake_drafter_callable(base_dir)
            n_drafter_calls = len(usage_drafter)

            def _close_drafter_cost(cycle: dict[str, Any], n_before: int) -> float:
                # The optimizer's per-cycle spend — ledgered and recorded on
                # the cycle whether the draft parsed or not (retries of a
                # failing drafter still bill).
                rows = usage_drafter[n_before:]
                cost = _text_call_cost(args.drafter_model, rows) if rows else 0.0
                if rows:
                    _add_cost(cost)
                    _append_agent_cost(
                        exp_dir, experiment_id=experiment_id, k=k, role="drafter",
                        model_id=args.drafter_model, usage_rows=rows,
                    )
                cycle["drafter"] = {
                    "model": args.drafter_model,
                    "context": args.drafter_context,
                    "n_calls": len(rows),
                    "cost_usd": round(cost, 6),
                    "input_tokens": _sum_usage_field(rows, "input_tokens"),
                    "output_tokens": _sum_usage_field(rows, "output_tokens"),
                    "cached_input_tokens": _sum_usage_field(rows, "cached_input_tokens"),
                }
                return cost

            try:
                # Transport errors retry inside _call_chat_with_retries; a
                # syntactically bad completion (empty / prose / broken JSON)
                # gets re-asked with the parse error echoed back instead of
                # skipping the cycle — the train batch is already paid for.
                raw, (proposed_files, removed), messages = call_and_parse_with_reask(
                    chat, messages,
                    parse=_proposal_from_llm_json,
                    parse_attempts=3,
                    on_parse_retry=lambda attempt, exc: _phase(
                        f"cycle {k}/{args.k_max}: drafter reply unparseable "
                        f"(attempt {attempt}/3: {str(exc)[:80]}) — re-asking"
                    ),
                    model_id=args.drafter_model,
                    reasoning_effort="high", timeout_s=300.0, retries=2, backoff_s=2.0,
                    # OpenAI cache routing: the drafter prefix (system + policy
                    # bundle) repeats across cycles while the policy is unchanged.
                    prompt_cache_key=(
                        f"rush:drafter:{args.area}:{state['current_version']}"
                        if args.drafter_model.startswith("openai/") else None
                    ),
                )
            except Exception as draft_exc:  # noqa: BLE001 - a bad draft skips the cycle
                drafter_cost = _close_drafter_cost(cycle, n_drafter_calls)
                cycle["status"] = "skipped"
                cycle["error"] = f"drafter_error: {type(draft_exc).__name__}: {draft_exc}"[:400]
                cycle["metrics"]["test"] = current_test_metrics
                cycle["generator_after"] = cycle["generator_before"]
                cycle["cost_usd"] = round(_run_cost(train_run["run_id"]) + drafter_cost, 6)
                cycle["closed_at"] = exp.utcnow_iso()
                _phase(f"cycle {k}/{args.k_max}: drafter failed — cycle skipped")
                _sync()
                continue
            _persist_agent_exchange(exp_dir, k=k, role="drafter", messages=messages, raw=raw)
            drafter_cost = _close_drafter_cost(cycle, n_drafter_calls)

            clip = exp.clip_changes(
                proposed_files, removed, base_dir=base_dir, max_changes=args.max_changes
            )
            cycle["n_changes_proposed"] = clip["n_proposed"]
            cycle["n_changes_applied"] = clip["n_applied"]
            cycle["edit_summary"] = clip["changes"]
            cycle["edit_clipped"] = clip["clipped"]
            if clip["n_applied"] == 0:
                cycle["status"] = "skipped"
                cycle["error"] = "drafter proposed no effective changes"
                cycle["metrics"]["test"] = current_test_metrics
                cycle["generator_after"] = cycle["generator_before"]
                cycle["cost_usd"] = round(_run_cost(train_run["run_id"]) + drafter_cost, 6)
                cycle["closed_at"] = exp.utcnow_iso()
                _phase(f"cycle {k}/{args.k_max}: no effective changes — cycle skipped")
                _sync()
                continue

            # 4-6. stage -> candidate eval -> gate rule. Contained per cycle:
            # a failure here archives the pending proposal, marks the CYCLE
            # failed, and moves on — it never abandons the whole experiment.
            try:
                from pipeline.policy_diff import ALLOWED_POLICY_MODELS

                proposal = propose_diff(
                    repo_root=ROOT,
                    run_id=train_run["run_id"],
                    base_version=state["current_version"],
                    domain=area,
                    model_id=(
                        args.drafter_model
                        if args.drafter_model in ALLOWED_POLICY_MODELS
                        else None
                    ),
                    proposed_files=clip["files"],
                    files_removed=clip["removed"],
                )
                cycle["proposal_id"] = inflight_proposal["id"] = proposal["proposal_id"]

                # 5. candidate eval on the gate partition. Fixed mode: the
                # run-long seeded partition (the stable yardstick). Resample
                # mode (--test-mode resample): a FRESH seeded partition each
                # cycle, drawn from the pool minus every train image used so
                # far (no optimization->test leakage), with the INCUMBENT
                # re-evaluated on it first — a paired eval on the same fresh
                # images, at ~+1 test eval per cycle. K-folds-over-parallel-
                # runs remains the cross-run variant of the same idea.
                gate_test_ids, gate_truth = test_ids, test_truth
                incumbent_cost = 0.0
                if args.test_mode == "resample":
                    eligible_test = [
                        r for r in records
                        if r.split == "dev_golden"
                        and r.sample_id not in used_train_ids
                    ]
                    if len(eligible_test) > args.test_n:
                        gate_test_ids, _rest = exp.partition_test_train(
                            eligible_test, seed=exp.cycle_seed(seed, k),
                            test_n=args.test_n,
                        )
                        gate_truth = exp.load_truth_labels(
                            manifest, task=task, restrict_ids=gate_test_ids
                        )
                        cycle["test_ids"] = gate_test_ids
                        _phase(
                            f"cycle {k}/{args.k_max}: re-evaluating incumbent "
                            f"{state['current_version']} on fresh test "
                            f"({len(gate_test_ids)})"
                        )
                        incumbent_run = _run_child(
                            models=models, area=area, sample_ids=gate_test_ids,
                            manifest=manifest, concurrency=args.concurrency,
                            batch_size=args.batch_size, live=live,
                            policy_version=state["current_version"],
                            label=f"k{k} incumbent test eval",
                            on_progress=_progress_phase(
                                f"cycle {k}/{args.k_max}: re-evaluating incumbent "
                                "on fresh test"
                            ),
                        )
                        last_run_id = cycle["test_run_id"] = incumbent_run["run_id"]
                        incumbent_cost = _run_cost(incumbent_run["run_id"])
                        _add_cost(incumbent_cost)
                        _ingest(incumbent_run["run_id"])
                        current_test_metrics = exp.load_run_panel_metrics(
                            _run_dir(incumbent_run["run_id"]), manifest, task=task,
                            restrict_ids=gate_test_ids, system_exclude=_sys_excl(),
                        )
                        current_test_run_id = incumbent_run["run_id"]
                    else:
                        _phase(
                            f"cycle {k}/{args.k_max}: resample pool too small "
                            f"({len(eligible_test)} <= test_n) — falling back to "
                            "the fixed partition"
                        )
                candidate_gen = f"{area}.{experiment_id}.k{k}"
                cycle["candidate_generator"] = candidate_gen
                candidate_dir = exp.materialize_candidate(
                    base_dir=base_dir,
                    out_dir=exp_dir / "candidates" / f"k{k}",
                    files=clip["files"],
                    removed=clip["removed"],
                )
                _phase(f"cycle {k}/{args.k_max}: evaluating candidate on test ({len(gate_test_ids)})")
                cand_run = _run_child(
                    models=models, area=area, sample_ids=gate_test_ids, manifest=manifest,
                    concurrency=args.concurrency, batch_size=args.batch_size, live=live,
                    policy_version=state["current_version"],
                    policy_dir=candidate_dir, policy_label=candidate_gen,
                    label=f"k{k} candidate test eval",
                    on_progress=_progress_phase(
                        f"cycle {k}/{args.k_max}: evaluating candidate on test"
                    ),
                )
                last_run_id = cycle["candidate_run_id"] = cand_run["run_id"]
                _add_cost(_run_cost(cand_run["run_id"]))
                _ingest(cand_run["run_id"])
                if cand_run.get("errored_calls"):
                    cycle["candidate_errored_calls"] = cand_run["errored_calls"]
                candidate_metrics = exp.load_run_panel_metrics(
                    _run_dir(cand_run["run_id"]), manifest, task=task,
                    restrict_ids=gate_test_ids, system_exclude=_sys_excl(),
                )
                cycle["metrics"]["test_candidate"] = candidate_metrics

                # 6. the PPO gate — before/after computed over the SAME
                # images (intersection of decided system verdicts), so
                # errored calls or majority ties can't flip the gate on
                # coverage alone. Non-compliant judges are deweighted from
                # BOTH sides' system verdicts (same rule, fair comparison).
                comparison = exp.gate_comparison(
                    exp.load_votes(_run_dir(current_test_run_id)),
                    exp.load_votes(_run_dir(cand_run["run_id"])),
                    gate_truth,
                    task=task,
                    system_exclude=_sys_excl(),
                )
            except ExperimentStopped:
                _archive_inflight_proposal("experiment stopped")
                raise
            except Exception as cycle_exc:  # noqa: BLE001 - contain to this cycle
                _archive_inflight_proposal("cycle failed")
                cycle["status"] = "failed"
                cycle["error"] = f"{type(cycle_exc).__name__}: {cycle_exc}"[:400]
                cycle["metrics"]["test"] = current_test_metrics
                cycle["generator_after"] = cycle["generator_before"]
                cand_cost = (
                    _run_cost(cycle["candidate_run_id"])
                    if cycle.get("candidate_run_id") else 0.0
                )
                cycle["cost_usd"] = round(
                    _run_cost(train_run["run_id"]) + drafter_cost + cand_cost, 6
                )
                cycle["closed_at"] = exp.utcnow_iso()
                _phase(f"cycle {k}/{args.k_max}: FAILED ({cycle['error']}) — continuing")
                _sync()
                continue

            value_before = comparison["value_before"]
            value_after = comparison["value_after"]
            metric_pass = exp.metric_passes(value_before, value_after, epsilon=args.epsilon)
            # Measured prompt mass — the bundle is every judge's context (a
            # 7B collapses under a bloated one; 2026-07-09 probe). Recorded on
            # every cycle (policy size over k = the parameter-count analog)
            # and handed to the gate agent as bloat-watch evidence.
            from pipeline.policy_iterator import load_policy_markdown as _lpm
            try:
                bundle_before = len(_lpm(base_dir))
                bundle_after = len(_lpm(candidate_dir))
            except Exception:  # noqa: BLE001 - evidence only, never block the gate
                bundle_before = bundle_after = None
            cycle["policy_bundle_chars"] = {
                "before": bundle_before, "candidate": bundle_after,
            }
            agent_verdict = None
            gate_error = None
            gate_raw = ""
            gate_cost = 0.0
            if gate_chat is not None:
                _phase(f"cycle {k}/{args.k_max}: gate review "
                       f"(F1 {value_before} -> {value_after} on {comparison['n_common']} "
                       f"common, rule {'pass' if metric_pass else 'fail'})")
                gate_messages = exp.build_gate_messages(
                    metric=exp.GATE_METRIC, value_before=value_before,
                    value_after=value_after, epsilon=args.epsilon,
                    metric_pass=metric_pass, metrics_before=current_test_metrics,
                    metrics_after=candidate_metrics,
                    diffs=get_proposal(repo_root=ROOT, proposal_id=proposal["proposal_id"])
                    .get("diffs", []),
                    anchors=anchors, k=k, comparison=comparison,
                    agent_is_sole_gate=args.gate_mode == "agent_only",
                    persona=args.gate_persona,
                    bundle_chars_before=bundle_before,
                    bundle_chars_after=bundle_after,
                    policy_blame=blame or None,
                )
                n_gate_calls = len(usage_gate)
                try:
                    gate_raw = _call_chat_with_retries(
                        gate_chat, gate_messages, model_id=args.gate_model,
                        reasoning_effort="high", timeout_s=120.0, retries=2, backoff_s=2.0,
                        # Gate packets are mostly unique per cycle; the shared
                        # prefix is the system prompt — still worth routing.
                        prompt_cache_key=(
                            f"rush:gate:{args.area}"
                            if args.gate_model.startswith("openai/") else None
                        ),
                    )
                    agent_verdict = exp.parse_gate_response(gate_raw)
                    _persist_agent_exchange(
                        exp_dir, k=k, role="gate", messages=gate_messages, raw=gate_raw
                    )
                except Exception as gate_exc:  # noqa: BLE001 - rule carries the decision
                    gate_error = f"{type(gate_exc).__name__}: {gate_exc}"[:400]
                if usage_gate[n_gate_calls:]:
                    gate_cost = _text_call_cost(args.gate_model, usage_gate[n_gate_calls:])
                    _add_cost(gate_cost)
                    _append_agent_cost(
                        exp_dir, experiment_id=experiment_id, k=k, role="gate",
                        model_id=args.gate_model, usage_rows=usage_gate[n_gate_calls:],
                    )

            if args.gate_mode in {"off", "agent_only"} and value_after is None:
                # Even without the metric rule enforced, an edit whose candidate
                # eval produced no decided system verdicts is unmeasurable —
                # never applied (the critic cannot outvote missing evidence).
                outcome = {
                    "decision": "skip",
                    "decided_by": "metric_rule",
                    "rationale": f"gate mode {args.gate_mode}, but the candidate "
                                 "produced no decided system verdicts on test — an "
                                 "unmeasurable edit is never applied",
                    "risk_flags": [],
                }
            else:
                outcome = exp.resolve_gate_decision(
                    metric_pass=metric_pass, agent=agent_verdict,
                    gate_off=args.gate_mode == "off",
                    agent_only=args.gate_mode == "agent_only",
                )
            cycle["gate"] = {
                "metric": exp.GATE_METRIC,
                "value_before": value_before,
                "value_after": value_after,
                "comparison": comparison,
                "epsilon": args.epsilon,
                "metric_pass": metric_pass,
                "decision": outcome["decision"],
                "decided_by": outcome["decided_by"],
                "rationale": outcome["rationale"],
                "risk_flags": outcome["risk_flags"],
                "gate_model": args.gate_model if gate_chat is not None else None,
                "persona": args.gate_persona if gate_chat is not None else None,
                "cost_usd": round(gate_cost, 6),
                "error": gate_error,
                "raw_response": gate_raw[:2000],
            }

            # 7. apply or archive. accept_proposal failures (e.g. a human
            # accepted a web proposal mid-run and moved the latest version)
            # stay experiment-fatal: the premise of the run is broken.
            inflight_proposal["id"] = None
            if outcome["decision"] == "accept":
                accepted = accept_proposal(
                    repo_root=ROOT, proposal_id=proposal["proposal_id"],
                    # Fixed-k=0 runs branch from the constant baseline, so the
                    # stale-base guard would otherwise abort run #2's first
                    # accept once run #1 minted a newer version.
                    allow_branch=True,
                    # v<run>.<k>: the version name says which run accepted it
                    # and at which cycle (Attila 2026-07-07). Falls back to
                    # the global mint on a name collision.
                    new_version=f"v{state['run_number']}.{k}",
                )
                state["current_version"] = accepted["new_version"]
                cycle["new_version"] = accepted["new_version"]
                cycle["generator_after"] = _gen_id(area, accepted["new_version"])
                cycle["status"] = "accepted"
                current_test_metrics = candidate_metrics
                current_test_run_id = cand_run["run_id"]
                _phase(
                    f"cycle {k}/{args.k_max}: ACCEPTED -> {accepted['new_version']} "
                    f"(system F1 {value_before} -> {value_after}, "
                    f"{clip['n_applied']} change{'s' if clip['n_applied'] != 1 else ''})"
                )
            else:
                try:
                    reject_proposal(repo_root=ROOT, proposal_id=proposal["proposal_id"])
                except Exception as reject_exc:  # noqa: BLE001 - archive is best effort
                    print(f"[experiment] warning: reject_proposal failed: {reject_exc}",
                          file=sys.stderr, flush=True)
                cycle["generator_after"] = cycle["generator_before"]
                cycle["status"] = "skipped"
                _phase(
                    f"cycle {k}/{args.k_max}: skipped ({outcome['decided_by']}; "
                    f"system F1 {value_before} -> {value_after})"
                )
            cycle["metrics"]["test"] = current_test_metrics
            cycle["cost_usd"] = round(
                _run_cost(train_run["run_id"]) + _run_cost(cand_run["run_id"])
                + incumbent_cost + drafter_cost + gate_cost, 6
            )
            cycle["closed_at"] = exp.utcnow_iso()
            _sync()

        # ---- optional locked-holdout before/after readout
        if args.holdout_final and holdout_ids:
            holdout: dict[str, Any] = {"n": len(holdout_ids)}
            # Bind into state BEFORE the legs run: a stop/failure between the
            # 'start' and 'final' passes must not discard the paid readout.
            state["holdout"] = holdout
            plan = [("start", base_version)]
            if state["current_version"] != base_version:
                plan.append(("final", state["current_version"]))
            for tag, version in plan:
                _phase(f"holdout readout: {version} on locked holdout ({len(holdout_ids)})")
                run = _run_child(
                    models=models, area=area, sample_ids=holdout_ids, manifest=manifest,
                    concurrency=args.concurrency, batch_size=args.batch_size, live=live,
                    policy_version=version, allow_holdout=True,
                    label=f"holdout eval {version}",
                    on_progress=_progress_phase(f"holdout readout: {version}"),
                )
                last_run_id = run["run_id"]
                _add_cost(_run_cost(run["run_id"]))
                _ingest(run["run_id"])
                holdout[tag] = {
                    "version": version,
                    "run_id": run["run_id"],
                    "metrics": exp.load_run_panel_metrics(
                        _run_dir(run["run_id"]), manifest, task=task,
                        restrict_ids=holdout_ids,
                    ),
                }
            if "final" not in holdout:
                # No accepted edits: final == start, no second spend needed.
                holdout["final"] = holdout["start"]

        # ---- optional FIXED cross-run benchmark readout (same images every
        # run; the numbers that compare run numbers / strategies fairly).
        if args.validation_final and validation_ids:
            benchmark: dict[str, Any] = {"n": len(validation_ids), "split": "validation"}
            # Same partial-persistence contract as the holdout block above.
            state["benchmark"] = benchmark
            plan = [("start", base_version)]
            if state["current_version"] != base_version:
                plan.append(("final", state["current_version"]))
            for tag, version in plan:
                _phase(f"benchmark readout: {version} on fixed validation "
                       f"({len(validation_ids)})")
                run = _run_child(
                    models=models, area=area, sample_ids=validation_ids, manifest=manifest,
                    concurrency=args.concurrency, batch_size=args.batch_size, live=live,
                    policy_version=version, allow_holdout=True,
                    label=f"benchmark eval {version}",
                    on_progress=_progress_phase(f"benchmark readout: {version}"),
                )
                last_run_id = run["run_id"]
                _add_cost(_run_cost(run["run_id"]))
                _ingest(run["run_id"])
                benchmark[tag] = {
                    "version": version,
                    "run_id": run["run_id"],
                    "metrics": exp.load_run_panel_metrics(
                        _run_dir(run["run_id"]), manifest, task=task,
                        restrict_ids=validation_ids,
                    ),
                }
            if "final" not in benchmark:
                benchmark["final"] = benchmark["start"]

        state["status"] = "completed"
    except ExperimentStopped as stop_exc:
        state["status"] = "stopped"
        _archive_inflight_proposal("experiment stopped")
        # Best-effort spend capture for a child terminated mid-flight.
        if getattr(stop_exc, "run_id", None):
            _add_cost(_run_cost(stop_exc.run_id))
        if state["cycles"] and state["cycles"][-1].get("status") == "open":
            state["cycles"][-1]["status"] = "stopped"
            state["cycles"][-1]["closed_at"] = exp.utcnow_iso()
        exit_status = 130
    except Exception as exc:  # noqa: BLE001 - finalize state, then re-raise info
        state["status"] = "failed"
        state["error"] = f"{type(exc).__name__}: {exc}"[:600]
        _archive_inflight_proposal("experiment failed")
        if state["cycles"] and state["cycles"][-1].get("status") == "open":
            state["cycles"][-1]["status"] = "failed"
            state["cycles"][-1]["error"] = state["error"]
            state["cycles"][-1]["closed_at"] = exp.utcnow_iso()
        print(f"[experiment] FAILED: {state['error']}", file=sys.stderr, flush=True)
        exit_status = 1

    state["finished_at"] = exp.utcnow_iso()
    accepted_cycles = [c for c in state["cycles"] if c.get("status") == "accepted"]
    state["phase"] = (
        f"{state['status']}: {len(accepted_cycles)} accepted / "
        f"{max(0, len(state['cycles']) - 1)} cycles, "
        f"{state['base_version']} -> {state['current_version']}"
    )
    # SME re-adjudication queue: every image whose latest evaluation in this
    # run is still misaligned (train under the cycle's policy in force,
    # test/holdout/benchmark under the final policy), with the consensus /
    # confidence / difficulty / gradient rank signals the queue tab sorts by.
    try:
        state["readjudication"] = exp.build_readjudication(
            state, load_misalignment=_load_misalignment,
            sha_by_image={r.sample_id: r.sha256 for r in records},
        )
        print(f"[experiment] flagged {state['readjudication']['n_flagged']} "
              "image(s) for SME re-adjudication", flush=True)
    except Exception as exc:  # noqa: BLE001 - flagging must never mask run status
        print(f"[experiment] re-adjudication flagging failed: {exc}",
              file=sys.stderr, flush=True)
        state["readjudication"] = None
    # End-of-run analysis record: per-scorer baseline/final/delta test metrics
    # + run metadata, for writing cross-run analyses without re-opening cycles.
    state["summary"] = exp.build_run_summary(state)
    _sync()

    # Human-readable summary
    print(f"\n=== RUSH experiment {experiment_id} (run #{state['run_number']}, seed {seed}) ===")
    gate_desc = "OFF (accept all clipped edits)" if args.gate_mode == "off" \
        else f"{args.gate_model} ({args.gate_mode}, {args.gate_persona})"
    print(f"panel: {', '.join(models)} | gate: {gate_desc}")
    print(f"policy: {state['base_version']} -> {state['current_version']} "
          f"({len(accepted_cycles)} accepted, "
          f"{sum(1 for c in state['cycles'] if c.get('status') == 'skipped')} skipped)")
    for cycle in state["cycles"]:
        if cycle.get("k") == 0:
            f1 = (cycle["metrics"]["test"].get(exp.SYSTEM_SCORER) or {}).get("macro_f1")
            print(f"  k=0 baseline: test system F1 {f1}")
            continue
        gate = cycle.get("gate") or {}
        print(
            f"  k={cycle['k']}: {cycle.get('status')} "
            f"(F1 {gate.get('value_before')} -> {gate.get('value_after')}, "
            f"{cycle.get('n_changes_applied', 0)} changes, "
            f"{cycle.get('n_misaligned', 0)} misaligned)"
        )
    print(f"total cost: ${state['cost_usd_total']:.4f} | state: "
          f"{exp.state_path(ROOT, experiment_id).relative_to(ROOT)}")

    # Machine-readable trailer (web job monitor picks the last JSON object)
    print(json.dumps({
        "kind": "experiment",
        "experiment_id": experiment_id,
        "run_id": last_run_id,
        "status": state["status"],
        "base_version": state["base_version"],
        "final_version": state["current_version"],
        "accepted": len(accepted_cycles),
        "cost_usd_total": state["cost_usd_total"],
    }))
    return exit_status


if __name__ == "__main__":
    raise SystemExit(main())
