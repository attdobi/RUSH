"""Best-effort Postgres sync for the experiment crank.

The portable ``experiment.json`` is the source of truth; these writers mirror
it into the ``rush`` schema (``experiment`` / ``experiment_cycle`` /
``experiment_metric`` / ``gate_decision`` / ``gate_review`` + the
``generator_version`` lineage) so cross-experiment paper queries are one SQL
statement. Every entry point is soft-fail: a missing psycopg, a dead server,
or a schema mismatch logs a warning and returns False — the crank never
stops for the database. Re-syncing the same state is idempotent (upserts).
"""
from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)

_METRIC_KEYS = (
    "accuracy",
    "macro_f1",
    "macro_precision",
    "macro_recall",
    "macro_fpr",
    "macro_fnr",
    "micro_f1",
    "micro_precision",
    "micro_recall",
    "micro_fpr",
    "micro_fnr",
)


# Schema application is idempotent but not free (the generator_version
# constraint migration takes an ACCESS EXCLUSIVE lock each replay) — run it
# once per process, not once per connection.
_schema_applied = False


def _connect():
    global _schema_applied
    from pipeline import labelstore

    conn = labelstore.connect()
    if not _schema_applied:
        labelstore.init_schema(conn)
        _schema_applied = True
    return conn


def _jsonb(value: Any):
    from psycopg.types.json import Jsonb

    return Jsonb(value)


def sync_experiment_state(conn, state: dict[str, Any]) -> None:
    """Upsert the whole experiment record: config, cycles, metrics, gates."""
    exp_id = state["experiment_id"]
    conn.execute(
        """
        INSERT INTO rush.experiment (
          experiment_id, run_number, area, seed, k_max, batch_n, test_n,
          judge_models, gate_model, drafter_model, strategy, max_changes,
          epsilon, base_generator, config, status, started_at, finished_at)
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
        ON CONFLICT (experiment_id) DO UPDATE SET
          status = EXCLUDED.status,
          finished_at = EXCLUDED.finished_at,
          config = EXCLUDED.config
        """,
        (
            exp_id,
            state.get("run_number"),
            state["area"],
            state["seed"],
            state["k_max"],
            state["batch_n"],
            state["test_n"],
            _jsonb(state.get("judge_models", [])),
            state.get("gate_model", ""),
            state.get("drafter_model", ""),
            state.get("strategy", "random_misalignment"),
            state.get("max_changes", 5),
            state.get("epsilon", 0),
            state.get("base_generator"),
            _jsonb(
                {
                    "dry_run": state.get("dry_run", False),
                    "concurrency": state.get("concurrency"),
                    "max_anchors": state.get("max_anchors"),
                    "gate_mode": state.get("gate_mode"),
                    "cost_usd_total": state.get("cost_usd_total"),
                    "current_version": state.get("current_version"),
                }
            ),
            state.get("status", "running"),
            state.get("started_at"),
            state.get("finished_at"),
        ),
    )

    for cycle in state.get("cycles", []):
        _sync_cycle(conn, exp_id, cycle)
    _sync_holdout(conn, state)
    conn.commit()


def _sync_holdout(conn, state: dict[str, Any]) -> None:
    """Mirror the locked-holdout before/after readout (--holdout-final).

    The driver stores it at experiment level, not on a cycle; the metric
    table's cycle FK is satisfied by anchoring 'start' to k=0 (baseline
    generator) and 'final' to the last recorded cycle (final generator).
    """
    holdout = state.get("holdout")
    cycles = state.get("cycles", [])
    if not holdout or not cycles:
        return
    exp_id = state["experiment_id"]
    last_k = max(int(c["k"]) for c in cycles if isinstance(c.get("k"), int))
    for tag, k in (("start", 0), ("final", last_k)):
        entry = holdout.get(tag)
        if not entry or not entry.get("metrics"):
            continue
        generator = entry.get("version")
        if generator and state.get("area"):
            area = state["area"]
            generator = f"{area}.{generator}" if area != "Generative_AI" else generator
        if not generator:
            continue
        _sync_generator(conn, generator)
        for scorer, metrics in entry["metrics"].items():
            _sync_metric(conn, exp_id, k, "holdout", scorer, generator, metrics)


def _sync_generator(
    conn,
    generator_id: str | None,
    *,
    experiment_id: str | None = None,
    parent_id: str | None = None,
    minibatch_k: int | None = None,
    n_changes: int | None = None,
    gate_status: str | None = None,
    accepted_as: str | None = None,
    diff_text: str | None = None,
    f1_before: float | None = None,
    f1_after: float | None = None,
) -> None:
    if not generator_id:
        return
    # Parent must exist first (self-referencing FK).
    if parent_id:
        conn.execute(
            "INSERT INTO rush.generator_version (generator_id) VALUES (%s) "
            "ON CONFLICT (generator_id) DO NOTHING",
            (parent_id,),
        )
    conn.execute(
        """
        INSERT INTO rush.generator_version (
          generator_id, minibatch_k, parent_id, diff_text, gate_status,
          experiment_id, n_changes, accepted_as, f1_val_before, f1_val_after)
        VALUES (%s,%s,%s,COALESCE(%s,''),COALESCE(%s,'pending'),%s,%s,%s,%s,%s)
        ON CONFLICT (generator_id) DO UPDATE SET
          minibatch_k = COALESCE(EXCLUDED.minibatch_k, rush.generator_version.minibatch_k),
          parent_id = COALESCE(EXCLUDED.parent_id, rush.generator_version.parent_id),
          diff_text = CASE WHEN EXCLUDED.diff_text <> '' THEN EXCLUDED.diff_text
                           ELSE rush.generator_version.diff_text END,
          gate_status = COALESCE(%s, rush.generator_version.gate_status),
          experiment_id = COALESCE(EXCLUDED.experiment_id, rush.generator_version.experiment_id),
          n_changes = COALESCE(EXCLUDED.n_changes, rush.generator_version.n_changes),
          accepted_as = COALESCE(EXCLUDED.accepted_as, rush.generator_version.accepted_as),
          f1_val_before = COALESCE(EXCLUDED.f1_val_before, rush.generator_version.f1_val_before),
          f1_val_after = COALESCE(EXCLUDED.f1_val_after, rush.generator_version.f1_val_after)
        """,
        (
            generator_id,
            minibatch_k,
            parent_id,
            diff_text,
            gate_status,
            experiment_id,
            n_changes,
            accepted_as,
            f1_before,
            f1_after,
            gate_status,
        ),
    )


def _sync_cycle(conn, exp_id: str, cycle: dict[str, Any]) -> None:
    k = cycle["k"]
    gate = cycle.get("gate") or {}

    # Generator lineage rows before anything that references them.
    _sync_generator(conn, cycle.get("generator_before"))
    candidate = cycle.get("candidate_generator")
    if candidate:
        gate_status = {
            "accepted": "accepted",
            "skipped": "skipped",
        }.get(str(cycle.get("status")), "pending")
        _sync_generator(
            conn,
            candidate,
            experiment_id=exp_id,
            parent_id=cycle.get("generator_before"),
            minibatch_k=k,
            n_changes=cycle.get("n_changes_applied"),
            gate_status=gate_status,
            accepted_as=cycle.get("generator_after")
            if cycle.get("status") == "accepted"
            else None,
            diff_text=json.dumps(cycle.get("edit_summary", [])),
            f1_before=gate.get("value_before"),
            f1_after=gate.get("value_after"),
        )
    if cycle.get("status") == "accepted" and cycle.get("generator_after"):
        _sync_generator(
            conn,
            cycle.get("generator_after"),
            experiment_id=exp_id,
            parent_id=cycle.get("generator_before"),
            minibatch_k=k,
            n_changes=cycle.get("n_changes_applied"),
            gate_status="accepted",
            diff_text=json.dumps(cycle.get("edit_summary", [])),
            f1_before=gate.get("value_before"),
            f1_after=gate.get("value_after"),
        )

    status = str(cycle.get("status", "open"))
    if status not in {
        "open", "baseline", "accepted", "skipped", "no_misalignments", "failed", "stopped",
    }:
        status = "open"
    conn.execute(
        """
        INSERT INTO rush.experiment_cycle (
          experiment_id, k, cycle_seed, generator_before, candidate_generator,
          generator_after, train_ids, n_misaligned, anchor_ids,
          n_changes_proposed, n_changes_applied, proposal_id, train_run_id,
          candidate_run_id, status, error, started_at, closed_at)
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
        ON CONFLICT (experiment_id, k) DO UPDATE SET
          generator_after = EXCLUDED.generator_after,
          status = EXCLUDED.status,
          error = EXCLUDED.error,
          closed_at = EXCLUDED.closed_at
        """,
        (
            exp_id,
            k,
            cycle.get("cycle_seed"),
            cycle.get("generator_before"),
            candidate,
            cycle.get("generator_after"),
            _jsonb(cycle.get("train_ids", [])),
            cycle.get("n_misaligned"),
            _jsonb(cycle.get("anchor_ids", [])),
            cycle.get("n_changes_proposed"),
            cycle.get("n_changes_applied"),
            cycle.get("proposal_id"),
            cycle.get("train_run_id"),
            cycle.get("candidate_run_id"),
            status,
            cycle.get("error"),
            cycle.get("started_at"),
            cycle.get("closed_at"),
        ),
    )

    for split, by_scorer in (cycle.get("metrics") or {}).items():
        # experiment.json keys: train / test / test_candidate / holdout_*.
        # The store keeps the CHECK-constrained axis; candidate test metrics
        # land as split='test' under the candidate's generator_id.
        if split == "train":
            store_split, generator = "train", cycle.get("generator_before")
        elif split == "test":
            store_split, generator = "test", cycle.get("generator_after")
        elif split == "test_candidate":
            store_split, generator = "test", candidate
        elif split.startswith("holdout"):
            store_split, generator = "holdout", cycle.get("generator_after")
        else:
            continue
        if not generator:
            continue
        for scorer, metrics in (by_scorer or {}).items():
            _sync_metric(conn, exp_id, k, store_split, scorer, generator, metrics)

    if gate:
        conn.execute(
            "DELETE FROM rush.gate_decision WHERE experiment_id = %s AND k = %s",
            (exp_id, k),
        )
        conn.execute(
            """
            INSERT INTO rush.gate_decision (
              experiment_id, k, baseline_generator, candidate_generator,
              gate_model, metric, value_before, value_after, metric_pass,
              decision, decided_by, rationale, raw_response, cost_usd)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            """,
            (
                exp_id,
                k,
                cycle.get("generator_before"),
                candidate or cycle.get("generator_before"),
                # metric_only / gate-off runs have no agent; the column is NOT NULL.
                gate.get("gate_model")
                or ("gate_off" if gate.get("decided_by") == "gate_off" else "metric_rule"),
                gate.get("metric", "test_system_macro_f1"),
                gate.get("value_before"),
                gate.get("value_after"),
                bool(gate.get("metric_pass")),
                gate.get("decision", "skip"),
                gate.get("decided_by", "metric_rule"),
                gate.get("rationale"),
                gate.get("raw_response"),
                gate.get("cost_usd"),
            ),
        )

    review = cycle.get("review")
    if review:
        _insert_gate_review(conn, exp_id, k, review)


def _sync_metric(
    conn, exp_id: str, k: int, split: str, scorer: str, generator_id: str, metrics: dict
) -> None:
    values = [metrics.get(key) for key in _METRIC_KEYS]
    conn.execute(
        f"""
        INSERT INTO rush.experiment_metric (
          experiment_id, k, split, scorer, generator_id, n, n_abstained,
          {", ".join(_METRIC_KEYS)}, per_class)
        VALUES (%s,%s,%s,%s,%s,%s,%s,{",".join(["%s"] * len(_METRIC_KEYS))},%s)
        ON CONFLICT (experiment_id, k, split, scorer, generator_id) DO UPDATE SET
          n = EXCLUDED.n,
          n_abstained = EXCLUDED.n_abstained,
          {", ".join(f"{key} = EXCLUDED.{key}" for key in _METRIC_KEYS)},
          per_class = EXCLUDED.per_class
        """,
        (
            exp_id,
            k,
            split,
            scorer,
            generator_id,
            metrics.get("n", 0),
            metrics.get("n_abstained", 0),
            *values,
            _jsonb(metrics.get("per_class", {})),
        ),
    )


def _insert_gate_review(conn, exp_id: str, k: int, review: dict[str, Any]) -> None:
    # One live review per (experiment, k): re-reviews replace, history stays
    # in experiment.json's audit trail if ever needed.
    conn.execute(
        "DELETE FROM rush.gate_review WHERE experiment_id = %s AND k = %s",
        (exp_id, k),
    )
    conn.execute(
        """
        INSERT INTO rush.gate_review (experiment_id, k, reviewer, verdict, comment)
        VALUES (%s,%s,%s,%s,%s)
        """,
        (
            exp_id,
            k,
            review.get("reviewer", "sme"),
            review.get("verdict", "unsure"),
            review.get("comment", ""),
        ),
    )


def try_sync_state(state: dict[str, Any]) -> bool:
    """Mirror the full experiment state into Postgres; False on any failure."""
    try:
        conn = _connect()
    except Exception as exc:  # noqa: BLE001 - soft-fail by design
        logger.warning("experiment store sync skipped (connect): %s", exc)
        return False
    try:
        sync_experiment_state(conn, state)
        return True
    except Exception as exc:  # noqa: BLE001
        logger.warning("experiment store sync failed: %s", exc)
        try:
            conn.rollback()
        except Exception:  # noqa: BLE001
            pass
        return False
    finally:
        conn.close()


def try_sync_gate_review(*, experiment_id: str, k: int, review: dict[str, Any]) -> bool:
    """Mirror one SME gate review; False on any failure."""
    try:
        conn = _connect()
    except Exception as exc:  # noqa: BLE001
        logger.warning("gate review store sync skipped (connect): %s", exc)
        return False
    try:
        _insert_gate_review(conn, experiment_id, k, review)
        conn.commit()
        return True
    except Exception as exc:  # noqa: BLE001
        logger.warning("gate review store sync failed: %s", exc)
        try:
            conn.rollback()
        except Exception:  # noqa: BLE001
            pass
        return False
    finally:
        conn.close()


def try_ingest_run(repo_root, run_id: str) -> bool:
    """Ingest one child run's verdicts into rush.llm_label (dedup-safe)."""
    try:
        conn = _connect()
    except Exception as exc:  # noqa: BLE001
        logger.warning("run ingest skipped (connect): %s", exc)
        return False
    try:
        import sys
        from pathlib import Path

        scripts_dir = str(Path(repo_root) / "scripts")
        if scripts_dir not in sys.path:
            sys.path.insert(0, scripts_dir)
        import labelstore_ingest  # noqa: PLC0415

        run_dir = Path(repo_root) / "data" / "runs" / run_id
        labelstore_ingest.ingest_run(conn, run_dir)
        return True
    except Exception as exc:  # noqa: BLE001
        logger.warning("run ingest failed for %s: %s", run_id, exc)
        try:
            conn.rollback()
        except Exception:  # noqa: BLE001
            pass
        return False
    finally:
        conn.close()
