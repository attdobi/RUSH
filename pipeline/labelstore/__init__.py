"""RUSH label store — Postgres-backed, per §4 of the re-adjudication loop MVP.

Design stance: run-dir JSONL files remain the portable transport and per-run
provenance (a fresh clone still demos with no database); this store is the
cross-run AGGREGATION layer — deduped LLM verdicts, human label history, and
golden-set state. Human labels (label_event / golden_label) live here and
only here once the loop is live.

Connection comes from ``RUSH_DB_URL`` (default ``postgresql:///adobi`` — the
local Postgres.app server). All objects live in the ``rush`` schema so
``DROP SCHEMA rush CASCADE`` removes the store without touching anything else
in the shared database.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

SCHEMA_PATH = Path(__file__).with_name("schema.sql")
DEFAULT_DB_URL = "postgresql:///adobi"

# §4.1 confidence tiers -> sampling weights.
TIER_WEIGHTS = {
    "seed_bpo": 1.0,
    "sme_1": 1.0,
    "sme_2_confirmed": 2.0,
    "sme_2_contested": 0.5,
    "sme_3": 3.0,
}

# Per-item cap on human touches (§4.1: sme_3 is "at cap; never queued again").
SME_CAP = 3


def human_confidence(num_agreeing: int) -> float:
    """Attila's human-label confidence: p = 1 - 1/(m + 0.2).

    ``m`` counts the human labels AGREEING with the current resolved label
    (per category, or the high-level binary label). A lone human label is
    weak evidence — m=1 -> 0.167, m=2 -> 0.545, m=3 -> 0.688 — agreement
    compounds it, saturating toward 1.
    """
    m = max(0, int(num_agreeing))
    # The raw formula goes negative below m=1 (1 - 1/0.2 = -4); clamp at 0 —
    # "no agreeing human evidence" is zero confidence, not negative.
    return max(0.0, 1.0 - 1.0 / (m + 0.2))


def db_url() -> str:
    return os.environ.get("RUSH_DB_URL") or DEFAULT_DB_URL


def connect():
    """Open a psycopg connection to the label store."""
    import psycopg

    return psycopg.connect(db_url())


def init_schema(conn) -> None:
    """Apply schema.sql (idempotent: CREATE IF NOT EXISTS throughout)."""
    conn.execute(SCHEMA_PATH.read_text(encoding="utf-8"))
    conn.commit()


def confidence_tier(sme_events: list[str], current_label: str) -> str:
    """§4.1 tier from the SME event history and the resolved label.

    ``sme_events`` is the chronological list of SME labels for one item
    (seed labels count as SME events, per the doc's sme_1 note).
    """
    n = len(sme_events)
    if n == 0:
        return "seed_bpo"
    if n == 1:
        return "sme_1"
    if n == 2:
        return "sme_2_confirmed" if sme_events[0] == sme_events[1] else "sme_2_contested"
    return "sme_3"


def resolve_golden(sme_events: list[str]) -> str | None:
    """Resolved label per §4.1: majority when >=3, latest event on a 1-1 split.

    Returns None when there are no events (caller falls back to seed data).
    """
    if not sme_events:
        return None
    n = len(sme_events)
    if n == 1:
        return sme_events[0]
    if n == 2:
        # Contested resolves to the LATEST event (queue-boosted for a tiebreak).
        return sme_events[-1] if sme_events[0] != sme_events[1] else sme_events[0]
    counts: dict[str, int] = {}
    for label in sme_events:
        counts[label] = counts.get(label, 0) + 1
    best = max(counts.values())
    winners = [label for label, c in counts.items() if c == best]
    if len(winners) == 1:
        return winners[0]
    # Tied majority (e.g. 3 events, 3 labels): latest event among the winners.
    for label in reversed(sme_events):
        if label in winners:
            return label
    return sme_events[-1]


def golden_row(sme_events: list[str], *, seed_source: str, last_epoch: int | None) -> dict[str, Any]:
    """Materialize the golden_label fields from an SME event history."""
    label = resolve_golden(sme_events)
    if label is None:
        raise ValueError("golden_row needs at least one label event")
    agree = sum(1 for event in sme_events if event == label)
    tier = confidence_tier(sme_events, label)
    return {
        "current_label": label,
        "seed_source": seed_source,
        "num_sme_labels": len(sme_events),
        "num_sme_agree_current": agree,
        "confidence_tier": tier,
        "human_confidence": round(human_confidence(agree), 6),
        "at_cap": len(sme_events) >= SME_CAP,
        "last_epoch": last_epoch,
    }
