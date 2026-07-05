#!/usr/bin/env python3
"""Aggregate durable cost ledgers across all runs for data analysis.

Scans ``data/runs/*/costs.jsonl`` (the current-pricing ledger) and emits a
single combined dataset (CSV and/or JSONL). Falls back to legacy
``llm_outputs.jsonl`` rows for runs that predate the ledger, flagging them with
``source=legacy_llm_outputs`` and ``pricing_version=legacy`` so analysis can
segment current vs. stale pricing.

Usage:
    python scripts/aggregate_costs.py [--runs-root data/runs] \\
        [--out-csv costs_combined.csv] [--out-jsonl costs_combined.jsonl] \\
        [--include-legacy]
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Iterator

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

FIELDNAMES = [
    "run_id",
    "batch_index",
    "batch_id",
    "image_id",
    "model_id",
    "input_tokens",
    "output_tokens",
    "latency_ms",
    "tokens_per_sec",
    "input_rate_per_mtok",
    "output_rate_per_mtok",
    "image_rate_per_image",
    "image_count",
    "cost_usd",
    "pricing_version",
    "recorded_at",
    "source",
]


def _read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def _rows_from_ledger(run_dir: Path) -> Iterator[dict[str, Any]]:
    for row in _read_jsonl(run_dir / "costs.jsonl"):
        row.setdefault("source", "ledger")
        yield row


def _rows_from_legacy(run_dir: Path) -> Iterator[dict[str, Any]]:
    """Reconstruct ledger-shaped rows from old llm_outputs.jsonl.

    These carry STALE stored cost_usd (do NOT trust for current-pricing
    analysis); flagged pricing_version=legacy.
    """
    run_id = run_dir.name
    for env in _read_jsonl(run_dir / "llm_outputs.jsonl"):
        out = env.get("output") if isinstance(env.get("output"), dict) else {}
        yield {
            "run_id": run_id,
            "batch_index": None,
            "batch_id": None,
            "image_id": env.get("image_id"),
            "model_id": env.get("model_id"),
            "input_tokens": out.get("input_tokens"),
            "output_tokens": out.get("output_tokens"),
            "input_rate_per_mtok": None,
            "output_rate_per_mtok": None,
            "image_rate_per_image": None,
            "image_count": 1,
            "cost_usd": out.get("cost_usd"),
            "pricing_version": "legacy",
            "recorded_at": env.get("recorded_at"),
            "source": "legacy_llm_outputs",
        }


def collect_rows(runs_root: Path, *, include_legacy: bool) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run_dir in sorted(p for p in runs_root.iterdir() if p.is_dir()):
        if (run_dir / "costs.jsonl").exists():
            rows.extend(_rows_from_ledger(run_dir))
        elif include_legacy and (run_dir / "llm_outputs.jsonl").exists():
            rows.extend(_rows_from_legacy(run_dir))
    return rows


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in FIELDNAMES})


def write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", default=str(_REPO_ROOT / "data" / "runs"))
    parser.add_argument("--out-csv", default=None)
    parser.add_argument("--out-jsonl", default=None)
    parser.add_argument("--include-legacy", action="store_true")
    args = parser.parse_args(argv)

    runs_root = Path(args.runs_root)
    if not runs_root.is_dir():
        print(f"runs-root not found: {runs_root}", file=sys.stderr)
        return 2

    rows = collect_rows(runs_root, include_legacy=args.include_legacy)
    out_csv = args.out_csv or str(runs_root.parent / "costs_combined.csv")
    write_csv(rows, Path(out_csv))
    if args.out_jsonl:
        write_jsonl(rows, Path(args.out_jsonl))

    ledger = sum(1 for r in rows if r.get("source") == "ledger")
    legacy = len(rows) - ledger
    print(f"aggregated {len(rows)} rows ({ledger} ledger, {legacy} legacy) -> {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
