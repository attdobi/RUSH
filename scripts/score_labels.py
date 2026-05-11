#!/usr/bin/env python3
"""Score a bulk-labeling run.

Writes canonical scoring artifacts under ``data/runs/<run_id>/scoring/`` plus
browser exports under ``data/runs/<run_id>/web/``. The implementation lives in
``pipeline.scoring.run_scoring`` so the local web API can invoke the same chain
in-process when a run completes.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline.scoring import run_scoring  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-id", required=True, help="run_id under data/runs/")
    p.add_argument(
        "--runs-root",
        default=str(ROOT / "data" / "runs"),
        help="root directory holding per-run dirs (default: data/runs)",
    )
    p.add_argument(
        "--manifest",
        default=str(
            ROOT / "data" / "images" / "genai-classification" / "manifests" / "combined_labels.jsonl"
        ),
        help="SME ground-truth manifest path",
    )
    p.add_argument("--policy-graph-version", default="Generative_AI.v0.1")
    p.add_argument(
        "--ground-truth-tier",
        default="gold,platinum,gold_candidate",
        help="comma-separated truth tiers to count as ground truth",
    )
    p.add_argument("--low-confidence-threshold", type=float, default=0.6)
    p.add_argument(
        "--validate-schemas",
        action="store_true",
        help="attempt JSON Schema validation (requires jsonschema installed)",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    tiers = tuple(t.strip() for t in args.ground_truth_tier.split(",") if t.strip())
    try:
        result = run_scoring(
            args.run_id,
            ROOT,
            runs_root=Path(args.runs_root),
            manifest=Path(args.manifest),
            policy_graph_version=args.policy_graph_version,
            ground_truth_tier=tiers,
            low_confidence_threshold=args.low_confidence_threshold,
            validate_schemas=args.validate_schemas,
        )
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    for name, path_text in result.get("written", {}).items():
        path = Path(path_text)
        try:
            display = path.relative_to(ROOT)
        except ValueError:
            display = path
        print(f"wrote {name}: {display}")

    mis = result.get("misalignment_summary", {})
    bord = result.get("borderline_summary", {})
    consensus = result.get("consensus_summary", {})
    cost = result.get("cost_summary", {})
    print(
        f"DQ: {result.get('decision_quality', {}).get('n_labelers', 0)} labelers | "
        f"misalignment: {mis} | borderline: {bord} | consensus: {consensus} | "
        f"total_cost_usd: {float(cost.get('total_cost_usd') or 0):.6f} | "
        f"cost_per_1000_labels: {cost.get('cost_per_1000_labels')}"
    )
    flip = result.get("flip_rate", {})
    if flip.get("skipped"):
        print(f"flip-rate skipped: {flip.get('reason')}")
    else:
        print(f"flip-rate wrote: {flip.get('output_dir')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
