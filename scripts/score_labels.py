#!/usr/bin/env python3
"""Score a bulk-labeling run.

Reads ``data/runs/<run_id>/label_votes.jsonl`` plus the SME manifest, writes:

    data/runs/<run_id>/scoring/decision_quality.json
    data/runs/<run_id>/scoring/misalignment.json
    data/runs/<run_id>/scoring/borderline.json
    data/runs/<run_id>/web/{summary,misalignment,borderline}.json

Stdlib-only; no network. Determinism: outputs are stable for a given input.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline.scoring import (  # noqa: E402
    borderline as borderline_mod,
    decision_quality as dq_mod,
    exporters as exporters_mod,
    misalignment as mis_mod,
)


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
    p.add_argument(
        "--policy-graph-version",
        default="Generative_AI.v0.1",
    )
    p.add_argument(
        "--ground-truth-tier",
        default="gold,platinum,gold_candidate",
        help="comma-separated truth tiers to count as ground truth",
    )
    p.add_argument(
        "--low-confidence-threshold",
        type=float,
        default=0.6,
    )
    p.add_argument(
        "--validate-schemas",
        action="store_true",
        help="attempt JSON Schema validation (requires jsonschema installed)",
    )
    return p.parse_args()


def _atomic_write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=False), encoding="utf-8")
    tmp.replace(path)


def main() -> int:
    args = parse_args()
    run_dir = Path(args.runs_root) / args.run_id
    votes_path = run_dir / "label_votes.jsonl"
    if not votes_path.exists():
        print(f"ERROR: missing {votes_path}", file=sys.stderr)
        return 2

    tiers = tuple(t.strip() for t in args.ground_truth_tier.split(",") if t.strip())
    schemas_dir = ROOT / "schemas" if args.validate_schemas else None

    dq = dq_mod.compute_decision_quality(
        votes_path,
        Path(args.manifest),
        policy_graph_version=args.policy_graph_version,
        ground_truth_tier=tiers,
        schemas_dir=schemas_dir,
    )
    mis = mis_mod.compute_misalignment(
        votes_path,
        Path(args.manifest),
        policy_graph_version=args.policy_graph_version,
        ground_truth_tier=tiers,
    )
    bord = borderline_mod.compute_borderline(
        votes_path,
        Path(args.manifest),
        policy_graph_version=args.policy_graph_version,
        ground_truth_tier=tiers,
        low_confidence_threshold=args.low_confidence_threshold,
    )

    scoring_dir = run_dir / "scoring"
    _atomic_write_json(scoring_dir / "decision_quality.json", dq)
    _atomic_write_json(scoring_dir / "misalignment.json", mis)
    _atomic_write_json(scoring_dir / "borderline.json", bord)

    written = exporters_mod.write_web_exports(
        run_dir,
        decision_quality=dq,
        misalignment=mis,
        borderline=bord,
        run_id=args.run_id,
    )
    for name, path in written.items():
        try:
            display = path.relative_to(ROOT)
        except ValueError:
            display = path
        print(f"wrote {name}: {display}")
    print(
        f"DQ: {len(dq.get('labelers', []))} labelers | "
        f"misalignment: {mis['summary']} | "
        f"borderline: {bord['summary']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
