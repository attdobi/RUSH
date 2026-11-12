"""Read-only adapters for the existing RUSH demo and durable handoff contracts.

The legacy adjudication API writes a separate JSONL log and cannot accept
input/output targets or pin a new workspace epoch. This adapter deliberately
does not forward writes or bypass existing experiment/spend authorization.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def read_rush_queue(repo_root: str | Path, area: str) -> dict:
    """Read RUSH's current ranked queue without relabeling or recomputing it.

Callers must map the returned legacy ontology and content references into
their configured labeling project before POST /api/labeling/items/import.
Source rank, policy, run and model evidence should survive that mapping.
"""
    from pipeline.experiment import aggregate_readjudication
    from pipeline.web.demo_area import normalize_policy_area

    validated_area = normalize_policy_area(area)
    return aggregate_readjudication(Path(repo_root).resolve(), area=validated_area)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Inspect the existing RUSH queue without changing it")
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--area", required=True)
    args = parser.parse_args(argv)
    print(json.dumps(read_rush_queue(args.repo_root, args.area), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
