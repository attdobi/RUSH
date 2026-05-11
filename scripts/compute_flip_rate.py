#!/usr/bin/env python3
"""Compute flip-rate artifacts across scored labeling runs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline.scoring import compute_flip_rate  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--runs-root",
        default=str(ROOT / "data" / "runs"),
        help="root directory holding per-run dirs (default: data/runs)",
    )
    p.add_argument("--min-runs", type=int, default=2)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    result = compute_flip_rate(repo_root=ROOT, runs_root=Path(args.runs_root), min_runs=args.min_runs)
    print(json.dumps(result, indent=2, sort_keys=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
