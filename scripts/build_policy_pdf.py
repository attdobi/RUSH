#!/usr/bin/env python3
"""CLI: Build a single bound PDF from a policy-graph version directory.

Usage:
    python scripts/build_policy_pdf.py \
        --source policy-graph/Generative_AI/v0.1 \
        --output web/policy.pdf

Or, for a per-run artifact:
    python scripts/build_policy_pdf.py \
        --source policy-graph/Generative_AI/v0.1 \
        --output data/runs/<run_id>/policy.pdf

The PDF is a build artifact; do not commit it. The web UI links to either
``web/policy.pdf`` (for the demo) or ``data/runs/<run_id>/policy.pdf``
(per-run, recommended once the bulk-labeling runner is wired).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Make `pipeline` importable when invoked as a script.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.pdf import PolicyPdfError, build_policy_pdf  # noqa: E402


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the bound RUSH policy PDF.")
    parser.add_argument(
        "--source",
        type=Path,
        default=ROOT / "policy-graph" / "Generative_AI" / "v0.1",
        help="Policy version directory containing .md files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "web" / "policy.pdf",
        help="Destination PDF path.",
    )
    parser.add_argument(
        "--policy-graph-version",
        default=None,
        help="Override the version label embedded in the cover page.",
    )
    parser.add_argument(
        "--examples-root",
        type=Path,
        default=ROOT / "data",
        help="Data directory, manifest file, or run directory to source node image examples from.",
    )
    parser.add_argument(
        "--examples-per-node",
        type=int,
        default=3,
        help="Maximum number of image examples to render under each policy node.",
    )
    parser.add_argument(
        "--no-examples",
        action="store_true",
        help="Skip image example discovery and render the markdown-only PDF.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit a JSON build summary on stdout.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        result = build_policy_pdf(
            args.source,
            args.output,
            policy_graph_version=args.policy_graph_version,
            examples_root=args.examples_root,
            examples_per_node=args.examples_per_node,
            include_examples=not args.no_examples,
        )
    except PolicyPdfError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(
            json.dumps(
                {
                    "output_path": str(result.output_path),
                    "source_dir": str(result.source_dir),
                    "policy_graph_version": result.policy_graph_version,
                    "file_count": result.file_count,
                    "page_count": result.page_count,
                    "byte_size": result.byte_size,
                    "sources": [str(p) for p in result.sources],
                }
            )
        )
    else:
        print(
            f"built {result.output_path} ({result.byte_size:,} bytes, "
            f"{result.page_count} page(s), {result.file_count} source markdown file(s))"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
