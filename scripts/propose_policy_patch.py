#!/usr/bin/env python3
"""Propose policy patches from a scored run.

Default behavior is **dry-run**: assembles the LLM prompt and prints it
without making any network call. Engineers MUST NOT execute live calls; the
``--execute`` flag is reserved for Pista when authorising spend.

Reads:
    data/runs/<run_id>/scoring/misalignment.json
    data/runs/<run_id>/scoring/borderline.json (optional)
    policy-graph/Generative_AI/v0.1/*.md

Writes (only on --execute):
    data/runs/<run_id>/policy_patches.jsonl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline.policy_iterator import (  # noqa: E402
    PolicyIterationInputs,
    load_policy_markdown,
    propose_policy_patches,
    write_patches_jsonl,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-id", required=True)
    p.add_argument("--runs-root", default=str(ROOT / "data" / "runs"))
    p.add_argument(
        "--policy-dir",
        default=str(ROOT / "policy-graph" / "Generative_AI" / "v0.1"),
    )
    p.add_argument("--policy-graph-version", default="Generative_AI.v0.1")
    p.add_argument("--model-id", default="openai/gpt-5.5")
    p.add_argument("--reasoning-effort", default="high")
    p.add_argument("--severity", default="high,medium")
    p.add_argument("--max-rows", type=int, default=40)
    p.add_argument(
        "--include-images",
        action="store_true",
        help="include downsampled images via X1 helper (NOT WIRED in dry-run)",
    )
    p.add_argument(
        "--execute",
        action="store_true",
        help="actually call the LLM (Pista only; default is dry-run)",
    )
    p.add_argument(
        "--prompt-out",
        default=None,
        help="optional path to write the assembled prompt JSON",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = Path(args.runs_root) / args.run_id
    mis_path = run_dir / "scoring" / "misalignment.json"
    bord_path = run_dir / "scoring" / "borderline.json"
    if not mis_path.exists():
        print(f"ERROR: missing {mis_path}; run scripts/score_labels.py first", file=sys.stderr)
        return 2
    misalignment = json.loads(mis_path.read_text(encoding="utf-8"))
    borderline = (
        json.loads(bord_path.read_text(encoding="utf-8")) if bord_path.exists() else None
    )
    policy_md = load_policy_markdown(Path(args.policy_dir))
    inputs = PolicyIterationInputs(
        misalignment=misalignment,
        borderline=borderline,
        policy_markdown=policy_md,
        policy_graph_version=args.policy_graph_version,
    )

    severity = tuple(s.strip() for s in args.severity.split(",") if s.strip())

    chat_callable = None
    if args.execute:
        try:
            from pipeline.providers import registry  # type: ignore
        except Exception as exc:
            print(
                "ERROR: --execute needs pipeline.providers.registry from X1 (not yet available): "
                f"{exc}",
                file=sys.stderr,
            )
            return 3
        # X1 will expose a get_chat_callable(model_id) -> ChatCallable factory.
        getter = getattr(registry, "get_chat_callable", None)
        if getter is None:
            print(
                "ERROR: pipeline.providers.registry.get_chat_callable not implemented yet",
                file=sys.stderr,
            )
            return 3
        chat_callable = getter(args.model_id)

    if args.include_images and not args.execute:
        print("(note) --include-images has no effect in dry-run; prompt will omit images.")

    result = propose_policy_patches(
        inputs=inputs,
        chat_callable=chat_callable,
        model_id=args.model_id,
        reasoning_effort=args.reasoning_effort,
        severity=severity,
        max_rows=args.max_rows,
        include_images=args.include_images and args.execute,
        # downsample_helper is only wired when X1 plugs it in:
        downsample_helper=None,
        image_root=ROOT if args.include_images and args.execute else None,
        schemas_dir=ROOT / "schemas",
    )

    if args.prompt_out:
        Path(args.prompt_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.prompt_out).write_text(
            json.dumps(result["prompt"], indent=2), encoding="utf-8"
        )
        print(f"wrote prompt -> {args.prompt_out}")

    if result.get("dry_run"):
        print("DRY RUN: no LLM call made.")
        print(f"prompt rows: {len(result['prompt']['misclassifications'])}")
        print(f"prompt size (bytes): {len(json.dumps(result['prompt']))}")
        return 0

    out_path = run_dir / "policy_patches.jsonl"
    write_patches_jsonl(out_path, result["patches"])
    try:
        display = out_path.relative_to(ROOT)
    except ValueError:
        display = out_path
    print(f"wrote {len(result['patches'])} patches -> {display}")
    if result["errors"]:
        print(f"validation errors: {len(result['errors'])}", file=sys.stderr)
        for e in result["errors"]:
            print(f"  - {e}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
