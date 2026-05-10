#!/usr/bin/env python3
"""CLI wrapper around ``pipeline.runner.run_labeling`` (X2).

Defaults are intentionally safe:
    * ``--dry-run`` is ON unless you pass ``--live`` AND ``--allow-spend``.
    * ``--allow-holdout`` is required to touch the holdout split.
    * Live mode requires X1's provider clients to be wired (currently NotImplemented).

Examples
--------

Dry run (offline, no provider calls), 5 dev_golden samples × 1 model:

    python scripts/run_bulk_labeling.py \
        --models openai/gpt-5.5 \
        --split dev_golden --limit 5

Print the run plan without writing anything:

    python scripts/run_bulk_labeling.py --models openai/gpt-5.5 --plan-only
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Make `pipeline` importable when running the script directly.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.io_paths import (  # noqa: E402  (after sys.path edit)
    DEFAULT_RUNS_ROOT,
    DEFAULT_SAMPLE_MANIFEST,
)
from pipeline.manifest import HOLDOUT_SPLITS, load_records, select_samples  # noqa: E402
from pipeline.runner import (  # noqa: E402
    DEFAULT_PROMPT_VERSION,
    ModelSpec,
    deterministic_fake_factory,
    run_labeling,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a bulk-labeling pass across N images × M models.",
    )
    parser.add_argument(
        "--models",
        required=True,
        help="Comma-separated model ids (e.g. openai/gpt-5.5,anthropic/claude-opus-4-6).",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_SAMPLE_MANIFEST,
        help=f"Path to combined_labels.jsonl (default: {DEFAULT_SAMPLE_MANIFEST}).",
    )
    parser.add_argument("--split", choices=["dev_golden", "holdout", "all"], default="dev_golden")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--sample-ids", default=None,
                        help="Comma-separated sample_ids (overrides --split/--limit filtering).")
    parser.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS_ROOT)
    parser.add_argument("--prompt-version", default=DEFAULT_PROMPT_VERSION)
    parser.add_argument("--concurrency", type=int, default=1,
                        help="In-flight provider calls per provider (default 1; max recommended: 4).")
    parser.add_argument("--allow-holdout", action="store_true",
                        help="Required to dispatch against the holdout split.")
    parser.add_argument("--live", action="store_true",
                        help="Use real provider clients (requires X1's pipeline.providers).")
    parser.add_argument("--allow-spend", action="store_true",
                        help="Confirms intentional spend; required alongside --live.")
    parser.add_argument("--plan-only", action="store_true",
                        help="Print the dispatch plan and exit without touching disk.")
    return parser.parse_args(argv)


def _resolve_factory(use_live: bool):
    """Return a client factory.

    In dry-run mode we use the deterministic fake. In live mode we lazy-import
    X1's registry (``pipeline.providers.registry.build_client``) and adapt it
    into the runner's ``(ModelSpec) -> LabelClient`` factory shape.
    """
    if not use_live:
        return deterministic_fake_factory

    try:
        from pipeline.providers.registry import build_client  # type: ignore
    except Exception as exc:  # noqa: BLE001 - explicit operator-facing error
        raise SystemExit(
            "[X2] --live requires pipeline.providers.registry.build_client "
            f"(X1's slice). Import failed: {type(exc).__name__}: {exc}"
        )

    def _factory(spec: ModelSpec):
        return build_client(spec.model_id)

    return _factory


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.live and not args.allow_spend:
        print("[X2] refusing to dispatch live calls without --allow-spend", file=sys.stderr)
        return 2

    if args.split in HOLDOUT_SPLITS and not args.allow_holdout:
        print(f"[X2] refusing to use split={args.split} without --allow-holdout", file=sys.stderr)
        return 2

    # If X1's registry is importable, enrich each ModelSpec with phase + params
    # so they show up in run_manifest.json. Stays optional for dry-run/CI.
    requested_ids = [m.strip() for m in args.models.split(",") if m.strip()]
    enriched: list[ModelSpec] = []
    try:
        from pipeline.providers.registry import MODEL_REGISTRY  # type: ignore
    except Exception:
        MODEL_REGISTRY = {}  # type: ignore[assignment]
    for mid in requested_ids:
        reg_spec = MODEL_REGISTRY.get(mid)  # type: ignore[union-attr]
        if reg_spec is not None:
            enriched.append(
                ModelSpec(
                    model_id=mid,
                    phase=f"phase-{reg_spec.phase}",
                    params=dict(reg_spec.params) if reg_spec.params else None,
                )
            )
        else:
            enriched.append(ModelSpec(model_id=mid))
    model_specs = enriched
    sample_ids = (
        [s.strip() for s in args.sample_ids.split(",") if s.strip()]
        if args.sample_ids
        else None
    )

    if args.plan_only:
        records = load_records(args.manifest)
        selected = select_samples(records, split=args.split, limit=args.limit, sample_ids=sample_ids)
        plan = {
            "models": [m.model_id for m in model_specs],
            "split": args.split,
            "limit": args.limit,
            "n_samples": len(selected),
            "n_calls": len(selected) * len(model_specs),
            "sample_ids_head": [r.sample_id for r in selected[:5]],
            "dry_run": not args.live,
        }
        json.dump(plan, sys.stdout, indent=2, sort_keys=True)
        sys.stdout.write("\n")
        return 0

    factory = _resolve_factory(use_live=args.live)
    summary = run_labeling(
        models=model_specs,
        sample_manifest_path=args.manifest,
        split=args.split,
        limit=args.limit,
        sample_ids=sample_ids,
        runs_root=args.runs_root,
        prompt_version=args.prompt_version,
        client_factory=factory,
        concurrency=args.concurrency,
        allow_holdout=args.allow_holdout,
        dry_run=not args.live,
    )

    payload = {
        "run_id": summary.run_id,
        "run_dir": str(summary.paths.root),
        "expected_calls": summary.expected_calls,
        "completed_calls": summary.completed_calls,
        "errored_calls": summary.errored_calls,
        "started_at": summary.started_at,
        "finished_at": summary.finished_at,
        "dry_run": summary.dry_run,
    }
    json.dump(payload, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0 if summary.errored_calls == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
