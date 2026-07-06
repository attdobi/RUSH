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
    MNIST_SAMPLE_MANIFEST,
    genai_manifest_default,
)
from pipeline.manifest import HOLDOUT_SPLITS, load_records, select_samples  # noqa: E402
from pipeline.providers._config import resolve_temperature  # noqa: E402
from pipeline.runner import (  # noqa: E402
    DEFAULT_PROMPT_VERSION,
    ModelSpec,
    deterministic_fake_factory,
    run_completed_with_results,
    run_labeling,
)
from pipeline.web.demo_area import DEFAULT_POLICY_AREA, MNIST_POLICY_AREA, normalize_policy_area  # noqa: E402


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
        default=None,
        help=(
            "Path to combined_labels.jsonl. Default: auto-selects the MNIST "
            "manifest for area=MNIST_Digits; for Generative_AI uses the "
            "portable manifest when RUSH_PORTABLE=1 or the full source image "
            f"tree is absent, otherwise {DEFAULT_SAMPLE_MANIFEST}."
        ),
    )
    parser.add_argument("--split", choices=["dev_golden", "holdout", "all"], default="dev_golden")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--sample-ids", default=None,
                        help="Comma-separated sample_ids (overrides --split/--limit filtering).")
    parser.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS_ROOT)
    parser.add_argument("--prompt-version", default=DEFAULT_PROMPT_VERSION)
    parser.add_argument(
        "--policy-version",
        default="v0.1",
        help="Policy graph version directory to label against (default: v0.1).",
    )
    parser.add_argument(
        "--area",
        default=DEFAULT_POLICY_AREA,
        help="Policy/demo area selecting ontology + policy graph (default: Generative_AI).",
    )
    parser.add_argument("--concurrency", type=int, default=1,
                        help="In-flight provider calls per provider (default 1; max recommended: 4).")
    parser.add_argument("--batch-size", type=int, default=20,
                        help="Images per logical provider batch (default 20).")
    parser.add_argument("--reasoning-effort", choices=["high", "xhigh"], default="xhigh",
                        help="OpenAI gpt-5.5 reasoning effort for this run (default: xhigh).")
    parser.add_argument(
        "--local-reasoning",
        default=None,
        help=(
            "Comma-separated local model reasoning toggles, e.g. "
            "local/qwen3.6-27b=on,local/gemma-4-26b-a4b-qat=off."
        ),
    )
    parser.add_argument("--allow-holdout", action="store_true",
                        help="Required to dispatch against the holdout split.")
    parser.add_argument("--live", action="store_true",
                        help="Use real provider clients (requires X1's pipeline.providers).")
    parser.add_argument("--allow-spend", action="store_true",
                        help="Confirms intentional spend; required alongside --live.")
    parser.add_argument("--plan-only", action="store_true",
                        help="Print the dispatch plan and exit without touching disk.")
    parser.add_argument("--no-score", action="store_true",
                        help="Skip automatic scoring after a successful labeling run.")
    return parser.parse_args(argv)


def _parse_local_reasoning_arg(raw: str | None) -> dict[str, bool]:
    """Parse ``model=on|off`` CSV into a local model reasoning map."""
    if raw is None or not raw.strip():
        return {}

    try:
        from pipeline.providers.registry import MODEL_REGISTRY  # type: ignore
    except Exception as exc:  # noqa: BLE001 - explicit operator-facing error
        raise ValueError(
            "local reasoning validation requires pipeline.providers.registry.MODEL_REGISTRY"
        ) from exc

    out: dict[str, bool] = {}
    for item in raw.split(","):
        part = item.strip()
        if not part:
            continue
        if "=" not in part:
            raise ValueError("--local-reasoning entries must use model=on|off")
        model_id, state = [s.strip() for s in part.split("=", 1)]
        if not model_id.startswith("local/"):
            raise ValueError("--local-reasoning model ids must start with local/")
        reg_spec = MODEL_REGISTRY.get(model_id)
        if reg_spec is None or reg_spec.provider != "local":
            raise ValueError(f"--local-reasoning unknown local model_id: {model_id}")
        if state not in {"on", "off"}:
            raise ValueError("--local-reasoning values must be on or off")
        out[model_id] = state == "on"
    return out


def _local_reasoning_runtime_params(model_id: str, enabled: bool) -> dict[str, int | str]:
    if not enabled:
        return {"reasoning_effort": "none", "max_completion_tokens": 4000}
    if model_id.startswith("local/qwen"):
        return {"reasoning_effort": "low", "max_completion_tokens": 6000}
    return {"reasoning_effort": "medium", "max_completion_tokens": 6000}


def _resolve_sample_manifest(area: str, manifest: Path | None) -> Path:
    """Resolve the effective sample manifest while preserving explicit overrides."""
    if manifest is not None:
        return manifest
    if area == MNIST_POLICY_AREA:
        return MNIST_SAMPLE_MANIFEST
    return genai_manifest_default()


def _resolve_factory(
    use_live: bool,
    *,
    reasoning_effort: str | None = None,
    local_reasoning: dict[str, bool] | None = None,
):
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

    local_reasoning = local_reasoning or {}

    def _factory(spec: ModelSpec):
        return build_client(
            spec.model_id,
            reasoning_effort=reasoning_effort,
            local_reasoning_on=local_reasoning.get(spec.model_id),
        )

    return _factory


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        local_reasoning = _parse_local_reasoning_arg(args.local_reasoning)
    except ValueError as exc:
        print(f"[X2] {exc}", file=sys.stderr)
        return 2

    try:
        area = normalize_policy_area(args.area)
    except ValueError as exc:
        print(f"[X2] {exc}", file=sys.stderr)
        return 2

    args.manifest = _resolve_sample_manifest(area, args.manifest)

    if args.live and not args.allow_spend:
        print("[X2] refusing to dispatch live calls without --allow-spend", file=sys.stderr)
        return 2

    if args.split in HOLDOUT_SPLITS and not args.allow_holdout:
        print(f"[X2] refusing to use split={args.split} without --allow-holdout", file=sys.stderr)
        return 2

    if args.batch_size < 1:
        print(f"[X2] --batch-size must be >= 1 (got {args.batch_size})", file=sys.stderr)
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
            params = dict(reg_spec.params)
            if mid == "openai/gpt-5.5":
                params["reasoning_effort"] = args.reasoning_effort
            if mid in local_reasoning:
                params.update(_local_reasoning_runtime_params(mid, local_reasoning[mid]))
            enriched.append(
                ModelSpec(
                    model_id=mid,
                    phase=f"phase-{reg_spec.phase}",
                    params=params if params else None,
                    resolved_temperature=resolve_temperature(reg_spec.provider_model_name),
                )
            )
        else:
            enriched.append(ModelSpec(model_id=mid, resolved_temperature=resolve_temperature(mid)))
    model_specs = enriched
    sample_ids = (
        [s.strip() for s in args.sample_ids.split(",") if s.strip()]
        if args.sample_ids
        else None
    )

    if args.plan_only:
        records = load_records(args.manifest)
        selected = select_samples(records, split=args.split, limit=args.limit, sample_ids=sample_ids)
        n_calls = len(selected) * len(model_specs)
        effective_batches = sum(
            (len(selected) + args.batch_size - 1) // args.batch_size
            for _ in model_specs
        )
        plan = {
            "models": [m.model_id for m in model_specs],
            "split": args.split,
            "limit": args.limit,
            "area": area,
            "policy_version": args.policy_version,
            "batch_size": args.batch_size,
            "effective_batches": effective_batches,
            "n_samples": len(selected),
            "n_calls": n_calls,
            "sample_ids_head": [r.sample_id for r in selected[:5]],
            "dry_run": not args.live,
        }
        json.dump(plan, sys.stdout, indent=2, sort_keys=True)
        sys.stdout.write("\n")
        return 0

    policy_graph_dir = ROOT / "policy-graph" / area / args.policy_version
    if not policy_graph_dir.is_dir():
        print(f"[X2] unknown policy version directory: {policy_graph_dir}", file=sys.stderr)
        return 2
    policy_graph_version = (
        f"{area}.{args.policy_version}" if area != DEFAULT_POLICY_AREA else args.policy_version
    )

    factory = _resolve_factory(
        use_live=args.live,
        reasoning_effort=args.reasoning_effort,
        local_reasoning=local_reasoning,
    )
    summary = run_labeling(
        models=model_specs,
        sample_manifest_path=args.manifest,
        split=args.split,
        limit=args.limit,
        sample_ids=sample_ids,
        runs_root=args.runs_root,
        policy_graph_dir=policy_graph_dir,
        policy_graph_version=policy_graph_version,
        policy_version=args.policy_version,
        area=area,
        prompt_version=args.prompt_version,
        client_factory=factory,
        concurrency=args.concurrency,
        batch_size=args.batch_size,
        allow_holdout=args.allow_holdout,
        dry_run=not args.live,
        reasoning_effort=args.reasoning_effort,
    )

    completed_with_errors = summary.errored_calls > 0 and run_completed_with_results(summary)
    fatal_error = summary.fatal_error

    scoring_result = None
    if not args.no_score and run_completed_with_results(summary):
        try:
            from pipeline.scoring.run_scoring import run_scoring  # noqa: PLC0415

            scoring_result = run_scoring(summary.run_id, ROOT, runs_root=args.runs_root)
        except Exception as exc:  # noqa: BLE001 - keep successful labeling runs successful
            print(
                f"[X2] warning: automatic scoring failed for run {summary.run_id}: "
                f"{type(exc).__name__}: {exc}",
                file=sys.stderr,
            )

    payload = {
        "run_id": summary.run_id,
        "run_dir": str(summary.paths.root),
        "expected_calls": summary.expected_calls,
        "completed_calls": summary.completed_calls,
        "errored_calls": summary.errored_calls,
        "completed_with_errors": completed_with_errors,
        "fatal_error": fatal_error,
        "batch_size": summary.batch_size,
        "effective_batches": summary.effective_batches,
        "started_at": summary.started_at,
        "finished_at": summary.finished_at,
        "dry_run": summary.dry_run,
        "scoring": scoring_result,
    }
    json.dump(payload, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0 if fatal_error is None else 1


if __name__ == "__main__":
    raise SystemExit(main())
