#!/usr/bin/env python3
"""Run the RUSH escalation cascade end to end.

The cascade is the thesis of the system: spend the expensive resource only where
the cheap one fails.

  Tier 1 (cheap)   : a panel of low-cost models labels EVERY image and reaches
                     consensus on the aligned majority.
  Tier 2 (escalate): only the images where tier 1 lacked consensus / split /
                     flagged a boundary are re-judged by a higher-reasoning pass.
  Tier 3 (SME)     : whatever tier 2 still can't resolve is the boundary residual
                     that a human adjudicates.

This orchestrator REUSES the existing run + scoring path — it runs
``run_bulk_labeling.py`` for tier 1, derives the escalation set straight from
tier 1's ``consensus.json`` (``is_split`` / not ``is_consensus`` / ``is_boundary``),
then runs tier 2 scoped to just those sample_ids (``--sample-ids``). It writes a
``cascade.json`` next to the tier-1 run summarizing what each tier resolved and
what it cost, so the saving is measurable.

For a free local demo: cheap = gemma + qwen (reasoning off, fast); escalate =
qwen with reasoning ON — reasoning depth is the escalation lever (the "H" in
RUSH). Example::

    scripts/run_cascade.py --area MNIST_Digits --split all --limit 20 \
        --cheap local/gemma-4-26b-a4b-qat,local/qwen3.6-35b-a3b \
        --escalate local/qwen3.6-35b-a3b --allow-holdout
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline.scoring.consensus import select_escalation_ids  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RUNNER = "scripts/run_bulk_labeling.py"


def _python() -> str:
    venv = ROOT / ".venv" / "bin" / "python"
    return str(venv) if venv.exists() else sys.executable


def _last_json_object(text: str) -> dict | None:
    """Return the last top-level ``{...}`` JSON object printed to stdout."""
    depth = 0
    start = None
    last: dict | None = None
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                try:
                    last = json.loads(text[start : i + 1])
                except json.JSONDecodeError:
                    pass
    return last


def _run_tier(
    *,
    models: list[str],
    area: str,
    policy_version: str,
    manifest: str | None,
    local_reasoning: str | None,
    concurrency: int,
    allow_holdout: bool,
    split: str | None = None,
    limit: int | None = None,
    sample_ids: list[str] | None = None,
    label: str,
) -> dict:
    argv = [
        _python(), "-u", RUNNER,
        "--area", area,
        "--models", ",".join(models),
        "--policy-version", policy_version,
        "--live", "--allow-spend",
        "--concurrency", str(concurrency),
        "--batch-size", "20",
    ]
    if manifest:
        argv += ["--manifest", manifest]
    if sample_ids is not None:
        argv += ["--sample-ids", ",".join(sample_ids)]
    else:
        argv += ["--split", split or "dev_golden", "--limit", str(limit or 20)]
    if allow_holdout:
        argv += ["--allow-holdout"]
    if local_reasoning:
        argv += ["--local-reasoning", local_reasoning]

    print(f"[cascade] tier '{label}': {' '.join(models)} "
          f"({'reasoning ' + local_reasoning if local_reasoning else 'default'})", file=sys.stderr)
    t0 = time.monotonic()
    proc = subprocess.run(argv, cwd=str(ROOT), capture_output=True, text=True)
    wall = time.monotonic() - t0
    payload = _last_json_object(proc.stdout)
    if payload is None or not payload.get("run_id"):
        sys.stderr.write(proc.stdout[-1500:] + "\n" + proc.stderr[-1500:] + "\n")
        raise SystemExit(f"[cascade] tier '{label}' produced no run payload (exit {proc.returncode})")
    payload["_wall_s"] = round(wall, 1)
    return payload


def _run_dir(run_id: str) -> Path:
    return ROOT / "data" / "runs" / run_id


def _partition_by_consensus(run_id: str) -> tuple[list[str], list[str], list[dict]]:
    """(escalate_ids, resolved_ids, records) from a scored run's consensus.json.

    Escalation uses the shared, unit-tested ``select_escalation_ids`` so the
    orchestrator and any future backend/UI agree on what "the cheap tier couldn't
    resolve" means.
    """
    path = _run_dir(run_id) / "web" / "consensus.json"
    records = json.loads(path.read_text(encoding="utf-8")).get("records", []) if path.exists() else []
    escalate = select_escalation_ids(records)
    escalate_set = set(escalate)
    resolved = [str(r.get("image_id")) for r in records if r.get("image_id") and str(r.get("image_id")) not in escalate_set]
    return escalate, resolved, records


def _tier_cost_time(run_id: str) -> dict:
    manifest = _run_dir(run_id) / "run_manifest.json"
    total = {}
    if manifest.exists():
        total = json.loads(manifest.read_text(encoding="utf-8")).get("per_model_timing", {}).get("total", {}) or {}
    return {
        "cost_usd": round(float(total.get("total_cost_usd") or 0.0), 6),
        "active_elapsed_s": total.get("active_elapsed_s"),
        "calls": total.get("calls"),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--area", default="MNIST_Digits")
    ap.add_argument("--split", default="all")
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--policy-version", default="v0.1")
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--cheap", default="local/gemma-4-26b-a4b-qat,local/qwen3.6-35b-a3b",
                    help="Tier-1 cheap panel (comma-separated). Needs >=2 for consensus signal.")
    ap.add_argument("--escalate", default="local/qwen3.6-35b-a3b",
                    help="Tier-2 high-reasoning panel (comma-separated).")
    ap.add_argument("--cheap-reasoning", default="off", choices=["off", "on"])
    ap.add_argument("--escalate-reasoning", default="on", choices=["off", "on"])
    ap.add_argument("--concurrency", type=int, default=2)
    ap.add_argument("--allow-holdout", action="store_true")
    args = ap.parse_args(argv)

    cheap = [m.strip() for m in args.cheap.split(",") if m.strip()]
    escalate = [m.strip() for m in args.escalate.split(",") if m.strip()]
    if len(cheap) < 2:
        print("[cascade] warning: cheap tier has <2 models, so consensus/split is trivial and "
              "nothing will escalate.", file=sys.stderr)

    def local_reasoning_arg(models: list[str], mode: str) -> str | None:
        locals_ = [m for m in models if m.startswith("local/")]
        return ",".join(f"{m}={mode}" for m in locals_) if locals_ else None

    # --- Tier 1: cheap panel labels everything ---
    tier1 = _run_tier(
        models=cheap, area=args.area, policy_version=args.policy_version, manifest=args.manifest,
        local_reasoning=local_reasoning_arg(cheap, args.cheap_reasoning),
        concurrency=args.concurrency, allow_holdout=args.allow_holdout,
        split=args.split, limit=args.limit, label="cheap",
    )
    run1 = tier1["run_id"]
    escalate_ids, resolved_ids, records = _partition_by_consensus(run1)
    n_total = len(records)

    # --- Tier 2: escalate only the disagreements ---
    tier2 = None
    residual_ids: list[str] = []
    if escalate_ids:
        tier2 = _run_tier(
            models=escalate, area=args.area, policy_version=args.policy_version, manifest=args.manifest,
            local_reasoning=local_reasoning_arg(escalate, args.escalate_reasoning),
            concurrency=args.concurrency, allow_holdout=args.allow_holdout,
            sample_ids=escalate_ids, label="escalate",
        )
        # Residual = still unresolved after the high-reasoning pass -> SME queue.
        residual_ids, _, _ = _partition_by_consensus(tier2["run_id"])

    cascade = {
        "area": args.area,
        "created_from_run": run1,
        "n_total": n_total,
        "tier1_cheap": {
            "run_id": run1, "models": cheap, "reasoning": args.cheap_reasoning,
            "resolved": len(resolved_ids), "escalated": len(escalate_ids),
            "wall_s": tier1["_wall_s"], **_tier_cost_time(run1),
        },
        "tier2_escalate": (
            {
                "run_id": tier2["run_id"], "models": escalate, "reasoning": args.escalate_reasoning,
                "judged": len(escalate_ids), "residual_to_sme": len(residual_ids),
                "wall_s": tier2["_wall_s"], **_tier_cost_time(tier2["run_id"]),
            }
            if tier2 else None
        ),
        "sme_queue": residual_ids,
        "resolved_cheap_fraction": round(len(resolved_ids) / n_total, 3) if n_total else None,
    }
    out_path = _run_dir(run1) / "cascade.json"
    out_path.write_text(json.dumps(cascade, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    # --- Human-readable summary ---
    t1 = cascade["tier1_cheap"]
    print(f"\n=== RUSH escalation cascade ({args.area}) ===")
    print(f"Tier 1 (cheap: {', '.join(cheap)}, reasoning {args.cheap_reasoning}):")
    print(f"  labeled {n_total} · resolved by consensus {t1['resolved']} · "
          f"escalated {t1['escalated']} · {t1['wall_s']}s · ${t1['cost_usd']:.4f}")
    if tier2:
        t2 = cascade["tier2_escalate"]
        print(f"Tier 2 (escalate: {', '.join(escalate)}, reasoning {args.escalate_reasoning}):")
        print(f"  re-judged {t2['judged']} · residual to SME {t2['residual_to_sme']} · "
              f"{t2['wall_s']}s · ${t2['cost_usd']:.4f}")
    frac = cascade["resolved_cheap_fraction"]
    if frac is not None:
        print(f"Cheap tier resolved {frac * 100:.0f}% of the stream; "
              f"only {len(escalate_ids)} escalated, {len(residual_ids)} reach a human.")
    print(f"wrote {out_path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
