#!/usr/bin/env python3
"""CLI for the label-noise simulation suites.

Usage (from the repo root, using the repo venv):
    ./.venv/bin/python sim/label-noise/run_sim.py --suite S2 --dataset genai
    ./.venv/bin/python sim/label-noise/run_sim.py --suite all --dataset both --seeds 12

Results land in sim/label-noise/results/<suite>_<dataset>.json (gitignored;
regenerate them — every suite is fully seeded).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from labelsim.experiments import SUITES, run_suite  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--suite", default="all",
                    choices=["all", *SUITES.keys()])
    ap.add_argument("--dataset", default="genai",
                    choices=["genai", "mnist", "both"])
    ap.add_argument("--seeds", type=int, default=12,
                    help="independent worlds per cell (default 12)")
    ap.add_argument("--out", default=str(HERE / "results"))
    args = ap.parse_args()

    suites = list(SUITES.keys()) if args.suite == "all" else [args.suite]
    datasets = ["genai", "mnist"] if args.dataset == "both" else [args.dataset]
    for suite in suites:
        for dataset in datasets:
            t0 = time.time()
            result = run_suite(suite, dataset=dataset, n_seeds=args.seeds,
                               out_dir=args.out)
            dt = time.time() - t0
            print(f"[{suite} {dataset}] {dt:.1f}s -> {args.out}/{suite.lower()}_{dataset}.json")
            _headline(result)
    return 0


def _headline(result: dict) -> None:
    suite = result["suite"]
    if suite in ("S1",):
        for c in result["cells"]:
            f1, ci = c["final_oracle_f1"]
            dd, _ = c["divergence_from_clean_final"]
            print(f"  {c['model']:9s} rate={c['rate']:.2f}  "
                  f"oracle-F1 {f1:.3f}±{ci:.3f}  div-from-clean {dd:.3f}")
    elif suite == "S2":
        f1c, cic = result["clean_reference_f1"]
        print(f"  clean reference     oracle-F1 {f1c:.3f}±{cic:.3f}")
        for c in result["cells"]:
            f1, ci = c["final_oracle_f1"]
            au, _ = c["detection_auroc"]
            ov, _ = c["emergent_overturn_rate"]
            print(f"  {c['arm']:20s} oracle-F1 {f1:.3f}±{ci:.3f}  "
                  f"det-AUROC {au:.3f}  overturn {ov:.2f}")
    elif suite == "S3":
        for p in result["pairs"]:
            dd, ci = p["decision_disagreement_final"]
            print(f"  rate={p['rate']:.2f}  deweight-vs-upweight disagreement "
                  f"{dd:.3f}±{ci:.3f}")
        infl = result["influence"]
        if infl:
            import numpy as np
            fl = [i["influence_dd"] for i in infl if i["is_flipped"]]
            cl = [i["influence_dd"] for i in infl if not i["is_flipped"]]
            if fl and cl:
                print(f"  point influence: flipped {np.mean(fl):.4f} vs "
                      f"clean {np.mean(cl):.4f} (n={len(fl)}/{len(cl)})")
    elif suite == "S4":
        for c in result["cells"]:
            fa, _ = c["gate_false_accept_rate"]
            fr, _ = c["gate_false_reject_rate"]
            f1, ci = c["final_oracle_f1"]
            print(f"  test-noise {c['rate']:.2f} readj_test={str(c['readj_test']):5s}  "
                  f"gate FA {fa:.2f} FR {fr:.2f}  oracle-F1 {f1:.3f}±{ci:.3f}")
    elif suite == "S5":
        for c in result["cells"]:
            f1, ci = c["final_oracle_f1"]
            g, _ = c["anchor_gini"]
            ct, _ = c["anchor_contamination"]
            print(f"  {c['strategy']:10s} rate={c['rate']:.2f}  "
                  f"oracle-F1 {f1:.3f}±{ci:.3f}  gini {g:.2f}  contam {ct:.2f}")


if __name__ == "__main__":
    raise SystemExit(main())
