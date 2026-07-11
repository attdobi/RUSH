"""Pre-registered simulation suites.

S1  Dose-response: how much does label noise bend the trajectory?
    (divergence from the clean twin universe + final oracle F1, by noise
    model x rate, no mitigation)
S2  Mitigation grid: does the weighting methodology converge to truth?
    (off / deweight / deweight_hard / upweight / re-adjudication by
    stack-rank / PPS / random / deweight+readj — final oracle F1, mislabel-
    detection AUROC, emergent overturn rate, residual label errors)
S3  Parallel universes: deweight-vs-upweight trajectory divergence at low vs
    high noise, plus per-point influence — do flipped points diverge the
    twin universes more than clean points ("never mattered much" test)?
S4  Test-set corruption: noise ONLY in the test partition — how often does
    the gate accept steps the oracle would reject (and vice versa), and does
    re-adjudicating the test set fix it?
S5  Sampling strategy: stack-rank vs PPS vs random anchor selection under
    noise — does deterministic stack-ranking hyper-focus on contaminated
    anchors (Gini concentration, anchor contamination)?

Every cell runs n_seeds independent worlds; aggregates are mean +/- 95% CI.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np

from .confidence import ConfidenceConfig
from .engine import (CrankConfig, PanelConfig, build_world, clean_labels,
                     divergence_series, point_influence, run_crank, run_pair)
from .metrics import gini, mean_ci
from .noise import NoiseConfig
from .readjudication import ReadjConfig

BASE_SEED = 13


def suite_noise(dataset: str, model: str = "boundary", rate: float = 0.25,
                one_way: bool | None = None) -> NoiseConfig:
    """Dataset-appropriate noise. GenAI defaults to ONE-WAY flips (gen_ai
    labeled not_gen_ai — directional adult-vs-racy-style confusion, which
    actually drags a boundary; two-sided flips mostly cancel). MNIST defaults
    to two-sided runner-up flips (pair directions don't cancel in
    multiclass)."""
    if one_way is None:
        one_way = dataset == "genai"
    return NoiseConfig(model=model, rate=rate, target="both",
                       flip_from=1 if (one_way and dataset == "genai") else None)


def base_config(dataset: str = "genai", seed: int = BASE_SEED, **kw) -> CrankConfig:
    defaults = dict(
        dataset=dataset, seed=seed, cycles=30, train_batch=40, test_n=100,
        n_anchors=10, lr=0.4, clip=0.5,
        noise=NoiseConfig(model="boundary", rate=0.15, target="both"),
    )
    defaults.update(kw)
    return CrankConfig(**defaults)


def _seeds(n: int):
    return [BASE_SEED + 100 * i for i in range(n)]


def _final(series, key):
    return series[key][-1]


def _gate_rates(series):
    cells = [c for c in series["gate_cell"] if c in ("TA", "FA", "FR", "TR")]
    n = max(len(cells), 1)
    return {c: cells.count(c) / n for c in ("TA", "FA", "FR", "TR")}


def s1_dose_response(dataset: str = "genai", n_seeds: int = 12) -> dict:
    rates = [0.0, 0.05, 0.1, 0.2, 0.3]
    variants = [("uniform", False), ("boundary", False)]
    if dataset == "genai":
        variants += [("uniform", True), ("boundary", True)]
    cells = []
    for model, one_way in variants:
        for rate in rates:
            finals, divs, div_series = [], [], []
            for seed in _seeds(n_seeds):
                cfg = base_config(dataset, seed=seed,
                                  noise=suite_noise(dataset, model=model,
                                                    rate=rate, one_way=one_way))
                if rate == 0.0:
                    res = run_crank(cfg)
                    finals.append(_final(res["series"], "policy_f1_true"))
                    divs.append(0.0)
                    div_series.append([0.0] * (cfg.cycles + 1))
                    continue
                # noisy universe vs its clean twin over the SAME world (CRN)
                ds, labels = build_world(cfg)
                res = run_crank(cfg, ds=ds, labels=labels)
                res_clean = run_crank(cfg, ds=ds, labels=clean_labels(ds))
                div = divergence_series(res, res_clean)
                finals.append(_final(res["series"], "policy_f1_true"))
                divs.append(div["decision_disagreement"][-1])
                div_series.append(div["decision_disagreement"])
            cells.append({
                "model": model, "one_way": one_way, "rate": rate,
                "final_oracle_f1": mean_ci(finals),
                "divergence_from_clean_final": mean_ci(divs),
                "divergence_series_mean": np.nanmean(np.array(div_series), axis=0).tolist(),
            })
    return {"suite": "S1", "dataset": dataset, "n_seeds": n_seeds, "cells": cells}


def s2_mitigation(dataset: str = "genai", n_seeds: int = 12,
                  rate: float = 0.3) -> dict:
    arms = {
        "off": {},
        "deweight": {"weighting": "deweight"},
        "deweight_hard": {"weighting": "deweight_hard"},
        "upweight": {"weighting": "upweight"},
        # mitigation reviews BOTH pools: he flagged that re-adjudicating the
        # validation/test set is critical, since the gate reads it every cycle
        "readj_stack": {"readj": ReadjConfig(strategy="stack_rank", budget=5,
                                             include_test=True, test_budget=3)},
        "readj_pps": {"readj": ReadjConfig(strategy="pps", budget=5,
                                           include_test=True, test_budget=3)},
        "readj_random": {"readj": ReadjConfig(strategy="random", budget=5,
                                              include_test=True, test_budget=3)},
        "deweight+readj_stack": {"weighting": "deweight",
                                 "readj": ReadjConfig(strategy="stack_rank", budget=5,
                                                      include_test=True, test_budget=3)},
    }
    noise = suite_noise(dataset, model="boundary", rate=rate)
    cells = []
    for name, over in arms.items():
        f1s, aurocs, overturns, residuals, gate_fa = [], [], [], [], []
        for seed in _seeds(n_seeds):
            cfg = base_config(dataset, seed=seed, noise=noise, **over)
            res = run_crank(cfg)
            s = res["series"]
            f1s.append(_final(s, "policy_f1_true"))
            aurocs.append(_final(s, "detection_auroc"))
            residuals.append(_final(s, "residual_label_errors"))
            gate_fa.append(_gate_rates(s)["FA"])
            overturns.append(res["overturned_total"] / res["reviewed_total"]
                             if res["reviewed_total"] else float("nan"))
        cells.append({
            "arm": name,
            "final_oracle_f1": mean_ci(f1s),
            "detection_auroc": mean_ci(aurocs),
            "residual_label_errors": mean_ci(residuals),
            "gate_false_accept_rate": mean_ci(gate_fa),
            "emergent_overturn_rate": mean_ci(overturns),
        })
    # clean reference
    f1s = [_final(run_crank(base_config(dataset, seed=s,
                                        noise=NoiseConfig(model="none", rate=0.0))
                            )["series"], "policy_f1_true") for s in _seeds(n_seeds)]
    return {"suite": "S2", "dataset": dataset, "n_seeds": n_seeds, "rate": rate,
            "clean_reference_f1": mean_ci(f1s), "cells": cells}


def s3_parallel_universe(dataset: str = "genai", n_seeds: int = 10,
                         influence_seeds: int = 3) -> dict:
    out = {"suite": "S3", "dataset": dataset, "n_seeds": n_seeds, "pairs": [],
           "influence": []}
    for rate in (0.1, 0.3):
        dd_final, f1_gap_final, dd_series = [], [], []
        for seed in _seeds(n_seeds):
            cfg = base_config(dataset, seed=seed,
                              noise=suite_noise(dataset, rate=rate))
            _ra, _rb, div = run_pair(cfg, a={"weighting": "deweight"},
                                     b={"weighting": "upweight"})
            dd_final.append(div["decision_disagreement"][-1])
            f1_gap_final.append(div["f1_gap"][-1])
            dd_series.append(div["decision_disagreement"])
        out["pairs"].append({
            "rate": rate,
            "decision_disagreement_final": mean_ci(dd_final),
            "f1_gap_final": mean_ci(f1_gap_final),
            "divergence_series_mean": np.nanmean(np.array(dd_series), axis=0).tolist(),
        })
    for seed in _seeds(influence_seeds):
        cfg = base_config(dataset, seed=seed,
                          noise=suite_noise(dataset, rate=0.2))
        out["influence"].extend(point_influence(cfg))
    return out


def s4_test_corruption(dataset: str = "genai", n_seeds: int = 12) -> dict:
    cells = []
    for rate in (0.1, 0.2):
        for readj_test in (False, True):
            f1s, fa, fr = [], [], []
            for seed in _seeds(n_seeds):
                readj = (ReadjConfig(strategy="stack_rank", budget=0,
                                     include_test=True, test_budget=5)
                         if readj_test else ReadjConfig(strategy="off"))
                cfg = base_config(dataset, seed=seed, readj=readj,
                                  noise=NoiseConfig(model="boundary", rate=rate,
                                                    target="test"))
                res = run_crank(cfg)
                rates = _gate_rates(res["series"])
                f1s.append(_final(res["series"], "policy_f1_true"))
                fa.append(rates["FA"])
                fr.append(rates["FR"])
            cells.append({
                "rate": rate, "readj_test": readj_test,
                "final_oracle_f1": mean_ci(f1s),
                "gate_false_accept_rate": mean_ci(fa),
                "gate_false_reject_rate": mean_ci(fr),
            })
    return {"suite": "S4", "dataset": dataset, "n_seeds": n_seeds, "cells": cells}


def s5_sampling(dataset: str = "genai", n_seeds: int = 12) -> dict:
    cells = []
    for strategy in ("stack_rank", "pps", "random"):
        for rate in (0.0, 0.2):
            f1s, ginis, contams = [], [], []
            for seed in _seeds(n_seeds):
                cfg = base_config(dataset, seed=seed, anchor_strategy=strategy,
                                  noise=suite_noise(dataset, model="uniform",
                                                    rate=rate))
                res = run_crank(cfg)
                f1s.append(_final(res["series"], "policy_f1_true"))
                # concentration among items that were ever ELIGIBLE (appeared
                # misaligned in some batch) — including the structurally-zero
                # rest of the dev pool would wash every strategy to ~0.85
                ever_eligible = res["eligible_counts"][res["dev_idx"]] > 0
                ginis.append(gini(res["anchor_counts"][res["dev_idx"]][ever_eligible]))
                contam = np.array(res["series"]["anchor_contamination"], dtype=float)
                contams.append(float(np.nanmean(contam)) if not np.all(np.isnan(contam))
                               else float("nan"))
            cells.append({
                "strategy": strategy, "rate": rate,
                "final_oracle_f1": mean_ci(f1s),
                "anchor_gini": mean_ci(ginis),
                "anchor_contamination": mean_ci(contams),
            })
    return {"suite": "S5", "dataset": dataset, "n_seeds": n_seeds, "cells": cells}


SUITES = {"S1": s1_dose_response, "S2": s2_mitigation, "S3": s3_parallel_universe,
          "S4": s4_test_corruption, "S5": s5_sampling}


def run_suite(name: str, dataset: str = "genai", n_seeds: int = 12,
              out_dir: Path | str | None = None) -> dict:
    result = SUITES[name](dataset=dataset, n_seeds=n_seeds)
    if out_dir:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        path = out / f"{name.lower()}_{dataset}.json"
        # sanitize first: bare NaN tokens are invalid strict JSON and break
        # JSON.parse / jq consumers — nan/inf become null.
        path.write_text(json.dumps(_jsonable(result), indent=1, allow_nan=False))
    return result


def _jsonable(o):
    if isinstance(o, dict):
        return {k: _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return _jsonable(o.tolist())
    if isinstance(o, (bool, np.bool_)):
        return bool(o)
    if isinstance(o, (float, np.floating)):
        return float(o) if np.isfinite(o) else None
    if isinstance(o, (int, np.integer)):
        return int(o)
    return o
