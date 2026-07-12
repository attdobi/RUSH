"""The simulated crank, paired universes, and per-point influence probes.

One cycle mirrors production: sample a train batch -> panel labels it under
the current policy -> misalignment (system vote vs human label) updates each
human label's confidence posterior -> budgeted re-adjudication spends SME
attention -> misaligned items become anchors (consensus x confidence-weighted,
selected by stack-rank / PPS / random) -> a clipped parameter step proposes a
candidate -> the gate compares candidate vs incumbent SYSTEM decisions against
the (possibly noisy) human test labels and accepts/rejects.

Ground truth is used ONLY for measurement (oracle metrics, gate-vs-oracle
confusion, detection AUROC) and inside the SME review model — never in the
loop's own signal.

Common random numbers: every stochastic subsystem draws from its own stream
keyed (seed, stream, cycle), so two runs that differ only in an intervention
(weighting mode, noise on/off, one pinned point) share every draw and their
divergence is attributable to the intervention alone.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field

import numpy as np

from .confidence import ConfidenceConfig, HumanConfidence, anchor_weight
from .datasets import make_dataset, probe_grid, split_indices
from .judges import Panel, default_panel
from .metrics import macro_f1
from .noise import NoiseConfig, apply_noise
from .policy import (LogisticPolicy, PrototypePolicy, decision_disagreement,
                     fit_oracle, param_distance)
from .readjudication import ReadjConfig, adjudicate, select_queue

_STREAMS = {"batch": 1, "judge": 2, "select": 3, "readj": 4,
            "sme": 5, "init": 6, "testdraw": 7, "testjudge": 8}


def _stream(seed: int, name: str, k: int = 0) -> np.random.Generator:
    return np.random.default_rng([int(seed), _STREAMS[name], int(k)])


@dataclass
class PanelConfig:
    noncompliant: bool = False       # swap in a constant-output judge
    constant_class: int = 0
    exclude_noncompliant: bool = False  # compliance deweight in the system vote


@dataclass
class CrankConfig:
    dataset: str = "genai"           # genai | mnist
    seed: int = 13
    cycles: int = 30
    train_batch: int = 40
    test_n: int = 100
    resample_test: bool = False      # fresh test draw from the pool per cycle
    n_anchors: int = 10
    anchor_strategy: str = "stack_rank"  # stack_rank | pps | random
    lr: float = 0.4
    clip: float = 0.5
    gate: str = "on"                 # on | off (off accepts every clipped step)
    gate_eps: float = 0.0
    weighting: str = "off"           # off | deweight | deweight_hard | upweight
    upweight_amp: float = 1.0
    noise: NoiseConfig = field(default_factory=NoiseConfig)
    conf: ConfidenceConfig = field(default_factory=ConfidenceConfig)
    readj: ReadjConfig = field(default_factory=ReadjConfig)
    panel: PanelConfig = field(default_factory=PanelConfig)
    v0_rotate_deg: tuple = (30.0, 60.0)  # genai v0 misspecification range
    v0_jitter: float = 0.8               # mnist v0 prototype jitter (data std units)
    # Vary ONLY the starting policy while keeping the world, labels, and every
    # other stream fixed (initial-condition sensitivity / Lyapunov probes).
    # None -> the run seed keys the init stream as usual.
    v0_seed: int | None = None
    dataset_kw: dict = field(default_factory=dict)


def _make_v0(cfg: CrankConfig, ds, oracle):
    """A deliberately rough starting policy — roughly-right-but-misspecified,
    like prod's generic v0 guidelines: a seeded 30-60 degree distortion OF THE
    ORACLE boundary. Be honest about what that means: v0 is truth-anchored
    (the oracle was fit on all ground-truth labels), so absolute
    recovery-of-truth levels are partly baked in by the starting point.
    What the design guarantees is that v0 is LABEL-NOISE-FREE and
    mode-independent — identical across twin universes and mitigation arms —
    so divergence at k=0 is exactly zero and every RELATIVE comparison is
    attributable to the loop. Supported claims are the relative ones."""
    rng = _stream(cfg.seed if cfg.v0_seed is None else cfg.v0_seed, "init")
    if ds.n_classes == 2:
        pol = LogisticPolicy(oracle.w, oracle.b)
        lo, hi = cfg.v0_rotate_deg
        theta = np.deg2rad(rng.uniform(lo, hi)) * (1 if rng.random() < 0.5 else -1)
        rot = np.array([[np.cos(theta), -np.sin(theta)],
                        [np.sin(theta), np.cos(theta)]])
        pol.w = rot @ pol.w
        pol.b += rng.normal(0, 0.4) * np.linalg.norm(pol.w)
        return pol
    protos = oracle.protos + rng.normal(0, cfg.v0_jitter,
                                        size=oracle.protos.shape)
    return PrototypePolicy(protos)


def _noise_eligible(cfg: CrankConfig, ds, dev_idx, test_pool_idx):
    return {"train": dev_idx, "test": test_pool_idx,
            "both": np.arange(ds.n)}[cfg.noise.target]


def build_world(cfg: CrankConfig):
    """(ds, labels) for a config — build once, share across paired universes."""
    ds = make_dataset(cfg.dataset, seed=cfg.seed, **cfg.dataset_kw)
    dev_idx, test_pool_idx = split_indices(ds.n, cfg.seed)
    eligible = _noise_eligible(cfg, ds, dev_idx, test_pool_idx)
    labels = apply_noise(ds, cfg.noise, cfg.seed, eligible=eligible)
    return ds, labels


def _sample_batch(dev_idx, used: np.ndarray, size: int, rng) -> np.ndarray:
    """Without replacement across cycles; when the pool runs dry it resets and
    tops up with reuse (the crank's used_train_ids semantics)."""
    remaining = dev_idx[~used[dev_idx]]
    if len(remaining) >= size:
        pick = rng.choice(remaining, size=size, replace=False)
        used[pick] = True
        return pick
    first = remaining
    used[dev_idx] = False
    rest_pool = dev_idx[~np.isin(dev_idx, first)]
    rest = rng.choice(rest_pool, size=size - len(first), replace=False)
    pick = np.concatenate([first, rest]).astype(int)
    used[pick] = True
    return pick


def run_crank(cfg: CrankConfig, ds=None, labels=None, weighting: str | None = None,
              pinned: dict | None = None) -> dict:
    """One full run. `ds`/`labels` overrides let paired universes share the
    exact world and noise; `weighting` overrides cfg.weighting; `pinned` maps
    dataset index -> forced anchor weight (per-point influence probes)."""
    mode = weighting if weighting is not None else cfg.weighting
    if ds is None:
        ds, labels = build_world(cfg)
    if labels is None:
        dev_i, test_p = split_indices(ds.n, cfg.seed)
        labels = apply_noise(ds, cfg.noise, cfg.seed,
                             eligible=_noise_eligible(cfg, ds, dev_i, test_p))
    y_human = labels[0].copy()   # readjudication mutates the copy
    flipped = labels[1].copy()

    dev_idx, test_pool_idx = split_indices(ds.n, cfg.seed)
    oracle = fit_oracle(ds)      # measurement + noise placement only
    panel = default_panel(noncompliant=cfg.panel.noncompliant,
                          constant_class=cfg.panel.constant_class)
    exclude = (frozenset({panel.specs[-1].name})
               if (cfg.panel.noncompliant and cfg.panel.exclude_noncompliant)
               else frozenset())
    conf = HumanConfidence(ds.n, cfg.conf)
    policy = _make_v0(cfg, ds, oracle)
    used = np.zeros(ds.n, dtype=bool)
    anchor_counts = np.zeros(ds.n, dtype=int)
    eligible_counts = np.zeros(ds.n, dtype=int)  # misaligned-pool appearances
    test_idx = test_pool_idx[:cfg.test_n]
    nc = ds.n_classes

    series: dict = {k: [] for k in (
        "k", "policy_f1_true", "sys_f1_true_test", "sys_f1_human_test",
        "accepted", "gate_cell", "n_anchors", "anchor_contamination",
        "anchor_margin_mean", "detection_auroc", "residual_label_errors",
        "reviewed_cum", "overturned_cum", "param_dist_from_v0", "w_min",
        "w_mean_flipped", "w_mean_clean")}
    snapshots = [policy.clone()]
    reviewed_total = overturned_total = 0
    v0_vec = policy.param_vector()

    def _eval_test(pol, k):
        # instantiate the same stream twice per cycle -> identical judge noise
        # for incumbent and candidate: a paired evaluation.
        _, sys_t, s_t = panel.vote(pol, ds.X[test_idx],
                                   _stream(cfg.seed, "testjudge", k), exclude)
        return sys_t, s_t

    def _record(k, accepted, gate_cell, anch_idx, sys_t_final):
        series["k"].append(k)
        series["policy_f1_true"].append(macro_f1(ds.y, policy.predict(ds.X), nc))
        series["sys_f1_true_test"].append(macro_f1(ds.y[test_idx], sys_t_final, nc))
        series["sys_f1_human_test"].append(macro_f1(y_human[test_idx], sys_t_final, nc))
        series["accepted"].append(accepted)
        series["gate_cell"].append(gate_cell)
        series["n_anchors"].append(len(anch_idx))
        mislab = y_human != ds.y
        if len(anch_idx):
            series["anchor_contamination"].append(float(mislab[anch_idx].mean()))
            series["anchor_margin_mean"].append(float(
                oracle.class_margin(ds.X[anch_idx], ds.y[anch_idx]).mean()))
        else:
            series["anchor_contamination"].append(float("nan"))
            series["anchor_margin_mean"].append(float("nan"))
        # peak suspicion vs the ORIGINAL flip mask on every seen item: a
        # stable detector score — re-adjudication arms must not censor the
        # very items their detector caught.
        seen = conf.n_seen > 0
        from .metrics import auroc
        series["detection_auroc"].append(
            auroc(conf.peak_suspicion[seen], flipped[seen]) if seen.any() else float("nan"))
        series["residual_label_errors"].append(int(mislab.sum()))
        series["reviewed_cum"].append(reviewed_total)
        series["overturned_cum"].append(overturned_total)
        series["param_dist_from_v0"].append(
            float(np.linalg.norm(policy.param_vector() - v0_vec) / np.sqrt(len(v0_vec))))
        w = conf.w
        series["w_min"].append(float(w[dev_idx].min()))
        series["w_mean_flipped"].append(float(w[flipped].mean()) if flipped.any() else float("nan"))
        series["w_mean_clean"].append(float(w[~flipped].mean()))

    # k = 0 baseline
    sys_t0, _ = _eval_test(policy, 0)
    _record(0, None, "", np.empty(0, dtype=int), sys_t0)

    for k in range(1, cfg.cycles + 1):
        if cfg.resample_test:
            test_idx = _stream(cfg.seed, "testdraw", k).choice(
                test_pool_idx, size=min(cfg.test_n, len(test_pool_idx)), replace=False)

        batch = _sample_batch(dev_idx, used, cfg.train_batch,
                              _stream(cfg.seed, "batch", k))
        _, sys_b, s_b = panel.vote(policy, ds.X[batch],
                                   _stream(cfg.seed, "judge", k), exclude)
        conf.observe(batch, sys_b != y_human[batch], s_b)

        # test partition gets panel evidence every cycle too (it is labeled
        # every eval in prod, so it accumulates disagreement signal)
        sys_t_inc, s_t_inc = _eval_test(policy, k)
        conf.observe(test_idx, sys_t_inc != y_human[test_idx], s_t_inc)

        # budgeted SME re-adjudication (before anchor selection)
        if cfg.readj.strategy != "off":
            rng_r = _stream(cfg.seed, "readj", k)
            rng_s = _stream(cfg.seed, "sme", k)
            for pool, budget in ((dev_idx, cfg.readj.budget),
                                 (test_pool_idx, cfg.readj.test_budget
                                  if cfg.readj.include_test else 0)):
                unresolved = pool[~conf.resolved[pool]]
                picked = select_queue(conf.queue_score, unresolved, budget,
                                      cfg.readj.strategy, rng_r)
                n_rev, n_over = adjudicate(picked, y_human, ds.y,
                                           cfg.readj.q_sme, rng_s)
                conf.mark_adjudicated(picked)
                reviewed_total += n_rev
                overturned_total += n_over

        # anchors: misaligned batch items, consensus x confidence-weighted
        disagree_b = sys_b != y_human[batch]
        elig = np.flatnonzero(disagree_b)
        eligible_counts[batch[elig]] += 1
        anch_idx = np.empty(0, dtype=int)
        weights_sel = np.empty(0)
        if len(elig):
            wt = anchor_weight(mode, conf.w[batch[elig]], amp=cfg.upweight_amp,
                               pinned=pinned, idx=batch[elig])
            score = s_b[elig] * wt
            live = score > 0
            elig, score = elig[live], score[live]
            if len(elig):
                m = min(cfg.n_anchors, len(elig))
                rng_sel = _stream(cfg.seed, "select", k)
                if cfg.anchor_strategy == "stack_rank":
                    order = np.argsort(-score, kind="stable")[:m]
                elif cfg.anchor_strategy == "pps":
                    order = rng_sel.choice(len(elig), size=m, replace=False,
                                           p=score / score.sum())
                elif cfg.anchor_strategy == "random":
                    order = rng_sel.choice(len(elig), size=m, replace=False)
                else:
                    raise ValueError(f"unknown anchor strategy {cfg.anchor_strategy!r}")
                anch_idx = batch[elig[order]]
                weights_sel = score[order]

        accepted = False
        gate_cell = "NOOP"
        sys_t_final = sys_t_inc
        if len(anch_idx):
            candidate = policy.clone()
            candidate.update(ds.X[anch_idx], y_human[anch_idx], weights_sel,
                             cfg.lr, cfg.clip)
            sys_t_cand, _ = _eval_test(candidate, k)
            f1_h_inc = macro_f1(y_human[test_idx], sys_t_inc, nc)
            f1_h_cand = macro_f1(y_human[test_idx], sys_t_cand, nc)
            f1_t_inc = macro_f1(ds.y[test_idx], sys_t_inc, nc)
            f1_t_cand = macro_f1(ds.y[test_idx], sys_t_cand, nc)
            accepted = cfg.gate == "off" or (f1_h_cand - f1_h_inc) >= cfg.gate_eps
            oracle_ok = (f1_t_cand - f1_t_inc) >= cfg.gate_eps
            gate_cell = {(True, True): "TA", (True, False): "FA",
                         (False, True): "FR", (False, False): "TR"}[(accepted, oracle_ok)]
            if accepted:
                policy = candidate
                anchor_counts[anch_idx] += 1
                sys_t_final = sys_t_cand
        _record(k, accepted, gate_cell, anch_idx, sys_t_final)
        snapshots.append(policy.clone())

    return {
        "cfg": cfg, "mode": mode, "series": series, "snapshots": snapshots,
        "policy": policy, "oracle": oracle, "ds": ds,
        "y_human_final": y_human, "flipped": flipped, "w_final": conf.w,
        "suspicion_final": conf.suspicion(), "resolved": conf.resolved.copy(),
        "n_seen": conf.n_seen.copy(), "anchor_counts": anchor_counts,
        "eligible_counts": eligible_counts,
        "dev_idx": dev_idx, "test_pool_idx": test_pool_idx,
        "reviewed_total": reviewed_total, "overturned_total": overturned_total,
    }


def divergence_series(res_a: dict, res_b: dict, probe: np.ndarray | None = None) -> dict:
    """Per-cycle divergence between two runs of the same world: parameter
    distance, decision disagreement on the probe grid, and oracle-F1 gap."""
    ds = res_a["ds"]
    probe = probe if probe is not None else probe_grid(ds)
    pd, dd = [], []
    for pa, pb in zip(res_a["snapshots"], res_b["snapshots"]):
        pd.append(param_distance(pa, pb))
        dd.append(decision_disagreement(pa, pb, probe))
    fa = np.asarray(res_a["series"]["policy_f1_true"])
    fb = np.asarray(res_b["series"]["policy_f1_true"])
    return {"param_dist": pd, "decision_disagreement": dd,
            "f1_gap": np.abs(fa - fb).tolist()}


def run_pair(cfg: CrankConfig, a: dict | None = None, b: dict | None = None):
    """Two universes over the SAME world and noise (common random numbers),
    differing only in the given run_crank overrides. Examples:
      noisy vs clean:      a={}, b={'labels': clean}   (see clean_labels())
      deweight vs upweight: a={'weighting':'deweight'}, b={'weighting':'upweight'}
    Returns (res_a, res_b, divergence dict)."""
    ds, labels = build_world(cfg)
    base = {"ds": ds, "labels": labels}
    res_a = run_crank(cfg, **{**base, **(a or {})})
    res_b = run_crank(cfg, **{**base, **(b or {})})
    return res_a, res_b, divergence_series(res_a, res_b)


def clean_labels(ds) -> tuple:
    """A labels override where every human label is ground truth."""
    return ds.y.copy(), np.zeros(ds.n, dtype=bool)


def point_influence(cfg: CrankConfig, m_suspects: int = 12, m_controls: int = 12,
                    horizon: int = 12) -> list[dict]:
    """His parallel-universe probe, one point at a time: for each candidate
    item run two short twin universes — the point's anchor weight pinned to 0
    (de-weight / assume human wrong) vs pinned to 2 (up-weight / assume human
    right) — and measure how far the two policies diverge. Points whose twin
    universes stay close 'never mattered much'; large divergence marks a label
    whose correctness the whole trajectory hinges on.

    Candidates: m random truly-flipped items and m random clean items, both
    drawn uniformly from the dev items the panel actually saw. (Random within
    group, NOT top-by-suspicion: selecting suspects by suspicion would
    confound the flipped-vs-clean influence comparison with the detector's
    own selection. The flip mask is oracle information — fine here, this is
    a measurement probe, not a mitigation.)"""
    base = run_crank(dataclasses.replace(cfg, readj=ReadjConfig(strategy="off")))
    ds = base["ds"]
    labels = (base["y_human_final"], base["flipped"])  # readj off: unchanged
    seen_dev = base["dev_idx"][base["n_seen"][base["dev_idx"]] > 0]
    susp = base["suspicion_final"]
    rng = _stream(cfg.seed, "select", 999_999)
    fl = seen_dev[base["flipped"][seen_dev]]
    cl = seen_dev[~base["flipped"][seen_dev]]
    suspects = (rng.choice(fl, size=min(m_suspects, len(fl)), replace=False)
                if len(fl) else np.empty(0, dtype=int))
    controls = (rng.choice(cl, size=min(m_controls, len(cl)), replace=False)
                if len(cl) else np.empty(0, dtype=int))
    short = dataclasses.replace(cfg, cycles=horizon, readj=ReadjConfig(strategy="off"))
    probe = probe_grid(ds)
    out = []
    for i in np.concatenate([suspects, controls]):
        i = int(i)
        res_lo = run_crank(short, ds=ds, labels=labels, weighting="off",
                           pinned={i: 0.0})
        res_hi = run_crank(short, ds=ds, labels=labels, weighting="off",
                           pinned={i: 2.0})
        div = divergence_series(res_lo, res_hi, probe)
        out.append({
            "idx": i,
            "influence_dd": float(np.mean(div["decision_disagreement"][-3:])),
            "influence_f1_gap": float(np.mean(div["f1_gap"][-3:])),
            "suspicion": float(susp[i]),
            "is_flipped": bool(base["flipped"][i]),
            "times_anchored": int(base["anchor_counts"][i]),
        })
    return out
