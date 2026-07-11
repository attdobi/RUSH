/* Label-noise crank simulator — client-side port of labelsim v0.1 (GenAI binary arm).
 *
 * Mirrors the SEMANTICS of sim/label-noise/labelsim/{datasets,policy,judges,
 * noise,confidence,readjudication,engine}.py — not numpy bit-for-bit. RNG is
 * mulberry32 seeded per (seed, stream, cycle), normals via Box-Muller, so two
 * universes keyed off the same seed share every draw (common random numbers).
 *
 * Twin-universe coupling: runSimulation() builds ONE world (points + injected
 * label noise), pre-draws the per-cycle stochastic tensors ONCE (batch indices,
 * batch judge noise, test judge noise) and hands the same tensors to both the
 * NOISY universe (corrupted human labels) and its CLEAN twin (labels = truth).
 * Divergence between the two policies is attributable to label noise alone.
 *
 * The file is split in two: a pure, DOM-free simulation core exported on
 * globalThis.LabelSim (smoke-testable in jsc/node with no d3), and a browser
 * UI layer that only runs when `document` and `d3` exist.
 */

(function (root) {
  'use strict';

  /* ================= RNG: mulberry32 + Box-Muller, stream-keyed ============ */

  // Stream ids mirror engine._STREAMS (plus ids for world construction).
  var ST = {
    batch: 1, judge: 2, select: 3, readj: 4, sme: 5, init: 6,
    testdraw: 7, testjudge: 8, data: 10, split: 11, noise: 12, oracle: 13
  };

  function hashKey(seed, stream, k) {
    var h = (seed | 0) ^ 0x9E3779B9;
    h = Math.imul(h ^ ((stream | 0) + 0x85EBCA6B), 0xC2B2AE35);
    h ^= h >>> 13;
    h = Math.imul(h ^ Math.imul((k | 0) + 1, 0x27D4EB2F), 0x165667B1);
    h ^= h >>> 16;
    return h >>> 0;
  }

  function makeStream(seed, stream, k) {
    var a = hashKey(seed, stream, k || 0);
    var spare = null;
    function random() {
      a = (a + 0x6D2B79F5) | 0;
      var t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    }
    return {
      random: random,
      normal: function () {           // Box-Muller with cached spare
        if (spare !== null) { var v0 = spare; spare = null; return v0; }
        var u = 0;
        do { u = random(); } while (u <= 1e-12);
        var v = random();
        var r = Math.sqrt(-2 * Math.log(u));
        spare = r * Math.sin(2 * Math.PI * v);
        return r * Math.cos(2 * Math.PI * v);
      },
      int: function (m) {             // uniform integer in [0, m)
        return m > 0 ? Math.min(m - 1, Math.floor(random() * m)) : 0;
      }
    };
  }

  function permutation(rng, n) {
    var p = new Int32Array(n);
    for (var i = 0; i < n; i++) p[i] = i;
    for (i = n - 1; i > 0; i--) {
      var j = rng.int(i + 1);
      var t = p[i]; p[i] = p[j]; p[j] = t;
    }
    return p;
  }

  function sampleWithoutReplacement(rng, pool, k) {
    var arr = Array.prototype.slice.call(pool);
    var out = new Array(k);
    for (var i = 0; i < k; i++) {
      var j = i + rng.int(arr.length - i);
      var t = arr[i]; arr[i] = arr[j]; arr[j] = t;
      out[i] = arr[i];
    }
    return out;
  }

  // Probability-proportional-to-size, without replacement (numpy choice p=..., replace=False).
  function weightedSampleWithoutReplacement(rng, pool, weights, k) {
    var w = Array.prototype.slice.call(weights);
    var items = Array.prototype.slice.call(pool);
    var total = 0;
    for (var i = 0; i < w.length; i++) total += w[i];
    var out = [];
    for (var c = 0; c < k; c++) {
      var r = rng.random() * total;
      var idx = -1;
      for (i = 0; i < items.length; i++) {
        if (w[i] <= 0) continue;
        r -= w[i];
        if (r <= 0) { idx = i; break; }
      }
      if (idx < 0) {                  // fp fallback: last live item
        for (i = items.length - 1; i >= 0; i--) if (w[i] > 0) { idx = i; break; }
      }
      out.push(items[idx]);
      total -= w[idx];
      w[idx] = 0;
    }
    return out;
  }

  function drawNormals(rng, count) {
    var out = new Float64Array(count);
    for (var i = 0; i < count; i++) out[i] = rng.normal();
    return out;
  }

  function clamp(v, lo, hi) { return v < lo ? lo : (v > hi ? hi : v); }

  /* ================= dataset: make_genai + split_indices =================== */

  // Binary world: 0 = not_gen_ai, 1 = gen_ai. Easy mass far from the boundary
  // plus a contested ridge at x ~ 0 where the classes interleave.
  function makeGenai(n, seed, hardFrac) {
    if (hardFrac === undefined) hardFrac = 0.35;
    var rng = makeStream(seed, ST.data, 0);
    var half0 = Math.floor(n / 2), half1 = n - half0;
    var X = new Float64Array(n * 2);
    var y = new Int8Array(n);
    var ptr = 0;
    var groups = [
      [0, half0, -2.2, 0.0, -0.30, 0.0],
      [1, half1, +2.2, 0.3, +0.30, 0.1]
    ];
    for (var g = 0; g < 2; g++) {
      var cls = groups[g][0], nCls = groups[g][1];
      var ex = groups[g][2], ey = groups[g][3], hx = groups[g][4], hy = groups[g][5];
      var nHard = Math.round(nCls * hardFrac);
      var nEasy = nCls - nHard;
      for (var i = 0; i < nEasy; i++) {
        X[ptr * 2] = ex + 0.9 * rng.normal();
        X[ptr * 2 + 1] = ey + 0.9 * rng.normal();
        y[ptr++] = cls;
      }
      for (i = 0; i < nHard; i++) {   // tight in x, tall in y
        X[ptr * 2] = hx + 0.55 * rng.normal();
        X[ptr * 2 + 1] = hy + 1.1 * rng.normal();
        y[ptr++] = cls;
      }
    }
    var perm = permutation(rng, n);
    var X2 = new Float64Array(n * 2), y2 = new Int8Array(n);
    for (i = 0; i < n; i++) {
      X2[i * 2] = X[perm[i] * 2];
      X2[i * 2 + 1] = X[perm[i] * 2 + 1];
      y2[i] = y[perm[i]];
    }
    return { n: n, X: X2, y: y2 };
  }

  // Seeded (devIdx, testPoolIdx); test pool = 35% of points.
  function splitIndices(n, seed, testPoolFrac) {
    if (testPoolFrac === undefined) testPoolFrac = 0.35;
    var perm = permutation(makeStream(seed, ST.split, 0), n);
    var nTest = Math.round(n * testPoolFrac);
    return {
      testPoolIdx: Array.prototype.slice.call(perm, 0, nTest),
      devIdx: Array.prototype.slice.call(perm, nTest)
    };
  }

  function probeGrid(ds, m) {
    var b = bbox(ds);
    var pts = new Float64Array(m * m * 2);
    var p = 0;
    for (var gy = 0; gy < m; gy++) {
      var yv = b.y0 + (b.y1 - b.y0) * gy / (m - 1);
      for (var gx = 0; gx < m; gx++) {
        pts[p * 2] = b.x0 + (b.x1 - b.x0) * gx / (m - 1);
        pts[p * 2 + 1] = yv;
        p++;
      }
    }
    return pts;
  }

  function bbox(ds) {  // padded data bbox (10% pad, matches probe_grid)
    var x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
    for (var i = 0; i < ds.n; i++) {
      var xv = ds.X[i * 2], yv = ds.X[i * 2 + 1];
      if (xv < x0) x0 = xv; if (xv > x1) x1 = xv;
      if (yv < y0) y0 = yv; if (yv > y1) y1 = yv;
    }
    var px = 0.1 * (x1 - x0), py = 0.1 * (y1 - y0);
    return { x0: x0 - px, x1: x1 + px, y0: y0 - py, y1: y1 + py };
  }

  /* ================= policy: LogisticPolicy ================================ */

  function clonePol(p) { return { w: [p.w[0], p.w[1]], b: p.b }; }

  function polPredictPoint(pol, x0, x1) {
    return (pol.w[0] * x0 + pol.w[1] * x1 + pol.b) > 0 ? 1 : 0;
  }

  function polPredictAll(pol, ds) {
    var out = new Int8Array(ds.n);
    for (var i = 0; i < ds.n; i++) out[i] = polPredictPoint(pol, ds.X[i * 2], ds.X[i * 2 + 1]);
    return out;
  }

  // Signed distance to the boundary, positive on the side of label y.
  function classMargin(pol, ds, i) {
    var nrm = Math.hypot(pol.w[0], pol.w[1]) + 1e-9;
    var z = (pol.w[0] * ds.X[i * 2] + pol.w[1] * ds.X[i * 2 + 1] + pol.b) / nrm;
    return ds.y[i] === 1 ? z : -z;
  }

  // One weighted logistic step toward labels, step-norm clipped
  // (the "1-5 discrete edits" analogue).
  function polUpdate(pol, ds, idxs, yTarget, sampleWeight, lr, clip) {
    var total = 0;
    for (var i = 0; i < sampleWeight.length; i++) total += sampleWeight[i];
    if (total <= 0) return;
    var g0 = 0, g1 = 0, gb = 0;
    for (i = 0; i < idxs.length; i++) {
      var di = idxs[i];
      var x0 = ds.X[di * 2], x1 = ds.X[di * 2 + 1];
      var z = clamp(pol.w[0] * x0 + pol.w[1] * x1 + pol.b, -30, 30);
      var p = 1 / (1 + Math.exp(-z));
      var err = (p - yTarget[di]) * (sampleWeight[i] / total);
      g0 += err * x0; g1 += err * x1; gb += err;
    }
    var s0 = -lr * g0, s1 = -lr * g1, sb = -lr * gb;
    var norm = Math.sqrt(s0 * s0 + s1 * s1 + sb * sb);
    if (norm > clip) { var f = clip / norm; s0 *= f; s1 *= f; sb *= f; }
    pol.w[0] += s0; pol.w[1] += s1; pol.b += sb;
  }

  // The true boundary: plain full-batch logistic regression on ground truth
  // (300 steps, lr 0.5, l2 1e-3 — LogisticPolicy.fit). Measurement + noise
  // placement only; the loop never sees it.
  function fitOracle(ds, seed) {
    var rng = makeStream(seed, ST.oracle, 0);
    var w0 = 0.01 * rng.normal(), w1 = 0.01 * rng.normal(), b = 0;
    var n = ds.n, lr = 0.5, l2 = 1e-3;
    for (var s = 0; s < 300; s++) {
      var g0 = 0, g1 = 0, gb = 0;
      for (var i = 0; i < n; i++) {
        var x0 = ds.X[i * 2], x1 = ds.X[i * 2 + 1];
        var z = clamp(w0 * x0 + w1 * x1 + b, -30, 30);
        var err = 1 / (1 + Math.exp(-z)) - ds.y[i];
        g0 += err * x0; g1 += err * x1; gb += err;
      }
      w0 -= lr * (g0 / n + l2 * w0);
      w1 -= lr * (g1 / n + l2 * w1);
      b -= lr * (gb / n);
    }
    return { w: [w0, w1], b: b };
  }

  // v0: a label-free seeded distortion of the oracle — rotate w by 30-60 deg
  // (random sign) and jitter b. Identical across twin universes.
  function makeV0(seed, oracle) {
    var rng = makeStream(seed, ST.init, 0);
    var deg = 30 + rng.random() * 30;
    var sign = rng.random() < 0.5 ? 1 : -1;
    var th = deg * Math.PI / 180 * sign;
    var c = Math.cos(th), s = Math.sin(th);
    var w0 = c * oracle.w[0] - s * oracle.w[1];
    var w1 = s * oracle.w[0] + c * oracle.w[1];
    var b = oracle.b + rng.normal() * 0.4 * Math.hypot(w0, w1);
    return { w: [w0, w1], b: b };
  }

  function decisionDisagreement(polA, polB, probe) {
    var m = probe.length / 2, dis = 0;
    for (var i = 0; i < m; i++) {
      if (polPredictPoint(polA, probe[i * 2], probe[i * 2 + 1]) !==
          polPredictPoint(polB, probe[i * 2], probe[i * 2 + 1])) dis++;
    }
    return dis / m;
  }

  /* ================= judges: 5-sigma panel, majority vote ================== */

  var SIGMAS = [0.15, 0.25, 0.35, 0.50, 0.70];

  // noise: pre-drawn Float64Array of length K * n * 2, laid out [judge][item][dim]
  // (one tensor per vote call, judges.Panel.vote CRN discipline).
  function panelVote(pol, ds, idxList, noise) {
    var K = SIGMAS.length, n = idxList.length;
    var sys = new Int8Array(n), cons = new Float64Array(n);
    for (var i = 0; i < n; i++) {
      var di = idxList[i];
      var x0 = ds.X[di * 2], x1 = ds.X[di * 2 + 1];
      var ones = 0;
      for (var j = 0; j < K; j++) {
        var base = (j * n + i) * 2;
        if (polPredictPoint(pol, x0 + SIGMAS[j] * noise[base],
                            x1 + SIGMAS[j] * noise[base + 1]) === 1) ones++;
      }
      var zeros = K - ones;
      sys[i] = ones > zeros ? 1 : 0;          // tie -> smallest class id
      cons[i] = Math.max(ones, zeros) / K;    // top-vote share
    }
    return { sys: sys, cons: cons };
  }

  /* ================= noise: uniform / boundary, optional one-way =========== */

  // Returns {yHuman, flipped}. Binary world: the flip target is always the
  // other class (runner-up == 1 - y for a logistic policy).
  function applyNoise(ds, cfg, seed, oracle) {
    var yHuman = Int8Array.from(ds.y);
    var flipped = new Uint8Array(ds.n);
    if (cfg.model === 'none' || cfg.rate <= 0) return { yHuman: yHuman, flipped: flipped };
    var rng = makeStream(seed, ST.noise, 0);
    var eligible = [];
    for (var i = 0; i < ds.n; i++) {
      if (cfg.flipFrom === null || cfg.flipFrom === undefined || ds.y[i] === cfg.flipFrom) {
        eligible.push(i);
      }
    }
    if (!eligible.length) return { yHuman: yHuman, flipped: flipped };
    // rate = fraction of ALL labels (comparable across one-way and two-sided
    // at the same rate), capped at the flip-from pool — mirrors labelsim.
    var nFlip = Math.round(cfg.rate * ds.n);
    if (nFlip <= 0) return { yHuman: yHuman, flipped: flipped };
    nFlip = Math.min(nFlip, eligible.length);
    var pick;
    if (cfg.model === 'uniform') {
      pick = sampleWithoutReplacement(rng, eligible, nFlip);
    } else {                                  // boundary: p ~ exp(-margin/tau)
      var weights = new Array(eligible.length);
      var live = 0;
      for (i = 0; i < eligible.length; i++) {
        var m = classMargin(oracle, ds, eligible[i]);
        weights[i] = Math.exp(-Math.max(m, 0) / cfg.tau);
        if (weights[i] > 0) live++;
      }
      nFlip = Math.min(nFlip, live);
      pick = weightedSampleWithoutReplacement(rng, eligible, weights, nFlip);
    }
    for (i = 0; i < pick.length; i++) {
      yHuman[pick[i]] = 1 - ds.y[pick[i]];
      flipped[pick[i]] = 1;
    }
    return { yHuman: yHuman, flipped: flipped };
  }

  /* ================= confidence: Bayesian per-label posterior ============== */

  function logit(p) {
    p = Math.min(Math.max(p, 1e-6), 1 - 1e-6);
    return Math.log(p / (1 - p));
  }

  // HumanConfidence: prior c0 at N=1; each panel observation multiplies the
  // odds by (likelihood ratio)^consensus, with assumed p_catch/p_false.
  function makeConfidence(n, c0) {
    var P_CATCH = 0.75, P_FALSE = 0.25, ADJ_CONF = 0.99, LMAX = 12;
    var llrDis = Math.log(P_FALSE / P_CATCH);           // disagreement: evidence human wrong
    var llrAgr = Math.log((1 - P_FALSE) / (1 - P_CATCH));
    var logOdds = new Float64Array(n);
    logOdds.fill(logit(c0));
    var nSeen = new Int32Array(n);
    var queueScore = new Float64Array(n);               // cumulative disagreement mass
    var resolved = new Uint8Array(n);
    return {
      resolved: resolved,
      queueScore: queueScore,
      nSeen: nSeen,
      w: function (i) { return 1 / (1 + Math.exp(-logOdds[i])); },
      wSnapshot: function () {
        var out = new Float32Array(n);
        for (var i = 0; i < n; i++) out[i] = 1 / (1 + Math.exp(-logOdds[i]));
        return out;
      },
      observe: function (idxList, disagree, cons) {
        for (var j = 0; j < idxList.length; j++) {
          var i = idxList[j];
          nSeen[i]++;
          if (resolved[i]) continue;                    // adjudicated labels are settled
          var step = (disagree[j] ? llrDis : llrAgr) * cons[j];
          logOdds[i] = clamp(logOdds[i] + step, -LMAX, LMAX);
          if (disagree[j]) queueScore[i] += cons[j];
        }
      },
      markAdjudicated: function (idxList) {
        for (var j = 0; j < idxList.length; j++) {
          var i = idxList[j];
          logOdds[i] = logit(ADJ_CONF);
          resolved[i] = 1;
          queueScore[i] = 0;
        }
      }
    };
  }

  function anchorWeight(mode, w) {
    if (mode === 'off') return 1;
    if (mode === 'deweight') return w;
    if (mode === 'deweight_hard') return w >= 0.5 ? 1 : 0;
    if (mode === 'upweight') return 1 + (1 - w);        // amp = 1
    throw new Error('unknown weighting mode ' + mode);
  }

  /* ================= re-adjudication: budgeted SME queue ==================== */

  function selectQueue(queueScore, eligible, budget, strategy, rng) {
    if (strategy === 'off' || budget <= 0) return [];
    var pool = [], scores = [];
    for (var i = 0; i < eligible.length; i++) {
      if (queueScore[eligible[i]] > 0) {                // needs nonzero mass
        pool.push(eligible[i]);
        scores.push(queueScore[eligible[i]]);
      }
    }
    if (!pool.length) return [];
    var k = Math.min(budget, pool.length);
    if (strategy === 'stack_rank') {
      var order = pool.map(function (_, j) { return j; })
        .sort(function (a, b) { return scores[b] - scores[a] || a - b; });
      return order.slice(0, k).map(function (j) { return pool[j]; });
    }
    if (strategy === 'pps') return weightedSampleWithoutReplacement(rng, pool, scores, k);
    if (strategy === 'random') return sampleWithoutReplacement(rng, pool, k);
    throw new Error('unknown re-adjudication strategy ' + strategy);
  }

  // SME at full attention: with prob q_sme the review returns truth,
  // otherwise it keeps the current label. Returns [nReviewed, nOverturned].
  function adjudicate(picked, yHuman, yTrue, qSme, rng) {
    var over = 0;
    for (var j = 0; j < picked.length; j++) {
      var i = picked[j];
      var hit = rng.random() < qSme;
      var nl = hit ? yTrue[i] : yHuman[i];
      if (nl !== yHuman[i]) over++;
      yHuman[i] = nl;
    }
    return [picked.length, over];
  }

  /* ================= metrics =============================================== */

  function macroF1(yTrue, yPred) {                      // binary, zero_division=0
    var sum = 0;
    for (var c = 0; c < 2; c++) {
      var tp = 0, fp = 0, fn = 0;
      for (var i = 0; i < yTrue.length; i++) {
        var t = yTrue[i] === c, p = yPred[i] === c;
        if (p && t) tp++;
        else if (p && !t) fp++;
        else if (!p && t) fn++;
      }
      var denom = 2 * tp + fp + fn;
      sum += denom > 0 ? 2 * tp / denom : 0;
    }
    return sum / 2;
  }

  /* ================= engine: batch sampling + one universe ================== */

  // Without replacement across cycles; when the pool runs dry it resets and
  // tops up with reuse (engine._sample_batch).
  function sampleBatch(devIdx, used, size, rng) {
    var remaining = [];
    for (var i = 0; i < devIdx.length; i++) if (!used[devIdx[i]]) remaining.push(devIdx[i]);
    var pick;
    if (remaining.length >= size) {
      pick = sampleWithoutReplacement(rng, remaining, size);
    } else {
      var first = remaining;
      for (i = 0; i < devIdx.length; i++) used[devIdx[i]] = 0;
      var inFirst = {};
      for (i = 0; i < first.length; i++) inFirst[first[i]] = 1;
      var restPool = [];
      for (i = 0; i < devIdx.length; i++) if (!inFirst[devIdx[i]]) restPool.push(devIdx[i]);
      pick = first.concat(sampleWithoutReplacement(rng, restPool, size - first.length));
    }
    for (i = 0; i < pick.length; i++) used[pick[i]] = 1;
    return pick;
  }

  // One universe over a shared world. `labels` = {yHuman, flipped}; `shared`
  // carries the pre-drawn per-cycle tensors both twins consume identically.
  function runUniverse(cfg, world, labels, shared) {
    var ds = world.ds, oracle = world.oracle;
    var devIdx = world.devIdx, testPoolIdx = world.testPoolIdx, testIdx = world.testIdx;
    var n = ds.n;
    var yHuman = Int8Array.from(labels.yHuman);         // readjudication mutates the copy
    var flipped = labels.flipped;
    var conf = makeConfidence(n, cfg.c0);
    var policy = makeV0(cfg.seed, oracle);
    var snapshots = [clonePol(policy)];
    var rec = {
      f1True: [], mislabeled: [], resolved: [], wSnap: [], anchors: [],
      accepted: [], gateCell: [], residualErrors: [], reviewedCum: [],
      overturnedCum: [], acceptedCum: [], faCum: [], wFlipped: [], wClean: []
    };
    var reviewedTotal = 0, overturnedTotal = 0, acceptedTotal = 0, faTotal = 0;
    var yTrueTest = new Int8Array(testIdx.length);
    for (var i = 0; i < testIdx.length; i++) yTrueTest[i] = ds.y[testIdx[i]];

    function record(k, accepted, gateCell, anchors) {
      rec.f1True.push(macroF1(ds.y, polPredictAll(policy, ds)));
      var mis = new Uint8Array(n), errs = 0;
      for (var i = 0; i < n; i++) if (yHuman[i] !== ds.y[i]) { mis[i] = 1; errs++; }
      rec.mislabeled.push(mis);
      rec.resolved.push(Uint8Array.from(conf.resolved));
      rec.wSnap.push(conf.wSnapshot());
      rec.anchors.push(anchors);
      rec.accepted.push(accepted);
      rec.gateCell.push(gateCell);
      rec.residualErrors.push(errs);
      rec.reviewedCum.push(reviewedTotal);
      rec.overturnedCum.push(overturnedTotal);
      rec.acceptedCum.push(acceptedTotal);
      rec.faCum.push(faTotal);
      var sf = 0, nf = 0, sc = 0, ncl = 0;
      for (i = 0; i < n; i++) {
        var wi = conf.w(i);
        if (flipped[i]) { sf += wi; nf++; } else { sc += wi; ncl++; }
      }
      rec.wFlipped.push(nf ? sf / nf : NaN);
      rec.wClean.push(ncl ? sc / ncl : NaN);
    }

    record(0, null, '', []);                            // k = 0 baseline

    for (var k = 1; k <= cfg.cycles; k++) {
      // -- panel labels the train batch under the current policy
      var batch = shared.batchIdx[k];
      var bv = panelVote(policy, ds, batch, shared.batchNoise[k]);
      var disB = new Array(batch.length);
      for (i = 0; i < batch.length; i++) disB[i] = bv.sys[i] !== yHuman[batch[i]];
      conf.observe(batch, disB, bv.cons);

      // -- test partition gets panel evidence every cycle too
      var tvInc = panelVote(policy, ds, testIdx, shared.testNoise[k]);
      var disT = new Array(testIdx.length);
      for (i = 0; i < testIdx.length; i++) disT[i] = tvInc.sys[i] !== yHuman[testIdx[i]];
      conf.observe(testIdx, disT, tvInc.cons);

      // -- budgeted SME re-adjudication (before anchor selection)
      if (cfg.readj.strategy !== 'off') {
        var rngR = makeStream(cfg.seed, ST.readj, k);
        var rngS = makeStream(cfg.seed, ST.sme, k);
        var pools = [
          [devIdx, cfg.readj.budget],
          [testPoolIdx, cfg.readj.includeTest ? cfg.readj.testBudget : 0]
        ];
        for (var pi = 0; pi < 2; pi++) {
          var pool = pools[pi][0], budget = pools[pi][1];
          var unresolved = [];
          for (i = 0; i < pool.length; i++) if (!conf.resolved[pool[i]]) unresolved.push(pool[i]);
          var picked = selectQueue(conf.queueScore, unresolved, budget, cfg.readj.strategy, rngR);
          var rres = adjudicate(picked, yHuman, ds.y, cfg.readj.qSme, rngS);
          conf.markAdjudicated(picked);
          reviewedTotal += rres[0];
          overturnedTotal += rres[1];
        }
      }

      // -- anchors: misaligned batch items, consensus x confidence-weighted
      var anchors = [], weightsSel = [];
      var scoredPos = [], scoredVal = [];
      for (i = 0; i < batch.length; i++) {
        if (bv.sys[i] !== yHuman[batch[i]]) {
          var sc2 = bv.cons[i] * anchorWeight(cfg.weighting, conf.w(batch[i]));
          if (sc2 > 0) { scoredPos.push(i); scoredVal.push(sc2); }
        }
      }
      if (scoredPos.length) {
        var m = Math.min(cfg.nAnchors, scoredPos.length);
        var rngSel = makeStream(cfg.seed, ST.select, k);
        var chosen;
        if (cfg.anchorStrategy === 'stack_rank') {
          chosen = scoredPos.map(function (_, j) { return j; })
            .sort(function (a, b) { return scoredVal[b] - scoredVal[a] || a - b; })
            .slice(0, m);
        } else if (cfg.anchorStrategy === 'pps') {
          chosen = weightedSampleWithoutReplacement(
            rngSel, scoredPos.map(function (_, j) { return j; }), scoredVal, m);
        } else if (cfg.anchorStrategy === 'random') {
          chosen = sampleWithoutReplacement(
            rngSel, scoredPos.map(function (_, j) { return j; }), m);
        } else {
          throw new Error('unknown anchor strategy ' + cfg.anchorStrategy);
        }
        for (i = 0; i < chosen.length; i++) {
          anchors.push(batch[scoredPos[chosen[i]]]);
          weightsSel.push(scoredVal[chosen[i]]);
        }
      }

      // -- candidate step + gate (paired eval: same test judge noise tensor)
      var accepted = false, gateCell = 'NOOP';
      if (anchors.length) {
        var cand = clonePol(policy);
        polUpdate(cand, ds, anchors, yHuman, weightsSel, cfg.lr, cfg.clip);
        var tvCand = panelVote(cand, ds, testIdx, shared.testNoise[k]);
        var yHumanTest = new Int8Array(testIdx.length);
        for (i = 0; i < testIdx.length; i++) yHumanTest[i] = yHuman[testIdx[i]];
        var f1hInc = macroF1(yHumanTest, tvInc.sys);
        var f1hCand = macroF1(yHumanTest, tvCand.sys);
        var f1tInc = macroF1(yTrueTest, tvInc.sys);
        var f1tCand = macroF1(yTrueTest, tvCand.sys);
        accepted = !cfg.gateOn || (f1hCand - f1hInc) >= 0;        // gate_eps = 0
        var oracleOk = (f1tCand - f1tInc) >= 0;
        gateCell = accepted ? (oracleOk ? 'TA' : 'FA') : (oracleOk ? 'FR' : 'TR');
        if (accepted) {
          policy = cand;
          acceptedTotal++;
          if (!oracleOk) faTotal++;                     // FA: accepted, truth-gate would reject
        }
      }
      record(k, accepted, gateCell, anchors);
      snapshots.push(clonePol(policy));
    }
    return { snapshots: snapshots, rec: rec, yHumanFinal: yHuman };
  }

  /* ================= top level: the twin universes ========================== */

  function runSimulation(cfg) {
    var ds = makeGenai(cfg.n, cfg.seed);
    var split = splitIndices(ds.n, cfg.seed);
    var oracle = fitOracle(ds, cfg.seed);
    var noise = applyNoise(ds, cfg.noise, cfg.seed, oracle);
    var testIdx = split.testPoolIdx.slice(0, Math.min(cfg.testN, split.testPoolIdx.length));
    var K = SIGMAS.length;

    // Pre-draw every per-cycle stochastic tensor ONCE; both universes consume
    // the exact same batches and judge-perception noise (common random numbers).
    // The candidate re-uses the incumbent's test tensor within a cycle, exactly
    // like engine._eval_test re-instantiating the (seed, testjudge, k) stream.
    var shared = { batchIdx: [null], batchNoise: [null], testNoise: [] };
    shared.testNoise.push(drawNormals(makeStream(cfg.seed, ST.testjudge, 0), K * testIdx.length * 2));
    var used = new Uint8Array(ds.n);
    for (var k = 1; k <= cfg.cycles; k++) {
      shared.batchIdx.push(sampleBatch(split.devIdx, used, cfg.trainBatch,
                                       makeStream(cfg.seed, ST.batch, k)));
      shared.batchNoise.push(drawNormals(makeStream(cfg.seed, ST.judge, k),
                                         K * cfg.trainBatch * 2));
      shared.testNoise.push(drawNormals(makeStream(cfg.seed, ST.testjudge, k),
                                        K * testIdx.length * 2));
    }

    var world = { ds: ds, oracle: oracle, devIdx: split.devIdx,
                  testPoolIdx: split.testPoolIdx, testIdx: testIdx };
    var cleanLabels = { yHuman: Int8Array.from(ds.y), flipped: new Uint8Array(ds.n) };

    var noisy = runUniverse(cfg, world, noise, shared);       // corrupted human labels
    var clean = runUniverse(cfg, world, cleanLabels, shared); // twin: labels = truth

    var probe = probeGrid(ds, 50);
    var divergence = [];
    for (k = 0; k <= cfg.cycles; k++) {
      divergence.push(decisionDisagreement(noisy.snapshots[k], clean.snapshots[k], probe));
    }
    var oracleCeiling = macroF1(ds.y, polPredictAll(oracle, ds));

    return {
      cfg: cfg, ds: ds, oracle: oracle, bbox: bbox(ds),
      devIdx: split.devIdx, testPoolIdx: split.testPoolIdx, testIdx: testIdx,
      flipped: noise.flipped,
      noisy: noisy, clean: clean,
      divergence: divergence, oracleCeiling: oracleCeiling
    };
  }

  root.LabelSim = {
    ST: ST, SIGMAS: SIGMAS,
    makeStream: makeStream, permutation: permutation,
    sampleWithoutReplacement: sampleWithoutReplacement,
    weightedSampleWithoutReplacement: weightedSampleWithoutReplacement,
    makeGenai: makeGenai, splitIndices: splitIndices, probeGrid: probeGrid, bbox: bbox,
    fitOracle: fitOracle, makeV0: makeV0, clonePol: clonePol,
    polPredictAll: polPredictAll, polUpdate: polUpdate, classMargin: classMargin,
    panelVote: panelVote, applyNoise: applyNoise,
    makeConfidence: makeConfidence, anchorWeight: anchorWeight,
    selectQueue: selectQueue, adjudicate: adjudicate,
    macroF1: macroF1, decisionDisagreement: decisionDisagreement,
    sampleBatch: sampleBatch, runUniverse: runUniverse, runSimulation: runSimulation
  };
})(typeof globalThis !== 'undefined' ? globalThis : this);


/* ======================= browser UI (d3) ================================== */

(function () {
  'use strict';
  if (typeof document === 'undefined' || typeof window === 'undefined') return;
  if (typeof d3 === 'undefined') return;

  var L = window.LabelSim;
  var CLASS_NAMES = ['not_gen_ai', 'gen_ai'];
  var COL = {
    class0: '#2a78d6', class1: '#eb6834', mislabel: '#d03b3b', halo: '#eda100',
    noisy: '#4a3aa7', clean: '#008300', oracle: '#898781', diverge: '#2a78d6'
  };

  function $(id) { return document.getElementById(id); }

  /* ---------- controls ---------- */

  function readControls() {
    return {
      seed: parseInt($('seed').value, 10) || 0,
      n: parseInt($('npoints').value, 10),
      cycles: parseInt($('cycles').value, 10),
      trainBatch: 40, testN: 100, nAnchors: 10, lr: 0.4, clip: 0.5,
      noise: {
        model: $('noiseModel').value,
        rate: parseFloat($('rate').value),
        tau: parseFloat($('tau').value),
        flipFrom: $('oneWay').checked ? 1 : null
      },
      weighting: $('weighting').value,
      c0: parseFloat($('c0').value),
      anchorStrategy: $('anchorStrategy').value,
      gateOn: $('gateOn').checked,
      readj: {
        strategy: $('readjStrategy').value,
        budget: parseInt($('budget').value, 10),
        includeTest: $('includeTest').checked,
        testBudget: parseInt($('testBudget').value, 10),
        qSme: parseFloat($('qSme').value)
      }
    };
  }

  function refreshBadgesAndDisable() {
    $('cyclesVal').textContent = $('cycles').value;
    $('rateVal').textContent = (+$('rate').value).toFixed(2);
    $('tauVal').textContent = (+$('tau').value).toFixed(2);
    $('c0Val').textContent = (+$('c0').value).toFixed(2);
    $('budgetVal').textContent = $('budget').value;
    $('testBudgetVal').textContent = $('testBudget').value;
    $('qSmeVal').textContent = (+$('qSme').value).toFixed(2);
    var readjOff = $('readjStrategy').value === 'off';
    $('ctlTau').classList.toggle('disabled', $('noiseModel').value === 'uniform');
    $('ctlBudget').classList.toggle('disabled', readjOff);
    $('ctlIncludeTest').classList.toggle('disabled', readjOff);
    $('ctlTestBudget').classList.toggle('disabled', readjOff || !$('includeTest').checked);
    $('ctlQSme').classList.toggle('disabled', readjOff);
  }

  /* ---------- scatter ---------- */

  var SC = { W: 660, H: 540, M: { t: 8, r: 8, b: 8, l: 8 } };
  var scatter = {};   // { svg, xs, ys, points, rings, adj, gHalo, lineNoisy, lineClean, lineOracle }

  function buildScatter(R) {
    var svg = d3.select('#scatter').attr('viewBox', '0 0 ' + SC.W + ' ' + SC.H);
    svg.selectAll('*').remove();
    var b = R.bbox;
    var xs = d3.scaleLinear().domain([b.x0, b.x1]).range([SC.M.l, SC.W - SC.M.r]);
    var ys = d3.scaleLinear().domain([b.y0, b.y1]).range([SC.H - SC.M.b, SC.M.t]);

    svg.append('rect')
      .attr('x', SC.M.l).attr('y', SC.M.t)
      .attr('width', SC.W - SC.M.l - SC.M.r).attr('height', SC.H - SC.M.t - SC.M.b)
      .attr('fill', 'none').attr('stroke', '#e1e0d9');

    var gHalo = svg.append('g');
    var gPts = svg.append('g');
    var gRing = svg.append('g');
    var gAdj = svg.append('g');
    var gLines = svg.append('g');

    var ds = R.ds;
    var idxs = d3.range(ds.n);
    var tip = $('scatterTip');

    var points = gPts.selectAll('circle').data(idxs).join('circle')
      .attr('cx', function (i) { return xs(ds.X[i * 2]); })
      .attr('cy', function (i) { return ys(ds.X[i * 2 + 1]); })
      .attr('r', 3)
      .attr('fill', function (i) { return ds.y[i] === 1 ? COL.class1 : COL.class0; })
      .attr('fill-opacity', 0.85)
      .on('mousemove', function (event, i) {
        var res = state.result, k = state.displayK;
        var rec = res.noisy.rec;
        var mis = rec.mislabeled[k][i] === 1;
        var lbl = mis ? 1 - ds.y[i] : ds.y[i];
        var parts = ['#' + i,
          'true: ' + CLASS_NAMES[ds.y[i]],
          'human label: ' + CLASS_NAMES[lbl],
          'w = ' + rec.wSnap[k][i].toFixed(2)];
        if (mis) parts.push('MISLABELED');
        if (rec.resolved[k][i]) parts.push('adjudicated');
        if (res.flipped[i]) parts.push('noise-flipped at k=0');
        tip.textContent = parts.join('  ·  ');
        var host = $('scatterCard').getBoundingClientRect();
        tip.style.display = 'block';
        tip.style.left = Math.min(event.clientX - host.left + 14, host.width - 260) + 'px';
        tip.style.top = (event.clientY - host.top + 14) + 'px';
      })
      .on('mouseleave', function () { tip.style.display = 'none'; });

    var rings = gRing.selectAll('circle').data(idxs).join('circle')
      .attr('cx', function (i) { return xs(ds.X[i * 2]); })
      .attr('cy', function (i) { return ys(ds.X[i * 2 + 1]); })
      .attr('r', 6)
      .attr('fill', 'none').attr('stroke', COL.mislabel).attr('stroke-width', 1.6)
      .attr('pointer-events', 'none')
      .style('display', 'none');

    var adj = gAdj.selectAll('path').data(idxs).join('path')
      .attr('transform', function (i) {
        return 'translate(' + xs(ds.X[i * 2]) + ',' + ys(ds.X[i * 2 + 1]) + ')';
      })
      .attr('d', 'M0,-5.5L5.5,0L0,5.5L-5.5,0Z')
      .attr('fill', 'none').attr('stroke', '#52514e').attr('stroke-width', 1.1)
      .attr('pointer-events', 'none')
      .style('display', 'none');

    function mkLine(color, dash, width) {
      return gLines.append('line')
        .attr('stroke', color).attr('stroke-width', width)
        .attr('stroke-dasharray', dash).attr('pointer-events', 'none');
    }
    var lineOracle = mkLine(COL.oracle, '2 4', 1.6);
    var lineClean = mkLine(COL.clean, '7 5', 2.2);
    var lineNoisy = mkLine(COL.noisy, null, 2.6);

    scatter = {
      xs: xs, ys: ys, bbox: b, points: points, rings: rings, adj: adj,
      gHalo: gHalo, lineNoisy: lineNoisy, lineClean: lineClean, lineOracle: lineOracle
    };
    setBoundary(lineOracle, R.oracle);
  }

  // Intersect w.x + b = 0 with the padded bbox; hide if outside the view.
  function boundarySegment(pol, b) {
    var w0 = pol.w[0], w1 = pol.w[1], bb = pol.b, eps = 1e-9, pts = [];
    if (Math.abs(w1) > eps) {
      [b.x0, b.x1].forEach(function (x) {
        var y = -(bb + w0 * x) / w1;
        if (y >= b.y0 - 1e-6 && y <= b.y1 + 1e-6) pts.push([x, y]);
      });
    }
    if (Math.abs(w0) > eps) {
      [b.y0, b.y1].forEach(function (y) {
        var x = -(bb + w1 * y) / w0;
        if (x >= b.x0 - 1e-6 && x <= b.x1 + 1e-6) pts.push([x, y]);
      });
    }
    var uniq = [];
    pts.forEach(function (p) {
      var dup = uniq.some(function (q) {
        return Math.abs(q[0] - p[0]) < 1e-6 && Math.abs(q[1] - p[1]) < 1e-6;
      });
      if (!dup) uniq.push(p);
    });
    return uniq.length >= 2 ? [uniq[0], uniq[1]] : null;
  }

  function setBoundary(lineSel, pol) {
    var seg = boundarySegment(pol, scatter.bbox);
    if (!seg) { lineSel.style('display', 'none'); return; }
    lineSel.style('display', null)
      .attr('x1', scatter.xs(seg[0][0])).attr('y1', scatter.ys(seg[0][1]))
      .attr('x2', scatter.xs(seg[1][0])).attr('y2', scatter.ys(seg[1][1]));
  }

  function updateScatter(R, k) {
    var rec = R.noisy.rec;
    var mis = rec.mislabeled[k], res = rec.resolved[k];
    scatter.rings.style('display', function (i) { return mis[i] ? null : 'none'; });
    scatter.adj.style('display', function (i) { return res[i] ? null : 'none'; });

    // anchors flash: key on idx+cycle so the CSS pulse restarts every cycle
    var ds = R.ds;
    scatter.gHalo.selectAll('circle')
      .data(rec.anchors[k], function (d) { return d + ':' + k; })
      .join(function (enter) {
        return enter.append('circle')
          .attr('class', 'halo-mark')
          .attr('cx', function (i) { return scatter.xs(ds.X[i * 2]); })
          .attr('cy', function (i) { return scatter.ys(ds.X[i * 2 + 1]); })
          .attr('r', 10)
          .attr('fill', COL.halo).attr('fill-opacity', 0.18)
          .attr('stroke', COL.halo).attr('stroke-width', 2.2)
          .attr('pointer-events', 'none');
      });

    setBoundary(scatter.lineNoisy, R.noisy.snapshots[k]);
    setBoundary(scatter.lineClean, R.clean.snapshots[k]);
  }

  /* ---------- line charts ---------- */

  var CH = { W: 360, H: 208, M: { t: 8, r: 10, b: 24, l: 38 } };

  function makeLineChart(containerId, spec) {
    var card = d3.select('#' + containerId);
    card.append('div').attr('class', 'card-title').text(spec.title);
    var legend = card.append('div').attr('class', 'clegend');
    spec.series.forEach(function (s) {
      var item = legend.append('span').attr('class', 'item');
      item.append('svg').attr('width', 20).attr('height', 8)
        .append('line')
        .attr('x1', 0).attr('x2', 20).attr('y1', 4).attr('y2', 4)
        .attr('stroke', s.color).attr('stroke-width', 2.2)
        .attr('stroke-dasharray', s.dash || null);
      item.append('span').text(s.label);
    });
    var svg = card.append('svg').attr('viewBox', '0 0 ' + CH.W + ' ' + CH.H);
    var innerW = CH.W - CH.M.l - CH.M.r, innerH = CH.H - CH.M.t - CH.M.b;
    var gGrid = svg.append('g');
    var gX = svg.append('g').attr('class', 'axis')
      .attr('transform', 'translate(0,' + (CH.H - CH.M.b) + ')');
    var gY = svg.append('g').attr('class', 'axis')
      .attr('transform', 'translate(' + CH.M.l + ',0)');
    var clipId = 'clip-' + containerId;
    svg.append('clipPath').attr('id', clipId).append('rect')
      .attr('x', CH.M.l).attr('y', 0).attr('width', 0).attr('height', CH.H);
    var gSeries = svg.append('g').attr('clip-path', 'url(#' + clipId + ')');
    var cursor = svg.append('line').attr('class', 'cursor-line')
      .attr('y1', CH.M.t).attr('y2', CH.H - CH.M.b).style('display', 'none');
    var overlay = svg.append('rect')
      .attr('x', CH.M.l).attr('y', CH.M.t).attr('width', innerW).attr('height', innerH)
      .attr('fill', 'transparent');
    var tip = card.append('div').attr('class', 'tip');

    var x = null, y = null, data = null, xMax = 0;

    overlay.on('mousemove', function (event) {
      if (!x || !data) return;
      var mx = d3.pointer(event, svg.node())[0];
      var k = Math.round(x.invert(mx));
      k = Math.max(0, Math.min(xMax, k));
      var rows = spec.series.map(function (s) {
        var v = data[s.key][k];
        return '<div class="row"><svg width="14" height="6"><line x1="0" x2="14" y1="3" y2="3" ' +
          'stroke="' + s.color + '" stroke-width="2"' +
          (s.dash ? ' stroke-dasharray="' + s.dash + '"' : '') + '></line></svg>' +
          s.label + ': <b>' + (isFinite(v) ? v.toFixed(3) : 'n/a') + '</b></div>';
      }).join('');
      tip.html('<div class="row k">cycle ' + k + '</div>' + rows)
        .style('display', 'block');
      var host = card.node().getBoundingClientRect();
      var tx = Math.min(event.clientX - host.left + 12, host.width - 150);
      tip.style('left', tx + 'px').style('top', (event.clientY - host.top + 12) + 'px');
    }).on('mouseleave', function () { tip.style('display', 'none'); });

    return {
      setData: function (xm, d) {
        xMax = xm; data = d;
        x = d3.scaleLinear().domain([0, xm]).range([CH.M.l, CH.W - CH.M.r]);
        var dom = spec.yDomain(d);
        y = d3.scaleLinear().domain(dom).range([CH.H - CH.M.b, CH.M.t]).nice(4);
        gX.call(d3.axisBottom(x).ticks(6).tickFormat(d3.format('d')).tickSizeOuter(0));
        gY.call(d3.axisLeft(y).ticks(4).tickFormat(d3.format(spec.yFmt || '.2f')).tickSizeOuter(0));
        gGrid.selectAll('line').data(y.ticks(4)).join('line')
          .attr('class', 'gridline')
          .attr('x1', CH.M.l).attr('x2', CH.W - CH.M.r)
          .attr('y1', function (v) { return y(v); }).attr('y2', function (v) { return y(v); });
        var mkPath = d3.line()
          .defined(function (v) { return isFinite(v); })
          .x(function (v, i2) { return x(i2); })
          .y(function (v) { return y(v); });
        gSeries.selectAll('path').data(spec.series).join('path')
          .attr('fill', 'none')
          .attr('stroke', function (s) { return s.color; })
          .attr('stroke-width', 2)
          .attr('stroke-dasharray', function (s) { return s.dash || null; })
          .attr('d', function (s) { return mkPath(d[s.key]); });
      },
      setCursor: function (k) {
        if (!x) return;
        d3.select('#' + clipId + ' rect').attr('width', Math.max(0, x(k) - CH.M.l + 1));
        cursor.style('display', null).attr('x1', x(k)).attr('x2', x(k));
      }
    };
  }

  var chartF1, chartDiv, chartConf;

  function buildCharts() {
    d3.select('#chartF1').selectAll('*').remove();
    d3.select('#chartDiv').selectAll('*').remove();
    d3.select('#chartConf').selectAll('*').remove();
    chartF1 = makeLineChart('chartF1', {
      title: 'Oracle macro-F1 (policy vs ground truth, all points)',
      series: [
        { key: 'noisy', label: 'noisy universe', color: COL.noisy },
        { key: 'clean', label: 'clean twin', color: COL.clean, dash: '6 4' },
        { key: 'ceiling', label: 'oracle ceiling', color: COL.oracle, dash: '2 4' }
      ],
      yDomain: function (d) {
        var lo = 1;
        ['noisy', 'clean', 'ceiling'].forEach(function (key) {
          d[key].forEach(function (v) { if (isFinite(v) && v < lo) lo = v; });
        });
        return [Math.max(0, lo - 0.05), 1];
      }
    });
    chartDiv = makeLineChart('chartDiv', {
      title: 'Twin divergence: decision disagreement (50 x 50 probe grid)',
      series: [{ key: 'dd', label: 'noisy vs clean twin', color: COL.diverge }],
      yDomain: function (d) {
        var hi = 0;
        d.dd.forEach(function (v) { if (isFinite(v) && v > hi) hi = v; });
        return [0, Math.max(0.05, hi * 1.15)];
      }
    });
    chartConf = makeLineChart('chartConf', {
      title: 'Mean label confidence w (noisy universe)',
      series: [
        { key: 'flipped', label: 'flipped labels', color: COL.mislabel },
        { key: 'clean', label: 'clean labels', color: COL.clean, dash: '6 4' }
      ],
      yDomain: function () { return [0, 1]; }
    });
  }

  function setChartData(R) {
    var K = R.cfg.cycles;
    var ceiling = R.noisy.rec.f1True.map(function () { return R.oracleCeiling; });
    chartF1.setData(K, { noisy: R.noisy.rec.f1True, clean: R.clean.rec.f1True, ceiling: ceiling });
    chartDiv.setData(K, { dd: R.divergence });
    chartConf.setData(K, { flipped: R.noisy.rec.wFlipped, clean: R.noisy.rec.wClean });
  }

  /* ---------- chips ---------- */

  function updateChips(R, k) {
    var rec = R.noisy.rec;
    $('chipCycle').textContent = k + ' / ' + R.cfg.cycles;
    $('chipErrors').textContent = rec.residualErrors[k];
    $('chipReviewed').textContent = rec.reviewedCum[k] + ' / ' + rec.overturnedCum[k];
    $('chipOverturn').textContent = rec.reviewedCum[k] > 0
      ? (100 * rec.overturnedCum[k] / rec.reviewedCum[k]).toFixed(0) + '%' : '—';
    $('chipAccept').textContent = rec.acceptedCum[k];
    $('chipFA').textContent = rec.faCum[k];
  }

  /* ---------- run state + animation ---------- */

  var state = { result: null, displayK: 0, playing: false, timer: null };

  function setDisplay(k) {
    state.displayK = k;
    var R = state.result;
    updateScatter(R, k);
    chartF1.setCursor(k);
    chartDiv.setCursor(k);
    chartConf.setCursor(k);
    updateChips(R, k);
  }

  function stopPlay() {
    if (state.timer) { clearInterval(state.timer); state.timer = null; }
    state.playing = false;
    $('btnPlay').textContent = 'Play';
  }

  function startPlay() {
    stopPlay();
    state.playing = true;
    $('btnPlay').textContent = 'Pause';
    state.timer = setInterval(function () {   // ~3 cycles per second
      if (state.displayK >= state.result.cfg.cycles) { stopPlay(); return; }
      setDisplay(state.displayK + 1);
    }, 333);
  }

  function recompute(autoplay) {
    refreshBadgesAndDisable();
    var cfg = readControls();
    state.result = L.runSimulation(cfg);     // full deterministic trajectory
    buildScatter(state.result);
    buildCharts();
    setChartData(state.result);
    setDisplay(0);
    if (autoplay) startPlay(); else stopPlay();
  }

  /* ---------- wiring ---------- */

  function init() {
    var rerun = function () { recompute(true); };
    [['seed', 'change'], ['npoints', 'change'], ['cycles', 'input'],
     ['noiseModel', 'change'], ['oneWay', 'change'], ['rate', 'input'],
     ['tau', 'input'], ['weighting', 'change'], ['c0', 'input'],
     ['anchorStrategy', 'change'], ['gateOn', 'change'],
     ['readjStrategy', 'change'], ['budget', 'input'], ['includeTest', 'change'],
     ['testBudget', 'input'], ['qSme', 'input']
    ].forEach(function (pair) { $(pair[0]).addEventListener(pair[1], rerun); });

    $('btnReset').addEventListener('click', function () { recompute(false); });
    $('btnStep').addEventListener('click', function () {
      stopPlay();
      setDisplay(Math.min(state.displayK + 1, state.result.cfg.cycles));
    });
    $('btnPlay').addEventListener('click', function () {
      if (state.playing) { stopPlay(); return; }
      if (state.displayK >= state.result.cfg.cycles) setDisplay(0);
      startPlay();
    });

    recompute(true);
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
