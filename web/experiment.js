// The RUSH loop — the main view. One experiment = a numbered, seeded PPO
// iteration run: k cycles of train mini-batch -> misalignment anchors ->
// one clipped policy edit -> candidate eval on a fixed test partition ->
// gate (auto-accept iff system macro-F1 improves; the gate agent may veto).
// This module owns: the view switcher (loop | inspect), the start panel
// (judge picker lives here), the learning curve (auto-scaled y), the per-run
// judge table, the gate ledger with expandable proposal diffs + anchor
// images, SME reviews (critic-of-the-critic), and KG auto-follow as the
// policy evolves.
(() => {
  const POLL_MS = 2500;
  const $ = (sel) => document.querySelector(sel);
  const esc = (value) => String(value ?? '').replace(/[&<>"']/g, (ch) => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
  })[ch]);

  const METRICS = [
    ['macro_f1', 'Macro F1 (gate metric)'],
    ['accuracy', 'Accuracy'],
    ['macro_precision', 'Precision (macro)'],
    ['macro_recall', 'Recall (macro)'],
    ['macro_fpr', 'FPR (macro)'],
    ['macro_fnr', 'FNR (macro)']
  ];
  // Theme palette (styles.css :root) — the page is dark.
  const SERIES_COLORS = ['#82b5ff', '#ffd166', '#4de0a6', '#d394ff', '#ff6f91', '#6fe3e0'];
  const SYSTEM_COLOR = '#edf4ff';
  const GRID_COLOR = '#2c3e68';
  const AXIS_TEXT = '#aab8d3';

  const state = {
    list: [],
    current: null,
    pollTimer: null,
    expandedCycles: new Set(),
    detailCache: {},      // proposal_id -> diffs payload
    anchorCache: {},      // train_run_id -> misalignment records
    followedVersion: null // KG auto-follow bookkeeping
  };

  // ---- view switcher (loop | inspect) --------------------------------------

  function applyView(view) {
    document.body.classList.toggle('view-loop', view === 'loop');
    document.body.classList.toggle('view-inspect', view === 'inspect');
    document.querySelectorAll('#viewSwitcher .view-switcher-option').forEach((button) => {
      button.setAttribute('aria-pressed', String(button.dataset.view === view));
    });
    try { sessionStorage.setItem('rush_view', view); } catch (err) { /* private mode */ }
  }

  function initViewSwitcher() {
    const switcher = $('#viewSwitcher');
    if (!switcher) return;
    switcher.querySelectorAll('.view-switcher-option').forEach((button) => {
      button.addEventListener('click', () => applyView(button.dataset.view));
    });
    // Deep links into inspect sections flip the view so the target is visible.
    const inspectAnchors = new Set(['sample', 'grow', 'label', 'score', 'quality', 'about', 'provenance']);
    const fromHash = location.hash.replace('#', '');
    let view = 'loop';
    try { view = sessionStorage.getItem('rush_view') || 'loop'; } catch (err) { /* ok */ }
    if (inspectAnchors.has(fromHash)) view = 'inspect';
    if (fromHash === 'experiment') view = 'loop';
    applyView(view);
    window.addEventListener('hashchange', () => {
      const anchor = location.hash.replace('#', '');
      if (inspectAnchors.has(anchor)) applyView('inspect');
      if (anchor === 'experiment' || anchor === 'policyEvolution') applyView('loop');
    });
  }

  // ---- helpers --------------------------------------------------------------

  function activeArea() {
    const demo = typeof window.rushActiveDemo === 'function' ? window.rushActiveDemo() : null;
    return demo?.policyGraph?.area || 'Generative_AI';
  }

  function activeDemoId() {
    const demo = typeof window.rushActiveDemo === 'function' ? window.rushActiveDemo() : null;
    return demo?.id || 'genai';
  }

  function selectedPanelModels() {
    return Array.from(
      document.querySelectorAll('#runTriggerModels input.model-select-input[type="checkbox"]:checked')
    ).map((input) => input.value).filter(Boolean);
  }

  function fmtPct(value, digits = 1) {
    return (value === null || value === undefined) ? '—' : `${(value * 100).toFixed(digits)}%`;
  }

  function fmtUsd(value) {
    return (typeof value === 'number') ? `$${value.toFixed(value >= 1 ? 2 : 4)}` : '—';
  }

  function nextRunNumber() {
    const nums = state.list.map((e) => e.run_number).filter((n) => Number.isInteger(n));
    return nums.length ? Math.max(...nums) + 1 : 1;
  }

  function statusChip(status) {
    const cls = {
      accepted: 'experiment-chip experiment-chip--accepted',
      skipped: 'experiment-chip experiment-chip--skipped',
      no_misalignments: 'experiment-chip experiment-chip--neutral',
      failed: 'experiment-chip experiment-chip--failed',
      stopped: 'experiment-chip experiment-chip--neutral',
      open: 'experiment-chip experiment-chip--neutral'
    }[status] || 'experiment-chip experiment-chip--neutral';
    const label = {
      accepted: 'accepted',
      skipped: 'skipped',
      no_misalignments: 'aligned — no edit',
      failed: 'failed',
      stopped: 'stopped',
      open: 'running'
    }[status] || status;
    return `<span class="${cls}">${esc(label)}</span>`;
  }

  // ---- start ----------------------------------------------------------------

  function renderPanelSummary() {
    const models = selectedPanelModels();
    const summaryEl = $('#experimentPanelSummary');
    if (summaryEl) {
      summaryEl.textContent = models.length
        ? `Panel (${models.length}): ${models.join(', ')} · fixed generator k=0 = v0.1 · holdout locked`
        : 'Pick 2–5 judge models above.';
    }
    const button = $('#experimentStart');
    if (button) button.textContent = `Start run #${nextRunNumber()}`;
  }

  async function startExperiment() {
    const statusEl = $('#experimentStartStatus');
    const button = $('#experimentStart');
    const models = selectedPanelModels();
    if (models.length < 2 || models.length > 5) {
      statusEl.textContent = 'Pick 2–5 judge models above first.';
      return;
    }
    if ($('#experimentAllowSpend')?.checked !== true) {
      statusEl.textContent = 'Tick "Allow spend" — runs call live judge APIs.';
      return;
    }
    const seedRaw = ($('#experimentSeed')?.value || '').trim();
    const gateChoice = $('#experimentGateModel')?.value || 'openai/gpt-5.5';
    const payload = {
      demo: activeDemoId(),
      area: activeArea(),
      models,
      seed: seedRaw ? Number(seedRaw) : null,
      k_max: Number($('#experimentKMax')?.value || 5),
      batch_n: Number($('#experimentBatchN')?.value || 20),
      test_n: Number($('#experimentTestN')?.value || 100),
      max_changes: Number($('#experimentMaxChanges')?.value || 5),
      gate_mode: gateChoice === 'metric_only' ? 'metric_only' : 'agent',
      gate_model: gateChoice === 'metric_only' ? 'openai/gpt-5.5' : gateChoice,
      // k=0 is FIXED: every run starts from the same baseline generator.
      policy_version: null,
      live: true,
      allow_spend: true
    };
    button.disabled = true;
    statusEl.textContent = `Starting run #${nextRunNumber()}…`;
    try {
      await window.rushApiPostJson('/api/experiments/start', payload);
      statusEl.textContent = 'Run started — cycle 0 (baseline) appears below shortly.';
      window.setTimeout(() => loadList(false), 1800);
      window.setTimeout(() => loadList(false), 6000);
    } catch (err) {
      statusEl.textContent = `Start failed: ${err?.message || err}`;
    } finally {
      button.disabled = false;
    }
  }

  // ---- list + detail --------------------------------------------------------

  async function loadList(preserveSelection = true) {
    let payload;
    try {
      payload = await window.rushApiGetJson('/api/experiments');
    } catch (err) {
      $('#experimentStatusLine').textContent = 'Experiments API unavailable.';
      return;
    }
    const area = activeArea();
    state.list = (payload?.experiments || []).filter((e) => e.area === area);
    const select = $('#experimentSelect');
    const previous = preserveSelection ? select.value : '';
    select.innerHTML = state.list.length
      ? state.list.map((e) => {
        const stamp = (e.started_at || '').slice(0, 16).replace('T', ' ');
        const label = `Run #${e.run_number ?? '?'} · seed ${e.seed} · ${e.status}${e.dry_run ? ' · dry-run' : ''} · ${stamp}`;
        return `<option value="${esc(e.experiment_id)}">${esc(label)}</option>`;
      }).join('')
      : '<option value="">No runs yet — start run #1 above</option>';
    if (previous && state.list.some((e) => e.experiment_id === previous)) {
      select.value = previous;
    }
    renderPanelSummary();
    if (select.value) {
      await loadDetail(select.value);
    } else {
      state.current = null;
      renderDetail();
    }
  }

  async function loadDetail(experimentId) {
    if (!experimentId) return;
    try {
      state.current = await window.rushApiGetJson(`/api/experiments/${encodeURIComponent(experimentId)}`);
    } catch (err) {
      $('#experimentStatusLine').textContent = `Failed to load ${experimentId} — retrying.`;
      schedulePoll();
      return;
    }
    renderDetail();
    schedulePoll();
  }

  function schedulePoll() {
    if (state.pollTimer) window.clearTimeout(state.pollTimer);
    state.pollTimer = null;
    if (state.current?.status === 'running') {
      state.pollTimer = window.setTimeout(() => {
        loadDetail(state.current.experiment_id);
      }, POLL_MS);
    }
  }

  // ---- KG auto-follow -------------------------------------------------------

  function kgShowVersion(version, scroll = false) {
    const versionSelect = document.querySelector('#policyGraphVersion');
    if (versionSelect && version) {
      if (!Array.from(versionSelect.options).some((o) => o.value === version)) {
        const option = document.createElement('option');
        option.value = version;
        option.textContent = version;
        versionSelect.appendChild(option);
      }
      versionSelect.value = version;
      versionSelect.dispatchEvent(new Event('change'));
    }
    if (scroll) {
      document.querySelector('#policyEvolution')?.scrollIntoView({ behavior: 'smooth' });
    }
  }

  function autoFollowPolicy() {
    // While a run is live, the KG below tracks its newest accepted version —
    // "watch the policy evolve" without clicking anything.
    const exp = state.current;
    if (!exp || exp.status !== 'running') return;
    const version = exp.current_version;
    if (version && version !== state.followedVersion) {
      state.followedVersion = version;
      kgShowVersion(version, false);
    }
  }

  // ---- rendering ------------------------------------------------------------

  function renderDetail() {
    const summary = $('#experimentSummary');
    const statusLine = $('#experimentStatusLine');
    const exp = state.current;
    if (!exp) {
      summary.innerHTML = '<p class="hint">No run selected. Configure the panel above and start run #1 — '
        + 'each run is fully reproducible from its seed.</p>';
      $('#experimentChart').innerHTML = '';
      $('#experimentJudgeTable').innerHTML = '';
      $('#experimentLedger').innerHTML = '';
      $('#experimentHoldout').innerHTML = '';
      statusLine.textContent = '';
      return;
    }
    statusLine.textContent = exp.status === 'running' ? (exp.phase || 'running…') : (exp.phase || exp.status);

    const cycles = exp.cycles || [];
    const accepted = cycles.filter((c) => c.status === 'accepted');
    const baseline = cycles.find((c) => c.k === 0);
    const latest = cycles[cycles.length - 1];
    const f1Start = baseline?.metrics?.test?.system?.macro_f1;
    const f1Now = latest?.metrics?.test?.system?.macro_f1;
    summary.innerHTML = `
      <div class="experiment-summary-grid">
        <div><span>Run</span><strong>#${esc(exp.run_number)} · seed ${esc(exp.seed)}${exp.dry_run ? ' · dry-run' : ''}</strong></div>
        <div><span>Policy</span><strong>${esc(exp.base_version)} → ${esc(exp.current_version)}</strong></div>
        <div><span>Accepted / cycles</span><strong>${accepted.length} / ${Math.max(0, cycles.length - 1)} of ${esc(exp.k_max)}</strong></div>
        <div><span>Test system F1</span><strong>${fmtPct(f1Start)} → ${fmtPct(f1Now)}</strong></div>
        <div><span>Splits</span><strong>test ${esc(exp.test_n)} · batch ${esc(exp.batch_n)}/cycle · holdout locked</strong></div>
        <div><span>Gate</span><strong>${esc(exp.gate_model || 'metric rule')} (${esc(exp.gate_mode)})</strong></div>
        <div><span>Cost</span><strong>${fmtUsd(exp.cost_usd_total)}</strong></div>
      </div>`;

    renderChart();
    renderJudgeTable();
    renderLedger();
    renderHoldout();
    autoFollowPolicy();
  }

  function collectSeries(metricKey) {
    const exp = state.current;
    const cycles = (exp?.cycles || []).filter((c) => typeof c.k === 'number');
    const scorers = new Set();
    cycles.forEach((c) => {
      Object.keys(c.metrics?.test || {}).forEach((s) => scorers.add(s));
      Object.keys(c.metrics?.train || {}).forEach((s) => scorers.add(s));
    });
    const ordered = Array.from(scorers).sort((a, b) => (
      (a === 'system') - (b === 'system') || a.localeCompare(b)
    ));
    return ordered.map((scorer, idx) => ({
      scorer,
      color: scorer === 'system' ? SYSTEM_COLOR : SERIES_COLORS[idx % SERIES_COLORS.length],
      test: cycles.map((c) => ({ k: c.k, v: c.metrics?.test?.[scorer]?.[metricKey] ?? null, status: c.status })),
      train: cycles.map((c) => ({ k: c.k, v: c.metrics?.train?.[scorer]?.[metricKey] ?? null }))
    }));
  }

  function niceDomain(values) {
    // Auto-scale so near-zero metrics (FPR/FNR) and near-one metrics (F1 on a
    // strong panel) are both readable; always pad, never invert.
    const defined = values.filter((v) => v !== null && v !== undefined);
    if (!defined.length) return [0, 1];
    let lo = Math.min(...defined);
    let hi = Math.max(...defined);
    const pad = Math.max((hi - lo) * 0.25, 0.01);
    lo = Math.max(0, lo - pad);
    hi = Math.min(1, hi + pad);
    if (hi - lo < 0.02) { // dead-flat series: open a readable window around it
      lo = Math.max(0, lo - 0.02);
      hi = Math.min(1, hi + 0.02);
    }
    return [lo, hi];
  }

  function renderChart() {
    const host = $('#experimentChart');
    const exp = state.current;
    const metricKey = $('#experimentMetric')?.value || 'macro_f1';
    const cycles = (exp?.cycles || []);
    if (!exp || cycles.length === 0) { host.innerHTML = ''; return; }
    const series = collectSeries(metricKey);
    const allValues = series.flatMap((s) => [...s.test, ...s.train].map((p) => p.v));
    const [lo, hi] = niceDomain(allValues);
    const kMax = Math.max(1, ...cycles.map((c) => c.k));
    const W = 760; const H = 280; const padL = 52; const padR = 128; const padT = 14; const padB = 34;
    const x = (k) => padL + (k / kMax) * (W - padL - padR);
    const y = (v) => padT + (1 - (v - lo) / (hi - lo)) * (H - padT - padB);

    let svg = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Decision quality by cycle">`;
    for (let grid = 0; grid <= 4; grid += 1) {
      const value = lo + (grid / 4) * (hi - lo);
      const digits = (hi - lo) < 0.05 ? 1 : 0;
      svg += `<line x1="${padL}" y1="${y(value)}" x2="${W - padR}" y2="${y(value)}" stroke="${GRID_COLOR}" stroke-width="1"/>`
        + `<text x="${padL - 8}" y="${y(value) + 4}" text-anchor="end" font-size="10" fill="${AXIS_TEXT}">${(value * 100).toFixed(digits)}%</text>`;
    }
    for (let k = 0; k <= kMax; k += 1) {
      svg += `<text x="${x(k)}" y="${H - padB + 16}" text-anchor="middle" font-size="10" fill="${AXIS_TEXT}">k=${k}</text>`;
    }
    cycles.filter((c) => c.status === 'accepted' && c.new_version).forEach((c) => {
      svg += `<text x="${x(c.k)}" y="${H - padB + 28}" text-anchor="middle" font-size="9" fill="#4de0a6">${esc(c.new_version)}</text>`;
    });

    const path = (points) => {
      let d = ''; let pen = false;
      points.forEach((p) => {
        if (p.v === null || p.v === undefined) { pen = false; return; }
        d += `${pen ? 'L' : 'M'}${x(p.k).toFixed(1)},${y(p.v).toFixed(1)}`;
        pen = true;
      });
      return d;
    };

    series.forEach((s) => {
      const width = s.scorer === 'system' ? 2.5 : 1.5;
      if (path(s.train)) {
        svg += `<path d="${path(s.train)}" fill="none" stroke="${s.color}" stroke-width="${width}" stroke-dasharray="4 4" opacity="0.55"/>`;
      }
      if (path(s.test)) {
        svg += `<path d="${path(s.test)}" fill="none" stroke="${s.color}" stroke-width="${width}"/>`;
      }
      s.test.forEach((p) => {
        if (p.v === null || p.v === undefined) return;
        if (p.status === 'accepted') {
          svg += `<path d="M${x(p.k)},${y(p.v) - 5} l5,8 h-10 z" fill="${s.color}"/>`;
        } else if (p.status === 'skipped' || p.status === 'no_misalignments') {
          svg += `<circle cx="${x(p.k)}" cy="${y(p.v)}" r="2.6" fill="#0a1020" stroke="${s.color}" stroke-width="1.5"/>`;
        } else {
          svg += `<circle cx="${x(p.k)}" cy="${y(p.v)}" r="2.6" fill="${s.color}"/>`;
        }
      });
      const lastPoint = [...s.test].reverse().find((p) => p.v !== null && p.v !== undefined);
      if (lastPoint) {
        const label = s.scorer === 'system' ? 'system (majority)' : s.scorer.split('/').pop();
        svg += `<text x="${W - padR + 6}" y="${y(lastPoint.v) + 3}" font-size="10" fill="${s.color}">${esc(label)}</text>`;
      }
    });
    svg += '</svg>';
    host.innerHTML = svg;
  }

  function renderJudgeTable() {
    const host = $('#experimentJudgeTable');
    const exp = state.current;
    const cycles = exp?.cycles || [];
    if (!exp || !cycles.length) { host.innerHTML = ''; return; }
    const baseline = cycles.find((c) => c.k === 0)?.metrics?.test || {};
    const finalMetrics = cycles[cycles.length - 1]?.metrics?.test || {};
    const scorers = Array.from(new Set([...Object.keys(baseline), ...Object.keys(finalMetrics)]))
      .sort((a, b) => ((a === 'system') - (b === 'system')) || a.localeCompare(b));
    if (!scorers.length) { host.innerHTML = ''; return; }

    const delta = (scorer) => {
      const before = baseline[scorer]?.macro_f1;
      const after = finalMetrics[scorer]?.macro_f1;
      if (before === null || before === undefined || after === null || after === undefined) return '';
      const diff = after - before;
      if (Math.abs(diff) < 0.0005) return '<span class="experiment-delta experiment-delta--flat">—</span>';
      const cls = diff > 0 ? 'experiment-delta--up' : 'experiment-delta--down';
      return `<span class="experiment-delta ${cls}">${diff > 0 ? '+' : ''}${(diff * 100).toFixed(1)}</span>`;
    };

    const rows = scorers.map((scorer) => {
      const m = finalMetrics[scorer] || {};
      const isSystem = scorer === 'system';
      const name = isSystem ? 'system (majority vote)' : scorer;
      return `<tr class="${isSystem ? 'experiment-judge-system' : ''}">
        <td>${esc(name)}</td>
        <td>${fmtPct(m.accuracy)}</td>
        <td>${fmtPct(m.macro_f1)} ${delta(scorer)}</td>
        <td>${fmtPct(m.macro_precision)}</td>
        <td>${fmtPct(m.macro_recall)}</td>
        <td>${fmtPct(m.macro_fpr, 2)}</td>
        <td>${fmtPct(m.macro_fnr, 2)}</td>
        <td>${m.n ?? '—'}${m.n_abstained ? ` <span class="hint">(+${m.n_abstained} abstain)</span>` : ''}</td>
      </tr>`;
    }).join('');
    host.innerHTML = `
      <table class="experiment-ledger-table experiment-judge-table">
        <thead><tr>
          <th>Judge</th><th>Accuracy</th><th>Macro F1 <span class="hint">(Δ vs k=0)</span></th>
          <th>Precision</th><th>Recall</th><th>FPR</th><th>FNR</th><th>n</th>
        </tr></thead>
        <tbody>${rows}</tbody>
      </table>`;
  }

  // ---- gate ledger with expandable evidence ---------------------------------

  function thumbnailUrl(repoRelPath) {
    return `/api/thumbnail?path=${encodeURIComponent(repoRelPath)}`;
  }

  function renderAnchorCards(anchors) {
    if (!anchors || !anchors.length) {
      return '<p class="hint">No anchor records stored for this cycle.</p>';
    }
    const cards = anchors.map((a) => {
      const votes = (a.votes || []).map((v) => {
        const wrong = v.label !== a.sme_truth;
        return `<span class="experiment-vote ${wrong ? 'experiment-vote--wrong' : 'experiment-vote--right'}"
          title="${esc(v.model)} · confidence ${v.confidence ?? '—'}">${esc(String(v.model || '').split('/').pop())}: ${esc(v.label)}</span>`;
      }).join(' ');
      const img = a.repo_rel_path
        ? `<img src="${esc(thumbnailUrl(a.repo_rel_path))}" alt="${esc(a.image_id)}" loading="lazy" />`
        : '';
      return `<figure class="experiment-anchor-card">
        ${img}
        <figcaption>
          <strong>truth: ${esc(a.sme_truth)}</strong>
          <span class="hint">${esc(a.image_id)} · ${esc(a.misalignment_type || '')}${a.severity ? ` · ${esc(a.severity)}` : ''}</span>
          <div class="experiment-anchor-votes">${votes}</div>
        </figcaption>
      </figure>`;
    }).join('');
    return `<div class="experiment-anchor-grid">${cards}</div>`;
  }

  function renderDiffBlocks(diffs) {
    if (!diffs || !diffs.length) return '<p class="hint">No diff available.</p>';
    return diffs.map((d) => {
      const lines = String(d.unified_diff || '').split('\n').map((line) => {
        let cls = '';
        if (line.startsWith('+') && !line.startsWith('+++')) cls = 'diff-add';
        else if (line.startsWith('-') && !line.startsWith('---')) cls = 'diff-del';
        else if (line.startsWith('@@')) cls = 'diff-hunk';
        return `<span class="${cls}">${esc(line)}</span>`;
      }).join('\n');
      return `<div class="experiment-diff-block">
        <div class="experiment-diff-head">${esc(d.change)} <code>${esc(d.path)}</code></div>
        <pre class="experiment-diff-pre">${lines}</pre>
      </div>`;
    }).join('');
  }

  async function fetchAnchors(cycle) {
    // New runs persist anchors on the cycle; older ones fall back to the
    // train run's misalignment artifact (statically served).
    if (cycle.anchors && cycle.anchors.length) return cycle.anchors;
    const runId = cycle.train_run_id;
    if (!runId) return [];
    if (!state.anchorCache[runId]) {
      try {
        const response = await fetch(window.cacheBust
          ? window.cacheBust(`/data/runs/${encodeURIComponent(runId)}/scoring/misalignment.json`)
          : `/data/runs/${encodeURIComponent(runId)}/scoring/misalignment.json`);
        const payload = await response.json();
        state.anchorCache[runId] = payload?.records || [];
      } catch (err) {
        state.anchorCache[runId] = [];
      }
    }
    const wanted = new Set(cycle.anchor_ids || []);
    return state.anchorCache[runId]
      .filter((r) => wanted.has(String(r.image_id)))
      .map((r) => ({
        image_id: r.image_id,
        repo_rel_path: r.repo_rel_path,
        sme_truth: r.sme_truth,
        misalignment_type: r.misalignment_type,
        severity: r.severity,
        votes: (r.votes || []).map((v) => ({
          model: v.labeler_id || v.model_id, label: v.label, confidence: v.confidence
        }))
      }));
  }

  async function fetchProposalDiffs(proposalId) {
    if (!proposalId) return null;
    if (!state.detailCache[proposalId]) {
      try {
        state.detailCache[proposalId] = await window.rushApiGetJson(
          `/api/policy/proposals/${encodeURIComponent(proposalId)}`
        );
      } catch (err) {
        state.detailCache[proposalId] = { diffs: [] };
      }
    }
    return state.detailCache[proposalId];
  }

  function renderLedger() {
    const host = $('#experimentLedger');
    const exp = state.current;
    const cycles = (exp?.cycles || []).filter((c) => c.k >= 1);
    if (!cycles.length) {
      host.innerHTML = exp?.status === 'running'
        ? '<p class="hint">Cycle 1 in progress…</p>'
        : '<p class="hint">No cycles yet.</p>';
      return;
    }
    const rows = cycles.map((c) => {
      const gate = c.gate || {};
      const delta = `${fmtPct(gate.value_before)} → ${fmtPct(gate.value_after)}`;
      const edits = (c.edit_summary || []).map((e) => `${esc(e.change)} ${esc(e.path)}`).join('<br>')
        || '<span class="hint">—</span>';
      const clipNote = c.edit_clipped
        ? ` <span class="experiment-chip experiment-chip--neutral" title="Drafter proposed ${c.n_changes_proposed}; clipped to ${c.n_changes_applied} for reviewability">clipped ${c.n_changes_proposed}→${c.n_changes_applied}</span>`
        : '';
      const rationale = gate.rationale
        ? `<div class="experiment-rationale" title="${esc(gate.decided_by)}">${esc(gate.rationale)}</div>`
        : (c.error ? `<div class="experiment-rationale">${esc(c.error)}</div>` : '');
      const kgLink = (c.status === 'accepted' && c.new_version)
        ? `<button type="button" class="experiment-kg-link" data-version="${esc(c.new_version)}">View ${esc(c.new_version)} in graph ↓</button>`
        : '';
      const review = c.review
        ? `<span class="experiment-chip ${c.review.verdict === 'correct' ? 'experiment-chip--accepted' : (c.review.verdict === 'incorrect' ? 'experiment-chip--failed' : 'experiment-chip--neutral')}" title="${esc(c.review.comment || '')} — ${esc(c.review.reviewer)}">SME: ${esc(c.review.verdict)}</span>`
        : (c.gate ? `
          <span class="experiment-review-buttons" data-k="${c.k}">
            <button type="button" data-verdict="correct" title="The gate decided correctly">✓</button>
            <button type="button" data-verdict="incorrect" title="The gate decided incorrectly">✗</button>
            <button type="button" data-verdict="unsure" title="Unsure">?</button>
          </span>` : '<span class="hint">—</span>');
      const expanded = state.expandedCycles.has(c.k);
      const detailButton = (c.proposal_id || (c.anchor_ids || []).length)
        ? `<button type="button" class="experiment-detail-toggle" data-k="${c.k}" aria-expanded="${expanded}">${expanded ? '▾ Hide' : '▸ Evidence'}</button>`
        : '';
      return `<tr>
        <td>k=${c.k} ${detailButton}</td>
        <td>${statusChip(c.status)}</td>
        <td title="misaligned in batch / anchors sampled">${c.n_misaligned ?? '—'} / ${(c.anchor_ids || []).length}</td>
        <td>${edits}${clipNote}</td>
        <td>${delta}${rationale}</td>
        <td>${kgLink}</td>
        <td>${review}</td>
      </tr>
      <tr class="experiment-detail-row" data-detail-k="${c.k}" ${expanded ? '' : 'hidden'}>
        <td colspan="7"><div class="experiment-detail-host" data-detail-host="${c.k}"></div></td>
      </tr>`;
    }).join('');
    host.innerHTML = `
      <table class="experiment-ledger-table">
        <thead><tr>
          <th>Cycle</th><th>Gate</th><th title="Misaligned in batch / anchors sampled (random misalignment anchors)">Misaligned / anchors</th>
          <th>Edit (≤${esc(exp.max_changes)} changes)</th>
          <th>Test system F1</th><th>Policy</th>
          <th title="Was the gate's decision correct? Your verdicts are recorded as training data for the critic agent.">SME review</th>
        </tr></thead>
        <tbody>${rows}</tbody>
      </table>`;

    host.querySelectorAll('.experiment-detail-toggle').forEach((button) => {
      button.addEventListener('click', () => {
        const k = Number(button.dataset.k);
        const row = host.querySelector(`[data-detail-k="${k}"]`);
        const detailHost = host.querySelector(`[data-detail-host="${k}"]`);
        if (state.expandedCycles.has(k)) {
          state.expandedCycles.delete(k);
          row.hidden = true;
          button.textContent = '▸ Evidence';
          button.setAttribute('aria-expanded', 'false');
        } else {
          row.hidden = false;
          button.textContent = '▾ Hide';
          button.setAttribute('aria-expanded', 'true');
          toggleCycleDetailInline(k, detailHost);
        }
      });
    });
    // Re-render already-expanded rows (poll refresh keeps them open).
    state.expandedCycles.forEach((k) => {
      const detailHost = host.querySelector(`[data-detail-host="${k}"]`);
      if (detailHost) toggleCycleDetailInline(k, detailHost);
    });

    host.querySelectorAll('.experiment-kg-link').forEach((button) => {
      button.addEventListener('click', () => kgShowVersion(button.dataset.version, true));
    });
    host.querySelectorAll('.experiment-review-buttons button').forEach((button) => {
      button.addEventListener('click', async () => {
        const k = Number(button.parentElement.dataset.k);
        const verdict = button.dataset.verdict;
        let comment = '';
        if (verdict !== 'correct') {
          comment = window.prompt('Optional note for the record (why?)', '') || '';
        }
        try {
          await window.rushApiPostJson(
            `/api/experiments/${encodeURIComponent(state.current.experiment_id)}/review`,
            { k, verdict, reviewer: 'sme', comment }
          );
          await loadDetail(state.current.experiment_id);
        } catch (err) {
          $('#experimentStatusLine').textContent = `Review failed: ${err?.message || err}`;
        }
      });
    });
  }

  async function toggleCycleDetailInline(k, detailHost) {
    // Fill (or refresh) an expanded row's evidence without collapsing it.
    state.expandedCycles.add(k);
    const cycle = (state.current?.cycles || []).find((c) => c.k === k);
    if (!cycle || !detailHost) return;
    if (!detailHost.dataset.loaded) {
      detailHost.innerHTML = '<p class="hint">Loading evidence…</p>';
    }
    const [anchors, proposal] = await Promise.all([
      fetchAnchors(cycle),
      fetchProposalDiffs(cycle.proposal_id)
    ]);
    const gate = cycle.gate || {};
    const coverage = gate.comparison
      ? `<p class="hint">Gate compared ${gate.comparison.n_common} common test images
         (baseline decided ${gate.comparison.n_before}, candidate ${gate.comparison.n_after},
         partition ${gate.comparison.n_expected}).</p>`
      : '';
    const risks = (gate.risk_flags || []).length
      ? `<p class="experiment-risk-flags">Gate agent flags: ${gate.risk_flags.map(esc).join(' · ')}</p>`
      : '';
    detailHost.dataset.loaded = '1';
    detailHost.innerHTML = `
      <div class="experiment-cycle-detail">
        <div>
          <h4>Anchor misalignments <span class="hint">— the images that drove this edit</span></h4>
          ${renderAnchorCards(anchors)}
        </div>
        <div>
          <h4>Proposed edit <span class="hint">— ${esc(cycle.n_changes_applied ?? 0)} change(s)${cycle.edit_clipped ? `, clipped from ${esc(cycle.n_changes_proposed)}` : ''}</span></h4>
          ${renderDiffBlocks(proposal?.diffs)}
          ${coverage}${risks}
        </div>
      </div>`;
  }

  function renderHoldout() {
    const host = $('#experimentHoldout');
    const holdout = state.current?.holdout;
    if (!holdout || !holdout.start) { host.innerHTML = ''; return; }
    const start = holdout.start.metrics?.system || {};
    const final = holdout.final?.metrics?.system || {};
    host.innerHTML = `
      <div class="experiment-holdout">
        <strong>Locked holdout (${esc(holdout.n)} images, untouched by the loop):</strong>
        system macro-F1 ${fmtPct(start.macro_f1)} (${esc(holdout.start.version)})
        → ${fmtPct(final.macro_f1)} (${esc(holdout.final?.version)})
        · accuracy ${fmtPct(start.accuracy)} → ${fmtPct(final.accuracy)}
      </div>`;
  }

  // ---- init -----------------------------------------------------------------

  function init() {
    if (!$('#experiment')) return;
    const metricSelect = $('#experimentMetric');
    metricSelect.innerHTML = METRICS.map(([key, label]) => `<option value="${key}">${esc(label)}</option>`).join('');
    metricSelect.addEventListener('change', renderChart);
    $('#experimentSelect').addEventListener('change', (event) => {
      state.expandedCycles.clear();
      loadDetail(event.target.value);
    });
    $('#experimentRefresh').addEventListener('click', () => loadList());
    $('#experimentStart').addEventListener('click', startExperiment);
    document.addEventListener('change', (event) => {
      if (event.target?.classList?.contains('model-select-input')) renderPanelSummary();
    });
    window.addEventListener('rush-api-catalog', () => { renderPanelSummary(); loadList(); });
    renderPanelSummary();
    loadList();
  }

  initViewSwitcher();
  if (typeof window.rushApiOnReady === 'function') {
    window.rushApiOnReady(() => init());
  } else {
    document.addEventListener('DOMContentLoaded', init);
  }
})();
