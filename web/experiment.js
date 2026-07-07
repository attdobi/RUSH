// §6 Optimize — the experiment crank panel.
// One experiment = a numbered, seeded PPO iteration run: k cycles of
// train mini-batch -> one clipped policy edit -> candidate eval on a fixed
// test partition -> gate (auto-accept iff system macro-F1 improves; the gate
// agent may veto). This panel starts experiments (judges come from the §3
// picker), plots the per-cycle decision-quality trajectory per judge + the
// system, lists every gate decision for SME review (the recorded
// critic-of-the-critic), and deep-links accepted versions into the §2 KG.
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
  const ACCEPT_COLOR = '#4de0a6';

  const state = {
    list: [],
    current: null,
    pollTimer: null,
    startedJobId: null
  };

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

  function fmtPct(value) {
    return (value === null || value === undefined) ? '—' : `${(value * 100).toFixed(1)}%`;
  }

  function fmtUsd(value) {
    return (typeof value === 'number') ? `$${value.toFixed(value >= 1 ? 2 : 4)}` : '—';
  }

  function statusChip(status) {
    const cls = {
      accepted: 'experiment-chip experiment-chip--accepted',
      skipped: 'experiment-chip experiment-chip--skipped',
      no_misalignments: 'experiment-chip experiment-chip--neutral',
      failed: 'experiment-chip experiment-chip--failed',
      open: 'experiment-chip experiment-chip--neutral'
    }[status] || 'experiment-chip experiment-chip--neutral';
    const label = {
      accepted: 'accepted',
      skipped: 'skipped',
      no_misalignments: 'aligned — no edit',
      failed: 'failed',
      open: 'running'
    }[status] || status;
    return `<span class="${cls}">${esc(label)}</span>`;
  }

  // ---- start ---------------------------------------------------------------

  function renderPanelSummary() {
    const models = selectedPanelModels();
    const el = $('#experimentPanelSummary');
    if (!el) return;
    el.textContent = models.length
      ? `Judge panel (from §3): ${models.join(', ')}`
      : 'Judge panel: select 2–5 models in §3 above.';
  }

  async function startExperiment() {
    const statusEl = $('#experimentStartStatus');
    const button = $('#experimentStart');
    const models = selectedPanelModels();
    if (models.length < 2 || models.length > 5) {
      statusEl.textContent = 'Select 2–5 judge models in the §3 panel first.';
      return;
    }
    const allowSpend = $('#runTriggerAllowSpend')?.checked === true;
    if (!allowSpend) {
      statusEl.textContent = 'Tick "Allow spend" in §3 — experiments run live.';
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
      policy_version: $('#runTriggerPolicyVersion')?.value || null,
      live: true,
      allow_spend: true
    };
    button.disabled = true;
    statusEl.textContent = 'Starting experiment…';
    try {
      const response = await window.rushApiPostJson('/api/experiments/start', payload);
      state.startedJobId = response?.job_id || null;
      statusEl.textContent = 'Experiment started — it will appear below as cycle 0 begins.';
      window.setTimeout(loadList, 1500);
    } catch (err) {
      statusEl.textContent = `Start failed: ${err?.message || err}`;
    } finally {
      button.disabled = false;
    }
  }

  // ---- list + detail -------------------------------------------------------

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
        const label = `#${e.run_number ?? '?'} · seed ${e.seed} · ${e.status}${e.dry_run ? ' · dry-run' : ''} · ${stamp}`;
        return `<option value="${esc(e.experiment_id)}">${esc(label)}</option>`;
      }).join('')
      : '<option value="">No experiments yet</option>';
    if (previous && state.list.some((e) => e.experiment_id === previous)) {
      select.value = previous;
    }
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
      // A transient fetch failure (server mid-restart, atomic-write race)
      // must not permanently stop the live poll loop.
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

  // ---- rendering -----------------------------------------------------------

  function renderDetail() {
    const summary = $('#experimentSummary');
    const statusLine = $('#experimentStatusLine');
    const exp = state.current;
    if (!exp) {
      summary.innerHTML = '<p class="hint">Start an experiment above, or pick one once it exists. '
        + 'Each run is fully reproducible from its seed.</p>';
      $('#experimentChart').innerHTML = '';
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
    renderLedger();
    renderHoldout();
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

  function renderChart() {
    const host = $('#experimentChart');
    const exp = state.current;
    const metricKey = $('#experimentMetric')?.value || 'macro_f1';
    const cycles = (exp?.cycles || []);
    if (!exp || cycles.length === 0) { host.innerHTML = ''; return; }
    const series = collectSeries(metricKey);
    const kMax = Math.max(1, ...cycles.map((c) => c.k));
    const W = 760; const H = 280; const padL = 46; const padR = 120; const padT = 14; const padB = 34;
    const x = (k) => padL + (k / kMax) * (W - padL - padR);
    const y = (v) => padT + (1 - v) * (H - padT - padB);

    let svg = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Decision quality by cycle">`;
    for (let grid = 0; grid <= 4; grid += 1) {
      const value = grid / 4;
      svg += `<line x1="${padL}" y1="${y(value)}" x2="${W - padR}" y2="${y(value)}" stroke="#2c3e68" stroke-width="1"/>`
        + `<text x="${padL - 8}" y="${y(value) + 4}" text-anchor="end" font-size="10" fill="#aab8d3">${(value * 100).toFixed(0)}%</text>`;
    }
    for (let k = 0; k <= kMax; k += 1) {
      svg += `<text x="${x(k)}" y="${H - padB + 16}" text-anchor="middle" font-size="10" fill="#aab8d3">k=${k}</text>`;
    }
    // Version labels under accepted cycles.
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

  function renderLedger() {
    const host = $('#experimentLedger');
    const exp = state.current;
    const cycles = (exp?.cycles || []).filter((c) => c.k >= 1);
    if (!cycles.length) {
      host.innerHTML = exp?.status === 'running'
        ? '<p class="hint">Cycle 1 in progress…</p>'
        : '';
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
        : '';
      const kgLink = (c.status === 'accepted' && c.new_version)
        ? `<button type="button" class="experiment-kg-link" data-version="${esc(c.new_version)}">View ${esc(c.new_version)} in KG →</button>`
        : '';
      const review = c.review
        ? `<span class="experiment-chip ${c.review.verdict === 'correct' ? 'experiment-chip--accepted' : (c.review.verdict === 'incorrect' ? 'experiment-chip--failed' : 'experiment-chip--neutral')}" title="${esc(c.review.comment || '')} — ${esc(c.review.reviewer)}">SME: ${esc(c.review.verdict)}</span>`
        : (c.gate ? `
          <span class="experiment-review-buttons" data-k="${c.k}">
            <button type="button" data-verdict="correct" title="The gate decided correctly">✓</button>
            <button type="button" data-verdict="incorrect" title="The gate decided incorrectly">✗</button>
            <button type="button" data-verdict="unsure" title="Unsure">?</button>
          </span>` : '<span class="hint">—</span>');
      return `<tr>
        <td>k=${c.k}</td>
        <td>${statusChip(c.status)}</td>
        <td>${c.n_misaligned ?? '—'} / ${(c.anchor_ids || []).length}</td>
        <td>${edits}${clipNote}</td>
        <td>${delta}${rationale}</td>
        <td>${kgLink}</td>
        <td>${review}</td>
      </tr>`;
    }).join('');
    host.innerHTML = `
      <table class="experiment-ledger-table">
        <thead><tr>
          <th>Cycle</th><th>Gate</th><th title="Misaligned in batch / anchors sampled (S1 random)">Misaligned / anchors</th>
          <th>Edit (≤${esc(exp.max_changes)} changes)</th>
          <th>Test system F1</th><th>Policy</th>
          <th title="Was the gate's decision correct? Your verdicts are recorded as training data for the critic agent.">SME review</th>
        </tr></thead>
        <tbody>${rows}</tbody>
      </table>`;

    host.querySelectorAll('.experiment-kg-link').forEach((button) => {
      button.addEventListener('click', () => {
        const versionSelect = document.querySelector('#policyGraphVersion');
        if (versionSelect) {
          const version = button.dataset.version;
          // A version accepted mid-session may not be in the picker yet —
          // add it so the deep-link never silently loads the wrong graph.
          if (!Array.from(versionSelect.options).some((o) => o.value === version)) {
            const option = document.createElement('option');
            option.value = version;
            option.textContent = version;
            versionSelect.appendChild(option);
          }
          versionSelect.value = version;
          versionSelect.dispatchEvent(new Event('change'));
        }
        document.querySelector('#grow')?.scrollIntoView({ behavior: 'smooth' });
      });
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

  // ---- init ----------------------------------------------------------------

  function init() {
    if (!$('#experiment')) return;
    const metricSelect = $('#experimentMetric');
    metricSelect.innerHTML = METRICS.map(([key, label]) => `<option value="${key}">${esc(label)}</option>`).join('');
    metricSelect.addEventListener('change', renderChart);
    $('#experimentSelect').addEventListener('change', (event) => loadDetail(event.target.value));
    $('#experimentRefresh').addEventListener('click', () => loadList());
    $('#experimentStart').addEventListener('click', startExperiment);
    document.addEventListener('change', (event) => {
      if (event.target?.classList?.contains('model-select-input')) renderPanelSummary();
    });
    window.addEventListener('rush-api-catalog', () => { renderPanelSummary(); loadList(); });
    renderPanelSummary();
    loadList();
  }

  if (typeof window.rushApiOnReady === 'function') {
    window.rushApiOnReady(() => init());
  } else {
    document.addEventListener('DOMContentLoaded', init);
  }
})();
