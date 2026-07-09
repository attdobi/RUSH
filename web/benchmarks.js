// Benchmarks tab — every live run of the active demo on the same yardstick.
// One row per run: config knobs + start→final SYSTEM macro-F1 on the fixed
// validation benchmark (cross-run comparable), with the locked-holdout
// readout alongside when the run paid for it. Data: GET /api/experiments
// (compact benchmark/holdout readouts included by pipeline.experiment
// .list_experiments). Dry runs are excluded — fake labels fake the numbers.
(() => {
  const $ = (sel) => document.querySelector(sel);

  function activeArea() {
    const demo = typeof window.rushActiveDemo === 'function' ? window.rushActiveDemo() : null;
    return demo?.policyGraph?.area || 'Generative_AI';
  }

  function esc(value) {
    return String(value ?? '').replace(/[&<>"']/g, (ch) => (
      { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[ch]
    ));
  }

  function fmtPct(value) {
    return (typeof value === 'number' && Number.isFinite(value))
      ? `${(value * 100).toFixed(1)}%` : '—';
  }

  function fmtUsd(value) {
    return (typeof value === 'number' && Number.isFinite(value))
      ? `$${value.toFixed(2)}` : '—';
  }

  function shortModel(id) {
    return String(id || '').split('/').pop() || '—';
  }

  const STRATEGY_LABELS = {
    random_misalignment: 'random (S1)',
    top_gradient: 'top |g|',
    top_importance: 'top importance',
  };

  // start→final readout with a delta chip; the delta is in percentage points
  // of macro-F1 (final − start), the number the whole crank optimizes.
  function readoutCell(block) {
    if (!block || typeof block.start_macro_f1 !== 'number') {
      return '<span class="hint">—</span>';
    }
    const start = block.start_macro_f1;
    const final = (typeof block.final_macro_f1 === 'number') ? block.final_macro_f1 : start;
    const deltaPp = (final - start) * 100;
    const cls = deltaPp > 0.05 ? 'up' : (deltaPp < -0.05 ? 'down' : 'flat');
    const sign = deltaPp > 0 ? '+' : '';
    return `<span class="benchmarks-readout" title="system macro-F1 under the START policy (${esc(block.start_version)}) → under the FINAL accepted policy (${esc(block.final_version || block.start_version)}) on the same ${esc(block.n)} images">
        ${fmtPct(start)} → ${fmtPct(final)}
        <span class="benchmarks-delta benchmarks-delta--${cls}">${sign}${deltaPp.toFixed(1)}pp</span>
      </span>
      <span class="hint">accuracy ${fmtPct(block.start_accuracy)} → ${fmtPct((typeof block.final_accuracy === 'number') ? block.final_accuracy : block.start_accuracy)} · n=${esc(block.n)}</span>`;
  }

  function statusChip(status) {
    const cls = status === 'completed' ? 'experiment-chip--accepted'
      : (status === 'failed' ? 'experiment-chip--failed' : 'experiment-chip--neutral');
    return `<span class="experiment-chip ${cls}">${esc(status || '?')}</span>`;
  }

  async function render() {
    const host = $('#benchmarksTable');
    const statusLine = $('#benchmarksStatusLine');
    if (!host) return;
    let payload;
    try {
      payload = await window.rushApiGetJson('/api/experiments');
    } catch (err) {
      if (statusLine) statusLine.textContent = 'Experiments API unavailable.';
      return;
    }
    const area = activeArea();
    const runs = (payload?.experiments || [])
      .filter((e) => e.area === area && !e.dry_run);
    if (statusLine) {
      const withBench = runs.filter((e) => e.benchmark).length;
      statusLine.textContent = runs.length
        ? `${runs.length} live run(s) · ${withBench} with a benchmark readout`
        : '';
    }
    if (!runs.length) {
      host.innerHTML = '<p class="hint">No live runs for this demo yet — start one on the Run the loop tab (tick "Benchmark readout" to land on this board).</p>';
      return;
    }
    const rows = runs.map((e) => {
      const started = String(e.started_at || '').slice(0, 16).replace('T', ' ');
      const gate = e.gate_mode === 'off' ? 'OFF'
        : (e.gate_mode === 'metric_only' ? 'metric rule'
          : `${shortModel(e.gate_model)} (${esc(e.gate_mode)}${e.gate_persona ? ` · ${esc(e.gate_persona)}` : ''})`);
      // Annotate the drafter's Input mode only when the run recorded one —
      // runs predating the knob attached images implicitly, so a fallback
      // label would misreport them.
      const inputLabel = e.drafter_context === 'text_and_images' ? ' · images+text'
        : (e.drafter_context === 'text_only' ? ' · text only' : '');
      const optimizer = `${shortModel(e.drafter_model)}${inputLabel}`;
      const anchors = `${STRATEGY_LABELS[e.strategy] || esc(e.strategy || '—')} · ${esc(e.max_anchors ?? '—')}/${esc(e.max_aligned_anchors ?? '—')}`;
      const loop = `${esc(e.k_max ?? '—')} · ${esc(e.batch_n ?? '—')} · ${esc(e.test_n ?? '—')}`;
      return `<tr class="${e.benchmark ? '' : 'benchmarks-row--none'}">
        <td class="benchmarks-run-cell">#${esc(e.run_number ?? '?')} · seed ${esc(e.seed)}<span class="hint">${esc(started)}</span></td>
        <td>${statusChip(e.status)}<span class="hint">${esc(e.accepted ?? 0)} accepted / ${esc(e.cycles_done ?? 0)}</span></td>
        <td>${esc(e.base_version)} → ${esc(e.current_version)}</td>
        <td>${gate}</td>
        <td>${esc(optimizer)}</td>
        <td>${anchors}</td>
        <td>${loop}</td>
        <td>${readoutCell(e.benchmark)}</td>
        <td>${readoutCell(e.holdout)}</td>
        <td>${fmtUsd(e.cost_usd_total)}</td>
      </tr>`;
    }).join('');
    host.innerHTML = `<div class="benchmarks-table-scroll">
      <table class="benchmarks-table">
        <thead><tr>
          <th>Run</th>
          <th>Status</th>
          <th>Policy</th>
          <th title="Gate mode (and the gate agent's persona when one reviews)">Gate</th>
          <th title="Drafter model · what it reads per anchor (Input knob)">Optimizer</th>
          <th title="Anchor selection method · misaligned/aligned counts per cycle">Anchors</th>
          <th title="Cycles k · train batch N · test size T">K · N · T</th>
          <th title="System macro-F1 on the FIXED validation split under the start policy → the final accepted policy. Same images every run — the honest cross-run comparison.">Benchmark F1 (start → final)</th>
          <th title="Same readout on the locked holdout, when the run paid for it (--holdout-final)">Holdout F1</th>
          <th>Cost</th>
        </tr></thead>
        <tbody>${rows}</tbody>
      </table>
    </div>`;
  }

  function init() {
    if (!$('#benchmarks')) return;
    $('#benchmarksRefresh')?.addEventListener('click', render);
    // Render when the tab becomes active (and once up front so the first
    // switch is instant); the table is cheap — one list fetch.
    window.addEventListener('rush-view-changed', (event) => {
      if (event?.detail?.view === 'benchmarks') render();
    });
    window.addEventListener('rush-api-catalog', render);
    render();
  }

  if (typeof window.rushApiOnReady === 'function') {
    window.rushApiOnReady(init);
  } else {
    document.addEventListener('DOMContentLoaded', init);
  }
})();
