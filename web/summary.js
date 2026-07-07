// Run summary view — Attila 2026-07-06: "select a run number and the mini
// batch k within the run … include the insights view where all of the LLM
// responses per image" are shown. Pick a run, a cycle, and which evaluation
// (train mini-batch / candidate test / baseline test); every image of that
// child run renders with each judge's full response — label, confidence,
// difficulty, justification, policy citations — ranked misaligned-first.
(() => {
  const $ = (sel) => document.querySelector(sel);
  const esc = (value) => String(value ?? '').replace(/[&<>"']/g, (ch) => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
  })[ch]);

  const state = {
    list: [],          // experiment summaries for the active area
    exp: null,         // selected experiment detail
    records: null,     // misalignment records of the selected child run
    recordsRunId: null,
    expanded: new Set(),
    loadToken: 0,
    expToken: 0        // orders overlapping /api/experiments/<id> loads
  };

  function activeArea() {
    const demo = typeof window.rushActiveDemo === 'function' ? window.rushActiveDemo() : null;
    return demo?.policyGraph?.area || 'Generative_AI';
  }

  function setStatus(text) {
    const el = $('#summaryStatusLine');
    if (el) el.textContent = text || '';
  }

  function fmtPct(value, digits = 1) {
    return (value === null || value === undefined) ? '—' : `${(value * 100).toFixed(digits)}%`;
  }

  // ---- pickers --------------------------------------------------------------

  async function loadRuns(preserve = true) {
    let payload;
    try {
      payload = await window.rushApiGetJson('/api/experiments');
    } catch (err) {
      setStatus('Experiments API unavailable.');
      return;
    }
    const area = activeArea();
    state.list = (payload?.experiments || []).filter((e) => e.area === area);
    const select = $('#summaryRunSelect');
    const previous = preserve ? select.value : '';
    select.innerHTML = state.list.length
      ? state.list.map((e) => {
        const label = `Run #${e.run_number ?? '?'} · seed ${e.seed} · ${e.status}${e.dry_run ? ' · dry-run' : ''}`;
        return `<option value="${esc(e.experiment_id)}">${esc(label)}</option>`;
      }).join('')
      : '<option value="">No runs yet</option>';
    if (previous && state.list.some((e) => e.experiment_id === previous)) select.value = previous;
    if (select.value) await loadExperiment(select.value);
    else { state.exp = null; renderAll(); }
  }

  async function loadExperiment(experimentId) {
    // Ordering guard: a stale response must never repopulate the pickers
    // for a run the user has already navigated away from.
    const token = ++state.expToken;
    let exp;
    try {
      exp = await window.rushApiGetJson(`/api/experiments/${encodeURIComponent(experimentId)}`);
    } catch (err) {
      if (token === state.expToken) setStatus(`Failed to load ${experimentId}.`);
      return;
    }
    if (token !== state.expToken) return; // superseded by a newer selection
    state.exp = exp;
    const cycles = (state.exp.cycles || []).filter((c) => typeof c.k === 'number');
    const cycleSelect = $('#summaryCycleSelect');
    cycleSelect.innerHTML = cycles.map((c) => {
      const label = c.k === 0 ? 'k=0 · baseline' : `k=${c.k} · ${c.status}`;
      return `<option value="${c.k}">${esc(label)}</option>`;
    }).join('') || '<option value="">—</option>';
    // Default to the last cycle with anything to show.
    const withRuns = cycles.filter((c) => c.train_run_id || c.candidate_run_id || c.test_run_id);
    if (withRuns.length) cycleSelect.value = String(withRuns[withRuns.length - 1].k);
    populateEvalSelect();
    await loadRecords();
  }

  function selectedCycle() {
    const k = Number($('#summaryCycleSelect')?.value);
    return (state.exp?.cycles || []).find((c) => c.k === k) || null;
  }

  function populateEvalSelect() {
    const cycle = selectedCycle();
    const select = $('#summaryEvalSelect');
    const options = [];
    if (cycle) {
      if (cycle.train_run_id) {
        options.push([cycle.train_run_id, `train mini-batch (${(cycle.train_ids || []).length || 'N'} imgs)`]);
      }
      if (cycle.candidate_run_id) options.push([cycle.candidate_run_id, 'candidate eval on test']);
      if (cycle.test_run_id) options.push([cycle.test_run_id, 'baseline eval on test']);
    }
    select.innerHTML = options.length
      ? options.map(([id, label]) => `<option value="${esc(id)}">${esc(label)}</option>`).join('')
      : '<option value="">no child runs for this cycle</option>';
  }

  // ---- records --------------------------------------------------------------

  async function loadRecords() {
    const runId = $('#summaryEvalSelect')?.value || '';
    const token = ++state.loadToken;
    state.expanded.clear();
    if (!runId) {
      state.records = null;
      state.recordsRunId = null;
      renderAll();
      return;
    }
    setStatus('Loading responses…');
    let records = null;
    try {
      // scoring/ carries the FULL audit trail (web/ exports truncate).
      const url = `/data/runs/${encodeURIComponent(runId)}/scoring/misalignment.json`;
      const response = await fetch(window.cacheBust ? window.cacheBust(url) : url);
      if (response.ok) records = (await response.json())?.records || [];
    } catch (err) { /* handled below */ }
    if (token !== state.loadToken) return;
    state.records = records;
    state.recordsRunId = runId;
    setStatus(records ? `${records.length} images · ${runId}` : 'No scoring artifact for that child run yet.');
    renderAll();
  }

  // ---- rendering ------------------------------------------------------------

  function degreeOf(row) {
    if (typeof window.rushMisalignmentDegree === 'function') return window.rushMisalignmentDegree(row);
    // Fallback: decisive (non-abstain) votes disagreeing with SME truth.
    const votes = row.votes || [];
    const decisive = votes.filter((v) => v.label && v.label !== 'abstain');
    const wrong = decisive.filter((v) => String(v.label) !== String(row.sme_truth)).length;
    const counts = {};
    decisive.forEach((v) => { counts[v.label] = (counts[v.label] || 0) + 1; });
    const top = Object.entries(counts).sort((a, b) => b[1] - a[1]);
    const majority = top.length && (top.length === 1 || top[0][1] > top[1][1]) ? top[0][0] : null;
    return { wrong, decisive: decisive.length, majorityWrong: majority !== null && String(majority) !== String(row.sme_truth) };
  }

  function majorityLabel(row) {
    const counts = {};
    (row.votes || []).forEach((v) => {
      if (v.label && v.label !== 'abstain') counts[v.label] = (counts[v.label] || 0) + 1;
    });
    const top = Object.entries(counts).sort((a, b) => b[1] - a[1]);
    if (!top.length) return '—';
    if (top.length > 1 && top[0][1] === top[1][1]) return 'tie';
    return top[0][0];
  }

  function renderCycleCard() {
    const host = $('#summaryCycleCard');
    const cycle = selectedCycle();
    const exp = state.exp;
    if (!host) return;
    if (!exp || !cycle) { host.innerHTML = ''; return; }
    const gate = cycle.gate || {};
    const items = [
      ['Run', `#${esc(exp.run_number)} · seed ${esc(exp.seed)}`],
      ['Cycle', cycle.k === 0 ? 'k=0 baseline' : `k=${cycle.k} · ${esc(cycle.status)}`],
      ['Policy', `${esc(cycle.generator_before || exp.base_generator || '')}${cycle.status === 'accepted' ? ` → ${esc(cycle.new_version)}` : ''}`],
      ['Gate F1', gate.value_before !== undefined ? `${fmtPct(gate.value_before)} → ${fmtPct(gate.value_after)}` : '—'],
      ['Misaligned / anchors', cycle.n_misaligned !== undefined ? `${cycle.n_misaligned} / ${(cycle.anchor_ids || []).length}` : '—'],
      ['Cost (k)', typeof cycle.cost_usd === 'number' ? `$${cycle.cost_usd.toFixed(2)}` : '—']
    ];
    host.innerHTML = `<div class="experiment-summary-grid">${items.map(([k, v]) =>
      `<div><span>${esc(k)}</span><strong>${v}</strong></div>`).join('')}</div>`;
  }

  function voteChip(vote, truth) {
    const label = String(vote.label ?? '—');
    const model = String(vote.model_id || vote.labeler_id || '').split('/').pop();
    let cls = 'summary-vote';
    if (label === 'abstain') cls += ' summary-vote--abstain';
    else cls += String(label) === String(truth) ? ' summary-vote--right' : ' summary-vote--wrong';
    const conf = (vote.confidence === null || vote.confidence === undefined) ? '' : ` · ${Number(vote.confidence).toFixed(2)}`;
    const boundary = vote.is_boundary ? ' ⧉' : '';
    const pair = (vote.is_boundary_between || []).map(String).join('↔');
    const title = `${vote.model_id || vote.labeler_id} · confidence ${vote.confidence ?? '—'} · difficulty ${vote.difficulty ?? '—'}`
      + (vote.is_boundary ? ` · boundary${pair ? ` ${pair}` : ''}` : '');
    return `<span class="${cls}" title="${esc(title)}">${esc(model)}: ${esc(label)}${esc(conf)}${boundary}</span>`;
  }

  // Keys the card lays out explicitly. Anything ELSE the parser kept from the
  // model reply falls through to a trailing "extra fields" strip, so a new
  // field in the output template shows up here without a UI change.
  // prepared_image_* is input provenance, not a model output — suppressed.
  const CARD_KEYS = new Set([
    'label', 'l2_label', 'confidence', 'difficulty', 'is_boundary',
    'is_boundary_between', 'justification', 'policy_citations',
    'policy_quotes', 'justification_too_long', 'input_tokens',
    'output_tokens', 'cost_usd', 'model_id', 'labeler_id'
  ]);

  function extraFields(vote) {
    return Object.entries(vote)
      .filter(([key, value]) => !CARD_KEYS.has(key) && !key.startsWith('prepared_image_')
        && value !== null && value !== undefined && value !== ''
        && !(Array.isArray(value) && !value.length))
      .map(([key, value]) => {
        const shown = Array.isArray(value) ? value.join(', ')
          : (typeof value === 'object' ? JSON.stringify(value) : value);
        return `<span class="summary-extra-field">${esc(key)}: ${esc(shown)}</span>`;
      }).join(' ');
  }

  function justificationCards(row) {
    const cards = (row.votes || []).map((vote) => {
      const citations = (vote.policy_citations || []).map((c) =>
        `<span class="summary-citation">${esc(c)}</span>`).join('');
      const quotes = (vote.policy_quotes || []).map((q) =>
        `<blockquote class="summary-policy-quote">${esc(q)}</blockquote>`).join('');
      const pair = (vote.is_boundary_between || []).map(String).join(' ↔ ');
      const boundary = vote.is_boundary
        ? `<span class="summary-flag summary-flag--boundary">is_boundary ${esc(pair) || 'true'}</span>`
        : '<span class="summary-flag">is_boundary false</span>';
      const tokens = (vote.input_tokens || vote.output_tokens)
        ? `${vote.input_tokens ?? '—'} in / ${vote.output_tokens ?? '—'} out tok` : null;
      const meta = [
        `label <strong>${esc(vote.label ?? '—')}</strong>`,
        vote.l2_label ? `node ${esc(vote.l2_label)}` : null,
        `confidence ${vote.confidence ?? '—'}`,
        `difficulty ${vote.difficulty ?? '—'}`,
        boundary,
        tokens,
        vote.cost_usd ? `$${Number(vote.cost_usd).toFixed(4)}` : null
      ].filter(Boolean).join(' · ');
      const extras = extraFields(vote);
      return `<div class="summary-justification-card">
        <h5>${esc(vote.model_id || vote.labeler_id || 'model')}</h5>
        <span class="hint">${meta}</span>
        ${citations}
        ${quotes}
        ${vote.justification ? `<p>${esc(vote.justification)}</p>` : '<p class="hint">no justification returned</p>'}
        ${vote.justification_too_long ? '<p class="hint">justification_too_long: reply exceeded the ~1500-char budget (shown in full)</p>' : ''}
        ${extras ? `<div class="summary-extra-fields">${extras}</div>` : ''}
      </div>`;
    }).join('');
    return `<div class="summary-justifications">${cards}</div>`;
  }

  function renderTable() {
    const host = $('#summaryTable');
    if (!host) return;
    if (!state.records) {
      host.innerHTML = state.exp
        ? '<p class="hint">Pick a cycle and evaluation above.</p>'
        : '<p class="hint">No run selected.</p>';
      return;
    }
    const onlyMis = $('#summaryOnlyMisaligned')?.checked === true;
    let rows = state.records;
    if (onlyMis) rows = rows.filter((r) => r.misalignment_type && r.misalignment_type !== 'all_agree');
    const ranked = rows.map((r) => ({ r, d: degreeOf(r) })).sort((a, b) => (
      (b.d.majorityWrong - a.d.majorityWrong)
      || ((b.d.decisive ? b.d.wrong / b.d.decisive : 0) - (a.d.decisive ? a.d.wrong / a.d.decisive : 0))
      || String(a.r.image_id).localeCompare(String(b.r.image_id))
    ));
    if (!ranked.length) {
      host.innerHTML = '<p class="hint">No rows match the current filter.</p>';
      return;
    }
    const typeLabel = {
      all_agree: 'all agree', model_vs_sme: 'models ≠ SME',
      model_vs_model: 'LLM split', consensus_wrong: 'majority ≠ SME'
    };
    const body = ranked.map(({ r, d }) => {
      const expanded = state.expanded.has(r.image_id);
      const img = r.repo_rel_path
        ? `<img src="/api/thumbnail?path=${encodeURIComponent(r.repo_rel_path)}" alt="${esc(r.image_id)}" loading="lazy" />`
        : '';
      const votes = (r.votes || []).map((v) => voteChip(v, r.sme_truth)).join(' ');
      const degree = d.decisive
        ? `${d.wrong} / ${d.decisive}${d.majorityWrong ? ' · majority ≠ SME' : ''}`
        : '—';
      const row = `<tr class="${d.majorityWrong ? 'summary-row--majority-wrong' : ''}">
        <td><button type="button" class="experiment-detail-toggle" data-image="${esc(r.image_id)}" aria-expanded="${expanded}">${expanded ? '▾' : '▸'}</button></td>
        <td>${img}<span class="hint">${esc(r.image_id)}</span></td>
        <td><strong>${esc(r.sme_truth)}</strong></td>
        <td>${esc(majorityLabel(r))}</td>
        <td>${votes}</td>
        <td>${esc(degree)}</td>
        <td>${esc(typeLabel[r.misalignment_type] || r.misalignment_type || '—')}</td>
      </tr>`;
      const detail = expanded
        ? `<tr><td colspan="7">${justificationCards(r)}</td></tr>`
        : '';
      return row + detail;
    }).join('');
    host.innerHTML = `
      <table class="summary-table">
        <thead><tr>
          <th></th><th>Image</th><th>SME truth</th><th>Majority</th>
          <th>Every judge's response</th>
          <th title="decisive votes disagreeing with SME truth">Misaligned</th><th>Agreement</th>
        </tr></thead>
        <tbody>${body}</tbody>
      </table>`;
    host.querySelectorAll('.experiment-detail-toggle').forEach((button) => {
      button.addEventListener('click', () => {
        const id = button.dataset.image;
        if (state.expanded.has(id)) state.expanded.delete(id);
        else state.expanded.add(id);
        renderTable();
      });
    });
  }

  function renderAll() {
    renderCycleCard();
    renderTable();
  }

  // ---- init -----------------------------------------------------------------

  function init() {
    if (!$('#summary')) return;
    $('#summaryRunSelect')?.addEventListener('change', (event) => loadExperiment(event.target.value));
    $('#summaryCycleSelect')?.addEventListener('change', () => { populateEvalSelect(); loadRecords(); });
    $('#summaryEvalSelect')?.addEventListener('change', () => loadRecords());
    $('#summaryOnlyMisaligned')?.addEventListener('change', renderTable);
    $('#summaryRefresh')?.addEventListener('click', () => loadRuns());
    window.addEventListener('rush-api-catalog', () => loadRuns());
    // Lazy-load the first time the view is opened (cheap no-op afterwards).
    window.addEventListener('rush-view-changed', (event) => {
      if (event.detail?.view === 'summary' && !state.list.length) loadRuns();
    });
    if (document.body.classList.contains('view-summary')) loadRuns();
  }

  if (typeof window.rushApiOnReady === 'function') {
    window.rushApiOnReady(() => init());
  } else {
    document.addEventListener('DOMContentLoaded', init);
  }
})();
