// Adjudication queue — the cross-run SME re-adjudication list.
// Ranked by the four-tier importance (see the About tab for the formalism):
//   T1 misaligned + high LLM-consensus  (unanimous & wrong — the worst)
//   T2 misaligned + low  LLM-consensus
//   T3 aligned    + low  LLM-consensus
//   T4 aligned    + high LLM-consensus  (the ideal state)
// Two DISTINCT signals drive it: SME agreement (LLM<->human) and LLM
// consensus (LLM<->LLM, SME-blind). Every column is click-sortable.
// Data: GET /api/adjudication?area=... (live runs only; dry runs excluded).
(() => {
  const $ = (sel) => document.querySelector(sel);
  const esc = (value) => String(value ?? '').replace(/[&<>"']/g, (ch) => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
  })[ch]);

  const state = {
    items: null,
    loaded: false,
    loadToken: 0,
    expanded: new Set(),
    sortKey: 'importance',
    sortDir: -1            // -1 desc, +1 asc
  };

  function activeArea() {
    const demo = typeof window.rushActiveDemo === 'function' ? window.rushActiveDemo() : null;
    return demo?.policyGraph?.area || 'Generative_AI';
  }

  function setStatus(text) {
    const el = $('#adjudicateStatusLine');
    if (el) el.textContent = text || '';
  }

  const fmt = (value, digits = 2) => (
    (value === null || value === undefined || Number.isNaN(value)) ? '—' : Number(value).toFixed(digits)
  );
  const isNum = (v) => typeof v === 'number' && Number.isFinite(v);

  function difficultyWord(score) {
    if (!isNum(score)) return '—';
    if (score >= 0.75) return 'high';
    if (score >= 0.25) return 'medium';
    return 'low';
  }

  const TIER_LABEL = {
    1: 'T1 · misaligned + high LLM-consensus — unanimous & wrong (worst)',
    2: 'T2 · misaligned + low LLM-consensus — the panel split and missed',
    3: 'T3 · aligned + low LLM-consensus — right, but the panel argued',
    4: 'T4 · aligned + high LLM-consensus — unanimous & right (ideal)'
  };

  function tierBadge(tier) {
    if (!tier) return '—';
    return `<span class="adjudicate-tier adjudicate-tier--${tier}" title="${esc(TIER_LABEL[tier] || '')}">T${tier}</span>`;
  }

  // --- columns: every one is click-sortable ---------------------------------
  // get() returns the sort value (null sorts last); cell() renders the td.
  // dir is the default direction when the column is first clicked.
  const COLUMNS = [
    { key: 'expand', label: '', sortable: false,
      cell: (it, exp) => `<td><button type="button" class="experiment-detail-toggle" data-key="${esc(it.key || it.image_id)}" aria-expanded="${exp}">${exp ? '▾' : '▸'}</button></td>` },
    { key: 'image', label: 'Image', dir: 1, get: (it) => String(it.image_id || ''),
      cell: (it) => {
        const img = it.repo_rel_path
          ? `<img src="/api/thumbnail?path=${encodeURIComponent(it.repo_rel_path)}" alt="${esc(it.image_id)}" loading="lazy" />` : '';
        return `<td>${img}<span class="hint">${esc(it.image_id)}<br/>${esc(it.split ?? '')}</span></td>`;
      } },
    { key: 'runs', label: 'Flagged by', dir: -1, get: (it) => it.n_runs || 0,
      title: 'how many runs flagged this image (run numbers shown)',
      cell: (it) => `<td>${(it.runs || []).map(runChip).join(' ')}</td>` },
    { key: 'tier', label: 'Tier', dir: 1, get: (it) => it.agg?.worst_tier ?? 9,
      title: 'worst four-tier bucket this item hit — T1 misaligned+high-consensus is the most important',
      cell: (it) => `<td>${tierBadge(it.agg?.worst_tier)}</td>` },
    { key: 'truth', label: 'SME truth', dir: 1, get: (it) => String(it.sme_truth || ''),
      title: 'the human (golden) label',
      cell: (it) => `<td><strong>${esc(it.sme_truth)}</strong></td>` },
    { key: 'sme', label: 'SME agree', dir: 1, get: (it) => it.agg?.sme_fraction,
      title: 'LLM↔HUMAN: fraction of judges matching the SME label (m/N). Low = misaligned. Sorts ascending (most misaligned first).',
      cell: (it) => {
        const s = (it.latest || {}).sme_agreement || {};
        const frac = it.agg?.sme_fraction;
        const md = s.decisive ? `${s.n_agree}/${s.decisive}` : '—';
        return `<td class="${isNum(frac) && frac < 0.5 ? 'adjudicate-misaligned' : ''}">${md} <span class="hint">${fmt(frac)}</span></td>`;
      } },
    { key: 'consensus', label: 'LLM consensus', dir: -1, get: (it) => it.agg?.consensus_fraction,
      title: 'LLM↔LLM (SME-blind): fraction of judges on the modal label. High + misaligned = systematic error (worst).',
      cell: (it) => {
        const c = (it.latest || {}).consensus || {};
        const frac = it.agg?.consensus_fraction;
        const md = c.decisive ? `${c.majority_count}/${c.decisive}${c.tie ? ' tie' : ''}` : '—';
        return `<td>${md} <span class="hint">${fmt(frac)}</span></td>`;
      } },
    { key: 'conf', label: 'Avg conf', dir: -1, get: (it) => it.agg?.avg_confidence,
      title: 'mean self-reported confidence across judges and runs',
      cell: (it) => `<td>${fmt(it.agg?.avg_confidence)}</td>` },
    { key: 'difficulty', label: 'Difficulty', dir: -1, get: (it) => it.agg?.difficulty_score,
      title: 'low=0, medium=0.5, high=1 averaged across judges and runs',
      cell: (it) => `<td>${difficultyWord(it.agg?.difficulty_score)} <span class="hint">${fmt(it.agg?.difficulty_score)}</span></td>` },
    { key: 'boundary', label: 'Boundary', dir: -1, get: (it) => it.agg?.boundary_rate,
      title: 'fraction of judges flagging is_boundary (a documented confusion case)',
      cell: (it) => `<td>${fmt(it.agg?.boundary_rate)}${it.agg?.any_boundary ? ' <span class="summary-flag summary-flag--boundary" title="≥1 judge flagged a boundary">⧉</span>' : ''}</td>` },
    { key: 'grad', label: '|g|', dir: -1, get: (it) => it.agg?.grad_magnitude,
      title: 'gradient magnitude |g| = 1 − p, p = confidence if correct else 1 − confidence; confident-wrong ≈ 1',
      cell: (it) => `<td>${fmt(it.agg?.grad_magnitude)}</td>` },
    { key: 'importance', label: 'Importance', dir: -1, get: (it) => it.agg?.importance,
      title: 'the four-tier re-adjudication priority: base(misalignment×consensus) × confidence × boundary × (1 − human-confidence). The default rank.',
      cell: (it) => `<td><strong>${fmt(it.agg?.importance, 3)}</strong></td>` }
  ];

  function runChip(run) {
    const where = run.kind === 'train' ? `k${run.k} train`
      : run.kind === 'test' ? 'test' : run.kind;
    const title = `${run.policy || ''} · ${run.misalignment_type || ''} · majority ${run.majority_label ?? 'tie'}`;
    return `<span class="adjudicate-run-chip" title="${esc(title)}">#${esc(run.run_number ?? '?')} · ${esc(where)}</span>`;
  }

  function voteChips(votes, truth) {
    return (votes || []).map((v) => {
      const label = String(v.label ?? '—');
      let cls = 'summary-vote';
      if (label === 'abstain') cls += ' summary-vote--abstain';
      else cls += label === String(truth) ? ' summary-vote--right' : ' summary-vote--wrong';
      const conf = (v.confidence === null || v.confidence === undefined) ? '' : ` · ${Number(v.confidence).toFixed(2)}`;
      const model = String(v.model || '').split('/').pop();
      const title = `${v.model} · difficulty ${v.difficulty ?? '—'}${v.is_boundary ? ' · boundary' : ''}`;
      return `<span class="${cls}" title="${esc(title)}">${esc(model)}: ${esc(label)}${esc(conf)}${v.is_boundary ? ' ⧉' : ''}</span>`;
    }).join(' ');
  }

  function detailBlock(item) {
    const runs = (item.runs || []).map((run) => {
      const c = run.consensus || {};
      const s = run.sme_agreement || {};
      const grad = run.gradient || {};
      const imp = run.importance || {};
      const meta = [
        `${esc(run.kind)}${run.k !== null && run.k !== undefined ? ` · k=${esc(run.k)}` : ''}`,
        `policy ${esc(run.policy ?? '—')}`,
        `tier ${esc(imp.tier ?? '—')}`,
        `SME agree ${esc(s.n_agree ?? '—')}/${esc(s.decisive ?? '—')}`,
        `LLM consensus ${esc(c.majority_count ?? '—')}/${esc(c.decisive ?? '—')}${c.tie ? ' tie' : ''}`,
        `avg conf ${fmt(run.avg_confidence)}`,
        `difficulty ${difficultyWord(run.difficulty_score)}`,
        `|g| ${fmt(grad.avg_magnitude)}`,
        `importance ${fmt(imp.readjudication, 3)}`
      ].join(' · ');
      return `<div class="adjudicate-detail">
        <h5>Run #${esc(run.run_number ?? '?')} <span class="hint">${esc(run.run_id ?? '')}</span></h5>
        <span class="hint">${meta}</span>
        <div>${voteChips(run.votes, item.sme_truth)}</div>
      </div>`;
    }).join('');
    return runs || '<p class="hint">no per-run evidence recorded</p>';
  }

  async function loadQueue() {
    const token = ++state.loadToken;
    setStatus('Loading queue…');
    let payload;
    try {
      payload = await window.rushApiGetJson(`/api/adjudication?area=${encodeURIComponent(activeArea())}`);
    } catch (err) {
      if (token === state.loadToken) setStatus('Adjudication API unavailable.');
      return;
    }
    if (token !== state.loadToken) return;
    state.items = payload?.items || [];
    state.loaded = true;
    state.expanded.clear();
    setStatus(`${state.items.length} item(s) awaiting SME re-adjudication`);
    render();
  }

  function comparator() {
    const col = COLUMNS.find((c) => c.key === state.sortKey) || COLUMNS.find((c) => c.key === 'importance');
    const dir = state.sortDir;
    return (a, b) => {
      const va = col.get ? col.get(a) : null;
      const vb = col.get ? col.get(b) : null;
      const na = va === null || va === undefined || (typeof va === 'number' && Number.isNaN(va));
      const nb = vb === null || vb === undefined || (typeof vb === 'number' && Number.isNaN(vb));
      if (na && nb) return String(a.image_id).localeCompare(String(b.image_id));
      if (na) return 1;   // nulls always last, both directions
      if (nb) return -1;
      if (va < vb) return -1 * dir;
      if (va > vb) return 1 * dir;
      return String(a.image_id).localeCompare(String(b.image_id));
    };
  }

  function render() {
    const host = $('#adjudicateTable');
    const summaryHost = $('#adjudicateSummary');
    if (!host) return;
    if (!state.items) { host.innerHTML = '<p class="hint">Loading…</p>'; return; }
    if (summaryHost) {
      const byTier = (t) => state.items.filter((it) => it.agg?.worst_tier === t).length;
      summaryHost.innerHTML = `<div class="experiment-summary-grid">
        <div><span>Queue</span><strong>${state.items.length} item(s)</strong></div>
        <div><span title="${esc(TIER_LABEL[1])}">Tier 1 — worst</span><strong>${byTier(1)}</strong></div>
        <div><span title="${esc(TIER_LABEL[2])}">Tier 2</span><strong>${byTier(2)}</strong></div>
        <div><span>Flagged by &gt;1 run</span><strong>${state.items.filter((it) => (it.n_runs || 0) > 1).length}</strong></div>
        <div><span>Boundary-flagged</span><strong>${state.items.filter((it) => it.agg?.any_boundary).length}</strong></div>
      </div>`;
    }
    if (!state.items.length) {
      host.innerHTML = '<p class="hint">Nothing to adjudicate — no completed live run has left misalignments behind yet.</p>';
      return;
    }
    const items = [...state.items].sort(comparator());
    const body = items.map((item) => {
      const key = item.key || item.image_id;
      const expanded = state.expanded.has(key);
      const row = `<tr>${COLUMNS.map((c) => c.cell(item, expanded)).join('')}</tr>`;
      const detail = expanded ? `<tr><td colspan="${COLUMNS.length}">${detailBlock(item)}</td></tr>` : '';
      return row + detail;
    }).join('');
    const head = COLUMNS.map((c) => {
      if (!c.sortable && c.sortable !== undefined) return `<th>${esc(c.label)}</th>`;
      if (c.get === undefined) return `<th>${esc(c.label)}</th>`;
      const active = state.sortKey === c.key;
      const arrow = active ? (state.sortDir === -1 ? ' ▾' : ' ▴') : '';
      return `<th class="adjudicate-sortable${active ? ' adjudicate-sorted' : ''}" data-sort="${c.key}" title="${esc(c.title || '')}">${esc(c.label)}${arrow}</th>`;
    }).join('');
    host.innerHTML = `<div class="adjudicate-table-scroll"><table class="summary-table adjudicate-table">
      <thead><tr>${head}</tr></thead><tbody>${body}</tbody></table></div>`;
    host.querySelectorAll('.experiment-detail-toggle').forEach((button) => {
      button.addEventListener('click', () => {
        const key = button.dataset.key;
        if (state.expanded.has(key)) state.expanded.delete(key); else state.expanded.add(key);
        render();
      });
    });
    host.querySelectorAll('.adjudicate-sortable').forEach((th) => {
      th.addEventListener('click', () => {
        const k = th.dataset.sort;
        const col = COLUMNS.find((c) => c.key === k);
        if (state.sortKey === k) state.sortDir = -state.sortDir;
        else { state.sortKey = k; state.sortDir = col?.dir ?? -1; }
        render();
      });
    });
  }

  function init() {
    if (!$('#adjudicate')) return;
    $('#adjudicateRefresh')?.addEventListener('click', () => loadQueue());
    window.addEventListener('rush-api-catalog', () => { if (state.loaded) loadQueue(); });
    window.addEventListener('rush-view-changed', (event) => {
      if (event.detail?.view === 'adjudicate' && !state.loaded) loadQueue();
    });
    if (document.body.classList.contains('view-adjudicate')) loadQueue();
  }

  if (typeof window.rushApiOnReady === 'function') window.rushApiOnReady(() => init());
  else document.addEventListener('DOMContentLoaded', init);
})();
