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

  // Floating metric tooltip: a fixed-position bubble on any [data-tip] element.
  // Fixed positioning escapes the table's horizontal-scroll container, which
  // (per the overflow spec) would clip a CSS ::after tooltip on both axes.
  (function initTips() {
    let tip = null;
    const show = (el) => {
      const text = el.getAttribute('data-tip');
      if (!text) return;
      if (!tip) { tip = document.createElement('div'); tip.className = 'rush-tip'; document.body.appendChild(tip); }
      tip.textContent = text;
      tip.style.display = 'block';
      const r = el.getBoundingClientRect();
      const w = tip.offsetWidth, h = tip.offsetHeight;
      // Clamp fully on-screen (max after min, so a narrow viewport can't push it negative).
      const left = Math.max(8, Math.min(r.left, window.innerWidth - w - 8));
      let top = r.bottom + 6;
      if (top + h > window.innerHeight - 8) top = r.top - h - 6; // flip above near the bottom edge
      top = Math.max(8, top);
      tip.style.left = `${left}px`;
      tip.style.top = `${top}px`;
    };
    const hide = () => { if (tip) tip.style.display = 'none'; };
    document.addEventListener('mouseover', (e) => {
      const el = e.target.closest && e.target.closest('[data-tip]');
      if (el) show(el);
    });
    document.addEventListener('mouseout', (e) => {
      if (e.target.closest && e.target.closest('[data-tip]')) hide();
    });
    document.addEventListener('click', hide, true); // dismiss on any click (e.g. sorting)
  })();

  // One page of rows at a time: every row carries a thumbnail <img>, so an
  // unpaged 500-item queue means 500 image fetches on open. 100/page bounds it.
  const PAGE_SIZE = 100;

  const state = {
    items: null,
    loaded: false,
    loadToken: 0,
    expanded: new Set(),
    sortKey: 'importance',
    sortDir: -1,           // -1 desc, +1 asc
    hideResolved: false,
    page: 0,               // 0-based page into the sorted+filtered list
    busy: null             // key currently posting a review
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
    return `<span class="adjudicate-tier adjudicate-tier--${tier}" data-tip="${esc(TIER_LABEL[tier] || '')}">T${tier}</span>`;
  }

  const RESOLUTION_LABEL = {
    open: 'open', uncertain: 'uncertain',
    confirmed: 'SME confirmed', overturned: 'SME overturned'
  };
  function resolutionBadge(item) {
    const r = item.review || {};
    const res = r.resolution || 'open';
    const m = r.sme_confirmations;
    let extra = '';
    if (res === 'confirmed') extra = ` ×${(m || 1) - 1}`;
    if (res === 'overturned' && r.overturned_from != null) extra = ` ${esc(r.overturned_from)}→${esc(r.effective_label ?? '')}`;
    // Resolved = ≥2 SMEs agree; a lone overturn (m=1) still needs a 2nd SME.
    const pending = !r.resolved && (res === 'overturned') ? ' · needs 2nd SME' : '';
    const cls = res + (r.resolved ? ' adjudicate-res--done' : '');
    return `<span class="adjudicate-res adjudicate-res--${cls}" title="human confidence ${fmt(item.agg?.human_confidence)} · ${m || 1} SME(s)">${RESOLUTION_LABEL[res] || res}${extra}${pending}</span>`;
  }

  // The tier to show: the re-scored tier after an overturn (which may differ
  // from the historical worst_tier), else the worst tier the item ever hit.
  function currentTier(item) {
    return item.agg?.recomputed?.tier ?? item.agg?.worst_tier;
  }

  // Distinct SME-truth / effective labels seen in the queue — the overturn picker.
  function classOptions() {
    const set = new Set();
    (state.items || []).forEach((it) => {
      if (it.sme_truth !== undefined && it.sme_truth !== null) set.add(String(it.sme_truth));
      const eff = it.review?.effective_label;
      if (eff !== undefined && eff !== null) set.add(String(eff));
    });
    return [...set].sort();
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
          ? `<img src="/api/thumbnail?path=${encodeURIComponent(it.repo_rel_path)}" alt="${esc(it.image_id)}" loading="lazy" class="adjudicate-thumb" data-evidence-key="${esc(it.key || it.image_id)}" title="Click for the full LLM evidence — per-judge justification, sub-category, boundary, difficulty, confidence" />` : '';
        return `<td>${img}<span class="hint">${esc(it.image_id)}<br/>${esc(it.split ?? '')}</span></td>`;
      } },
    { key: 'runs', label: 'Flagged by', dir: -1, get: (it) => it.n_runs || 0,
      title: 'how many runs flagged this image (run numbers shown)',
      cell: (it) => `<td>${(it.runs || []).map(runChip).join(' ')}</td>` },
    { key: 'tier', label: 'Tier', dir: 1, get: (it) => currentTier(it) ?? 9,
      title: 'four-tier bucket — after an overturn, the tier RE-SCORED against the new golden label; else the worst tier the item hit. T1 misaligned+high-consensus is most important',
      cell: (it) => `<td>${tierBadge(currentTier(it))}</td>` },
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
    { key: 'importance', label: 'Importance', dir: -1,
      get: (it) => (it.agg?.effective_importance ?? it.agg?.importance),
      title: 'four-tier re-adjudication priority AFTER SME actions: base(misalignment×consensus) × confidence × boundary × (1 − human-confidence). Confirmed/overturned items fade. The default rank.',
      cell: (it) => {
        const eff = it.agg?.effective_importance ?? it.agg?.importance;
        const raw = it.agg?.importance;
        const faded = it.review && it.review.resolution !== 'open' && it.review.resolution !== 'uncertain';
        return `<td title="${faded ? `raw ${fmt(raw, 3)} before SME action` : ''}"><strong>${fmt(eff, 3)}</strong></td>`;
      } },
    { key: 'status', label: 'Status', dir: 1, get: (it) => it.review?.resolution || 'open',
      title: 'SME re-adjudication verdict; confirm raises human confidence (fades), overturn re-scores against the new golden label',
      cell: (it) => `<td>${resolutionBadge(it)}</td>` }
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
    return (runs || '<p class="hint">no per-run evidence recorded</p>') + actionPanel(item);
  }

  function actionPanel(item) {
    const key = item.key || item.image_id;
    const busy = state.busy === key;
    const majority = (item.latest || {}).majority_label;
    const opts = classOptions().map((c) =>
      `<option value="${esc(c)}"${String(c) === String(majority) ? ' selected' : ''}>${esc(c)}</option>`).join('');
    const r = item.review || {};
    const done = r.count
      ? `<span class="hint">${r.count} SME action(s) · latest: ${esc(RESOLUTION_LABEL[r.resolution] || r.resolution)}${r.resolution === 'overturned' ? ` (${esc(r.overturned_from ?? '')}→${esc(r.effective_label ?? '')})` : ''}</span>`
      : '';
    return `<div class="adjudicate-action" data-action-key="${esc(key)}">
      <strong>SME re-adjudication</strong>
      <span class="hint">SME truth <strong>${esc(item.sme_truth)}</strong> · the panel's majority was ${esc(majority ?? 'tie')}.</span>
      <div class="adjudicate-action-row">
        <button type="button" class="adjudicate-btn adjudicate-btn--confirm" data-verdict="confirm" ${busy ? 'disabled' : ''} title="the golden label is right; the LLMs were wrong. Raises human confidence and fades this item.">Confirm label ${esc(item.sme_truth)}</button>
        <span class="adjudicate-overturn">
          <button type="button" class="adjudicate-btn adjudicate-btn--overturn" data-verdict="overturn" ${busy ? 'disabled' : ''} title="the golden label was wrong; re-score the panel against a new truth.">Overturn →</button>
          <select class="adjudicate-newlabel" aria-label="new label">${opts}</select>
        </span>
        <button type="button" class="adjudicate-btn" data-verdict="uncertain" ${busy ? 'disabled' : ''} title="needs another SME / more review — stays in the queue.">Uncertain</button>
        ${busy ? '<span class="hint">saving…</span>' : done}
      </div>
    </div>`;
  }

  async function postReview(item, verdict, newLabel) {
    const key = item.key || item.image_id;
    state.busy = key;
    render();
    try {
      const payload = {
        area: item.area || activeArea(), key, image_id: item.image_id,
        verdict, prior_truth: item.sme_truth
      };
      if (verdict === 'overturn') payload.new_label = newLabel;
      const res = await window.rushApiPostJson('/api/adjudication/review', payload);
      if (res?.queue?.items) {
        state.items = res.queue.items;
        setStatus(`${res.queue.n_open} open · ${state.items.length} total · recorded ${verdict}`);
      } else {
        await loadQueue();
      }
    } catch (err) {
      setStatus(`Review failed: ${err?.message || err}`);
    } finally {
      state.busy = null;
      render();
    }
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
    state.page = 0;
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
    // Resolved = the backend's ≥2-SME flag (a lone overturn stays open).
    const isResolved = (it) => !!(it.review && it.review.resolved);
    if (summaryHost) {
      const byTier = (t) => state.items.filter((it) => currentTier(it) === t && !isResolved(it)).length;
      const resolved = state.items.filter(isResolved).length;
      summaryHost.innerHTML = `<div class="experiment-summary-grid">
        <div><span>Open</span><strong>${state.items.length - resolved} item(s)</strong></div>
        <div><span title="${esc(TIER_LABEL[1])}">Tier 1 — worst</span><strong>${byTier(1)}</strong></div>
        <div><span title="${esc(TIER_LABEL[2])}">Tier 2</span><strong>${byTier(2)}</strong></div>
        <div><span title="SME confirmed or overturned">Resolved</span><strong>${resolved}</strong></div>
        <div><span>Boundary-flagged</span><strong>${state.items.filter((it) => it.agg?.any_boundary && !isResolved(it)).length}</strong></div>
      </div>`;
    }
    if (!state.items.length) {
      host.innerHTML = '<p class="hint">Nothing to adjudicate — no completed live run has left misalignments behind yet.</p>';
      return;
    }
    let items = [...state.items];
    if (state.hideResolved) items = items.filter((it) => !isResolved(it));
    items.sort(comparator());
    if (!items.length) {
      host.innerHTML = '<p class="hint">All items resolved. Uncheck "hide resolved" to review them.</p>';
      return;
    }
    // Page AFTER sort+filter so page 1 is always the current top of the rank.
    const pages = Math.max(1, Math.ceil(items.length / PAGE_SIZE));
    state.page = Math.min(Math.max(0, state.page), pages - 1);
    const first = state.page * PAGE_SIZE;
    const pageItems = items.slice(first, first + PAGE_SIZE);
    const rangeBtns = Array.from({ length: pages }, (_, p) => {
      const lo = p * PAGE_SIZE + 1;
      const hi = Math.min((p + 1) * PAGE_SIZE, items.length);
      return `<button type="button" class="adjudicate-page-btn${p === state.page ? ' adjudicate-page-btn--active' : ''}" data-page="${p}">${lo}–${hi}</button>`;
    }).join('');
    const pager = pages > 1
      ? `<div class="adjudicate-pager">
          <button type="button" class="adjudicate-page-btn" data-page-delta="-1" ${state.page === 0 ? 'disabled' : ''}>‹ Prev</button>
          ${rangeBtns}
          <button type="button" class="adjudicate-page-btn" data-page-delta="1" ${state.page >= pages - 1 ? 'disabled' : ''}>Next ›</button>
          <span>${items.length} item(s)</span>
        </div>`
      : '';
    const body = pageItems.map((item) => {
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
      const tip = c.title ? ` data-tip="${esc(c.title)}"` : '';
      return `<th class="adjudicate-sortable${active ? ' adjudicate-sorted' : ''}" data-sort="${c.key}"${tip}>${esc(c.label)}${arrow}</th>`;
    }).join('');
    host.innerHTML = `${pager}<div class="adjudicate-table-scroll"><table class="summary-table adjudicate-table">
      <thead><tr>${head}</tr></thead><tbody>${body}</tbody></table></div>${pager}`;
    host.querySelectorAll('.adjudicate-page-btn').forEach((btn) => {
      btn.addEventListener('click', () => {
        if (btn.dataset.page !== undefined) state.page = Number(btn.dataset.page);
        else state.page += Number(btn.dataset.pageDelta || 0);
        render();
        host.scrollIntoView({ block: 'start' });
      });
    });
    // Thumbnail click -> the full-evidence drawer (per-judge justification,
    // l2_label, boundary pair, difficulty, confidence, quotes, tokens).
    host.querySelectorAll('[data-evidence-key]').forEach((el) => {
      el.addEventListener('click', () => {
        const item = (state.items || []).find(
          (it) => (it.key || it.image_id) === el.dataset.evidenceKey);
        if (!item || typeof window.rushShowEvidence !== 'function') return;
        const latest = item.latest || {};
        window.rushShowEvidence({
          image_id: item.image_id,
          repo_rel_path: item.repo_rel_path,
          sme_truth: item.sme_truth,
          run_id: latest.run_id,
          votes: latest.votes || []
        });
      });
    });
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
        state.page = 0;   // a new ranking starts back at the top
        render();
      });
    });
    host.querySelectorAll('.adjudicate-action').forEach((panel) => {
      const key = panel.dataset.actionKey;
      const item = (state.items || []).find((it) => (it.key || it.image_id) === key);
      if (!item) return;
      panel.querySelectorAll('[data-verdict]').forEach((btn) => {
        btn.addEventListener('click', () => {
          const verdict = btn.dataset.verdict;
          const newLabel = panel.querySelector('.adjudicate-newlabel')?.value;
          postReview(item, verdict, newLabel);
        });
      });
    });
  }

  function init() {
    if (!$('#adjudicate')) return;
    $('#adjudicateRefresh')?.addEventListener('click', () => loadQueue());
    $('#adjudicateHideResolved')?.addEventListener('change', (e) => {
      state.hideResolved = e.target.checked; state.page = 0; render();
    });
    window.addEventListener('rush-api-catalog', () => { if (state.loaded) loadQueue(); });
    window.addEventListener('rush-view-changed', (event) => {
      if (event.detail?.view === 'adjudicate' && !state.loaded) loadQueue();
    });
    if (document.body.classList.contains('view-adjudicate')) loadQueue();
  }

  if (typeof window.rushApiOnReady === 'function') window.rushApiOnReady(() => init());
  else document.addEventListener('DOMContentLoaded', init);
})();
