// Adjudication queue — Attila 2026-07-06: "keep a running list of the items
// to adjudicate stack ranked by llm consensus count (or lack thereof), the
// confidence score, and difficulty rating. Averaged across llm judges. Be
// sure to indicate which run number(s) the item came from. (also stack rank
// by the gradient descent formalism)".
// Data: GET /api/adjudication?area=... — aggregated at read time from every
// experiment.json readjudication block (live runs only; dry runs excluded).
(() => {
  const $ = (sel) => document.querySelector(sel);
  const esc = (value) => String(value ?? '').replace(/[&<>"']/g, (ch) => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
  })[ch]);

  const state = {
    items: null,      // aggregated queue items for the active area
    loaded: false,
    loadToken: 0,
    expanded: new Set()
  };

  function activeArea() {
    const demo = typeof window.rushActiveDemo === 'function' ? window.rushActiveDemo() : null;
    return demo?.policyGraph?.area || 'Generative_AI';
  }

  function setStatus(text) {
    const el = $('#adjudicateStatusLine');
    if (el) el.textContent = text || '';
  }

  const nz = (value, fallback) => (
    typeof value === 'number' && Number.isFinite(value) ? value : fallback
  );

  // Rank modes. "composite" is the default stack rank he specified:
  // least consensus first, then least confident, then hardest — items with
  // no machine signal at all (all-abstain panels) sort to the very top.
  // "gradient" is the rush.sample_gradient formalism: |g| = 1 - p with
  // p = c if correct else 1 - c, so confident-wrong panels lead.
  const SORTS = {
    composite: (a, b) => (
      (nz(a.agg.consensus_fraction, -1) - nz(b.agg.consensus_fraction, -1))
      || (nz(a.agg.avg_confidence, -1) - nz(b.agg.avg_confidence, -1))
      || (nz(b.agg.difficulty_score, 0) - nz(a.agg.difficulty_score, 0))
      || String(a.image_id).localeCompare(String(b.image_id))
    ),
    gradient: (a, b) => (
      (nz(b.agg.grad_magnitude, -1) - nz(a.agg.grad_magnitude, -1))
      || String(a.image_id).localeCompare(String(b.image_id))
    ),
    loss: (a, b) => (
      (nz(b.agg.loss, -1) - nz(a.agg.loss, -1))
      || String(a.image_id).localeCompare(String(b.image_id))
    ),
    runs: (a, b) => (
      ((b.n_runs || 0) - (a.n_runs || 0))
      || (nz(b.agg.grad_magnitude, 0) - nz(a.agg.grad_magnitude, 0))
      || String(a.image_id).localeCompare(String(b.image_id))
    )
  };

  async function loadQueue() {
    const token = ++state.loadToken;
    setStatus('Loading queue…');
    let payload;
    try {
      payload = await window.rushApiGetJson(
        `/api/adjudication?area=${encodeURIComponent(activeArea())}`
      );
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

  const fmt = (value, digits = 2) => (
    (value === null || value === undefined) ? '—' : Number(value).toFixed(digits)
  );

  function difficultyWord(score) {
    if (score === null || score === undefined) return '—';
    if (score >= 0.75) return `high (${fmt(score)})`;
    if (score >= 0.25) return `medium (${fmt(score)})`;
    return `low (${fmt(score)})`;
  }

  function runChip(run) {
    const where = run.kind === 'train' ? `k${run.k} train`
      : run.kind === 'test' ? 'test'
        : run.kind; // holdout / benchmark
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
      const consensus = run.consensus || {};
      const grad = run.gradient || {};
      const meta = [
        `${esc(run.kind)}${run.k !== null && run.k !== undefined ? ` · k=${esc(run.k)}` : ''}`,
        `policy ${esc(run.policy ?? '—')}`,
        `majority ${esc(run.majority_label ?? (consensus.tie ? 'tie' : '—'))}`,
        `consensus ${esc(consensus.majority_count ?? '—')}/${esc(consensus.decisive ?? '—')}`,
        `avg conf ${fmt(run.avg_confidence)}`,
        `difficulty ${esc(difficultyWord(run.difficulty_score))}`,
        `|g| ${fmt(grad.avg_magnitude)} · loss ${fmt(grad.avg_loss)}`,
        esc(run.misalignment_type ?? '')
      ].join(' · ');
      return `<div class="adjudicate-detail">
        <h5>Run #${esc(run.run_number ?? '?')} <span class="hint">${esc(run.run_id ?? '')}</span></h5>
        <span class="hint">${meta}</span>
        <div>${voteChips(run.votes, item.sme_truth)}</div>
      </div>`;
    }).join('');
    return runs || '<p class="hint">no per-run evidence recorded</p>';
  }

  function render() {
    const host = $('#adjudicateTable');
    const summaryHost = $('#adjudicateSummary');
    if (!host) return;
    if (!state.items) {
      host.innerHTML = '<p class="hint">Loading…</p>';
      return;
    }
    if (summaryHost) {
      const boundary = state.items.filter((it) => it.agg?.any_boundary).length;
      const multi = state.items.filter((it) => (it.n_runs || 0) > 1).length;
      summaryHost.innerHTML = `<div class="experiment-summary-grid">
        <div><span>Queue</span><strong>${state.items.length} item(s)</strong></div>
        <div><span>Flagged by &gt;1 run</span><strong>${multi}</strong></div>
        <div><span>Boundary-flagged</span><strong>${boundary}</strong></div>
      </div>`;
    }
    if (!state.items.length) {
      host.innerHTML = '<p class="hint">Nothing to adjudicate — no completed live run has left misalignments behind (or none has finished since this was added).</p>';
      return;
    }
    const mode = $('#adjudicateSort')?.value || 'composite';
    const items = [...state.items].sort(SORTS[mode] || SORTS.composite);
    const body = items.map((item) => {
      const key = item.key || item.image_id;
      const expanded = state.expanded.has(key);
      const latest = item.latest || {};
      const consensus = latest.consensus || {};
      const img = item.repo_rel_path
        ? `<img src="/api/thumbnail?path=${encodeURIComponent(item.repo_rel_path)}" alt="${esc(item.image_id)}" loading="lazy" />`
        : '';
      const consensusText = consensus.decisive
        ? `${consensus.majority_count}/${consensus.decisive}${consensus.tie ? ' · tie' : ''} (${fmt(item.agg.consensus_fraction)})`
        : 'no decisive votes';
      const row = `<tr>
        <td><button type="button" class="experiment-detail-toggle" data-key="${esc(key)}" aria-expanded="${expanded}">${expanded ? '▾' : '▸'}</button></td>
        <td>${img}<span class="hint">${esc(item.image_id)}<br/>${esc(item.split ?? '')}</span></td>
        <td>${(item.runs || []).map(runChip).join(' ')}</td>
        <td><strong>${esc(item.sme_truth)}</strong></td>
        <td>${esc(latest.majority_label ?? (consensus.tie ? 'tie' : '—'))}</td>
        <td>${esc(consensusText)}</td>
        <td>${fmt(item.agg.avg_confidence)}</td>
        <td>${esc(difficultyWord(item.agg.difficulty_score))}</td>
        <td>${fmt(item.agg.grad_magnitude)}${item.agg.any_boundary ? ' <span class="summary-flag summary-flag--boundary" title="at least one judge flagged a boundary case">⧉</span>' : ''}</td>
      </tr>`;
      const detail = expanded
        ? `<tr><td colspan="9">${detailBlock(item)}</td></tr>`
        : '';
      return row + detail;
    }).join('');
    host.innerHTML = `
      <table class="summary-table">
        <thead><tr>
          <th></th><th>Image</th><th>Flagged by</th><th>SME truth</th>
          <th title="majority label in the most recent flagging run">Majority</th>
          <th title="majority count / decisive votes (cross-run avg fraction)">Consensus</th>
          <th title="average self-reported confidence across judges and runs">Avg conf</th>
          <th title="low=0, medium=0.5, high=1 averaged across judges and runs">Difficulty</th>
          <th title="gradient magnitude |g| = 1 - p, p = confidence if correct else 1 - confidence; confident-wrong ≈ 1">|g|</th>
        </tr></thead>
        <tbody>${body}</tbody>
      </table>`;
    host.querySelectorAll('.experiment-detail-toggle').forEach((button) => {
      button.addEventListener('click', () => {
        const key = button.dataset.key;
        if (state.expanded.has(key)) state.expanded.delete(key);
        else state.expanded.add(key);
        render();
      });
    });
  }

  function init() {
    if (!$('#adjudicate')) return;
    $('#adjudicateSort')?.addEventListener('change', render);
    $('#adjudicateRefresh')?.addEventListener('click', () => loadQueue());
    window.addEventListener('rush-api-catalog', () => { if (state.loaded) loadQueue(); });
    window.addEventListener('rush-view-changed', (event) => {
      if (event.detail?.view === 'adjudicate' && !state.loaded) loadQueue();
    });
    if (document.body.classList.contains('view-adjudicate')) loadQueue();
  }

  if (typeof window.rushApiOnReady === 'function') {
    window.rushApiOnReady(() => init());
  } else {
    document.addEventListener('DOMContentLoaded', init);
  }
})();
