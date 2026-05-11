// Per-image evidence drawer. Loaded from index.html after policy-graph.js.
(() => {
  const DRAWER_ID = 'justificationsDrawer';
  const STYLE_ID = 'justificationsDrawerStyles';
  const DEFAULT_POLICY_MODEL = 'openai/gpt-5.5';

  const esc = value => String(value ?? '').replace(/[&<>'"]/g, char => ({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}[char]));
  const attr = esc;
  const isNumber = value => typeof value === 'number' && Number.isFinite(value);

  function injectStyles() {
    if (document.getElementById(STYLE_ID)) return;
    const style = document.createElement('style');
    style.id = STYLE_ID;
    style.textContent = `
      .justifications-backdrop { position: fixed; inset: 0; z-index: 99; background: rgba(2, 6, 23, 0.45); backdrop-filter: blur(2px); }
      .justifications-drawer { position: fixed; top: 0; right: 0; bottom: 0; z-index: 100; width: min(480px, 94vw); overflow-y: auto; background: #0f172a; color: #e5edf8; border-left: 1px solid rgba(148, 163, 184, 0.28); box-shadow: -28px 0 60px rgba(0, 0, 0, 0.45); padding: 18px; }
      .justifications-head { display: flex; align-items: flex-start; justify-content: space-between; gap: 12px; margin-bottom: 14px; }
      .justifications-head h2 { margin: 0; font-size: 1.1rem; }
      .justifications-head code, .justifications-drawer code { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; color: #bfdbfe; }
      .justifications-close { border: 1px solid rgba(148, 163, 184, 0.4); background: rgba(15, 23, 42, 0.8); color: #e5edf8; border-radius: 10px; padding: 6px 10px; cursor: pointer; }
      .justifications-image { width: 100%; display: grid; place-items: center; border: 1px solid rgba(148, 163, 184, 0.22); border-radius: 16px; background: rgba(2, 6, 23, 0.45); min-height: 120px; margin-bottom: 14px; overflow: hidden; }
      .justifications-image img { max-width: 320px; max-height: 320px; width: auto; height: auto; display: block; }
      .justifications-meta { display: grid; gap: 8px; margin-bottom: 14px; }
      .justifications-pill { display: inline-flex; align-items: center; width: fit-content; gap: 6px; border: 1px solid rgba(148, 163, 184, 0.25); border-radius: 999px; padding: 4px 9px; background: rgba(30, 41, 59, 0.78); font-size: 0.8rem; color: #dbeafe; }
      .justifications-cost { display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; margin: 12px 0; }
      .justifications-cost div, .justifications-vote { border: 1px solid rgba(148, 163, 184, 0.2); border-radius: 14px; background: rgba(15, 23, 42, 0.72); padding: 10px; }
      .justifications-cost span, .justifications-vote span { display: block; color: #94a3b8; font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.06em; }
      .justifications-cost strong { display: block; margin-top: 4px; color: #f8fafc; }
      .justifications-votes { display: grid; gap: 10px; margin-top: 12px; }
      .justifications-vote header { display: flex; justify-content: space-between; gap: 12px; align-items: baseline; margin-bottom: 8px; }
      .justifications-vote h3 { margin: 0; font-size: 0.95rem; color: #f8fafc; }
      .justifications-vote p { margin: 8px 0 0; line-height: 1.45; color: #dbeafe; }
      .justifications-vote dl { display: grid; grid-template-columns: 92px 1fr; gap: 4px 8px; margin: 0; font-size: 0.85rem; }
      .justifications-vote dt { color: #94a3b8; }
      .justifications-vote dd { margin: 0; color: #e5edf8; }
      .justifications-actions { display: flex; gap: 8px; align-items: center; margin: 14px 0; flex-wrap: wrap; }
      .justifications-actions button { border: 0; border-radius: 12px; padding: 9px 12px; cursor: pointer; color: #06121f; background: #93c5fd; font-weight: 700; }
      .justifications-actions button:disabled { opacity: 0.55; cursor: not-allowed; }
      .justifications-status { color: #bfdbfe; font-size: 0.82rem; }
      .justifications-muted { color: #94a3b8; }
    `;
    document.head.appendChild(style);
  }

  function recordsFromBorderline(groups) {
    if (Array.isArray(groups)) return groups.flatMap(group => Array.isArray(group?.items) ? group.items : []);
    if (groups && typeof groups === 'object') return Object.values(groups).flatMap(items => Array.isArray(items) ? items : []);
    return [];
  }

  function findRecord(imageId) {
    const state = window.runState || {};
    const id = String(imageId || '');
    const candidates = [];
    candidates.push(...(Array.isArray(state.misalignment?.rows) ? state.misalignment.rows : []));
    candidates.push(...(Array.isArray(state.misalignment?.records) ? state.misalignment.records : []));
    candidates.push(...recordsFromBorderline(state.borderline?.groups));
    candidates.push(...(Array.isArray(state.consensus?.records) ? state.consensus.records : []));

    const merged = {};
    for (const record of candidates) {
      const recordId = record?.image_id || record?.sample_id;
      if (String(recordId || '') !== id) continue;
      Object.assign(merged, record);
      if (Array.isArray(record.votes) && record.votes.length) merged.votes = record.votes;
      if (!merged.votes && Array.isArray(record.voters) && record.voters.length) merged.votes = record.voters;
      if (!merged.repo_rel_path && record.repo_rel_path) merged.repo_rel_path = record.repo_rel_path;
      if (!merged.sme_truth && (record.sme_truth || record.truth)) merged.sme_truth = record.sme_truth || record.truth;
    }
    return Object.keys(merged).length ? merged : { image_id: id };
  }

  function sourceImageSrc(record) {
    const path = String(record.repo_rel_path || record.synthetic_repo_rel_path || '').replace(/^\.\//, '').replace(/^\/+/, '');
    if (!path) return '';
    return window.RUSH_API?.available ? `/api/thumbnail?path=${encodeURIComponent(path)}` : `../${path}`;
  }

  function tokenValue(vote, ...keys) {
    for (const key of keys) {
      const value = vote?.[key];
      if (isNumber(value)) return value;
      const coerced = Number(value);
      if (Number.isFinite(coerced)) return coerced;
    }
    return null;
  }

  function voteStats(vote) {
    const input = tokenValue(vote, 'input_tokens', 'prompt_tokens');
    const output = tokenValue(vote, 'output_tokens', 'completion_tokens');
    const total = tokenValue(vote, 'total_tokens') ?? ((input ?? 0) + (output ?? 0) || null);
    const cost = tokenValue(vote, 'cost_usd', 'total_cost_usd');
    return { input, output, total, cost };
  }

  function formatCost(value) {
    return isNumber(value) ? `$${value.toFixed(5)}` : '—';
  }

  function renderVote(vote) {
    const stats = voteStats(vote);
    const model = vote.labeler_id || vote.model_id || 'unknown model';
    const conf = isNumber(vote.confidence) ? vote.confidence.toFixed(2) : '—';
    const boundary = vote.is_boundary ? 'yes' : (vote.is_boundary === false ? 'no' : '—');
    return `<article class="justifications-vote">
      <header><h3>${esc(model)}</h3><span>${formatCost(stats.cost)}</span></header>
      <dl>
        <dt>label</dt><dd>${esc(vote.label || '—')}</dd>
        <dt>confidence</dt><dd>${esc(conf)}</dd>
        <dt>l2_label</dt><dd><code>${esc(vote.l2_label || '—')}</code></dd>
        <dt>boundary</dt><dd>${esc(boundary)}</dd>
        <dt>tokens</dt><dd>${esc(stats.total ?? '—')} total (${esc(stats.input ?? '—')} in / ${esc(stats.output ?? '—')} out)</dd>
      </dl>
      <p>${esc(vote.justification || 'No justification text available for this vote.')}</p>
    </article>`;
  }

  function totals(votes) {
    return votes.reduce((acc, vote) => {
      const stats = voteStats(vote);
      acc.input += stats.input ?? 0;
      acc.output += stats.output ?? 0;
      acc.cost += stats.cost ?? 0;
      return acc;
    }, { input: 0, output: 0, cost: 0 });
  }

  function renderDrawer(record) {
    injectStyles();
    closeDrawer();
    const imageId = record.image_id || record.sample_id || '';
    const votes = Array.isArray(record.votes) ? record.votes : (Array.isArray(record.voters) ? record.voters : []);
    const total = totals(votes);
    const image = sourceImageSrc(record);
    const runId = window.runState?.selectedRunId || record.run_id || '';

    const backdrop = document.createElement('div');
    backdrop.className = 'justifications-backdrop';
    backdrop.dataset.justificationsClose = 'true';

    const drawer = document.createElement('aside');
    drawer.id = DRAWER_ID;
    drawer.className = 'justifications-drawer';
    drawer.setAttribute('role', 'dialog');
    drawer.setAttribute('aria-modal', 'true');
    drawer.setAttribute('aria-label', `Evidence for ${imageId}`);
    drawer.innerHTML = `
      <div class="justifications-head">
        <div><span class="justifications-pill">image evidence</span><h2><code>${esc(imageId)}</code></h2></div>
        <button class="justifications-close" type="button" data-justifications-close="true" aria-label="Close">×</button>
      </div>
      <div class="justifications-image">${image ? `<img src="${attr(image)}" alt="${attr(imageId)}" loading="lazy" decoding="async" />` : '<span class="justifications-muted">No source image path available.</span>'}</div>
      <div class="justifications-meta">
        <div><span class="justifications-pill">SME truth: ${esc(record.sme_truth || record.truth || '—')}</span></div>
        <div><span class="justifications-pill">run: <code>${esc(runId || '—')}</code></span></div>
        ${record.repo_rel_path ? `<code>${esc(record.repo_rel_path)}</code>` : ''}
      </div>
      <section class="justifications-cost" aria-label="Token and cost breakdown">
        <div><span>input tokens</span><strong>${esc(total.input || '—')}</strong></div>
        <div><span>output tokens</span><strong>${esc(total.output || '—')}</strong></div>
        <div><span>total cost</span><strong>${formatCost(total.cost || null)}</strong></div>
      </section>
      <div class="justifications-actions">
        <button type="button" data-propose-row-diff="${attr(imageId)}" ${runId ? '' : 'disabled'}>Propose diff from this row</button>
        <span class="justifications-status" id="justificationsStatus"></span>
      </div>
      <section class="justifications-votes">
        ${votes.length ? votes.map(renderVote).join('') : '<div class="justifications-vote">No per-model vote details available for this row.</div>'}
      </section>
    `;
    document.body.append(backdrop, drawer);
    drawer.querySelector('.justifications-close')?.focus();
  }

  function closeDrawer() {
    document.getElementById(DRAWER_ID)?.remove();
    document.querySelector('.justifications-backdrop')?.remove();
  }

  function setStatus(message, isError = false) {
    const el = document.getElementById('justificationsStatus');
    if (!el) return;
    el.style.color = isError ? '#fca5a5' : '#bfdbfe';
    el.textContent = message || '';
  }

  async function proposeDiffFromRow(button) {
    const imageId = button.dataset.proposeRowDiff || '';
    const runId = window.runState?.selectedRunId || '';
    if (!runId) return setStatus('No run selected.', true);
    if (!window.RUSH_API?.available || typeof window.rushApiPostJson !== 'function') {
      return setStatus('Local API is not available.', true);
    }
    const baseVersion = window.RUSH_API?.catalog?.currentPolicyVersion || 'v0.1';
    button.disabled = true;
    setStatus('Requesting GPT-5.5 policy diff…');
    try {
      const payload = await window.rushApiPostJson('/api/policy/propose-diff', {
        run_id: runId,
        base_version: baseVersion,
        model_id: DEFAULT_POLICY_MODEL,
        image_id: imageId
      });
      setStatus(`Created proposal ${payload.proposal_id || 'unknown'}.`);
      window.dispatchEvent(new CustomEvent('rush-policy-proposal-created', { detail: payload }));
    } catch (error) {
      setStatus(`Proposal failed: ${error.message}`, true);
    } finally {
      button.disabled = false;
    }
  }

  document.addEventListener('click', event => {
    const target = event.target;
    if (!(target instanceof Element)) return;
    const close = target.closest('[data-justifications-close]');
    if (close) {
      closeDrawer();
      return;
    }
    const proposeButton = target.closest('[data-propose-row-diff]');
    if (proposeButton) {
      proposeDiffFromRow(proposeButton);
      return;
    }
    const explicitOpen = target.closest('[data-open-justifications]');
    if (explicitOpen) {
      renderDrawer(findRecord(explicitOpen.dataset.openJustifications));
      return;
    }
    if (target.closest('a, button, input, select, textarea')) return;
    const row = target.closest('#misalignmentTable [data-image-id], #consensusTable [data-image-id], #borderlineGroups [data-image-id]');
    if (!row) return;
    renderDrawer(findRecord(row.dataset.imageId));
  });

  document.addEventListener('keydown', event => {
    if (event.key === 'Escape') closeDrawer();
  });
})();
