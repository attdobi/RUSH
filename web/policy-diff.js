(() => {
  const DEFAULT_POLICY_MODEL = 'openai/gpt-5.5';
  const state = { proposals: [], selected: '', lastLoadedProposal: null };

  function status(message, isError = false) {
    rushApiStatus('#proposalStatus', message, isError);
  }

  function setUnavailable() {
    rushApiUnavailable('#grow');
    $('#proposalSummary').innerHTML = '';
    $('#proposalDiffViewer').innerHTML = '';
  }

  function proposalLabel(proposal) {
    const id = proposal.proposal_id || 'unknown';
    const statusText = proposal.status || 'pending';
    const version = proposal.base_version || 'base ?';
    return `${id} · ${statusText} · ${version}`;
  }

  function syncProposalActions(payload = state.lastLoadedProposal) {
    const proposalId = $('#proposalPicker')?.value || state.selected || '';
    const statusText = String(payload?.status || '').toLowerCase();
    const hasProposal = !!proposalId && !!payload;
    const canAccept = hasProposal && statusText === 'pending';
    const canReject = hasProposal && ['pending', 'parse_error'].includes(statusText);
    const accept = $('#acceptProposal');
    const reject = $('#rejectProposal');
    if (accept) {
      accept.disabled = !canAccept;
      accept.title = canAccept ? 'Accept this pending proposal' : 'Only pending proposals can be accepted';
    }
    if (reject) {
      reject.disabled = !canReject;
      reject.title = canReject ? 'Reject this proposal' : 'Only pending or parse-error proposals can be rejected';
    }
  }

  function populateControls() {
    const currentRun = $('#proposalRunId')?.value || window.RUSH_API?.catalog?.runs?.[0]?.run_id || '';
    const runSelect = $('#proposalRunId');
    if (runSelect) {
      runSelect.innerHTML = rushApiRunOptions(currentRun, false);
      if (currentRun) runSelect.value = currentRun;
    }
    const currentVersion = window.RUSH_API?.catalog?.currentPolicyVersion || '';
    for (const id of ['proposalBaseVersionChip', 'proposalBuildVersionChip']) {
      const chip = $(`#${id}`);
      if (chip) chip.textContent = `${currentVersion} · current`;
    }
  }

  function populateProposalPicker() {
    const select = $('#proposalPicker');
    if (!select) return;
    if (!state.proposals.length) {
      select.innerHTML = rushApiOptionHtml('', 'No proposals found', true);
      state.selected = '';
      syncProposalActions(null);
      return;
    }
    const selected = state.selected || state.proposals[0].proposal_id;
    select.innerHTML = state.proposals.map(proposal => rushApiOptionHtml(proposal.proposal_id || '', proposalLabel(proposal), selected === proposal.proposal_id)).join('');
    select.value = selected;
    state.selected = selected;
  }

  function diffLineClass(line) {
    if (line.startsWith('+++') || line.startsWith('---')) return 'diff-file';
    if (line.startsWith('@@')) return 'diff-hunk';
    if (line.startsWith('+')) return 'diff-add';
    if (line.startsWith('-')) return 'diff-del';
    return 'diff-context';
  }

  function renderUnifiedDiff(diffText) {
    const lines = String(diffText || '').split('\n');
    return `<pre class="policy-diff-pre">${lines.map(line => `<span class="${diffLineClass(line)}">${esc(line || ' ')}</span>`).join('\n')}</pre>`;
  }

  function renderProposal(payload) {
    const summary = $('#proposalSummary');
    const viewer = $('#proposalDiffViewer');
    if (!summary || !viewer) return;
    if (!payload) {
      summary.innerHTML = '';
      viewer.innerHTML = '<div class="empty-state">Select a proposal to view its diff.</div>';
      syncProposalActions(null);
      return;
    }
    syncProposalActions(payload);
    const diffs = Array.isArray(payload.diffs) ? payload.diffs : [];
    const cards = [
      ['Proposal', payload.proposal_id || '—', payload.status || '—'],
      ['Base version', payload.base_version || '—', payload.model_id || DEFAULT_POLICY_MODEL],
      ['Draft model', payload.model_id || DEFAULT_POLICY_MODEL, `Drafted by ${payload.model_id || DEFAULT_POLICY_MODEL} (high reasoning)`],
      ['Changed', payload.files_changed?.length ?? diffs.filter(d => d.change === 'modified').length, (payload.files_changed || []).join(' · ')],
      ['Added / removed', `${payload.files_added?.length ?? 0} / ${payload.files_removed?.length ?? 0}`, [...(payload.files_added || []), ...(payload.files_removed || [])].join(' · ')]
    ];
    summary.innerHTML = cards.map(([label, value, note]) => `<article class="stat-card"><span>${esc(label)}</span><strong>${esc(value)}</strong><p>${esc(note || '')}</p></article>`).join('');
    const statusText = payload.status || 'pending';
    const statusBanner = `<div class="proposal-state-banner"><strong>${esc(statusText)}</strong><span>${esc(statusText === 'pending' ? 'SME action needed before this graph changes.' : 'Proposal state is recorded; only pending proposals can change the graph.')}</span></div>`;
    if (!diffs.length) {
      viewer.innerHTML = `${statusBanner}<div class="empty-state">This proposal has no diff records.</div>`;
      return;
    }
    viewer.innerHTML = statusBanner + diffs.map(diff => `<article class="diff-file-card"><h3>${esc(diff.path || 'unknown file')} <span>${esc(diff.change || '')}</span></h3>${renderUnifiedDiff(diff.unified_diff || '')}</article>`).join('');
  }

  async function loadProposals(selectFirst = true) {
    if (!window.RUSH_API?.available) {
      setUnavailable();
      return;
    }
    try {
      status('Loading proposals…');
      const payload = await rushApiGetJson('/api/policy/proposals');
      state.proposals = Array.isArray(payload.proposals) ? payload.proposals : [];
      if (selectFirst && !state.selected && state.proposals.length) state.selected = state.proposals[0].proposal_id;
      populateProposalPicker();
      if (state.selected) await loadProposal(state.selected);
      else renderProposal(null);
      status(`Loaded ${state.proposals.length} proposal(s).`);
    } catch (error) {
      $('#proposalDiffViewer').innerHTML = `<div class="empty-state">${esc(error.message)}</div>`;
      syncProposalActions(null);
      status(`Proposal list failed: ${error.message}`, true);
    }
  }

  async function loadProposal(proposalId) {
    state.selected = proposalId || '';
    if (!state.selected) {
      renderProposal(null);
      return;
    }
    try {
      status(`Loading proposal ${state.selected}…`);
      const payload = await rushApiGetJson(`/api/policy/proposals/${encodeURIComponent(state.selected)}`);
      state.lastLoadedProposal = payload;
      renderProposal(payload);
      status(`Loaded proposal ${state.selected}.`);
    } catch (error) {
      $('#proposalDiffViewer').innerHTML = `<div class="empty-state">${esc(error.message)}</div>`;
      syncProposalActions(null);
      status(`Proposal failed: ${error.message}`, true);
    }
  }

  async function proposeDiff() {
    const runId = $('#proposalRunId')?.value || '';
    const baseVersion = window.RUSH_API?.catalog?.currentPolicyVersion || 'v0.1';
    if (!runId) {
      status('Select a run before proposing a diff.', true);
      return;
    }
    try {
      status(`Proposing diff from ${runId}…`);
      $('#proposeDiff').disabled = true;
      const payload = await rushApiPostJson('/api/policy/propose-diff', {
        run_id: runId,
        base_version: baseVersion,
        model_id: DEFAULT_POLICY_MODEL
      });
      state.selected = payload.proposal_id || '';
      await loadProposals(false);
      if (state.selected) await loadProposal(state.selected);
      status(`Created proposal ${payload.proposal_id || 'unknown'}.`);
    } catch (error) {
      status(`Propose diff failed: ${error.message}`, true);
    } finally {
      $('#proposeDiff').disabled = false;
    }
  }

  async function acceptProposal() {
    const proposalId = $('#proposalPicker')?.value || state.selected;
    if (!proposalId) return status('Select a proposal to accept.', true);
    const proposal = state.lastLoadedProposal?.proposal_id === proposalId ? state.lastLoadedProposal : null;
    try {
      status(`Accepting ${proposalId}…`);
      $('#acceptProposal').disabled = true;
      const payload = await rushApiPostJson(`/api/policy/proposals/${encodeURIComponent(proposalId)}/accept`, {});
      status(`Accepted into ${payload.new_version || 'new version'}.`);
      window.dispatchEvent(new CustomEvent('rush-policy-accepted', {
        detail: {
          new_version: payload.new_version,
          files_added: Array.isArray(proposal?.files_added) ? proposal.files_added : [],
          files_changed: Array.isArray(proposal?.files_changed) ? proposal.files_changed : []
        }
      }));
      await rushApiLoadCatalog();
      await loadProposals(false);
    } catch (error) {
      status(`Accept failed: ${error.message}`, true);
    } finally {
      syncProposalActions(state.lastLoadedProposal);
    }
  }

  async function rejectProposal() {
    const proposalId = $('#proposalPicker')?.value || state.selected;
    if (!proposalId) return status('Select a proposal to reject.', true);
    try {
      status(`Rejecting ${proposalId}…`);
      $('#rejectProposal').disabled = true;
      await rushApiPostJson(`/api/policy/proposals/${encodeURIComponent(proposalId)}/reject`, {});
      status(`Rejected ${proposalId}.`);
      state.selected = '';
      await loadProposals(true);
    } catch (error) {
      status(`Reject failed: ${error.message}`, true);
    } finally {
      syncProposalActions(state.lastLoadedProposal);
    }
  }

  async function buildPdf() {
    const version = window.RUSH_API?.catalog?.currentPolicyVersion || 'v0.1';
    try {
      status(`Building policy PDF for ${version}…`);
      $('#buildPolicyPdf').disabled = true;
      const payload = await rushApiPostJson('/api/policy/build-pdf', { version, model_id: DEFAULT_POLICY_MODEL });
      status(`Built PDF: ${payload.output_path || payload.path || version}.`);
    } catch (error) {
      status(`Build PDF failed: ${error.message}`, true);
    } finally {
      $('#buildPolicyPdf').disabled = false;
    }
  }

  function bind() {
    $('#proposalPicker')?.addEventListener('change', event => loadProposal(event.target.value));
    $('#proposeDiff')?.addEventListener('click', proposeDiff);
    $('#acceptProposal')?.addEventListener('click', acceptProposal);
    $('#rejectProposal')?.addEventListener('click', rejectProposal);
    $('#buildPolicyPdf')?.addEventListener('click', buildPdf);
  }

  async function initPolicyDiff(api) {
    if (!api.available) {
      setUnavailable();
      return;
    }
    await rushApiLoadCatalog();
    populateControls();
    bind();
    syncProposalActions(null);
    await loadProposals(true);
  }

  rushApiOnReady(initPolicyDiff);
  window.addEventListener('rush-api-catalog', populateControls);
  window.rushLoadProposal = async id => {
    state.selected = id || '';
    await loadProposal(state.selected);
  };
  window.rushRefreshPolicyProposals = async selectedId => {
    if (selectedId) state.selected = selectedId;
    await loadProposals(false);
  };
})();
