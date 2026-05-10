(() => {
  const DEFAULT_POLICY_MODEL = 'anthropic/claude-opus-4-7';
  const state = { proposals: [], selected: '' };

  function status(message, isError = false) {
    rushApiStatus('#proposalStatus', message, isError);
  }

  function setUnavailable() {
    rushApiUnavailable('#policy-proposals');
    $('#proposalSummary').innerHTML = '';
    $('#proposalDiffViewer').innerHTML = '';
  }

  function proposalLabel(proposal) {
    const id = proposal.proposal_id || 'unknown';
    const statusText = proposal.status || 'pending';
    const version = proposal.base_version || 'base ?';
    return `${id} · ${statusText} · ${version}`;
  }

  function populateControls() {
    const currentRun = $('#proposalRunId')?.value || window.RUSH_API?.catalog?.runs?.[0]?.run_id || '';
    const runSelect = $('#proposalRunId');
    if (runSelect) {
      runSelect.innerHTML = rushApiRunOptions(currentRun, false);
      if (currentRun) runSelect.value = currentRun;
    }
    const currentVersion = window.RUSH_API?.catalog?.currentPolicyVersion || '';
    for (const id of ['proposalBaseVersion', 'proposalBuildVersion']) {
      const select = $(`#${id}`);
      if (!select) continue;
      const selected = select.value || currentVersion;
      select.innerHTML = rushApiPolicyVersionOptions(selected, false);
      if (selected) select.value = selected;
    }
  }

  function populateProposalPicker() {
    const select = $('#proposalPicker');
    if (!select) return;
    if (!state.proposals.length) {
      select.innerHTML = rushApiOptionHtml('', 'No proposals found', true);
      state.selected = '';
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
      return;
    }
    const diffs = Array.isArray(payload.diffs) ? payload.diffs : [];
    const cards = [
      ['Proposal', payload.proposal_id || '—', payload.status || '—'],
      ['Base version', payload.base_version || '—', payload.model_id || DEFAULT_POLICY_MODEL],
      ['Changed', payload.files_changed?.length ?? diffs.filter(d => d.change === 'modified').length, (payload.files_changed || []).join(' · ')],
      ['Added / removed', `${payload.files_added?.length ?? 0} / ${payload.files_removed?.length ?? 0}`, [...(payload.files_added || []), ...(payload.files_removed || [])].join(' · ')]
    ];
    summary.innerHTML = cards.map(([label, value, note]) => `<article class="stat-card"><span>${esc(label)}</span><strong>${esc(value)}</strong><p>${esc(note || '')}</p></article>`).join('');
    if (!diffs.length) {
      viewer.innerHTML = '<div class="empty-state">This proposal has no diff records.</div>';
      return;
    }
    viewer.innerHTML = diffs.map(diff => `<article class="diff-file-card"><h3>${esc(diff.path || 'unknown file')} <span>${esc(diff.change || '')}</span></h3>${renderUnifiedDiff(diff.unified_diff || '')}</article>`).join('');
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
      renderProposal(payload);
      status(`Loaded proposal ${state.selected}.`);
    } catch (error) {
      $('#proposalDiffViewer').innerHTML = `<div class="empty-state">${esc(error.message)}</div>`;
      status(`Proposal failed: ${error.message}`, true);
    }
  }

  async function proposeDiff() {
    const runId = $('#proposalRunId')?.value || '';
    const baseVersion = $('#proposalBaseVersion')?.value || window.RUSH_API?.catalog?.currentPolicyVersion || 'v0.1';
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
    try {
      status(`Accepting ${proposalId}…`);
      $('#acceptProposal').disabled = true;
      const payload = await rushApiPostJson(`/api/policy/proposals/${encodeURIComponent(proposalId)}/accept`, {});
      status(`Accepted into ${payload.new_version || 'new version'}.`);
      await rushApiLoadCatalog();
      await loadProposals(false);
    } catch (error) {
      status(`Accept failed: ${error.message}`, true);
    } finally {
      $('#acceptProposal').disabled = false;
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
      $('#rejectProposal').disabled = false;
    }
  }

  async function buildPdf() {
    const version = $('#proposalBuildVersion')?.value || window.RUSH_API?.catalog?.currentPolicyVersion || 'v0.1';
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
    await loadProposals(true);
  }

  rushApiOnReady(initPolicyDiff);
  window.addEventListener('rush-api-catalog', populateControls);
})();
