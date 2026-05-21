(() => {
  const DEFAULT_POLICY_MODEL = 'openai/gpt-5.5';
  const state = { proposals: [], selected: '', lastLoadedProposal: null };

  const STATUS_THEMES = {
    idle: { label: 'IDLE', color: '#94a3b8' },
    loading: { label: 'LOADING', color: '#38bdf8' },
    building: { label: 'BUILDING', color: '#f59e0b' },
    building_retry: { label: 'BUILDING', color: '#f97316' },
    success: { label: 'SUCCESS', color: '#22c55e' },
    parse_error: { label: 'PARSE ERROR', color: '#fb7185' },
    failed: { label: 'FAILED', color: '#ef4444' }
  };

  function status(message, state = 'idle', details = {}) {
    const legacyError = state === true;
    const statusState = legacyError ? 'failed' : (state || 'idle');
    const theme = STATUS_THEMES[statusState] || STATUS_THEMES.idle;
    const retry = details.retry_count ? ` retry ${details.retry_count}/${details.max_retries || '?'}` : '';
    const spinner = ['loading', 'building', 'building_retry'].includes(statusState) ? '⏳ ' : '';
    const text = `${spinner}[${theme.label}${retry}] ${message || ''}`.trim();
    const isError = legacyError || ['failed', 'parse_error'].includes(statusState);
    rushApiStatus('#proposalStatus', text, isError);
    const el = $('#proposalStatus');
    if (!el) return;
    el.dataset.state = statusState;
    el.style.color = theme.color;
    el.style.fontWeight = statusState === 'idle' ? '' : '700';
  }

  function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
  }

  function jobStatusMessage(job) {
    const reason = job?.error || job?.message || '';
    if (job?.status === 'building_retry' || (job?.status === 'building' && job?.retry_count)) {
      return `Generating proposal… retry ${job.retry_count || 0}/${job.max_retries || '?'} after a timeout.`;
    }
    if (job?.status === 'queued') return 'Queued proposal generation…';
    if (job?.status === 'building') return 'Generating proposal… (still working, this can take a minute)';
    if (job?.status === 'parse_error') return `Model returned malformed JSON${reason ? `: ${reason}` : '.'}`;
    if (job?.status === 'failed') return `Proposal generation failed${reason ? `: ${reason}` : '.'}`;
    if (job?.status === 'success') return `Created proposal ${job.proposal_id || job.result?.proposal_id || 'unknown'}.`;
    return job?.message || 'Generating proposal…';
  }

  function renderJobStatus(job) {
    const stateName = job?.status || 'building';
    status(jobStatusMessage(job), stateName, {
      retry_count: job?.retry_count || 0,
      max_retries: job?.max_retries || 0
    });
  }

  async function waitForProposalJob(initialJob) {
    let job = initialJob;
    renderJobStatus(job);
    for (let polls = 0; polls < 240; polls += 1) {
      if (['success', 'parse_error', 'failed'].includes(job?.status)) return job;
      await sleep(1500);
      job = await rushApiGetJson(job.status_url || `/api/policy/propose-diff/jobs/${encodeURIComponent(job.job_id)}`);
      renderJobStatus(job);
    }
    throw new Error('proposal job is still running after several minutes');
  }

  function setUnavailable() {
    rushApiUnavailable('#grow');
    $('#proposalSummary').innerHTML = '';
    $('#proposalDiffViewer').innerHTML = '';
  }

  function proposalStatusKey(statusText) {
    const status = String(statusText || 'pending').toLowerCase();
    if (status.includes('accept')) return 'accepted';
    if (status.includes('reject')) return 'rejected';
    if (status.includes('parse') || status.includes('error') || status.includes('fail')) return 'parse_error';
    return 'pending';
  }

  function proposalStatusClass(statusText) {
    return proposalStatusKey(statusText);
  }

  function proposalStatusLabel(statusText, count = null) {
    const key = proposalStatusKey(statusText);
    const labels = {
      pending: 'pending',
      accepted: 'accepted',
      rejected: 'rejected',
      parse_error: 'parse error'
    };
    if (count != null) {
      return key === 'parse_error'
        ? `${count} parse error${count === 1 ? '' : 's'}`
        : `${count} ${labels[key]} proposal${count === 1 ? '' : 's'}`;
    }
    return labels[key] || String(statusText || 'pending');
  }

  function proposalLabel(proposal) {
    const id = proposal.proposal_id || 'unknown';
    const statusText = proposal.status || 'pending';
    const version = proposal.base_version || 'base ?';
    return `${statusText.toUpperCase()} · ${version} · ${id}`;
  }

  function isParseErrorProposal(proposal) {
    return proposalStatusKey(proposal?.status) === 'parse_error';
  }

  function proposalTime(proposal) {
    return Date.parse(proposal.accepted_at || proposal.updated_at || proposal.created_at || '') || 0;
  }

  function proposalVersionText(proposal) {
    const base = proposal.base_version || 'base ?';
    if (proposal.accepted_into_version) return `${base} → ${proposal.accepted_into_version}`;
    if (proposal.build_version) return `${base} → ${proposal.build_version}`;
    return `${base} · build pending`;
  }

  function sortProposalList(proposals) {
    return [...proposals].sort((a, b) => proposalTime(b) - proposalTime(a) || String(b.proposal_id || '').localeCompare(String(a.proposal_id || '')));
  }

  function proposalStatusGroups(proposals) {
    const groups = proposals.reduce((acc, proposal) => {
      const key = proposalStatusKey(proposal.status);
      (acc[key] ||= []).push(proposal);
      return acc;
    }, {});
    const priority = ['pending', 'accepted', 'rejected'];
    return [
      ...priority.filter(statusText => groups[statusText]?.length),
      ...Object.keys(groups).filter(statusText => !priority.includes(statusText))
    ].map(statusText => [statusText, sortProposalList(groups[statusText])]);
  }

  function orderedProposalList() {
    const reviewable = state.proposals.filter(proposal => !isParseErrorProposal(proposal));
    const parseErrors = sortProposalList(state.proposals.filter(isParseErrorProposal));
    return [
      ...proposalStatusGroups(reviewable).flatMap(([, proposals]) => proposals),
      ...parseErrors
    ];
  }

  function defaultProposalId() {
    const ordered = orderedProposalList();
    return ordered[0]?.proposal_id || '';
  }

  function renderProposalCard(proposal) {
    const selected = proposal.proposal_id === state.selected;
    const statusKey = proposalStatusKey(proposal.status);
    return `<button type="button" class="proposal-card proposal-card-${statusKey}${selected ? ' selected' : ''}" data-proposal-id="${attr(proposal.proposal_id || '')}">
      <span class="proposal-status-chip ${statusKey}">${esc(proposalStatusLabel(proposal.status))}</span>
      <strong>${esc(proposal.proposal_id || 'unknown')}</strong>
      <small>${esc(proposalVersionText(proposal))}</small>
    </button>`;
  }

  function renderProposalGroup(statusText, proposals) {
    const statusKey = proposalStatusKey(statusText);
    return `<section class="proposal-status-group proposal-status-group-${statusKey}">
      <h4><span class="proposal-status-chip ${statusKey}">${esc(proposalStatusLabel(statusText, proposals.length))}</span></h4>
      <div class="proposal-card-list">${proposals.map(renderProposalCard).join('')}</div>
    </section>`;
  }

  function renderProposalCards() {
    const target = $('#proposalGroupedList');
    if (!target) return;
    if (!state.proposals.length) {
      target.innerHTML = '<div class="empty-state compact-empty">No policy proposals yet. Pick a scored run, then suggest changes.</div>';
      return;
    }
    const reviewable = state.proposals.filter(proposal => !isParseErrorProposal(proposal));
    const parseErrors = state.proposals.filter(isParseErrorProposal);
    const reviewGroups = proposalStatusGroups(reviewable).map(([statusText, proposals]) => renderProposalGroup(statusText, proposals));
    const parseErrorDetails = parseErrors.length ? `<details class="proposal-status-group proposal-parse-errors">
      <summary>Parse errors (${parseErrors.length})</summary>
      <div class="proposal-card-list">${parseErrors.map(renderProposalCard).join('')}</div>
    </details>` : '';
    target.innerHTML = [
      ...reviewGroups,
      parseErrorDetails
    ].filter(Boolean).join('') || '<div class="empty-state compact-empty">Only parse-error proposals are available. Expand parse errors below to inspect them.</div>';
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

  function syncProposalVersionChips(payload = state.lastLoadedProposal) {
    const baseChip = $('#proposalBaseVersionChip');
    const buildChip = $('#proposalBuildVersionChip');
    const buildLabel = $('#proposalBuildVersionLabel');
    const arrow = $('#proposalVersionArrow');
    const currentVersion = window.RUSH_API?.catalog?.currentPolicyVersion || '—';
    const hasProposal = !!payload && !!(payload.proposal_id || state.selected);
    if (!hasProposal) {
      if (baseChip) baseChip.textContent = `Policy ${currentVersion} · current`;
      if (buildChip) buildChip.textContent = '';
      if (buildLabel) buildLabel.hidden = true;
      if (arrow) arrow.hidden = true;
      return;
    }
    const baseVersion = payload.base_version || currentVersion;
    const buildVersion = payload.build_version || (payload.base_version ? `${payload.base_version}+` : `${baseVersion}+`);
    if (baseChip) baseChip.textContent = baseVersion;
    if (buildChip) buildChip.textContent = buildVersion;
    if (buildLabel) buildLabel.hidden = false;
    if (arrow) arrow.hidden = false;
  }

  function populateControls() {
    const currentRun = $('#proposalRunId')?.value || window.RUSH_API?.catalog?.runs?.[0]?.run_id || '';
    const runSelect = $('#proposalRunId');
    if (runSelect) {
      runSelect.innerHTML = rushApiRunOptions(currentRun, false);
      if (currentRun) runSelect.value = currentRun;
    }
    syncProposalVersionChips();
  }

  function populateProposalPicker() {
    const select = $('#proposalPicker');
    if (!select) return;
    if (!state.proposals.length) {
      select.innerHTML = rushApiOptionHtml('', 'No proposals found', true);
      state.selected = '';
      renderProposalCards();
      syncProposalActions(null);
      syncProposalVersionChips(null);
      return;
    }
    const ordered = orderedProposalList();
    const hasSelected = ordered.some(proposal => proposal.proposal_id === state.selected);
    const selected = hasSelected ? state.selected : defaultProposalId();
    select.innerHTML = ordered.map(proposal => rushApiOptionHtml(proposal.proposal_id || '', proposalLabel(proposal), selected === proposal.proposal_id)).join('');
    select.value = selected;
    state.selected = selected;
    renderProposalCards();
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

  function diffHeaderPath(line) {
    const value = String(line || '').replace(/^[+-]{3}\s+/, '').trim().split(/\s+/)[0] || '';
    if (!value || value === '/dev/null') return '';
    return value.replace(/^[ab]\//, '');
  }

  function diffPath(diff) {
    const lines = String(diff?.unified_diff || '').split('\n');
    return diffHeaderPath(lines.find(line => line.startsWith('+++ ')))
      || diffHeaderPath(lines.find(line => line.startsWith('--- ')))
      || diff?.path
      || 'unknown file';
  }

  function splitUnifiedDiffRecord(diff) {
    const diffText = String(diff?.unified_diff || '');
    const lines = diffText.split('\n');
    const fileStarts = lines.reduce((starts, line, index) => {
      if (line.startsWith('--- ') && lines[index + 1]?.startsWith('+++ ')) starts.push(index);
      return starts;
    }, []);
    if (fileStarts.length <= 1) return [diff];
    return fileStarts.map((start, index) => {
      const end = fileStarts[index + 1] ?? lines.length;
      const unifiedDiff = lines.slice(start, end).join('\n');
      const record = { ...diff, unified_diff: unifiedDiff };
      record.path = diffPath(record);
      return record;
    });
  }

  function normalizeDiffRecords(diffs) {
    return diffs.flatMap(splitUnifiedDiffRecord);
  }

  function diffStats(diffText) {
    return String(diffText || '').split('\n').reduce((stats, line) => {
      if (line.startsWith('+') && !line.startsWith('+++')) stats.additions += 1;
      if (line.startsWith('-') && !line.startsWith('---')) stats.deletions += 1;
      return stats;
    }, { additions: 0, deletions: 0 });
  }

  function renderDiffFileCard(diff) {
    return `<article class="diff-file-card"><h3>${esc(diffPath(diff))} <span>${esc(diff.change || '')}</span></h3>${renderUnifiedDiff(diff.unified_diff || '')}</article>`;
  }

  function renderDiffFile(diff, index, total) {
    const card = renderDiffFileCard(diff);
    if (total <= 3 || index < 3) return card;
    const stats = diffStats(diff.unified_diff || '');
    const summary = `📄 ${diffPath(diff)} (${stats.additions}+ / ${stats.deletions}-)`;
    return `<details class="diff-file-collapsed"><summary>${esc(summary)}</summary>${card}</details>`;
  }

  function renderProposal(payload) {
    const summary = $('#proposalSummary');
    const viewer = $('#proposalDiffViewer');
    if (!summary || !viewer) return;
    if (!payload) {
      state.lastLoadedProposal = null;
      summary.innerHTML = '';
      viewer.innerHTML = '<div class="empty-state">Select a proposal to view its diff.</div>';
      syncProposalActions(null);
      syncProposalVersionChips(null);
      return;
    }
    syncProposalActions(payload);
    syncProposalVersionChips(payload);
    const diffs = normalizeDiffRecords(Array.isArray(payload.diffs) ? payload.diffs : []);
    const cards = [
      ['Proposal', payload.proposal_id || '—', `Status: ${payload.status || 'pending'}`],
      ['Base version', payload.base_version || '—', payload.model_id || DEFAULT_POLICY_MODEL],
      ['Draft model', payload.model_id || DEFAULT_POLICY_MODEL, `Drafted by ${payload.model_id || DEFAULT_POLICY_MODEL} (high reasoning)`],
      ['Changed', payload.files_changed?.length ?? diffs.filter(d => d.change === 'modified').length, (payload.files_changed || []).join(' · ')],
      ['Added / removed', `${payload.files_added?.length ?? 0} / ${payload.files_removed?.length ?? 0}`, [...(payload.files_added || []), ...(payload.files_removed || [])].join(' · ')]
    ];
    summary.innerHTML = cards.map(([label, value, note]) => `<article class="stat-card"><span>${esc(label)}</span><strong>${esc(value)}</strong><p>${esc(note || '')}</p></article>`).join('');
    const statusText = payload.status || 'pending';
    const statusKey = proposalStatusClass(statusText);
    const statusCopy = statusKey === 'pending'
      ? 'SME action needed before this graph changes.'
      : (statusKey === 'parse_error'
        ? 'Malformed proposal output was saved for review; reject it without changing the graph.'
        : 'Proposal state is recorded; only pending proposals can change the graph.');
    const statusBanner = `<div class="proposal-state-banner ${statusKey}"><strong>${esc(proposalStatusLabel(statusText))}</strong><span>${esc(statusCopy)}</span></div>`;
    if (!diffs.length) {
      viewer.innerHTML = `${statusBanner}<div class="empty-state">This proposal has no diff records.</div>`;
      return;
    }
    viewer.innerHTML = statusBanner + diffs.map((diff, index) => renderDiffFile(diff, index, diffs.length)).join('');
  }

  async function loadProposals(selectFirst = true) {
    if (!window.RUSH_API?.available) {
      setUnavailable();
      return;
    }
    try {
      status('Loading proposals…');
      const payload = await rushApiGetJson('/api/policy/proposals?include_errors=true');
      state.proposals = Array.isArray(payload.proposals) ? payload.proposals : [];
      if (selectFirst && !state.selected && state.proposals.length) state.selected = defaultProposalId();
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
      state.lastLoadedProposal = null;
      renderProposal(null);
      return;
    }
    try {
      status(`Loading proposal ${state.selected}…`);
      const payload = await rushApiGetJson(`/api/policy/proposals/${encodeURIComponent(state.selected)}`);
      state.lastLoadedProposal = payload;
      renderProposal(payload);
      renderProposalCards();
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
      status('Select a run before proposing a diff.', 'failed');
      return;
    }
    try {
      status(`Starting proposal from ${runId}…`, 'building');
      $('#proposeDiff').disabled = true;
      const started = await rushApiPostJson('/api/policy/propose-diff?async=1', {
        run_id: runId,
        base_version: baseVersion,
        model_id: DEFAULT_POLICY_MODEL
      });
      const job = started.job_id ? await waitForProposalJob(started) : { status: 'success', result: started, proposal_id: started.proposal_id };
      const payload = job.result || job;
      if (job.status === 'failed') {
        status(jobStatusMessage(job), 'failed');
        return;
      }
      state.selected = payload.proposal_id || '';
      await loadProposals(false);
      if (state.selected) await loadProposal(state.selected);
      if (job.status === 'parse_error' || payload.status === 'parse_error') {
        status(jobStatusMessage(job), 'parse_error');
        return;
      }
      status(`Created proposal ${payload.proposal_id || 'unknown'}.`, 'success');
    } catch (error) {
      const message = /\b524\b/.test(String(error.message || ''))
        ? 'The proposal request timed out at the gateway, but the server may still be working. Refresh proposals in a moment.'
        : `Proposal generation could not start: ${error.message}`;
      status(message, 'failed');
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
    $('#proposalGroupedList')?.addEventListener('click', event => {
      const card = event.target.closest('[data-proposal-id]');
      if (!card) return;
      const proposalId = card.dataset.proposalId || '';
      const select = $('#proposalPicker');
      if (select) select.value = proposalId;
      loadProposal(proposalId);
    });
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
