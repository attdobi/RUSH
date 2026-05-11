(() => {
  const POLL_MS = 2000;
  const MAX_POLL_MS = 30 * 60 * 1000;

  const MODEL_GROUPS = [
    {
      phase: 'Phase 1 · defaults',
      checked: true,
      models: [
        'openai/gpt-5.5',
        'anthropic/claude-opus-4-6',
        'google/gemini-3.1-pro-preview'
      ]
    },
    {
      phase: 'Phase 2 · optional sweep',
      checked: false,
      models: [
        'anthropic/claude-opus-4-7',
        'openai/gpt-5.4-mini',
        'google/gemini-3.1-flash-lite-preview'
      ]
    }
  ];

  // Mirror of pipeline/providers/pricing.py — keep in sync.
  const PRICING_PER_MTOK = {
    'openai/gpt-5.5': { input: 1.25, output: 10.0 },
    'google/gemini-3.1-pro-preview': { input: 1.25, output: 5.0 },
    'anthropic/claude-opus-4-6': { input: 15.0, output: 75.0 },
    'anthropic/claude-opus-4-7': { input: 15.0, output: 75.0 },
    'openai/gpt-5.4-mini': { input: 0.15, output: 0.60 },
    'google/gemini-3.1-flash-lite-preview': { input: 0.10, output: 0.40 }
  };

  const state = { runId: '', pollTimer: null, pollStartedAt: 0, finished: false };

  function estimatePerThousandLabels(model) {
    const pricing = PRICING_PER_MTOK[model];
    if (!pricing) return null;
    return ((pricing.input * 800 + pricing.output * 400) / 1_000_000) * 1000;
  }

  function selectedModels() {
    const picker = $('#runTriggerModels');
    return Array.from(picker?.querySelectorAll('input[type="checkbox"]:checked') || [])
      .map(input => input.value)
      .filter(Boolean);
  }

  function renderModelPick(model, checked) {
    const estimate = estimatePerThousandLabels(model);
    const estimateText = estimate === null ? 'rough estimate unavailable' : `$${estimate.toFixed(4)} / 1k labels (rough estimate)`;
    return `<label class="model-pick"><input type="checkbox" value="${attr(model)}"${checked ? ' checked' : ''} /><span><code>${esc(model)}</code><em class="rough-estimate">${esc(estimateText)}</em></span></label>`;
  }

  function populateModels() {
    const picker = $('#runTriggerModels');
    if (!picker) return;
    picker.innerHTML = MODEL_GROUPS.map(group => `
      <div class="model-picker-phase">${esc(group.phase)}</div>
      ${group.models.map(model => renderModelPick(model, group.checked)).join('')}
    `).join('');
  }

  function populatePolicies() {
    const current = window.RUSH_API?.catalog?.currentPolicyVersion || '';
    const select = $('#runTriggerPolicyVersion');
    if (select) select.innerHTML = rushApiPolicyVersionOptions(current, false);
  }

  function status(message, isError = false) {
    rushApiStatus('#runTriggerStatusLine', message, isError);
  }

  function buildStartPayload() {
    const models = selectedModels();
    const split = ($('#runTriggerSplit')?.value || 'dev_golden').trim() || 'dev_golden';
    const sampleIds = ($('#runTriggerSampleIds')?.value || '').trim();
    const limitText = ($('#runTriggerLimit')?.value || '').trim();
    const limit = limitText ? Number.parseInt(limitText, 10) : null;
    const allowSpend = $('#runTriggerAllowSpend')?.checked === true;
    if (!models.length) throw new Error('Select at least one model.');
    if (sampleIds && limit !== null) throw new Error('Use either limit or sample_ids, not both.');
    if (!sampleIds && limit === null) throw new Error('Provide either a limit or sample_ids.');
    if (limit !== null && (!Number.isInteger(limit) || limit < 1)) throw new Error('Limit must be a positive integer.');
    return {
      models,
      split,
      limit: sampleIds ? null : limit,
      sample_ids: sampleIds || null,
      policy_version: $('#runTriggerPolicyVersion')?.value || window.RUSH_API?.catalog?.currentPolicyVersion || 'v0.1',
      mode: $('#runTriggerMode')?.value || 'cold_start',
      allow_spend: allowSpend,
      allow_holdout: split === 'holdout' && allowSpend,
      concurrency: 1
    };
  }

  function stopPolling() {
    if (state.pollTimer) window.clearTimeout(state.pollTimer);
    state.pollTimer = null;
  }

  function schedulePoll() {
    stopPolling();
    if (!state.runId || state.finished) return;
    if (Date.now() - state.pollStartedAt > MAX_POLL_MS) {
      status('Stopped polling after 30 minutes. Refresh status manually by starting from the returned run id.', true);
      return;
    }
    state.pollTimer = window.setTimeout(() => pollStatus(), POLL_MS);
  }

  function renderStatus(payload) {
    const panel = $('#runTriggerStatusPanel');
    if (panel) panel.hidden = false;
    const completed = Number(payload.completed_calls || 0);
    const expected = Number(payload.expected_calls || 0);
    const errored = Number(payload.errored_calls || 0);
    const progress = isNumber(payload.progress) ? payload.progress : (expected > 0 ? completed / expected : 0);
    const width = Math.max(0, Math.min(100, progress * 100));
    const runId = payload.run_id || state.runId;
    const runIdEl = $('#runTriggerRunId');
    if (runIdEl) runIdEl.textContent = runId ? `Run ${runId}` : 'Run status';
    const progressEl = $('#runTriggerProgressText');
    if (progressEl) progressEl.textContent = `${completed} / ${expected || '—'} calls${errored ? ` · ${errored} errored` : ''}`;
    const costBadge = $('#runTriggerCostBadge');
    if (costBadge) {
      if (isNumber(payload.running_cost_usd_estimate)) {
        costBadge.textContent = `$${payload.running_cost_usd_estimate.toFixed(4)} spent`;
        costBadge.hidden = false;
      } else {
        costBadge.hidden = true;
      }
    }
    const bar = $('#runTriggerProgressBar');
    if (bar) bar.style.width = `${width.toFixed(1)}%`;
    const log = $('#runTriggerLogTail');
    if (log) log.textContent = Array.isArray(payload.log_tail) ? payload.log_tail.join('\n') : '';
    const running = payload.running === true;
    const score = $('#scoreRunNow');
    if (score) score.hidden = running || payload.scoring_done === true;
    state.finished = !running;
    status(running ? `Run ${runId} is running…` : `Run ${runId} finished${payload.scoring_done ? ' and is already scored.' : '.'}`);
    if (!running) {
      stopPolling();
      rushApiLoadCatalog().catch(() => {});
    }
  }

  async function pollStatus() {
    if (!state.runId) return;
    try {
      const payload = await rushApiGetJson(`/api/runs/${encodeURIComponent(state.runId)}/status`);
      renderStatus(payload);
      if (payload.running === true) schedulePoll();
    } catch (error) {
      status(`Status check failed: ${error.message}`, true);
      schedulePoll();
    }
  }

  async function startRun() {
    try {
      const payload = buildStartPayload();
      status('Starting labeling run…');
      $('#startLabelingRun').disabled = true;
      const response = await rushApiPostJson('/api/runs/start', payload);
      state.runId = response.run_id || response.job_id || '';
      state.pollStartedAt = Date.now();
      state.finished = false;
      if (!state.runId) throw new Error('API did not return a run_id.');
      status(`Started ${state.runId}; polling status…`);
      await pollStatus();
    } catch (error) {
      status(`Could not start run: ${error.message}`, true);
    } finally {
      $('#startLabelingRun').disabled = false;
    }
  }

  async function scoreRun() {
    if (!state.runId) return;
    try {
      status(`Scoring ${state.runId}…`);
      $('#scoreRunNow').disabled = true;
      const payload = await rushApiPostJson(`/api/runs/${encodeURIComponent(state.runId)}/score`, {});
      status(`Scored ${payload.run_id || state.runId}.`);
      await pollStatus();
      await rushApiLoadCatalog();
    } catch (error) {
      status(`Score failed: ${error.message}`, true);
    } finally {
      $('#scoreRunNow').disabled = false;
    }
  }

  function bind() {
    $('#startLabelingRun')?.addEventListener('click', startRun);
    $('#scoreRunNow')?.addEventListener('click', scoreRun);
  }

  async function initRunTrigger(api) {
    const section = $('#run-trigger');
    const hint = $('#apiUnavailableHint');
    if (!api.available) {
      if (section) section.hidden = true;
      if (hint) hint.hidden = false;
      return;
    }
    if (section) section.hidden = false;
    if (hint) hint.hidden = true;
    populateModels();
    bind();
    await rushApiLoadCatalog();
    populatePolicies();
    status('Local API connected. Configure a run and click Start labeling run.');
  }

  rushApiOnReady(initRunTrigger);
  window.addEventListener('rush-api-catalog', populatePolicies);
})();
