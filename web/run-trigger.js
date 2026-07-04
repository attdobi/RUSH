(() => {
  const POLL_MS = 2000;
  const MAX_POLL_MS = 30 * 60 * 1000;

  const MODEL_GROUPS = [
    {
      phase: 'Phase 1 · defaults',
      models: [
        { id: 'openai/gpt-5.5-xhigh', checked: false },
        { id: 'openai/gpt-5.5-high', checked: true },
        { id: 'anthropic/claude-opus-4-6', checked: true },
        { id: 'google/gemini-3.1-pro-preview', checked: true }
      ]
    },
    {
      phase: 'Phase 2 · optional sweep',
      models: [
        { id: 'anthropic/claude-opus-4-7', checked: false },
        { id: 'openai/gpt-5.4-mini-xhigh', checked: false },
        { id: 'openai/gpt-5.4-mini-high', checked: false },
        { id: 'google/gemini-3.1-flash-lite-preview', checked: false }
      ]
    }
  ];

  // Mirror of pipeline/providers/pricing.py — keep in sync. GPT reasoning variants mirror their base model prices.
  const PRICING_PER_MTOK = {
    'openai/gpt-5.5': { input: 1.25, output: 10.0 },
    'openai/gpt-5.5-xhigh': { input: 1.25, output: 10.0 },
    'openai/gpt-5.5-high': { input: 1.25, output: 10.0 },
    'google/gemini-3.1-pro-preview': { input: 1.25, output: 5.0 },
    'anthropic/claude-opus-4-6': { input: 15.0, output: 75.0 },
    'anthropic/claude-opus-4-7': { input: 15.0, output: 75.0 },
    'openai/gpt-5.4-mini': { input: 0.15, output: 0.60 },
    'openai/gpt-5.4-mini-xhigh': { input: 0.15, output: 0.60 },
    'openai/gpt-5.4-mini-high': { input: 0.15, output: 0.60 },
    'google/gemini-3.1-flash-lite-preview': { input: 0.10, output: 0.40 }
  };

  const state = { runId: '', pollTimer: null, pollStartedAt: 0, finished: false, lastPayload: null };

  function activeDemo() {
    return typeof window.rushActiveDemo === 'function'
      ? window.rushActiveDemo()
      : { id: 'genai', policyGraph: { area: 'Generative_AI' } };
  }

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
      ${group.models.map(model => renderModelPick(model.id || model, model.checked ?? group.checked)).join('')}
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
    const split = ($('#runTriggerSplit')?.value || 'all').trim() || 'all';
    const sampleIds = ($('#runTriggerSampleIds')?.value || '').trim();
    const limitText = ($('#runTriggerLimit')?.value || '').trim();
    const batchSizeText = ($('#runTriggerBatchSize')?.value || '').trim();
    const limit = sampleIds ? null : (limitText ? Number.parseInt(limitText, 10) : (Number.parseInt(batchSizeText, 10) || 20));
    const allowSpend = $('#runTriggerAllowSpend')?.checked === true;
    if (!models.length) throw new Error('Select at least one model.');
    if (limit !== null && (!Number.isInteger(limit) || limit < 1)) throw new Error('Batch size must be a positive integer.');
    return {
      demo: activeDemo().id || 'genai',
      area: activeDemo().policyGraph?.area || 'Generative_AI',
      models,
      split,
      limit: sampleIds ? null : limit,
      sample_ids: sampleIds || null,
      policy_version: $('#runTriggerPolicyVersion')?.value || window.RUSH_API?.catalog?.currentPolicyVersion || 'v0.1',
      mode: $('#runTriggerMode')?.value || 'cold_start',
      allow_spend: allowSpend,
      allow_holdout: (split === 'holdout' || split === 'all') && allowSpend,
      concurrency: 1,
      batch_size: limit || 20
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

  function formatUsd(value) {
    return isNumber(value) ? `$${value.toFixed(value >= 1 ? 2 : 4)}` : '—';
  }

  function compactModelName(model) {
    return String(model || 'unknown')
      .replace(/^openai\//, 'OpenAI ')
      .replace(/^anthropic\//, 'Anthropic ')
      .replace(/^google\//, 'Google ');
  }

  function renderStatusSummary(payload, completed, expected, errored) {
    const target = $('#runTriggerSummary');
    if (!target) return;
    const models = selectedModels();
    const modelCount = models.length || '—';
    const imageCount = expected && models.length ? Math.ceil(expected / models.length) : ($('#runTriggerBatchSize')?.value || '20');
    const succeeded = Math.max(0, completed - errored);
    const started = payload.started_at ? new Date(payload.started_at).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }) : '—';
    const finished = payload.finished_at ? new Date(payload.finished_at).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }) : (payload.running ? 'running' : '—');
    const modelBreakdown = models.length ? models.map(compactModelName).join(' · ') : 'No models selected';
    const cards = [
      ['Images', imageCount, `split ${$('#runTriggerSplit')?.value || 'all'}`],
      ['Models', modelCount, modelBreakdown],
      ['Time', finished === 'running' ? 'Running' : finished, `started ${started}`],
      ['Cost', formatUsd(payload.running_cost_usd_estimate), 'estimated live spend'],
      ['Calls', `${succeeded}/${expected || '—'}`, errored ? `${errored} error(s)` : 'no errors reported']
    ];
    target.innerHTML = cards.map(([label, value, note]) => `
      <article class="run-summary-metric">
        <span>${esc(label)}</span>
        <strong>${esc(value)}</strong>
        <p>${esc(note)}</p>
      </article>`).join('');
  }

  function renderStatus(payload) {
    state.lastPayload = payload;
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
    renderStatusSummary(payload, completed, expected, errored);
    const raw = $('#runTriggerRawJson');
    if (raw) raw.textContent = JSON.stringify(payload, null, 2);
    const log = $('#runTriggerLogTail');
    if (log) log.textContent = Array.isArray(payload.log_tail) ? payload.log_tail.slice(-8).join('\n') : '';
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
    const section = $('#label');
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
    status('Local API connected. Default run is N=20 across train+test (split=all).');
  }

  rushApiOnReady(initRunTrigger);
  window.addEventListener('rush-api-catalog', populatePolicies);
})();
