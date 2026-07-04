(() => {
  const POLL_MS = 2000;
  const MAX_POLL_MS = 30 * 60 * 1000;

  // Flat model list. The panel groups these by PROVIDER (see populateModels),
  // ordering each family cheapest-first. Each row still shows its computed
  // $/1k estimate PLUS a cost-tier badge (HIGH/MEDIUM/LOW/LOCAL) derived from
  // that estimate via costTierFor, so relative cost stays visible at a glance.
  // Only the default-checked selection (a cheap, diverse set) is curated here.
  const MODEL_LIST = [
    { id: 'openai/gpt-5.5-xhigh', checked: false },
    { id: 'openai/gpt-5.5-high', checked: false },
    { id: 'openai/gpt-5.5-medium', checked: false },
    { id: 'openai/gpt-5.5-low', checked: true },
    { id: 'anthropic/claude-opus-4-6', checked: false },
    { id: 'anthropic/claude-opus-4-7', checked: false },
    { id: 'google/gemini-3.1-pro-preview', checked: false },
    { id: 'anthropic/claude-sonnet-4-6', checked: false },
    { id: 'anthropic/claude-sonnet-5-low', checked: true },
    { id: 'anthropic/claude-sonnet-5-medium', checked: false },
    { id: 'openai/gpt-5.4-mini-xhigh', checked: false },
    { id: 'openai/gpt-5.4-mini-high', checked: false },
    { id: 'openai/gpt-5.4-mini-medium', checked: false },
    { id: 'openai/gpt-5.4-mini-low', checked: false },
    { id: 'google/gemini-3.5-flash', checked: true },
    // TODO(attila-confirm): distinct gemini-3.1-flash SKU/rate is UNVERIFIED
    // (public sources show the 3.1 gen as Pro + Flash-Lite; the full Flash is
    // 3.5). Rate mirrors gemini-3-flash-preview (0.50/3.00) — may be identical.
    { id: 'google/gemini-3.1-flash', checked: false },
    { id: 'google/gemini-3-flash-preview', checked: false },
    { id: 'anthropic/claude-haiku-4-5-low', checked: false },
    { id: 'anthropic/claude-haiku-4-5-medium', checked: false },
    { id: 'google/gemini-3.1-flash-lite', checked: false },
    { id: 'local/qwen3.6-27b', checked: false },
    { id: 'local/gemma-4-26b-a4b-qat', checked: true }
  ];

  // Ordered PROVIDER display groups (Attila's reorg): scan by family, not by
  // cost tier. Each row still carries a cost-tier BADGE so relative price is
  // never lost. Within a group, rows sort by price high-to-low.
  const PROVIDER_GROUP_ORDER = ['OpenAI', 'Anthropic', 'Gemini', 'Local / Open'];

  function providerGroupFor(modelId) {
    const id = String(modelId || '');
    if (id.startsWith('openai/')) return 'OpenAI';
    if (id.startsWith('anthropic/')) return 'Anthropic';
    if (id.startsWith('google/')) return 'Gemini';
    return 'Local / Open';
  }

  // ---- Measured-token cost model (mirror of pipeline/providers/pricing.py) ----
  // Keep tokens/tiers/thresholds in EXACT sync with Python (sync-tested).
  // The old "appetite" model was fantasy (heavy=11200 output tokens made
  // gpt-5.4-mini-xhigh ~$10.6/1k). REAL data (data/runs/*/llm_outputs.jsonl):
  // input ~6,300-8,200 (ontology prompt dominates, ~model-independent), output
  // ~200-1,670 (grows modestly by effort tier). Cost is INPUT-DOMINATED.
  //   estimate_$/1k = (input_rate*INPUT + output_rate*OUTPUT_BY_TIER[tier]) / 1000
  // INPUT_TOKENS_PER_LABEL is the measured median; prompt-driven (grows with
  // ontology size). OUTPUT_TOKENS_BY_TIER calibrated to the real gpt-5.5 family.
  const INPUT_TOKENS_PER_LABEL = 7500;
  const OUTPUT_TOKENS_BY_TIER = { none: 300, low: 450, medium: 950, high: 1160, xhigh: 1670 };
  // Optional per-model measured medians; empty by default (anchor to tiers).
  const MEASURED_OUTPUT_TOKENS = {};
  // Buckets on the real input-dominated scale: HIGH>=$20, MEDIUM>=$5, else LOW.
  const COST_TIER_THRESHOLDS = { high: 20.0, medium: 5.0 };
  const REASONING_SUFFIXES = ['xhigh', 'high', 'medium', 'low'];

  // Mirror of pipeline/providers/pricing.py — keep in EXACT sync. GPT reasoning variants mirror their base model prices.
  // Note: gpt-5.5 input of 1.25 looks like the cached-input rate.
  const PRICING_PER_MTOK = {
    'openai/gpt-5.5': { input: 1.25, output: 10.0 },
    'openai/gpt-5.5-xhigh': { input: 1.25, output: 10.0 },
    'openai/gpt-5.5-high': { input: 1.25, output: 10.0 },
    'openai/gpt-5.5-medium': { input: 1.25, output: 10.0 },
    'openai/gpt-5.5-low': { input: 1.25, output: 10.0 },
    'google/gemini-3.1-pro-preview': { input: 2.0, output: 12.0 },
    // Opus 4.6 (dated but kept): verified 5 / 25 per Mtok.
    'anthropic/claude-opus-4-6': { input: 5.0, output: 25.0 },
    // Opus 4.7: same list price, but newer tokenizer emits ~30% more tokens (effective ~1.3x).
    'anthropic/claude-opus-4-7': { input: 5.0, output: 25.0 },
    'openai/gpt-5.4-mini': { input: 0.15, output: 0.60 },
    'openai/gpt-5.4-mini-xhigh': { input: 0.15, output: 0.60 },
    'openai/gpt-5.4-mini-high': { input: 0.15, output: 0.60 },
    'openai/gpt-5.4-mini-medium': { input: 0.15, output: 0.60 },
    'openai/gpt-5.4-mini-low': { input: 0.15, output: 0.60 },
    'anthropic/claude-sonnet-4-6': { input: 3.0, output: 15.0 },
    // Sonnet 5: INTRO 2.0/10.0 through 2026-08-31 (standard 3.0/15.0 after; +30% tokenizer).
    'anthropic/claude-sonnet-5-low': { input: 2.0, output: 10.0 },
    'anthropic/claude-sonnet-5-medium': { input: 2.0, output: 10.0 },
    // Haiku 4.5: cheap/fast vision model.
    'anthropic/claude-haiku-4-5-low': { input: 1.0, output: 5.0 },
    'anthropic/claude-haiku-4-5-medium': { input: 1.0, output: 5.0 },
    'google/gemini-3.5-flash': { input: 1.50, output: 9.0 },
    // TODO(attila-confirm): gemini-3.1-flash rate UNVERIFIED — mirrors
    // gemini-3-flash-preview (0.50/3.00); may be the same SKU. Keep in sync
    // with pipeline/providers/pricing.py (sync-tested).
    'google/gemini-3.1-flash': { input: 0.50, output: 3.0 },
    'google/gemini-3-flash-preview': { input: 0.50, output: 3.0 },
    'google/gemini-3.1-flash-lite': { input: 0.25, output: 1.50 },
    'local/qwen3.6-27b': { input: 0.0, output: 0.0 },
    'local/gemma-4-26b-a4b-qat': { input: 0.0, output: 0.0 }
  };

  const state = { runId: '', pollTimer: null, pollStartedAt: 0, finished: false, lastPayload: null };

  function activeDemo() {
    return typeof window.rushActiveDemo === 'function'
      ? window.rushActiveDemo()
      : { id: 'genai', policyGraph: { area: 'Generative_AI' } };
  }

  function reasoningTierFor(model) {
    const tail = String(model || '').split('-').pop();
    return REASONING_SUFFIXES.includes(tail) ? tail : 'none';
  }

  // Estimated OUTPUT tokens: measured median when present, else the tier table.
  function estimateOutputTokensFor(model) {
    const measured = MEASURED_OUTPUT_TOKENS[model];
    if (measured !== undefined) return measured;
    return OUTPUT_TOKENS_BY_TIER[reasoningTierFor(model)];
  }

  // Measured-token estimate: input ~constant (prompt-driven), output by tier.
  // Cost is input-dominated.
  function estimatePerThousandLabels(model) {
    const pricing = PRICING_PER_MTOK[model];
    if (!pricing) return null;
    const outputTokens = estimateOutputTokensFor(model);
    const perLabel = (pricing.input * INPUT_TOKENS_PER_LABEL
      + pricing.output * outputTokens) / 1_000_000;
    return perLabel * 1000;
  }

  // Derive the display bucket from the computed estimate (Bug 2). Locals get
  // their own dedicated tier (Bug 3).
  function costTierFor(model) {
    if (isLocalModel(model)) return 'LOCAL';
    const estimate = estimatePerThousandLabels(model);
    if (estimate === null) return 'LOW';
    if (estimate >= COST_TIER_THRESHOLDS.high) return 'HIGH';
    if (estimate >= COST_TIER_THRESHOLDS.medium) return 'MEDIUM';
    return 'LOW';
  }

  function selectedModels() {
    const picker = $('#runTriggerModels');
    return Array.from(picker?.querySelectorAll('input[type="checkbox"]:checked') || [])
      .map(input => input.value)
      .filter(Boolean);
  }

  function isLocalModel(modelId) {
    return String(modelId || '').startsWith('local/');
  }

  function parsePositiveInt(raw, fallback, label) {
    const text = String(raw ?? '').trim();
    if (text && !/^\d+$/.test(text)) {
      throw new Error(`${label} must be a positive integer.`);
    }
    const value = text ? Number.parseInt(text, 10) : fallback;
    if (!Number.isInteger(value) || value < 1) {
      throw new Error(`${label} must be a positive integer.`);
    }
    return value;
  }

  function renderModelPick(model, checked, isLocal) {
    const badgeTier = isLocal ? 'LOCAL' : costTierFor(model);
    let estimateText;
    if (isLocal) {
      estimateText = '$0.00 / 1k labels · free';
    } else {
      const estimate = estimatePerThousandLabels(model);
      const tier = reasoningTierFor(model);
      const tierNote = tier === 'none' ? 'rough estimate' : `${tier} reasoning · rough estimate`;
      estimateText = estimate === null ? 'rough estimate unavailable' : `$${estimate.toFixed(2)} / 1k labels (${tierNote})`;
    }
    const localClass = isLocal ? ' model-pick--local' : '';
    const badge = `<span class="cost-badge cost-badge--${badgeTier.toLowerCase()}">${esc(badgeTier)}</span>`;
    return `<label class="model-pick${localClass}"><input type="checkbox" value="${attr(model)}"${checked ? ' checked' : ''} /><span class="model-pick-body"><code>${esc(model)}</code><em class="rough-estimate">${esc(estimateText)}</em></span>${badge}</label>`;
  }

  function populateModels() {
    const picker = $('#runTriggerModels');
    if (!picker) return;
    // Group every model by PROVIDER so a family is easy to scan (Attila's
    // reorg). Within a group, sort by price high-to-low; each row still carries its
    // computed $/1k estimate and a cost-tier badge.
    const groups = new Map(PROVIDER_GROUP_ORDER.map(group => [group, []]));
    for (const entry of MODEL_LIST) {
      const group = providerGroupFor(entry.id);
      (groups.get(group) || groups.get('Local / Open')).push(entry);
    }
    picker.innerHTML = PROVIDER_GROUP_ORDER.map(group => {
      const entries = groups.get(group);
      if (!entries.length) return '';
      entries.sort((a, b) => (estimatePerThousandLabels(b.id) || 0) - (estimatePerThousandLabels(a.id) || 0));
      const localClass = group === 'Local / Open' ? ' model-picker-provider--local' : '';
      return `
      <div class="model-picker-provider${localClass}">${esc(group)}</div>
      ${entries.map(entry => renderModelPick(entry.id, entry.checked === true, isLocalModel(entry.id))).join('')}`;
    }).join('');
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
    const kPerSplit = parsePositiveInt($('#runTriggerBatchSize')?.value, 20, 'k per split');
    const requestedImagesPerCall = parsePositiveInt($('#runTriggerImagesPerCall')?.value, 5, 'Images per API call');
    const allSelectedModelsAreLocal = models.length > 0 && models.every(isLocalModel);
    const batchSize = allSelectedModelsAreLocal ? 1 : requestedImagesPerCall;
    const limit = sampleIds ? null : kPerSplit;
    const allowSpend = $('#runTriggerAllowSpend')?.checked === true;
    if (!models.length) throw new Error('Select at least one model.');
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
      batch_size: batchSize
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

  // Prefer the manifest-recorded per-batch total once the run finalizes; fall
  // back to the live running estimate summed from label votes mid-run.
  function costValue(payload) {
    if (payload && isNumber(payload.recorded_cost_usd)) return payload.recorded_cost_usd;
    return payload ? payload.running_cost_usd_estimate : undefined;
  }

  function costNote(payload) {
    const cost = payload && payload.cost;
    if (cost && Array.isArray(cost.per_batch) && cost.per_batch.length) {
      const cpi = isNumber(cost.cost_per_image_usd) ? `$${cost.cost_per_image_usd.toFixed(4)}/img` : '—/img';
      return `${cost.per_batch.length} batch(es) recorded · ${cpi}`;
    }
    return 'estimated live spend';
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
      ['Cost', formatUsd(costValue(payload)), costNote(payload)],
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
      const cv = costValue(payload);
      if (isNumber(cv)) {
        costBadge.textContent = `$${cv.toFixed(4)} ${isNumber(payload.recorded_cost_usd) ? 'recorded' : 'spent'}`;
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

  function currentK() {
    const raw = Number.parseInt(($('#runTriggerBatchSize')?.value || '').trim(), 10);
    return Number.isInteger(raw) && raw > 0 ? raw : 20;
  }

  function refreshRunButtonLabel() {
    const btn = $('#startLabelingRun');
    if (!btn) return;
    const ids = ($('#runTriggerSampleIds')?.value || '').trim();
    btn.textContent = ids ? 'Run panel · sample IDs' : `Run panel · k=${currentK()}`;
  }

  function bind() {
    $('#startLabelingRun')?.addEventListener('click', startRun);
    $('#scoreRunNow')?.addEventListener('click', scoreRun);
    $('#runTriggerBatchSize')?.addEventListener('input', refreshRunButtonLabel);
    $('#runTriggerSampleIds')?.addEventListener('input', refreshRunButtonLabel);
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
    refreshRunButtonLabel();
    await rushApiLoadCatalog();
    populatePolicies();
    status('Local API connected. Set k per split (split=all runs up to k train + k test).');
  }

  rushApiOnReady(initRunTrigger);
  window.addEventListener('rush-api-catalog', populatePolicies);
})();
