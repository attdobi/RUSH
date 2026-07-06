(() => {
  const POLL_MS = 2000;
  // Safety cap only — the poll loop already stops when the run reports
  // running:false, so this just guards a truly stuck tab. Long local runs
  // (slow single-GPU models) routinely exceed 30 min, so keep it generous.
  const MAX_POLL_MS = 12 * 60 * 60 * 1000;

  // Flat model list. The panel groups these by PROVIDER (see populateModels),
  // ordering each family cheapest-first. Each row still shows its computed
  // $/1k estimate PLUS a cost-tier badge (HIGH/MEDIUM/LOW/LOCAL) derived from
  // that estimate via costTierFor, so relative cost stays visible at a glance.
  // Default-checked = the single CHEAPEST hosted model per provider + BOTH free
  // locals (Attila's init preference). Everything else off by default.
  const MODEL_LIST = [
    { id: 'openai/gpt-5.5-xhigh', checked: false },
    { id: 'openai/gpt-5.5-high', checked: false },
    { id: 'openai/gpt-5.5-medium', checked: false },
    { id: 'openai/gpt-5.5-low', checked: false },
    { id: 'anthropic/claude-opus-4-6', checked: false },
    { id: 'anthropic/claude-opus-4-7', checked: false },
    { id: 'google/gemini-3.1-pro-preview', checked: false },
    { id: 'anthropic/claude-sonnet-4-6', checked: false },
    { id: 'anthropic/claude-sonnet-5-low', checked: false },
    { id: 'anthropic/claude-sonnet-5-medium', checked: false },
    { id: 'openai/gpt-5.4-mini-xhigh', checked: false },
    { id: 'openai/gpt-5.4-mini-high', checked: false },
    { id: 'openai/gpt-5.4-mini-medium', checked: false },
    { id: 'openai/gpt-5.4-mini-low', checked: true },
    { id: 'google/gemini-3.5-flash', checked: false },
    // TODO(attila-confirm): distinct gemini-3.1-flash SKU/rate is UNVERIFIED
    // (public sources show the 3.1 gen as Pro + Flash-Lite; the full Flash is
    // 3.5). Rate mirrors gemini-3-flash-preview (0.50/3.00) — may be identical.
    { id: 'google/gemini-3.1-flash', checked: false },
    { id: 'google/gemini-3-flash-preview', checked: false },
    { id: 'anthropic/claude-haiku-4-5-low', checked: true },
    { id: 'anthropic/claude-haiku-4-5-medium', checked: false },
    { id: 'google/gemini-3.1-flash-lite', checked: true },
    { id: 'local/qwen3.6-27b', checked: false },
    { id: 'local/qwen3.6-35b-a3b', checked: false },
    { id: 'local/qwen2.5-vl-7b', checked: true },
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
  const LOCAL_REASONING_SESSION_KEY = 'rush_local_reasoning_overrides_v1';
  const localReasoningOverrides = readLocalReasoningOverrides();

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
    'local/qwen3.6-27b-low': { input: 0.0, output: 0.0 },
    'local/qwen3.6-35b-a3b': { input: 0.0, output: 0.0 },
    'local/qwen2.5-vl-7b': { input: 0.0, output: 0.0 },
    'local/gemma-4-26b-a4b-qat': { input: 0.0, output: 0.0 }
  };

  const state = {
    runId: '',
    runModels: [],
    pollTimer: null,
    pollStartedAt: 0,
    finished: false,
    lastPayload: null,
    elapsedTimer: null,
    canceling: false,
    cancelError: ''
  };

  // Format a duration in seconds as mm:ss, or hh:mm:ss once it crosses an hour.
  function formatElapsed(totalSeconds) {
    const s = Math.max(0, Math.floor(Number(totalSeconds) || 0));
    const hh = Math.floor(s / 3600);
    const mm = Math.floor((s % 3600) / 60);
    const ss = s % 60;
    const pad = (n) => String(n).padStart(2, '0');
    return hh > 0 ? `${hh}:${pad(mm)}:${pad(ss)}` : `${pad(mm)}:${pad(ss)}`;
  }

  function payloadStatusValues(payload) {
    return [
      payload?.status,
      payload?.run_status,
      payload?.state,
      payload?.liveness_status,
      payload?.liveness?.status,
      payload?.liveness?.state
    ].filter(value => value !== undefined && value !== null).map(value => String(value).toLowerCase());
  }

  function runErroredCount(payload) {
    const direct = Number(payload?.errored_calls);
    if (Number.isFinite(direct)) return direct;
    const totals = payload?.totals && typeof payload.totals === 'object' ? Number(payload.totals.errored_calls) : 0;
    return Number.isFinite(totals) ? totals : 0;
  }

  function truthyApiFlag(value) {
    if (value === true || value === 1) return true;
    if (typeof value === 'string') return ['1', 'true', 'yes'].includes(value.toLowerCase());
    return false;
  }

  function completedWithErrors(payload) {
    const values = payloadStatusValues(payload);
    if (values.some(value => value.includes('abort') || value.includes('fail') || value.includes('cancel'))) return false;
    if (truthyApiFlag(payload?.completed_with_errors)) return true;
    if (values.some(value => value.includes('completed_with_errors'))) return true;
    const errored = runErroredCount(payload);
    if (errored <= 0) return false;
    if (values.some(value => value.includes('complete') || value.includes('success') || value.includes('finish'))) return true;
    return !!payload?.finished_at && payload?.running !== true;
  }

  function runStateLabel(payload) {
    const values = payloadStatusValues(payload);
    if (payload?.stale === true || payload?.is_stale === true || payload?.liveness?.stale === true) return 'stale';
    if (values.some(value => value.includes('stale'))) return 'stale';
    if (values.some(value => value === 'cancel' || value.includes('canceled') || value.includes('cancelled'))) return 'canceled';
    if (values.some(value => value.includes('abort'))) return 'aborted';
    if (completedWithErrors(payload)) return 'completed-with-errors';
    if (payload?.returncode !== undefined && payload?.returncode !== null && Number(payload.returncode) !== 0) return 'failed';
    if (values.some(value => value.includes('fail') || value.includes('error'))) return 'failed';
    if (payload?.running === true) {
      const start = new Date(payload.started_at || '').getTime();
      if (Number.isFinite(start) && Date.now() - start > MAX_POLL_MS) return 'stale';
      return 'running';
    }
    if (values.some(value => value.includes('complete') || value.includes('success') || value.includes('finish'))) return 'finished';
    if (payload?.finished_at) return 'finished';
    if (payload?.running === false && payload?.finished_at == null) return 'stopped';
    return payload ? 'finished' : 'unknown';
  }

  function isLiveRun(payload) {
    return runStateLabel(payload) === 'running';
  }

  // Live elapsed: from started_at to finished_at (frozen) or now only while live.
  function elapsedSecondsFor(payload) {
    if (!payload || !payload.started_at) return null;
    const start = new Date(payload.started_at).getTime();
    if (!Number.isFinite(start)) return null;
    if (!isLiveRun(payload) && !payload.finished_at) {
      return isNumber(payload.elapsed_seconds) && payload.elapsed_seconds < (MAX_POLL_MS / 1000)
        ? payload.elapsed_seconds
        : null;
    }
    const end = payload.finished_at ? new Date(payload.finished_at).getTime() : Date.now();
    if (!Number.isFinite(end)) return null;
    return Math.max(0, (end - start) / 1000);
  }

  function elapsedTextFor(payload) {
    const label = runStateLabel(payload);
    if (['aborted', 'canceled', 'stale', 'failed', 'stopped'].includes(label) && !payload?.finished_at) return label;
    const secs = elapsedSecondsFor(payload);
    return secs === null ? '—' : formatElapsed(secs);
  }

  function updateElapsed() {
    const el = $('#runTriggerElapsed');
    if (!el) return;
    el.textContent = elapsedTextFor(state.lastPayload);
  }

  function stopElapsedTicker() {
    if (state.elapsedTimer) window.clearInterval(state.elapsedTimer);
    state.elapsedTimer = null;
  }

  function startElapsedTicker() {
    stopElapsedTicker();
    updateElapsed();
    state.elapsedTimer = window.setInterval(updateElapsed, 1000);
  }

  function arrayFromMaybeObject(value) {
    if (Array.isArray(value)) return value;
    if (!value || typeof value !== 'object') return [];
    return Object.entries(value).map(([key, row]) => (
      row && typeof row === 'object' ? { model: key, ...row } : { model: key, value: row }
    ));
  }

  function numericField(row, fields) {
    for (const field of fields) {
      const value = row?.[field];
      if (isNumber(value)) return value;
      if (typeof value === 'string' && value.trim() !== '') {
        const parsed = Number(value);
        if (Number.isFinite(parsed)) return parsed;
      }
    }
    return null;
  }

  function modelIdFromRow(row) {
    return String(row?.model || row?.model_id || row?.labeler_id || row?.id || '').trim();
  }

  function extractRequestedModels(payload) {
    const candidates = [
      payload?.models,
      payload?.model_ids,
      payload?.request?.models,
      payload?.manifest?.models
    ];
    for (const candidate of candidates) {
      const models = arrayFromMaybeObject(candidate).map(item => {
        if (typeof item === 'string') return item;
        return item?.model || item?.model_id || item?.id || '';
      }).filter(Boolean);
      if (models.length) return Array.from(new Set(models.map(String)));
    }
    return Array.isArray(state.runModels) ? state.runModels : [];
  }

  function mergeModelTelemetry(speedRows, costRows) {
    const byModel = new Map();
    const mergeRow = (base, row, model) => {
      const merged = { ...(base || {}), model };
      for (const [key, value] of Object.entries(row || {})) {
        if (value === undefined || value === null || value === '') continue;
        merged[key] = value;
      }
      return merged;
    };
    for (const row of speedRows) {
      const model = modelIdFromRow(row);
      if (!model) continue;
      byModel.set(model, mergeRow(byModel.get(model), row, model));
    }
    for (const row of costRows) {
      const model = modelIdFromRow(row);
      if (!model) continue;
      byModel.set(model, mergeRow(byModel.get(model), row, model));
    }
    return Array.from(byModel.values());
  }

  function modelSpeedSummaryRows(summary) {
    const nestedRows = [
      ...arrayFromMaybeObject(summary?.models),
      ...arrayFromMaybeObject(summary?.per_model)
    ];
    const hasNestedShape = summary && typeof summary === 'object' && !Array.isArray(summary)
      && ('models' in summary || 'per_model' in summary);
    if (nestedRows.length || hasNestedShape) {
      return nestedRows;
    }
    return arrayFromMaybeObject(summary);
  }

  function normalizedModelTelemetryRows(payload) {
    // Consumed telemetry fields:
    // model/model_id, avg_s_per_call (fallback avg_latency_ms/1000),
    // tokens_per_sec, total_input/output_tokens, total_cost/total_cost_usd, calls_done.
    const speedRows = [
      ...modelSpeedSummaryRows(payload?.model_speed_summary),
      ...arrayFromMaybeObject(payload?.model_speed),
      ...arrayFromMaybeObject(payload?.per_model_speed),
      ...arrayFromMaybeObject(payload?.speed?.per_model),
      ...arrayFromMaybeObject(payload?.telemetry?.per_model),
      ...arrayFromMaybeObject(payload?.per_model)
    ];
    const costRows = arrayFromMaybeObject(payload?.cost?.per_model);
    let rows = mergeModelTelemetry(speedRows, costRows);
    const requested = extractRequestedModels(payload);
    if (requested.length) {
      const allowed = new Set(requested);
      const filtered = rows.filter(row => allowed.has(modelIdFromRow(row)));
      rows = filtered.length ? filtered : [];
    }
    return rows.map(row => {
      const avgSec = numericField(row, ['avg_s_per_call', 'avg_seconds_per_call', 'avg_sec_per_call', 'avg_latency_s', 'avg_latency_seconds']);
      const avgMs = numericField(row, ['avg_latency_ms', 'latency_ms_avg']);
      const inputTokens = numericField(row, ['total_input_tokens', 'input_tokens']);
      const outputTokens = numericField(row, ['total_output_tokens', 'output_tokens']);
      const totalTokens = numericField(row, ['total_tokens', 'tokens_total'])
        ?? (inputTokens !== null || outputTokens !== null ? (inputTokens || 0) + (outputTokens || 0) : null);
      const callsDone = numericField(row, ['calls_done', 'n_calls', 'completed_calls', 'calls', 'images']);
      const callsTotal = numericField(row, ['calls_total', 'expected_calls', 'total_calls']);
      return {
        model: modelIdFromRow(row),
        avg_s_per_call: avgSec !== null ? avgSec : (avgMs !== null ? avgMs / 1000 : null),
        tokens_per_sec: numericField(row, ['tokens_per_sec', 'tokens_per_second', 'output_tokens_per_sec', 'output_tokens_per_second']),
        images_per_min: numericField(row, ['images_per_min', 'imgs_per_min', 'images_per_minute', 'throughput_imgs_per_min']),
        total_tokens: totalTokens,
        total_cost: numericField(row, ['total_cost', 'total_cost_usd', 'cost_usd']),
        calls_done: callsDone,
        calls_total: callsTotal,
        done: row?.done === true || (callsDone !== null && callsTotal !== null && callsTotal > 0 && callsDone >= callsTotal)
      };
    }).filter(row => row.model);
  }

  function formatNumber(value, digits = 1) {
    return isNumber(value) ? value.toLocaleString(undefined, { maximumFractionDigits: digits }) : '—';
  }

  function formatInteger(value) {
    return isNumber(value) ? Math.round(value).toLocaleString() : '—';
  }

  function renderModelSpeed(payload) {
    const target = $('#runTriggerModelSpeed');
    if (!target) return;
    const rows = normalizedModelTelemetryRows(payload);
    if (!rows.length) { target.innerHTML = ''; return; }
    const body = rows.map((r) => {
      const callTotal = isNumber(r.calls_total) ? `/${esc(formatInteger(r.calls_total))}` : '';
      const doneText = r.done ? ' · done' : '';
      const callNote = isNumber(r.calls_done)
        ? `<small>${esc(formatInteger(r.calls_done))}${callTotal} call(s)${doneText}</small>`
        : (r.done ? `<small>done</small>` : '');
      return `<tr class="run-model-speed-row${r.done ? ' run-model-speed-row--done' : ''}">`
        + `<td><code>${esc(compactModelName(r.model))}</code>${callNote}</td>`
        + `<td>${esc(isNumber(r.avg_s_per_call) ? r.avg_s_per_call.toFixed(2) : '—')}</td>`
        + `<td>${esc(formatNumber(r.tokens_per_sec, 1))}</td>`
        + `<td>${esc(formatNumber(r.images_per_min, 1))}</td>`
        + `<td>${esc(formatInteger(r.total_tokens))}</td>`
        + `<td>${esc(formatUsd(r.total_cost))}</td></tr>`;
    }).join('');
    target.innerHTML = `<div class="compact-table"><table class="run-model-speed-table misalignment"><thead><tr>`
      + `<th>Model</th><th>Avg s/call</th><th>Tokens/sec</th><th>Images/min</th><th>Total tokens</th><th>Total cost</th>`
      + `</tr></thead><tbody>${body}</tbody></table></div>`;
  }

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
    return Array.from(picker?.querySelectorAll('input.model-select-input[type="checkbox"]:checked') || [])
      .map(input => input.value)
      .filter(Boolean);
  }

  function isLocalModel(modelId) {
    return String(modelId || '').startsWith('local/');
  }

  function activePolicyArea() {
    return activeDemo().policyGraph?.area || 'Generative_AI';
  }

  function defaultLocalReasoningForArea(area = activePolicyArea()) {
    return String(area || '') !== 'MNIST_Digits';
  }

  function readLocalReasoningOverrides() {
    try {
      const raw = window.sessionStorage.getItem(LOCAL_REASONING_SESSION_KEY);
      if (!raw) return {};
      const parsed = JSON.parse(raw);
      return parsed && typeof parsed === 'object' && !Array.isArray(parsed) ? parsed : {};
    } catch (error) {
      return {};
    }
  }

  function writeLocalReasoningOverrides() {
    try {
      window.sessionStorage.setItem(LOCAL_REASONING_SESSION_KEY, JSON.stringify(localReasoningOverrides));
    } catch (error) {
      /* sessionStorage is optional; the in-memory object still tracks this page. */
    }
  }

  function hasLocalReasoningOverride(modelId) {
    return Object.prototype.hasOwnProperty.call(localReasoningOverrides, modelId);
  }

  function localReasoningEnabled(modelId) {
    if (hasLocalReasoningOverride(modelId)) return localReasoningOverrides[modelId] === true;
    return defaultLocalReasoningForArea();
  }

  function localReasoningForSelectedModels(models = selectedModels()) {
    return models.filter(isLocalModel).reduce((payload, modelId) => {
      payload[modelId] = localReasoningEnabled(modelId);
      return payload;
    }, {});
  }

  function renderLocalReasoningToggle(model) {
    const enabled = localReasoningEnabled(model);
    const stateText = enabled ? 'On' : 'Off';
    return `
      <div class="local-reasoning-control" data-local-reasoning-for="${attr(model)}">
        <span class="local-reasoning-label">Reasoning</span>
        <label class="local-reasoning-switch" aria-label="Reasoning for ${attr(model)}">
          <input class="local-reasoning-input" type="checkbox" data-local-reasoning-model="${attr(model)}"${enabled ? ' checked' : ''} />
          <span class="local-reasoning-slider" aria-hidden="true"></span>
          <span class="local-reasoning-state">${esc(stateText)}</span>
        </label>
      </div>`;
  }

  function syncLocalReasoningControls() {
    document.querySelectorAll('.local-reasoning-input').forEach(input => {
      const model = input.dataset.localReasoningModel || '';
      const enabled = localReasoningEnabled(model);
      input.checked = enabled;
      const stateEl = input.closest('.local-reasoning-switch')?.querySelector('.local-reasoning-state');
      if (stateEl) stateEl.textContent = enabled ? 'On' : 'Off';
    });
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
    // Measured speed from this machine's recent runs (seconds/image), when we have it.
    const speed = MODEL_SECONDS_PER_IMAGE[model];
    const speedText = (speed && speed.seconds)
      ? `<em class="model-speed" title="median seconds/image over ${speed.samples} recent run(s) on this machine">~${speed.seconds < 10 ? speed.seconds.toFixed(1) : Math.round(speed.seconds)}s/img</em>`
      : '';
    const localClass = isLocal ? ' model-pick--local' : '';
    const badge = `<span class="cost-badge cost-badge--${badgeTier.toLowerCase()}">${esc(badgeTier)}</span>`;
    const reasoningToggle = isLocal ? renderLocalReasoningToggle(model) : '';
    return `
      <div class="model-pick${localClass}">
        <label class="model-pick-select">
          <input class="model-select-input" type="checkbox" value="${attr(model)}"${checked ? ' checked' : ''} />
          <span class="model-pick-body"><code>${esc(model)}</code><em class="rough-estimate">${esc(estimateText)}</em>${speedText}</span>
          ${badge}
        </label>
        ${reasoningToggle}
      </div>`;
  }

  // model_id -> { seconds: median seconds/image, samples: n runs } aggregated
  // from this machine's recent runs' per-model timing (server-recorded).
  let MODEL_SECONDS_PER_IMAGE = {};

  async function refreshModelSpeedEstimates() {
    try {
      // Speed is per-model / demo-agnostic; pull both demo lists for more samples.
      const results = await Promise.all(
        ['mnist', 'genai'].map(demo =>
          rushApiGetJson(`/api/runs?demo=${encodeURIComponent(demo)}`).catch(() => null))
      );
      const perModel = {};
      for (const data of results) {
        for (const run of (data && Array.isArray(data.runs) ? data.runs : [])) {
          const rows = run.model_speed_summary && Array.isArray(run.model_speed_summary.models)
            ? run.model_speed_summary.models : [];
          for (const m of rows) {
            const s = Number(m.avg_s_per_call);
            if (m.model_id && Number.isFinite(s) && s > 0) {
              (perModel[m.model_id] = perModel[m.model_id] || []).push(s);
            }
          }
        }
      }
      const next = {};
      for (const [id, arr] of Object.entries(perModel)) {
        arr.sort((a, b) => a - b);
        next[id] = { seconds: arr[Math.floor(arr.length / 2)], samples: arr.length };
      }
      MODEL_SECONDS_PER_IMAGE = next;
      populateModels();  // re-render rows with the measured speed
    } catch (_e) { /* speed estimate is best-effort */ }
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
      // Each provider is a SELF-CONTAINED block: a full-width header sitting
      // directly above a 2-up grid of ONLY that provider's model rows. Groups
      // stack vertically so a header can never split away from its models
      // across a column break (Attila's screenshot bug).
      const rows = entries.map(entry => renderModelPick(entry.id, entry.checked === true, isLocalModel(entry.id))).join('');
      return `
      <div class="model-picker-group">
        <div class="model-picker-provider${localClass}">${esc(group)}</div>
        <div class="model-picker-grid">${rows}</div>
      </div>`;
    }).join('');
    syncLocalReasoningControls();
  }

  function populatePolicies() {
    const current = window.RUSH_API?.catalog?.currentPolicyVersion || '';
    const select = $('#runTriggerPolicyVersion');
    if (select) select.innerHTML = rushApiPolicyVersionOptions(current, false);
  }

  function status(message, isError = false) {
    rushApiStatus('#runTriggerStatusLine', message, isError);
  }

  function lifecycleTitle(lifecycle, payload = null) {
    if (lifecycle === 'completed-with-errors') {
      const count = runErroredCount(payload);
      return `Completed · ${count} errored`;
    }
    const titles = {
      aborted: 'Aborted',
      canceled: 'Canceled',
      failed: 'Failed',
      finished: 'Finished',
      running: 'Running',
      stale: 'Stale',
      stopped: 'Stopped',
      unknown: 'Unknown'
    };
    return titles[lifecycle] || lifecycle;
  }

  function runToken(payload) {
    return payload?.run_id || payload?.job_id || state.runId;
  }

  function ensureRunStatusControls() {
    const row = $('#runTriggerStatusPanel .progress-row');
    if (!row) return {};

    let lifecycleBadge = $('#runTriggerLifecycleBadge');
    if (!lifecycleBadge) {
      lifecycleBadge = document.createElement('span');
      lifecycleBadge.id = 'runTriggerLifecycleBadge';
      lifecycleBadge.className = 'run-status-chip';
      row.appendChild(lifecycleBadge);
    }

    let cancelButton = $('#cancelRunButton');
    if (!cancelButton) {
      cancelButton = document.createElement('button');
      cancelButton.id = 'cancelRunButton';
      cancelButton.type = 'button';
      cancelButton.className = 'cancel-run-button';
      cancelButton.textContent = 'Cancel run';
      cancelButton.addEventListener('click', cancelRun);
      row.appendChild(cancelButton);
    }

    let cancelMessage = $('#runTriggerCancelMessage');
    if (!cancelMessage) {
      cancelMessage = document.createElement('small');
      cancelMessage.id = 'runTriggerCancelMessage';
      cancelMessage.className = 'cancel-run-message';
      cancelMessage.hidden = true;
      row.appendChild(cancelMessage);
    }

    return { lifecycleBadge, cancelButton, cancelMessage };
  }

  function renderRunStatusControls(payload, lifecycle, running) {
    const { lifecycleBadge, cancelButton, cancelMessage } = ensureRunStatusControls();
    if (lifecycleBadge) {
      lifecycleBadge.textContent = lifecycleTitle(lifecycle, payload);
      lifecycleBadge.className = `run-status-chip status-${lifecycle}`;
      lifecycleBadge.hidden = lifecycle === 'unknown';
    }
    if (cancelButton) {
      cancelButton.hidden = !running;
      cancelButton.disabled = state.canceling;
      cancelButton.textContent = state.canceling ? 'canceling…' : 'Cancel run';
      cancelButton.dataset.runToken = runToken(payload) || '';
    }
    if (cancelMessage) {
      cancelMessage.textContent = state.cancelError;
      cancelMessage.hidden = !state.cancelError;
    }
  }

  function ensureRunErrorsPanel() {
    let panel = $('#runTriggerErrors');
    if (panel) return panel;
    const host = $('#runTriggerModelSpeed') || $('#runTriggerSummary');
    if (!host || !host.parentNode) return null;
    panel = document.createElement('div');
    panel.id = 'runTriggerErrors';
    panel.className = 'run-error-panel';
    panel.hidden = true;
    host.insertAdjacentElement('afterend', panel);
    return panel;
  }

  async function updateRunErrors(payload) {
    const panel = ensureRunErrorsPanel();
    if (!panel) return;
    const runId = payload?.run_id || state.runId;
    const errored = runErroredCount(payload);
    if (!runId || errored <= 0) {
      panel.hidden = true;
      panel.innerHTML = '';
      return;
    }
    const rows = typeof window.rushFetchRunErrors === 'function'
      ? await window.rushFetchRunErrors(runId)
      : [];
    if ((payload?.run_id || state.runId) !== runId) return;
    const renderer = window.rushRenderRunErrorDetails;
    panel.innerHTML = typeof renderer === 'function'
      ? renderer(rows, errored)
      : `<p class="muted">${errored} errored call(s) recorded.</p>`;
    panel.hidden = false;
  }

  async function cancelRun() {
    const token = runToken(state.lastPayload);
    if (!token || state.canceling) return;
    if (!window.confirm('Cancel this run? The job will be terminated.')) return;
    state.canceling = true;
    state.cancelError = '';
    renderRunStatusControls(state.lastPayload, runStateLabel(state.lastPayload), true);
    try {
      await rushApiPostJson(`/api/runs/${encodeURIComponent(token)}/cancel`, {});
      status(`Cancel requested for ${token}; waiting for terminal status…`);
      schedulePoll();
    } catch (error) {
      state.canceling = false;
      state.cancelError = `Cancel failed: ${error.message}`;
      renderRunStatusControls(state.lastPayload, runStateLabel(state.lastPayload), isLiveRun(state.lastPayload));
      status(state.cancelError, true);
    }
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
    let concurrency = parsePositiveInt($('#runTriggerConcurrency')?.value, 4, 'Parallelism');
    if (concurrency > 4) concurrency = 4;
    if (!models.length) throw new Error('Select at least one model.');
    const localReasoning = localReasoningForSelectedModels(models);
    return {
      demo: activeDemo().id || 'genai',
      area: activePolicyArea(),
      models,
      local_reasoning: localReasoning,
      split,
      limit: sampleIds ? null : limit,
      sample_ids: sampleIds || null,
      policy_version: $('#runTriggerPolicyVersion')?.value || window.RUSH_API?.catalog?.currentPolicyVersion || 'v0.1',
      mode: $('#runTriggerMode')?.value || 'cold_start',
      allow_spend: allowSpend,
      allow_holdout: (split === 'holdout' || split === 'all') && allowSpend,
      concurrency,
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
      status('Stopped polling after 12 hours. Refresh the page to resume watching the run.', true);
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
    const models = extractRequestedModels(payload);
    const modelCount = models.length || '—';
    const imageCount = expected && models.length ? Math.ceil(expected / models.length) : ($('#runTriggerBatchSize')?.value || '20');
    const succeeded = completed;
    const started = payload.started_at ? new Date(payload.started_at).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }) : '—';
    const lifecycle = runStateLabel(payload);
    const modelBreakdown = models.length ? models.map(compactModelName).join(' · ') : 'No models selected';
    const elapsedText = elapsedTextFor(payload);
    const timeNote = `started ${started}${payload.finished_at ? ' · final' : lifecycle !== 'running' ? ` · ${lifecycle}` : ''}`;
    const cards = [
      ['Images', imageCount, `split ${$('#runTriggerSplit')?.value || 'all'}`],
      ['Models', modelCount, modelBreakdown],
      ['Time', `__ELAPSED__${elapsedText}`, timeNote],
      ['Cost', formatUsd(costValue(payload)), costNote(payload)],
      ['Calls', `${succeeded}/${expected || '—'}`, errored ? `${errored} error(s)` : 'no errors reported']
    ];
    target.innerHTML = cards.map(([label, value, note]) => {
      const raw = String(value);
      const valueHtml = raw.startsWith('__ELAPSED__')
        ? `<strong id="runTriggerElapsed">${esc(raw.slice('__ELAPSED__'.length))}</strong>`
        : `<strong>${esc(raw)}</strong>`;
      return `
      <article class="run-summary-metric">
        <span>${esc(label)}</span>
        ${valueHtml}
        <p>${esc(note)}</p>
      </article>`;
    }).join('');
  }

  function renderStatus(payload) {
    state.lastPayload = payload;
    const requestedModels = extractRequestedModels(payload);
    if (requestedModels.length) state.runModels = requestedModels;
    const panel = $('#runTriggerStatusPanel');
    if (panel) panel.hidden = false;
    const completed = Number(payload.completed_calls || 0);
    const expected = Number(payload.expected_calls || 0);
    const errored = Number(payload.errored_calls || 0);
    const attempted = completed + errored;
    const progress = completedWithErrors(payload) && expected > 0
      ? Math.min(1, attempted / expected)
      : (isNumber(payload.progress) ? payload.progress : (expected > 0 ? attempted / expected : 0));
    const width = Math.max(0, Math.min(100, progress * 100));
    const runId = payload.run_id || state.runId;
    const runIdEl = $('#runTriggerRunId');
    if (runIdEl) runIdEl.textContent = runId ? `Run ${runId}` : 'Run status';
    const progressEl = $('#runTriggerProgressText');
    if (progressEl) progressEl.textContent = `${completed} / ${expected || '—'} ${errored ? 'successful calls' : 'calls'}${errored ? ` · ${errored} errored` : ''}`;
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
    renderModelSpeed(payload);
    updateRunErrors(payload).catch(() => {});
    const raw = $('#runTriggerRawJson');
    if (raw) raw.textContent = JSON.stringify(payload, null, 2);
    const log = $('#runTriggerLogTail');
    if (log) log.textContent = Array.isArray(payload.log_tail) ? payload.log_tail.slice(-8).join('\n') : '';
    const lifecycle = runStateLabel(payload);
    const running = lifecycle === 'running';
    if (!running) {
      state.canceling = false;
      state.cancelError = '';
    }
    renderRunStatusControls(payload, lifecycle, running);
    const score = $('#scoreRunNow');
    if (score) score.hidden = running || payload.scoring_done === true;
    state.finished = !running;
    if (running) {
      startElapsedTicker();
    } else {
      stopElapsedTicker();
      updateElapsed();
    }
    const runNoun = state.kind === 'cascade' ? 'Cascade' : 'Run';
    status(running ? `${runNoun} ${runId} is running…` : `${runNoun} ${runId} ${lifecycleTitle(lifecycle, payload).toLowerCase()}${payload.scoring_done ? ' and is already scored.' : '.'}`, ['aborted', 'stale', 'failed'].includes(lifecycle));
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
      else refreshModelSpeedEstimates();  // run finished -> update measured seconds/image
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
      state.runModels = payload.models || [];
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

  // Tier-2 judge choices for the cascade button: reasoning-capable steps up
  // from the cheap default panel. First entry is the default.
  const CASCADE_ESCALATE_OPTIONS = [
    'anthropic/claude-sonnet-5-medium',
    'anthropic/claude-haiku-4-5-medium',
    'openai/gpt-5.5-medium',
    'google/gemini-3.1-pro-preview',
    'anthropic/claude-opus-4-7',
    'local/qwen3.6-35b-a3b'
  ];

  function populateCascadeEscalate() {
    const select = $('#cascadeEscalateModel');
    if (!select) return;
    select.innerHTML = CASCADE_ESCALATE_OPTIONS
      .map(id => `<option value="${attr(id)}">${esc(id)}</option>`)
      .join('');
  }

  async function startCascade() {
    try {
      const payload = buildStartPayload();
      // Cascade takes a split+limit slice; explicit sample IDs stay a
      // plain-run feature.
      payload.sample_ids = null;
      if (payload.limit == null) payload.limit = currentK();
      const escalateModel = $('#cascadeEscalateModel')?.value || CASCADE_ESCALATE_OPTIONS[0];
      payload.escalate_models = [escalateModel];
      status('Starting escalation cascade (tier 1 cheap panel → tier 2 judge)…');
      $('#startCascadeRun').disabled = true;
      const response = await rushApiPostJson('/api/runs/start-cascade', payload);
      state.runId = response.run_id || response.job_id || '';
      state.runModels = payload.models || [];
      state.kind = 'cascade';
      state.pollStartedAt = Date.now();
      state.finished = false;
      if (!state.runId) throw new Error('API did not return a run_id.');
      status(`Started cascade ${state.runId}; tier banners appear in the log tail…`);
      await pollStatus();
    } catch (error) {
      status(`Could not start cascade: ${error.message}`, true);
    } finally {
      $('#startCascadeRun').disabled = false;
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
    if (btn) {
      const ids = ($('#runTriggerSampleIds')?.value || '').trim();
      btn.textContent = ids ? 'Run panel · sample IDs' : `Run panel · k=${currentK()}`;
    }
    const cascadeBtn = $('#startCascadeRun');
    if (cascadeBtn) cascadeBtn.textContent = `Run cascade · k=${currentK()}`;
  }

  function bind() {
    $('#startLabelingRun')?.addEventListener('click', startRun);
    $('#startCascadeRun')?.addEventListener('click', startCascade);
    $('#scoreRunNow')?.addEventListener('click', scoreRun);
    $('#runTriggerBatchSize')?.addEventListener('input', refreshRunButtonLabel);
    $('#runTriggerSampleIds')?.addEventListener('input', refreshRunButtonLabel);
    $('#runTriggerModels')?.addEventListener('change', event => {
      const input = event.target;
      if (!(input instanceof HTMLInputElement) || !input.classList.contains('local-reasoning-input')) return;
      const model = input.dataset.localReasoningModel || '';
      if (!isLocalModel(model)) return;
      localReasoningOverrides[model] = input.checked === true;
      writeLocalReasoningOverrides();
      syncLocalReasoningControls();
    });
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
    populateCascadeEscalate();
    bind();
    refreshRunButtonLabel();
    await rushApiLoadCatalog();
    populatePolicies();
    status('Local API connected. Set k per split (split=all runs up to k train + k test).');
    resumeActiveRun();
    refreshModelSpeedEstimates();  // fill in measured seconds/image (async, best-effort)
  }

  // On (re)load, re-attach to an already-running run so a browser refresh does
  // not abandon the live view (the job keeps running server-side regardless).
  async function resumeActiveRun() {
    if (state.runId) return;
    try {
      const data = await rushApiGetJson('/api/runs');
      const runs = (data && data.runs) || [];
      const active = runs.find(r => r && r.running === true && r.run_id);
      if (!active) return;
      state.runId = active.run_id;
      state.runModels = Array.isArray(active.models) ? active.models : [];
      state.pollStartedAt = Date.now();
      state.finished = false;
      status(`Resumed watching in-flight run ${state.runId}\u2026`);
      await pollStatus();
    } catch (error) {
      /* non-fatal: no active run or API hiccup */
    }
  }

  rushApiOnReady(initRunTrigger);
  window.addEventListener('rush-api-catalog', populatePolicies);
})();
