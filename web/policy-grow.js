(() => {
  const DEFAULT_MODEL = 'openai/gpt-5.5';
  const ALT_MODEL = 'anthropic/claude-opus-4-7';
  const DEFAULT_TASK_DESCRIPTION = 'Classify whether a given image is AI-generated. Use visual evidence (hand/finger anatomy, text/typography glitches, surface texture, scene geometry/reflections) as positive evidence; treat conventional photo edits, CGI/game renders, and low-quality uncertain inputs as boundaries; surface explicit synthetic provenance as a separate evidence class.';

  const state = {
    mode: 'warm',
    baseVersion: null,
    runId: null,
    batchIndex: 0,
    batchSize: 20,
    latestProposalId: null,
    history: [],
    busy: false
  };

  const qs = selector => document.querySelector(selector);
  const esc = value => String(value ?? '').replace(/[&<>'"]/g, char => ({
    '&': '&amp;',
    '<': '&lt;',
    '>': '&gt;',
    "'": '&#39;',
    '"': '&quot;'
  }[char]));
  const attr = esc;

  function status(message, isError = false) {
    rushApiStatus('#policyGrowStatus', message, isError);
  }

  function selectedDemoMode() {
    return qs('[name="demoMode"]:checked')?.value || 'cold_start';
  }

  function modeBaseVersion(mode = state.mode) {
    if (mode === 'warm_v0_1') return 'v0.1';
    if (mode === 'warm_v0_2') return 'v0.2';
    return window.RUSH_API?.catalog?.currentPolicyVersion || qs('#policyGraphVersion')?.value || 'v0.1';
  }

  function latestRunId() {
    return qs('#runPicker')?.value || window.RUSH_API?.catalog?.runs?.[0]?.run_id || '';
  }

  function modelOptions(selected = DEFAULT_MODEL) {
    return [DEFAULT_MODEL, ALT_MODEL]
      .map(model => `<option value="${attr(model)}"${model === selected ? ' selected' : ''}>${esc(model)}</option>`)
      .join('');
  }

  function setBusy(busy) {
    state.busy = !!busy;
    updateControls();
  }

  function ensureControls() {
    const slot = qs('#policyGrowControlsSlot');
    if (!slot || slot.dataset.policyGrowReady === 'true') return slot;
    slot.dataset.policyGrowReady = 'true';
    slot.innerHTML = `
      <div class="policy-grow-header">
        <div>
          <h3>Seed or suggest the next graph update</h3>
          <p class="body-copy">Use cold start to seed a graph, or use a scored labeling round to draft SME-reviewable changes.</p>
        </div>
        <span id="policyGrowStatus" class="status-line" role="status"></span>
      </div>

      <div class="policy-grow-mode-cold" data-policy-grow-panel="cold">
        <label>Task description
          <textarea id="policyGrowTaskDescription" rows="4">${esc(DEFAULT_TASK_DESCRIPTION)}</textarea>
        </label>
        <div class="policy-grow-form-row">
          <label>Model
            <select id="policyGrowColdModel">${modelOptions()}</select>
          </label>
          <button id="policyGrowSeedBtn" class="primary-action" type="button">Seed policy graph</button>
        </div>
      </div>

      <div class="policy-grow-mode-warm" data-policy-grow-panel="warm" hidden>
        <div class="policy-grow-chip-row">
          <span class="tag tag-current" id="policyGrowBaseVersionChip">Base version: —</span>
          <span class="tag tag-current" id="policyGrowRunChip">Run: —</span>
          <span class="policy-grow-disabled-notice" id="policyGrowRunNotice" hidden>Run a labeling pass first</span>
        </div>
        <div class="policy-grow-form-row">
          <label>Batch size
            <input id="policyGrowBatchSize" type="number" min="2" value="20" />
          </label>
          <label>Next batch
            <output id="policyGrowBatchIndex">0</output>
          </label>
          <label>Model
            <select id="policyGrowWarmModel">${modelOptions()}</select>
          </label>
          <button id="policyGrowRunBatchBtn" class="primary-action" type="button">Suggest changes from batch</button>
        </div>
        <p id="policyGrowBatchMeta" class="policy-grow-batch-meta">After review, accept or reject the proposal below, then return to §3 for the next labeling round.</p>
      </div>`;

    qs('#policyGrowSeedBtn')?.addEventListener('click', seedColdStart);
    qs('#policyGrowRunBatchBtn')?.addEventListener('click', runNextBatch);
    qs('#policyGrowBatchSize')?.addEventListener('input', event => {
      state.batchSize = Math.max(2, Number.parseInt(event.target.value, 10) || 20);
    });
    return slot;
  }

  function ensureHistoryContainer() {
    const grow = qs('#grow');
    if (!grow) return null;
    let history = qs('.policy-version-history');
    if (history) return history;
    history = document.createElement('div');
    history.className = 'policy-version-history';
    history.setAttribute('aria-label', 'Policy version history');
    const proposals = qs('.policy-proposals-block');
    if (proposals) proposals.insertAdjacentElement('afterend', history);
    else grow.appendChild(history);
    return history;
  }

  function updateControls() {
    const slot = ensureControls();
    if (!slot) return;
    state.mode = selectedDemoMode();
    state.baseVersion = state.mode === 'cold_start' ? null : modeBaseVersion(state.mode);
    state.runId = latestRunId();

    const isCold = state.mode === 'cold_start';
    const coldPanel = qs('[data-policy-grow-panel="cold"]');
    const warmPanel = qs('[data-policy-grow-panel="warm"]');
    if (coldPanel) coldPanel.hidden = !isCold;
    if (warmPanel) warmPanel.hidden = isCold;

    const baseChip = qs('#policyGrowBaseVersionChip');
    if (baseChip) baseChip.textContent = `Base version: ${state.baseVersion || '—'}`;
    const runChip = qs('#policyGrowRunChip');
    if (runChip) runChip.textContent = state.runId ? `Run: ${state.runId}` : 'Run: —';
    const runNotice = qs('#policyGrowRunNotice');
    if (runNotice) runNotice.hidden = !!state.runId;
    const batchIndex = qs('#policyGrowBatchIndex');
    if (batchIndex) batchIndex.value = String(state.batchIndex);

    const seedBtn = qs('#policyGrowSeedBtn');
    if (seedBtn) seedBtn.disabled = state.busy || !window.RUSH_API?.available;
    const batchBtn = qs('#policyGrowRunBatchBtn');
    if (batchBtn) batchBtn.disabled = state.busy || !window.RUSH_API?.available || !state.runId;
  }

  function renderHistory() {
    const history = ensureHistoryContainer();
    if (!history) return;
    const versions = (window.RUSH_API?.catalog?.policyVersions || [])
      .map(item => item?.version || item)
      .filter(Boolean);
    state.history = versions;
    if (!versions.length) {
      history.innerHTML = '<span class="muted">No policy versions found yet.</span>';
      return;
    }
    const current = window.RUSH_API?.catalog?.currentPolicyVersion || versions[versions.length - 1];
    history.innerHTML = versions.map(version => {
      const isCurrent = version === current;
      return `<button type="button" class="version-chip${isCurrent ? ' current' : ''}" data-policy-version="${attr(version)}">${esc(version)}${isCurrent ? ' · current' : ''}</button>`;
    }).join('');
    history.querySelectorAll('[data-policy-version]').forEach(button => {
      button.addEventListener('click', () => {
        const select = qs('#policyGraphVersion');
        if (!select) return;
        select.value = button.dataset.policyVersion || '';
        select.dispatchEvent(new Event('change', { bubbles: true }));
      });
    });
  }

  async function routeProposalToDiff(proposalId) {
    const full = await rushApiGetJson(`/api/policy/proposals/${encodeURIComponent(proposalId)}`);
    window.dispatchEvent(new CustomEvent('rush-proposal-loaded', { detail: full }));
    await rushApiLoadCatalog();
    if (typeof window.rushRefreshPolicyProposals === 'function') await window.rushRefreshPolicyProposals(proposalId);
    if (typeof window.rushLoadProposal === 'function') await window.rushLoadProposal(proposalId);
    renderHistory();
    return full;
  }

  async function seedColdStart() {
    const taskDescription = qs('#policyGrowTaskDescription')?.value?.trim() || DEFAULT_TASK_DESCRIPTION;
    const modelId = qs('#policyGrowColdModel')?.value || DEFAULT_MODEL;
    try {
      setBusy(true);
      status('Calling LLM (this may take 30s)…');
      const resp = await rushApiPostJson('/api/policy/cold-start', {
        task_description: taskDescription,
        domain: 'Generative_AI',
        model_id: modelId
      });
      state.latestProposalId = resp.proposal_id || null;
      if (state.latestProposalId) await routeProposalToDiff(state.latestProposalId);
      status(`Created cold-start proposal ${state.latestProposalId || 'unknown'}.`);
    } catch (error) {
      status(error.message || 'Cold-start proposal failed.', true);
    } finally {
      setBusy(false);
    }
  }

  async function runNextBatch() {
    state.runId = latestRunId();
    state.baseVersion = modeBaseVersion(state.mode);
    state.batchSize = Math.max(2, Number.parseInt(qs('#policyGrowBatchSize')?.value, 10) || state.batchSize || 20);
    const batchIndex = state.batchIndex;
    const modelId = qs('#policyGrowWarmModel')?.value || DEFAULT_MODEL;
    if (!state.runId) {
      status('Run a labeling pass first.', true);
      return;
    }
    try {
      setBusy(true);
      status('Calling LLM (this may take 30s)…');
      const resp = await rushApiPostJson('/api/policy/grow-batch', {
        run_id: state.runId,
        base_version: state.baseVersion,
        batch_index: batchIndex,
        batch_size: state.batchSize,
        model_id: modelId
      });
      state.latestProposalId = resp.proposal_id || null;
      if (state.latestProposalId) await routeProposalToDiff(state.latestProposalId);
      const batch = resp.batch || {};
      const positives = batch.n_positives ?? 0;
      const negatives = batch.n_negatives ?? 0;
      const meta = qs('#policyGrowBatchMeta');
      if (meta) meta.textContent = `Batch ${batchIndex}: ${positives} positives, ${negatives} negatives`;
      state.batchIndex = batchIndex + 1;
      updateControls();
      status(`Created grow-batch proposal ${state.latestProposalId || 'unknown'}.`);
    } catch (error) {
      status(error.message || 'Grow-batch proposal failed.', true);
    } finally {
      setBusy(false);
    }
  }

  async function initPolicyGrow(api) {
    ensureControls();
    ensureHistoryContainer();
    if (api.available) await rushApiLoadCatalog();
    updateControls();
    renderHistory();
    if (!api.available) status('Local API offline — start the rush web server to grow policy.', true);
  }

  document.querySelectorAll('[name="demoMode"]').forEach(radio => {
    radio.addEventListener('change', () => {
      updateControls();
      renderHistory();
    });
  });
  qs('#runPicker')?.addEventListener('change', updateControls);
  window.addEventListener('rush-api-catalog', () => {
    updateControls();
    renderHistory();
  });
  window.addEventListener('rush-policy-accepted', async () => {
    if (window.RUSH_API?.available) await rushApiLoadCatalog();
    renderHistory();
  });

  rushApiOnReady(initPolicyGrow);
})();
