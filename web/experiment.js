// The RUSH loop — the main view. One experiment = a numbered, seeded PPO
// iteration run: k cycles of train mini-batch -> misalignment anchors ->
// one clipped policy edit -> candidate eval on a fixed test partition ->
// gate (auto-accept iff system macro-F1 improves; the gate agent may veto).
// This module owns: the view switcher (loop | inspect), the start panel
// (judge picker lives here), the learning curve (auto-scaled y), the per-run
// judge table, the gate ledger with expandable proposal diffs + anchor
// images, SME reviews (critic-of-the-critic), and KG auto-follow as the
// policy evolves.
(() => {
  const POLL_MS = 2500;
  const $ = (sel) => document.querySelector(sel);
  const esc = (value) => String(value ?? '').replace(/[&<>"']/g, (ch) => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
  })[ch]);

  const METRICS = [
    ['macro_f1', 'Macro F1 (gate metric)'],
    ['accuracy', 'Accuracy'],
    ['macro_precision', 'Precision (macro)'],
    ['macro_recall', 'Recall (macro)'],
    ['macro_fpr', 'FPR (macro)'],
    ['macro_fnr', 'FNR (macro)']
  ];
  // Theme palette (styles.css :root) — the page is dark.
  const SERIES_COLORS = ['#82b5ff', '#ffd166', '#4de0a6', '#d394ff', '#ff6f91', '#6fe3e0'];
  const SYSTEM_COLOR = '#edf4ff';
  const GRID_COLOR = '#2c3e68';
  const AXIS_TEXT = '#aab8d3';

  const state = {
    list: [],
    current: null,
    pollTimer: null,
    expandedCycles: new Set(),
    detailCache: {},       // proposal_id -> diffs payload
    anchorCache: {},       // train_run_id -> misalignment records
    followedVersion: null, // KG auto-follow bookkeeping
    kgCycleK: null,        // KG cycle stepper: which k the graph is showing
    kgManual: false,       // true once the SME steps the graph by hand
    pendingRunNumber: null, // run just started — auto-select it when it appears
    mintPollTimer: null,    // GenAI split mint in flight — poll until it lands
    splitSeedTouched: false, // user edited the split seed; stop auto-filling it
    bundleCache: {},       // `${area}/${version}` -> { ids, texts } node markdown
    runSummaryCache: {},   // child run_id -> web/summary.json payload (successes only)
    policyChangesToken: 0, // guards overlapping async node-diff renders
    confusionToken: 0,     // guards overlapping async confusion-grid renders
    renderSigs: {},        // panel -> JSON signature of its inputs (skip
                           // unchanged re-renders so polling never yanks the
                           // scroll out from under an expanded evidence row)
    jobToken: null,        // registry job id of the in-flight run (live card)
    liveTimer: null,
    listRetryTimer: null   // keeps polling the list while pendingRunNumber set
  };

  // Re-render a panel only when its input signature changed.
  function sigChanged(key, value) {
    const sig = JSON.stringify(value);
    if (state.renderSigs[key] === sig) return false;
    state.renderSigs[key] = sig;
    return true;
  }

  // ---- view switcher (loop | summary | adjudicate | benchmarks | about) -----

  const VIEWS = ['loop', 'summary', 'adjudicate', 'benchmarks', 'about'];

  function applyView(view) {
    VIEWS.forEach((v) => document.body.classList.toggle(`view-${v}`, view === v));
    document.querySelectorAll('#viewSwitcher .view-switcher-option').forEach((button) => {
      button.setAttribute('aria-pressed', String(button.dataset.view === view));
    });
    try { sessionStorage.setItem('rush_view', view); } catch (err) { /* private mode */ }
    window.dispatchEvent(new CustomEvent('rush-view-changed', { detail: { view } }));
  }

  function initViewSwitcher() {
    const switcher = $('#viewSwitcher');
    if (!switcher) return;
    switcher.querySelectorAll('.view-switcher-option').forEach((button) => {
      button.addEventListener('click', () => applyView(button.dataset.view));
    });
    const fromHash = location.hash.replace('#', '');
    let view = 'loop';
    try { view = sessionStorage.getItem('rush_view') || 'loop'; } catch (err) { /* ok */ }
    if (!VIEWS.includes(view)) view = 'loop';
    if (fromHash === 'experiment' || fromHash === 'policyEvolution') view = 'loop';
    else if (VIEWS.includes(fromHash)) view = fromHash;
    applyView(view);
    window.addEventListener('hashchange', () => {
      const anchor = location.hash.replace('#', '');
      if (anchor === 'experiment' || anchor === 'policyEvolution') applyView('loop');
      else if (VIEWS.includes(anchor)) applyView(anchor);
    });
  }

  // ---- helpers --------------------------------------------------------------

  function activeArea() {
    const demo = typeof window.rushActiveDemo === 'function' ? window.rushActiveDemo() : null;
    return demo?.policyGraph?.area || 'Generative_AI';
  }

  function activeDemoId() {
    const demo = typeof window.rushActiveDemo === 'function' ? window.rushActiveDemo() : null;
    return demo?.id || 'genai';
  }

  function selectedPanelModels() {
    return Array.from(
      document.querySelectorAll('#runTriggerModels input.model-select-input[type="checkbox"]:checked')
    ).map((input) => input.value).filter(Boolean);
  }

  function fmtPct(value, digits = 1) {
    return (value === null || value === undefined) ? '—' : `${(value * 100).toFixed(digits)}%`;
  }

  function fmtUsd(value) {
    return (typeof value === 'number') ? `$${value.toFixed(value >= 1 ? 2 : 4)}` : '—';
  }

  function nextRunNumber() {
    const nums = state.list.map((e) => e.run_number).filter((n) => Number.isInteger(n));
    return nums.length ? Math.max(...nums) + 1 : 1;
  }

  function statusChip(status) {
    const cls = {
      accepted: 'experiment-chip experiment-chip--accepted',
      skipped: 'experiment-chip experiment-chip--skipped',
      no_misalignments: 'experiment-chip experiment-chip--neutral',
      failed: 'experiment-chip experiment-chip--failed',
      stopped: 'experiment-chip experiment-chip--neutral',
      open: 'experiment-chip experiment-chip--neutral'
    }[status] || 'experiment-chip experiment-chip--neutral';
    const label = {
      accepted: 'accepted',
      skipped: 'skipped',
      no_misalignments: 'aligned — no edit',
      failed: 'failed',
      stopped: 'stopped',
      open: 'running'
    }[status] || status;
    return `<span class="${cls}">${esc(label)}</span>`;
  }

  // ---- start ----------------------------------------------------------------

  function renderPanelSummary() {
    const models = selectedPanelModels();
    const summaryEl = $('#experimentPanelSummary');
    if (summaryEl) {
      summaryEl.textContent = models.length
        ? `Panel (${models.length}): ${models.join(', ')} · fixed generator k=0 = v0.1 · holdout locked`
        : 'Pick 2–5 judge models above.';
    }
    const button = $('#experimentStart');
    if (button) button.textContent = `Start run #${nextRunNumber()}`;
  }

  async function startExperiment() {
    const statusEl = $('#experimentStartStatus');
    const button = $('#experimentStart');
    const models = selectedPanelModels();
    if (models.length < 2 || models.length > 5) {
      statusEl.textContent = 'Pick 2–5 judge models above first.';
      return;
    }
    if ($('#experimentAllowSpend')?.checked !== true) {
      statusEl.textContent = 'Tick "Allow spend" — runs call live judge APIs.';
      return;
    }
    const seedRaw = ($('#experimentSeed')?.value || '').trim();
    const gateChoice = $('#experimentGateModel')?.value || 'metric_only';
    // Select values: 'metric_only' | 'off' | '<model>' (metric rule + agent
    // veto) | 'agent_only:<model>' (the critic's verdict alone decides —
    // metric recorded as advisory, never enforced).
    const agentOnly = gateChoice.startsWith('agent_only:');
    const gateMode = gateChoice === 'metric_only' ? 'metric_only'
      : (gateChoice === 'off' ? 'off' : (agentOnly ? 'agent_only' : 'agent'));
    const gateModel = agentOnly ? gateChoice.slice('agent_only:'.length) : gateChoice;
    const payload = {
      demo: activeDemoId(),
      area: activeArea(),
      models,
      seed: seedRaw ? Number(seedRaw) : null,
      k_max: Number($('#experimentKMax')?.value || 5),
      batch_n: Number($('#experimentBatchN')?.value || 20),
      test_n: Number($('#experimentTestN')?.value || 100),
      max_changes: Number($('#experimentMaxChanges')?.value || 3),
      max_anchors: Number($('#experimentMaxAnchors')?.value || 15),
      max_aligned_anchors: Number($('#experimentMaxAlignedAnchors')?.value ?? 5),
      gate_mode: gateMode,
      gate_model: (gateMode === 'agent' || gateMode === 'agent_only')
        ? gateModel : 'openai/gpt-5.5',
      gate_persona: $('#experimentGatePersona')?.value || 'lenient',
      drafter_model: $('#experimentDrafterModel')?.value || 'openai/gpt-5.5',
      drafter_context: $('#experimentDrafterContext')?.value || 'text_only',
      strategy: $('#experimentStrategy')?.value || 'random_misalignment',
      // Fixed cross-run benchmark readout (validation split, start + final).
      validation_final: $('#experimentValidationFinal')?.checked === true,
      // k=0 is FIXED: every run starts from the same baseline generator.
      policy_version: null,
      live: true,
      allow_spend: true
    };
    const pinInput = $('#experimentLaunchPin');
    const launchPin = String(pinInput?.value || '').trim();
    if (!launchPin) {
      statusEl.textContent = 'Enter code to run.';
      pinInput?.focus();
      return;
    }
    payload.launch_pin = launchPin;
    button.disabled = true;
    const runNumber = nextRunNumber();
    statusEl.textContent = `Starting run #${runNumber}…`;
    try {
      const started = await window.rushApiPostJson('/api/experiments/start', payload);
      state.pendingRunNumber = runNumber;
      state.jobToken = started?.job_id || started?.run_id || null;
      statusEl.textContent = `Run #${runNumber} starting — it is auto-selected below as soon as the driver checks in.`;
      // Keep polling the list until the driver's state file shows up (a slow
      // first cycle used to strand the panel on the PREVIOUS run forever).
      if (state.listRetryTimer) window.clearInterval(state.listRetryTimer);
      let attempts = 0;
      state.listRetryTimer = window.setInterval(() => {
        attempts += 1;
        if (state.pendingRunNumber === null || attempts > 40) {
          window.clearInterval(state.listRetryTimer);
          state.listRetryTimer = null;
          return;
        }
        loadList(false);
      }, 3000);
      window.setTimeout(() => loadList(false), 1500);
    } catch (err) {
      statusEl.textContent = `Start failed: ${err?.message || err}`;
    } finally {
      if (pinInput) pinInput.value = '';
      button.disabled = false;
    }
  }

  // ---- list + detail --------------------------------------------------------

  async function loadList(preserveSelection = true) {
    let payload;
    try {
      payload = await window.rushApiGetJson('/api/experiments');
    } catch (err) {
      $('#experimentStatusLine').textContent = 'Experiments API unavailable.';
      return;
    }
    const area = activeArea();
    state.list = (payload?.experiments || []).filter((e) => e.area === area);
    const select = $('#experimentSelect');
    const previous = preserveSelection ? select.value : '';
    select.innerHTML = state.list.length
      ? state.list.map((e) => {
        const stamp = (e.started_at || '').slice(0, 16).replace('T', ' ');
        const label = `Run #${e.run_number ?? '?'} · seed ${e.seed} · ${e.status}${e.dry_run ? ' · dry-run' : ''} · ${stamp}`;
        return `<option value="${esc(e.experiment_id)}">${esc(label)}</option>`;
      }).join('')
      : '<option value="">No runs yet — start run #1 above</option>';
    if (previous && state.list.some((e) => e.experiment_id === previous)) {
      select.value = previous;
    }
    // A run the SME just started wins the selection the moment it appears.
    if (state.pendingRunNumber !== null) {
      const started = state.list.find((e) => e.run_number === state.pendingRunNumber);
      if (started) {
        select.value = started.experiment_id;
        state.pendingRunNumber = null;
        state.kgManual = false;
        state.kgCycleK = null;
        state.followedVersion = null;
        state.expandedCycles.clear();
        state.renderSigs = {};
        const startStatus = $('#experimentStartStatus');
        if (startStatus) startStatus.textContent = `Run #${started.run_number} is live below.`;
      }
    }
    renderPanelSummary();
    if (select.value) {
      await loadDetail(select.value);
    } else {
      state.current = null;
      renderDetail();
    }
  }

  async function loadDetail(experimentId) {
    if (!experimentId) return;
    try {
      state.current = await window.rushApiGetJson(`/api/experiments/${encodeURIComponent(experimentId)}`);
    } catch (err) {
      $('#experimentStatusLine').textContent = `Failed to load ${experimentId} — retrying.`;
      schedulePoll();
      return;
    }
    renderDetail();
    schedulePoll();
  }

  function schedulePoll() {
    if (state.pollTimer) window.clearTimeout(state.pollTimer);
    state.pollTimer = null;
    if (state.current?.status === 'running') {
      state.pollTimer = window.setTimeout(() => {
        loadDetail(state.current.experiment_id);
      }, POLL_MS);
    }
  }

  // ---- KG auto-follow -------------------------------------------------------

  function kgShowVersion(version, scroll = false) {
    const versionSelect = document.querySelector('#policyGraphVersion');
    if (versionSelect && version) {
      if (!Array.from(versionSelect.options).some((o) => o.value === version)) {
        const option = document.createElement('option');
        option.value = version;
        option.textContent = version;
        versionSelect.appendChild(option);
      }
      versionSelect.value = version;
      versionSelect.dispatchEvent(new Event('change'));
    }
    if (scroll) {
      document.querySelector('#policyEvolution')?.scrollIntoView({ behavior: 'smooth' });
    }
  }

  function syncPolicyGraphVersion() {
    // Keep the KG panel on the version the SELECTED run implies: the version
    // in force at the stepper's k, the base version for a fresh run with no
    // closed cycles yet (a restarted run must snap back to k=0), the newest
    // accepted version while a run is live. Manual chip-stepping pauses this
    // for the run.
    const exp = state.current;
    if (!exp || state.kgManual) return;
    const cycles = (exp.cycles || []).filter((c) => typeof c.k === 'number' && c.status !== 'open');
    const version = cycles.length
      ? versionInForceAfter(cycles, state.kgCycleK ?? cycles[cycles.length - 1].k)
      : (exp.base_version || 'v0.1');
    if (version && version !== state.followedVersion) {
      state.followedVersion = version;
      kgShowVersion(version, false);
    }
  }

  // ---- policy evolution ↔ cycles: step the graph through k ------------------

  function versionInForceAfter(cycles, k) {
    let version = state.current?.base_version || 'v0.1';
    cycles.forEach((c) => {
      if (c.k <= k && c.status === 'accepted' && c.new_version) version = c.new_version;
    });
    return version;
  }

  function renderKgCycles() {
    const host = $('#experimentKgCycles');
    if (!host) return;
    const exp = state.current;
    const cycles = (exp?.cycles || []).filter((c) => typeof c.k === 'number' && c.status !== 'open');
    if (!exp || !cycles.length) { host.innerHTML = ''; return; }
    const ks = cycles.map((c) => c.k);
    if (!state.kgManual || state.kgCycleK === null || !ks.includes(state.kgCycleK)) {
      state.kgCycleK = ks[ks.length - 1];
    }
    const chips = cycles.map((c) => {
      const accepted = c.status === 'accepted';
      const active = c.k === state.kgCycleK;
      const text = c.k === 0
        ? `k=0 ${esc(exp.base_version)}`
        : (accepted ? `k=${c.k} → ${esc(c.new_version)}` : `k=${c.k}`);
      const title = c.k === 0
        ? 'baseline generator (fixed for every run)'
        : (accepted ? `accepted — policy became ${c.new_version}` : `${c.status} — policy unchanged`);
      return `<button type="button" class="experiment-kg-chip${active ? ' experiment-kg-chip--active' : ''}${accepted ? ' experiment-kg-chip--accepted' : ''}" data-kg-k="${c.k}" title="${esc(title)}">${text}</button>`;
    }).join('');
    const version = versionInForceAfter(cycles, state.kgCycleK);
    host.innerHTML = `
      <span class="experiment-kg-strip-label">Graph by cycle</span>
      <button type="button" class="experiment-kg-step" data-kg-step="-1" aria-label="Previous cycle">‹</button>
      <div class="experiment-kg-chip-row">${chips}</div>
      <button type="button" class="experiment-kg-step" data-kg-step="1" aria-label="Next cycle">›</button>
      <span class="experiment-kg-note">after k=${state.kgCycleK}: <strong>${esc(version)}</strong> in force</span>`;
    const applyK = (k) => {
      state.kgCycleK = k;
      state.kgManual = true;
      kgShowVersion(versionInForceAfter(cycles, k), false);
      renderKgCycles();
      renderPolicyChanges();
    };
    host.querySelectorAll('.experiment-kg-chip').forEach((chip) => {
      chip.addEventListener('click', () => applyK(Number(chip.dataset.kgK)));
    });
    host.querySelectorAll('.experiment-kg-step').forEach((button) => {
      button.addEventListener('click', () => {
        const idx = ks.indexOf(state.kgCycleK);
        const next = ks[Math.min(ks.length - 1, Math.max(0, idx + Number(button.dataset.kgStep)))];
        applyK(next);
      });
    });
  }

  // ---- rendering ------------------------------------------------------------

  function renderDetail() {
    const summary = $('#experimentSummary');
    const statusLine = $('#experimentStatusLine');
    const exp = state.current;
    if (!exp) {
      summary.innerHTML = '<p class="hint">No run selected. Configure the panel above and start run #1 — '
        + 'each run is fully reproducible from its seed.</p>';
      $('#experimentChart').innerHTML = '';
      $('#experimentJudgeTable').innerHTML = '';
      $('#experimentLedger').innerHTML = '';
      $('#experimentHoldout').innerHTML = '';
      ['#experimentKgCycles', '#experimentPolicyChanges', '#experimentConfusion'].forEach((sel) => {
        const el = $(sel);
        if (el) el.innerHTML = '';
      });
      // Invalidate any in-flight async renders so they can't repaint the
      // hosts we just cleared.
      state.policyChangesToken += 1;
      state.confusionToken += 1;
      state.renderSigs = {};
      state.jobToken = null;
      const liveCard = $('#experimentLiveCard');
      if (liveCard) { liveCard.hidden = true; liveCard.innerHTML = ''; }
      statusLine.textContent = '';
      return;
    }
    // Live runs stream their phase (incl. "137/500 calls · $0.42" while a
    // child labels); the spinner marks the run as in flight at a glance.
    if (exp.status === 'running') {
      statusLine.innerHTML = `<span class="spinner" aria-hidden="true"></span> ${esc(exp.phase || 'running…')}`;
    } else {
      statusLine.textContent = exp.phase || exp.status;
    }

    const cycles = exp.cycles || [];
    const accepted = cycles.filter((c) => c.status === 'accepted');
    const baseline = cycles.find((c) => c.k === 0);
    const latest = cycles[cycles.length - 1];
    const f1Start = baseline?.metrics?.test?.system?.macro_f1;
    const f1Now = latest?.metrics?.test?.system?.macro_f1;
    summary.innerHTML = `
      <div class="experiment-summary-grid">
        <div><span>Run</span><strong>#${esc(exp.run_number)} · seed ${esc(exp.seed)}${exp.dry_run ? ' · dry-run' : ''}</strong></div>
        <div><span>Policy</span><strong>${esc(exp.base_version)} → ${esc(exp.current_version)}</strong></div>
        <div><span>Accepted / cycles</span><strong>${accepted.length} / ${Math.max(0, cycles.length - 1)} of ${esc(exp.k_max)}</strong></div>
        <div><span>Test system F1</span><strong>${fmtPct(f1Start)} → ${fmtPct(f1Now)}</strong></div>
        <div><span>Splits</span><strong>test ${esc(exp.test_n)} · batch ${esc(exp.batch_n)}/cycle · holdout locked</strong></div>
        <div><span>Gate</span><strong>${exp.gate_mode === 'off'
          ? 'OFF — accepts every edit'
          : `${esc(exp.gate_model || 'metric rule')} (${esc(exp.gate_mode)}${(exp.gate_persona && exp.gate_mode !== 'metric_only') ? ` · ${esc(exp.gate_persona)}` : ''})`}</strong></div>
        <div title="The drafter that proposes each policy edit — its model, what it sees per anchor, and its total spend across all cycles so far"><span>Optimizer</span><strong>${esc(exp.drafter_model || 'openai/gpt-5.5')}${exp.drafter_context === 'text_only' ? ' · text only' : ''}${(() => {
          const spent = cycles.reduce((sum, c) => sum + (c.drafter?.cost_usd || 0), 0);
          return spent > 0 ? ` · ${fmtUsd(spent)}` : '';
        })()}</strong></div>
        <div><span>Cost</span><strong>${fmtUsd(exp.cost_usd_total)}${exp.status === 'completed'
          ? ' <span class="experiment-cost-note">final</span>'
          : (exp.status === 'running' ? ' <span class="experiment-cost-note">so far</span>' : '')}</strong></div>
      </div>`;

    // Heavy panels re-render ONLY when their inputs changed — the 2.5s poll
    // must never rebuild the DOM under an expanded evidence row (scroll
    // jumps). Control listeners call the renderers directly, bypassing sigs.
    // status is part of every signature: when a run flips running→completed
    // the cycle array may be unchanged, but the empty-state text and the
    // "in progress" chart hint must repaint once (and only once).
    const id = exp.experiment_id;
    const st = exp.status;
    if (sigChanged('chart', [id, st, cycles])) renderChart();
    if (sigChanged('judges', [id, st, cycles])) renderJudgeTable();
    if (sigChanged('ledger', [id, st, cycles])) renderLedger();
    if (sigChanged('holdout', [id, st, exp.holdout, exp.benchmark])) renderHoldout();
    if (sigChanged('kg', [id, st, cycles])) renderKgCycles();
    syncPolicyGraphVersion();
    if (sigChanged('confusion', [id, st, cycles])) renderConfusion();      // async; caches per child run
    if (sigChanged('policyChanges', [id, st, cycles, state.kgCycleK])) renderPolicyChanges(); // async; token-guarded
    renderLiveCard();       // async; own host, cheap
  }

  // ---- live labeling card (per-model progress + cancel) ---------------------

  async function discoverJobToken() {
    // Page reloaded while a run is in flight: find the running experiment job
    // in the registry so the live card + cancel button work again.
    try {
      const payload = await window.rushApiGetJson('/api/jobs?running=1');
      const jobs = payload?.jobs || [];
      const experimentId = state.current?.experiment_id;
      const running = jobs.find((j) => j?.kind === 'experiment'
        && (!j.experiment_id || !experimentId || j.experiment_id === experimentId));
      if (running) state.jobToken = running.job_id || null;
    } catch (err) { /* registry unavailable — the card stays hidden */ }
  }

  async function renderLiveCard() {
    const host = $('#experimentLiveCard');
    if (!host) return;
    const exp = state.current;
    if (!exp || exp.status !== 'running') {
      host.hidden = true;
      if (host.innerHTML) host.innerHTML = '';
      return;
    }
    if (!state.jobToken) await discoverJobToken();
    if (!state.jobToken) { host.hidden = true; return; }
    let payload;
    try {
      payload = await window.rushApiGetJson(`/api/runs/${encodeURIComponent(state.jobToken)}/status`);
    } catch (err) { host.hidden = true; return; }
    if (state.current?.status !== 'running') { host.hidden = true; return; }
    const completed = payload?.completed_calls ?? 0;
    const expected = payload?.expected_calls ?? 0;
    const cost = payload?.recorded_cost_usd ?? payload?.running_cost_usd_estimate;
    const pct = expected ? Math.min(100, Math.round((completed / expected) * 100)) : 0;
    // The registry tracks the CURRENT labeling child; when that child has
    // finalized but the job is still alive, the driver is between passes
    // (drafting / gating / scoring) — say so instead of showing a stuck bar.
    const betweenPasses = Boolean(payload?.finished_at);
    const fmtNum = (v, d = 1) => (typeof v === 'number' && Number.isFinite(v)) ? v.toFixed(d) : '—';
    const rows = (payload?.per_model || []).map((m) => {
      const name = String(m.model || m.model_id || '');
      const done = m.calls_done ?? m.n_calls ?? 0;
      const total = m.calls_total ?? '—';
      // Provider-consistent processed-token total from the backend (Anthropic
      // cache reads/writes added back in); fall back to in+out for old runs.
      const toks = m.total_tokens
        || ((m.total_input_tokens || 0) + (m.total_output_tokens || 0));
      const cached = m.total_cached_input_tokens || 0;
      const cachedPct = (toks > 0 && cached > 0) ? Math.round((cached / toks) * 100) : 0;
      const cachedNote = cachedPct > 0
        ? `<span class="hint" title="${cached.toLocaleString()} input tokens served from the provider's prompt cache at a discounted rate"> · ${cachedPct}% cached</span>`
        : '';
      const modelCost = m.total_cost_usd ?? m.total_cost;
      return `<tr class="${m.done ? 'experiment-live-row--done' : ''}">
        <td>${esc(name)}<span class="hint"> ${esc(done)}/${esc(total)} call(s)${m.done ? ' · done' : ''}</span></td>
        <td>${fmtNum(m.avg_s_per_call, 2)}</td>
        <td>${fmtNum(m.tokens_per_sec)}</td>
        <td>${fmtNum(m.images_per_min ?? m.throughput_imgs_per_min)}</td>
        <td>${toks ? toks.toLocaleString() : '—'}${cachedNote}</td>
        <td>${typeof modelCost === 'number' ? `$${modelCost.toFixed(4)}` : '—'}</td>
      </tr>`;
    }).join('');
    host.hidden = false;
    host.innerHTML = `
      <div class="experiment-live-head">
        <strong>${betweenPasses ? 'Optimizing (between labeling passes)' : 'Labeling now'}</strong>
        <span class="hint experiment-live-phase">${esc(exp.phase || payload?.run_id || '')}</span>
        <span class="experiment-live-progress">${esc(completed)} / ${esc(expected)} calls${typeof cost === 'number' ? ` · $${cost.toFixed(4)}` : ''}${betweenPasses ? ' · last pass' : ''}</span>
        <button type="button" class="experiment-live-cancel" data-cancel-run>Cancel run</button>
      </div>
      <div class="experiment-live-bar${betweenPasses ? ' experiment-live-bar--idle' : ''}"><div style="width:${pct}%"></div></div>
      ${rows ? `<table class="experiment-live-table">
        <thead><tr><th>Model</th><th>Avg s/call</th><th>Tokens/sec</th><th>Images/min</th><th>Total tokens</th><th>Cost</th></tr></thead>
        <tbody>${rows}</tbody>
      </table>` : ''}`;
  }

  async function cancelCurrentRun() {
    if (!state.jobToken) return;
    if (!window.confirm('Cancel the in-flight run? The driver finalizes what it already paid for.')) return;
    try {
      await window.rushApiPostJson(`/api/runs/${encodeURIComponent(state.jobToken)}/cancel`, {});
      const statusLine = $('#experimentStatusLine');
      if (statusLine) statusLine.textContent = 'Cancel requested — the run finalizes as stopped.';
    } catch (err) {
      const statusLine = $('#experimentStatusLine');
      if (statusLine) statusLine.textContent = `Cancel failed: ${err?.message || err}`;
    }
  }

  function renderChartLegend(xMode, showTrain) {
    const el = $('#experimentChartLegend');
    if (!el) return;
    const swatch = (dash) => `<svg class="chart-legend-swatch" width="18" height="6" aria-hidden="true">`
      + `<line x1="1" y1="3" x2="17" y2="3" stroke="${SYSTEM_COLOR}" stroke-width="2"${dash ? ' stroke-dasharray="4 3" opacity="0.7"' : ''}/></svg>`;
    const bits = [`${swatch(false)} test — the run's fixed gate partition`];
    if (showTrain) bits.push(`${swatch(true)} train mini-batch — a fresh sample each k`);
    bits.push('▲ accepted → new policy version');
    bits.push(xMode === 'steps'
      ? '○ skipped candidate — sampling noise, no new tick (nothing was learned)'
      : '○ skipped candidate’s score — the solid line holds the incumbent');
    el.innerHTML = bits.join(' <span class="legend-sep">·</span> ');
  }

  function niceDomain(values) {
    // Auto-scale so near-zero metrics (FPR/FNR) and near-one metrics (F1 on a
    // strong panel) are both readable; always pad, never invert.
    const defined = values.filter((v) => v !== null && v !== undefined);
    if (!defined.length) return [0, 1];
    let lo = Math.min(...defined);
    let hi = Math.max(...defined);
    const pad = Math.max((hi - lo) * 0.25, 0.01);
    lo = Math.max(0, lo - pad);
    hi = Math.min(1, hi + pad);
    if (hi - lo < 0.02) { // dead-flat series: open a readable window around it
      lo = Math.max(0, lo - 0.02);
      hi = Math.min(1, hi + 0.02);
    }
    return [lo, hi];
  }

  function renderChart() {
    const host = $('#experimentChart');
    const exp = state.current;
    const metricKey = $('#experimentMetric')?.value || 'macro_f1';
    const xMode = $('#experimentXAxis')?.value || 'steps';
    const showTrain = $('#experimentShowTrain')?.checked === true;
    renderChartLegend(xMode, showTrain);
    const cycles = (exp?.cycles || []).filter((c) => typeof c.k === 'number');
    if (!exp || cycles.length === 0) { host.innerHTML = ''; return; }

    // -- x layout ------------------------------------------------------------
    // steps mode (default): a tick exists ONLY for k=0 and accepted cycles.
    // If nothing was accepted, nothing was learned — the axis says so instead
    // of stretching flat inherited values across k. Skipped candidates render
    // as hollow ghosts parked after the step they failed to beat.
    const steps = [];
    const baselineCycle = cycles.find((c) => c.k === 0);
    if (baselineCycle) {
      steps.push({ x: 0, k: 0, label: exp.base_version || 'v0.1', cycle: baselineCycle });
    }
    cycles.filter((c) => c.k >= 1 && c.status === 'accepted').forEach((c) => {
      steps.push({ x: steps.length, k: c.k, label: c.new_version || `k=${c.k}`, cycle: c });
    });
    const ghostCycles = cycles.filter((c) => c.k >= 1 && c.status !== 'accepted' && c.status !== 'open');
    const ghostX = new Map();
    let trailingGhosts = false;
    if (xMode === 'steps' && steps.length) {
      steps.forEach((left, i) => {
        const right = steps[i + 1] || null;
        const inGap = ghostCycles.filter((c) => c.k > left.k && (!right || c.k < right.k));
        inGap.forEach((c, idx) => {
          ghostX.set(c.k, left.x + (idx + 1) / (inGap.length + 1));
        });
        if (!right && inGap.length) trailingGhosts = true;
      });
    }
    const kMax = Math.max(1, ...cycles.map((c) => c.k));
    const xMax = xMode === 'steps'
      ? Math.max(1, (steps.length - 1) + (trailingGhosts ? 1 : 0))
      : kMax;
    const xOf = (k) => {
      if (xMode !== 'steps') return k;
      const step = steps.find((s) => s.k === k);
      if (step) return step.x;
      if (ghostX.has(k)) return ghostX.get(k);
      return steps.length ? steps[steps.length - 1].x : 0;
    };

    // -- series ---------------------------------------------------------------
    const scorers = new Set();
    cycles.forEach((c) => {
      Object.keys(c.metrics?.test || {}).forEach((s) => scorers.add(s));
      Object.keys(c.metrics?.train || {}).forEach((s) => scorers.add(s));
    });
    const ordered = Array.from(scorers).sort((a, b) => (
      (a === 'system') - (b === 'system') || a.localeCompare(b)
    ));
    const lineCycles = xMode === 'steps' ? steps.map((s) => s.cycle) : cycles;
    const series = ordered.map((scorer, idx) => ({
      scorer,
      color: scorer === 'system' ? SYSTEM_COLOR : SERIES_COLORS[idx % SERIES_COLORS.length],
      test: lineCycles.map((c) => ({
        k: c.k, v: c.metrics?.test?.[scorer]?.[metricKey] ?? null, status: c.status
      })),
      train: cycles.filter((c) => c.k >= 1).map((c) => ({
        k: c.k, v: c.metrics?.train?.[scorer]?.[metricKey] ?? null
      }))
    }));
    // Rejected candidates' SYSTEM score — the noise the trust region filtered.
    const ghostPoints = ghostCycles.map((c) => ({
      k: c.k,
      status: c.status,
      v: c.metrics?.test_candidate?.system?.[metricKey] ?? null
    })).filter((p) => p.v !== null && p.v !== undefined);

    const allValues = series.flatMap((s) => [
      ...s.test.map((p) => p.v),
      ...(showTrain ? s.train.map((p) => p.v) : [])
    ]).concat(ghostPoints.map((p) => p.v));
    const [lo, hi] = niceDomain(allValues);
    const W = 760; const H = 280; const padL = 52; const padR = 128; const padT = 14; const padB = 40;
    const x = (k) => padL + (xOf(k) / xMax) * (W - padL - padR);
    const y = (v) => padT + (1 - (v - lo) / (hi - lo)) * (H - padT - padB);

    let svg = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Decision quality per ${xMode === 'steps' ? 'accepted policy step' : 'cycle'}">`;
    for (let grid = 0; grid <= 4; grid += 1) {
      const value = lo + (grid / 4) * (hi - lo);
      const digits = (hi - lo) < 0.05 ? 1 : 0;
      svg += `<line x1="${padL}" y1="${y(value)}" x2="${W - padR}" y2="${y(value)}" stroke="${GRID_COLOR}" stroke-width="1"/>`
        + `<text x="${padL - 8}" y="${y(value) + 4}" text-anchor="end" font-size="10" fill="${AXIS_TEXT}">${(value * 100).toFixed(digits)}%</text>`;
    }

    // -- x axis ---------------------------------------------------------------
    if (xMode === 'steps') {
      steps.forEach((s) => {
        svg += `<text x="${x(s.k)}" y="${H - padB + 16}" text-anchor="middle" font-size="10" fill="${s.k === 0 ? AXIS_TEXT : '#4de0a6'}">${esc(s.label)}</text>`
          + `<text x="${x(s.k)}" y="${H - padB + 28}" text-anchor="middle" font-size="9" fill="${AXIS_TEXT}">k=${s.k}</text>`;
      });
      if (trailingGhosts) {
        const cx = padL + ((steps[steps.length - 1].x + 0.5) / xMax) * (W - padL - padR);
        svg += `<text x="${cx}" y="${H - padB + 16}" text-anchor="middle" font-size="9" fill="${AXIS_TEXT}" opacity="0.7">skipped ○</text>`;
      }
      if (steps.length === 1 && !ghostPoints.length && exp.status === 'running') {
        svg += `<text x="${(padL + W - padR) / 2}" y="${padT + 18}" text-anchor="middle" font-size="10" fill="${AXIS_TEXT}">waiting for the first gate decision — a new tick appears only when an edit is accepted</text>`;
      }
    } else {
      for (let k = 0; k <= kMax; k += 1) {
        svg += `<text x="${x(k)}" y="${H - padB + 16}" text-anchor="middle" font-size="10" fill="${AXIS_TEXT}">k=${k}</text>`;
      }
      cycles.filter((c) => c.status === 'accepted' && c.new_version).forEach((c) => {
        svg += `<text x="${x(c.k)}" y="${H - padB + 28}" text-anchor="middle" font-size="9" fill="#4de0a6">${esc(c.new_version)}</text>`;
      });
    }

    const path = (points) => {
      let d = ''; let pen = false;
      points.forEach((p) => {
        if (p.v === null || p.v === undefined) { pen = false; return; }
        d += `${pen ? 'L' : 'M'}${x(p.k).toFixed(1)},${y(p.v).toFixed(1)}`;
        pen = true;
      });
      return d;
    };

    const labels = [];
    series.forEach((s) => {
      const width = s.scorer === 'system' ? 2.5 : 1.5;
      if (showTrain && path(s.train)) {
        svg += `<path d="${path(s.train)}" fill="none" stroke="${s.color}" stroke-width="${width}" stroke-dasharray="4 4" opacity="0.5"/>`;
      }
      if (path(s.test)) {
        svg += `<path d="${path(s.test)}" fill="none" stroke="${s.color}" stroke-width="${width}"/>`;
      }
      s.test.forEach((p) => {
        if (p.v === null || p.v === undefined) return;
        if (p.status === 'accepted') {
          svg += `<path d="M${x(p.k)},${y(p.v) - 5} l5,8 h-10 z" fill="${s.color}"><title>k=${p.k} accepted — ${(p.v * 100).toFixed(1)}%</title></path>`;
        } else if (p.status === 'skipped' || p.status === 'no_misalignments' || p.status === 'failed') {
          svg += `<circle cx="${x(p.k)}" cy="${y(p.v)}" r="2.6" fill="#0a1020" stroke="${s.color}" stroke-width="1.5"/>`;
        } else {
          svg += `<circle cx="${x(p.k)}" cy="${y(p.v)}" r="2.6" fill="${s.color}"/>`;
        }
      });
      const lastPoint = [...s.test].reverse().find((p) => p.v !== null && p.v !== undefined);
      if (lastPoint) {
        labels.push({
          text: s.scorer === 'system' ? 'system (majority)' : s.scorer.split('/').pop(),
          color: s.color,
          anchorY: y(lastPoint.v) + 3
        });
      }
    });

    // Rejected-candidate ghosts (dashed hollow circles, system score).
    ghostPoints.forEach((p) => {
      svg += `<circle cx="${x(p.k)}" cy="${y(p.v)}" r="3.2" fill="none" stroke="${SYSTEM_COLOR}" stroke-width="1.2" stroke-dasharray="2 2" opacity="0.65">`
        + `<title>k=${p.k} candidate ${esc(p.status)} — system ${(p.v * 100).toFixed(1)}%</title></circle>`;
    });

    // Right-edge series labels: greedy de-overlap (sort by y, enforce a
    // minimum gap, clamp to the plot, leader lines when displaced).
    labels.sort((a, b) => a.anchorY - b.anchorY);
    const gap = 12;
    let prevY = padT - gap;
    labels.forEach((l) => { l.y = Math.max(l.anchorY, prevY + gap); prevY = l.y; });
    let limit = H - padB - 2;
    for (let i = labels.length - 1; i >= 0; i -= 1) {
      if (labels[i].y > limit) labels[i].y = limit;
      limit = labels[i].y - gap;
    }
    labels.forEach((l) => {
      if (Math.abs(l.y - l.anchorY) > 5) {
        svg += `<line x1="${W - padR + 1}" y1="${l.anchorY - 3}" x2="${W - padR + 6}" y2="${l.y - 3}" stroke="${l.color}" stroke-width="0.8" opacity="0.55"/>`;
      }
      svg += `<text x="${W - padR + 8}" y="${l.y}" font-size="10" fill="${l.color}">${esc(l.text)}</text>`;
    });
    svg += '</svg>';
    host.innerHTML = svg;
  }

  function renderJudgeTable() {
    const host = $('#experimentJudgeTable');
    const exp = state.current;
    const cycles = exp?.cycles || [];
    if (!exp || !cycles.length) { host.innerHTML = ''; return; }
    const baseline = cycles.find((c) => c.k === 0)?.metrics?.test || {};
    const finalMetrics = cycles[cycles.length - 1]?.metrics?.test || {};
    const scorers = Array.from(new Set([...Object.keys(baseline), ...Object.keys(finalMetrics)]))
      .sort((a, b) => ((a === 'system') - (b === 'system')) || a.localeCompare(b));
    if (!scorers.length) { host.innerHTML = ''; return; }

    // Δ vs k=0 on every key metric; FPR/FNR are lower-is-better so the
    // green/red coloring inverts for them.
    const delta = (scorer, key, lowerBetter = false) => {
      const before = baseline[scorer]?.[key];
      const after = finalMetrics[scorer]?.[key];
      if (before === null || before === undefined || after === null || after === undefined) return '';
      const diff = after - before;
      if (Math.abs(diff) < 0.0005) return '<span class="experiment-delta experiment-delta--flat">—</span>';
      const good = lowerBetter ? diff < 0 : diff > 0;
      const cls = good ? 'experiment-delta--up' : 'experiment-delta--down';
      return `<span class="experiment-delta ${cls}">${diff > 0 ? '+' : ''}${(diff * 100).toFixed(1)}</span>`;
    };

    const rows = scorers.map((scorer) => {
      const m = finalMetrics[scorer] || {};
      const isSystem = scorer === 'system';
      const name = isSystem ? 'system (majority vote)' : scorer;
      return `<tr class="${isSystem ? 'experiment-judge-system' : ''}">
        <td>${esc(name)}</td>
        <td>${fmtPct(m.accuracy)} ${delta(scorer, 'accuracy')}</td>
        <td>${fmtPct(m.macro_f1)} ${delta(scorer, 'macro_f1')}</td>
        <td>${fmtPct(m.macro_precision)} ${delta(scorer, 'macro_precision')}</td>
        <td>${fmtPct(m.macro_recall)} ${delta(scorer, 'macro_recall')}</td>
        <td>${fmtPct(m.macro_fpr, 2)} ${delta(scorer, 'macro_fpr', true)}</td>
        <td>${fmtPct(m.macro_fnr, 2)} ${delta(scorer, 'macro_fnr', true)}</td>
        <td>${m.n ?? '—'}${m.n_abstained ? ` <span class="hint">(+${m.n_abstained} abstain)</span>` : ''}</td>
      </tr>`;
    }).join('');
    host.innerHTML = `
      <table class="experiment-ledger-table experiment-judge-table">
        <thead><tr>
          <th>Judge</th><th>Accuracy</th><th>Macro F1</th>
          <th>Precision</th><th>Recall</th><th>FPR</th><th>FNR</th><th>n</th>
        </tr></thead>
        <tbody>${rows}</tbody>
      </table>
      <p class="legend-note">Δ chips = final vs k=0 on this run's fixed test partition; green is better
      (FPR/FNR improve downward). Recorded to the run's summary block at completion.</p>`;
  }

  // ---- gate ledger with expandable evidence ---------------------------------

  function thumbnailUrl(repoRelPath) {
    return `/api/thumbnail?path=${encodeURIComponent(repoRelPath)}`;
  }

  // image_id -> the last-rendered anchor record; the click handler below
  // hands it to the full-evidence drawer (justifications.js).
  const evidenceIndex = {};

  function renderAnchorCards(anchors) {
    if (!anchors || !anchors.length) {
      return '<p class="hint">No anchor records stored for this cycle.</p>';
    }
    const cards = anchors.map((a) => {
      evidenceIndex[String(a.image_id)] = a;
      const votes = (a.votes || []).map((v) => {
        const wrong = v.label !== a.sme_truth;
        return `<span class="experiment-vote ${wrong ? 'experiment-vote--wrong' : 'experiment-vote--right'}"
          title="${esc(v.model)} · confidence ${v.confidence ?? '—'}">${esc(String(v.model || '').split('/').pop())}: ${esc(v.label)}</span>`;
      }).join(' ');
      const img = a.repo_rel_path
        ? `<img src="${esc(thumbnailUrl(a.repo_rel_path))}" alt="${esc(a.image_id)}" loading="lazy" class="experiment-anchor-thumb" data-evidence-image="${esc(a.image_id)}" title="Click for the full LLM evidence — per-judge justification, sub-category, boundary, difficulty, confidence" />`
        : '';
      return `<figure class="experiment-anchor-card">
        ${img}
        <figcaption>
          <strong>truth: ${esc(a.sme_truth)}</strong>
          <span class="hint">${esc(a.image_id)} · ${esc(a.misalignment_type || '')}${a.severity ? ` · ${esc(a.severity)}` : ''}</span>
          <div class="experiment-anchor-votes">${votes}</div>
        </figcaption>
      </figure>`;
    }).join('');
    return `<div class="experiment-anchor-grid">${cards}</div>`;
  }

  document.addEventListener('click', (event) => {
    const el = event.target instanceof Element
      ? event.target.closest('[data-evidence-image]') : null;
    if (!el) return;
    const record = evidenceIndex[el.dataset.evidenceImage];
    if (record && typeof window.rushShowEvidence === 'function') {
      window.rushShowEvidence(record);
    }
  });

  function renderDiffBlocks(diffs) {
    if (!diffs || !diffs.length) return '<p class="hint">No diff available.</p>';
    return diffs.map((d) => {
      const lines = String(d.unified_diff || '').split('\n').map((line) => {
        let cls = '';
        if (line.startsWith('+') && !line.startsWith('+++')) cls = 'diff-add';
        else if (line.startsWith('-') && !line.startsWith('---')) cls = 'diff-del';
        else if (line.startsWith('@@')) cls = 'diff-hunk';
        return `<span class="${cls}">${esc(line)}</span>`;
      }).join('\n');
      return `<div class="experiment-diff-block">
        <div class="experiment-diff-head">${esc(d.change)} <code>${esc(d.path)}</code></div>
        <pre class="experiment-diff-pre">${lines}</pre>
      </div>`;
    }).join('');
  }

  async function loadMisalignmentRecords(runId) {
    if (!runId) return [];
    if (!state.anchorCache[runId]) {
      try {
        const response = await fetch(window.cacheBust
          ? window.cacheBust(`/data/runs/${encodeURIComponent(runId)}/scoring/misalignment.json`)
          : `/data/runs/${encodeURIComponent(runId)}/scoring/misalignment.json`);
        const payload = await response.json();
        state.anchorCache[runId] = payload?.records || [];
      } catch (err) {
        state.anchorCache[runId] = [];
      }
    }
    return state.anchorCache[runId];
  }

  async function fetchAnchors(cycle) {
    // New runs persist anchor blocks on the cycle; the train run's
    // misalignment artifact carries the FULL per-judge responses
    // (justification, l2_label, citations, quotes) — merge them in so the
    // evidence drawer has everything, falling back to the lean block votes.
    const runId = cycle.train_run_id;
    const records = await loadMisalignmentRecords(runId);
    const fullByImage = new Map(records.map((r) => [String(r.image_id), r]));
    const fromRecord = (r) => ({
      image_id: r.image_id,
      repo_rel_path: r.repo_rel_path,
      sme_truth: r.sme_truth,
      misalignment_type: r.misalignment_type,
      severity: r.severity,
      run_id: runId,
      votes: (r.votes || []).map((v) => ({
        ...v, model: v.labeler_id || v.model_id || v.model
      }))
    });
    if (cycle.anchors && cycle.anchors.length) {
      return cycle.anchors.map((a) => {
        const full = fullByImage.get(String(a.image_id));
        return full ? { ...a, ...fromRecord(full) } : a;
      });
    }
    const wanted = new Set(cycle.anchor_ids || []);
    return records
      .filter((r) => wanted.has(String(r.image_id)))
      .map(fromRecord);
  }

  async function fetchProposalDiffs(proposalId) {
    if (!proposalId) return null;
    if (!state.detailCache[proposalId]) {
      try {
        state.detailCache[proposalId] = await window.rushApiGetJson(
          `/api/policy/proposals/${encodeURIComponent(proposalId)}`
        );
      } catch (err) {
        state.detailCache[proposalId] = { diffs: [] };
      }
    }
    return state.detailCache[proposalId];
  }

  function renderLedger() {
    const host = $('#experimentLedger');
    const exp = state.current;
    const cycles = (exp?.cycles || []).filter((c) => c.k >= 1);
    if (!cycles.length) {
      host.innerHTML = exp?.status === 'running'
        ? '<p class="hint">Cycle 1 in progress…</p>'
        : '<p class="hint">No cycles yet.</p>';
      return;
    }
    const rows = cycles.map((c) => {
      const gate = c.gate || {};
      const delta = `${fmtPct(gate.value_before)} → ${fmtPct(gate.value_after)}`;
      const edits = (c.edit_summary || []).map((e) => `${esc(e.change)} ${esc(e.path)}`).join('<br>')
        || '<span class="hint">—</span>';
      const clipNote = c.edit_clipped
        ? ` <span class="experiment-chip experiment-chip--neutral" title="Drafter proposed ${c.n_changes_proposed}; clipped to ${c.n_changes_applied} for reviewability">clipped ${c.n_changes_proposed}→${c.n_changes_applied}</span>`
        : '';
      const rationale = gate.rationale
        ? `<div class="experiment-rationale" title="${esc(gate.decided_by)}">${esc(gate.rationale)}</div>`
        : (c.error ? `<div class="experiment-rationale">${esc(c.error)}</div>` : '');
      const kgLink = (c.status === 'accepted' && c.new_version)
        ? `<button type="button" class="experiment-kg-link" data-version="${esc(c.new_version)}">View ${esc(c.new_version)} in graph ↓</button>`
        : '';
      const review = c.review
        ? `<span class="experiment-chip ${c.review.verdict === 'correct' ? 'experiment-chip--accepted' : (c.review.verdict === 'incorrect' ? 'experiment-chip--failed' : 'experiment-chip--neutral')}" title="${esc(c.review.comment || '')} — ${esc(c.review.reviewer)}">SME: ${esc(c.review.verdict)}</span>`
        : (c.gate ? `
          <span class="experiment-review-buttons" data-k="${c.k}">
            <button type="button" data-verdict="correct" title="The gate decided correctly">✓</button>
            <button type="button" data-verdict="incorrect" title="The gate decided incorrectly">✗</button>
            <button type="button" data-verdict="unsure" title="Unsure">?</button>
          </span>` : '<span class="hint">—</span>');
      const expanded = state.expandedCycles.has(c.k);
      const detailButton = (c.proposal_id || (c.anchor_ids || []).length)
        ? `<button type="button" class="experiment-detail-toggle" data-k="${c.k}" aria-expanded="${expanded}">${expanded ? '▾ Hide' : '▸ Evidence'}</button>`
        : '';
      return `<tr>
        <td>k=${c.k} ${detailButton}</td>
        <td>${statusChip(c.status)}</td>
        <td title="misaligned in batch / anchors sampled">${c.n_misaligned ?? '—'} / ${(c.anchor_ids || []).length}</td>
        <td>${edits}${clipNote}</td>
        <td>${delta}${rationale}</td>
        <td title="train batch + candidate eval + optimizer (drafter) + gate agent for this cycle">${fmtUsd(c.cost_usd)}${c.drafter && typeof c.drafter.cost_usd === 'number' && c.drafter.cost_usd > 0
          ? `<div class="hint" title="The drafter call(s) that proposed this cycle's edit${c.drafter.cached_input_tokens ? ` — ${Number(c.drafter.cached_input_tokens).toLocaleString()} prompt tokens served from cache` : ''}">opt ${fmtUsd(c.drafter.cost_usd)}</div>`
          : ''}</td>
        <td>${kgLink}</td>
        <td>${review}</td>
      </tr>
      <tr class="experiment-detail-row" data-detail-k="${c.k}" ${expanded ? '' : 'hidden'}>
        <td colspan="8"><div class="experiment-detail-host" data-detail-host="${c.k}"></div></td>
      </tr>`;
    }).join('');
    host.innerHTML = `
      <table class="experiment-ledger-table">
        <thead><tr>
          <th>Cycle</th><th>Gate</th><th title="Misaligned in batch / anchors sampled (random misalignment anchors)">Misaligned / anchors</th>
          <th>Edit (≤${esc(exp.max_changes)} changes)</th>
          <th>Test system F1</th>
          <th title="What this cycle spent: train batch + candidate eval + optimizer (drafter) + gate agent">Cost (k)</th>
          <th>Policy</th>
          <th title="Was the gate's decision correct? Your verdicts are recorded as training data for the critic agent.">SME review</th>
        </tr></thead>
        <tbody>${rows}</tbody>
      </table>`;

    host.querySelectorAll('.experiment-detail-toggle').forEach((button) => {
      button.addEventListener('click', () => {
        const k = Number(button.dataset.k);
        const row = host.querySelector(`[data-detail-k="${k}"]`);
        const detailHost = host.querySelector(`[data-detail-host="${k}"]`);
        if (state.expandedCycles.has(k)) {
          state.expandedCycles.delete(k);
          row.hidden = true;
          button.textContent = '▸ Evidence';
          button.setAttribute('aria-expanded', 'false');
        } else {
          row.hidden = false;
          button.textContent = '▾ Hide';
          button.setAttribute('aria-expanded', 'true');
          toggleCycleDetailInline(k, detailHost);
        }
      });
    });
    // Re-render already-expanded rows (poll refresh keeps them open).
    state.expandedCycles.forEach((k) => {
      const detailHost = host.querySelector(`[data-detail-host="${k}"]`);
      if (detailHost) toggleCycleDetailInline(k, detailHost);
    });

    host.querySelectorAll('.experiment-kg-link').forEach((button) => {
      button.addEventListener('click', () => kgShowVersion(button.dataset.version, true));
    });
    host.querySelectorAll('.experiment-review-buttons button').forEach((button) => {
      button.addEventListener('click', async () => {
        const k = Number(button.parentElement.dataset.k);
        const verdict = button.dataset.verdict;
        let comment = '';
        if (verdict !== 'correct') {
          comment = window.prompt('Optional note for the record (why?)', '') || '';
        }
        try {
          await window.rushApiPostJson(
            `/api/experiments/${encodeURIComponent(state.current.experiment_id)}/review`,
            { k, verdict, reviewer: 'sme', comment }
          );
          await loadDetail(state.current.experiment_id);
        } catch (err) {
          $('#experimentStatusLine').textContent = `Review failed: ${err?.message || err}`;
        }
      });
    });
  }

  async function toggleCycleDetailInline(k, detailHost) {
    // Fill (or refresh) an expanded row's evidence without collapsing it.
    state.expandedCycles.add(k);
    const cycle = (state.current?.cycles || []).find((c) => c.k === k);
    if (!cycle || !detailHost) return;
    if (!detailHost.dataset.loaded) {
      detailHost.innerHTML = '<p class="hint">Loading evidence…</p>';
    }
    const [anchors, proposal] = await Promise.all([
      fetchAnchors(cycle),
      fetchProposalDiffs(cycle.proposal_id)
    ]);
    const gate = cycle.gate || {};
    const coverage = gate.comparison
      ? `<p class="hint">Gate compared ${gate.comparison.n_common} common test images
         (baseline decided ${gate.comparison.n_before}, candidate ${gate.comparison.n_after},
         partition ${gate.comparison.n_expected}).</p>`
      : '';
    const risks = (gate.risk_flags || []).length
      ? `<p class="experiment-risk-flags">Gate agent flags: ${gate.risk_flags.map(esc).join(' · ')}</p>`
      : '';
    detailHost.dataset.loaded = '1';
    detailHost.innerHTML = `
      <div class="experiment-cycle-detail">
        <div>
          <h4>Anchor misalignments <span class="hint">— the images that drove this edit</span></h4>
          ${renderAnchorCards(anchors)}
        </div>
        <div>
          <h4>Proposed edit <span class="hint">— ${esc(cycle.n_changes_applied ?? 0)} change(s)${cycle.edit_clipped ? `, clipped from ${esc(cycle.n_changes_proposed)}` : ''}</span></h4>
          ${renderDiffBlocks(proposal?.diffs)}
          ${coverage}${risks}
        </div>
      </div>`;
  }

  function renderHoldout() {
    const host = $('#experimentHoldout');
    const blocks = [];
    const readout = (block, title) => {
      if (!block || !block.start) return '';
      const start = block.start.metrics?.system || {};
      const final = block.final?.metrics?.system || {};
      return `
      <div class="experiment-holdout">
        <strong>${title} (${esc(block.n)} images):</strong>
        system macro-F1 ${fmtPct(start.macro_f1)} (${esc(block.start.version)})
        → ${fmtPct(final.macro_f1)} (${esc(block.final?.version)})
        · accuracy ${fmtPct(start.accuracy)} → ${fmtPct(final.accuracy)}
      </div>`;
    };
    blocks.push(readout(state.current?.holdout, 'Locked holdout — untouched by the loop'));
    blocks.push(readout(state.current?.benchmark,
      'Fixed validation benchmark — the same images every run (cross-run comparable)'));
    host.innerHTML = blocks.filter(Boolean).join('');
  }

  // ---- confusion grid: final policy on this run's test partition ------------

  async function fetchRunWebSummary(runId) {
    if (!runId) return null;
    if (runId in state.runSummaryCache) return state.runSummaryCache[runId];
    try {
      const url = `/data/runs/${encodeURIComponent(runId)}/web/summary.json`;
      const response = await fetch(window.cacheBust ? window.cacheBust(url) : url);
      if (!response.ok) return null; // transient/absent: retry next poll, never cache
      const payload = await response.json();
      state.runSummaryCache[runId] = payload; // cache successes only
      return payload;
    } catch (err) {
      return null;
    }
  }

  function isEnsembleLabeler(labeler) {
    if (typeof window.rushIsEnsembleRow === 'function' && window.rushIsEnsembleRow(labeler)) return true;
    return String(labeler?.labeler_id || '').toLowerCase() === 'majority_vote';
  }

  async function renderConfusion() {
    const host = $('#experimentConfusion');
    if (!host) return;
    // Token first: every invocation (including early-return clears)
    // invalidates in-flight renders so a slow fetch can't repaint the host.
    const token = ++state.confusionToken;
    const exp = state.current;
    const cycles = (exp?.cycles || []);
    if (!exp || !cycles.length) { host.innerHTML = ''; return; }
    // The final measured test eval: the last accepted candidate's run (its
    // metrics ARE the final policy's), else the k=0 baseline eval.
    const lastAccepted = [...cycles].reverse().find((c) => c.status === 'accepted' && c.candidate_run_id);
    const runId = lastAccepted?.candidate_run_id || cycles.find((c) => c.k === 0)?.test_run_id;
    if (!runId) { host.innerHTML = ''; return; }
    const summary = await fetchRunWebSummary(runId);
    if (token !== state.confusionToken) return; // superseded by a newer render
    const labelers = Array.isArray(summary?.labelers) ? summary.labelers : [];
    const withCM = labelers.filter((l) => l?.metrics?.confusion_matrix);
    const labeler = withCM.find(isEnsembleLabeler) || withCM[0];
    if (!labeler) {
      host.innerHTML = '<p class="hint">No multiclass confusion artifact for this run yet'
        + ' (the grid appears once the final test eval is scored).</p>';
      return;
    }
    const m = labeler.metrics;
    const cm = m.confusion_matrix;
    const perClass = m.per_class || {};
    const classes = Object.keys(cm).sort((a, b) => String(a).localeCompare(String(b), undefined, { numeric: true }));
    let maxOff = 0; let totalOff = 0; let total = 0;
    classes.forEach((t) => classes.forEach((p) => {
      const v = cm[t]?.[p] || 0;
      total += v;
      if (t !== p) { maxOff = Math.max(maxOff, v); totalOff += v; }
    }));
    const cellHtml = (t, p) => {
      const v = cm[t]?.[p] || 0;
      let cls = 'cm-cell';
      let style = '';
      if (v > 0 && t === p) cls += ' cm-correct';
      else if (v > 0) {
        cls += ' cm-confusion';
        const intensity = maxOff ? (0.18 + 0.62 * (v / maxOff)) : 0.4;
        style = `background:rgba(255,111,145,${intensity.toFixed(2)});`;
      }
      return `<td class="${cls}" style="${style}" title="${esc(`truth ${t} → predicted ${p}: ${v}`)}">${v || ''}</td>`;
    };
    const perClassCell = (c, field, digits = 2) => {
      const v = perClass[c]?.[field];
      return `<td class="cm-f1" title="${esc(`class ${c} ${field}`)}">${(v === null || v === undefined) ? '—' : Number(v).toFixed(digits)}</td>`;
    };
    const headCols = classes.map((c) => `<th class="cm-head" scope="col">${esc(c)}</th>`).join('');
    const rows = classes.map((t) => `
      <tr>
        <th class="cm-head cm-row-head" scope="row">${esc(t)}</th>
        ${classes.map((p) => cellHtml(t, p)).join('')}
        ${perClassCell(t, 'f1')}${perClassCell(t, 'recall')}${perClassCell(t, 'fpr', 3)}
      </tr>`).join('');
    const overall = [
      ['accuracy', m.accuracy], ['macro F1', m.macro_f1],
      ['precision', m.macro_precision], ['recall', m.macro_recall],
      ['FPR', m.macro_fpr], ['FNR', m.macro_fnr]
    ].map(([k, v]) => `${k} <strong>${fmtPct(v, k === 'FPR' || k === 'FNR' ? 2 : 1)}</strong>`).join(' · ');
    host.innerHTML = `
      <div class="cm-header">
        <h3>${esc(isEnsembleLabeler(labeler) ? 'System (majority vote)' : labeler.labeler_id)} under ${esc(exp.current_version)} — ${esc(String(totalOff))}/${esc(String(total))} confusions</h3>
        <p class="cm-sub">${overall} · from ${lastAccepted ? `the accepted k=${lastAccepted.k} candidate eval` : 'the k=0 baseline eval'}
        <code>${esc(runId)}</code>. Rows = SME truth, columns = predicted.</p>
      </div>
      <div class="cm-scroll">
        <table class="cm-table">
          <thead><tr><th class="cm-corner"><span>truth ＼ pred</span></th>${headCols}<th class="cm-head cm-f1-head">F1</th><th class="cm-head cm-f1-head">Recall</th><th class="cm-head cm-f1-head">FPR</th></tr></thead>
          <tbody>${rows}</tbody>
        </table>
      </div>`;
  }

  // ---- node changes vs k=0: per-node diff + image evidence ------------------

  function lineDiff(beforeText, afterText) {
    // LCS line diff — policy nodes are ~60 lines, O(n·m) is nothing.
    const a = String(beforeText ?? '').split('\n');
    const b = String(afterText ?? '').split('\n');
    const n = a.length; const m = b.length;
    const dp = Array.from({ length: n + 1 }, () => new Array(m + 1).fill(0));
    for (let i = n - 1; i >= 0; i -= 1) {
      for (let j = m - 1; j >= 0; j -= 1) {
        dp[i][j] = a[i] === b[j] ? dp[i + 1][j + 1] + 1 : Math.max(dp[i + 1][j], dp[i][j + 1]);
      }
    }
    const ops = [];
    let i = 0; let j = 0;
    while (i < n && j < m) {
      if (a[i] === b[j]) { ops.push([' ', a[i]]); i += 1; j += 1; }
      else if (dp[i + 1][j] >= dp[i][j + 1]) { ops.push(['-', a[i]]); i += 1; }
      else { ops.push(['+', b[j]]); j += 1; }
    }
    while (i < n) { ops.push(['-', a[i]]); i += 1; }
    while (j < m) { ops.push(['+', b[j]]); j += 1; }
    return ops;
  }

  function renderCompactDiff(ops, context = 2) {
    const keep = new Array(ops.length).fill(false);
    ops.forEach((op, idx) => {
      if (op[0] === ' ') return;
      for (let d = -context; d <= context; d += 1) {
        const t = idx + d;
        if (t >= 0 && t < ops.length) keep[t] = true;
      }
    });
    const out = [];
    let skipping = false;
    ops.forEach((op, idx) => {
      if (!keep[idx]) {
        if (!skipping) { out.push(['~', '⋯ unchanged ⋯']); skipping = true; }
        return;
      }
      skipping = false;
      out.push(op);
    });
    return out.map(([tag, line]) => {
      const cls = tag === '+' ? 'diff-add' : (tag === '-' ? 'diff-del' : (tag === '~' ? 'diff-hunk' : ''));
      const prefix = tag === '~' ? '' : `${tag === ' ' ? ' ' : tag} `;
      return `<span class="${cls}">${esc(prefix + line)}</span>`;
    }).join('\n');
  }

  async function fetchVersionBundle(version) {
    const area = activeArea();
    const key = `${area}/${version}`;
    if (state.bundleCache[key]) return state.bundleCache[key];
    // Cache SUCCESSES only: a transient failure must not pin an empty/bogus
    // bundle (and thus a wrong diff) for the rest of the session.
    try {
      const graph = await window.rushApiGetJson(
        `/api/policy/graph?area=${encodeURIComponent(area)}&version=${encodeURIComponent(version)}`
      );
      const ids = (graph?.nodes || []).map((node) => node.id).filter(Boolean);
      const texts = {};
      let failed = false;
      await Promise.all(ids.map(async (id) => {
        try {
          const url = `/policy-graph/${encodeURIComponent(area)}/${encodeURIComponent(version)}/${encodeURIComponent(id)}.md`;
          const response = await fetch(window.cacheBust ? window.cacheBust(url) : url);
          if (response.ok) texts[id] = await response.text();
          else if (response.status === 404) texts[id] = null; // genuinely absent
          else failed = true;
        } catch (err) {
          failed = true;
        }
      }));
      const bundle = { ids, texts };
      if (!failed) state.bundleCache[key] = bundle;
      return bundle;
    } catch (err) {
      return { ids: [], texts: {} };
    }
  }

  function anchorStatsByTruth(cycles) {
    // Evidence pressure per class node across this run's anchors: how many
    // anchors carried each SME truth, and what it was most misread as.
    const stats = {};
    cycles.forEach((c) => (c.anchors || []).forEach((anchor) => {
      const truth = String(anchor.sme_truth ?? '');
      if (!truth) return;
      const entry = stats[truth] = stats[truth] || { count: 0, confusion: {} };
      entry.count += 1;
      (anchor.votes || []).forEach((vote) => {
        const label = String(vote.label ?? '');
        if (label && label !== truth && label !== 'abstain') {
          entry.confusion[label] = (entry.confusion[label] || 0) + 1;
        }
      });
    }));
    return stats;
  }

  function topConfusion(confusion) {
    const entries = Object.entries(confusion || {}).sort((a, b) => b[1] - a[1]);
    return entries.length ? entries[0] : null;
  }

  async function renderPolicyChanges() {
    const host = $('#experimentPolicyChanges');
    if (!host) return;
    // Token first: even early-return clears must invalidate an in-flight
    // render, or a slow diff for the previous run repaints the cleared host.
    const token = ++state.policyChangesToken;
    const exp = state.current;
    const cycles = (exp?.cycles || []).filter((c) => typeof c.k === 'number');
    if (!exp || !cycles.length) { host.innerHTML = ''; return; }
    const shownVersion = versionInForceAfter(cycles, state.kgCycleK ?? cycles[cycles.length - 1].k);
    const baseVersion = exp.base_version || 'v0.1';

    const stats = anchorStatsByTruth(cycles);
    const statChips = Object.entries(stats)
      .sort((a, b) => b[1].count - a[1].count)
      .map(([truth, s]) => {
        const top = topConfusion(s.confusion);
        return `<span class="experiment-node-stat" title="anchors with SME truth ${esc(truth)} across this run's cycles">
          truth <strong>${esc(truth)}</strong> · ${s.count} anchor${s.count === 1 ? '' : 's'}${top ? ` · misread as <strong>${esc(top[0])}</strong> ×${top[1]}` : ''}</span>`;
      }).join('');
    const statsBlock = statChips
      ? `<div class="experiment-node-stats" aria-label="Anchor evidence by class">${statChips}</div>`
      : '';

    if (shownVersion === baseVersion) {
      host.innerHTML = `${statsBlock}<p class="hint">Showing ${esc(baseVersion)} (k=0 baseline) — no accepted changes yet.
        Step the cycle chips above to an accepted version to see per-node diffs.</p>`;
      return;
    }
    host.innerHTML = `${statsBlock}<p class="hint">Computing node diffs ${esc(baseVersion)} → ${esc(shownVersion)}…</p>`;
    const [baseBundle, currentBundle] = await Promise.all([
      fetchVersionBundle(baseVersion), fetchVersionBundle(shownVersion)
    ]);
    if (token !== state.policyChangesToken) return; // superseded by a newer render

    const allIds = Array.from(new Set([...(baseBundle.ids || []), ...(currentBundle.ids || [])])).sort();
    const versionK = new Map();
    cycles.forEach((c) => { if (c.status === 'accepted' && c.new_version) versionK.set(c.new_version, c.k); });
    const shownK = versionK.get(shownVersion) ?? Infinity;

    const cards = [];
    allIds.forEach((id) => {
      const before = baseBundle.texts?.[id];
      const after = currentBundle.texts?.[id];
      if (before === after) return;
      const change = before === null || before === undefined
        ? 'added' : (after === null || after === undefined ? 'removed' : 'modified');
      // Which cycles touched this node file, up to the shown version.
      const touched = cycles.filter((c) => c.k >= 1 && c.k <= shownK
        && (c.edit_summary || []).some((e) => e.path === `${id}.md`));
      const acceptedTouches = touched.filter((c) => c.status === 'accepted');
      const cycleChips = touched.map((c) => {
        const accepted = c.status === 'accepted';
        return `<span class="experiment-chip ${accepted ? 'experiment-chip--accepted' : 'experiment-chip--neutral'}"
          title="${esc(accepted ? `accepted → ${c.new_version}` : `${c.status}`)}">k=${c.k}${accepted ? ` → ${esc(c.new_version)}` : ` · ${esc(c.status)}`}</span>`;
      }).join(' ');
      const evidence = acceptedTouches.flatMap((c) => c.anchors || []).slice(0, 6);
      const diffHtml = change === 'modified'
        ? renderCompactDiff(lineDiff(before, after))
        : renderCompactDiff(lineDiff(change === 'added' ? '' : before, change === 'added' ? after : ''));
      cards.push(`
        <div class="experiment-node-card">
          <div class="experiment-node-card-head">
            <code>${esc(id)}</code>
            <span class="experiment-chip ${change === 'added' ? 'experiment-chip--accepted' : 'experiment-chip--neutral'}">${esc(change)}</span>
            <span class="experiment-node-cycle-chips">${cycleChips || '<span class="hint">changed outside this run</span>'}</span>
          </div>
          <div class="experiment-node-card-body">
            <div class="experiment-diff-block">
              <div class="experiment-diff-head">${esc(baseVersion)} → ${esc(shownVersion)} <code>${esc(id)}.md</code></div>
              <pre class="experiment-diff-pre">${diffHtml}</pre>
            </div>
            <div>
              <h4>Image evidence <span class="hint">— anchors behind the accepting cycle(s)</span></h4>
              ${renderAnchorCards(evidence)}
            </div>
          </div>
        </div>`);
    });
    host.innerHTML = statsBlock + (cards.length
      ? cards.join('')
      : `<p class="hint">${esc(baseVersion)} → ${esc(shownVersion)}: no node-level differences found.</p>`);
  }

  // ---- init -----------------------------------------------------------------

  // Demo-aware form defaults, sized from the area's REAL manifest via
  // /api/area-stats. The static HTML defaults fit MNIST (dev_golden ~2k);
  // the GenAI pool is far smaller (40 on portable clones, 100+ where the
  // source tree exists), so Test/Train scale to the pool, and the benchmark
  // readout enables only when a fixed validation split actually exists.
  // Every other knob is identical across demos by design.
  async function applyDemoFormDefaults() {
    const genai = activeDemoId() === 'genai';
    const testN = $('#experimentTestN');
    const batchN = $('#experimentBatchN');
    const benchmark = $('#experimentValidationFinal');
    // Static fallback (portable GenAI pool = 40) so the form is sane even if
    // the stats endpoint is unreachable.
    if (genai) {
      if (testN) testN.value = '20';
      if (batchN) batchN.value = '10';
      if (benchmark) { benchmark.checked = false; benchmark.disabled = true; }
      const note = $('#experimentBenchmarkNote');
      if (note) note.textContent = 'needs a validation split — mint one with sample_genai_gold_sets.py --n-validation on the data host';
    }
    let stats = null;
    try {
      stats = await window.rushApiGetJson(`/api/area-stats?demo=${encodeURIComponent(activeDemoId())}`);
    } catch (err) { /* fall back to the static defaults above */ }
    if (!stats || activeDemoId() !== (genai ? 'genai' : 'mnist')) return;
    const pool = Number(stats?.splits?.dev_golden) || 0;
    if (genai && pool > 0) {
      // Half the pool tests, the rest is the train pool; clamp to the
      // MNIST-sized defaults so a big future pool converges to parity.
      if (testN) testN.value = String(Math.max(10, Math.min(100, Math.floor(pool / 2))));
      if (batchN) batchN.value = String(Math.max(2, Math.min(20, Math.floor(pool / 4))));
    }
    const hasValidation = (Number(stats?.splits?.validation) || 0) > 0;
    if (benchmark) {
      benchmark.disabled = !hasValidation;
      benchmark.checked = hasValidation;
      // A silently disabled checkbox reads as broken — say WHY inline.
      const note = $('#experimentBenchmarkNote');
      if (note) {
        note.textContent = hasValidation
          ? ''
          : 'needs a benchmark split — mint one in the Splits row above';
      }
      const label = benchmark.closest('label');
      if (label && !hasValidation) {
        label.title = `The ${stats?.area || 'active'} manifest has no fixed validation split yet — mint one in the Splits row to enable the cross-run benchmark readout.`;
      }
    }
    renderSplitsRow(stats, genai);
  }

  // ---- GenAI split minting (the seed IS the cross-machine contract) ----------

  function renderSplitsRow(stats, genai) {
    const row = $('#genaiSplitsRow');
    if (!row) return;
    row.hidden = !genai;
    if (!genai || !stats) return;
    const s = stats.splits || {};
    const statusEl = $('#genaiSplitsStatus');
    const usingPortable = String(stats.manifest || '').endsWith('portable.jsonl');
    if (statusEl && !state.mintPollTimer) {
      statusEl.textContent = usingPortable
        ? `portable fixture in use (dev ${s.dev_golden ?? 0} · holdout ${s.holdout ?? 0}) — mint to switch to the full source tree`
        : `current: dev ${s.dev_golden ?? 0} · holdout ${s.holdout ?? 0} · bench ${s.validation ?? 0}${stats.sampling_seed != null ? ` · seed ${stats.sampling_seed}` : ''}`;
    }
    const seedInput = $('#genaiSplitSeed');
    if (seedInput && stats.sampling_seed != null && !state.splitSeedTouched) {
      seedInput.value = String(stats.sampling_seed);
    }
    if (stats.mint_running) beginMintPoll();
  }

  function beginMintPoll() {
    if (state.mintPollTimer) return;
    const statusEl = $('#genaiSplitsStatus');
    const button = $('#genaiSplitsMint');
    if (button) button.disabled = true;
    let ticks = 0;
    state.mintPollTimer = window.setInterval(async () => {
      ticks += 1;
      if (statusEl) statusEl.textContent = `minting splits — hashing the source tree… (${ticks * 4}s)`;
      let stats = null;
      try {
        stats = await window.rushApiGetJson(`/api/area-stats?demo=${encodeURIComponent(activeDemoId())}`);
      } catch (err) { return; }
      if (!stats.mint_running || ticks > 150) {
        window.clearInterval(state.mintPollTimer);
        state.mintPollTimer = null;
        if (button) button.disabled = false;
        if (statusEl) statusEl.textContent = 'splits minted ✓ — form defaults refreshed';
        // Re-size T/N, re-enable the benchmark checkbox, refresh the row text.
        applyDemoFormDefaults();
      }
    }, 4000);
  }

  async function mintSplits() {
    const statusEl = $('#genaiSplitsStatus');
    const payload = {
      seed: Number($('#genaiSplitSeed')?.value || 20260510),
      n_dev: Number($('#genaiSplitDev')?.value || 2000),
      n_holdout: Number($('#genaiSplitHoldout')?.value || 1000),
      n_validation: Number($('#genaiSplitValidation')?.value ?? 200),
    };
    const warning = 'Re-mint the GenAI split manifests?\n\n'
      + `seed ${payload.seed} · dev ${payload.n_dev} · holdout ${payload.n_holdout} · benchmark ${payload.n_validation}\n\n`
      + 'Changing the seed or sizes re-deals which images belong to which split: '
      + 'prior runs’ benchmark numbers stop being same-images-comparable with new runs, '
      + 'and other machines must mint with the SAME seed + sizes to stay aligned.';
    if (!window.confirm(warning)) return;
    try {
      await window.rushApiPostJson('/api/genai/splits/mint', payload);
      if (statusEl) statusEl.textContent = 'minting splits — hashing the source tree…';
      beginMintPoll();
    } catch (err) {
      if (statusEl) statusEl.textContent = `Mint failed: ${err?.message || err}`;
    }
  }

  function init() {
    if (!$('#experiment')) return;
    applyDemoFormDefaults();
    const metricSelect = $('#experimentMetric');
    metricSelect.innerHTML = METRICS.map(([key, label]) => `<option value="${key}">${esc(label)}</option>`).join('');
    metricSelect.addEventListener('change', renderChart);
    $('#experimentXAxis')?.addEventListener('change', renderChart);
    $('#experimentShowTrain')?.addEventListener('change', renderChart);
    $('#experimentSelect').addEventListener('change', (event) => {
      state.expandedCycles.clear();
      state.kgManual = false;
      state.kgCycleK = null;
      state.followedVersion = null;
      state.renderSigs = {};
      state.jobToken = null;  // the live card re-discovers the job for the newly selected run
      loadDetail(event.target.value);
    });
    $('#experimentRefresh').addEventListener('click', () => {
      state.renderSigs = {};
      loadList();
    });
    $('#experimentStart').addEventListener('click', startExperiment);
    $('#genaiSplitsMint')?.addEventListener('click', mintSplits);
    // A user-edited seed must survive the row's poll refreshes.
    $('#genaiSplitSeed')?.addEventListener('input', () => { state.splitSeedTouched = true; });
    // Delegated: the live card rebuilds every poll, the listener must not.
    $('#experimentLiveCard')?.addEventListener('click', (event) => {
      if (event.target?.closest?.('[data-cancel-run]')) cancelCurrentRun();
    });
    document.addEventListener('change', (event) => {
      if (event.target?.classList?.contains('model-select-input')) renderPanelSummary();
    });
    window.addEventListener('rush-api-catalog', () => { renderPanelSummary(); loadList(); });
    renderPanelSummary();
    loadList();
  }

  initViewSwitcher();
  if (typeof window.rushApiOnReady === 'function') {
    window.rushApiOnReady(() => init());
  } else {
    document.addEventListener('DOMContentLoaded', init);
  }
})();
