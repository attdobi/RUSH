const $ = selector => document.querySelector(selector);
const esc = value => String(value ?? '').replace(/[&<>'"]/g, char => ({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}[char]));
const attr = esc;
const isNumber = value => typeof value === 'number' && Number.isFinite(value);

const MANIFESTS = {
  dev: '../data/images/genai-classification/manifests/dev_golden_labels.csv',
  holdout: '../data/images/genai-classification/manifests/holdout_labels.csv',
  summary: '../data/images/genai-classification/manifests/sampling_summary.json'
};

const policyNodes = [
  { id: 'GA.visual_artifacts.anatomy.hands', label: 'Hands / anatomy', type: 'positive evidence', summary: 'Impossible hands, fingers, limbs, teeth, or repeated body details.' },
  { id: 'GA.visual_artifacts.text_symbols', label: 'Text + symbols', type: 'positive evidence', summary: 'Garbled text, pseudo-logos, malformed UI, or impossible integrated typography.' },
  { id: 'GA.surface_texture.plastic_skin', label: 'Synthetic texture', type: 'positive evidence', summary: 'Waxy skin, over-smoothed surfaces, diffusion texture repetition, or pore absence.' },
  { id: 'GA.scene_geometry.inconsistent_perspective', label: 'Scene geometry', type: 'positive evidence', summary: 'Impossible reflections, shadows, object intersections, or inconsistent perspective.' },
  { id: 'GA.boundary.photo_editing', label: 'Edited real photo', type: 'hard negative', summary: 'Filters, retouching, compression, and conventional edits are not GenAI by themselves.' },
  { id: 'GA.boundary.cgi_game_render', label: 'CGI / game render', type: 'hard negative', summary: 'Stylized renders are not GenAI unless generative provenance is established.' },
  { id: 'GA.boundary.low_quality_uncertain', label: 'Low-quality uncertain', type: 'abstain / SME review', summary: 'Blurred, cropped, or ambiguous evidence should route to SME review.' }
];

const demoState = {
  result: null,
  source: 'loading',
  overrides: {}
};

function parseCsv(text) {
  const rows = [];
  let row = [];
  let cell = '';
  let quoted = false;
  for (let i = 0; i < text.length; i += 1) {
    const char = text[i];
    const next = text[i + 1];
    if (quoted) {
      if (char === '"' && next === '"') { cell += '"'; i += 1; }
      else if (char === '"') quoted = false;
      else cell += char;
    } else if (char === '"') quoted = true;
    else if (char === ',') { row.push(cell); cell = ''; }
    else if (char === '\n') { row.push(cell); rows.push(row); row = []; cell = ''; }
    else if (char !== '\r') cell += char;
  }
  if (cell || row.length) { row.push(cell); rows.push(row); }
  const [header, ...body] = rows.filter(r => r.some(v => v !== ''));
  if (!header) return [];
  return body.map(values => Object.fromEntries(header.map((key, index) => [key, values[index] ?? ''])));
}

async function fetchText(path) {
  const response = await fetch(path, { cache: 'no-store' });
  if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
  return response.text();
}

async function loadLocalManifests() {
  const [devText, holdoutText, summaryText] = await Promise.all([
    fetchText(MANIFESTS.dev),
    fetchText(MANIFESTS.holdout),
    fetchText(MANIFESTS.summary)
  ]);
  return {
    devGolden: parseCsv(devText).map(row => normalizeManifestRow(row, 'dev_golden')),
    holdout: parseCsv(holdoutText).map(row => normalizeManifestRow(row, 'holdout')),
    manifestSummary: JSON.parse(summaryText)
  };
}

function normalizeManifestRow(row, split) {
  const repoPath = row.repo_rel_path || row.synthetic_repo_rel_path || '';
  return {
    ...row,
    sample_id: row.sample_id,
    split,
    label_int: Number.parseInt(row.label_int, 10),
    seed: Number.parseInt(row.seed, 10) || 20260510,
    sha256: row.sha256 || '',
    repo_rel_path: repoPath,
    synthetic_repo_rel_path: repoPath,
    original_filename: row.original_filename || repoPath.split('/').pop(),
    truth_tier: row.truth_tier || 'gold_candidate',
    policy_use: row.policy_use || (split === 'holdout' ? 'locked_holdout_decision_quality' : 'develop_policy'),
    llm_status: 'pending_bulk_llm_labeling',
    human_override_label: null,
    human_override_note: '',
    human_override_applied: false
  };
}

function countBy(records, keyFn) {
  return records.reduce((acc, row) => {
    const key = keyFn(row) || 'unknown';
    acc[key] = (acc[key] || 0) + 1;
    return acc;
  }, {});
}

function take(records, n) {
  return records.slice(0, Math.max(0, n));
}

function readSamplerOptions() {
  return {
    seed: Number.parseInt($('#samplerSeed')?.value, 10) || 20260510,
    mode: $('#samplerMode')?.value || 'cold_start',
    nDev: Number.parseInt($('#samplerDevN')?.value, 10) || 100,
    nHoldout: Number.parseInt($('#samplerHoldoutN')?.value, 10) || 100
  };
}

function buildSummary(devGolden, holdout, options, source, manifestSummary = null) {
  const combined = [...devGolden, ...holdout];
  return {
    seed: options.seed,
    mode: options.mode,
    source,
    manifestSeed: manifestSummary?.seed,
    samplingVersion: manifestSummary?.sampling_version,
    n_dev_golden: devGolden.length,
    n_holdout: holdout.length,
    total: combined.length,
    byClass: countBy(combined, row => row.label),
    byDataset: countBy(combined, row => row.dataset),
    bySplit: { dev_golden: devGolden.length, holdout: holdout.length }
  };
}

async function runRealOrSyntheticSampler() {
  const options = readSamplerOptions();
  try {
    const local = await loadLocalManifests();
    const manifestSeed = Number.parseInt(local.manifestSummary?.seed, 10);
    if (options.mode !== 'cold_start') throw new Error('Local manifests are cold-start only; using synthetic fallback for warm-start preview.');
    if (manifestSeed && options.seed !== manifestSeed) throw new Error('Requested seed differs from local manifest seed; using synthetic fallback so sampling changes visibly.');
    const devGolden = take(local.devGolden, options.nDev);
    const holdout = take(local.holdout, options.nHoldout);
    return {
      devGolden,
      holdout,
      combined: [...devGolden, ...holdout],
      summary: buildSummary(devGolden, holdout, options, 'local manifests + real image files', local.manifestSummary),
      leakageChecks: { ok: true, devHoldoutDisjoint: true, note: `Local manifests were generated by ${local.manifestSummary?.sampling_version || 'scripts/sample_genai_gold_sets.py'} with path/hash leakage checks.` }
    };
  } catch (error) {
    if (!window.RushGenaiSampler?.runDemoReset) throw error;
    const fallback = window.RushGenaiSampler.runDemoReset(options);
    return {
      ...fallback,
      summary: {
        ...fallback.summary,
        source: 'browser synthetic fallback',
        total: fallback.combined.length,
        byClass: countBy(fallback.combined, row => row.label),
        byDataset: countBy(fallback.combined, row => row.dataset),
        bySplit: { dev_golden: fallback.devGolden.length, holdout: fallback.holdout.length }
      }
    };
  }
}

function labelBadge(row) {
  return row.label === 'ai_generated' ? 'ai-generated' : 'not-ai';
}

function overrideFor(sampleId) {
  return demoState.overrides[sampleId] || { label: 'none', note: '' };
}

function imgSrc(row) {
  const path = row.repo_rel_path || row.synthetic_repo_rel_path;
  return path ? `../${path}` : '';
}

function safeImageFallback(label = 'image unavailable', detail = 'local path missing') {
  const fallback = document.createElement('div');
  fallback.className = 'thumb-fallback';
  const strong = document.createElement('strong');
  strong.textContent = label;
  const span = document.createElement('span');
  span.textContent = detail;
  fallback.append(strong, span);
  return fallback;
}

function renderThumb(row) {
  const src = imgSrc(row);
  const fallback = `<div class="thumb-fallback"><strong>${esc(row.dataset)}</strong><span>${esc(row.label)}</span></div>`;
  if (demoState.source.startsWith('local') && src) {
    return `<img src="${attr(src)}" alt="${attr(row.sample_id)} ${attr(row.label)}" loading="lazy" onerror="this.replaceWith(safeImageFallback())" />`;
  }
  return fallback;
}

function renderSampleCard(row, compact = false) {
  const override = overrideFor(row.sample_id);
  const locked = row.split === 'holdout';
  const policyCue = row.label === 'ai_generated' ? 'positive evidence search' : 'boundary / hard-negative check';
  return `<article class="sample-card ${locked ? 'locked' : ''} ${compact ? 'compact' : ''}" data-sample-id="${attr(row.sample_id)}">
    <div class="sample-image">${renderThumb(row)}</div>
    <div class="sample-meta">
      <div class="sample-badges">
        <span class="badge ${locked ? 'holdout' : 'dev'}">${locked ? 'locked holdout' : 'dev golden'}</span>
        <span class="badge ${labelBadge(row)}">${esc(row.label)}</span>
      </div>
      <h3>${esc(row.sample_id)}</h3>
      <p>${esc(row.dataset)} · ${esc(row.original_filename || 'demo image')}</p>
      <p><strong>Directory label:</strong> ${esc(row.source_label_dir || 'synthetic')} → ${esc(row.label)}<br><strong>Policy cue:</strong> ${esc(policyCue)}<br><strong>LLM status:</strong> pending bulk labeling</p>
      ${compact ? '' : `<div class="override-controls">
        <label>SME/human override
          <select class="override-label" data-sample-id="${attr(row.sample_id)}">
            ${['none','ai_generated','not_ai_generated','needs_review'].map(value => `<option value="${value}" ${override.label === value ? 'selected' : ''}>${esc(value)}</option>`).join('')}
          </select>
        </label>
        <label>Review note
          <input class="override-note" data-sample-id="${attr(row.sample_id)}" type="text" value="${attr(override.note)}" placeholder="policy cue, uncertainty, or correction" />
        </label>
      </div>`}
    </div>
  </article>`;
}

function renderStats() {
  const summary = demoState.result?.summary;
  if (!summary) return;
  const classText = Object.entries(summary.byClass || {}).map(([k, v]) => `${k}: ${v}`).join(' · ');
  const sourceText = Object.entries(summary.byDataset || {}).map(([k, v]) => `${k}: ${v}`).join(' · ');
  $('#demoStats').innerHTML = [
    ['Sampled records', summary.total ?? (summary.n_dev_golden + summary.n_holdout), 'default N = 100 per split'],
    ['Truth tier', 'gold candidates', 'SME quality assumed for demo'],
    ['Source', summary.source || demoState.source, summary.samplingVersion ? `manifest ${summary.samplingVersion}` : 'real manifests if available'],
    ['Leakage check', demoState.result.leakageChecks?.ok ? 'pass' : 'review', 'dev/holdout path + hash separation']
  ].map(([label, value, note]) => `<article class="stat-card"><span>${esc(label)}</span><strong>${esc(value)}</strong><p>${esc(note)}</p></article>`).join('') +
  `<div class="wide-note"><strong>Class balance:</strong> ${esc(classText || 'n/a')}<br><strong>Dataset balance:</strong> ${esc(sourceText || 'n/a')}</div>`;
}

function renderGallery() {
  const result = demoState.result;
  if (!result) return;
  const visible = [...result.devGolden.slice(0, 12), ...result.holdout.slice(0, 12)];
  $('#sampleGallery').innerHTML = `
    <div class="gallery-head">
      <div><h3>Visible sample gallery</h3><p>Showing ${visible.length} records from the sampled set. These same records drive the policy loop and decision-quality preview below.</p></div>
      <span class="quiet-pill">${esc(result.summary.mode || 'cold_start')}</span>
    </div>
    <div class="sample-grid">${visible.map(row => renderSampleCard(row)).join('')}</div>`;
}

function renderPolicy() {
  $('#policyNodeList').innerHTML = policyNodes.map(node => `<article class="node-card ${node.type.includes('negative') ? 'negative' : ''}"><span>${esc(node.type)}</span><h3>${esc(node.label)}</h3><p>${esc(node.summary)}</p><code>${esc(node.id)}</code></article>`).join('');
  const summary = demoState.result?.summary;
  const total = summary?.total || 0;
  $('#policyLoop').innerHTML = [
    ['Sample', `${total} candidates sampled`, 'Balanced GenAI/not-GenAI examples create a cold-start policy surface.'],
    ['Annotate', 'SME override controls ready', 'Human corrections are captured before LLM labels exist.'],
    ['Label', 'LLM labeling comes next', 'Models will cite policy nodes and return structured confidence/justification.'],
    ['Patch', 'Policy diffs from clusters', 'Repeated disagreements become SME-reviewable graph changes.']
  ].map(([k, v, d]) => `<div class="timeline-row"><span>${esc(k)}</span><strong>${esc(v)}</strong><p>${esc(d)}</p></div>`).join('');
}

function overrideCounts() {
  const values = Object.values(demoState.overrides).filter(item => item.label !== 'none' || item.note.trim());
  return { total: values.length, labels: values.filter(item => item.label !== 'none').length };
}

function renderQuality() {
  const result = demoState.result;
  if (!result) return;
  const overrides = overrideCounts();
  const ai = result.combined.filter(row => row.label === 'ai_generated').length;
  const notAi = result.combined.length - ai;
  $('#qualityCards').innerHTML = [
    ['SME truth baseline', 'assumed ready', 'Demo labels are treated as SME-quality candidates until actual review tiers are wired.'],
    ['Model outputs', 'not started', 'Next phase runs multiple models with API keys and structured policy-node evidence.'],
    ['Review pressure', `${overrides.total} notes`, `${overrides.labels} explicit SME label overrides captured in this browser session.`],
    ['Class mix', `${ai} / ${notAi}`, 'AI-generated vs not-AI candidates in the active sample.']
  ].map(([label, value, note]) => `<article class="quality-card"><span>${esc(label)}</span><strong>${esc(value)}</strong><p>${esc(note)}</p></article>`).join('');

  const queue = [
    { title: 'Consensus failure audit', body: 'Even 3/3 LLM agreement gets sampled for SME audit to catch correlated failure.', rows: result.devGolden.slice(0, 2) },
    { title: 'Boundary hard negatives', body: 'Real edits, CGI/game renders, and low-quality uncertain cases are routed separately from positives.', rows: result.combined.filter(r => r.label === 'not_ai_generated').slice(0, 2) },
    { title: 'Policy gap finder', body: 'Repeated LLM/SME disagreement clusters become candidate Markdown graph diffs.', rows: result.combined.slice(2, 4) }
  ];
  $('#misalignmentQueue').innerHTML = queue.map(item => `<article class="queue-card"><h3>${esc(item.title)}</h3><p>${esc(item.body)}</p>${item.rows.map(row => renderSampleCard(row, true)).join('')}</article>`).join('');
}

function renderAll() {
  renderStats();
  renderGallery();
  renderPolicy();
  renderQuality();
}

function setSamplerLoading(isLoading) {
  const loading = $('#samplerLoading');
  const runButton = $('#runSampler');
  if (loading) loading.hidden = !isLoading;
  if (runButton) {
    runButton.disabled = isLoading;
    runButton.setAttribute('aria-busy', String(isLoading));
  }
}

async function runSamplerDemo() {
  const status = $('#samplerStatus');
  setSamplerLoading(true);
  status.textContent = 'Loading sampled images and labels…';
  status.classList.remove('error');
  demoState.overrides = {};
  try {
    demoState.result = await runRealOrSyntheticSampler();
    demoState.source = demoState.result.summary.source || 'browser synthetic fallback';
    status.textContent = `Sample ready: ${demoState.result.summary.total} records from ${demoState.source}. No LLM calls made.`;
    renderAll();
  } catch (error) {
    console.error('Sampler failed:', error);
    status.classList.add('error');
    status.textContent = `Sampler failed: ${error.message}`;
  } finally {
    setSamplerLoading(false);
  }
}

// ---------- Bulk-labeling runs (X4) ----------
// Web reads run outputs written by the pipeline runner under
//   data/runs/<run_id>/web/{summary,borderline,misalignment}.json
// All shapes are tolerant: missing fields render an empty-state row, never a crash.
//
// IMPORTANT: provider calls downsample every image (longest edge ≤1024,
// JPEG quality≈85) before submission. JSON outputs never embed image bytes;
// when present, optional prepared-image audit metadata is surfaced in the UI:
//   prepared_image: { sha256, width, height, byte_size, mime_type, longest_edge, jpeg_quality }

const RUNS_INDEX_URL = '../data/runs/index.json';
const RUNS_DIR_URL = '../data/runs/';
const KNOWN_LABELS = ['gen_ai', 'not_gen_ai', 'abstain'];

const runState = {
  available: [],
  selectedRunId: null,
  summary: null,
  borderline: null,
  misalignment: null,
  consensus: null,
  consensusFilter: 'all'
};

async function fetchJsonOptional(path) {
  try {
    const response = await fetch(path, { cache: 'no-store' });
    if (!response.ok) return null;
    return await response.json();
  } catch (error) {
    return null;
  }
}

async function discoverRuns() {
  // Preferred: an index file produced by the runner / scoring exporter.
  const index = await fetchJsonOptional(RUNS_INDEX_URL);
  if (index && Array.isArray(index.runs)) {
    return index.runs.filter(r => r && r.run_id).map(r => ({
      run_id: r.run_id,
      label: r.label || r.run_id,
      started_at: r.started_at || null
    })).sort((a, b) => (b.started_at || '').localeCompare(a.started_at || ''));
  }
  return [];
}

async function loadRun(runId) {
  if (!runId) {
    runState.selectedRunId = null;
    runState.summary = null;
    runState.borderline = null;
    runState.misalignment = null;
    renderRun();
    return;
  }
  const status = $('#runStatus');
  if (status) {
    status.classList.remove('error');
    status.textContent = `Loading run ${runId}…`;
  }
  const base = `${RUNS_DIR_URL}${encodeURIComponent(runId)}/web`;
  const [summary, borderline, misalignment, consensus] = await Promise.all([
    fetchJsonOptional(`${base}/summary.json`),
    fetchJsonOptional(`${base}/borderline.json`),
    fetchJsonOptional(`${base}/misalignment.json`),
    fetchJsonOptional(`${base}/consensus.json`)
  ]);
  runState.selectedRunId = runId;
  runState.summary = summary;
  runState.borderline = borderline;
  runState.misalignment = misalignment;
  runState.consensus = consensus;
  if (status) {
    if (!summary && !borderline && !misalignment && !consensus) {
      status.classList.add('error');
      status.textContent = `No web exports found for run ${runId}.`;
    } else {
      status.textContent = `Loaded run ${runId}.`;
    }
  }
  // Wire the per-run policy.pdf link if the file exists at the expected path.
  const policyLink = $('#policyPdfLink');
  if (policyLink) {
    policyLink.href = `${RUNS_DIR_URL}${encodeURIComponent(runId)}/policy.pdf`;
    policyLink.dataset.runId = runId;
  }
  renderRun();
}

function renderRunPicker() {
  const picker = $('#runPicker');
  if (!picker) return;
  if (!runState.available.length) {
    picker.innerHTML = '<option value="">— no runs found —</option>';
    picker.disabled = true;
    return;
  }
  picker.disabled = false;
  picker.innerHTML = runState.available.map(r => {
    const label = r.started_at ? `${r.run_id} · ${r.started_at}` : r.run_id;
    const selected = r.run_id === runState.selectedRunId ? ' selected' : '';
    return `<option value="${attr(r.run_id)}"${selected}>${esc(label)}</option>`;
  }).join('');
}

function renderRunSummary() {
  const target = $('#runSummary');
  if (!target) return;
  if (!runState.summary) {
    target.innerHTML = '';
    return;
  }
  const s = runState.summary;
  const cards = [
    ['Run id', s.run_id || runState.selectedRunId || '—', s.started_at || ''],
    ['Models', Array.isArray(s.models) ? s.models.length : (s.model_count ?? '—'), Array.isArray(s.models) ? s.models.join(' · ') : ''],
    ['Images', s.image_count ?? s.n_images ?? '—', s.split ? `split: ${s.split}` : ''],
    ['Policy graph', s.policy_graph_version || '—', s.prompt_version ? `prompt ${s.prompt_version}` : '']
  ];
  target.innerHTML = cards.map(([k, v, n]) =>
    `<article class="stat-card"><span>${esc(k)}</span><strong>${esc(v)}</strong><p>${esc(n || '')}</p></article>`
  ).join('');
}

function preparedMetaLine(prepared) {
  if (!prepared || typeof prepared !== 'object') return '';
  const bits = [];
  if (prepared.width && prepared.height) bits.push(`${prepared.width}×${prepared.height}px`);
  if (typeof prepared.byte_size === 'number') bits.push(`${prepared.byte_size.toLocaleString()} bytes`);
  if (prepared.mime_type) bits.push(esc(prepared.mime_type));
  if (prepared.longest_edge) bits.push(`longest edge ≤${prepared.longest_edge}`);
  if (typeof prepared.jpeg_quality === 'number') bits.push(`q≈${prepared.jpeg_quality}`);
  if (prepared.sha256) bits.push(`sha256 ${esc(String(prepared.sha256).slice(0, 12))}…`);
  if (!bits.length) return '';
  return `<p class="prepared-meta" title="Bytes the providers actually saw"><strong>Prepared image:</strong> ${bits.join(' · ')}</p>`;
}

function renderBorderline() {
  const empty = $('#borderlineEmpty');
  const target = $('#borderlineGroups');
  if (!target) return;
  const data = runState.borderline;
  if (!data || !Array.isArray(data.groups) || data.groups.length === 0) {
    if (empty) empty.hidden = false;
    target.innerHTML = '';
    return;
  }
  if (empty) empty.hidden = true;
  target.innerHTML = data.groups.map(group => {
    const items = (group.items || []).slice(0, 12);
    const itemHtml = items.map(item => {
      const id = item.image_id || item.sample_id || '';
      const reason = item.reason || item.borderline_reason || '';
      const conf = (typeof item.confidence === 'number') ? `confidence ${item.confidence.toFixed(2)}` : '';
      const diff = item.difficulty ? `difficulty ${esc(item.difficulty)}` : '';
      return `<li><strong>${esc(id)}</strong>${reason ? ` — ${esc(reason)}` : ''}<span class="row-meta">${[conf, diff].filter(Boolean).join(' · ')}</span>${preparedMetaLine(item.prepared_image)}</li>`;
    }).join('');
    const heading = group.l0 || group.label || group.bucket || 'unbucketed';
    return `<article class="borderline-group"><header><span class="badge ${KNOWN_LABELS.includes(heading) ? heading.replace('_', '-') : 'dev'}">${esc(heading)}</span><strong>${(group.items || []).length} case(s)</strong></header><ul>${itemHtml || '<li class="muted">no items</li>'}</ul></article>`;
  }).join('');
}

function renderMisalignment() {
  const empty = $('#misalignmentEmpty');
  const target = $('#misalignmentTable');
  if (!target) return;
  const data = runState.misalignment;
  if (!data || !Array.isArray(data.rows) || data.rows.length === 0) {
    if (empty) empty.hidden = false;
    target.innerHTML = '';
    return;
  }
  if (empty) empty.hidden = true;
  const models = Array.isArray(data.models) ? data.models : [];
  const headerCells = ['<th>image</th>', '<th>SME truth</th>',
    ...models.map(m => `<th>${esc(m)}</th>`),
    '<th>agreement</th>', '<th>reason</th>', '<th>patch</th>'].join('');
  const rows = data.rows.slice(0, 100).map(row => {
    const id = row.image_id || row.sample_id || '';
    const thumb = row.image_path ? `<img class="row-thumb" src="${attr('../' + row.image_path.replace(/^\.\//, ''))}" alt="${attr(id)}" loading="lazy" onerror="this.replaceWith(safeImageFallback('image unavailable','local path missing'))" />` : '';
    const sme = row.sme_truth || row.truth || '—';
    const perModel = models.map(m => {
      const vote = (row.model_labels && row.model_labels[m]) || '—';
      const cls = KNOWN_LABELS.includes(vote) ? vote.replace('_', '-') : 'dev';
      return `<td><span class="badge ${cls}">${esc(vote)}</span></td>`;
    }).join('');
    const agreement = row.agreement || (row.unanimous ? 'unanimous' : 'split');
    const reason = row.disagreement_reason || '';
    const patch = row.policy_patch_id
      ? `<a href="${attr(row.policy_patch_url || '#')}">${esc(row.policy_patch_id)}</a>`
      : '<span class="muted">—</span>';
    return `<tr><td><div class="thumb-wrap">${thumb}<div><strong>${esc(id)}</strong>${preparedMetaLine(row.prepared_image)}</div></div></td><td><span class="badge ${KNOWN_LABELS.includes(sme) ? sme.replace('_', '-') : 'dev'}">${esc(sme)}</span></td>${perModel}<td>${esc(agreement)}</td><td>${esc(reason)}</td><td>${patch}</td></tr>`;
  }).join('');
  target.innerHTML = `<table class="misalignment"><thead><tr>${headerCells}</tr></thead><tbody>${rows}</tbody></table>`;
}

function renderConsensus() {
  const data = runState.consensus;
  const summaryTarget = $('#consensusSummary');
  const emptyEl = $('#consensusEmpty');
  const tableTarget = $('#consensusTable');
  if (!tableTarget) return;
  if (!data || !Array.isArray(data.records) || data.records.length === 0) {
    if (summaryTarget) summaryTarget.innerHTML = '';
    if (emptyEl) emptyEl.hidden = false;
    tableTarget.innerHTML = '';
    return;
  }
  if (emptyEl) emptyEl.hidden = true;

  const s = data.summary || {};
  if (summaryTarget) {
    const cards = [
      ['Images', s.n_images_total ?? data.records.length, ''],
      ['Unanimous', s.n_images_unanimous ?? '—', ''],
      ['Split', s.n_images_split ?? '—', ''],
      ['Ties', s.n_images_with_tie ?? '—', ''],
      ['Boundary-flagged', s.n_images_with_boundary_flag ?? '—', ''],
      ['Majority vs SME', isNumber(s.majority_vs_sme_accuracy) ? `${(s.majority_vs_sme_accuracy * 100).toFixed(1)}%` : '—',
        isNumber(s.majority_vs_sme_compared) ? `${s.majority_vs_sme_correct ?? 0} / ${s.majority_vs_sme_compared}` : '']
    ];
    summaryTarget.innerHTML = cards.map(([k, v, n]) =>
      `<article class="stat-card"><span>${esc(k)}</span><strong>${esc(v)}</strong><p>${esc(n || '')}</p></article>`
    ).join('');
  }

  // Build sme_truth lookup from misalignment payload (which carries SME truth per image).
  const smeMap = {};
  const misRecords = runState.misalignment?.records || [];
  for (const r of misRecords) {
    if (r && r.image_id) smeMap[r.image_id] = r.sme_truth;
  }

  const filter = runState.consensusFilter || 'all';
  const records = data.records.filter(r => {
    if (filter === 'unanimous') return !!r.is_unanimous;
    if (filter === 'split') return !!r.is_split;
    if (filter === 'boundary') return !!r.any_boundary_flag;
    return true;
  });
  const statusEl = $('#consensusStatus');
  if (statusEl) statusEl.textContent = `Showing ${records.length} of ${data.records.length} image(s).`;

  if (records.length === 0) {
    tableTarget.innerHTML = '<p class="muted">No images match the current filter.</p>';
    return;
  }

  // Determine column order of models across the dataset (stable, sorted).
  const modelSet = new Set();
  for (const r of data.records) {
    for (const v of (r.voters || [])) modelSet.add(v.labeler_id || v.model_id || 'unknown');
  }
  const models = Array.from(modelSet).sort();

  const headerCells = [
    '<th>image</th>',
    '<th>SME truth</th>',
    ...models.map(m => `<th>${esc(m)}</th>`),
    '<th>consensus</th>',
    '<th>distribution</th>'
  ].join('');

  const chipFor = r => {
    const tot = r.n_votes_total ?? (r.voters || []).length;
    if (r.tie) return `<span class="badge consensus-tie" title="tie">⚠ tie ${r.majority_count}/${tot}</span>`;
    if (r.is_unanimous) return `<span class="badge consensus-unanimous" title="all voters agreed">✓ unanimous ${r.majority_count}/${tot}</span>`;
    if (r.majority_label) {
      const dec = r.n_votes_decisive ?? r.majority_count;
      return `<span class="badge consensus-majority" title="majority among decisive voters">majority ${r.majority_count}/${dec}</span>`;
    }
    return '<span class="badge dev" title="no majority">no majority</span>';
  };

  const rows = records.slice(0, 200).map(r => {
    const sme = smeMap[r.image_id];
    const smeBadge = sme
      ? `<span class="badge ${KNOWN_LABELS.includes(sme) ? sme.replace('_', '-') : 'dev'}">${esc(sme)}</span>`
      : '<span class="muted">—</span>';
    const voterById = {};
    for (const v of (r.voters || [])) voterById[v.labeler_id || v.model_id || 'unknown'] = v;
    const perModel = models.map(m => {
      const v = voterById[m];
      if (!v) return '<td><span class="muted">—</span></td>';
      const cls = KNOWN_LABELS.includes(v.label) ? v.label.replace('_', '-') : 'dev';
      const boundary = v.is_boundary ? ' · boundary' : '';
      const conf = isNumber(v.confidence) ? ` (${v.confidence.toFixed(2)})` : '';
      return `<td><span class="badge ${cls}" title="confidence${conf}${boundary}">${esc(v.label)}</span></td>`;
    }).join('');
    const distChips = Object.entries(r.vote_distribution || {})
      .sort((a, b) => b[1] - a[1])
      .map(([lbl, cnt]) => `<span class="dist-chip ${KNOWN_LABELS.includes(lbl) ? lbl.replace('_', '-') : 'dev'}">${esc(lbl)}: ${cnt}</span>`)
      .join(' ');
    const mismatch = sme && r.majority_label && r.majority_label !== sme;
    const rowCls = mismatch ? ' class="row-mismatch"' : '';
    return `<tr${rowCls}><td><strong>${esc(r.image_id)}</strong>${mismatch ? '<p class="row-meta mismatch-note">majority ≠ SME</p>' : ''}</td><td>${smeBadge}</td>${perModel}<td>${chipFor(r)}</td><td>${distChips || '<span class="muted">—</span>'}</td></tr>`;
  }).join('');
  tableTarget.innerHTML = `<table class="misalignment"><thead><tr>${headerCells}</tr></thead><tbody>${rows}</tbody></table>`;
}

function renderRun() {
  renderRunPicker();
  renderRunSummary();
  renderBorderline();
  renderMisalignment();
  renderConsensus();
}

async function refreshRuns(autoSelectMostRecent = true) {
  const status = $('#runStatus');
  if (status) {
    status.classList.remove('error');
    status.textContent = 'Looking for runs…';
  }
  runState.available = await discoverRuns();
  if (status) {
    if (!runState.available.length) {
      status.textContent = 'No runs found yet. Run scripts/run_bulk_labeling.py to create one.';
    } else {
      status.textContent = `Found ${runState.available.length} run(s).`;
    }
  }
  renderRunPicker();
  if (autoSelectMostRecent && runState.available.length) {
    await loadRun(runState.available[0].run_id);
  } else {
    renderRun();
  }
}

function bindControls() {
  $('#runSampler')?.addEventListener('click', runSamplerDemo);
  $('#randomSamplerSeed')?.addEventListener('click', () => {
    $('#samplerSeed').value = String(Math.floor(100000 + Math.random() * 2140000000));
    runSamplerDemo();
  });
  document.body.addEventListener('input', event => {
    const target = event.target;
    const sampleId = target?.dataset?.sampleId;
    if (!sampleId) return;
    const current = overrideFor(sampleId);
    if (target.classList.contains('override-note')) current.note = target.value;
    if (target.classList.contains('override-label')) current.label = target.value;
    demoState.overrides[sampleId] = current;
    renderQuality();
  });
  document.body.addEventListener('change', event => {
    const target = event.target;
    const sampleId = target?.dataset?.sampleId;
    if (!sampleId || !target.classList.contains('override-label')) return;
    demoState.overrides[sampleId] = { ...overrideFor(sampleId), label: target.value };
    renderQuality();
  });
}

function bindRunControls() {
  $('#runPicker')?.addEventListener('change', event => loadRun(event.target.value));
  $('#refreshRuns')?.addEventListener('click', () => refreshRuns(true));
  $('#consensusFilter')?.addEventListener('change', event => {
    runState.consensusFilter = event.target.value || 'all';
    renderConsensus();
  });
}

function init() {
  $('#policyNodeList').innerHTML = '';
  bindControls();
  bindRunControls();
  runSamplerDemo();
  refreshRuns(true);
}

init();
