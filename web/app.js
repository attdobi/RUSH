const $ = selector => document.querySelector(selector);
const esc = value => String(value ?? '').replace(/[&<>'"]/g, char => ({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}[char]));
const attr = esc;
const isNumber = value => typeof value === 'number' && Number.isFinite(value);

window.rushIsEnsembleRow = function(row) {
  if (!row) return false;
  const id = String(row.labeler_id || row.model_id || '').toLowerCase();
  const type = String(row.labeler_type || '').toLowerCase();
  return id === 'majority_vote' || type === 'ensemble';
};

window.rushSortEnsembleLast = function(rows, primaryCompare) {
  const cmp = primaryCompare || (() => 0);
  return rows.slice().sort((a, b) => {
    const ae = window.rushIsEnsembleRow(a) ? 1 : 0;
    const be = window.rushIsEnsembleRow(b) ? 1 : 0;
    if (ae !== be) return ae - be;
    return cmp(a, b);
  });
};

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
  overrides: {},
  galleryFilter: 'all',
  galleryVisibleCount: 24
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

function thumbnailSrcForPath(repoRelPath) {
  const path = String(repoRelPath || '').replace(/^\.\//, '').replace(/^\/+/, '');
  if (!path) return '';
  if (window.RUSH_API?.available) return `/api/thumbnail?path=${encodeURIComponent(path)}`;
  return `../${path}`;
}

function imgSrc(row) {
  return thumbnailSrcForPath(row.repo_rel_path || row.synthetic_repo_rel_path);
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
  if (src) {
    return `<img class="thumb-loading" src="${attr(src)}" alt="${attr(row.sample_id)} ${attr(row.label)}" loading="lazy" decoding="async" onload="this.classList.remove('thumb-loading')" onerror="this.replaceWith(safeImageFallback())" />`;
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

function zipSplits(devGolden = [], holdout = []) {
  const rows = [];
  const max = Math.max(devGolden.length, holdout.length);
  for (let index = 0; index < max; index += 1) {
    if (devGolden[index]) rows.push(devGolden[index]);
    if (holdout[index]) rows.push(holdout[index]);
  }
  return rows;
}

function interleaveByLabel(records) {
  const buckets = {
    ai_generated: records.filter(row => row.label === 'ai_generated'),
    not_ai_generated: records.filter(row => row.label === 'not_ai_generated'),
    other: records.filter(row => row.label !== 'ai_generated' && row.label !== 'not_ai_generated')
  };
  const rows = [];
  const max = Math.max(buckets.ai_generated.length, buckets.not_ai_generated.length);
  for (let index = 0; index < max; index += 1) {
    if (buckets.ai_generated[index]) rows.push(buckets.ai_generated[index]);
    if (buckets.not_ai_generated[index]) rows.push(buckets.not_ai_generated[index]);
  }
  return rows.concat(buckets.other);
}

function isNeedsReview(row) {
  return overrideFor(row.sample_id).label === 'needs_review' || row.human_override_label === 'needs_review';
}

function galleryRecords() {
  const result = demoState.result;
  if (!result) return [];
  const ordered = interleaveByLabel(zipSplits(result.devGolden || [], result.holdout || []));
  const filter = demoState.galleryFilter || 'all';
  if (filter === 'all') return ordered;
  if (filter === 'needs_review') return ordered.filter(isNeedsReview);
  return ordered.filter(row => row.label === filter);
}

function updateGalleryControls(total, visibleCount) {
  document.querySelectorAll('[data-gallery-filter]').forEach(button => {
    const active = button.dataset.galleryFilter === demoState.galleryFilter;
    button.classList.toggle('active', active);
    button.setAttribute('aria-pressed', String(active));
  });
  const loadMore = $('#galleryLoadMore');
  if (loadMore) {
    const hasMore = visibleCount < total;
    loadMore.hidden = !hasMore;
    loadMore.textContent = hasMore ? `Load more (${visibleCount}/${total})` : 'All samples loaded';
  }
}

function renderGallery() {
  const result = demoState.result;
  if (!result) return;
  const records = galleryRecords();
  const visibleCount = Math.min(demoState.galleryVisibleCount || 24, records.length);
  const visible = records.slice(0, visibleCount);
  updateGalleryControls(records.length, visible.length);
  const status = $('#galleryStatus');
  if (status) {
    const filter = demoState.galleryFilter === 'all' ? 'balanced mix' : demoState.galleryFilter;
    status.textContent = `Showing ${visible.length} of ${records.length} ${filter} sample(s).`;
  }
  $('#sampleGallery').innerHTML = `
    <div class="gallery-head">
      <div><h3>Visible sample gallery</h3><p>Showing a balanced, split-aware sample order from the sampled set. These same records drive the policy loop and decision-quality preview below.</p></div>
      <span class="quiet-pill">${esc(result.summary.mode || 'cold_start')}</span>
    </div>
    ${visible.length ? `<div class="sample-grid">${visible.map(row => renderSampleCard(row)).join('')}</div>` : '<div class="empty-state">No samples match this filter yet.</div>'}`;
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

function renderAll() {
  renderStats();
  renderGallery();
  renderPolicy();
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
  demoState.galleryVisibleCount = 24;
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

// Single source of truth for label → CSS class.
// IMPORTANT: replaceAll, not replace — `not_gen_ai` has TWO underscores.
function labelBadgeClass(label) {
  if (!label || !KNOWN_LABELS.includes(label)) return 'dev';
  return label.replaceAll('_', '-');
}
window.rushLabelBadgeClass = labelBadgeClass;

const runState = {
  available: [],
  selectedRunId: null,
  summary: null,
  borderline: null,
  misalignment: null,
  consensus: null,
  consensusFilter: 'all',
  expandedRows: {}
};
window.runState = runState;

const flipRateState = {
  data: null,
  byImage: {},
  source: null
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

function assertFlipRateShape(data) {
  const warn = message => console.warn(`flip_rate.json shape: ${message}`);
  if (!data || typeof data !== 'object' || Array.isArray(data)) {
    warn('expected a top-level object.');
    return false;
  }

  const expectedTop = ['summary', 'records'];
  for (const key of expectedTop) {
    if (!(key in data)) warn(`missing top-level key "${key}".`);
  }
  const extraTop = Object.keys(data).filter(key => !expectedTop.includes(key));
  if (extraTop.length) warn(`extra top-level key(s): ${extraTop.join(', ')}.`);

  const summary = data.summary || {};
  const expectedSummary = [
    'n_pairs_total', 'n_pairs_stable', 'n_pairs_flipped', 'n_pairs_single_run',
    'mean_flip_rate', 'per_model_flip_rate', 'top_flipped_images', 'computed_at'
  ];
  if (!data.summary || typeof data.summary !== 'object' || Array.isArray(data.summary)) {
    warn('missing or invalid summary object.');
  } else {
    for (const key of expectedSummary) {
      if (!(key in summary)) warn(`summary missing key "${key}".`);
    }
    const extraSummary = Object.keys(summary).filter(key => !expectedSummary.includes(key));
    if (extraSummary.length) warn(`summary extra key(s): ${extraSummary.join(', ')}.`);
  }

  if (!Array.isArray(data.records)) {
    warn('records should be an array.');
    return false;
  }
  const sample = data.records.find(record => record && typeof record === 'object');
  if (sample) {
    const expectedRecord = [
      'image_id', 'model_id', 'n_runs', 'labels_observed', 'label_counts', 'distinct_label_count',
      'flip_count', 'flip_rate', 'stable_label', 'abstain_count', 'confidence_min', 'confidence_max',
      'confidence_mean', 'first_seen_run_id', 'last_seen_run_id', 'run_ids', 'single_run_only'
    ];
    for (const key of expectedRecord) {
      if (!(key in sample)) warn(`record missing key "${key}".`);
    }
    const extraRecord = Object.keys(sample).filter(key => !expectedRecord.includes(key));
    if (extraRecord.length) warn(`record extra key(s): ${extraRecord.join(', ')}.`);
  }
  return true;
}

function buildFlipRateIndex(records) {
  return (records || []).reduce((acc, record) => {
    const imageId = record?.image_id;
    if (!imageId) return acc;
    const key = String(imageId);
    if (!acc[key]) acc[key] = [];
    acc[key].push(record);
    return acc;
  }, {});
}

function formatFlipRate(value) {
  return isNumber(value) ? value.toFixed(2) : '—';
}

function formatPct(numerator, denominator) {
  if (!isNumber(numerator) || !isNumber(denominator) || denominator <= 0) return '—';
  return `${((numerator / denominator) * 100).toFixed(1)}%`;
}

function flipLabelPills(labels) {
  if (!Array.isArray(labels) || labels.length === 0) return '<span class="muted">—</span>';
  return labels.map(label => `<span class="flip-label-pill">${esc(label)}</span>`).join(' ');
}

function renderFlipBadgeForImage(imageId) {
  if (!flipRateState.data) return '';
  const records = flipRateState.byImage[String(imageId || '')] || [];
  const multiRunRecords = records.filter(record => !record.single_run_only && Number(record.n_runs || 0) > 1);
  if (!multiRunRecords.length) {
    return `<span class="flip-badge single" title="No multi-run flip-rate data for this image">single run</span>`;
  }
  const maxFlipCount = Math.max(...multiRunRecords.map(record => Number(record.flip_count || 0)));
  if (maxFlipCount >= 2) {
    return `<span class="flip-badge bad" title="At least one model flipped ${maxFlipCount} times for this image">2+ flips</span>`;
  }
  if (maxFlipCount === 1) {
    return '<span class="flip-badge warn" title="At least one model flipped once for this image">1 flip</span>';
  }
  return '<span class="flip-badge stable" title="All multi-run model pairs were stable for this image">stable</span>';
}

function unstableImageCount(records) {
  const ids = new Set();
  for (const record of records || []) {
    if (record?.image_id && Number(record.flip_count || 0) > 0) ids.add(record.image_id);
  }
  return ids.size;
}

async function loadFlipRate() {
  const fixture = (typeof window !== 'undefined' && window.__FLIP_RATE_DEV__ && typeof window.__FLIP_RATE_DEV__ === 'object')
    ? window.__FLIP_RATE_DEV__
    : null;
  const data = fixture || await fetchJsonOptional('flip_rate.json');
  if (data) {
    assertFlipRateShape(data);
    flipRateState.data = data;
    flipRateState.byImage = buildFlipRateIndex(Array.isArray(data.records) ? data.records : []);
    flipRateState.source = fixture ? 'dev fixture' : 'flip_rate.json';
  } else {
    flipRateState.data = null;
    flipRateState.byImage = {};
    flipRateState.source = null;
  }
  renderFlipRate();
  renderConsensus();
}

function renderFlipRate() {
  const panel = $('#flip-rate-panel');
  if (!panel) return;
  const empty = $('#flipRateEmpty');
  const cardsTarget = $('#flipRateCards');
  const barsTarget = $('#flipRateBars');
  const tableTarget = $('#flipRateTable');
  const data = flipRateState.data;
  if (!data || !Array.isArray(data.records)) {
    panel.hidden = !runState.selectedRunId;
    showComputeEmpty(empty, 'flip-rate', 'No flip-rate data yet — need ≥2 scored runs of the same images.');
    if (cardsTarget) cardsTarget.innerHTML = '';
    if (barsTarget) barsTarget.innerHTML = '';
    if (tableTarget) tableTarget.innerHTML = '';
    return;
  }
  panel.hidden = false;
  if (empty) empty.hidden = true;

  const summary = data.summary || {};
  const total = Number(summary.n_pairs_total || 0);
  if (cardsTarget) {
    const cards = [
      ['Mean Flip Rate', formatFlipRate(summary.mean_flip_rate), flipRateState.source || ''],
      ['% Stable', formatPct(Number(summary.n_pairs_stable || 0), total), `${summary.n_pairs_stable ?? 0} / ${summary.n_pairs_total ?? 0} pairs`],
      ['% Flipped', formatPct(Number(summary.n_pairs_flipped || 0), total), `${summary.n_pairs_flipped ?? 0} / ${summary.n_pairs_total ?? 0} pairs`],
      ['Unstable Images', unstableImageCount(data.records), 'distinct images with ≥1 flipped model pair']
    ];
    cardsTarget.innerHTML = cards.map(([label, value, note]) =>
      `<article class="stat-card"><span>${esc(label)}</span><strong>${esc(value)}</strong><p>${esc(note || '')}</p></article>`
    ).join('');
  }

  if (barsTarget) {
    const perModel = summary.per_model_flip_rate || {};
    const modelEntries = Object.entries(perModel)
      .sort((a, b) => Number(b[1]?.mean_flip_rate || 0) - Number(a[1]?.mean_flip_rate || 0));
    if (!modelEntries.length) {
      barsTarget.innerHTML = '<div class="empty-state">No per-model flip-rate records available.</div>';
    } else {
      const observedMax = Math.max(...modelEntries.map(([, stats]) => Number(stats?.mean_flip_rate || 0)), 0);
      const denominator = Math.max(0.5, observedMax);
      barsTarget.innerHTML = modelEntries.map(([modelId, stats]) => {
        const rate = Number(stats?.mean_flip_rate || 0);
        const width = denominator > 0 ? Math.max(0, Math.min(100, (rate / denominator) * 100)) : 0;
        return `<article class="flip-bar">
          <header><strong>${esc(modelId)}</strong><span>${formatFlipRate(rate)} · ${esc(stats?.n_pairs_flipped ?? 0)} / ${esc(stats?.n_pairs ?? 0)} flipped</span></header>
          <div class="flip-bar-track" aria-hidden="true"><span style="width:${width.toFixed(1)}%"></span></div>
        </article>`;
      }).join('');
    }
  }

  if (tableTarget) {
    const topSource = Array.isArray(summary.top_flipped_images) && summary.top_flipped_images.length
      ? summary.top_flipped_images
      : data.records;
    const topRows = topSource
      .filter(record => Number(record.flip_count || 0) > 0)
      .slice()
      .sort((a, b) => Number(b.flip_rate || 0) - Number(a.flip_rate || 0) || Number(b.flip_count || 0) - Number(a.flip_count || 0))
      .slice(0, 20);
    if (!topRows.length) {
      tableTarget.innerHTML = '<div class="empty-state">No flipped images yet — all multi-run pairs are stable.</div>';
    } else {
      const rows = topRows.map(row => `<tr><td><strong>${esc(row.image_id)}</strong></td><td>${esc(row.model_id)}</td><td>${esc(row.n_runs ?? '—')}</td><td>${esc(row.flip_count ?? '—')}</td><td>${formatFlipRate(Number(row.flip_rate))}</td><td>${flipLabelPills(row.labels_observed)}</td></tr>`).join('');
      tableTarget.innerHTML = `<table class="misalignment"><thead><tr><th>image</th><th>model</th><th>runs</th><th>flips</th><th>flip rate</th><th>labels observed</th></tr></thead><tbody>${rows}</tbody></table>`;
    }
  }
}

async function discoverRuns() {
  if (window.RUSH_API?.ready) await window.RUSH_API.ready.catch(() => null);
  if (window.RUSH_API?.available) {
    const payload = await rushApiGetJson('/api/runs').catch(() => null);
    if (payload && Array.isArray(payload.runs)) {
      return payload.runs.filter(r => r && r.run_id).map(r => ({
        run_id: r.run_id,
        label: r.label || r.run_id,
        started_at: r.started_at || null,
        scoring_done: !!r.scoring_done
      })).sort((a, b) => (b.started_at || '').localeCompare(a.started_at || ''));
    }
  }
  // Fallback: an index file produced by the runner / scoring exporter.
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
  if (runState.selectedRunId !== runId) runState.expandedRows = {};
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
    const suffix = r.scoring_done === false ? ' · unscored' : '';
    const label = r.started_at ? `${r.run_id} · ${r.started_at}${suffix}` : `${r.run_id}${suffix}`;
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

function normalizedBorderlineGroups(groups) {
  if (Array.isArray(groups)) return groups;
  if (groups && typeof groups === 'object') {
    return Object.entries(groups).map(([label, items]) => ({ label, items: Array.isArray(items) ? items : [] }));
  }
  return [];
}

function showComputeEmpty(empty, panel, reason) {
  if (!empty) return;
  if (!runState.selectedRunId) {
    empty.hidden = true;
    return;
  }
  empty.hidden = false;
  empty.innerHTML = `${esc(reason)} <button type="button" data-compute-target="${attr(panel)}" data-run-id="">Compute now</button>`;
}

function expandedKey(panel, imageId) {
  return `${panel}:${String(imageId || '')}`;
}

function isExpanded(panel, imageId) {
  return !!runState.expandedRows[expandedKey(panel, imageId)];
}

function expandButton(panel, imageId) {
  const expanded = isExpanded(panel, imageId);
  return `<button type="button" class="expand-row" data-expand-row="${attr(panel)}" data-image-id="${attr(imageId)}" aria-expanded="${expanded ? 'true' : 'false'}" title="${expanded ? 'Hide' : 'Show'} model justifications">${expanded ? '▾' : '▸'}</button>`;
}

function voteNumber(value, digits = 2) {
  if (value == null || value === '') return '—';
  const n = Number(value);
  return Number.isFinite(n) ? n.toFixed(digits) : String(value);
}

function voteCost(value) {
  if (value == null || value === '') return '—';
  const n = Number(value);
  return Number.isFinite(n) ? `$${n.toFixed(5)}` : String(value);
}

function voteBool(value) {
  if (value === true) return 'yes';
  if (value === false) return 'no';
  return '—';
}

function voteToken(vote, key) {
  const value = vote?.[key];
  if (value == null || value === '') return '—';
  const n = Number(value);
  return Number.isFinite(n) ? n.toLocaleString() : String(value);
}

function votesForInline(row) {
  if (Array.isArray(row?.votes)) return row.votes;
  if (Array.isArray(row?.voters)) return row.voters;
  return [];
}

// Plurality tally over non-ensemble votes. Returns {label, isTie, counts, total}.
function tallyVotes(row) {
  const counts = new Map();
  let total = 0;
  for (const vote of (Array.isArray(row.votes) ? row.votes : [])) {
    if (window.rushIsEnsembleRow(vote)) continue;
    const label = vote?.label;
    if (!label) continue;
    counts.set(label, (counts.get(label) || 0) + 1);
    total += 1;
  }
  if (!counts.size) return { label: null, isTie: false, counts, total };
  const ranked = Array.from(counts.entries()).sort((a, b) => b[1] - a[1]);
  const topCount = ranked[0][1];
  const tied = ranked.filter(([, c]) => c === topCount);
  if (tied.length > 1) {
    return { label: null, isTie: true, tiedLabels: tied.map(([l]) => l), counts, total };
  }
  return { label: ranked[0][0], isTie: false, counts, total };
}
window.rushTallyVotes = tallyVotes;

function majorityPill(row) {
  const t = tallyVotes(row);
  if (t.isTie) {
    const tip = `tie between ${t.tiedLabels.join(' / ')}`;
    return `<span class="badge dev" title="${attr(tip)}">tie</span>`;
  }
  if (!t.label) return '<span class="muted">—</span>';
  return `<span class="badge ${labelBadgeClass(t.label)}">${esc(t.label)}</span>`;
}

function policyCitationHtml(citation) {
  const id = String(citation || '').trim();
  if (!id) return '';
  return `<button type="button" class="policy-citation-chip" data-policy-node-id="${attr(id)}" title="Open policy node ${attr(id)}">${esc(id)}</button>`;
}

function renderInlineJustificationsRow(panel, row, colSpan) {
  const imageId = row?.image_id || row?.sample_id || '';
  if (!isExpanded(panel, imageId)) return '';
  const votes = votesForInline(row);
  const cards = votes.length ? votes.map(vote => {
    const model = vote.labeler_id || vote.model_id || 'unknown model';
    const citations = Array.isArray(vote.policy_citations) ? vote.policy_citations.filter(Boolean) : [];
    const quotes = Array.isArray(vote.policy_quotes) ? vote.policy_quotes.filter(Boolean) : [];
    return `<article class="inline-justification-card">
      <header><code>${esc(model)}</code><span>${voteCost(vote.cost_usd)}</span></header>
      ${citations.length ? `<div class="policy-citation-row">${citations.map(policyCitationHtml).join('')}</div>` : ''}
      <dl>
        <dt>label</dt><dd><span class="badge ${labelBadgeClass(vote.label)}">${esc(vote.label || '—')}</span></dd>
        <dt>l2_label</dt><dd><code>${esc(vote.l2_label || '—')}</code></dd>
        <dt>confidence</dt><dd><code>${voteNumber(vote.confidence)}</code></dd>
        <dt>boundary</dt><dd>${esc(voteBool(vote.is_boundary))}</dd>
        <dt>difficulty</dt><dd>${esc(vote.difficulty || '—')}${vote.justification_too_long ? ' <span class="mini-chip">too long</span>' : ''}</dd>
        <dt>input</dt><dd><code>${voteToken(vote, 'input_tokens')}</code></dd>
        <dt>output</dt><dd><code>${voteToken(vote, 'output_tokens')}</code></dd>
      </dl>
      <p>${esc(vote.justification || 'No justification text available for this vote.')}</p>
      ${quotes.length ? `<div class="policy-quotes">${quotes.map(quote => `<blockquote>${esc(quote)}</blockquote>`).join('')}</div>` : ''}
    </article>`;
  }).join('') : '<p class="muted">No per-model vote details available for this row.</p>';
  return `<tr class="justification-row" data-image-id="${attr(imageId)}"><td colspan="${colSpan}"><div class="inline-justification-grid">${cards}</div></td></tr>`;
}

function initInlineJustificationStyles() {
  if (document.getElementById('inlineJustificationStyles')) return;
  const style = document.createElement('style');
  style.id = 'inlineJustificationStyles';
  style.textContent = `
    .misalignment-table { overflow-x: auto; }
    .expand-row { border: 1px solid rgba(148, 163, 184, 0.35); border-radius: 999px; background: rgba(15, 23, 42, 0.04); cursor: pointer; width: 1.8rem; height: 1.8rem; }
    .image-id-button { border: 0; background: transparent; color: inherit; cursor: pointer; padding: 0; font: inherit; text-align: left; }
    .image-id-button strong { text-decoration: underline; text-decoration-style: dotted; text-underline-offset: 3px; }
    .justification-row td { background: rgba(15, 23, 42, 0.035); }
    .inline-justification-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 12px; padding: 10px 0; }
    .inline-justification-card { border: 1px solid rgba(148, 163, 184, 0.32); border-radius: 14px; padding: 10px; background: linear-gradient(180deg, rgba(17, 27, 51, .94), rgba(9, 16, 31, .94)); }
    .inline-justification-card header { display: flex; justify-content: space-between; gap: 10px; align-items: baseline; margin-bottom: 8px; }
    .inline-justification-card code { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 0.82em; }
    .inline-justification-card dl { display: grid; grid-template-columns: 84px 1fr; gap: 4px 8px; margin: 0; font-size: 0.85rem; }
    .inline-justification-card dt { color: var(--muted); }
    .inline-justification-card dd { margin: 0; }
    .inline-justification-card p { margin: 8px 0 0; line-height: 1.45; white-space: normal; }
    .policy-citation-row { display: flex; flex-wrap: wrap; gap: 6px; margin: 0 0 8px; }
    .policy-citation-chip { border: 1px solid rgba(37, 99, 235, 0.28); border-radius: 999px; background: rgba(219, 234, 254, 0.78); color: #1e3a8a; cursor: pointer; font: 600 0.75rem ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; padding: 3px 7px; }
    .policy-quotes { display: grid; gap: 6px; margin-top: 8px; }
    .policy-quotes blockquote { margin: 0; border-left: 3px solid rgba(59, 130, 246, 0.45); padding-left: 9px; color: var(--muted); font-style: italic; }
  `;
  document.head.appendChild(style);
}

function renderBorderline() {
  const empty = $('#borderlineEmpty');
  const target = $('#borderlineGroups');
  if (!target) return;
  const data = runState.borderline;
  const groups = normalizedBorderlineGroups(data?.groups).filter(group => (group.items || []).length > 0);
  if (!data || groups.length === 0) {
    showComputeEmpty(empty, 'borderline', 'No borderline data yet for this run.');
    target.innerHTML = '';
    return;
  }
  if (empty) empty.hidden = true;
  target.innerHTML = groups.map(group => {
    const items = (group.items || []).slice(0, 12);
    const body = items.map(item => {
      const id = item.image_id || item.sample_id || '';
      const reason = item.reason || item.borderline_reason || (Array.isArray(item.reasons) ? item.reasons.join(' · ') : '');
      const thumbSrc = thumbnailSrcForPath(item.repo_rel_path || '');
      const thumb = thumbSrc ? `<img class="row-thumb thumb-loading" src="${attr(thumbSrc)}" alt="${attr(id)}" loading="lazy" decoding="async" onload="this.classList.remove('thumb-loading')" onerror="this.replaceWith(safeImageFallback('image unavailable','local path missing'))" />` : '';
      const votes = votesForInline(item);
      const confs = votes.map(v => Number(v.confidence)).filter(Number.isFinite);
      const conf = confs.length ? `avg confidence ${(confs.reduce((a, b) => a + b, 0) / confs.length).toFixed(2)}` : '';
      const diffs = [...new Set(votes.map(v => v.difficulty).filter(Boolean))];
      const diff = diffs.length ? `difficulty ${diffs.join(', ')}` : '';
      const primary = `<tr data-image-id="${attr(id)}"><td>${expandButton('borderline', id)}</td><td><div class="thumb-wrap">${thumb}<div><button type="button" class="image-id-button" data-open-justifications="${attr(id)}"><strong>${esc(id)}</strong></button>${preparedMetaLine(item.prepared_image)}</div></div></td><td>${esc(reason || '—')}</td><td><span class="row-meta">${esc([conf, diff].filter(Boolean).join(' · ') || '—')}</span></td></tr>`;
      return primary + renderInlineJustificationsRow('borderline', item, 4);
    }).join('');
    const heading = group.l0 || group.label || group.bucket || 'unbucketed';
    return `<article class="borderline-group"><header><span class="badge ${labelBadgeClass(heading)}">${esc(heading)}</span><strong>${(group.items || []).length} case(s)</strong></header><div class="misalignment-table"><table class="misalignment"><thead><tr><th></th><th>image</th><th>reason</th><th>model notes</th></tr></thead><tbody>${body}</tbody></table></div></article>`;
  }).join('');
}

function renderMisalignment() {
  const empty = $('#misalignmentEmpty');
  const target = $('#misalignmentTable');
  if (!target) return;
  const data = runState.misalignment;
  const sourceRows = Array.isArray(data?.rows) ? data.rows : (Array.isArray(data?.records) ? data.records : []);
  if (!data || sourceRows.length === 0) {
    showComputeEmpty(empty, 'misalignment', 'No misalignment data yet for this run.');
    target.innerHTML = '';
    return;
  }
  if (empty) empty.hidden = true;
  const modelMap = new Map();
  for (const m of (Array.isArray(data.models) ? data.models : [])) modelMap.set(String(m), { labeler_id: String(m), model_id: String(m) });
  for (const row of sourceRows) {
    for (const vote of (row.votes || [])) {
      const key = vote.labeler_id || vote.model_id || 'unknown';
      if (!modelMap.has(key)) modelMap.set(key, vote);
    }
    for (const key of Object.keys(row.model_labels || {})) {
      if (!modelMap.has(key)) modelMap.set(key, { labeler_id: key, model_id: key });
    }
  }
  const modelRows = window.rushSortEnsembleLast(Array.from(modelMap.values()), (a, b) =>
    String(a.model_id || a.labeler_id || '').localeCompare(String(b.model_id || b.labeler_id || ''))
  );
  const modelId = model => model.labeler_id || model.model_id || 'unknown';
  const ensembleSuffix = model => window.rushIsEnsembleRow(model) ? ' <small class="muted">· ensemble</small>' : '';
  const formatCost = cost => {
    if (cost == null) return '';
    const value = Number(cost);
    return Number.isFinite(value) ? `$${value.toFixed(4)}` : String(cost);
  };
  const headerCells = ['<th></th>', '<th>image</th>', '<th>SME truth</th>', '<th>majority</th>',
    ...modelRows.map(model => `<th>${esc(modelId(model))}${ensembleSuffix(model)}</th>`),
    '<th>agreement</th>', '<th>reason</th>', '<th>patch</th>'].join('');
  const rows = sourceRows.slice(0, 100).map(row => {
    const id = row.image_id || row.sample_id || '';
    const repoRelPath = row.repo_rel_path || '';
    const thumbSrc = thumbnailSrcForPath(repoRelPath);
    const thumb = thumbSrc ? `<img class="row-thumb thumb-loading" src="${attr(thumbSrc)}" alt="${attr(id)}" loading="lazy" decoding="async" onload="this.classList.remove('thumb-loading')" onerror="this.replaceWith(safeImageFallback('image unavailable','local path missing'))" />` : '';
    const sme = row.sme_truth || row.truth || '—';
    const voteById = {};
    for (const vote of (row.votes || [])) voteById[vote.labeler_id || vote.model_id || 'unknown'] = vote;
    const perModel = modelRows.map(model => {
      const key = modelId(model);
      const voteRow = voteById[key];
      const vote = voteRow?.label || (row.model_labels && row.model_labels[key]) || '—';
      const cls = labelBadgeClass(vote);
      const cost = formatCost(voteRow?.cost_usd);
      const title = cost ? ` title="${attr(cost)}"` : '';
      return `<td${title}><span class="badge ${cls}">${esc(vote)}</span></td>`;
    }).join('');
    const agreement = row.agreement || (row.unanimous ? 'unanimous' : (row.misalignment_type || 'split'));
    const reason = row.disagreement_reason || row.reason || '';
    const patch = row.policy_patch_id
      ? `<a href="${attr(row.policy_patch_url || '#')}">${esc(row.policy_patch_id)}</a>`
      : '<span class="muted">—</span>';
    const primary = `<tr data-image-id="${attr(id)}"><td>${expandButton('misalignment', id)}</td><td><div class="thumb-wrap">${thumb}<div><button type="button" class="image-id-button" data-open-justifications="${attr(id)}"><strong>${esc(id)}</strong></button>${preparedMetaLine(row.prepared_image)}</div></div></td><td><span class="badge ${labelBadgeClass(sme)}">${esc(sme)}</span></td><td>${majorityPill(row)}</td>${perModel}<td>${esc(agreement)}</td><td>${esc(reason)}</td><td>${patch}</td></tr>`;
    return primary + renderInlineJustificationsRow('misalignment', row, modelRows.length + 7);
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
    showComputeEmpty(emptyEl, 'consensus', 'No consensus data yet for this run.');
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

  // Determine column order of voters across the dataset (stable, sorted; synthetic ensemble last).
  const voterMap = new Map();
  for (const r of data.records) {
    for (const v of (r.voters || [])) {
      const key = v.labeler_id || v.model_id || 'unknown';
      if (!voterMap.has(key)) voterMap.set(key, v);
    }
  }
  const voterColumns = window.rushSortEnsembleLast(Array.from(voterMap.values()), (a, b) =>
    String(a.model_id || a.labeler_id || '').localeCompare(String(b.model_id || b.labeler_id || ''))
  );
  const voterId = voter => voter.labeler_id || voter.model_id || 'unknown';
  const ensembleSuffix = voter => window.rushIsEnsembleRow(voter) ? ' <small class="muted">· ensemble</small>' : '';

  const headerCells = [
    '<th></th>',
    '<th>image</th>',
    '<th>flip rate</th>',
    '<th>SME truth</th>',
    ...voterColumns.map(v => `<th>${esc(voterId(v))}${ensembleSuffix(v)}</th>`),
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
    const sme = r.sme_truth || smeMap[r.image_id];
    const smeBadge = sme
      ? `<span class="badge ${labelBadgeClass(sme)}">${esc(sme)}</span>`
      : '<span class="muted">—</span>';
    const voterById = {};
    for (const v of (r.voters || [])) voterById[v.labeler_id || v.model_id || 'unknown'] = v;
    const perModel = voterColumns.map(voter => {
      const v = voterById[voterId(voter)];
      if (!v) return '<td><span class="muted">—</span></td>';
      const cls = labelBadgeClass(v.label);
      const boundary = v.is_boundary ? ' · boundary' : '';
      const conf = isNumber(v.confidence) ? ` (${v.confidence.toFixed(2)})` : '';
      const rawCost = Number(v.cost_usd);
      const cost = v.cost_usd != null && Number.isFinite(rawCost) ? ` · cost $${rawCost.toFixed(4)}` : '';
      return `<td><span class="badge ${cls}" title="confidence${conf}${boundary}${cost}">${esc(v.label)}</span></td>`;
    }).join('');
    const distChips = Object.entries(r.vote_distribution || {})
      .sort((a, b) => b[1] - a[1])
      .map(([lbl, cnt]) => `<span class="dist-chip ${labelBadgeClass(lbl)}">${esc(lbl)}: ${cnt}</span>`)
      .join(' ');
    const mismatch = sme && r.majority_label && r.majority_label !== sme;
    const rowCls = mismatch ? ' class="row-mismatch"' : '';
    const flipBadge = renderFlipBadgeForImage(r.image_id) || '<span class="muted">—</span>';
    const thumbSrc = thumbnailSrcForPath(r.repo_rel_path || '');
    const thumb = thumbSrc ? `<img class="row-thumb thumb-loading" src="${attr(thumbSrc)}" alt="${attr(r.image_id)}" loading="lazy" decoding="async" onload="this.classList.remove('thumb-loading')" onerror="this.replaceWith(safeImageFallback('image unavailable','local path missing'))" />` : '';
    const primary = `<tr data-image-id="${attr(r.image_id)}"${rowCls}><td>${expandButton('consensus', r.image_id)}</td><td><div class="thumb-wrap">${thumb}<div><button type="button" class="image-id-button" data-open-justifications="${attr(r.image_id)}"><strong>${esc(r.image_id)}</strong></button>${mismatch ? '<p class="row-meta mismatch-note">majority ≠ SME</p>' : ''}</div></div></td><td>${flipBadge}</td><td>${smeBadge}</td>${perModel}<td>${chipFor(r)}</td><td>${distChips || '<span class="muted">—</span>'}</td></tr>`;
    return primary + renderInlineJustificationsRow('consensus', r, voterColumns.length + 6);
  }).join('');
  tableTarget.innerHTML = `<table class="misalignment"><thead><tr>${headerCells}</tr></thead><tbody>${rows}</tbody></table>`;
}

function renderRun() {
  renderRunPicker();
  renderRunSummary();
  renderBorderline();
  renderMisalignment();
  renderConsensus();
  renderFlipRate();
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
      status.textContent = 'No runs found yet. Use the Run panel above to start one.';
    } else {
      status.textContent = `Found ${runState.available.length} run(s).`;
    }
  }
  renderRunPicker();
  if (autoSelectMostRecent && runState.available.length) {
    const preferred = runState.available.find(run => run.scoring_done !== false) || runState.available[0];
    await loadRun(preferred.run_id);
  } else {
    renderRun();
  }
}

function bindControls() {
  $('#runSampler')?.addEventListener('click', runSamplerDemo);
  document.querySelectorAll('[data-gallery-filter]').forEach(button => {
    button.addEventListener('click', () => {
      demoState.galleryFilter = button.dataset.galleryFilter || 'all';
      demoState.galleryVisibleCount = 24;
      renderGallery();
    });
  });
  $('#galleryLoadMore')?.addEventListener('click', () => {
    demoState.galleryVisibleCount = Math.max(24, (demoState.galleryVisibleCount || 24) * 2);
    renderGallery();
  });
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
  });
  document.body.addEventListener('change', event => {
    const target = event.target;
    const sampleId = target?.dataset?.sampleId;
    if (!sampleId || !target.classList.contains('override-label')) return;
    demoState.overrides[sampleId] = { ...overrideFor(sampleId), label: target.value };
    renderGallery();
  });
}

function bindRunControls() {
  $('#runPicker')?.addEventListener('change', event => loadRun(event.target.value));
  $('#refreshRuns')?.addEventListener('click', () => refreshRuns(true));
  $('#consensusFilter')?.addEventListener('change', event => {
    runState.consensusFilter = event.target.value || 'all';
    renderConsensus();
  });
  document.addEventListener('click', async event => {
    const target = event.target;
    if (!(target instanceof Element)) return;
    const expand = target.closest('[data-expand-row]');
    if (expand) {
      const panel = expand.dataset.expandRow || '';
      const imageId = expand.dataset.imageId || '';
      const key = expandedKey(panel, imageId);
      runState.expandedRows[key] = !runState.expandedRows[key];
      renderRun();
      return;
    }
    const policyNode = target.closest('[data-policy-node-id]');
    if (policyNode) {
      const opened = typeof window.rushOpenPolicyNode === 'function' && window.rushOpenPolicyNode(policyNode.dataset.policyNodeId || '');
      if (!opened) $('#policyGraphStatus') && ($('#policyGraphStatus').textContent = `Policy node ${policyNode.dataset.policyNodeId || ''} not loaded.`);
      return;
    }
    const computeButton = target.closest('[data-compute-target]');
    if (!computeButton) return;
    const runId = computeButton.dataset.runId || $('#runPicker')?.value || runState.selectedRunId || '';
    if (!runId) return;
    const status = $('#runStatus');
    computeButton.disabled = true;
    if (status) status.textContent = `Computing scoring exports for ${runId}…`;
    try {
      await rushApiPostJson(`/api/runs/${encodeURIComponent(runId)}/compute-now`, {});
      await loadRun(runId);
      await loadFlipRate();
      if (status) status.textContent = `Computed and refreshed ${runId}.`;
    } catch (error) {
      if (status) {
        status.classList.add('error');
        status.textContent = `Compute failed: ${error.message}`;
      }
    } finally {
      computeButton.disabled = false;
    }
  });
}

function initActiveNav() {
  const links = Array.from(document.querySelectorAll('.nav-pills a[href^="#"]'));
  if (!links.length || !('IntersectionObserver' in window)) return;
  const byId = new Map(links.map(link => [link.getAttribute('href').slice(1), link]));
  const observer = new IntersectionObserver(entries => {
    const visible = entries.filter(entry => entry.isIntersecting).sort((a, b) => b.intersectionRatio - a.intersectionRatio)[0];
    if (!visible) return;
    links.forEach(link => link.classList.remove('active'));
    byId.get(visible.target.id)?.classList.add('active');
  }, { rootMargin: '-18% 0px -65% 0px', threshold: [0.01, 0.25, 0.5] });
  byId.forEach((link, id) => {
    const section = document.getElementById(id);
    if (section) observer.observe(section);
  });
}

function init() {
  $('#policyNodeList').innerHTML = '';
  initInlineJustificationStyles();
  bindControls();
  bindRunControls();
  initActiveNav();
  initApi();
  runSamplerDemo();
  loadFlipRate();
  refreshRuns(true);
}

init();

// ---------- Local Web API bootstrap (X5) ----------
function initApi() {
  if (window.RUSH_API?.ready) return window.RUSH_API.ready;
  const api = window.RUSH_API || {};
  window.RUSH_API = api;
  api.available = false;
  api.health = null;
  api.catalog = api.catalog || { runs: [], policyVersions: [], currentPolicyVersion: '' };
  api.getJson = rushApiGetJson;
  api.postJson = rushApiPostJson;
  window.rushApiGetJson = rushApiGetJson;
  window.rushApiPostJson = rushApiPostJson;
  api.ready = fetch('/api/health', { cache: 'no-store' })
    .then(async response => {
      if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
      api.health = await response.json();
      api.available = true;
      rushApiApplyAvailability(true);
      return api.health;
    })
    .catch(error => {
      api.health = null;
      api.available = false;
      api.error = error;
      rushApiApplyAvailability(false);
      return null;
    });
  return api.ready;
}

function rushApiApplyAvailability(available) {
  document.body.classList.toggle('rush-api-available', !!available);
  document.body.classList.toggle('rush-api-unavailable', !available);
  const hint = $('#apiUnavailableHint');
  if (hint) hint.hidden = !!available;
  if (!available) {
    document.querySelectorAll('.api-section').forEach(section => {
      if (section.id === 'run-trigger') section.hidden = true;
      else rushApiUnavailable(section);
    });
  } else {
    document.querySelectorAll('.api-section').forEach(section => { section.hidden = false; });
  }
  window.dispatchEvent(new CustomEvent('rush-api-ready', { detail: { available: !!available, health: window.RUSH_API?.health || null } }));
}

function rushApiUnavailable(sectionOrId, message = 'Local API offline — start the rush web server to enable this view.') {
  const section = typeof sectionOrId === 'string' ? $(sectionOrId) : sectionOrId;
  if (!section) return;
  const existing = section.querySelector('.api-placeholder');
  const html = `<div class="api-placeholder empty-state">${esc(message)}</div>`;
  if (existing) existing.outerHTML = html;
  else section.insertAdjacentHTML('beforeend', html);
}

function rushApiOnReady(callback) {
  const api = window.RUSH_API || (window.RUSH_API = { available: false });
  if (api.ready) {
    api.ready.then(() => callback(api));
  } else {
    window.addEventListener('rush-api-ready', () => callback(api), { once: true });
  }
}

async function rushApiGetJson(path) {
  const response = await fetch(path, { cache: 'no-store' });
  return rushApiParseJson(response);
}

async function rushApiPostJson(path, body = {}) {
  const response = await fetch(path, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body)
  });
  return rushApiParseJson(response);
}

async function rushApiParseJson(response) {
  let payload = null;
  try { payload = await response.json(); }
  catch (error) { payload = null; }
  if (!response.ok) {
    const errorText = typeof payload?.error === 'string' ? payload.error : payload?.error?.message;
    const message = errorText || payload?.message || `${response.status} ${response.statusText}`;
    throw new Error(message);
  }
  return payload;
}

function rushApiStatus(target, message, isError = false) {
  const el = typeof target === 'string' ? $(target) : target;
  if (!el) return;
  el.classList.toggle('error', !!isError);
  el.textContent = message || '';
}

function rushApiOptionHtml(value, label = value, selected = false) {
  return `<option value="${attr(value)}"${selected ? ' selected' : ''}>${esc(label)}</option>`;
}

async function rushApiLoadCatalog() {
  if (!window.RUSH_API?.available) return window.RUSH_API?.catalog || { runs: [], policyVersions: [], currentPolicyVersion: '' };
  const [runsPayload, versionsPayload] = await Promise.all([
    rushApiGetJson('/api/runs').catch(() => ({ runs: [] })),
    rushApiGetJson('/api/policy/versions').catch(() => ({ versions: [], current: '' }))
  ]);
  const versions = Array.isArray(versionsPayload.versions) ? versionsPayload.versions : [];
  window.RUSH_API.catalog = {
    runs: Array.isArray(runsPayload.runs) ? runsPayload.runs : [],
    policyVersions: versions,
    currentPolicyVersion: versionsPayload.current || versions[0]?.version || ''
  };
  window.dispatchEvent(new CustomEvent('rush-api-catalog', { detail: window.RUSH_API.catalog }));
  return window.RUSH_API.catalog;
}

function rushApiRunOptions(selected = '', includeAll = false, allLabel = 'All scored runs') {
  const runs = window.RUSH_API?.catalog?.runs || [];
  const prefix = includeAll ? rushApiOptionHtml('', allLabel, !selected) : '';
  if (!runs.length) return prefix || rushApiOptionHtml('', 'No runs found', true);
  return prefix + runs.map(run => {
    const label = [run.run_id, run.started_at].filter(Boolean).join(' · ');
    return rushApiOptionHtml(run.run_id || '', label || run.run_id || '', selected === run.run_id);
  }).join('');
}

function rushApiPolicyVersionOptions(selected = '', includeAll = false, allLabel = 'All policy versions') {
  const versions = window.RUSH_API?.catalog?.policyVersions || [];
  const prefix = includeAll ? rushApiOptionHtml('', allLabel, !selected) : '';
  if (!versions.length) return prefix || rushApiOptionHtml('', 'No policy versions found', true);
  return prefix + versions.map(item => {
    const version = item.version || item;
    return rushApiOptionHtml(version, version, selected === version);
  }).join('');
}

function rushApiFormatMetric(value, digits = 1) {
  return isNumber(value) ? `${(value * 100).toFixed(digits)}%` : '—';
}

function rushApiShort(value, fallback = '—') {
  const text = String(value ?? '').trim();
  return text || fallback;
}
