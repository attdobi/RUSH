const $ = selector => document.querySelector(selector);
const esc = value => String(value ?? '').replace(/[&<>'"]/g, char => ({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}[char]));
const attr = esc;
const isNumber = value => typeof value === 'number' && Number.isFinite(value);
const BUILD_ID = (document.querySelector('meta[name="build-id"]')?.content) || '';
function cacheBust(url) {
  if (!BUILD_ID) return url;
  if (/^https?:\/\//i.test(url) || url.startsWith('//')) return url;
  const sep = url.includes('?') ? '&' : '?';
  return url + sep + 'v=' + encodeURIComponent(BUILD_ID);
}
window.cacheBust = cacheBust;

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

// Active demo config comes from web/demos.js. Everything demo-specific
// (manifest URLs, label space, thumbnails, hero copy, policy graph) reads
// through activeDemo() so genai and mnist share one code path.
function activeDemo() {
  if (typeof window.rushActiveDemo === 'function') return window.rushActiveDemo();
  // Defensive fallback if demos.js failed to load: keep the historical GenAI config.
  return {
    id: 'genai', kind: 'binary', classes: ['gen_ai', 'not_gen_ai'], positiveClass: 'gen_ai',
    manifests: {
      dev: '../data/images/genai-classification/manifests/dev_golden_labels.csv',
      holdout: '../data/images/genai-classification/manifests/holdout_labels.csv',
      summary: '../data/images/genai-classification/manifests/sampling_summary.json'
    },
    thumbnailsDir: 'data/images/genai-classification/thumbnails',
    heroCopy: { eyebrow: '', h1: '', cta: 'Start demo' }
  };
}
function activeDemoIsMnist() { return activeDemo().id === 'mnist'; }
function activeManifests() { return activeDemo().manifests; }
function activeDemoId() { return activeDemo().id || 'genai'; }
function activePolicyGraphArea() { return activeDemo().policyGraph?.area || 'Generative_AI'; }


const demoState = {
  result: null,
  source: 'loading',
  previewPerClass: 4
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
  const response = await fetch(cacheBust(path), { cache: 'no-store' });
  if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
  return response.text();
}

async function loadLocalManifests() {
  const manifests = activeManifests();
  const [devText, holdoutText, summaryText] = await Promise.all([
    fetchText(manifests.dev),
    fetchText(manifests.holdout),
    fetchText(manifests.summary)
  ]);
  return {
    devGolden: parseCsv(devText).map(row => normalizeManifestRow(row, 'dev_golden')),
    holdout: parseCsv(holdoutText).map(row => normalizeManifestRow(row, 'holdout')),
    manifestSummary: JSON.parse(summaryText)
  };
}

function normalizeManifestRow(row, sampleSet) {
  const repoPath = row.repo_rel_path || row.synthetic_repo_rel_path || '';
  return {
    ...row,
    sample_id: row.sample_id,
    split: row.split || sampleSet,
    sample_set: sampleSet,
    label_int: Number.parseInt(row.label_int, 10),
    seed: Number.parseInt(row.seed, 10) || 20260510,
    sha256: row.sha256 || '',
    repo_rel_path: repoPath,
    synthetic_repo_rel_path: repoPath,
    truth_tier: row.truth_tier || 'gold_candidate',
    policy_use: row.policy_use || (sampleSet === 'holdout' ? 'locked_holdout_decision_quality' : 'develop_policy'),
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
  // The warm-start / seed-mismatch guards below only add value when a
  // browser-side synthetic fallback is available (GenAI only). For demos
  // without a synthetic sampler (e.g. mnist) throwing here just surfaces a
  // confusing "Sampler failed" instead of showing the real local manifest
  // data, so we skip the guards for those demos.
  const hasSyntheticFallback = activeDemo().id === 'genai' && !!window.RushGenaiSampler?.runDemoReset;
  try {
    const local = await loadLocalManifests();
    const manifestSeed = Number.parseInt(local.manifestSummary?.seed, 10);
    if (hasSyntheticFallback && options.mode !== 'cold_start') throw new Error('Local manifests are cold-start only; using synthetic fallback for warm-start preview.');
    if (hasSyntheticFallback && manifestSeed && options.seed !== manifestSeed) throw new Error('Requested seed differs from local manifest seed; using synthetic fallback so sampling changes visibly.');
    const devGolden = take(local.devGolden, options.nDev);
    const holdout = take(local.holdout, options.nHoldout);
    return {
      devGolden,
      holdout,
      combined: [...devGolden, ...holdout],
      previewRecords: [...local.devGolden, ...local.holdout],
      summary: buildSummary(devGolden, holdout, options, 'local manifests + real image files', local.manifestSummary),
      leakageChecks: { ok: true, devHoldoutDisjoint: true, note: `Local manifests were generated by ${local.manifestSummary?.sampling_version || 'scripts/sample_genai_gold_sets.py'} with path/hash leakage checks.` }
    };
  } catch (error) {
    // The client-side synthetic sampler only knows the GenAI label space.
    // For other demos (e.g. mnist) surface the real error instead of
    // fabricating gen_ai/not_gen_ai rows.
    if (activeDemo().id !== 'genai' || !window.RushGenaiSampler?.runDemoReset) throw error;
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
  const label = String(row?.label || '').trim();
  if (label === 'ai_generated') return 'ai-generated';
  if (label === 'not_ai_generated') return 'not-ai';
  // Multiclass demos (mnist): render one badge class per digit / class.
  const classes = activeDemo().classes || [];
  if (classes.includes(label)) return `digit-${label.replace(/[^a-z0-9_-]/gi, '')}`;
  return 'dev';
}

function thumbnailSrcForPath(repoRelPath) {
  const path = String(repoRelPath || '').replace(/^\.\//, '').replace(/^\/+/, '');
  if (!path) return '';
  if (window.RUSH_API?.available) return `/api/thumbnail?path=${encodeURIComponent(path)}`;
  return `../${path}`;
}

// Deterministic synthetic SVG thumb for browser-only demo (no real image bytes available).
function syntheticThumbDataUri(row) {
  const label = String(row?.label || 'unknown');
  const dataset = String(row?.dataset || 'demo');
  const sampleId = String(row?.sample_id || row?.synthetic_repo_rel_path || dataset);
  const hash = String(row?.sha256 || '').slice(0, 8) || sampleId.slice(-8);
  // Deterministic hue from sha256/sample_id
  let h = 0;
  for (let i = 0; i < hash.length; i += 1) h = (h * 31 + hash.charCodeAt(i)) >>> 0;
  const hue = h % 360;
  const isAi = label === 'ai_generated';
  const accent = isAi ? `hsl(${(hue + 320) % 360}, 70%, 56%)` : `hsl(${(hue + 150) % 360}, 60%, 52%)`;
  const bgA = `hsl(${hue}, 38%, 18%)`;
  const bgB = `hsl(${(hue + 40) % 360}, 42%, 28%)`;
  const labelText = isAi ? 'AI' : 'REAL';
  const idShort = sampleId.length > 14 ? `${sampleId.slice(0, 12)}…` : sampleId;
  const datasetShort = dataset.length > 10 ? `${dataset.slice(0, 9)}…` : dataset;
  const svg = `<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 200 200'>` +
    `<defs><linearGradient id='g' x1='0' y1='0' x2='1' y2='1'>` +
      `<stop offset='0' stop-color='${bgA}'/><stop offset='1' stop-color='${bgB}'/>` +
    `</linearGradient></defs>` +
    `<rect width='200' height='200' rx='14' fill='url(#g)'/>` +
    `<circle cx='150' cy='52' r='30' fill='${accent}' opacity='0.85'/>` +
    `<rect x='14' y='14' width='${datasetShort.length * 8 + 14}' height='22' rx='11' fill='rgba(0,0,0,0.45)'/>` +
    `<text x='${14 + 7}' y='30' font-family='Inter,system-ui,sans-serif' font-size='12' font-weight='700' fill='#dce8ff'>${esc(datasetShort)}</text>` +
    `<text x='100' y='118' text-anchor='middle' font-family='Inter,system-ui,sans-serif' font-size='44' font-weight='900' fill='#fff' opacity='0.92'>${labelText}</text>` +
    `<text x='100' y='154' text-anchor='middle' font-family='Inter,system-ui,sans-serif' font-size='11' font-weight='600' fill='#dce8ff' opacity='0.8'>${esc(idShort)}</text>` +
    `<text x='100' y='180' text-anchor='middle' font-family='Inter,system-ui,sans-serif' font-size='9' font-weight='500' fill='#aab8d3' opacity='0.65'>demo placeholder · no image bytes</text>` +
  `</svg>`;
  return `data:image/svg+xml;charset=utf-8,${encodeURIComponent(svg)}`;
}

function thumbnailSrcForImageId(imageId) {
  if (!imageId) return null;
  const dir = activeDemo().thumbnailsDir || 'data/images/genai-classification/thumbnails';
  const path = `${dir}/${imageId}.jpg`;
  return `/api/thumbnail?path=${encodeURIComponent(path)}`;
}

window.thumbnailSrcForPath = thumbnailSrcForPath;
window.thumbnailSrcForImageId = thumbnailSrcForImageId;
window.syntheticThumbDataUri = syntheticThumbDataUri;

function imgSrc(row) {
  if (row && row.is_synthetic_demo_candidate === true) return syntheticThumbDataUri(row);
  return thumbnailSrcForPath(row?.repo_rel_path || row?.synthetic_repo_rel_path);
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

window.safeImageFallback = safeImageFallback;

function renderThumb(row) {
  const src = imgSrc(row);
  const datasetLabel = row.dataset || activeDemo().title || 'demo';
  const fallback = `<div class="thumb-fallback"><strong>${esc(datasetLabel)}</strong><span>${esc(row.label)}</span></div>`;
  // MNIST source PNGs are 28x28 — keep them crisp when upscaled in the grid.
  const thumbClass = activeDemoIsMnist() ? 'thumb-loading mnist-thumb' : 'thumb-loading';
  if (src) {
    return `<img class="${thumbClass}" src="${attr(src)}" alt="${attr(row.sample_id)} ${attr(row.label)}" loading="lazy" decoding="async" onload="this.classList.remove('thumb-loading')" onerror="this.replaceWith(safeImageFallback())" />`;
  }
  return fallback;
}

function displaySplitLabel(row) {
  const raw = String(row?.split || row?.sample_set || '').trim();
  const normalized = raw.toLowerCase();
  if (normalized === 'val' || normalized === 'validation') return 'test';
  if (normalized === 'dev_golden') return 'dev';
  return raw || 'unknown';
}

function renderSampleCard(row, compact = false) {
  const demo = activeDemo();
  const locked = row.sample_set === 'holdout' || row.split === 'holdout';
  const policyNodeId = demo.classNodeId ? demo.classNodeId(row.label) : '';
  const policyChip = policyNodeId
    ? `<button type="button" class="policy-citation-chip" data-policy-node-id="${attr(policyNodeId)}" title="Open policy node ${attr(policyNodeId)}">${esc(policyNodeId)}</button>`
    : '';
  const dirLabel = row.source_label_dir || row.dataset || 'source';
  return `<article class="sample-card ${locked ? 'locked' : ''} ${compact ? 'compact' : ''}" data-sample-id="${attr(row.sample_id)}" title="${attr(row.sample_id)}">
    <div class="sample-image">${renderThumb(row)}</div>
    <div class="sample-meta">
      <div class="sample-badges">
        <span class="badge ${labelBadge(row)}">${esc(row.label)}</span>
        <span class="badge ${locked ? 'holdout' : 'dev'}">${esc(displaySplitLabel(row))}</span>
        ${policyChip}
      </div>
      <details class="sample-details">
        <summary>Details</summary>
        <p><strong>Human label:</strong> ${esc(row.label)}<br><strong>Split:</strong> ${esc(displaySplitLabel(row))}<br><strong>Source:</strong> ${esc(dirLabel)}</p>
      </details>
    </div>
  </article>`;
}

function gcd(a, b) {
  let x = Math.abs(Number(a) || 0);
  let y = Math.abs(Number(b) || 0);
  while (y) [x, y] = [y, x % y];
  return x || 1;
}

function classRatio(summary) {
  const counts = Object.values(summary.byClass || {}).map(value => Number(value) || 0).filter(value => value > 0);
  if (!counts.length) return 'n/a';
  if (counts.length !== 2) return counts.join(':');
  const divisor = gcd(counts[0], counts[1]);
  return `${counts[0] / divisor}:${counts[1] / divisor}`;
}

function renderSummaryStrip() {
  const strip = $('#demoSummaryStrip');
  if (!strip) return;
  const summary = demoState.result?.summary;
  if (!summary) {
    strip.hidden = true;
    strip.textContent = '';
    return;
  }
  const total = summary.total ?? ((summary.n_dev_golden || 0) + (summary.n_holdout || 0));
  const dev = summary.n_dev_golden ?? summary.bySplit?.dev_golden ?? 0;
  const holdout = summary.n_holdout ?? summary.bySplit?.holdout ?? 0;
  const leakage = demoState.result?.leakageChecks?.ok ? '✓' : 'review';
  strip.textContent = `${total} records · ${dev} dev / ${holdout} holdout · class ${classRatio(summary)} · seed ${summary.seed || readSamplerOptions().seed} · leakage ${leakage}`;
  strip.hidden = false;
}

function renderStats() {
  renderSummaryStrip();
  const summary = demoState.result?.summary;
  if (!summary) {
    const stats = $('#demoStats');
    if (stats) stats.innerHTML = '';
    return;
  }
  const classText = Object.entries(summary.byClass || {}).map(([k, v]) => `${k}: ${v}`).join(' · ');
  const sourceText = Object.entries(summary.byDataset || {}).map(([k, v]) => `${k}: ${v}`).join(' · ');
  $('#demoStats').innerHTML = [
    ['Sampled records', summary.total ?? (summary.n_dev_golden + summary.n_holdout), 'default N = 100 per split'],
    ['Truth tier', 'gold candidates', 'SME quality assumed for demo'],
    ['Label', 'LLM labeling comes next', 'This preview makes no model calls; bulk LLM labeling comes next, then SME/human override feeds the policy graph.'],
    ['Source', summary.source || demoState.source, summary.samplingVersion ? `manifest ${summary.samplingVersion}` : 'real manifests if available'],
    ['Leakage check', demoState.result.leakageChecks?.ok ? 'pass' : 'review', 'dev/holdout path + hash separation']
  ].map(([label, value, note]) => `<article class="stat-card"><span>${esc(label)}</span><strong>${esc(value)}</strong><p>${esc(note)}</p></article>`).join('') +
  `<div class="wide-note"><strong>Class balance:</strong> ${esc(classText || 'n/a')}<br><strong>Dataset balance:</strong> ${esc(sourceText || 'n/a')}</div>`;
}

function hashString(value) {
  let hash = 2166136261;
  const text = String(value || '');
  for (let index = 0; index < text.length; index += 1) {
    hash ^= text.charCodeAt(index);
    hash = Math.imul(hash, 16777619);
  }
  return hash >>> 0;
}

function seededRandom(seed) {
  let state = seed >>> 0;
  return () => {
    state += 0x6D2B79F5;
    let value = state;
    value = Math.imul(value ^ (value >>> 15), value | 1);
    value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
    return ((value ^ (value >>> 14)) >>> 0) / 4294967296;
  };
}

function seededShuffle(records, seedKey) {
  const shuffled = records.slice();
  const random = seededRandom(hashString(seedKey));
  for (let index = shuffled.length - 1; index > 0; index -= 1) {
    const swapIndex = Math.floor(random() * (index + 1));
    [shuffled[index], shuffled[swapIndex]] = [shuffled[swapIndex], shuffled[index]];
  }
  return shuffled;
}

function selectPreviewRows(rows, n, seedKey) {
  const bySplit = new Map();
  for (const row of rows) {
    const split = displaySplitLabel(row);
    if (!bySplit.has(split)) bySplit.set(split, []);
    bySplit.get(split).push(row);
  }
  const splitKeys = seededShuffle(Array.from(bySplit.keys()), `${seedKey}:splits`);
  const buckets = new Map(splitKeys.map(split => [split, seededShuffle(bySplit.get(split), `${seedKey}:${split}`)]));
  const selected = [];
  let offset = 0;
  while (selected.length < n && selected.length < rows.length) {
    let added = false;
    for (const split of splitKeys) {
      const row = buckets.get(split)?.[offset];
      if (!row) continue;
      selected.push(row);
      added = true;
      if (selected.length >= n) break;
    }
    if (!added) break;
    offset += 1;
  }
  return selected;
}

function readPreviewPerClass() {
  const input = $('#samplePreviewPerClass');
  const value = Number.parseInt(input?.value, 10);
  return Math.min(10, Math.max(1, Number.isFinite(value) ? value : 4));
}

function previewRecords() {
  const result = demoState.result;
  if (!result) return [];
  return result.previewRecords || result.combined || [];
}

function previewGroups() {
  const demo = activeDemo();
  const classes = Array.isArray(demo.classes) && demo.classes.length ? demo.classes : [];
  const buckets = new Map(classes.map(cls => [String(cls), []]));
  for (const row of previewRecords()) {
    const label = String(row.label || '');
    if (!buckets.has(label)) buckets.set(label, []);
    buckets.get(label).push(row);
  }
  const seed = Number.parseInt($('#samplerSeed')?.value, 10) || demoState.result?.summary?.seed || 20260510;
  const n = readPreviewPerClass();
  return Array.from(buckets.entries()).map(([label, rows]) => ({
    label,
    total: rows.length,
    rows: selectPreviewRows(rows, n, `${activeDemoId()}:${seed}:${label}`)
  }));
}

function renderGallery() {
  const result = demoState.result;
  if (!result) return;
  const n = readPreviewPerClass();
  demoState.previewPerClass = n;
  const groups = previewGroups();
  const shown = groups.reduce((sum, group) => sum + group.rows.length, 0);
  const total = groups.reduce((sum, group) => sum + group.total, 0);
  const status = $('#galleryStatus');
  if (status) {
    status.textContent = `Data preview: ${shown} random example(s) across ${groups.length} human-label categories from ${total} loaded manifest row(s).`;
  }
  const html = groups.map(group => `
    <section class="preview-class-group" aria-label="Human label ${attr(group.label)}">
      <div class="preview-class-head">
        <h3>${esc(group.label)}</h3>
        <span>${group.rows.length}/${group.total} shown</span>
      </div>
      ${group.rows.length ? `<div class="sample-grid preview-class-grid">${group.rows.map(row => renderSampleCard(row, true)).join('')}</div>` : '<div class="empty-state">No loaded rows for this category.</div>'}
    </section>
  `).join('');
  $('#sampleGallery').innerHTML = html || '<div class="empty-state">No loaded manifest rows are available yet.</div>';
}


function renderAll() {
  renderStats();
  renderGallery();
  renderResidualMisalignments();
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
const MNIST_EMPTY_RUN_MESSAGE = 'No scored MNIST run yet — run labeling to populate.';

// Single source of truth for label → CSS class.
// IMPORTANT: replaceAll, not replace — `not_gen_ai` has TWO underscores.
function labelBadgeClass(label) {
  const value = String(label || '');
  if (!value) return 'dev';
  const classes = Array.isArray(activeDemo().classes) ? activeDemo().classes.map(String) : [];
  if (classes.includes(value)) {
    if (activeDemoIsMnist()) return `digit-${value.replace(/[^a-z0-9_-]/gi, '')}`;
    return value.replaceAll('_', '-');
  }
  if (KNOWN_LABELS.includes(value)) return value.replaceAll('_', '-');
  return 'dev';
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
  if (window.RUSH_API?.ready) await window.RUSH_API.ready.catch(() => null);
  if (window.RUSH_API?.available) {
    const payload = await rushApiGetJson(`/api/runs?demo=${encodeURIComponent(activeDemoId())}`)
      .catch(() => rushApiGetJson('/api/runs').catch(() => null));
    if (payload && Array.isArray(payload.runs)) {
      return filterRunsForActiveDemo(payload.runs).filter(r => r && r.run_id).map(r => ({
        ...r,
        run_id: r.run_id,
        label: r.label || r.run_id,
        started_at: r.started_at || null,
        scoring_done: r.scoring_done === false ? false : (r.scoring_done === true ? true : undefined)
      })).sort((a, b) => (b.started_at || '').localeCompare(a.started_at || ''));
    }
  }
  // Fallback: an index file produced by the runner / scoring exporter.
  const index = await fetchJsonOptional(RUNS_INDEX_URL);
  if (index && Array.isArray(index.runs)) {
    return filterRunsForActiveDemo(index.runs).filter(r => r && r.run_id).map(r => ({
      ...r,
      run_id: r.run_id,
      label: r.label || r.run_id,
      started_at: r.started_at || null
    })).sort((a, b) => (b.started_at || '').localeCompare(a.started_at || ''));
  }
  return [];
}

function runDemoTokens(run) {
  return [
    run?.demo,
    run?.demo_id,
    run?.track,
    run?.area,
    run?.policy_area,
    run?.policy_graph_area,
    run?.domain,
    run?.policy_domain,
    run?.policy_graph?.area
  ].filter(value => value != null).map(value => String(value).toLowerCase());
}

function runIdSuggestsDemo(run, demoId) {
  return String(run?.run_id || run?.label || '').toLowerCase().includes(demoId);
}

function runMatchesActiveDemo(run) {
  const demo = activeDemo();
  const demoId = String(demo.id || 'genai').toLowerCase();
  const area = String(demo.policyGraph?.area || '').toLowerCase();
  const tokens = runDemoTokens(run);
  const hasExplicitDemoToken = tokens.length > 0;
  const matchesActive = tokens.some(token => token === demoId || token === area) || runIdSuggestsDemo(run, demoId);
  if (demoId === 'mnist') return matchesActive;
  const isExplicitMnist = tokens.some(token => token === 'mnist' || token === 'mnist_digits') || runIdSuggestsDemo(run, 'mnist');
  return hasExplicitDemoToken ? matchesActive || !isExplicitMnist : !isExplicitMnist;
}

function filterRunsForActiveDemo(runs) {
  return (Array.isArray(runs) ? runs : []).filter(runMatchesActiveDemo);
}

async function loadRun(runId) {
  if (!runId) {
    runState.selectedRunId = null;
    runState.summary = null;
    runState.borderline = null;
    runState.misalignment = null;
    renderRun();
    window.dispatchEvent(new CustomEvent('rush-score-run-selected', { detail: { runId: '' } }));
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
  window.dispatchEvent(new CustomEvent('rush-score-run-selected', { detail: { runId } }));
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

function metricNumber(value) {
  return isNumber(value) ? value : (Number.isFinite(Number(value)) ? Number(value) : null);
}

function runMetaForSelected() {
  return runState.available.find(row => row.run_id === runState.selectedRunId) || {};
}

function runSummaryImageCount(summary) {
  const labelers = Array.isArray(summary?.labelers) ? summary.labelers : [];
  const nValues = labelers
    .filter(row => !window.rushIsEnsembleRow(row))
    .map(row => metricNumber(row?.metrics?.n))
    .filter(value => value !== null);
  if (nValues.length) return Math.max(...nValues);
  const consensus = summary?.consensus_summary || {};
  return consensus.n_images ?? consensus.image_count ?? consensus.total_images ?? '—';
}

function runSummaryCost(summary) {
  if (isNumber(summary?.cost?.total_cost_usd)) return summary.cost.total_cost_usd;
  const labelers = Array.isArray(summary?.labelers) ? summary.labelers : [];
  const modelCosts = labelers
    .filter(row => !window.rushIsEnsembleRow(row))
    .map(row => {
      const n = metricNumber(row?.metrics?.n);
      const per1k = metricNumber(row?.metrics?.cost_per_1000_labels);
      return n !== null && per1k !== null ? (n * per1k / 1000) : null;
    })
    .filter(value => value !== null);
  if (!modelCosts.length) return null;
  return modelCosts.reduce((total, value) => total + value, 0);
}

function formatRunTime(meta) {
  const started = meta.started_at ? new Date(meta.started_at).toLocaleString([], { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' }) : '—';
  const finished = meta.finished_at ? new Date(meta.finished_at).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }) : (meta.running ? 'running' : '—');
  return { started, finished };
}

function renderRunSummary() {
  const target = $('#runSummary');
  if (!target) return;
  if (!runState.summary) {
    target.innerHTML = '';
    return;
  }
  const s = runState.summary;
  const meta = runMetaForSelected();
  const labelers = Array.isArray(s.labelers) ? s.labelers : [];
  const modelRows = labelers.filter(row => !window.rushIsEnsembleRow(row));
  const modelNames = modelRows.map(row => row.labeler_id).filter(Boolean);
  const totals = meta.totals || {};
  const errored = metricNumber(totals.errored_calls) || 0;
  const completed = metricNumber(totals.completed_calls) ?? metricNumber(totals.successful_calls) ?? null;
  const expected = metricNumber(totals.expected_calls);
  const cost = runSummaryCost(s);
  const time = formatRunTime(meta);
  const cards = [
    ['Images', runSummaryImageCount(s), meta.split ? `split: ${meta.split}` : 'scored images'],
    ['Models', modelRows.length || '—', modelNames.length ? modelNames.join(' · ') : 'model breakdown unavailable'],
    ['Time', time.finished === 'running' ? 'Running' : time.finished, `started ${time.started}`],
    ['Cost', cost === null ? '—' : `$${cost.toFixed(cost >= 1 ? 2 : 4)}`, 'estimated from scored calls'],
    ['Success / errors', `${completed ?? '—'} / ${errored}`, expected ? `${expected} expected calls` : 'from run manifest']
  ];
  target.innerHTML = `
    <div class="run-summary-metrics">
      ${cards.map(([k, v, n]) => `<article class="stat-card"><span>${esc(k)}</span><strong>${esc(v)}</strong><p>${esc(n || '')}</p></article>`).join('')}
    </div>
    <details class="raw-json-details run-summary-raw">
      <summary>View raw summary JSON</summary>
      <pre class="log-tail raw-json">${esc(JSON.stringify(s, null, 2))}</pre>
    </details>`;
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
  if (activeDemoIsMnist()) {
    empty.hidden = false;
    empty.textContent = MNIST_EMPTY_RUN_MESSAGE;
    return;
  }
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

function boundaryPairChip(record) {
  const pair = record?.is_boundary === true && Array.isArray(record?.is_boundary_between) && record.is_boundary_between.length === 2
    ? record.is_boundary_between
    : null;
  if (!pair) return '';
  return `<span class="boundary-pair-chip" title="boundary pair">${esc(pair[0])} ↔ ${esc(pair[1])}</span>`;
}
window.rushBoundaryPairChip = boundaryPairChip;

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

const TRAIN_SPLIT_ALIASES = new Set(['dev_golden', 'train', 'training', 'development']);
const TEST_SPLIT_ALIASES = new Set(['holdout', 'val', 'validation', 'test', 'testing', 'locked_holdout', 'locked_holdout_decision_quality']);

function splitKind(value) {
  const normalized = String(value || '').trim().toLowerCase();
  if (TRAIN_SPLIT_ALIASES.has(normalized)) return 'train';
  if (TEST_SPLIT_ALIASES.has(normalized)) return 'test';
  return '';
}

function splitLabel(value) {
  const kind = splitKind(value);
  if (kind === 'train') return 'training';
  if (kind === 'test') return 'testing';
  return value ? String(value) : 'not exported';
}

function manifestSplitLookup() {
  const rows = Array.isArray(demoState.result?.combined) ? demoState.result.combined : [];
  const byId = new Map();
  for (const row of rows) {
    const split = row.split || row.policy_use || '';
    for (const key of [row.sample_id, row.image_id]) {
      if (key) byId.set(String(key), split);
    }
  }
  return byId;
}

function splitForScoredRow(row, lookup = manifestSplitLookup()) {
  const direct = row?.split || row?.dataset_split || row?.policy_use || '';
  if (direct) return direct;
  const id = row?.sample_id || row?.image_id || '';
  return id ? (lookup.get(String(id)) || '') : '';
}

function scoreKPerSplitValue() {
  const own = Number.parseInt($('#scoreKPerSplit')?.value, 10);
  if (Number.isInteger(own) && own > 0) return own;
  const run = Number.parseInt($('#runTriggerBatchSize')?.value, 10);
  return Number.isInteger(run) && run > 0 ? run : 20;
}

function scoreSplitValue() {
  return $('#scoreSplitMirror')?.value || $('#runTriggerSplit')?.value || 'all';
}

function scoreSplitPhrase(split) {
  if (split === 'all') return 'training + testing';
  if (splitKind(split) === 'train') return 'training only';
  if (splitKind(split) === 'test') return 'testing only';
  return split || 'all';
}

function decisionQualityLabelers(payload) {
  return Array.isArray(payload?.labelers) ? payload.labelers : [];
}

function decisionQualityMetric(row, key) {
  return metricNumber(row?.metrics?.[key] ?? row?.[key]);
}

function decisionQualityImageCount(payload) {
  const own = metricNumber(payload?.n_images);
  if (own !== null) return own;
  const nValues = decisionQualityLabelers(payload)
    .map(row => decisionQualityMetric(row, 'n'))
    .filter(value => value !== null);
  return nValues.length ? Math.max(...nValues) : '—';
}

function decisionQualityHeadlineLabeler(payload) {
  const labelers = decisionQualityLabelers(payload);
  return labelers.find(row => window.rushIsEnsembleRow(row))
    || labelers.find(row => String(row?.labeler_id || '').toLowerCase() === 'majority_vote')
    || labelers.find(row => row);
}

function decisionQualityLabelerName(row) {
  return row?.labeler_id || row?.model_id || 'majority_vote';
}

function renderScoreReportedMetrics() {
  const target = $('#scoreReportedMetrics');
  if (!target) return;
  const summary = runState.summary || {};
  if (!summary.reported || typeof summary.reported !== 'object') {
    target.innerHTML = '';
    return;
  }
  const reported = summary.reported;
  const reportedRow = decisionQualityHeadlineLabeler(reported);
  const reportedAccuracy = decisionQualityMetric(reportedRow, 'accuracy');
  const reportedF1 = decisionQualityMetric(reportedRow, 'f1');
  const reportedSplit = String(summary.reported_split || 'test').toUpperCase();
  const cards = [
    ['TEST set · reported', decisionQualityImageCount(reported), `${reportedSplit} only; train rows excluded from headline`],
    ['Reported majority accuracy', rushApiFormatMetric(reportedAccuracy), `${decisionQualityLabelerName(reportedRow)} · test only`]
  ];
  if (reportedF1 !== null) cards.push(['Reported majority F1', rushApiFormatMetric(reportedF1), 'test-only decision quality']);

  const train = summary.by_split?.train;
  if (train && typeof train === 'object') {
    const trainRow = decisionQualityHeadlineLabeler(train);
    const trainAccuracy = decisionQualityMetric(trainRow, 'accuracy');
    const trainNote = trainAccuracy !== null
      ? `secondary reference · majority accuracy ${rushApiFormatMetric(trainAccuracy)}`
      : 'secondary reference only';
    cards.push(['train (updates, not reported)', decisionQualityImageCount(train), trainNote]);
  }

  target.innerHTML = cards.map(([k, v, n]) =>
    `<article class="stat-card"><span>${esc(k)}</span><strong>${esc(v)}</strong><p>${esc(n || '')}</p></article>`
  ).join('');
}

function renderScoreUpdateCandidates() {
  const target = $('#scoreUpdateCandidates');
  if (!target) return;
  const candidates = Array.isArray(runState.summary?.update_candidates) ? runState.summary.update_candidates : [];
  if (!candidates.length) {
    target.innerHTML = '';
    return;
  }
  const rows = candidates.slice(0, 8).map(candidate => `
    <tr>
      <td><strong>${esc(candidate.image_id || candidate.sample_id || '—')}</strong></td>
      <td>${esc(candidate.misalignment_type || '—')}</td>
      <td>${esc(candidate.severity || '—')}</td>
    </tr>`).join('');
  const clipped = candidates.length > 8 ? `<p class="row-meta">Showing 8 of ${candidates.length} train-derived update candidate(s).</p>` : '';
  target.innerHTML = `
    <article class="score-algo-lane train-lane">
      <span>TRAIN update candidates</span>
      <strong>Policy/prompt updates are driven by these training misalignments.</strong>
      <p>Live backend input is exported here; SME review and applying the proposal remain the manual next step.</p>
      <div class="misalignment-table residual-table">
        <table class="misalignment"><thead><tr><th>image</th><th>misalignment</th><th>severity</th></tr></thead><tbody>${rows}</tbody></table>
      </div>
      ${clipped}
    </article>`;
}

function renderScoreAlgorithmState() {
  const summary = runState.summary || {};
  const hasReported = !!summary.reported && typeof summary.reported === 'object';
  const badge = $('#scoreAlgoBadge');
  if (badge) {
    badge.textContent = hasReported ? 'live · test-reported / train-updates' : 'intended pipeline · defined-not-executed';
  }
  const rule = $('#scoreAlgoRule');
  if (rule) {
    rule.textContent = hasReported
      ? 'Learn from train, report on test — never leak. Live split-separated export loaded: TEST decision quality is reported, while TRAIN residuals and update candidates drive policy work.'
      : 'Learn from train, report on test — never leak. Current backend scoring exports one combined run-level decision-quality snapshot, so split-separated train updates vs test metrics are labeled here as the intended pipeline until backend separation is wired.';
  }
  renderScoreReportedMetrics();
  renderScoreUpdateCandidates();
}

function renderScoreAlgorithmControls() {
  const k = scoreKPerSplitValue();
  const split = scoreSplitValue();
  const note = $('#scoreKNote');
  if (note) {
    const total = split === 'all' ? `up to ${k} training + ${k} test images` : `up to ${k} ${scoreSplitPhrase(split)} images`;
    note.textContent = `Mirrors §3 Batch size and Split for the next run. k=${k} per split means ${total}. Set k=10 for a 10+10 pass. §1 N per class builds the candidate pool.`;
  }
  const defaultBatch = $('#labelDefaultBatch');
  if (defaultBatch) defaultBatch.textContent = `k=${k} per split · ${scoreSplitPhrase(split)}`;
}

function syncScoreControlsFromRunTrigger() {
  const kInput = $('#scoreKPerSplit');
  const splitMirror = $('#scoreSplitMirror');
  const runK = $('#runTriggerBatchSize')?.value || '20';
  const runSplit = $('#runTriggerSplit')?.value || 'all';
  if (kInput && kInput.value !== runK) kInput.value = runK;
  if (splitMirror && splitMirror.value !== runSplit) splitMirror.value = runSplit;
  renderScoreAlgorithmControls();
}

function pushScoreControlsToRunTrigger() {
  const k = scoreKPerSplitValue();
  const split = scoreSplitValue();
  const runK = $('#runTriggerBatchSize');
  const runSplit = $('#runTriggerSplit');
  if (runK) runK.value = String(k);
  if (runSplit) runSplit.value = split;
  renderScoreAlgorithmControls();
}

function bindScoreAlgorithmControls() {
  syncScoreControlsFromRunTrigger();
  $('#scoreKPerSplit')?.addEventListener('input', pushScoreControlsToRunTrigger);
  $('#scoreSplitMirror')?.addEventListener('change', pushScoreControlsToRunTrigger);
  $('#runTriggerBatchSize')?.addEventListener('input', (event) => {
    if (event?.target) event.target.dataset.userEdited = '1';
  });
  $('#runTriggerBatchSize')?.addEventListener('input', syncScoreControlsFromRunTrigger);
  $('#runTriggerSplit')?.addEventListener('change', syncScoreControlsFromRunTrigger);
}

function residualRows() {
  const data = runState.misalignment;
  const rows = Array.isArray(data?.rows) ? data.rows : (Array.isArray(data?.records) ? data.records : []);
  return rows.filter(row => row && row.misalignment_type !== 'all_agree');
}

function residualRowTable(entries, emptyMessage) {
  if (!entries.length) return `<div class="empty-state">${esc(emptyMessage)}</div>`;
  const body = entries.slice(0, 30).map(entry => {
    const row = entry.row;
    const id = row.image_id || row.sample_id || '';
    const sme = row.sme_truth || row.truth || '—';
    const reason = row.disagreement_reason || row.reason || row.misalignment_type || '—';
    const pairChip = boundaryPairChip(row);
    return `<tr>
      <td><strong>${esc(id)}</strong></td>
      <td>${esc(splitLabel(entry.split))}</td>
      <td><span class="badge ${labelBadgeClass(sme)}">${esc(sme)}</span></td>
      <td>${majorityPill(row)}</td>
      <td>${esc(reason)}${pairChip ? `<div class="boundary-pair-row">${pairChip}</div>` : ''}</td>
      <td><button type="button" class="residual-label-update" disabled title="Coming soon: update a human label before guideline building so one bad label does not propagate.">Update human label</button></td>
    </tr>`;
  }).join('');
  const clipped = entries.length > 30 ? `<p class="row-meta">Showing 30 of ${entries.length} residual row(s).</p>` : '';
  return `<div class="misalignment-table residual-table"><table class="misalignment"><thead><tr><th>image</th><th>split</th><th>human label</th><th>majority</th><th>residual</th><th>human label</th></tr></thead><tbody>${body}</tbody></table></div>${clipped}`;
}

function renderResidualMisalignments() {
  const target = $('#residualMisalignments');
  if (!target) return;
  const rows = residualRows();
  const lookup = manifestSplitLookup();
  const annotated = rows.map(row => ({ row, split: splitForScoredRow(row, lookup) }));
  const trainRows = annotated.filter(entry => splitKind(entry.split) === 'train');
  const testRows = annotated.filter(entry => splitKind(entry.split) === 'test');
  const unknownRows = annotated.filter(entry => !splitKind(entry.split));
  const splitExported = rows.some(row => Object.prototype.hasOwnProperty.call(row, 'split') && splitKind(row.split));
  const status = rows.length
    ? `${rows.length} real residual row(s) in the selected run export${splitExported ? ', grouped by split where available.' : ', but per-row split is not exported in the current scoring payload.'}`
    : 'No residual misalignment rows are available for the selected run.';
  const splitState = splitExported
    ? 'Live: train residuals drive policy updates; test residuals are reported-only.'
    : 'Train/update vs test/report separation remains the intended pipeline until per-row split is exported.';
  target.innerHTML = `
    <div class="residual-misalign-status">${esc(status)} ${esc(splitState)}</div>
    <article class="residual-lane train-lane">
      <header><span>TRAIN residuals</span><strong>${trainRows.length}</strong></header>
      ${residualRowTable(trainRows, 'No split-tagged training residual rows are exported for this run yet.')}
    </article>
    <article class="residual-lane test-lane">
      <header><span>TEST residuals</span><strong>${testRows.length}</strong></header>
      ${residualRowTable(testRows, 'No split-tagged test residual rows are exported for this run yet.')}
    </article>
    ${unknownRows.length ? `<article class="residual-lane residual-unknown"><header><span>Unassigned residual rows</span><strong>${unknownRows.length}</strong></header><p class="row-meta">Real misalignment rows from the current export; split could not be assigned from row data or the loaded manifest.</p>${residualRowTable(unknownRows, '')}</article>` : ''}
  `;
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
        <dt>boundary</dt><dd>${esc(voteBool(vote.is_boundary))}${boundaryPairChip(vote)}</dd>
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
      return `<td${title}><span class="badge ${cls}">${esc(vote)}</span>${boundaryPairChip(voteRow)}</td>`;
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
    '<th>SME truth</th>',
    '<th>majority</th>',
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
    const majorityTally = r.majority_label ? null : tallyVotes({ votes: r.voters });
    const majorityLabel = r.majority_label || majorityTally?.label;
    const isMajorityTie = !!r.tie || !!majorityTally?.isTie;
    const tiedLabels = Array.isArray(r.tied_labels) ? r.tied_labels : (majorityTally?.tiedLabels || []);
    const majorityCell = (() => {
      if (isMajorityTie) {
        const tied = tiedLabels.length ? tiedLabels.join(' / ') : (majorityLabel || '—');
        return `<td><span class="badge dev" title="tie between ${attr(tied)}">tie</span></td>`;
      }
      if (!majorityLabel) return '<td><span class="muted">—</span></td>';
      return `<td><span class="badge ${labelBadgeClass(majorityLabel)}">${esc(majorityLabel)}</span></td>`;
    })();
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
      return `<td><span class="badge ${cls}" title="confidence${conf}${boundary}${cost}">${esc(v.label)}</span>${boundaryPairChip(v)}</td>`;
    }).join('');
    const distChips = Object.entries(r.vote_distribution || {})
      .sort((a, b) => b[1] - a[1])
      .map(([lbl, cnt]) => `<span class="dist-chip ${labelBadgeClass(lbl)}">${esc(lbl)}: ${cnt}</span>`)
      .join(' ');
    const mismatch = sme && majorityLabel && majorityLabel !== sme;
    const rowCls = mismatch ? ' class="row-mismatch"' : '';
    const thumbSrc = thumbnailSrcForPath(r.repo_rel_path || '');
    const thumb = thumbSrc ? `<img class="row-thumb thumb-loading" src="${attr(thumbSrc)}" alt="${attr(r.image_id)}" loading="lazy" decoding="async" onload="this.classList.remove('thumb-loading')" onerror="this.replaceWith(safeImageFallback('image unavailable','local path missing'))" />` : '';
    const primary = `<tr data-image-id="${attr(r.image_id)}"${rowCls}><td>${expandButton('consensus', r.image_id)}</td><td><div class="thumb-wrap">${thumb}<div><button type="button" class="image-id-button" data-open-justifications="${attr(r.image_id)}"><strong>${esc(r.image_id)}</strong></button>${mismatch ? '<p class="row-meta mismatch-note">majority ≠ SME</p>' : ''}</div></div></td><td>${smeBadge}</td>${majorityCell}${perModel}<td>${chipFor(r)}</td><td>${distChips || '<span class="muted">—</span>'}</td></tr>`;
    return primary + renderInlineJustificationsRow('consensus', r, voterColumns.length + 6);
  }).join('');
  tableTarget.innerHTML = `<table class="misalignment"><thead><tr>${headerCells}</tr></thead><tbody>${rows}</tbody></table>`;
}

function renderRun() {
  if (!document.querySelector('.score-tab-panel:not([hidden])')) selectScoreTab('consensus');
  renderScoreAlgorithmControls();
  renderScoreAlgorithmState();
  renderRunPicker();
  renderRunSummary();
  renderBorderline();
  renderMisalignment();
  renderConsensus();
  renderResidualMisalignments();
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
      status.textContent = activeDemoIsMnist() ? MNIST_EMPTY_RUN_MESSAGE : 'No runs found yet. Use the Run panel above to start one.';
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

function selectScoreTab(tabName = 'consensus') {
  const selected = tabName || 'consensus';
  document.querySelectorAll('.score-tabs [data-score-tab]').forEach(tab => {
    tab.setAttribute('aria-selected', String(tab.dataset.scoreTab === selected));
  });
  document.querySelectorAll('.score-tab-panel[data-score-panel]').forEach(panel => {
    panel.hidden = panel.dataset.scorePanel !== selected;
  });
}

function initScoreTabs() {
  const current = document.querySelector('.score-tabs [data-score-tab][aria-selected="true"]')?.dataset.scoreTab || 'consensus';
  selectScoreTab(current);
  document.querySelectorAll('.score-tabs [data-score-tab]').forEach(tab => {
    tab.addEventListener('click', () => selectScoreTab(tab.dataset.scoreTab || 'consensus'));
  });
}

// Set text/html for an element if it exists and a value was supplied.
function setNodeText(id, value) {
  if (value == null) return;
  const el = document.getElementById(id);
  if (el) el.textContent = value;
}
function setNodeHtml(id, value) {
  if (value == null) return;
  const el = document.getElementById(id);
  if (el) el.innerHTML = value;
}

// Apply demo-specific page chrome: title, hero copy, selector value, body flag,
// plus per-section MNIST copy overrides. GenAI keeps its HTML defaults because
// its sectionCopy is unset.
function applyDemoChrome() {
  const demo = activeDemo();
  const hero = demo.heroCopy || {};
  const copy = demo.sectionCopy || {};
  document.title = demo.title ? `RUSH · ${demo.title}` : 'RUSH Demo';
  setNodeText('heroEyebrow', hero.eyebrow);
  setNodeText('heroH1', hero.h1);
  setNodeText('heroLede', hero.lede);
  setNodeText('heroCta', hero.cta);
  // Section copy overrides (only set when a value is provided).
  setNodeText('sampleH2', copy.sampleH2);
  setNodeText('sampleSub', copy.sampleSub);
  setNodeText('growSub', copy.growSub);
  setNodeText('growLoopStep2Body', copy.growLoopStep2Body);
  setNodeText('labelH2', copy.labelH2);
  setNodeText('labelSub', copy.labelSub);
  setNodeText('labelStartButton', copy.labelStartButton);
  const startBtn = document.getElementById('startLabelingRun');
  if (startBtn && copy.labelStartButton) startBtn.textContent = copy.labelStartButton;
  setNodeText('labelDefaultBatch', copy.labelDefaultBatch);
  setNodeHtml('labelDefaultDetail', copy.labelDefaultDetail);
  setNodeText('scoreSub', copy.scoreSub);
  setNodeText('consensusH3', copy.consensusH3);
  setNodeText('consensusSub', copy.consensusSub);
  setNodeText('misalignmentH3', copy.misalignmentH3);
  setNodeText('misalignmentSub', copy.misalignmentSub);
  setNodeText('borderlineH3', copy.borderlineH3);
  setNodeHtml('borderlineSub', copy.borderlineSub);
  setNodeText('qualityH2', copy.qualityH2);
  setNodeText('qualitySub', copy.qualitySub);
  if (copy.policyGraphTitle) setNodeText('policyGraphTitle', copy.policyGraphTitle);
  if (copy.policyGraphBlurb) setNodeHtml('policyGraphBlurb', copy.policyGraphBlurb);
  // X1 polish/switcher: support the prominent segmented demo control.
  const selector = document.getElementById('demoSelector');
  if (selector) {
    if ('value' in selector) selector.value = demo.id;
    selector.querySelectorAll('[data-demo-id]').forEach(button => {
      const active = button.dataset.demoId === demo.id;
      button.classList.toggle('active', active);
      button.setAttribute('aria-pressed', active ? 'true' : 'false');
      if (active) button.setAttribute('aria-current', 'page');
      else button.removeAttribute('aria-current');
    });
  }
  document.body.dataset.rushDemo = demo.id;
  // Apply the per-demo default k (images per split). MNIST defaults to 50
  // (5 per digit class) to maximize learnings per batch; genai stays at 20.
  if (Number.isFinite(demo.defaultK)) {
    const runK = document.getElementById('runTriggerBatchSize');
    if (runK && !runK.dataset.userEdited) runK.value = String(demo.defaultK);
    const scoreK = document.getElementById('scoreKPerSplit');
    if (scoreK && !scoreK.dataset.userEdited) scoreK.value = String(demo.defaultK);
  }
  const benchmark = document.getElementById('benchmarkComparison');
  if (benchmark) benchmark.hidden = demo.id !== 'mnist';
  const provenance = document.getElementById('gp0Provenance');
  if (provenance) provenance.hidden = demo.id !== 'mnist';
  rebuildConsensusFilter();
}

// Rebuild the §4 consensus filter <select> from demo.consensusFilters when the
// demo supplies them (e.g. mnist adds pair:X-Y options). GenAI keeps its
// four hardcoded options untouched.
function rebuildConsensusFilter() {
  const demo = activeDemo();
  const select = document.getElementById('consensusFilter');
  const options = Array.isArray(demo.consensusFilters) ? demo.consensusFilters : null;
  if (!select || !options || !options.length) return;
  const current = runState.consensusFilter || select.value || 'all';
  select.innerHTML = options.map(opt =>
    `<option value="${attr(opt.value)}">${esc(opt.label)}</option>`
  ).join('');
  if (options.some(opt => opt.value === current)) select.value = current;
}

// Header demo selector: persist choice and reload with ?demo= (simple + robust).
function bindDemoSelector() {
  const selector = document.getElementById('demoSelector');
  if (!selector) return;
  // X1 polish/switcher: segmented buttons are the primary UI; keep a select
  // fallback for older markup or test fixtures.
  const activateDemo = id => {
    if (!window.RUSH_DEMOS || !window.RUSH_DEMOS[id]) return;
    try { window.localStorage.setItem('rush_active_demo', id); } catch (error) { /* ignore */ }
    const url = new URL(window.location.href);
    url.searchParams.set('demo', id);
    window.location.href = url.toString();
  };
  selector.addEventListener('click', event => {
    const target = event.target;
    if (!(target instanceof Element)) return;
    const button = target.closest('[data-demo-id]');
    if (!button) return;
    activateDemo(button.dataset.demoId);
  });
  if ('value' in selector) {
    selector.addEventListener('change', () => activateDemo(selector.value));
  }
}

function bindControls() {
  $('#runSampler')?.addEventListener('click', runSamplerDemo);
  $('#randomSamplerSeed')?.addEventListener('click', () => {
    $('#samplerSeed').value = String(Math.floor(100000 + Math.random() * 2140000000));
    runSamplerDemo();
  });
  $('#samplePreviewPerClass')?.addEventListener('input', renderGallery);
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
  initInlineJustificationStyles();
  if (typeof window.rushResolveActiveDemoId === 'function') window.rushResolveActiveDemoId();
  applyDemoChrome();
  bindDemoSelector();
  bindControls();
  bindRunControls();
  bindScoreAlgorithmControls();
  initScoreTabs();
  initActiveNav();
  initApi();
  runSamplerDemo();
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
      if (section.id === 'label') section.hidden = true;
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
  const demoId = activeDemoId();
  const area = activePolicyGraphArea();
  const [runsPayload, versionsPayload] = await Promise.all([
    rushApiGetJson(`/api/runs?demo=${encodeURIComponent(demoId)}`)
      .catch(() => rushApiGetJson('/api/runs').catch(() => ({ runs: [] }))),
    rushApiGetJson(`/api/policy/versions?area=${encodeURIComponent(area)}`)
      .catch(() => ({ versions: [{ version: activeDemo().policyGraph?.version || 'v0.1' }], current: activeDemo().policyGraph?.version || 'v0.1' }))
  ]);
  const versions = (Array.isArray(versionsPayload.versions) ? versionsPayload.versions : [])
    .map(version => (typeof version === 'string' ? { version } : version))
    .filter(version => version?.version);
  window.RUSH_API.catalog = {
    runs: filterRunsForActiveDemo(Array.isArray(runsPayload.runs) ? runsPayload.runs : []),
    policyVersions: versions,
    currentPolicyVersion: versionsPayload.current || versions[0]?.version || activeDemo().policyGraph?.version || ''
  };
  window.dispatchEvent(new CustomEvent('rush-api-catalog', { detail: window.RUSH_API.catalog }));
  return window.RUSH_API.catalog;
}

function rushApiRunOptions(selected = '', includeAll = false, allLabel = 'All scored runs') {
  const runs = filterRunsForActiveDemo(window.RUSH_API?.catalog?.runs || []);
  const prefix = includeAll ? rushApiOptionHtml('', allLabel, !selected) : '';
  if (!runs.length) return prefix || rushApiOptionHtml('', 'No runs found', true);
  return prefix + runs.map(run => {
    const runId = run.run_id || '';
    return rushApiOptionHtml(runId, runId || 'unknown run', selected === runId);
  }).join('');
}

function rushApiPolicyVersionOptions(selected = '', includeAll = false, allLabel = 'All policy versions') {
  const versions = window.RUSH_API?.catalog?.policyVersions || [];
  const prefix = includeAll ? rushApiOptionHtml('', allLabel, !selected) : '';
  if (!versions.length) return prefix || rushApiOptionHtml('', 'No policy versions found', true);
  const currentVersion = window.RUSH_API?.catalog?.currentPolicyVersion || '';
  return prefix + versions.map(item => {
    const version = item.version || item;
    const label = version === currentVersion ? `${version} · current` : version;
    return rushApiOptionHtml(version, label, selected === version);
  }).join('');
}

function rushApiFormatMetric(value, digits = 1) {
  return isNumber(value) ? `${(value * 100).toFixed(digits)}%` : '—';
}

function rushApiShort(value, fallback = '—') {
  const text = String(value ?? '').trim();
  return text || fallback;
}
