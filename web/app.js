const graphNodes = [
  {id:'GA.root', title:'GenAI Image Classification', type:'root', polarity:'mixed', x:50, y:12, rule:'Root decision rule for gen_ai vs not_gen_ai.', edges:['GA.visual_artifacts.anatomy.hands','GA.visual_artifacts.text_symbols','GA.surface_texture.plastic_skin','GA.scene_geometry.inconsistent_perspective','GA.provenance.synthetic_disclosure','GA.boundary.photo_editing','GA.boundary.cgi_game_render','GA.boundary.low_quality_uncertain']},
  {id:'GA.visual_artifacts.anatomy.hands', title:'Hand / limb artifacts', type:'category', polarity:'positive', x:19, y:36, rule:'Extra, missing, fused, duplicated, or impossible hands/limbs.'},
  {id:'GA.visual_artifacts.text_symbols', title:'Garbled text', type:'category', polarity:'positive', x:40, y:36, rule:'Pseudo-text, malformed logos, impossible integrated typography.'},
  {id:'GA.surface_texture.plastic_skin', title:'Plastic skin', type:'category', polarity:'positive', x:61, y:36, rule:'Synthetic waxy/poreless texture not explained by filters.'},
  {id:'GA.scene_geometry.inconsistent_perspective', title:'Geometry', type:'category', polarity:'positive', x:82, y:36, rule:'Impossible reflections, shadows, object intersections, perspective.'},
  {id:'GA.provenance.synthetic_disclosure', title:'Provenance', type:'category', polarity:'positive', x:22, y:70, rule:'Watermark, metadata, caption, or source evidence identifies generation.'},
  {id:'GA.boundary.photo_editing', title:'Photo editing', type:'boundary', polarity:'hard-negative', x:45, y:74, rule:'Filters, retouching, compression, and healing-brush artifacts are not GenAI by themselves.'},
  {id:'GA.boundary.cgi_game_render', title:'CGI / game', type:'boundary', polarity:'hard-negative', x:65, y:74, rule:'Rendered or stylized assets are not GenAI unless generative provenance is established.'},
  {id:'GA.boundary.low_quality_uncertain', title:'Uncertain low quality', type:'boundary', polarity:'negative', x:85, y:74, rule:'Blurred/cropped/ambiguous cases require abstain or SME review.'},
  {id:'GA.negative.authentic_photo', title:'Authentic photo', type:'category', polarity:'negative', x:8, y:74, rule:'Authentic unmodified or conventionally-edited photograph with no generative provenance.', edges:[]},
  {id:'GA.exception.compression_artifacts', title:'Compression artifacts', type:'exception', polarity:'hard-negative', x:30, y:90, rule:'JPEG/WebP compression artifacts that mimic GenAI texture. Not gen_ai unless independent evidence.', edges:[]},
  {id:'GA.exception.medical_prosthetic', title:'Medical / prosthetic', type:'exception', polarity:'hard-negative', x:52, y:90, rule:'Polydactyly, syndactyly, prosthetics, post-surgical — not gen_ai.', edges:[]},
  {id:'GA.visual_artifacts.repeated_details', title:'Repeated details', type:'category', polarity:'positive', x:72, y:50, rule:'Repeated teeth, jewelry, texture tiling from diffusion model failures.', edges:[]}
];

const graphEdges = [
  ['GA.root','GA.visual_artifacts.anatomy.hands'],
  ['GA.root','GA.visual_artifacts.text_symbols'],
  ['GA.root','GA.surface_texture.plastic_skin'],
  ['GA.root','GA.scene_geometry.inconsistent_perspective'],
  ['GA.root','GA.provenance.synthetic_disclosure'],
  ['GA.root','GA.boundary.photo_editing'],
  ['GA.root','GA.boundary.cgi_game_render'],
  ['GA.root','GA.boundary.low_quality_uncertain'],
  ['GA.surface_texture.plastic_skin','GA.boundary.photo_editing'],
  ['GA.visual_artifacts.anatomy.hands','GA.boundary.photo_editing'],
  ['GA.scene_geometry.inconsistent_perspective','GA.boundary.cgi_game_render'],
  ['GA.visual_artifacts.text_symbols','GA.boundary.low_quality_uncertain'],
  ['GA.negative.authentic_photo','GA.root'],
  ['GA.exception.compression_artifacts','GA.root'],
  ['GA.exception.compression_artifacts','GA.surface_texture.plastic_skin'],
  ['GA.exception.compression_artifacts','GA.boundary.low_quality_uncertain'],
  ['GA.exception.medical_prosthetic','GA.visual_artifacts.anatomy.hands'],
  ['GA.exception.medical_prosthetic','GA.surface_texture.plastic_skin'],
  ['GA.visual_artifacts.repeated_details','GA.root'],
  ['GA.visual_artifacts.repeated_details','GA.boundary.photo_editing']
];

const seedFiles = {
  images: '../data/seed/image-records.json',
  labels: '../data/seed/label-records.json',
  suggestions: '../data/seed/policy-suggestions.json',
  metrics: '../data/seed/metrics.json',
  decisionQuality: '../data/seed/decision-quality.json'
};

const $ = selector => document.querySelector(selector);
const esc = value => String(value ?? '').replace(/[&<>'"]/g, char => ({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}[char]));
const labelize = key => String(key ?? '').replaceAll('_',' ').replace(/\b\w/g, char => char.toUpperCase());
const isNumber = value => typeof value === 'number' && Number.isFinite(value);

function setupTabs(){
  document.querySelectorAll('.tab').forEach(btn=>btn.addEventListener('click',()=>{
    document.querySelectorAll('.tab').forEach(b=>{b.classList.remove('active');b.setAttribute('aria-selected','false')});
    btn.classList.add('active');btn.setAttribute('aria-selected','true');
    document.querySelectorAll('.panel').forEach(p=>p.classList.remove('active'));
    const panel = document.getElementById(btn.dataset.tab);
    if (panel) panel.classList.add('active');
  }));
}

async function fetchJson(path){
  const response = await fetch(path, {cache:'no-store'});
  if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
  return response.json();
}

async function loadSeedData(){
  const entries = await Promise.all(Object.entries(seedFiles).map(async ([key, path])=>{
    try {
      return [key, {data: await fetchJson(path), error: null}];
    } catch (error) {
      console.error(`Failed to load ${path}:`, error);
      return [key, {data: null, error}];
    }
  }));
  return Object.fromEntries(entries);
}

function failedPanel(selector){
  const el = $(selector);
  if (el) el.innerHTML = '<div class="load-error">Failed to load data</div>';
}

function byId(id){return graphNodes.find(n=>n.id===id)}
function drawGraph(){
  const c=$('#graphCanvas');
  c.innerHTML='<div class="mock-banner">Graph positions are static mock UI; policy truth lives in Markdown nodes and edges.json.</div>';
  graphEdges.forEach(([a,b])=>{
    const na=byId(a), nb=byId(b);
    if(!na || !nb) return;
    const line=document.createElement('div');line.className='graph-line';
    const update=()=>{const w=c.clientWidth,h=c.clientHeight;const x1=na.x*w/100,y1=na.y*h/100,x2=nb.x*w/100,y2=nb.y*h/100;const len=Math.hypot(x2-x1,y2-y1);line.style.width=len+'px';line.style.left=x1+'px';line.style.top=y1+'px';line.style.transform=`rotate(${Math.atan2(y2-y1,x2-x1)}rad)`};
    c.appendChild(line);update();window.addEventListener('resize',update)
  });
  graphNodes.forEach(n=>{const el=document.createElement('button');el.className=`graph-node ${n.type} ${n.polarity}`;el.style.left=n.x+'%';el.style.top=n.y+'%';el.innerHTML=`${esc(n.title)}<br><span style="color:#aab8d3;font-weight:650">${esc(n.id)}</span>`;el.addEventListener('click',()=>showNode(n));c.appendChild(el)});
  showNode(graphNodes[0]);
}
function showNode(n){$('#nodeDetails').innerHTML=`<h3>${esc(n.title)}</h3><p><strong>${esc(n.id)}</strong> · ${esc(n.type)} · ${esc(n.polarity)}</p><p>${esc(n.rule)}</p><ul><li>Version: Generative_AI.v0.1</li><li>Stores source anchors, coverage targets, examples, and boundary/confusion edges in Markdown frontmatter.</li><li>Future labelers cite this node when producing justification and confidence.</li></ul>`}

function labelByImage(labels){
  return labels.reduce((acc,label)=>{acc[label.image_id]=label;return acc},{});
}
function tierClass(tier){
  const normalized = String(tier || 'provisional').toLowerCase();
  if (normalized === 'gold' || normalized === 'platinum') return 'gold';
  if (normalized === 'silver') return 'silver';
  return 'provisional';
}
function splitClass(split){
  const normalized = String(split || '').toLowerCase();
  if (normalized.includes('holdout')) return 'split-holdout';
  if (normalized.includes('validation')) return 'split-validation';
  return 'split-development';
}
function renderGolden(images, labels){
  const labelsByImage = labelByImage(labels);
  $('#goldenGrid').innerHTML=images.map(image=>{
    const label = labelsByImage[image.image_id] || {};
    const tier = label.label_tier || 'unlabeled';
    const split = image.split || 'unassigned';
    const confidence = isNumber(label.confidence) ? `${Math.round(label.confidence*100)}%` : 'n/a';
    const nodes = Array.isArray(label.node_ids) && label.node_ids.length ? label.node_ids.map(esc).join('<br>') : 'no node evidence yet';
    return `<article class="golden-card ${splitClass(split)}"><div class="thumb">image pending</div><span class="badge ${tierClass(tier)}">${esc(tier)}</span><span class="badge split ${splitClass(split)}">${esc(split)}</span><span class="badge mock-only">mock seed</span><h3>${esc(image.image_id)}</h3><p>${esc(image.metadata?.note || 'Seed record pending real media.')}</p><p><strong>${esc(label.label || 'unlabeled')}</strong> · ${esc(confidence)}</p><p>${nodes}</p></article>`;
  }).join('')
}
function renderLabels(labels){
  $('#labelTable').innerHTML=`<table><thead><tr><th>Image</th><th>Labeler</th><th>Label</th><th>Conf.</th><th>Policy evidence</th><th>Tier</th></tr></thead><tbody>${labels.map(l=>`<tr><td>${esc(l.image_id)}</td><td>${esc(l.labeler_id || l.labeler_type)}</td><td>${esc(l.label)}</td><td>${isNumber(l.confidence) ? Math.round(l.confidence*100)+'%' : 'n/a'}</td><td>${esc(l.justification || '')}<br><span class="node-list">${esc((l.node_ids || []).join(', '))}</span></td><td><span class="badge ${tierClass(l.label_tier)}">${esc(l.label_tier || 'provisional')}</span></td></tr>`).join('')}</tbody></table>`
}
function renderDiffs(suggestions){
  $('#diffList').innerHTML=suggestions.length ? suggestions.map(s=>`<article class="diff-card"><p class="eyebrow">${esc(s.suggestion_type)} · ${esc(s.status)} · mock seed</p><h3>${esc(s.patch_id)}</h3><p>${esc(s.rationale)}</p><pre>${esc(JSON.stringify(s.proposed_diff, null, 2))}</pre></article>`).join('') : '<div class="load-error">No suggested diffs yet.</div>'
}

function denominatorTotal(snapshot){
  const d = snapshot?.denominators || {};
  if (isNumber(d.n_total)) return d.n_total;
  if (isNumber(d.paired_truth_prediction_n)) return d.paired_truth_prediction_n;
  if (isNumber(d.truth_n)) return d.truth_n;
  return 0;
}
function ciFor(snapshot, key){
  const intervals = snapshot?.confidence_intervals?.intervals;
  if (!intervals || !Object.prototype.hasOwnProperty.call(intervals, key)) return '';
  const interval = intervals[key];
  const lower = Array.isArray(interval) ? interval[0] : interval?.lower ?? interval?.low;
  const upper = Array.isArray(interval) ? interval[1] : interval?.upper ?? interval?.high;
  if (!isNumber(lower) || !isNumber(upper)) return '';
  return `[${formatMetricNumber(lower, key)}, ${formatMetricNumber(upper, key)}]`;
}
function formatMetricNumber(value, key){
  if (!isNumber(value)) return '—';
  if (key==='fpr'||key==='fnr'||key.includes('proportion')||key.includes('coverage')||key.includes('mass')||key.includes('rate')) return `${Math.round(value*100)}%`;
  if (Number.isInteger(value) && (key.endsWith('count') || key.includes('_nodes') || key.includes('_n'))) return String(value);
  return Number(value).toFixed(2);
}
function metricEntries(snapshot){
  const entries = Object.entries(snapshot.metrics || {});
  if (snapshot.macro_metrics) {
    for (const [key, value] of Object.entries(snapshot.macro_metrics)) {
      if (!entries.some(([existing])=>existing===key)) entries.push([key, value]);
    }
  }
  if (snapshot.calibration && !entries.some(([key])=>key==='calibration_ece')) entries.push(['calibration_ece', snapshot.calibration.ece]);
  if (snapshot.graph_location_metrics && !entries.some(([key])=>key==='gray_zone_mass')) entries.push(['gray_zone_mass', snapshot.graph_location_metrics.gray_zone_mass]);
  return entries;
}
function renderMetricCard(key, value, snapshot, group){
  const nTotal = denominatorTotal(snapshot);
  const structuralGraphKeys = new Set(['node_count','edge_count','orphan_nodes','coverage']);
  const shouldMask = value === null || value === undefined || (group === 'decision' && nTotal < 30) || (group === 'graph' && !structuralGraphKeys.has(key) && nTotal < 30);
  const display = shouldMask ? '—' : formatMetricNumber(value, key);
  const ci = ciFor(snapshot, key);
  return `<article class="metric-card ${shouldMask ? 'placeholder' : ''}"><div class="metric-label">${esc(key.replaceAll('_',' '))}</div><div class="metric-value">${display}</div><p class="metric-meta">n = ${esc(nTotal)}</p>${ci ? `<p class="ci-range">CI ${esc(ci)}</p>` : ''}<p>${group === 'graph' ? 'graph health' : 'decision quality'}</p></article>`;
}
function renderMetrics(snapshot){
  const graphEntries = Object.entries(snapshot.graph_health || {}).filter(([,value])=>value === null || typeof value !== 'object');
  $('#metricCards').innerHTML=[
    ...metricEntries(snapshot).map(([k,v])=>renderMetricCard(k, v, snapshot, 'decision')),
    ...graphEntries.map(([k,v])=>renderMetricCard(k, v, snapshot, 'graph'))
  ].join('');
  const d = snapshot.denominators || {};
  const n = denominatorTotal(snapshot);
  $('#metricNote').textContent = `${snapshot.warning || 'Metrics are placeholders until real labels exist.'} n_total=${n}; positive=${d.n_positive ?? d.positive_truth_n ?? 0}; negative=${d.n_negative ?? d.negative_truth_n ?? 0}. Gold/platinum labels are the only reportable truth tiers.`;
}

function dqName(id){
  const names = {
    'gpt-5.4':'GPT-5.4',
    'gpt-5.5':'GPT-5.5',
    'gpt-5.5-high':'GPT-5.5-high',
    'gemini-3.1-pro':'Gemini-3.1-pro',
    'majority_vote':'Majority Vote',
    'non_expert':'Non-expert'
  };
  return names[id] || labelize(id);
}
function dqMetric(value, key){
  if (value === null || value === undefined) return '<span class="dim">—</span>';
  if (key === 'n') return esc(value);
  return esc(formatMetricNumber(value, key));
}
function renderDecisionQuality(snapshot){
  const rows = snapshot.labelers || [];
  $('#dqTable').innerHTML=`<table><thead><tr><th>Labeler</th><th>Type</th><th>Accuracy</th><th>F1</th><th>Precision</th><th>Recall</th><th>FPR</th><th>FNR</th><th>Pos. Prop.</th><th>N</th><th>Informedness</th></tr></thead><tbody>${rows.map(row=>{
    const m = row.metrics || {};
    const n = m.n ?? 0;
    return `<tr class="${n === 0 ? 'dim-row' : ''}"><td>${esc(dqName(row.labeler_id))}</td><td>${esc(labelize(row.labeler_type))}</td><td>${dqMetric(m.accuracy,'accuracy')}</td><td>${dqMetric(m.f1,'f1')}</td><td>${dqMetric(m.precision,'precision')}</td><td>${dqMetric(m.recall,'recall')}</td><td>${dqMetric(m.fpr,'fpr')}</td><td>${dqMetric(m.fnr,'fnr')}</td><td>${dqMetric(m.positive_proportion,'positive_proportion')}</td><td>${esc(n)}</td><td>${dqMetric(m.informedness,'informedness')}</td></tr>`;
  }).join('')}</tbody></table>`;
}

async function init(){
  setupTabs();
  drawGraph();
  const seed = await loadSeedData();

  if (seed.images.data && seed.labels.data) renderGolden(seed.images.data, seed.labels.data);
  else failedPanel('#goldenGrid');

  if (seed.labels.data) renderLabels(seed.labels.data);
  else failedPanel('#labelTable');

  if (seed.suggestions.data) renderDiffs(seed.suggestions.data);
  else failedPanel('#diffList');

  if (seed.metrics.data) renderMetrics(seed.metrics.data);
  else { failedPanel('#metricCards'); $('#metricNote').textContent = 'Failed to load data'; }

  if (seed.decisionQuality.data) renderDecisionQuality(seed.decisionQuality.data);
  else failedPanel('#dqTable');
}

init().catch(error=>{
  console.error('Failed to initialize RUSH web UI:', error);
  failedPanel('#goldenGrid');
  failedPanel('#labelTable');
  failedPanel('#diffList');
  failedPanel('#metricCards');
  failedPanel('#dqTable');
});
