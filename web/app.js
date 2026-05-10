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


// GenAI sampler demo: loads ignored local manifests/images when available, with deterministic synthetic fallback.
const samplerState = { result: null, overrides: {} };
const manifestPaths = {
  dev: '../data/images/genai-classification/manifests/dev_golden_labels.csv',
  holdout: '../data/images/genai-classification/manifests/holdout_labels.csv',
  summary: '../data/images/genai-classification/manifests/sampling_summary.json'
};

function parseCsv(text){
  const rows=[]; let row=[]; let cell=''; let quoted=false;
  for(let i=0;i<text.length;i++){
    const char=text[i], next=text[i+1];
    if(quoted){
      if(char==='"' && next==='"'){cell+='"'; i++;}
      else if(char==='"') quoted=false;
      else cell+=char;
    } else if(char==='"') quoted=true;
    else if(char===','){row.push(cell); cell='';}
    else if(char==='\n'){row.push(cell); rows.push(row); row=[]; cell='';}
    else if(char !== '\r') cell+=char;
  }
  if(cell || row.length){row.push(cell); rows.push(row);}
  const [header,...body]=rows.filter(r=>r.some(v=>v!==''));
  if(!header) return [];
  return body.map(values=>Object.fromEntries(header.map((key,index)=>[key, values[index] ?? ''])));
}

async function fetchText(path){
  const response = await fetch(path, {cache:'no-store'});
  if(!response.ok) throw new Error(`${response.status} ${response.statusText}`);
  return response.text();
}

function normalizeManifestRow(row, split){
  const repoPath=row.repo_rel_path || row.synthetic_repo_rel_path || '';
  return {
    ...row,
    split,
    label_int: Number.parseInt(row.label_int, 10),
    seed: Number.parseInt(row.seed, 10) || 20260510,
    repo_rel_path: repoPath,
    synthetic_repo_rel_path: repoPath,
    original_filename: row.original_filename || repoPath.split('/').pop(),
    llm_status: 'LLM labeling comes next — no model votes generated in this demo.'
  };
}

async function loadLocalSamplerManifests(){
  const [devText, holdoutText, summaryText] = await Promise.all([
    fetchText(manifestPaths.dev),
    fetchText(manifestPaths.holdout),
    fetchText(manifestPaths.summary)
  ]);
  return {
    devGolden: parseCsv(devText).map(row=>normalizeManifestRow(row, 'dev_golden')),
    holdout: parseCsv(holdoutText).map(row=>normalizeManifestRow(row, 'holdout')),
    manifestSummary: JSON.parse(summaryText)
  };
}

function readSamplerOptions(){
  return {
    seed: Number.parseInt($('#samplerSeed')?.value, 10) || 20260510,
    mode: $('#samplerMode')?.value || 'cold_start',
    nDev: Number.parseInt($('#samplerDevN')?.value, 10) || 100,
    nHoldout: Number.parseInt($('#samplerHoldoutN')?.value, 10) || 100
  };
}

function countBy(records, fn){
  return records.reduce((acc,row)=>{const key=fn(row)||'unknown'; acc[key]=(acc[key]||0)+1; return acc;}, {});
}

function take(records, n){ return records.slice(0, Math.max(0, n)); }

async function runRealOrSyntheticSampler(){
  const options=readSamplerOptions();
  try{
    const local=await loadLocalSamplerManifests();
    const manifestSeed=Number.parseInt(local.manifestSummary?.seed, 10);
    if(options.mode !== 'cold_start') throw new Error('Local manifests are cold-start only; falling back for warm-start preview.');
    if(manifestSeed && options.seed !== manifestSeed) throw new Error('Seed differs from local manifest seed; falling back so random seed changes the sample.');
    const devGolden=take(local.devGolden, options.nDev);
    const holdout=take(local.holdout, options.nHoldout);
    return {
      devGolden,
      holdout,
      combined: devGolden.concat(holdout),
      summary: {seed: options.seed, mode: options.mode, source: 'local manifests + image files', samplingVersion: local.manifestSummary?.sampling_version, n_dev_golden: devGolden.length, n_holdout: holdout.length},
      leakageChecks: {ok: true, devHoldoutDisjoint: true, note: 'Local manifests were generated by the Python sampler with path/hash leakage checks.'}
    };
  } catch(error){
    if(!window.RushGenaiSampler?.runDemoReset) throw error;
    const fallback=window.RushGenaiSampler.runDemoReset(options);
    return {...fallback, summary: {...fallback.summary, source: 'synthetic browser fallback'}};
  }
}

function setSamplerLoading(isLoading, text){
  const banner=document.querySelector('.sampler-banner');
  const loadingText=$('#samplerLoadingText');
  if(banner){banner.classList.toggle('loading', isLoading); banner.classList.toggle('loading-ready', !isLoading);}
  if(loadingText) loadingText.textContent=text;
}

function labelText(row){
  return row.label === 'ai_generated' ? 'AI generated' : 'Not AI generated';
}

function renderImage(row){
  const path=row.repo_rel_path || row.synthetic_repo_rel_path;
  if(path && row.repo_rel_path){
    return `<img src="../${attr(path)}" alt="${attr(row.sample_id)} ${attr(labelText(row))}" loading="lazy" onerror="this.closest('.sample-thumb').innerHTML='<div class=&quot;thumb-fallback&quot;><strong>image unavailable</strong><span>check local file path</span></div>'" />`;
  }
  return `<div class="thumb-fallback"><strong>${esc(row.dataset)}</strong><span>${esc(labelText(row))}</span></div>`;
}

function overrideFor(sampleId){
  return samplerState.overrides[sampleId] || {label:'none', note:''};
}

function renderBalance(title, counts){
  return `<div class="balance-card"><h4>${esc(title)}</h4>${Object.entries(counts || {}).map(([key,value])=>`<div><span>${esc(key)}</span><strong>${esc(value)}</strong></div>`).join('')}</div>`;
}

function renderSamplerSummary(){
  const result=samplerState.result;
  const combined=result?.combined || [];
  const summary=result?.summary || {};
  const byClass=countBy(combined, row=>row.label);
  const byDataset=countBy(combined, row=>row.dataset);
  const bySplit=countBy(combined, row=>row.split);
  $('#samplerSummary').innerHTML=`
    <div class="sampler-meta">
      <span class="pill">seed ${esc(summary.seed)}</span>
      <span class="pill">${esc(summary.mode)}</span>
      <span class="pill">${esc(summary.source || 'demo')}</span>
      ${summary.samplingVersion ? `<span class="pill">${esc(summary.samplingVersion)}</span>` : ''}
    </div>
    <div class="sampler-balances">
      ${renderBalance('Class labels', byClass)}
      ${renderBalance('Datasets', byDataset)}
      ${renderBalance('Splits', bySplit)}
      ${renderBalance('Leakage', {ok: result.leakageChecks?.ok ? 'pass' : 'review'})}
    </div>`;
}

function renderSampleCard(row){
  const locked=row.split === 'holdout';
  const override=overrideFor(row.sample_id);
  return `<article class="sample-card ${locked ? 'locked' : ''}" data-sample-id="${attr(row.sample_id)}">
    <div class="sample-thumb">${renderImage(row)}</div>
    <div class="sample-card-head">
      <span class="badge ${locked ? 'split-holdout' : 'split-development'}">${locked ? 'locked holdout' : 'dev golden'}</span>
      <span class="badge ${row.label === 'ai_generated' ? 'gold' : 'silver'}">SME/source label: ${esc(labelText(row))}</span>
    </div>
    <h3>${esc(row.dataset)} / ${esc(row.source_label_dir || 'source')}</h3>
    <p>${esc(row.original_filename || row.sample_id)}<br><strong>Directory label:</strong> ${esc(row.source_label_dir || 'n/a')} → ${esc(labelText(row))}<br><strong>Status:</strong> ${esc(row.llm_status || 'LLM labeling comes next')}</p>
    <div class="override-controls">
      <label>SME/human override
        <select class="override-label" data-sample-id="${attr(row.sample_id)}">
          ${['none','ai_generated','not_ai_generated','needs_review'].map(value=>`<option value="${value}" ${override.label===value?'selected':''}>${esc(value)}</option>`).join('')}
        </select>
      </label>
      <label>Note
        <input class="override-note" data-sample-id="${attr(row.sample_id)}" type="text" value="${attr(override.note)}" placeholder="Reason, uncertainty, or SME cue" />
      </label>
    </div>
  </article>`;
}

function renderSamplerRecords(){
  const result=samplerState.result;
  if(!result) return;
  $('#samplerRecords').innerHTML=`
    <div class="override-summary" id="overrideSummary">Loaded ${esc(result.combined.length)} records. Showing all sampled images/labels below.</div>
    <div class="sample-section"><h3>Generated dev golden candidates</h3><div class="sample-grid">${result.devGolden.map(renderSampleCard).join('')}</div></div>
    <div class="sample-section"><h3>Locked holdout candidates</h3><div class="sample-grid">${result.holdout.map(renderSampleCard).join('')}</div></div>`;
}

function updateOverrideSummary(){
  const values=Object.values(samplerState.overrides).filter(item=>item.label !== 'none' || item.note.trim());
  const summary=$('#overrideSummary');
  if(summary) summary.textContent=`Loaded ${samplerState.result?.combined?.length || 0} records. ${values.length} SME override note(s) in this browser session.`;
}

async function runSamplerDemo(){
  const status=$('#samplerStatus');
  if(!status) return;
  setSamplerLoading(true, 'Loading sampled images and labels…');
  status.classList.remove('error');
  status.textContent='Loading manifests, labels, and thumbnails…';
  try{
    samplerState.result=await runRealOrSyntheticSampler();
    samplerState.overrides={};
    renderSamplerSummary();
    renderSamplerRecords();
    status.textContent=`Demo loaded: ${samplerState.result.combined.length} records with directory-derived SME/source labels.`;
    setSamplerLoading(false, 'Loaded. Images and labels are ready below.');
  }catch(error){
    console.error('Sampler demo failed:', error);
    status.classList.add('error');
    status.textContent=`Sampler demo failed: ${error.message}`;
    setSamplerLoading(false, 'Load failed. See error below.');
  }
}

function initSamplerDemo(){
  const runButton=$('#runSampler');
  if(!runButton) return;
  runButton.addEventListener('click', runSamplerDemo);
  $('#randomSamplerSeed')?.addEventListener('click',()=>{
    $('#samplerSeed').value=String(Math.floor(100000 + Math.random()*2140000000));
    runSamplerDemo();
  });
  $('#samplerRecords')?.addEventListener('input', event=>{
    const target=event.target;
    const sampleId=target?.dataset?.sampleId;
    if(!sampleId) return;
    const current=overrideFor(sampleId);
    if(target.classList.contains('override-note')) current.note=target.value;
    if(target.classList.contains('override-label')) current.label=target.value;
    samplerState.overrides[sampleId]=current;
    updateOverrideSummary();
  });
  $('#samplerRecords')?.addEventListener('change', event=>{
    const target=event.target;
    const sampleId=target?.dataset?.sampleId;
    if(!sampleId || !target.classList.contains('override-label')) return;
    samplerState.overrides[sampleId]={...overrideFor(sampleId), label: target.value};
    updateOverrideSummary();
  });
}

async function init(){
  setupTabs();
  drawGraph();
  initSamplerDemo();
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
