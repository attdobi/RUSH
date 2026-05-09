const graphNodes = [
  {id:'GA.root', title:'GenAI Image Classification', type:'root', polarity:'mixed', x:50, y:12, rule:'Root decision rule for gen_ai vs not_gen_ai.', edges:['GA.visual_artifacts.anatomy.hands','GA.visual_artifacts.text_symbols','GA.surface_texture.plastic_skin','GA.scene_geometry.inconsistent_perspective','GA.provenance.synthetic_disclosure','GA.boundary.photo_editing','GA.boundary.cgi_game_render','GA.boundary.low_quality_uncertain']},
  {id:'GA.visual_artifacts.anatomy.hands', title:'Hand / limb artifacts', type:'category', polarity:'positive', x:19, y:36, rule:'Extra, missing, fused, duplicated, or impossible hands/limbs.'},
  {id:'GA.visual_artifacts.text_symbols', title:'Garbled text', type:'category', polarity:'positive', x:40, y:36, rule:'Pseudo-text, malformed logos, impossible integrated typography.'},
  {id:'GA.surface_texture.plastic_skin', title:'Plastic skin', type:'category', polarity:'positive', x:61, y:36, rule:'Synthetic waxy/poreless texture not explained by filters.'},
  {id:'GA.scene_geometry.inconsistent_perspective', title:'Geometry', type:'category', polarity:'positive', x:82, y:36, rule:'Impossible reflections, shadows, object intersections, perspective.'},
  {id:'GA.provenance.synthetic_disclosure', title:'Provenance', type:'category', polarity:'positive', x:22, y:70, rule:'Watermark, metadata, caption, or source evidence identifies generation.'},
  {id:'GA.boundary.photo_editing', title:'Photo editing', type:'boundary', polarity:'hard-negative', x:45, y:74, rule:'Filters, retouching, compression, and healing-brush artifacts are not GenAI by themselves.'},
  {id:'GA.boundary.cgi_game_render', title:'CGI / game', type:'boundary', polarity:'hard-negative', x:65, y:74, rule:'Rendered or stylized assets are not GenAI unless generative provenance is established.'},
  {id:'GA.boundary.low_quality_uncertain', title:'Uncertain low quality', type:'boundary', polarity:'negative', x:85, y:74, rule:'Blurred/cropped/ambiguous cases require abstain or SME review.'}
];
const graphEdges = [
  ['GA.root','GA.visual_artifacts.anatomy.hands'],['GA.root','GA.visual_artifacts.text_symbols'],['GA.root','GA.surface_texture.plastic_skin'],['GA.root','GA.scene_geometry.inconsistent_perspective'],['GA.root','GA.provenance.synthetic_disclosure'],['GA.root','GA.boundary.photo_editing'],['GA.root','GA.boundary.cgi_game_render'],['GA.root','GA.boundary.low_quality_uncertain'],['GA.surface_texture.plastic_skin','GA.boundary.photo_editing'],['GA.visual_artifacts.anatomy.hands','GA.boundary.photo_editing'],['GA.scene_geometry.inconsistent_perspective','GA.boundary.cgi_game_render'],['GA.visual_artifacts.text_symbols','GA.boundary.low_quality_uncertain']
];
const fallbackSeed = {
  images: [
    {image_id:'ga_seed_0001', split:'development', metadata:{note:'Mock placeholder impossible-hand positive.'}},
    {image_id:'ga_seed_0002', split:'validation', metadata:{note:'Mock placeholder conventional-edit hard negative.'}},
    {image_id:'ga_seed_0003', split:'locked_holdout', metadata:{note:'Mock placeholder low-quality consensus-audit case.'}}
  ],
  labels: [
    {image_id:'ga_seed_0001', labeler_id:'sme_placeholder', label:'gen_ai', confidence:.86, node_ids:['GA.visual_artifacts.anatomy.hands'], justification:'Mock placeholder evidence.', label_tier:'gold'},
    {image_id:'ga_seed_0002', labeler_id:'sme_placeholder', label:'not_gen_ai', confidence:.72, node_ids:['GA.boundary.photo_editing'], justification:'Mock placeholder evidence.', label_tier:'gold'},
    {image_id:'ga_seed_0003', labeler_id:'future_high_reasoning_ensemble', label:'abstain', confidence:.48, node_ids:['GA.boundary.low_quality_uncertain'], justification:'Mock placeholder evidence.', label_tier:'provisional'}
  ],
  suggestions: [],
  metrics: {status:'mock_only', mock_only:true, metrics:{accuracy:null, precision:null, recall:null, fpr:null, positive_proportion:null, informedness:null}, graph_health:{coverage:.08, gray_zone_mass:null}, warning:'Mock-only fallback data.'}
};

const $ = selector => document.querySelector(selector);
const esc = value => String(value ?? '').replace(/[&<>'"]/g, char => ({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}[char]));

async function fetchJson(path, fallback){
  try {
    const response = await fetch(path, {cache:'no-store'});
    if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
    return await response.json();
  } catch (error) {
    console.warn(`Using fallback data for ${path}:`, error);
    return fallback;
  }
}

async function loadSeedData(){
  const [images, labels, suggestions, metrics] = await Promise.all([
    fetchJson('../data/seed/image-records.json', fallbackSeed.images),
    fetchJson('../data/seed/label-records.json', fallbackSeed.labels),
    fetchJson('../data/seed/policy-suggestions.json', fallbackSeed.suggestions),
    fetchJson('../data/seed/metrics.json', fallbackSeed.metrics)
  ]);
  return {images, labels, suggestions, metrics};
}

document.querySelectorAll('.tab').forEach(btn=>btn.addEventListener('click',()=>{
  document.querySelectorAll('.tab').forEach(b=>{b.classList.remove('active');b.setAttribute('aria-selected','false')});
  btn.classList.add('active');btn.setAttribute('aria-selected','true');
  document.querySelectorAll('.panel').forEach(p=>p.classList.remove('active'));
  document.getElementById(btn.dataset.tab).classList.add('active');
}));

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
  graphNodes.forEach(n=>{const el=document.createElement('button');el.className=`graph-node ${n.type==='boundary'?'boundary':n.polarity}`;el.style.left=n.x+'%';el.style.top=n.y+'%';el.innerHTML=`${esc(n.title)}<br><span style="color:#aab8d3;font-weight:650">${esc(n.id)}</span>`;el.addEventListener('click',()=>showNode(n));c.appendChild(el)});
  showNode(graphNodes[0]);
}
function showNode(n){$('#nodeDetails').innerHTML=`<h3>${esc(n.title)}</h3><p><strong>${esc(n.id)}</strong> · ${esc(n.type)} · ${esc(n.polarity)}</p><p>${esc(n.rule)}</p><ul><li>Version: Generative_AI.v0.1</li><li>Stores source anchors, coverage targets, examples, and boundary/confusion edges in Markdown frontmatter.</li><li>Future labelers cite this node when producing justification and confidence.</li></ul>`}

function labelByImage(labels){
  return labels.reduce((acc,label)=>{acc[label.image_id]=label;return acc},{});
}
function tierClass(tier){return tier === 'gold' || tier === 'platinum' ? 'gold' : 'provisional'}
function renderGolden(images, labels){
  const labelsByImage = labelByImage(labels);
  $('#goldenGrid').innerHTML=images.map(image=>{
    const label = labelsByImage[image.image_id] || {};
    const tier = label.label_tier || 'unlabeled';
    const confidence = typeof label.confidence === 'number' ? `${Math.round(label.confidence*100)}%` : 'n/a';
    const nodes = Array.isArray(label.node_ids) && label.node_ids.length ? label.node_ids.join('<br>') : 'no node evidence yet';
    return `<article class="golden-card"><div class="thumb">image pending</div><span class="badge ${tierClass(tier)}">${esc(tier)}</span><span class="badge mock-only">mock seed</span><h3>${esc(image.image_id)}</h3><p>${esc(image.metadata?.note || 'Seed record pending real media.')}</p><p><strong>${esc(label.label || 'unlabeled')}</strong> · ${esc(confidence)} · ${esc(image.split)}</p><p>${nodes}</p></article>`;
  }).join('')
}
function renderLabels(labels){
  $('#labelTable').innerHTML=`<table><thead><tr><th>Image</th><th>Labeler</th><th>Label</th><th>Conf.</th><th>Policy evidence</th><th>Tier</th></tr></thead><tbody>${labels.map(l=>`<tr><td>${esc(l.image_id)}</td><td>${esc(l.labeler_id || l.labeler_type)}</td><td>${esc(l.label)}</td><td>${typeof l.confidence === 'number' ? Math.round(l.confidence*100)+'%' : 'n/a'}</td><td>${esc(l.justification || '')}<br><span class="node-list">${esc((l.node_ids || []).join(', '))}</span></td><td><span class="badge ${tierClass(l.label_tier)}">${esc(l.label_tier || 'provisional')}</span></td></tr>`).join('')}</tbody></table>`
}
function renderDiffs(suggestions){
  $('#diffList').innerHTML=suggestions.map(s=>`<article class="diff-card"><p class="eyebrow">${esc(s.suggestion_type)} · ${esc(s.status)} · mock seed</p><h3>${esc(s.patch_id)}</h3><p>${esc(s.rationale)}</p><pre>${esc(JSON.stringify(s.proposed_diff, null, 2))}</pre></article>`).join('')
}
function metricDisplay(value, key){
  if (value === null || value === undefined) return 'NED';
  if (key==='fpr'||key.includes('proportion')||key.includes('coverage')||key.includes('mass')) return `${Math.round(value*100)}%`;
  return Number(value).toFixed(2);
}
function renderMetrics(snapshot){
  const metricEntries = Object.entries(snapshot.metrics || {});
  const graphEntries = Object.entries({coverage:snapshot.graph_health?.coverage, gray_zone_mass:snapshot.graph_health?.gray_zone_mass});
  $('#metricCards').innerHTML=[...metricEntries, ...graphEntries].map(([k,v])=>`<article class="metric-card ${v == null ? 'not-enough-data' : ''}"><div class="metric-label">${esc(k.replaceAll('_',' '))}</div><div class="metric-value">${metricDisplay(v,k)}</div><p>${v == null ? 'not enough data' : 'mock seed'}</p></article>`).join('');
  const n = snapshot.denominators || {};
  $('#metricNote').textContent = `${snapshot.warning || 'Metrics are mock-only until real labels exist.'} Denominators: truth_n=${n.truth_n ?? 0}, paired_truth_prediction_n=${n.paired_truth_prediction_n ?? 0}. Gold/platinum labels are the only reportable truth tiers.`;
}

loadSeedData().then(seed=>{
  drawGraph();
  renderGolden(seed.images, seed.labels);
  renderLabels(seed.labels);
  renderDiffs(seed.suggestions);
  renderMetrics(seed.metrics);
});
