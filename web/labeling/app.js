import { createClient } from './api.js';

const $ = (selector, root=document) => root.querySelector(selector);
const esc = value => String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const icons = {
  search:'<circle cx="10.5" cy="10.5" r="6.5"/><path d="m16 16 4 4"/>',
  chevron:'<path d="m7 10 5 5 5-5"/>',left:'<path d="m14 6-6 6 6 6"/>',right:'<path d="m10 6 6 6-6 6"/>',
  image:'<rect x="3" y="3" width="18" height="18" rx="3"/><circle cx="8" cy="8" r="1.5"/><path d="m3 17 5-5 4 4 4-6 5 7"/>',
  pin:'<path d="m15 3 6 6-4 1-4 6-2-2-6 6 6-6-2-2 6-4z"/>',
  assistant:'<rect x="3" y="5" width="18" height="14" rx="4"/><path d="M9 11h.01M15 11h.01M9 15h6M12 2v3"/>',
  sparkle:'<path d="m12 3 2.3 6.7L21 12l-6.7 2.3L12 21l-2.3-6.7L3 12l6.7-2.3z"/>',
  arrow:'<path d="M5 12h14m-5-5 5 5-5 5"/>',check:'<path d="m5 12 4 4L19 6"/>',
  checkcircle:'<circle cx="12" cy="12" r="9"/><path d="m8 12 3 3 5-6"/>',
  lock:'<rect x="5" y="10" width="14" height="11" rx="2"/><path d="M8 10V7a4 4 0 0 1 8 0v3m-4 4v3"/>',
  eye:'<path d="M2 12s3.5-7 10-7 10 7 10 7-3.5 7-10 7S2 12 2 12z"/><circle cx="12" cy="12" r="3"/>',
  route:'<circle cx="5" cy="5" r="2"/><circle cx="19" cy="5" r="2"/><circle cx="12" cy="19" r="2"/><path d="M5 7v3a3 3 0 0 0 3 3h8a3 3 0 0 0 3-3V7m-7 6v4"/>',
  history:'<path d="M3 11a9 9 0 1 1 2.5 7M3 4v7h7m2-4v6l4 2"/>',
  book:'<path d="M12 6S7 2 3 4v15c4-2 9 2 9 2s5-4 9-2V4c-4-2-9 2-9 2zm0 0v15"/>',
  flag:'<path d="M5 21V4m0 0c5-5 9 5 14 0v10c-5 5-9-5-14 0"/>',
  refresh:'<path d="M20 7a9 9 0 0 0-15-2L2 8m0-6v6h6m-4 9a9 9 0 0 0 15 2l3-3m0 6v-6h-6"/>',
  layers:'<path d="m12 3 10 5-10 5L2 8zm-9 10 9 5 9-5M3 18l9 5 9-5"/>',
  download:'<path d="M12 3v12m-5-5 5 5 5-5M4 16v5h16v-5"/>',
  upload:'<path d="M12 16V4m-5 5 5-5 5 5M4 16v5h16v-5"/>',
  close:'<path d="m6 6 12 12M6 18 18 6"/>',expand:'<path d="M8 3H3v5m13-5h5v5M3 16v5h5m13-5v5h-5"/>',
  person:'<circle cx="12" cy="8" r="4"/><path d="M4 21v-2a8 8 0 0 1 16 0v2"/>',
  chart:'<path d="M4 3v18h17M8 16v-5m5 5V7m5 9V3"/>',
  activity:'<path d="M2 12h5l3-8 4 16 3-8h5"/>',
  database:'<ellipse cx="12" cy="5" rx="8" ry="3"/><path d="M4 5v14c0 4 16 4 16 0V5M4 12c0 4 16 4 16 0"/>',
  info:'<circle cx="12" cy="12" r="9"/><path d="M12 11v6m0-10v.01"/>',
  inbox:'<path d="M4 4h16l2 12v4H2v-4zm-2 12h6l2 3h4l2-3h6"/>',
  bolt:'<path d="m13 2-9 12h7l-1 8 10-13h-7z"/>',
  link:'<path d="m10 13 4-4m-5 7-2 2a4 4 0 0 1-6-6l4-4a4 4 0 0 1 6 0m2 1 2-2a4 4 0 0 1 6 6l-4 4a4 4 0 0 1-6 0"/>'
};
function icon(name,size=16) { return `<svg class="icon" width="${size}" height="${size}" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${icons[name]||icons.layers}</svg>`; }
const pct = value => `${Math.round(Number(value||0)*100)}%`;
const date = value => value ? new Date(value).toLocaleString(undefined,{month:'short',day:'numeric',hour:'2-digit',minute:'2-digit'}) : '—';
const statusLabels = {open:'Open',in_review:'In review',approved:'Ready for gold',escalated:'Policy review',published:'Published'};
const statusColors = {open:'',in_review:'blue',approved:'mint',escalated:'amber',published:'purple'};
const typeLabels = {image:'Image',pin:'Pin',assistant:'AI assistant'};
const state={client:null,boot:null,items:[],allItems:[],selected:null,item:null,tab:'content',view:'review',filters:{q:'',type:'',category:'',status:'',project:''},drafts:{},events:[],busy:false,error:'',requestSequence:0,keys:new Map()};
let toastTimer,searchTimer;

function categoryName(id){return state.boot?.categories.find(c=>c.id===id)?.name||id;}
function policyName(id){return state.boot?.policies.find(p=>p.id===id)?.title||id;}
function badge(status){return `<span class="badge ${statusColors[status]||''}">${esc(statusLabels[status]||status)}</span>`;}
function currentDraft(){
  const item=state.item;if(!item)return null;
  if(!state.drafts[item.id])state.drafts[item.id]={labels:Object.fromEntries(item.targets.map(t=>[t.id,t.staged_label||item.independent_labels?.[t.id]||''])),rationale:'',policy_refs:(item.route?.nodes||[]).filter(id=>state.boot.policies.some(p=>p.id===id))};
  return state.drafts[item.id];
}
function mutationKey(operation,payload){const signature=operation+JSON.stringify(payload);if(!state.keys.has(signature))state.keys.set(signature,crypto.randomUUID());return state.keys.get(signature);}
function notify(message,error=false){const element=$('#toast');element.textContent=message;element.className=`toast show${error?' error':''}`;clearTimeout(toastTimer);toastTimer=setTimeout(()=>element.classList.remove('show'),5000);}
function option(value,label,selected){return `<option value="${esc(value)}" ${value===selected?'selected':''}>${esc(label)}</option>`;}
function filterSelect(name,label,options){return `<label class="filter"><span class="sr-only">${label}</span><select data-filter="${name}" aria-label="${label}" data-testid="filter-${name}">${options.map(([value,title])=>option(value,title,state.filters[name])).join('')}</select>${icon('chevron',12)}</label>`;}
function button(action,text,cls='btn',ico='',extra=''){return `<button type="button" class="${cls}" data-action="${action}" ${extra}>${ico?icon(ico,14):''}${text}</button>`;}
function noItems(title,description,ico='inbox'){return `<div class="empty-state">${icon(ico,32)}<h3>${esc(title)}</h3><p>${esc(description)}</p></div>`;}

function renderShell(){
  const user=state.boot.viewer;
  $('#app').innerHTML=`<header class="topbar"><a href="#review" class="brand" aria-label="RUSH labeling home"><span class="brand-mark">R</span><span class="brand-name">RUSH<small>LABELING WORKSPACE</small></span></a><nav class="nav" aria-label="Workspace navigation">${[['review','Review queue'],['gold','Golden sets'],['insights','Policy insights'],['activity','Activity']].map(([id,label])=>`<button type="button" data-view="${id}" class="${state.view===id?'active':''}" aria-current="${state.view===id?'page':'false'}">${label}</button>`).join('')}</nav><div class="topbar-end"><span class="demo-badge">Synthetic examples</span><div class="user" title="${esc(user.name)} · ${esc(user.role)} reviewer"><span class="avatar">${esc(user.name.split(' ').map(w=>w[0]).join('').slice(0,2))}</span><span><strong>${esc(user.name)}</strong><small>${esc(user.role)} reviewer</small></span></div></div></header><main class="main" id="main-content"></main>`;
  renderPage();
}
function setNav(){document.querySelectorAll('[data-view]').forEach(button=>{const active=button.dataset.view===state.view;button.classList.toggle('active',active);button.setAttribute('aria-current',active?'page':'false');});}
function renderPage(){setNav();if(state.view==='review')renderReview();else if(state.view==='gold')renderGold();else if(state.view==='insights')renderInsights();else renderActivity();}
function pageHead(title,subtitle,eyebrow,actions=''){return `<div class="page-head"><div><p class="eyebrow">${eyebrow}</p><h1>${title}</h1><p class="lede">${subtitle}</p></div><div class="head-actions">${actions}</div></div>`;}
function publishButton(){const ready=state.boot.stats.approved;return button('publish',`Publish epoch ${icon('arrow',14)}`,'btn btn-primary','',`data-testid="publish-epoch" ${!ready?'disabled title="Approve a training item before publishing a new epoch"':''}`);}
function renderReview(){
  const stats=state.boot.stats;
  $('#main-content').innerHTML=pageHead('Review queue','Turn the hardest decisions into better gold.','HUMAN EXPERTISE · CONTINUOUS LEARNING',button('import','Import items','btn btn-quiet','upload')+button('gold',`Epoch ${state.boot.epoch}`,'btn','layers')+publishButton())+
    `<div class="toolbar">${filterSelect('project','Filter by project',[['','All projects'],...state.boot.projects.map(p=>[p,p])])}${filterSelect('category','Filter by category',[['','All policy areas'],...state.boot.categories.map(c=>[c.id,c.name])])}${filterSelect('type','Filter by content type',[['','All content'],['assistant','AI assistant'],['pin','Pins'],['image','Images']])}${filterSelect('status','Filter by review status',[['','All statuses'],...Object.entries(statusLabels)])}<div class="counter-pills"><span class="counter-pill"><span class="dot blue"></span><b>${stats.open}</b> to review</span><span class="counter-pill"><span class="dot amber"></span><b>${stats.escalated}</b> with Policy</span><span class="counter-pill"><span class="dot"></span><b>${stats.approved}</b> ready for gold</span></div></div>
    <div class="workspace"><section class="panel queue-panel" aria-label="Ranked item queue"><div class="panel-header queue-header"><div class="queue-title"><h2>Prioritized for review <span id="queue-count" class="inline-count">${state.items.length}</span></h2><span>${icon('bolt',11)} APO rank</span></div><div class="search">${icon('search',14)}<input type="search" id="queue-search" aria-label="Search review items" data-testid="queue-search" value="${esc(state.filters.q)}" placeholder="Search items or IDs"><span class="small-muted">/</span></div></div><div class="queue-list" id="queue-list" aria-live="polite"></div><div class="queue-footer"><span>Upstream review priority</span><span><kbd>J</kbd> <kbd>K</kbd> to move</span></div></section><section class="panel content-panel" id="content-panel" aria-label="Original content and policy"></section><section class="panel decision-panel" id="decision-panel" aria-label="Human review decisions"></section></div>`;
  renderQueue();renderItem();
}
function renderQueue(){
  const container=$('#queue-list');if(!container)return;
  $('#queue-count').textContent=state.items.length;
  container.innerHTML=state.items.length?state.items.map((item,index)=>{
    const image=item.content?.image||item.thumbnail;
    return `<button class="queue-item ${item.id===state.selected?'selected':''}" data-item="${esc(item.id)}" type="button" aria-label="Review ${esc(item.id)} ${esc(item.title)}" aria-pressed="${item.id===state.selected}"><div class="queue-item-top"><span class="queue-type">${icon(item.type,11)}${typeLabels[item.type]||esc(item.type)}</span><span class="queue-score" title="Imported human-review priority score; not a probability">${icon('bolt',10)}${Number(item.priority?.score||0).toFixed(3)}</span></div><div class="queue-card">${image?`<img class="queue-thumb" src="${esc(image)}" alt="" loading="lazy">`:`<span class="queue-thumb queue-thumb-placeholder">${icon(item.type,22)}</span>`}<div><h3>${esc(item.title)}</h3><p>${esc(categoryName(item.category))}${item.split==='holdout'?icon('lock',10):''}</p></div></div><div class="queue-item-bottom"><span class="reason">${esc(item.priority?.reason||'Coverage sample')}</span>${badge(item.status)}</div></button>`;
  }).join(''):noItems('No matching items','Try another policy area, content type or search.');
}
function renderItem(){
  if(!state.item){$('#content-panel').innerHTML=noItems('Select an item','Choose an example from the queue to begin.','eye');$('#decision-panel').innerHTML=noItems('Your judgment matters','Review the original context, then record each decision.','checkcircle');return;}
  const item=state.item;
  $('#content-panel').innerHTML=`<div class="content-heading"><div class="content-kicker"><span>${icon(item.type,13)}</span>${esc(item.id)}${badge(item.status)}<span class="push"></span><button class="icon-btn" data-action="previous" aria-label="Previous item" title="Previous item (K)" ${state.items.findIndex(x=>x.id===state.selected)<=0?'disabled':''}>${icon('left',15)}</button><button class="icon-btn" data-action="next" aria-label="Next item" title="Next item (J)" data-testid="next-item" ${state.items.findIndex(x=>x.id===state.selected)>=state.items.length-1?'disabled':''}>${icon('right',15)}</button></div><h2>${esc(item.title)}</h2><div class="content-meta"><span>${esc(item.project)}</span><span>${esc(typeLabels[item.type])}</span><span>Policy v${esc(item.policy_version)}</span><span>Revision ${item.revision}</span><span>${item.assignee ? (item.assignee===state.boot.viewer.id ? 'Claimed by you' : 'Assigned to another reviewer') : button('claim','Claim review','link-btn')}</span></div></div><div class="tabs" role="tablist" aria-label="Item context">${[['content','Original content','eye'],['policy','Policy context','book'],['history','History','history']].map(([id,title,ico])=>`<button role="tab" type="button" data-tab="${id}" aria-selected="${state.tab===id}" class="${state.tab===id?'active':''}">${icon(ico,12)}${title}${id==='history'?`<span class="tab-count">${item.events?.length||0}</span>`:''}</button>`).join('')}</div><div class="scroll-body content-body" id="content-body">${state.tab==='content'?renderOriginal(item):state.tab==='policy'?renderPolicy(item):renderHistory(item.events)}</div>`;
  renderDecision();
}
function renderRoute(item){const nodes=item.route?.nodes||[];return `<div class="route-strip">${icon('route',15)}<div class="route-copy"><div class="route-title">Current policy route</div><div class="route-path">${nodes.length?nodes.map(id=>esc(policyName(id))).join(' <span class="small-muted">·</span> '):'Broad policy review'}</div><div class="route-details">${(item.route?.current_route_changed||item.route?.trace_stale)?'Human correction recorded':item.route?.fallback?'Fallback · review all relevant areas':'High-recall routing'} · ${esc(item.route?.version||'Unversioned')}</div></div>${button('route','Adjust','link-btn')}</div>`;}
function contentImage(image,title){return `<div class="visual-wrap"><img src="${esc(image)}" alt="${esc(title)}"><button type="button" class="visual-expand" data-action="expand-image" data-image="${esc(image)}" aria-label="Enlarge content image">${icon('expand',15)}</button></div>`;}
function renderOriginal(item){
  let content=renderRoute(item);
  if(item.split==='holdout')content+=`<div class="protected-note">${icon('lock',14)}Protected evaluation item. Review is allowed; optimization publication is blocked.</div>`;
  if(item.type==='assistant') {
    content+=`<div class="section-label">Conversation context<span class="muted">Input and output judged separately</span></div><div class="conversation">${(item.content.messages||[]).map((message,index)=>`<article class="message ${message.role==='assistant'?'assistant':''}"><span class="message-avatar">${icon(message.role==='assistant'?'assistant':'person',16)}</span><div class="message-main"><div class="message-author">${message.role==='assistant'?'AI assistant':'User'}<span class="badge ${message.role==='assistant'?'mint':''}">${message.role==='assistant'?'OUTPUT':'INPUT'}</span></div><div class="message-bubble">${message.image?`<img class="message-image" src="${esc(message.image)}" alt="User-provided attachment">`:index===0&&item.content.image?`<img class="message-image" src="${esc(item.content.image)}" alt="Illustrative attachment">`:''}${esc(message.text).replace(/\n/g,'<br>')}</div></div></article>`).join('')}</div>`;
    content+=`<div class="content-context"><strong>Review context</strong>${esc(item.description||item.content.description||'Assess the user request and assistant response independently under the selected policy.')}</div>`;
  } else {
    const query=item.content.query||item.content.search_query;
    if(query)content+=`<div class="query-card"><small>Search query</small><p>${icon('search',13)} ${esc(query)}</p></div>`;
    content+=`<div class="section-label">${item.type==='pin'?'Pin content':'Original image'}<span class="muted">Full context available</span></div><article class="pin-card">${item.content.image?contentImage(item.content.image,item.content.title||item.title):''}<div class="pin-copy"><h3>${esc(item.content.title||item.title)}</h3><p>${esc(item.content.description||item.description)}</p><div class="pin-source">${icon('link',12)}${esc(item.content.source||item.source?.name||'Synthetic example')}</div></div></article>`;
    if(item.description&&item.description!==item.content.description)content+=`<p class="neutral-note">${esc(item.description)}</p>`;
  }
  content+=renderPriority(item);
  return content;
}
function renderPriority(item){
  const p=item.priority||{},c=p.components||{};
  const value=(v,format='score')=>v===undefined||v===null?'Not supplied':format==='fraction'?pct(v):format==='count'?esc(v):Number(v).toFixed(3);
  const signals=[
    ['Review priority',p.score,'score','Imported upstream score used to order the human-review queue.'],
    ['Anchor value',p.anchor_value??c.anchor_value??p.anchor??p.policy_anchor_score,'score','Value as a policy-learning anchor. Separate from review priority.'],
    ['SME agreement',p.sme_agreement??c.sme_agreement??p.sme_fraction,'fraction','Model ↔ human: fraction of judges matching the expert label.'],
    ['Panel consensus',p.panel_consensus??c.panel_consensus??p.consensus_fraction??p.consensus,'fraction','Model ↔ model: fraction agreeing on the modal label, independent of gold.'],
    ['Mean |g|',p.mean_grad??c.mean_abs_gradient??p.grad_magnitude??p.gradient,'score','Heuristic from self-reported confidence and agreement with gold.'],
    ['Boundary rate',p.boundary_rate??c.boundary_rate,'fraction','Fraction of judges flagging a documented boundary case.'],
    ['Human confirmations',p.human_confirmations??c.human_confirmations??p.sme_confirmations,'count','Repeated expert confirmation reduces upstream review priority.'],
    ['Disagreement signal',p.disagreement,'score','Imported disagreement score; not assumed to equal 1 − SME agreement.']
  ];
  const synthetic=/synthetic|example/i.test(`${item.source?.name||''} ${item.source?.run_id||''}`);
  return `<details class="why-item" data-testid="why-this-item"><summary><span>${icon('bolt',13)}Why this item</span><span class="why-score">Review score ${value(p.score)} ${icon('chevron',12)}</span></summary><div class="why-body"><p class="why-reason">${esc(p.reason||'Selected for coverage')} ${p.tier?`<span class="badge amber">${esc(p.tier)}</span>`:''}</p><p class="why-intro">Human re-adjudication priority and APO anchor value answer different questions. The queue preserves the rank supplied upstream.${c.target_id?` These signals describe the <strong>${esc(c.target_id)}</strong> target${c.judge_count?` across ${esc(c.judge_count)} judges`:''}.`:''}</p><div class="signal-detail-grid">${signals.map(([name,v,format,description])=>`<div class="signal-detail"><div><span>${esc(name)}</span><strong class="${v===undefined||v===null?'not-supplied':''}">${value(v,format)}</strong></div><p>${esc(description)}</p></div>`).join('')}</div>${c.formula?`<details class="signal-formula"><summary>Imported scoring formula ${icon('chevron',11)}</summary><p>${esc(c.formula)}</p><p>Base importance ${value(c.i_base)} · confirmation confidence ${value(c.human_confidence??c.p_human,'fraction')}${c.human_priority!==undefined?` · raw score ${value(c.human_priority)}`:''}</p><p>The confirmation confidence is a repeat-review heuristic, separate from SME agreement.</p></details>`:''}<p class="signal-caveat">Mean |g| is a heuristic, not a calibrated error probability. Human confirmations can fade priority after an upstream refresh; this workspace does not recalculate the rank.</p><p class="signal-source">${synthetic||c.source_kind==='synthetic'?'Synthetic example signals':'Imported upstream signals'} · ${esc(item.source?.run_id||item.source?.name||'Source unavailable')}</p></div></details>`;
}
function renderPolicy(item){
  const routes=new Set(item.route?.nodes||[]),trace=item.guidance_visible?item.route?.trace:null;
  const referenced=new Set([...(trace?.served_ids||[]),...(trace?.cited_ids||[])]);
  const policies=state.boot.policies.filter(p=>routes.has(p.id)||referenced.has(p.id)||p.category===item.category);
  return `<p class="policy-explainer">Policy owns the interpretation of these rules. Expand the route if another area may apply, or escalate an unresolved boundary.</p>${renderGraphInspector(item)}<div class="section-label">Rules available for review<span class="muted">Current catalog</span></div><div class="policy-list">${policies.map(p=>`<article class="policy-card" id="policy-${esc(p.id)}"><div class="policy-title">${icon('book',14)}${esc(p.title)}</div><p>${esc(p.body).replace(/\n/g,'<br>')}</p><div class="policy-foot"><span>${esc(p.id)} · v${esc(p.version)}${p.node_type?` · ${esc(p.node_type)}`:''}</span><span>${p.locked?`<span class="badge">${icon('lock',9)}Locked</span> `:''}${routes.has(p.id)?'<span class="badge blue">Current route</span>':''}</span></div>${p.body_sha256?`<div class="policy-provenance">Body ${esc(p.body_sha256.slice(0,12))}… · ${esc(p.provenance||'Imported')}</div>`:''}</article>`).join('')}</div>`;
}
function renderGraphInspector(item){
  const graph=state.boot.graph;
  if(!graph)return renderRoute(item);
  const catalog=new Map((graph.nodes||[]).map(node=>typeof node==='string'?[node,state.boot.policies.find(p=>p.id===node)||{id:node,title:node}]:[node.id,node]));
  const trace=item.guidance_visible?item.route?.trace:null,current=new Set(item.route?.nodes||[]);
  const sets={selected:new Set(trace?.selected_ids||[]),served:new Set(trace?.served_ids||[]),cited:new Set(trace?.cited_ids||[]),pinned:new Set(trace?.pinned_ids||[])};
  const unservedCitations=[...sets.cited].filter(id=>!sets.served.has(id)),missingPinned=[...sets.pinned].filter(id=>!sets.served.has(id));
  const traceDiagnostics=(unservedCitations.length?`<p class="trace-alert">Cited without being served: ${unservedCitations.map(esc).join(', ')}. The original evidence is retained for review.</p>`:'')+(missingPinned.length?`<p class="trace-alert">Pinned context missing from the recorded prompt: ${missingPinned.map(esc).join(', ')}.</p>`:'');
  const mismatch=!!(trace?.graph_hash&&graph.source_sha256&&trace.graph_hash!==graph.source_sha256);
  const focus=new Set(current);
  if(trace&&!mismatch)Object.values(sets).forEach(ids=>ids.forEach(id=>focus.add(id)));
  if(!focus.size)for(const node of catalog.values())if(node.category===item.category)focus.add(node.id);
  for(const id of [...focus]){let node=catalog.get(id),seen=new Set();while(node?.parent&&!seen.has(node.parent)){seen.add(node.parent);focus.add(node.parent);node=catalog.get(node.parent);}}
  const nodes=[...focus].map(id=>catalog.get(id)||{id,title:'Unresolved policy node',node_type:'unknown'});
  const edges=(graph.edges||[]).filter(edge=>focus.has(edge.source_node_id)&&focus.has(edge.target_node_id));
  const chips=ids=>ids.length?ids.map(id=>`<button class="policy-ref" data-policy="${esc(id)}">${esc(id)}</button>`).join(''):'<span class="small-muted">None</span>';
  const signalCards=[['Selected','Proposed narrow context',sets.selected,'amber'],['Served','Actually sent to the judge',sets.served,'blue'],['Cited','Referenced in AI reasoning',sets.cited,'mint'],['Pinned','Always included by the router',sets.pinned,'purple']];
  const history=trace?`<div class="trace-header"><span>${icon('history',12)}Historical judge run</span><span class="badge ${trace.source_kind==='synthetic'?'purple':'blue'}">${esc(trace.effective_mode||'Recorded')}</span></div><div class="route-counts">${signalCards.map(([title,meaning,ids,color])=>`<div class="route-count"><strong class="${color}">${ids.size}</strong><span>${title}<small>${meaning}</small></span></div>`).join('')}</div>${traceDiagnostics}${trace.selection_note?`<p class="trace-note">${esc(trace.selection_note)}</p>`:''}${trace.fallback_reason?`<p class="trace-alert">${icon('flag',12)}Fallback: ${esc(trace.fallback_reason)}</p>`:''}${mismatch?'<p class="trace-alert">The recorded graph hash differs from the current catalog. The diagram shows the current route; historical IDs are preserved below.</p>':''}${(item.route.current_route_changed||item.route.trace_stale)?'<p class="trace-alert">The current route was corrected after this run. The recorded selection, served context and citations remain unchanged.</p>':''}<details class="trace-details"><summary>Routing record and scores ${icon('chevron',12)}</summary><div class="trace-metadata"><div><small>Requested → effective mode</small><span>${esc(trace.requested_mode||'Unknown')} → ${esc(trace.effective_mode||'Unknown')}</span></div><div><small>Router / policy version</small><span>${esc(trace.router_version||'Unknown')} / ${esc(trace.policy_version||'Unknown')}</span></div><div><small>Configured recall target</small><span>${trace.recall_target==null?'Not supplied':pct(trace.recall_target)}</span></div><div><small>Retained score mass</small><span>${trace.retained_score_mass==null?'Not supplied':`${(100*trace.retained_score_mass).toFixed(1)}%`}</span></div></div><p class="signal-caveat">Routing scores are uncalibrated relevance estimates. The configured target and retained score mass are not measured recall.</p><div class="trace-id">Run ${esc(trace.run_id||'Unavailable')}<br>Recorded graph ${esc(trace.graph_hash?.slice(0,16)||'Unavailable')}${trace.graph_hash?'…':''}</div><div class="trace-sets">${signalCards.map(([title,meaning,ids])=>`<div><strong>${title}</strong><span>${chips([...ids])}</span></div>`).join('')}</div>${trace.decisions?.length?`<div class="routing-decisions"><div class="routing-decision heading"><span>Node / selection</span><span>Score p</span><span>Confidence</span></div>${trace.decisions.map(d=>`<div class="routing-decision"><span><strong>${esc(d.node_id)}</strong><small>${esc(d.choice)} · ${esc(d.why)}${d.pinned?' · pinned':''}</small></span><span>${d.p==null?'—':Number(d.p).toFixed(3)}</span><span>${d.confidence==null?'—':Number(d.confidence).toFixed(3)}</span></div>`).join('')}</div><p class="signal-caveat">Confidence reflects distance from the router’s decision boundary. It does not establish correctness or calibration.</p>`:''}</details>`:`<div class="trace-blind">${icon('lock',13)}${item.guidance_visible?'No recorded router trace was supplied for this item.':'Save your independent first pass to reveal the historical routing trace and AI citations.'}</div>`;
  return `<section class="policy-inspector" data-testid="policy-inspector"><header><div><h3>${icon('route',14)}Policy subgraph</h3><p>${esc(graph.graph_id)} · v${esc(graph.version)} · ${nodes.length} nodes in view</p></div><span class="badge">Versioned</span></header><div class="current-route"><span>Current review route</span><div>${current.size?chips([...current]):'<span class="small-muted">Broad review · no narrow route</span>'}</div>${button('route','Adjust','link-btn')}</div>${history}<details class="subgraph-view" ${nodes.length<=6?'open':''}><summary>${trace&&!mismatch?'Selected and served context':'Current policy context'}<span>${nodes.length} nodes ${icon('chevron',11)}</span></summary><div class="graph-visual">${renderSubgraph(nodes,edges,current,mismatch?null:sets,!!trace&&!mismatch)}</div><div class="graph-legend"><span><i class="graph-legend-line"></i>Arrows: child → parent</span><span>${trace&&!mismatch?'Node badges describe the recorded run':'Current route and supporting context'}</span></div><details class="edge-details"><summary>${edges.length} typed relationships ${icon('chevron',11)}</summary><div class="edge-list">${edges.length?edges.map(e=>`<div class="typed-edge"><div><code>${esc(e.source_node_id)}</code><span class="edge-type">${esc(e.edge_type.replace(/_/g,' '))}</span>${icon('arrow',11)}<code>${esc(e.target_node_id)}</code></div>${e.via&&e.via!=='parent'?`<p>${esc(e.via)}</p>`:''}</div>`).join(''):'<p class="small-muted">No relationships supplied between these nodes.</p>'}</div></details></details><div class="graph-provenance">Catalog SHA-256 ${esc(graph.source_sha256?.slice(0,16)||'Unavailable')}${graph.source_sha256?'…':''} · Rule bodies retain their own hashes</div></section>`;
}
function renderSubgraph(nodes,edges,current,sets,hasTrace){
  const lookup=new Map(nodes.map(n=>[n.id,n])),levels=new Map(),positions=new Map(),width=620,nodeWidth=180,nodeHeight=86;
  const depth=(node,seen=new Set())=>{if(!node.parent||!lookup.has(node.parent)||seen.has(node.id))return 0;seen.add(node.id);return 1+depth(lookup.get(node.parent),seen);};
  for(const node of nodes){const d=depth(node);if(!levels.has(d))levels.set(d,[]);levels.get(d).push(node);}
  let row=0;
  for(const level of [...levels.keys()].sort((a,b)=>a-b)){
    const group=levels.get(level);
    for(let i=0;i<group.length;i+=3){const segment=group.slice(i,i+3),offset=(width-segment.length*nodeWidth-(segment.length-1)*18)/2;segment.forEach((node,col)=>positions.set(node.id,{x:offset+col*(nodeWidth+18),y:16+row*126}));row++;}
  }
  const lineMarkup=edges.filter(e=>e.edge_type==='subtype_of').map(e=>{
    const from=positions.get(e.source_node_id),to=positions.get(e.target_node_id);if(!from||!to)return '';
    const x1=from.x+nodeWidth/2,y1=from.y,x2=to.x+nodeWidth/2,y2=to.y+nodeHeight;
    return `<path d="M${x1} ${y1} C${x1} ${y1-22} ${x2} ${y2+22} ${x2} ${y2+4}" class="graph-edge" marker-end="url(#subgraph-arrow)"/>`;
  }).join('');
  const nodeMarkup=nodes.map(node=>{
    const {x,y}=positions.get(node.id),title=String(node.title||node.id),words=title.split(/\s+/),lines=[''];
    for(const word of words){if((lines[lines.length-1]+' '+word).trim().length>24&&lines.length<2)lines.push(word);else lines[lines.length-1]=(lines[lines.length-1]+' '+word).trim();}
    lines[1]=lines[1]?.length>26?lines[1].slice(0,24)+'…':lines[1];
    const tags=hasTrace?[sets.pinned.has(node.id)?'Pinned':null,sets.selected.has(node.id)?'Selected':null,sets.served.has(node.id)?'Served':null,sets.cited.has(node.id)?'Cited':null].filter(Boolean):[current.has(node.id)?'Current route':'Context'];
    const color=hasTrace&&sets.cited.has(node.id)?'cited':hasTrace&&sets.served.has(node.id)?'served':current.has(node.id)?'current':'';
    return `<g class="graph-node ${color}"><title>${esc(node.id)}: ${esc(title)}. ${esc(tags.join(', '))}</title><rect x="${x}" y="${y}" width="${nodeWidth}" height="${nodeHeight}" rx="9"/><text x="${x+12}" y="${y+19}" class="graph-node-id">${esc(node.id)}</text><text x="${x+nodeWidth-12}" y="${y+19}" class="graph-node-kind" text-anchor="end">${esc(node.node_type||'node')}</text>${lines.filter(Boolean).map((line,i)=>`<text x="${x+12}" y="${y+39+i*15}" class="graph-node-title">${esc(line)}</text>`).join('')}<text x="${x+12}" y="${y+73}" class="graph-node-tags">${esc(tags.join(' · ')||'Context')}</text></g>`;
  }).join('');
  return `<svg viewBox="0 0 ${width} ${Math.max(118,row*126)}" role="img" aria-label="Policy subgraph with ${nodes.length} nodes; hierarchy arrows point from child rules to parent sections"><defs><marker id="subgraph-arrow" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto"><path d="M0 0 L6 3 L0 6" fill="#56769a"/></marker></defs>${lineMarkup}${nodeMarkup}</svg>`;
}
function eventTitle(event){if(event.summary)return event.summary;const type=event.type||event.action||event.event_type||event.phase;return ({independent:'Independent first pass saved',claim:'Review claimed',route:'Policy route adjusted',published:'Gold label published',epoch:'Golden-set epoch published',rerun:'APO handoff queued',import:'New candidate imported',adjudication:event.decision==='approve'?'Expert decision approved':event.decision==='escalate'?'Escalated to Policy':'Expert review saved'})[type]||String(type||'Review event').replace(/_/g,' ');}
function actorName(event){return typeof event.actor==='object'?event.actor?.name||event.actor?.id:event.actor||event.reviewer_name||event.reviewer_id||'Reviewer';}
function renderHistory(events=[]){return events.length?`<div class="timeline">${[...events].sort((a,b)=>String(b.created_at||b.timestamp).localeCompare(String(a.created_at||a.timestamp))).map(event=>`<article class="timeline-event"><h3>${esc(eventTitle(event))}${event.epoch?` <span class="badge purple">Epoch ${esc(event.epoch)}</span>`:''}</h3><div class="event-meta">${esc(actorName(event))} · ${esc(date(event.created_at||event.timestamp))}${event.item_id?` · <button class="row-link" data-item="${esc(event.item_id)}">${esc(event.item_id)}</button>`:''}</div>${event.rationale||event.reason?`<p>${esc(event.rationale||event.reason)}</p>`:''}${event.labels?`<div class="event-labels">${Object.entries(event.labels).map(([target,label])=>`<span>${esc(target)}: ${esc(label)}</span>`).join('')}</div>`:''}${event.ai_visible===false?'<p class="small-muted">Recorded before AI guidance was revealed</p>':''}</article>`).join('')}</div>`:noItems('A clean review history','Each human decision, routing change and publication is recorded here.','history');}

function renderDecision(){
  if(!state.item||!$('#decision-panel'))return;
  const item=state.item,draft=currentDraft(),visible=item.guidance_visible,canAdjudicate=['sme','policy'].includes(state.boot.viewer.role),protectedSplit=item.split==='holdout';
  $('#decision-panel').innerHTML=`<div class="panel-header decision-header">${icon('checkcircle',16)}<h2>Your decision</h2><span class="step">${icon(visible?'eye':'lock',11)}${visible?'Adjudication':'Independent review'}</span></div><div class="scroll-body decision-body">${state.error?`<div class="error-inline" role="alert">${esc(state.error)}</div>`:''}<div class="review-mode ${visible?'':'blind'}">${icon(visible?'checkcircle':'lock',14)}<span>${visible?'Your independent first pass is recorded. Compare the evidence and finalize each label.':'AI guidance and current gold are hidden. Save your own labels first to keep this review independent.'}</span></div>${item.targets.map(target=>renderTarget(target,item,draft)).join('')}<div class="rationale-wrap"><label class="rationale-label" for="rationale">Decision rationale <small>Required</small></label><textarea id="rationale" data-testid="decision-rationale" placeholder="What in the content and policy supports your decision?">${esc(draft.rationale)}</textarea><div class="policy-select">${icon('book',11)}Citing ${draft.policy_refs.length} policy node${draft.policy_refs.length===1?'':'s'} ${button('policy-refs','Choose references','link-btn')}</div></div></div><div class="decision-footer">${!visible?button('first-pass','Save first pass & reveal','btn btn-primary','eye',`data-testid="save-first-pass" ${state.busy?'disabled':''}`):button('approve','Approve for gold','btn btn-primary','check',`data-testid="approve-review" ${state.busy||!canAdjudicate||protectedSplit?'disabled':''} title="${protectedSplit?'Protected evaluation data is excluded from optimization':!canAdjudicate?'An SME or Policy reviewer must adjudicate':'Stage approved labels for the next gold epoch'}"`)}${visible?`<div class="secondary-actions">${button('save-review','Save review','btn btn-quiet','',state.busy||!canAdjudicate?'disabled':'')}${button('escalate','Escalate to Policy','btn btn-amber','flag',`data-testid="escalate-review" ${state.busy||!canAdjudicate?'disabled':''}`)}</div>`:''}<div class="footer-note"><span>${visible?'Approval is staged until the next epoch is published.':'First-pass labels are preserved in the audit history.'}</span>${state.busy?'<span>Saving…</span>':''}</div></div>`;
}
function renderTarget(target,item,draft){
  const ai=target.ai,gold=target.gold;const selected=draft.labels[target.id];
  return `<section class="target-block"><div class="target-heading"><h3>${esc(target.name)}</h3><span class="target-hint">${target.id==='input'?'USER ABUSE':target.id==='output'?'AI ABUSE':target.id==='relevance'?'RELEVANCE':'CONTENT'}</span></div><div class="label-options" role="group" aria-label="${esc(target.name)} label">${target.choices.map(choice=>`<button type="button" class="label-option ${selected===choice?'selected':''} ${choice==='Violation'?'is-violation':choice==='Uncertain'?'is-uncertain':''}" data-label-target="${esc(target.id)}" data-label="${esc(choice)}" aria-pressed="${selected===choice}" aria-label="${esc(target.name)}: ${esc(choice)}">${selected===choice?icon('check',11):''}${esc(choice)}</button>`).join('')}</div>${item.guidance_visible?`<div class="comparison"><div><small>AI label <span class="confidence" title="${esc(ai?.confidence_kind||'Model confidence')} model score">${ai?pct(ai.confidence):'—'}</span></small><strong class="${ai?.label!==gold?.label?'label-different':'label-same'}">${esc(ai?.label||'Not available')}</strong></div><div><small>Current gold <span>r${esc(gold?.revision||0)}</span></small><strong>${esc(gold?.label||'Unlabeled')}</strong></div></div>${ai?`<details class="evidence" ${target.id==='input'||item.targets.length===1?'open':''}><summary>${icon('sparkle',11)}AI decision rationale ${icon('chevron',12)}</summary><p>${esc(ai.rationale)}</p>${ai.evidence?`<blockquote>“${esc(ai.evidence)}”</blockquote>`:''}<div>${(ai.policy_refs||[]).map(id=>`<button type="button" class="policy-ref" data-policy="${esc(id)}">${esc(id)}</button>`).join('')}</div><div class="model-info">${esc(ai.model||'Example judge')} · prompt ${esc(ai.prompt_version||'—')}<br>${esc(ai.confidence_kind||'Uncalibrated')} confidence · advisory</div></details>`:''}`:''}</section>`;
}

function renderGold(){const b=state.boot,s=b.stats;$('#main-content').innerHTML=pageHead('Golden sets','Approved decisions, reproducible snapshots, and a clear path back to APO.','VERSIONED HUMAN GROUND TRUTH',button('rerun','Queue APO handoff','btn','bolt','data-testid="queue-apo"')+publishButton())+`<div class="overview-grid"><article class="overview-card"><span class="overview-label">${icon('layers',14)}Current epoch</span><strong>${b.epoch}</strong><p>Immutable snapshot of local approved gold</p></article><article class="overview-card"><span class="overview-label">${icon('checkcircle',14)}Ready to publish</span><strong>${s.approved}</strong><p>Expert decisions staged for the next epoch</p></article><article class="overview-card"><span class="overview-label">${icon('database',14)}Gold items</span><strong>${s.gold_items}</strong><p>Training items eligible for the next snapshot</p></article></div><section class="wide-panel"><header class="wide-panel-head"><div><h2>Epoch history</h2><p>Every export keeps its item, target, policy and content identity.</p></div><span class="badge mint">Local gold store</span></header><div class="table-scroll"><table><thead><tr><th>Epoch</th><th>Published</th><th>Items</th><th>Changes</th><th>Snapshot digest</th><th><span class="sr-only">Actions</span></th></tr></thead><tbody>${[...b.epochs].sort((a,c)=>c.epoch-a.epoch).map(e=>`<tr><td><strong>Epoch ${e.epoch}</strong>${e.epoch===b.epoch?' <span class="badge mint">Current</span>':''}</td><td>${esc(date(e.created_at))}<small>${esc(e.created_by)}</small></td><td>${e.item_count}</td><td>${e.changed_count}</td><td><code title="${esc(e.digest)}">${esc(e.digest?.slice(0,14))}…</code></td><td><div class="table-actions">${button('export','Export JSON','btn btn-small btn-quiet','download',`data-epoch="${e.epoch}"`)}${button('rerun','APO handoff','btn btn-small','bolt',`data-epoch="${e.epoch}"`)}</div></td></tr>`).join('')}</tbody></table></div></section><section class="wide-panel"><header class="wide-panel-head"><div><h2>APO handoffs</h2><p>Requests pin an exact epoch and digest. An external worker must consume them.</p></div><span class="badge amber">Connector required</span></header>${b.runs.length?`<div class="table-scroll"><table><thead><tr><th>Request</th><th>Gold epoch</th><th>Created</th><th>Status</th></tr></thead><tbody>${b.runs.map(r=>`<tr><td><strong>${esc(r.id)}</strong><small>Digest ${esc(r.digest?.slice(0,14))}…</small></td><td>Epoch ${r.epoch}</td><td>${esc(date(r.created_at))}</td><td><span class="badge amber">Awaiting connector</span></td></tr>`).join('')}</tbody></table></div>`:noItems('Ready for the next learning cycle','Publish your approved labels, then queue an APO handoff pinned to that snapshot.','bolt')}</section><div class="integration-grid"><article class="integration-item">${icon('database',23)}<div><h3>Production golden sets <span class="badge">Not connected</span></h3><p>This workspace publishes local, versioned gold. Exported snapshots are ready for a production connector to validate and write back.</p></div></article><article class="integration-item">${icon('route',23)}<div><h3>APO and Prompt Manager <span class="badge">Separate services</span></h3><p>APO requests stay queued until connected. Prompt evaluation, final Policy approval and production promotion remain downstream.</p></div></article></div>`;}
function renderInsights(){
  const escalated=state.allItems.filter(i=>i.status==='escalated'),fallback=state.allItems.filter(i=>i.route?.fallback),categories=state.boot.categories;const counts=categories.map(c=>({...c,count:state.allItems.filter(i=>i.category===c.id).length})),max=Math.max(1,...counts.map(c=>c.count));
  $('#main-content').innerHTML=pageHead('Policy insights','Let difficult examples point to the rules that need a clearer interpretation.','EVIDENCE FOR POLICY DEVELOPMENT',button('refresh','Refresh','btn btn-quiet','refresh'))+`<div class="insights-grid"><article class="insight-card"><div class="section-label">${icon('flag',14)}Awaiting Policy</div><div class="count" style="color:var(--amber)">${escalated.length}</div><p>Escalations requiring an expert interpretation before becoming gold.</p></article><article class="insight-card"><div class="section-label">${icon('route',14)}Broad review</div><div class="count" style="color:var(--blue)">${fallback.length}</div><p>Examples without a confident policy route. Check for missing areas or new leaves.</p></article><article class="insight-card"><div class="section-label">${icon('book',14)}Policy areas</div><div class="count" style="color:var(--mint)">${categories.length}</div><p>Areas represented by the example dataset. Coverage is distinct from label volume.</p></article></div><section class="wide-panel"><header class="wide-panel-head"><div><h2>Policy escalation queue</h2><p>Resolve the interpretation, then return the item for adjudication.</p></div><span class="badge amber">Policy ownership</span></header>${escalated.length?`<div class="table-scroll"><table><thead><tr><th>Example</th><th>Policy area</th><th>Why it matters</th><th>Owner</th><th></th></tr></thead><tbody>${escalated.map(i=>`<tr><td><button class="row-link" data-item="${esc(i.id)}">${esc(i.title)}</button><small>${esc(i.id)}</small></td><td>${esc(categoryName(i.category))}</td><td>${esc(i.priority?.reason)}</td><td>${esc(i.escalation_owner||'Policy · '+categoryName(i.category))}</td><td><button class="btn btn-small" data-item="${esc(i.id)}">Review ${icon('arrow',12)}</button></td></tr>`).join('')}</tbody></table></div>`:noItems('No unresolved Policy escalations','Escalate an ambiguous review to collect its context, evidence and decision history here.','checkcircle')}</section><section class="wide-panel"><header class="wide-panel-head"><div><h2>Example coverage by policy area</h2><p>Counts reflect this synthetic dataset, not measured production coverage.</p></div></header><div class="wide-panel-body coverage-list">${counts.map(c=>`<div class="coverage-row"><div class="name"><span>${esc(c.name)}</span><strong>${c.count}</strong></div><div class="signal-bar"><span style="width:${Math.round(c.count/max*100)}%"></span></div></div>`).join('')}</div></section><div class="integration-item">${icon('info',22)}<div><h3>Evidence informs policy. Policy determines meaning.</h3><p>Recurring disagreements may indicate a label error, unclear boundary or emerging pattern. A Policy decision is distinct from a label correction. Re-review affected gold when the meaning of a rule changes.</p></div></div>`;
}
function renderActivity(){
  $('#main-content').innerHTML=pageHead('Activity','A durable trail from the first human judgment to the next gold epoch.','AUDIT AND TRACEABILITY',button('refresh','Refresh activity','btn btn-quiet','refresh'))+`<section class="wide-panel activity-wrap"><header class="wide-panel-head"><div><h2>Workspace audit trail</h2><p>Reviewer identity, rationale, policy version and publication history.</p></div><span class="badge">${state.events.length} events</span></header><div class="wide-panel-body">${renderHistory(state.events)}</div></section>`;
}

async function refreshData(){
  const [boot,all,queue]=await Promise.all([state.client.bootstrap(),state.client.items(),state.client.items(state.filters)]);
  state.boot=boot;state.allItems=all.items;state.items=queue.items;
  if(state.view==='activity')state.events=(await state.client.events()).events;
}
async function selectItem(id){
  const sequence=++state.requestSequence;
  state.selected=id;state.error='';state.tab='content';
  const item=await state.client.item(id);
  if(sequence!==state.requestSequence)return;
  state.item=item.item||item;
  if(state.view!=='review'){state.view='review';location.hash='review';renderPage();}else{renderQueue();renderItem();}
}
async function moveItem(delta){const index=state.items.findIndex(i=>i.id===state.selected);const next=state.items[index+delta];if(next)await selectItem(next.id);}
async function loadQueue(){const sequence=++state.requestSequence;const result=await state.client.items(state.filters);if(sequence!==state.requestSequence)return;state.items=result.items;if(!state.items.some(item=>item.id===state.selected)){if(state.items.length){await selectItem(state.items[0].id);return;}state.selected=null;state.item=null;renderItem();}renderQueue();}
async function navigate(view){state.view=view;state.error='';if(location.hash!==`#${view}`)history.replaceState(null,'',`#${view}`);if(view==='activity')state.events=(await state.client.events()).events;renderPage();}
async function submitReview(decision){
  const item=state.item,draft=currentDraft();
  if(!item||state.busy)return;
  for(const target of item.targets)if(!draft.labels[target.id]){state.error=`Choose a label for ${target.name}.`;renderDecision();return;}
  if(!draft.rationale.trim()){state.error='Add a short rationale so the next reviewer understands this decision.';renderDecision();$('#rationale')?.focus();return;}
  if(decision==='approve'&&Object.values(draft.labels).includes('Uncertain')){state.error='An uncertain decision needs Policy clarification. Choose “Escalate to Policy.”';renderDecision();return;}
  const payload={base_revision:item.revision,phase:item.guidance_visible?'adjudication':'independent',decision,labels:{...draft.labels},rationale:draft.rationale,policy_refs:draft.policy_refs};
  state.busy=true;state.error='';renderDecision();
  try {
    const result=await state.client.review(item.id,payload,mutationKey(`review:${item.id}`,payload));state.item=result.item||result;
    await refreshData();
    const message=payload.phase==='independent'?'Independent review saved. AI guidance is now available.':decision==='approve'?'Decision approved. Ready for the next gold epoch.':decision==='escalate'?'Escalated to Policy with your context and rationale.':'Review saved with a complete audit history.';
    notify(message);renderReview();
  }catch(error){state.error=error.message;notify(error.message,true);}
  finally{state.busy=false;renderDecision();}
}

function openDialog(title,description,body,footer,cls=''){
  const dialog=$('#modal');dialog.className=cls;dialog.innerHTML=`<div class="dialog-head"><div><h2 id="dialog-title">${title}</h2><p>${description}</p></div><button class="icon-btn" data-action="close-modal" aria-label="Close dialog">${icon('close',18)}</button></div><div class="dialog-body">${body}<div id="dialog-error" role="alert"></div></div><div class="dialog-foot">${button('close-modal','Cancel','btn btn-quiet')}${footer}</div>`;
  dialog.showModal();
}
function dialogError(message){const element=$('#dialog-error');if(element)element.innerHTML=`<div class="error-inline" style="margin-top:12px">${esc(message)}</div>`;}
function closeDialog(){$('#modal').close();}
function openPublish(){
  const count=state.boot.stats.approved;
  openDialog(`Publish epoch ${state.boot.epoch+1}`,`Create a reproducible snapshot from ${count} approved item${count===1?'':'s'}.`,`<div class="dialog-note">${icon('checkcircle',15)} Approved decisions become current local gold. Human history is retained, protected evaluation data is excluded, and the prior epoch remains unchanged.</div><label class="field"><span>Publication note</span><textarea id="publish-note" placeholder="What changed in this golden-set update?">Approved expert corrections from the review queue.</textarea></label><p class="small-muted">This publishes to the local golden store. Production writeback requires a configured connector.</p>`,button('confirm-publish',`Publish epoch ${state.boot.epoch+1}`,'btn btn-primary','layers',`data-testid="confirm-publish" ${!count?'disabled':''}`));
}
function openRerun(epoch=state.boot.epoch){
  openDialog('Queue an APO handoff','Pin the next optimization request to an exact published gold snapshot.',`<div class="dialog-note amber">This creates a durable handoff request. It will remain <strong>awaiting connector</strong> until an external APO worker consumes it. No experiment is launched here.</div><label class="field"><span>Golden-set epoch</span><select id="rerun-epoch" aria-label="APO golden-set epoch">${[...state.boot.epochs].sort((a,b)=>b.epoch-a.epoch).map(e=>option(String(e.epoch),`Epoch ${e.epoch} · ${e.item_count} items`,String(epoch))).join('')}</select></label><label class="field"><span>Request note</span><textarea id="rerun-note" placeholder="What should this run investigate?">Evaluate the latest expert corrections against the existing judge.</textarea></label>`,button('confirm-rerun','Queue handoff','btn btn-primary','bolt','data-testid="confirm-rerun"'));
}
function openRoute(refsOnly=false){
  const selected=new Set(refsOnly?currentDraft().policy_refs:state.item.route?.nodes||[]);
  const checks=state.boot.policies.map(p=>`<label class="check-row"><input type="checkbox" name="policy-node" value="${esc(p.id)}" ${selected.has(p.id)?'checked':''}><span><strong>${esc(p.title)}</strong><small>${esc(categoryName(p.category))} · ${esc(p.id)} · v${esc(p.version)}</small></span></label>`).join('');
  openDialog(refsOnly?'Cite the relevant policy':'Adjust the policy route',refsOnly?'Choose the policy nodes that support your human decision.':'Select every relevant area. Leaving all nodes unchecked requests broad review.',`<div style="max-height:300px;overflow:auto;margin-bottom:15px">${checks}</div>${refsOnly?'':`<label class="field"><span>Why does the route need to change?</span><textarea id="route-reason" placeholder="Name the missing area or explain why broader review is needed."></textarea></label>`}`,button(refsOnly?'confirm-refs':'confirm-route',refsOnly?'Use these references':'Apply route','btn btn-primary','check','data-testid="apply-route"'));
}
function openImport(){
  openDialog('Import review candidates','Add image, pin or assistant examples without overwriting existing review history.',`<label class="field"><span>Items JSON</span><textarea id="import-json" style="min-height:220px;font-family:ui-monospace,monospace" placeholder='{"items": [{"id": "new-example", ...}]}'></textarea><span class="hint">Use the item schema in docs/labeling-contract.md. Existing IDs are skipped. Imported items always begin as unlabeled review candidates.</span></label><label class="field"><span>Or choose a JSON file</span><input type="file" id="import-file" accept="application/json,.json"></label>`,button('confirm-import','Import candidates','btn btn-primary','upload'));
}
async function modalMutation(buttonElement,callback){
  buttonElement.disabled=true;
  try{await callback();}catch(error){dialogError(error.message);}finally{buttonElement.disabled=false;}
}
function downloadJson(value,name){const url=URL.createObjectURL(new Blob([JSON.stringify(value,null,2)],{type:'application/json'}));const link=document.createElement('a');link.href=url;link.download=name;document.body.append(link);link.click();link.remove();setTimeout(()=>URL.revokeObjectURL(url),2000);}

document.addEventListener('click',async event=>{
  const buttonElement=event.target.closest('button');if(!buttonElement||buttonElement.disabled)return;
  try {
    if(buttonElement.dataset.view){await navigate(buttonElement.dataset.view);return;}
    if(buttonElement.dataset.item){await selectItem(buttonElement.dataset.item);return;}
    if(buttonElement.dataset.tab){state.tab=buttonElement.dataset.tab;renderItem();return;}
    if(buttonElement.dataset.labelTarget){const target=buttonElement.dataset.labelTarget,label=buttonElement.dataset.label;currentDraft().labels[target]=label;state.error='';renderDecision();document.querySelector(`[data-label-target="${CSS.escape(target)}"][data-label="${CSS.escape(label)}"]`)?.focus();return;}
    if(buttonElement.dataset.policy){state.tab='policy';renderItem();document.getElementById('policy-'+buttonElement.dataset.policy)?.scrollIntoView({block:'nearest',behavior:'smooth'});return;}
    const action=buttonElement.dataset.action;
    if(action==='previous')return await moveItem(-1);
    if(action==='next')return await moveItem(1);
    if(action==='gold')return await navigate('gold');
    if(action==='claim'){const payload={base_revision:state.item.revision};state.item=await state.client.claim(state.item.id,payload,mutationKey('claim:'+state.item.id,payload));renderItem();notify('Review claimed. Other reviewers cannot overwrite your work.');return;}
    if(action==='first-pass'||action==='save-review')return await submitReview('save');
    if(action==='approve')return await submitReview('approve');
    if(action==='escalate')return await submitReview('escalate');
    if(action==='publish')return openPublish();
    if(action==='rerun')return openRerun(Number(buttonElement.dataset.epoch||state.boot.epoch));
    if(action==='route')return openRoute();
    if(action==='policy-refs')return openRoute(true);
    if(action==='import')return openImport();
    if(action==='close-modal')return closeDialog();
    if(action==='expand-image')return openDialog('Original image','Inspect the complete example without changing its review.',`<img src="${esc(buttonElement.dataset.image)}" alt="Enlarged original content">`,'','image-dialog');
    if(action==='refresh'){await refreshData();if(state.item)state.item=await state.client.item(state.item.id);renderPage();notify('Workspace is up to date.');return;}
    if(action==='export'){const epoch=Number(buttonElement.dataset.epoch);downloadJson(await state.client.exportEpoch(epoch),`rush-gold-epoch-${epoch}.json`);notify(`Epoch ${epoch} exported with its snapshot digest.`);return;}
    if(action==='confirm-publish')return await modalMutation(buttonElement,async()=>{const payload={expected_epoch:state.boot.epoch,note:$('#publish-note').value};const result=await state.client.publish(payload,mutationKey('publish',payload));closeDialog();await refreshData();if(state.item)state.item=await state.client.item(state.item.id);await navigate('gold');notify(`Epoch ${result.epoch} published. Previous snapshots are preserved.`);});
    if(action==='confirm-rerun')return await modalMutation(buttonElement,async()=>{const payload={epoch:Number($('#rerun-epoch').value),note:$('#rerun-note').value};await state.client.rerun(payload,mutationKey('rerun',payload));closeDialog();await refreshData();await navigate('gold');notify('APO handoff queued. It is awaiting a production connector.');});
    if(action==='confirm-refs'){currentDraft().policy_refs=[...document.querySelectorAll('[name="policy-node"]:checked')].map(x=>x.value);closeDialog();renderDecision();return;}
    if(action==='confirm-route')return await modalMutation(buttonElement,async()=>{const payload={base_revision:state.item.revision,nodes:[...document.querySelectorAll('[name="policy-node"]:checked')].map(x=>x.value),reason:$('#route-reason').value};if(!payload.reason.trim())throw new Error('Add a reason for the routing change.');state.item=await state.client.route(state.item.id,payload,mutationKey('route:'+state.item.id,payload));closeDialog();await refreshData();renderReview();notify('Policy route updated and recorded in history.');});
    if(action==='confirm-import')return await modalMutation(buttonElement,async()=>{let parsed;try{parsed=JSON.parse($('#import-json').value);}catch{throw new Error('Enter a valid JSON array or an object with an items array.');}const items=Array.isArray(parsed)?parsed:parsed.items;if(!Array.isArray(items))throw new Error('The JSON must contain an items array.');const result=await state.client.importItems(items,mutationKey('import',items));closeDialog();await refreshData();renderPage();notify(`${result.imported} candidates imported; ${result.skipped} existing IDs skipped.`);});
  }catch(error){notify(error.message,true);}
});
document.addEventListener('input',event=>{
  if(event.target.id==='rationale'&&state.item)currentDraft().rationale=event.target.value;
  if(event.target.id==='queue-search'){state.filters.q=event.target.value;clearTimeout(searchTimer);searchTimer=setTimeout(()=>loadQueue().catch(error=>notify(error.message,true)),160);}
});
document.addEventListener('change',async event=>{
  try{
    if(event.target.dataset.filter){state.filters[event.target.dataset.filter]=event.target.value;await loadQueue();}
    if(event.target.id==='import-file'&&event.target.files[0]){const file=event.target.files[0];if(file.size>4_000_000)throw new Error('Choose a JSON file smaller than 4 MB.');$('#import-json').value=await file.text();}
  }catch(error){notify(error.message,true);}
});
document.addEventListener('keydown',event=>{
  if($('#modal').open||state.view!=='review'||event.ctrlKey||event.metaKey||event.altKey||/INPUT|TEXTAREA|SELECT/.test(event.target.tagName)||event.target.isContentEditable)return;
  if(event.key==='j'||event.key==='J'){event.preventDefault();moveItem(1).catch(error=>notify(error.message,true));}
  if(event.key==='k'||event.key==='K'){event.preventDefault();moveItem(-1).catch(error=>notify(error.message,true));}
  if(event.key==='/'){event.preventDefault();$('#queue-search')?.focus();}
});
window.addEventListener('hashchange',()=>{const view=location.hash.slice(1);if(['review','gold','insights','activity'].includes(view)&&state.boot)navigate(view).catch(error=>notify(error.message,true));});

async function start(){
  try{
    state.client=await createClient();await refreshData();
    const view=location.hash.slice(1);if(['review','gold','insights','activity'].includes(view))state.view=view;
    if(state.items.length){state.selected=state.items[0].id;state.item=await state.client.item(state.selected);}
    if(state.view==='activity')state.events=(await state.client.events()).events;
    renderShell();
  }catch(error){
    $('#app').innerHTML=`<div class="initial-loader"><span class="brand-mark">R</span><h1>Workspace unavailable</h1><p style="max-width:460px;text-align:center">${esc(error.message)}</p>${error.status===401?'<label class="field" style="width:350px"><span>Reviewer access token</span><input id="access-token" type="password" autocomplete="off" placeholder="Enter your configured reviewer token"></label><button id="connect-service" class="btn btn-primary">Connect to review service</button>':'<button id="retry-start" class="btn">Try again</button>'}</div>`;
    $('#retry-start')?.addEventListener('click',start);
    $('#connect-service')?.addEventListener('click',()=>{sessionStorage.setItem('rush-labeling-token',$('#access-token').value.trim());start();});
  }
}
start();
