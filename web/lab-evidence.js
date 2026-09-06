/* Additive investigation surface. The native run selector, graph, and APIs own
 * state. There is no /api/studio discovery, alternate file root, or fake run. */
(() => {
  'use strict';
  const C=window.RushEvidence, $=id=>document.getElementById(id);
  const esc=v=>String(v??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  let run=null, sequence=0, detailSeq=0, timer=null, playing=false, debounce=null;
  let selected=null, controllers=new Set();
  const area=()=>window.rushActiveDemo?.()?.policyGraph?.area||'Generative_AI';
  async function read(url) {
    const controller=new AbortController();controllers.add(controller);
    const timeout=setTimeout(()=>controller.abort(),15000);
    try {const r=await fetch(url,{cache:'no-store',signal:controller.signal});if(!r.ok)throw new Error(`${url}: HTTP ${r.status}`);return await r.json();}
    finally{clearTimeout(timeout);controllers.delete(controller);}
  }
  function cancel(){controllers.forEach(c=>c.abort());controllers.clear();}
  function stop(){playing=false;clearTimeout(timer);if($('evPlay'))$('evPlay').textContent='▶ Replay this run';}
  function clearEvidence(){
    run=null;selected=null;++detailSeq;stop();
    $('evTrail').replaceChildren();$('evFrame').textContent='';$('evDetail').hidden=true;
    $('evDiff').textContent='';$('evPlay').disabled=true;
    document.querySelectorAll('.policy-node').forEach(el=>el.classList.remove('ev-added','ev-edited','ev-proposed'));
  }
  function active(){const chip=document.querySelector('.experiment-kg-chip--active');return Number(chip?.dataset.kgK??run?.cycles?.at(-1)?.k??0);}
  function highlight(){
    const c=run?.cycles.find(x=>x.k===active());
    const changes=new Map(C.evidence(c).map(e=>[e.id,e.change]));
    const expected=run?C.frames(run).find(f=>f.k===active())?.version:null;
    const shown=$('policyGraphSvg')?.getAttribute('aria-label')||'';
    document.querySelectorAll('.policy-node').forEach(el=>{
      const id=el.__data__?.id, change=changes.get(id);
      const matching=Boolean(expected&&shown===`Policy graph ${expected}`);
      el.classList.toggle('ev-added',matching&&c?.status==='accepted'&&change==='added');
      el.classList.toggle('ev-edited',matching&&c?.status==='accepted'&&change&&change!=='added');
      el.classList.toggle('ev-proposed',matching&&c?.status!=='accepted'&&Boolean(change));
      el.classList.toggle('ev-search-dim',Boolean($('evSearch')?.value)&&!`${id} ${el.__data__?.title}`.toLowerCase().includes($('evSearch').value.toLowerCase()));
    });
  }
  async function load(){
    const id=$('experimentSelect')?.value;if(!id){clearEvidence();$('evSource').textContent='No run selected. The native policy graph is available below.';return;}
    const token=++sequence;cancel();if(run?.experiment_id!==id)clearEvidence();
    $('evSource').textContent=`Reading /api/experiments/${id} …`;
    try {
      const data=await read(`/api/experiments/${encodeURIComponent(id)}`);
      if(token!==sequence||$('experimentSelect').value!==id)return;
      if(data.experiment_id!==id||data.area!==area())throw new Error('Native experiment identity mismatch');
      C.frames(data);run=data;
      $('evSource').textContent=`Native API · run #${data.run_number??'?'} · seed ${data.seed??'?'} · ${data.dry_run===false?'recorded model run':data.dry_run===true?'DRY RUN — synthetic labels':'label origin unspecified'}`;
      $('evPlay').disabled=data.cycles.length<2;
      show();
    }catch(error){if(token!==sequence)return;clearEvidence();$('evSource').textContent=`Evidence panel: ${error.message}. The original graph and lab remain available.`;$('evTrail').replaceChildren();stop();}
  }
  async function show(){
    if(!run||run.experiment_id!==$('experimentSelect')?.value||run.area!==area())return;
    const token=++detailSeq,k=active(),c=run.cycles.find(x=>x.k===k);if(!c)return;
    selected=c;
    const f=C.frames(run).find(x=>x.k===k),m=C.metrics(c),changes=C.evidence(c);
    $('evFrame').textContent=`k=${k} · ${c.status||c.kind||'baseline'} · ${f?.version||'version unavailable'} in force`;
    $('evTrail').innerHTML=`<div class="ev-stage"><span class="ev-label">01 / GOLDEN EVIDENCE</span><h4>${(c.anchor_ids||c.anchors||[]).length} selected anchors</h4><div id="evAnchors" class="ev-anchors"></div><p>Reference labels and judge responses—not majority votes relabeled as truth.</p></div>
      <div class="ev-stage"><span class="ev-label">02 / POLICY UPDATE</span><h4>${changes.length} proposed node changes</h4><div class="ev-changes">${changes.map(e=>`<button type="button" data-ev-rule="${esc(e.id)}"><b>${esc(e.change)}</b> ${esc(e.id)}</button>`).join('')||'<p>No edit proposed in this step.</p>'}</div><p>${c.status==='accepted'?'Accepted changes are highlighted on the native graph.':'The incumbent graph is retained. Dashed rings identify touched existing nodes, not accepted edits.'}</p></div>
      <div class="ev-stage"><span class="ev-label">03 / MEASURED RESPONSE</span><h4>${k===0?'Baseline':c.status==='accepted'?'Accepted candidate':'Candidate, not incumbent'} · development split</h4><dl class="ev-metrics"><div><dt>Macro FPR</dt><dd>${C.pct(m.fpr)}</dd></div><div><dt>Macro FNR</dt><dd>${C.pct(m.fnr)}</dd></div><div><dt>Decided n</dt><dd>${C.finite(m.n)?m.n:'—'}</dd></div><div><dt>Coverage</dt><dd>${C.pct(m.coverage)}</dd></div></dl><p>Stored scores. Historical macro-F1 needs re-scoring after the count-based correction; this view does not relabel old verdicts.</p></div>`;
    $('evTrail').querySelectorAll('[data-ev-rule]').forEach(button=>button.addEventListener('click',()=>{
      const id=button.dataset.evRule;
      const opened=window.rushOpenPolicyNode?.(id);
      $('evDetailStatus').textContent=opened?`Inspecting ${id}`:`${id} is not in the incumbent. Its proposed edit is shown below.`;
      proposal(c,id);
    }));
    $('evDetail').hidden=true;highlight();
    let anchors=Array.isArray(c.anchors)?c.anchors:[];
    try {
      if(c.train_run_id && (c.anchor_ids?.length||anchors.length)){
        const payload=await read(`/data/runs/${encodeURIComponent(c.train_run_id)}/scoring/misalignment.json`);
        const byId=new Map((payload.records||[]).map(a=>[String(a.image_id),a]));
        anchors=anchors.length?anchors.map(a=>({...a,...byId.get(String(a.image_id))})):(c.anchor_ids||[]).map(id=>byId.get(String(id))).filter(Boolean);
      }
    }catch(_){/* Stored lean anchors are still genuine evidence; do not invent images. */}
    if(token!==detailSeq||c!==selected)return;
    $('evAnchors').innerHTML=anchors.slice(0,12).map((a,i)=>`<button type="button" data-ev-anchor="${i}" title="${esc(a.image_id)} — reference ${esc(a.sme_truth)}">${a.repo_rel_path?`<img loading="lazy" src="/api/thumbnail?path=${encodeURIComponent(a.repo_rel_path)}" alt="Anchor ${esc(a.image_id)}"/>`:'<span>No image</span>'}<small>${esc(a.sme_truth??'reference missing')}</small></button>`).join('')||'<p>No anchor images stored for this step.</p>';
    $('evAnchors').querySelectorAll('img').forEach(img=>img.addEventListener('error',()=>{
      const missing=document.createElement('span');missing.className='ev-image-missing';
      missing.textContent='Image file unavailable';missing.title=img.alt;img.replaceWith(missing);
    },{once:true}));
    $('evAnchors').querySelectorAll('[data-ev-anchor]').forEach(button=>button.addEventListener('click',()=>{
      const a=anchors[Number(button.dataset.evAnchor)];
      window.rushShowEvidence?.({...a,run_id:c.train_run_id,votes:(a.votes||[]).map(v=>({...v,model:v.model||v.labeler_id||v.model_id}))});
    }));
  }
  async function proposal(c,id){
    const token=++detailSeq;$('evDetail').hidden=false;$('evDiff').textContent='Reading the saved proposal…';
    if(!c.proposal_id){$('evDiff').textContent='No proposal identifier is stored for this cycle.';return;}
    try{
      const p=await read(`/api/policy/proposals/${encodeURIComponent(c.proposal_id)}`);
      if(token!==detailSeq||selected!==c)return;
      const diffs=(p.diffs||[]).filter(d=>!id||d.path?.split('/').pop()===id+'.md');
      $('evDiff').textContent=diffs.map(d=>`${d.path}\n${d.unified_diff||''}`).join('\n\n')||'No saved diff for this node in the proposal response.';
    }catch(e){if(token===detailSeq)$('evDiff').textContent=e.message;}
  }
  function replay(){
    if(playing){stop();return;}
    const chips=()=>[...document.querySelectorAll('#experimentKgCycles [data-kg-k]')];
    if(chips().length<2)return;
    playing=true;$('evPlay').textContent='Ⅱ Pause';chips()[0].click();
    const next=()=>{
      if(!playing)return;const list=chips(),i=list.findIndex(b=>Number(b.dataset.kgK)===active());
      if(i<0||i===list.length-1){stop();return;}
      list[i+1].click();timer=setTimeout(next,2400);
    };timer=setTimeout(next,2400);
  }
  function mount(){
    if(!C||!$('policyEvolution')||$('evToolbar'))return;
    document.body.classList.add('native-research');
    const graph=$('policyEvolution'),runPicker=$('experimentSelect')?.closest('.run-controls');
    if(runPicker) {runPicker.after(graph); graph.before($('experimentSummary'));}
    const tools=document.createElement('div');tools.id='evToolbar';tools.innerHTML=`<div><span class="ev-label">POLICY DEVELOPMENT / CONNECTED TO THE EXISTING LAB</span><h2>Evidence → rule → measured response</h2><p id="evSource" role="status">Waiting for the native run selector…</p></div><div class="ev-controls"><button type="button" id="evPlay" disabled>▶ Replay this run</button><input id="evSearch" type="search" placeholder="Find a policy node…" aria-label="Find a policy node"/><span id="evFrame"></span></div>`;
    graph.prepend(tools);
    const trail=document.createElement('div');trail.id='evTrail';tools.after(trail);
    const detail=document.createElement('details');detail.id='evDetail';detail.hidden=true;detail.open=true;detail.innerHTML='<summary id="evDetailStatus">Recorded rule change</summary><pre id="evDiff"></pre>';trail.after(detail);
    const config=document.querySelector('.experiment-config');
    if(config){const box=document.createElement('details');box.className='ev-config';box.open=true;box.innerHTML='<summary>Experiment configuration — judges, optimization, sampling, and gates</summary>';config.before(box);box.append(config);}
    const notes=document.createElement('p');notes.className='ev-legend';notes.textContent='Solid highlight: accepted addition/edit · dashed ring: proposed change only · graph edges and rule text come from the original policy API.';tools.after(notes);
    $('evPlay').addEventListener('click',replay);$('evSearch').addEventListener('input',highlight);
    $('experimentSelect')?.addEventListener('change',()=>{stop();load();});
    $('experimentRefresh')?.addEventListener('click',()=>setTimeout(load,500));
    new MutationObserver(()=>{clearTimeout(debounce);debounce=setTimeout(load,180);}).observe($('experimentSummary'),{childList:true});
    new MutationObserver(()=>show()).observe($('experimentKgCycles'),{childList:true});
    new MutationObserver(()=>highlight()).observe($('policyGraphSvgWrap'),{childList:true,subtree:true});
    if(window.RUSH_REVIEW_ONLY){
      const banner=document.createElement('div');banner.className='ev-preview';banner.textContent='READ-ONLY CONNECTED PREVIEW · native data is proxied from your working RUSH server. All writes are blocked; run experiments and adjudicate in the original application.';document.body.prepend(banner);
      if(config)config.querySelectorAll('input,select,button').forEach(el=>el.disabled=true);
      read('/__review__/source').then(source=>{banner.textContent+=` Source: ${source.upstream}`;}).catch(()=>{});
    }
    document.addEventListener('visibilitychange',()=>{if(document.hidden)stop();});
    window.addEventListener('pagehide',()=>{stop();cancel();});
    load();
  }
  if(document.readyState==='loading')document.addEventListener('DOMContentLoaded',mount,{once:true});else mount();
})();
