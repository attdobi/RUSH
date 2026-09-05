/* Executive policy explorer. All server access is GET-only; the lab owns mutations. */
(() => {
  'use strict';
  const C = window.RushStudioCore, F = window.RushStudioFixtures;
  const $ = id => document.getElementById(id);
  const esc = v => String(v ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const colors = {root:'#e8c785',category:'#96dfbc',rule:'#96dfbc',boundary:'#9caee8',exception:'#c3a6dc',guideline:'#a0adb9'};
  const state = {demo:new URLSearchParams(location.search).get('demo') === 'mnist' ? 'mnist' : 'genai',source:'recorded',series:[],frames:[],index:0,mode:'knowledge',selected:null,graph:null,previous:null,playing:false,scenario:0};
  const snapshots = new Map(), positions = new Map(), requests = new Set();
  let requestSeq = 0, loadSeq = 0, timer = null, pan = {x:0,y:0,scale:1}, drag = null;
  const reducedMotion = matchMedia('(prefers-reduced-motion: reduce)');
  const fixture = () => F.build(state.demo);
  const area = () => state.demo === 'mnist' ? 'MNIST_Digits' : 'Generative_AI';
  const currentFrame = () => state.frames[state.index] || {};
  const currentSeries = () => state.series.find(s => s.id === $('seriesSelect').value);
  const color = n => colors[n.node_type] || (String(n.id).includes('boundary') ? colors.boundary : String(n.id).includes('exception') ? colors.exception : colors.guideline);
  function cancelRequests() { for (const c of requests) c.abort(); requests.clear(); }
  async function getJSON(url) {
    const controller = new AbortController(); requests.add(controller);
    const timeout = setTimeout(() => controller.abort(), 7000);
    try {
      const response = await fetch(url, {cache:'no-store',signal:controller.signal});
      if (!response.ok) throw new Error(`History API returned HTTP ${response.status}.`);
      return await response.json();
    } finally { clearTimeout(timeout); requests.delete(controller); }
  }
  function pause() { state.playing = false; clearTimeout(timer); $('playFrames').textContent = 'Play evolution'; }
  function empty(message, action = '') {
    $('graphEmpty').hidden = false;
    $('graphEmpty').innerHTML = `<h3>${esc(message)}</h3>${action}`;
  }
  function sourceStatus(message) {
    $('sourceSelect').value = state.source;
    $('sourceBadge').textContent = state.source === 'recorded' ? 'RECORDED · READ ONLY' : 'ILLUSTRATIVE · NOT A MEASURED RUN';
    $('sourceBadge').classList.toggle('recorded', state.source === 'recorded');
    $('sourceNote').textContent = message;
  }
  function updateURL() {
    const url = new URL(location.href); url.searchParams.set('demo', state.demo);
    try { history.replaceState(null, '', url); } catch (_) { /* Embedded previews may forbid URL mutation. */ }
    $('labLink').href = `lab.html?demo=${encodeURIComponent(state.demo)}#loop`;
  }
  async function loadData(requested = state.source) {
    const token = ++loadSeq; ++requestSeq; cancelRequests(); pause();
    state.source = requested; state.mode = 'knowledge'; state.selected = null; state.graph = null; state.previous = null;
    positions.clear(); snapshots.clear(); pan = {x:0,y:0,scale:1};
    $('demoSelect').value = state.demo; updateURL();
    empty('Loading policy evidence…'); $('graphSvg').innerHTML = ''; $('graphCounts').textContent = '';
    try {
      if (requested === 'recorded') {
        const payload = await getJSON(`/api/studio/history?area=${encodeURIComponent(area())}`);
        if (token !== loadSeq) return;
        if (!Array.isArray(payload.series) || !payload.series.length) throw new Error('No complete local history found.');
        state.series = payload.series.filter(s => typeof s.id === 'string' && Array.isArray(s.frames) && s.frames.length);
        if (!state.series.length) throw new Error('No readable policy series found.');
        sourceStatus(`Existing files only. No model calls or policy changes. ${(payload.warnings || []).length ? payload.warnings.join(' ') : 'Historical scores need re-scoring after the F1 correction.'}`);
      } else {
        state.series = [{id:'story',title:'Guided development story',lineage:true,frames:fixture().frames}];
        sourceStatus('Scripted policy examples and supplied scenario facts. No measured accuracy, coverage or production lineage.');
      }
    } catch (error) {
      if (token !== loadSeq) return;
      state.source = 'illustrative';
      state.series = [{id:'story',title:'Guided development story',lineage:true,frames:fixture().frames}];
      sourceStatus('Local recorded history is unavailable here. Showing a clearly labeled illustrative walkthrough—not measured results.');
    }
    $('seriesSelect').innerHTML = state.series.map(s => `<option value="${esc(s.id)}">${esc(s.title)}</option>`).join('');
    $('seriesLabel').hidden = state.source !== 'recorded';
    const selected = state.series.find(s => s.lineage) || state.series[0];
    $('seriesSelect').value = selected.id; state.frames = selected.frames;
    await showFrame(0);
  }
  async function readSnapshot(frame) {
    if (state.source === 'illustrative') return C.normalizeGraph(frame);
    const key = `${area()}:${frame.version}`;
    if (!snapshots.has(key)) snapshots.set(key, getJSON(`/api/studio/snapshot?area=${encodeURIComponent(area())}&version=${encodeURIComponent(frame.version)}`).then(C.normalizeGraph).catch(e => {snapshots.delete(key); throw e;}));
    return snapshots.get(key);
  }
  async function showFrame(index) {
    if (!state.frames.length) return;
    const token = ++requestSeq;
    index = Math.max(0, Math.min(index, state.frames.length - 1));
    const frame = state.frames[index];
    $('graphStage').setAttribute('aria-busy','true');
    $('exportTrace').disabled = true;
    try {
      const [graph, previous] = await Promise.all([readSnapshot(frame), index ? readSnapshot(state.frames[index-1]) : Promise.resolve(null)]);
      if (token !== requestSeq) return;
      state.index = index; state.graph = graph; state.previous = previous;
      if (state.selected && !graph.nodes.some(n => n.id === state.selected)) state.selected = null;
      if (state.mode === 'paths' && state.source === 'illustrative' && index !== 5) state.mode = 'knowledge';
      $('graphEmpty').hidden = true;
      render();
    } catch (error) {
      if (token !== requestSeq) return;
      pause(); state.graph = null; state.previous = null;
      $('graphSvg').innerHTML = ''; $('graphCounts').textContent = '';
      $('inspectorContent').innerHTML = '<h3>Evidence unavailable.</h3><p>The selected snapshot could not be read. No previous snapshot is being presented as this version.</p>';
      empty('This recorded snapshot could not be loaded.', '<p>Refresh history or choose the illustrative walkthrough. No measured result is substituted.</p>');
    } finally {
      if (token === requestSeq) { $('graphStage').setAttribute('aria-busy','false'); $('exportTrace').disabled = !state.graph; }
    }
  }
  function layout(nodes) {
    const byId = new Map(nodes.map(n => [n.id,n]));
    const roots = nodes.filter(n => n.node_type === 'root' || !n.parent);
    const rootIds = new Set(roots.map(n => n.id));
    const families = nodes.filter(n => rootIds.has(n.parent)).sort((a,b) => a.id.localeCompare(b.id));
    const hash = id => [...id].reduce((a,c) => ((a*31+c.charCodeAt(0)) >>> 0),7) / 4294967296;
    roots.forEach((n,i) => {if(!positions.has(n.id)) positions.set(n.id,{x:470+(i ? 65*Math.cos(i*2.4) : 0),y:242+(i ? 65*Math.sin(i*2.4) : 0)});});
    for (const n of nodes) {
      if (positions.has(n.id)) continue;
      let family = n, depth = 1, seen = new Set([n.id]);
      while (family.parent && byId.has(family.parent) && !rootIds.has(family.parent) && !seen.has(family.parent)) {
        seen.add(family.parent); family = byId.get(family.parent); depth++;
      }
      const fi = families.findIndex(f => f.id === family.id);
      const angle = fi < 0 ? hash(n.id)*Math.PI*2 : fi/Math.max(1,families.length)*Math.PI*2-Math.PI/2;
      const siblings = nodes.filter(s => s.parent === n.parent).sort((a,b) => a.id.localeCompare(b.id));
      const spread = depth > 1 ? (siblings.findIndex(s => s.id === n.id) - (siblings.length-1)/2)*.29 : 0;
      let x = 470 + Math.cos(angle+spread)*(depth === 1 ? 245 : Math.min(385,295+depth*28));
      let y = 242 + Math.sin(angle+spread)*(depth === 1 ? 152 : Math.min(211,165+depth*18));
      for (let k=0;k<14;k++) {
        let moved = false;
        for (const p of positions.values()) {
          const dx=x-p.x,dy=y-p.y,dist=Math.hypot(dx,dy);
          if (dist<53) {x+=(dx || .7)/Math.max(dist,1)*7;y+=(dy || .4)/Math.max(dist,1)*7;moved=true;}
        }
        x=Math.max(74,Math.min(866,x));y=Math.max(48,Math.min(445,y));
        if (!moved) break;
      }
      positions.set(n.id,{x,y});
    }
    return positions;
  }
  function camera() { const el=$('graphCamera'); if(el) el.setAttribute('transform',`translate(${pan.x},${pan.y}) scale(${pan.scale})`); $('graphSvg').classList.toggle('zoomed',pan.scale>1.3); }
  function labelLines(title) {
    const lines=[''];
    for(const word of String(title).split(/\s+/)){const i=lines.length-1;if(lines[i]&&(lines[i]+' '+word).length>17)lines.push(word);else lines[i]+=(lines[i]?' ':'')+word;}
    return lines.slice(0,3).map((line,i)=>esc(line.length>18?line.slice(0,16)+'…':line)+(i===2&&lines.length>3?'…':''));
  }
  function renderKnowledge() {
    const graph = state.graph, diff = C.graphDiff(state.previous, graph);
    const added = new Set(state.previous ? diff.added : []), changed = new Set(diff.changed);
    const pos = layout(graph.nodes);
    const connected = new Set([state.selected]);
    if (state.selected) graph.edges.forEach(e => {if(e.source === state.selected) connected.add(e.target);if(e.target === state.selected) connected.add(e.source);});
    const nodesHTML = (nodes, removed=false) => nodes.map(n => {
      const p = pos.get(n.id); if(!p) return '';
      const root=n.node_type==='root',radius=root?15:n.node_type==='category'?10:8,c=color(n);
      const lines = labelLines(n.title);
      const compact = graph.nodes.length>30 && !root && n.node_type!=='category' && !added.has(n.id);
      return `<g class="graph-node ${compact?'compact':''} ${root?'root-node':''} ${added.has(n.id)?'entering':''} ${changed.has(n.id)?'changed':''} ${removed?'removed':''} ${state.selected===n.id?'selected':''} ${state.selected&&!connected.has(n.id)?'dim':''}" transform="translate(${p.x},${p.y})" data-node="${esc(n.id)}" tabindex="${removed?'-1':'0'}" role="button" aria-label="${esc(n.title)}${removed?' (retired)':''}" aria-pressed="${state.selected===n.id}"><title>${esc(n.id)} · ${esc(n.node_type)}</title><circle class="halo" r="${radius+8}" stroke="${c}"/><circle class="orb" r="${radius}" stroke="${c}"/><circle r="${root?4:2}" fill="${c}"/><text y="${radius+21}">${lines.map((line,i)=>`<tspan x="0" dy="${i?13:0}">${line}</tspan>`).join('')}</text></g>`;
    }).join('');
    $('graphSvg').innerHTML = `<defs><marker id="kgArrow" viewBox="0 0 10 10" refX="19" refY="5" markerWidth="4" markerHeight="4" orient="auto"><path d="M0 0L10 5L0 10" fill="#657685"/></marker></defs><g id="graphCamera"><g opacity=".12" fill="none" stroke="#8298a4"><ellipse cx="470" cy="242" rx="245" ry="152"/><ellipse cx="470" cy="242" rx="360" ry="205"/></g>${graph.edges.map(e => {
      const a=pos.get(e.source),b=pos.get(e.target);if(!a||!b)return '';
      return `<line class="edge ${e.type==='subtype_of'?'':'cross'} ${state.selected&&e.source!==state.selected&&e.target!==state.selected?'dim':''}" x1="${a.x}" y1="${a.y}" x2="${b.x}" y2="${b.y}" marker-end="url(#kgArrow)"><title>${esc(e.source)} → ${esc(e.target)} · ${esc(e.type)}${e.synthetic?' (synthetic)':''}</title></line>`;
    }).join('')}${nodesHTML((state.previous?.nodes||[]).filter(n=>diff.removed.includes(n.id)),true)}${nodesHTML(graph.nodes)}</g>`;
    camera(); bindNodes();
    $('graphLegend').hidden = false; $('scenarioBar').hidden = true;
    $('graphHint').textContent = 'Select a node to inspect its rule · scroll to zoom · drag the canvas to pan';
    $('graphVersion').textContent = `${state.source==='illustrative'?'ILLUSTRATION':'POLICY'} / ${currentFrame().version}`;
    $('graphCounts').innerHTML = `<span><b>${graph.nodes.length}</b> nodes</span><span><b>${graph.edges.length}</b> edges</span><span><b>${diff.changed.length}</b> edits</span>`;
    $('deltaNote').textContent = `${state.previous?`Compared with ${state.frames[state.index-1].version}: +${diff.added.length} added · ${diff.changed.length} changed · −${diff.removed.length} retired.`:'Starting snapshot.'} ${[...(graph.warnings||[])].length ? 'Unresolved edges are disclosed, never fabricated.' : 'Graph size is not a quality score.'}`;
  }
  function bindNodes() {
    document.querySelectorAll('#graphSvg [data-node]').forEach(el => {
      const select = () => {if(state.graph.nodes.some(n=>n.id===el.dataset.node)){state.selected=el.dataset.node;render();}};
      el.addEventListener('click',select);
      el.addEventListener('keydown',e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();select();}});
    });
  }
  function pathResult() {
    const f=fixture();
    return {program:f.program,facts:f.scenarios[state.scenario].facts,result:C.evaluateProgram(f.program,f.scenarios[state.scenario].facts,{policy_area:state.graph.area,policy_version:state.graph.version,policy_node_ids:state.graph.nodes.map(n=>n.id)})};
  }
  function renderPaths() {
    $('graphLegend').hidden = true;
    $('graphVersion').textContent = 'EXPLICIT DECISION PROGRAM / SHADOW ONLY';
    $('graphCounts').textContent = 'Typed predicates · no model calls';
    $('graphHint').textContent = 'A deterministic route over supplied facts—not an image inference.';
    if (state.source !== 'illustrative') {
      $('graphSvg').innerHTML = ''; $('scenarioBar').hidden = true;
      empty('No executable program is attached to this recorded snapshot.', '<p>A knowledge graph is not silently compiled into decision code. Inspect the separately labeled shadow example instead.</p><button type="button" id="openShadow" class="primary">Open illustrative shadow paths</button>');
      $('openShadow').addEventListener('click',async()=>{await loadData('illustrative');await showFrame(5);state.mode='paths';render();});
      $('deltaNote').textContent='Recorded policy remains unchanged. No decision-quality improvement is claimed.';
      return;
    }
    const {program,result}=pathResult();
    const activeNodes=new Set([...result.trace.map(t=>t.node_id),result.terminal_id]);
    const activeEdges=new Set(result.trace.map(t=>`${t.node_id}:${t.outcome}`));
    const positions={first:{x:140,y:240},second:{x:410,y:110},third:{x:490,y:375},review:{x:790,y:245},nine:{x:760,y:75},four:{x:790,y:425},generated:{x:760,y:65},evidence:{x:790,y:425}};
    const edgeHTML=program.nodes.filter(n=>n.kind==='rule').flatMap(n=>Object.entries(n.next).map(([outcome,target])=>{
      const a=positions[n.id],b=positions[target],active=activeEdges.has(`${n.id}:${outcome}`);
      const offset=outcome==='unknown'?12:outcome==='false'?-9:0;
      const ax=a.x+88,ay=a.y+offset,bx=b.x-90,by=b.y;
      return `<path class="path-edge ${active?'active':''} ${outcome==='unknown'?'unknown':''}" d="M${ax},${ay} C${ax+65},${ay} ${bx-65},${by} ${bx},${by}"/><text class="path-label" x="${ax+35}" y="${ay+(outcome==='unknown'?20:-8)}">${outcome==='unknown'?'?':outcome}</text>`;
    })).join('');
    $('graphSvg').innerHTML=`${edgeHTML}${program.nodes.map(n=>{
      const p=positions[n.id];return `<g class="path-node ${activeNodes.has(n.id)?'active':''} ${n.id==='review'?'review':''}" transform="translate(${p.x},${p.y})" role="img" aria-label="${esc(n.title)}${activeNodes.has(n.id)?', on selected route':''}"><rect x="-90" y="-30" width="180" height="60" rx="${n.kind==='rule'?7:22}"/><text y="-2">${esc(n.title)}</text><text y="15" class="path-sub">${n.kind==='rule'?'TYPED PREDICATE':'SHADOW ACTION'}</text></g>`;
    }).join('')}`;
    $('graphEmpty').hidden=true; $('scenarioBar').hidden=false;
    $('scenarioBar').innerHTML='<span>SUPPLIED SCENARIO</span>'+fixture().scenarios.map((s,i)=>`<button type="button" data-scenario="${i}" class="${i===state.scenario?'active':''}" aria-pressed="${i===state.scenario}">${esc(s.title)}</button>`).join('');
    document.querySelectorAll('[data-scenario]').forEach(b=>b.addEventListener('click',()=>{state.scenario=Number(b.dataset.scenario);render();}));
    $('deltaNote').textContent='Same pinned program + same typed facts → same route. Unknown or untrusted evidence goes to review. No live labels are changed.';
  }
  function renderInspector() {
    const node=state.graph.nodes.find(n=>n.id===state.selected);
    $('clearSelection').hidden=!node;
    if(node){
      $('inspectorKicker').textContent='POLICY NODE / EVIDENCE';
      const previous=state.previous?.nodes.find(n=>n.id===node.id);
      const changed=previous&&C.graphDiff({nodes:[previous]},{nodes:[node]}).changed.length;
      $('inspectorContent').innerHTML=`<div class="detail-type">${esc(node.node_type)}</div><h3>${esc(node.title)}</h3><div class="detail-id">${esc(node.id)}</div><hr><div class="body-text">${esc(node.body||'No rule body is attached to this snapshot.')}</div>${changed?`<details><summary>Previous rule text</summary><div class="body-text">${esc(previous.body||'No previous text available.')}</div></details>`:''}<p class="note">${state.source==='illustrative'?'Scripted explanation. Not a production rule or a measured result.':`Source: ${esc(node.source||'recorded policy snapshot')}<br>Body SHA-256: ${esc(node.content_hash||'unavailable')}`}</p>`;
      return;
    }
    if(state.mode==='paths'){
      $('inspectorKicker').textContent='DECISION TRACE / SUPPLIED EVIDENCE';
      if(state.source!=='illustrative'){$('inspectorContent').innerHTML='<h3>Explicit, not inferred.</h3><p>Attach a reviewed decision program before claiming deterministic execution of a recorded policy.</p>';return;}
      const {result}=pathResult();
      $('inspectorContent').innerHTML=`<span class="step-number">SHADOW EXECUTION</span><h3>Follow the facts.<br>Keep the uncertainty.</h3>${result.trace.map((t,i)=>`<div class="trace-row">${i+1}. ${esc(t.field)} → <b>${esc(t.outcome)}</b><br><code>${esc(t.policy_node_id)}</code></div>`).join('')}<div class="trace-outcome">${esc(result.action.replaceAll('_',' '))}</div><p class="note">Facts are supplied by the selected scenario. Source tags are caller assertions, not authenticated provenance. No model inference or live classification is performed.</p>`;
      return;
    }
    const f=currentFrame(),diff=C.graphDiff(state.previous,state.graph);
    $('inspectorKicker').textContent=state.source==='illustrative'?'THE DEVELOPMENT LOOP':'RECORDED POLICY HISTORY';
    $('inspectorContent').innerHTML=`<span class="step-number">${String(state.index+1).padStart(2,'0')} / ${String(state.frames.length).padStart(2,'0')}</span><h3>${esc(f.headline||f.title||f.version)}</h3><p>${esc(f.detail||'Inspect the rule text and its explicit relationships.')}</p><div class="loop-mini"><span>Evidence</span><i>→</i><span>Policy</span><i>→</i><span>Evaluation</span></div>${state.previous?`<div class="delta-pills"><span class="positive">+${diff.added.length} added</span><span>${diff.changed.length} edited</span><span class="negative">−${diff.removed.length} retired</span></div>`:''}<p class="note">${state.source==='illustrative'?'Illustrative walkthrough. The gate outcome is scripted, not an evaluation result.':currentSeries()?.lineage?'Replay follows the experiment’s recorded version sequence. Historical scores require re-scoring before comparison.':'Snapshot catalog, not a lineage. Adjacent versions may belong to unrelated runs.'}</p>${(state.graph.warnings||[]).length?`<details><summary>Graph integrity notes (${state.graph.warnings.length})</summary><p>${state.graph.warnings.map(esc).join('<br>')}</p></details>`:''}`;
  }
  function renderTimeline() {
    $('timeline').innerHTML=state.frames.map((f,i)=>`<button type="button" data-frame="${i}" class="${i===state.index?'active':''} ${['rejected','skipped'].includes(f.status)?'rejected':''}" aria-current="${i===state.index?'step':'false'}"><span class="timeline-kicker">${String(i+1).padStart(2,'0')} / ${esc(f.version)}</span><span class="timeline-title">${esc(f.title||f.version)}</span></button>`).join('');
    document.querySelectorAll('[data-frame]').forEach(b=>b.addEventListener('click',()=>{pause();showFrame(Number(b.dataset.frame));}));
    $('frameRange').max=state.frames.length-1; $('frameRange').value=state.index;
    $('prevFrame').disabled=state.index===0; $('nextFrame').disabled=state.index===state.frames.length-1;
    $('playFrames').disabled=state.frames.length<2;
    $('playFrames').textContent=state.playing?'Pause':currentSeries()?.lineage?'Play evolution':'Browse snapshots';
    $('timelineNote').textContent=state.source==='illustrative'?'A guided story: evidence → boundary → candidate → gate → governed rule.':currentSeries()?.lineage?'This run’s recorded sequence. Rejected edits retain the incumbent.':'Catalog navigation only. This sequence is not asserted to be a policy lineage.';
  }
  function render() {
    if(!state.graph)return;
    $('knowledgeTab').setAttribute('aria-pressed',state.mode==='knowledge');$('pathsTab').setAttribute('aria-pressed',state.mode==='paths');
    if(state.mode==='knowledge')renderKnowledge();else renderPaths();
    renderInspector();renderTimeline();
    $('exportTrace').textContent=state.mode==='paths'&&state.source==='illustrative'?'Export route + facts JSON ↗':'Export snapshot JSON ↗';
  }
  async function play() {
    if(state.playing){pause();renderTimeline();return;}
    if(state.frames.length<2)return;
    state.mode='knowledge';state.selected=null;
    if(state.index>=state.frames.length-1)await showFrame(0);
    state.playing=true;renderTimeline();
    const tick=async()=>{
      if(!state.playing)return;
      if(state.index>=state.frames.length-1){pause();renderTimeline();return;}
      await showFrame(state.index+1);
      if(state.playing)timer=setTimeout(tick,2600);
    };
    timer=setTimeout(tick,1800);
  }
  function pageRoute() {
    if (['#loop','#summary','#adjudicate','#benchmarks','#experiment'].includes(location.hash)) {
      location.replace(`lab.html${location.search}${location.hash}`); return;
    }
    const about=location.hash==='#about';
    $('studioView').hidden=about;$('aboutView').hidden=!about;
    document.querySelectorAll('[data-nav]').forEach(a=>a.classList.toggle('active',a.dataset.nav===(about?'about':'studio')));
    if(about)pause();
  }
  $('demoSelect').addEventListener('change',()=>{state.demo=$('demoSelect').value;state.scenario=0;loadData();});
  $('sourceSelect').addEventListener('change',()=>loadData($('sourceSelect').value));
  $('retryHistory').addEventListener('click',()=>loadData('recorded'));
  $('seriesSelect').addEventListener('change',()=>{pause();state.frames=currentSeries().frames;state.selected=null;positions.clear();showFrame(0);});
  $('knowledgeTab').addEventListener('click',()=>{state.mode='knowledge';$('graphEmpty').hidden=true;render();});
  $('pathsTab').addEventListener('click',async()=>{pause();state.selected=null;if(state.source==='illustrative')await showFrame(5);state.mode='paths';render();});
  $('clearSelection').addEventListener('click',()=>{state.selected=null;render();});
  $('prevFrame').addEventListener('click',()=>{pause();showFrame(state.index-1);});
  $('nextFrame').addEventListener('click',()=>{pause();showFrame(state.index+1);});
  $('frameRange').addEventListener('input',()=>{pause();showFrame(Number($('frameRange').value));});
  $('playFrames').addEventListener('click',play);
  $('heroPlay').addEventListener('click',()=>{if(location.hash==='#about')location.hash='studio';$('workspace').scrollIntoView({behavior:reducedMotion.matches?'instant':'smooth',block:'start'});play();});
  $('nodeSearch').addEventListener('input',()=>{
    const q=$('nodeSearch').value.toLowerCase().trim();$('searchResults').hidden=!q;
    if(!q||!state.graph)return;
    const matches=state.graph.nodes.filter(n=>`${n.id} ${n.title}`.toLowerCase().includes(q)).slice(0,12);
    $('searchResults').innerHTML=matches.length?matches.map(n=>`<button type="button" data-result="${esc(n.id)}">${esc(n.title)}</button>`).join(''):'<p>No matching policy nodes.</p>';
    document.querySelectorAll('[data-result]').forEach(b=>b.addEventListener('click',()=>{state.selected=b.dataset.result;$('nodeSearch').value='';$('searchResults').hidden=true;render();}));
  });
  const zoom=factor=>{if(state.mode!=='knowledge')return;const old=pan.scale;pan.scale=Math.max(.55,Math.min(3.5,old*factor));pan.x=470-(470-pan.x)*pan.scale/old;pan.y=250-(250-pan.y)*pan.scale/old;camera();};
  $('zoomIn').addEventListener('click',()=>zoom(1.2));$('zoomOut').addEventListener('click',()=>zoom(1/1.2));
  $('fitGraph').addEventListener('click',()=>{pan={x:0,y:0,scale:1};camera();});
  $('graphSvg').addEventListener('wheel',e=>{if(state.mode==='knowledge'){e.preventDefault();zoom(e.deltaY<0?1.07:1/1.07);}},{passive:false});
  $('graphSvg').addEventListener('pointerdown',e=>{if(state.mode!=='knowledge'||e.target.closest('[data-node]'))return;drag={x:e.clientX,y:e.clientY,px:pan.x,py:pan.y};$('graphSvg').setPointerCapture(e.pointerId);});
  $('graphSvg').addEventListener('pointermove',e=>{if(drag){const scale=940/$('graphSvg').getBoundingClientRect().width;pan.x=drag.px+(e.clientX-drag.x)*scale;pan.y=drag.py+(e.clientY-drag.y)*scale;camera();}});
  ['pointerup','pointercancel','lostpointercapture'].forEach(type=>$('graphSvg').addEventListener(type,()=>{drag=null;}));
  $('fullscreen').addEventListener('click',async()=>{try{if(document.fullscreenElement)await document.exitFullscreen();else await $('workspace').requestFullscreen();}catch(e){$('sourceNote').textContent='Fullscreen is not available in this browser. The explorer remains interactive.';}});
  $('exportTrace').addEventListener('click',()=>{
    if(!state.graph)return;
    let value={origin:state.source,frame:currentFrame(),snapshot:state.graph};
    if(state.mode==='paths'&&state.source==='illustrative')value={origin:'illustrative',...pathResult(),notice:'Shadow route only; facts supplied by a scripted scenario. No live inference.'};
    const blob=new Blob([JSON.stringify(value,null,2)+'\n'],{type:'application/json'}),url=URL.createObjectURL(blob),a=document.createElement('a');
    a.href=url;a.download=`rush-${state.demo}-${state.mode}-${state.index}.json`;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
  });
  document.addEventListener('visibilitychange',()=>{if(document.hidden){pause();if(state.frames.length)renderTimeline();}});
  window.addEventListener('hashchange',pageRoute);
  window.addEventListener('pagehide',()=>{pause();cancelRequests();});
  pageRoute();loadData();
})();
