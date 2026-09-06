/* Research workbench: the original experimental controls remain the write path.
 * This module reads saved artifacts only. It never starts a provider call. */
(() => {
  'use strict';
  const C = window.RushResearchCore;
  if (!C || !document.getElementById('experiment')) return;
  const $ = id => document.getElementById(id);
  const esc = v => String(v ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const reduced = matchMedia('(prefers-reduced-motion: reduce)');
  const state = { series: [], frames: [], index: 0, graph: null, previous: null, run: null, selected: null,
    layout: 'network', playing: false, points: new Map(), sourceId: '', area: '', external: false };
  let seq = 0, controller = null, timer = null, animation = null;
  let pan = { x: 0, y: 0, scale: 1 }, pointer = null;
  const cache = new Map();
  const area = () => window.rushActiveDemo?.()?.policyGraph?.area || (new URLSearchParams(location.search).get('demo') === 'mnist' ? 'MNIST_Digits' : 'Generative_AI');
  const classes = () => area() === 'MNIST_Digits' ? ['0','1','2','3','4','5','6','7','8','9'] : ['gen_ai','not_gen_ai'];
  const frame = () => state.frames[state.index] || {};
  const cycle = () => state.run?.cycles?.find(c => c.k === frame().k);
  const pct = v => C.finite(v) ? `${(v * 100).toFixed(1)}%` : '—';
  const number = v => C.finite(v) ? v.toLocaleString() : '—';
  const short = (v, n = 26) => String(v).length > n ? String(v).slice(0, n - 1) + '…' : String(v);
  function pause() { state.playing = false; clearTimeout(timer); if ($('rchPlay')) $('rchPlay').textContent = '▶ Replay'; }
  function begin() { controller?.abort(); controller = new AbortController(); return ++seq; }
  async function json(url, signal) {
    const request = new AbortController(), abort = () => request.abort();
    if (signal?.aborted) abort(); else signal?.addEventListener('abort', abort, {once:true});
    const timeout = setTimeout(abort, 12000);
    try {
      const response = await fetch(url, {cache:'no-store', signal:request.signal});
      if (!response.ok) throw new Error(`Evidence API returned HTTP ${response.status}`);
      return await response.json();
    } finally {clearTimeout(timeout); signal?.removeEventListener('abort', abort);}
  }
  function message(text, error = false) {
    $('rchStatus').textContent = text;
    $('rchStatus').classList.toggle('rch-error', error);
  }
  function unavailable(text) {
    pause(); state.graph = null; state.previous = null; state.run = null; state.frames = []; state.index = 0; cancelAnimationFrame(animation);
    $('rchSvg').innerHTML = '';
    $('rchEmpty').hidden = false;
    $('rchEmpty').innerHTML = `<span class="rch-empty-symbol" aria-hidden="true">G</span><h3>No evidence substituted.</h3><p>${esc(text)}</p><p>The experiment controls below remain available. Record a run or configure the read-only evidence root, then refresh.</p>`;
    $('rchInspector').innerHTML = '<h3>Awaiting a recorded policy</h3><p>There is no synthetic learning curve or automatic illustrative fallback in this research view.</p>';
    $('rchStats').innerHTML = ''; $('rchTimeline').innerHTML = ''; $('rchAudit').innerHTML = '';
    $('rchVersion').textContent = 'Evidence unavailable'; $('rchEvidenceTitle').textContent = ''; $('rchEvidenceNote').textContent = ''; $('rchDelta').textContent = ''; $('rchMatch').textContent = ''; $('rchStep').textContent = '0 / 0'; $('rchGateStatus').textContent = 'UNAVAILABLE';
    $('rchPrev').disabled = true; $('rchNext').disabled = true;
    $('rchRange').disabled = true; $('rchPlay').disabled = true; $('rchExport').disabled = true;
    message(text, true);
  }
  function syncLabRun(id) {
    const select = $('experimentSelect');
    if (!state.external && select && [...select.options].some(o => o.value === id) && select.value !== id) {
      select.value = id; select.dispatchEvent(new Event('change', { bubbles: true }));
    }
  }
  async function refresh() {
    const token = begin(), signal = controller.signal;
    pause(); $('rchStage').setAttribute('aria-busy', 'true');
    message('Reading policy snapshots and experiment lineage…');
    try {
      const a = area();
      const payload = await json(`/api/studio/history?area=${encodeURIComponent(a)}`, signal);
      if (token !== seq) return;
      state.external = payload.external_evidence === true;
      state.series = (Array.isArray(payload.series) ? payload.series : []).filter(s => typeof s.id === 'string' && Array.isArray(s.frames) && s.frames.length);
      if (!state.series.length) throw new Error('No complete recorded policy history is available.');
      $('rchSeries').innerHTML = state.series.map(s => `<option value="${esc(s.id)}">${esc(s.title)}</option>`).join('');
      const chosen = state.series.find(s => s.id === state.sourceId && state.area === a) || state.series.find(s => s.lineage) || state.series[0];
      $('rchSeries').value = chosen.id;
      $('rchRootNote').hidden = !state.external;
      $('rchRootNote').textContent = 'External evidence root · read-only. Experiment actions below operate on this checkout, not the external records.';
      await selectSeries(chosen.id, token, signal);
    } catch (error) {
      if (token === seq) unavailable(error.message || 'The evidence service is unavailable.');
    } finally { if (token === seq) $('rchStage').setAttribute('aria-busy', 'false'); }
  }
  async function selectSeries(id, existingToken, existingSignal) {
    const token = existingToken ?? begin(), signal = existingSignal || controller.signal;
    pause();
    try {
      const selected = state.series.find(s => s.id === id);
      if (!selected) throw new Error('Unknown evidence series');
      const changed = state.sourceId !== id || state.area !== area();
      state.sourceId = id; state.area = area(); state.run = null;
      if (changed) { state.points.clear(); cache.clear(); state.selected = null; pan = { x: 0, y: 0, scale: 1 }; }
      if (selected.lineage) {
        const run = await json(`/api/studio/research-run?area=${encodeURIComponent(state.area)}&id=${encodeURIComponent(id)}`, signal);
        if (token !== seq) return;
        if (!run || !Array.isArray(run.frames) || !run.frames.length) throw new Error('This experiment has no verifiable graph lineage.');
        state.run = run; state.frames = run.frames;
      } else {
        state.frames = selected.frames.map(f => ({ ...f, k: null, before_version: null }));
      }
      syncLabRun(id);
      await showFrame(changed || $('rchFollow').checked ? state.frames.length - 1 : Math.min(state.index, state.frames.length - 1), token, signal);
    } catch (error) { if (token === seq) unavailable(error.message || 'The selected experiment could not be read.'); }
  }
  async function snapshot(version, signal) {
    const requestedArea = state.area, key = `${requestedArea}:${version}`;
    if (cache.has(key)) return cache.get(key);
    const payload = C.graph(await json(`/api/studio/snapshot?area=${encodeURIComponent(requestedArea)}&version=${encodeURIComponent(version)}`, signal));
    if (payload.area !== requestedArea || payload.version !== version) throw new Error('Snapshot identity does not match the requested policy pin');
    cache.set(key, payload); if (cache.size > 20) cache.delete(cache.keys().next().value);
    return payload;
  }
  async function showFrame(index, existingToken, existingSignal) {
    if (!state.frames.length) return;
    const token = existingToken ?? begin(), signal = existingSignal || controller.signal;
    index = Math.max(0, Math.min(index, state.frames.length - 1));
    const f = state.frames[index]; $('rchStage').setAttribute('aria-busy', 'true');
    try {
      const [graph, before] = await Promise.all([snapshot(f.version, signal), f.before_version ? snapshot(f.before_version, signal) : Promise.resolve(null)]);
      if (token !== seq) return;
      state.index = index; state.graph = graph; state.previous = before;
      if (state.selected && !graph.nodes.some(n => n.id === state.selected) && !before?.nodes.some(n => n.id === state.selected)) state.selected = null;
      $('rchEmpty').hidden = true; $('rchExport').disabled = false;
      render();
    } catch (error) { if (token === seq) unavailable(error.message || 'The selected policy snapshot is unavailable.'); }
    finally { if (token === seq) $('rchStage').setAttribute('aria-busy', 'false'); }
  }
  function type(n) {
    const text = `${n.node_type} ${n.id}`.toLowerCase();
    return text.includes('root') ? 'root' : text.includes('exception') || text.includes('negative') ? 'exception' : text.includes('boundary') || text.includes('confus') ? 'boundary' : 'rule';
  }
  function camera() { $('rchCamera')?.setAttribute('transform', `translate(${pan.x},${pan.y}) scale(${pan.scale})`); }
  function drawGraph() {
    cancelAnimationFrame(animation);
    if (!state.graph) return;
    const g = state.graph, diff = C.difference(state.previous, g);
    const add = new Set(state.previous ? diff.added.map(n => n.id) : []), edit = new Set(diff.changed.map(n => n.id));
    const retired = state.previous ? diff.removed : [];
    const all = [...g.nodes, ...retired];
    const targets = C.initialLayout(all, state.layout);
    for (const [id, target] of targets) {
      if (!state.points.has(id)) {
        const n = all.find(x => x.id === id), parent = state.points.get(n?.parent);
        state.points.set(id, parent && add.has(id) ? {x: parent.x + 14, y: parent.y + 14} : {...target});
      }
    }
    for (const id of state.points.keys()) if (!targets.has(id)) state.points.delete(id);
    if (state.layout === 'hierarchy') state.points = new Map([...targets].map(([id, p]) => [id, {...p}]));
    else if (reduced.matches) C.relax(state.points, g.edges, targets, 70);
    const connected = new Set([state.selected]);
    if (state.selected) g.edges.forEach(e => { if (e.source === state.selected) connected.add(e.target); if (e.target === state.selected) connected.add(e.source); });
    const query = $('rchSearch').value.trim().toLowerCase();
    const matches = new Set(all.filter(n => `${n.title} ${n.id} ${n.body}`.toLowerCase().includes(query)).map(n => n.id));
    const degree = new Map(all.map(n => [n.id, 0])); g.edges.forEach(e => {degree.set(e.source, degree.get(e.source) + 1); degree.set(e.target, degree.get(e.target) + 1);});
    const showCross = $('rchRelations').checked, labels = $('rchLabels').checked;
    $('rchSvg').innerHTML = `<defs><pattern id="rchGrid" width="25" height="25" patternUnits="userSpaceOnUse"><circle cx="1" cy="1" r=".7" fill="currentColor" opacity=".17"/></pattern><marker id="rchArrow" viewBox="0 0 10 10" refX="18" refY="5" markerWidth="4" markerHeight="4" orient="auto"><path d="M0 0L10 5L0 10" fill="#657889"/></marker></defs><rect width="950" height="580" fill="url(#rchGrid)"/><g id="rchCamera"><g class="rch-edges">${g.edges.filter(e => showCross || e.type === 'subtype_of').map(e => `<line data-edge-source="${esc(e.source)}" data-edge-target="${esc(e.target)}" class="${e.type === 'subtype_of' ? 'hierarchy-edge' : 'relation-edge'} ${e.synthetic ? 'synthetic-edge' : ''} ${state.selected && e.source !== state.selected && e.target !== state.selected ? 'dim' : ''}" marker-end="url(#rchArrow)"><title>${esc(e.source)} → ${esc(e.target)} · ${esc(e.type)}${e.synthetic ? ' · synthetic' : ''}</title></line>`).join('')}</g><g class="rch-nodes">${all.map(n => {
      const gone = retired.some(x => x.id === n.id), kind = type(n), d = degree.get(n.id) || 0;
      const radius = kind === 'root' ? 18 : $('rchSize').value === 'degree' ? Math.min(19, 7 + Math.sqrt(d) * 2) : kind === 'rule' ? 9 : 8;
      const dim = (query && !matches.has(n.id)) || (state.selected && !connected.has(n.id));
      const show = labels || state.selected === n.id || add.has(n.id) || kind === 'root' || (query && matches.has(n.id));
      return `<g class="rch-node ${kind} ${dim ? 'dim' : ''} ${gone ? 'retired' : ''} ${add.has(n.id) ? 'added' : ''} ${edit.has(n.id) ? 'edited' : ''} ${state.selected === n.id ? 'selected' : ''}" data-rch-node="${esc(n.id)}" tabindex="0" role="button" aria-pressed="${state.selected === n.id}" aria-label="${esc(n.title)}${gone ? ', retired' : ''}"><title>${esc(n.id)} · ${esc(n.node_type)} · degree ${d}</title><circle class="rch-aura" r="${radius + 8}"/><circle class="rch-hit" r="24"/><circle class="rch-dot" r="${radius}"/><circle class="rch-center" r="2.5"/>${show ? `<text y="${radius + 20}" text-anchor="middle">${esc(short(n.title, 27))}</text>` : ''}${add.has(n.id) ? `<text class="rch-tag" x="${radius + 6}" y="-${radius + 3}">+</text>` : edit.has(n.id) ? `<text class="rch-tag" x="${radius + 6}" y="-${radius + 3}">Δ</text>` : ''}</g>`;
    }).join('')}</g></g>`;
    camera();
    const nodeElements = [...$('rchSvg').querySelectorAll('[data-rch-node]')];
    const edgeElements = [...$('rchSvg').querySelectorAll('[data-edge-source]')];
    const tick = () => {
      for (const el of nodeElements) {const p = state.points.get(el.dataset.rchNode); el.setAttribute('transform', `translate(${p.x.toFixed(2)},${p.y.toFixed(2)})`);}
      for (const el of edgeElements) {const a = state.points.get(el.dataset.edgeSource), b = state.points.get(el.dataset.edgeTarget); el.setAttribute('x1', a.x); el.setAttribute('y1', a.y); el.setAttribute('x2', b.x); el.setAttribute('y2', b.y);}
    };
    tick(); let ticks = 0;
    const animate = () => { C.relax(state.points, g.edges, targets, 2); tick(); if (++ticks < 75) animation = requestAnimationFrame(animate); };
    if (!reduced.matches && state.layout === 'network') animation = requestAnimationFrame(animate);
    for (const el of nodeElements) {
      const select = () => { state.selected = el.dataset.rchNode; drawGraph(); inspector(); };
      el.addEventListener('click', select);
      el.addEventListener('keydown', e => {if (e.key === 'Enter' || e.key === ' ') {e.preventDefault(); select();}});
    }
    $('rchMatch').textContent = query ? `${matches.size} matching nodes` : `${g.nodes.length} nodes · ${g.edges.length} explicit edges`;
    $('rchGraphHeading').textContent = state.layout === 'hierarchy' ? 'Policy hierarchy' : 'Policy knowledge graph';
    $('rchVersion').textContent = `${state.area} / ${frame().version}`;
    $('rchDelta').innerHTML = state.previous ? `<span class="rch-add">+${diff.added.length} added</span><span class="rch-change">Δ ${diff.changed.length} edited</span><span class="rch-remove">−${diff.removed.length} retired</span>` : '<span>No parent comparison asserted</span>';
  }
  function trajectory() {
    const cycles = (state.run?.cycles || []).filter(c => Number.isInteger(c.k)).sort((a, b) => a.k - b.k);
    const rows = cycles.map(c => ({ k:c.k, status:state.frames.find(f => f.k === c.k)?.status,
      metrics:C.fromCounts(c.metrics?.test?.system, classes()) || c.metrics?.test?.system || {} }));
    if (!rows.some(r => C.finite(r.metrics.macro_fpr) || C.finite(r.metrics.macro_fnr))) return '';
    const maxK = Math.max(1, ...rows.map(r => r.k)), x = k => 30 + k / maxK * 230, y = v => 137 - v * 110;
    return `<div class="rch-trajectory"><h4>Evaluated candidates · development</h4><svg viewBox="0 0 286 172" role="img" aria-label="Saved candidate FPR and FNR by cycle; rejected candidates are hollow markers"><g class="rch-trajectory-grid">${[0,.5,1].map(v => `<line x1="30" y1="${y(v)}" x2="262" y2="${y(v)}"/><text x="22" y="${y(v)+3}" text-anchor="end">${v*100}%</text>`).join('')}</g>${['macro_fpr','macro_fnr'].map((key,j) => {
      const valid = rows.filter(r => C.finite(r.metrics[key]) && r.metrics[key] >= 0 && r.metrics[key] <= 1);
      return `<g class="rch-rate-${j}">${valid.map(r => `<circle cx="${x(r.k)}" cy="${y(r.metrics[key])}" r="${r.k === frame().k ? 4.5 : 3}" class="${['accepted','baseline'].includes(r.status) ? 'promoted' : ''}"><title>k=${r.k} ${key}: ${pct(r.metrics[key])}; ${esc(r.status || 'status unknown')}</title></circle>`).join('')}</g>`;
    }).join('')}<line class="rch-trajectory-cursor" x1="${x(frame().k || 0)}" x2="${x(frame().k || 0)}" y1="23" y2="141"/><text x="30" y="157">k=0</text><text x="260" y="157" text-anchor="end">k=${maxK}</text></svg><div class="rch-trajectory-key"><span>● FPR</span><span>● FNR</span></div><p class="rch-footnote">Each point is an evaluated candidate. Hollow markers were not promoted. Not an incumbent learning curve or a final test result.</p></div>`;
  }
  function inspector() {
    if (!state.graph) return;
    const g = state.graph, diff = C.difference(state.previous, g), selected = state.selected;
    if (!selected) {
      const focus = state.previous ? [...diff.added, ...diff.changed, ...diff.removed] : g.nodes.filter(n => type(n) !== 'root').slice(0, 8);
      $('rchInspector').innerHTML = `<div class="rch-eyebrow">RULE INSPECTOR</div><h3>What changed—and why?</h3><p>Select a node to read the actual policy, compare the previous wording, and trace its explicit relations.</p>${trajectory()}<div class="rch-inspector-list">${focus.slice(0, 24).map(n => `<button type="button" data-inspect="${esc(n.id)}"><span class="rch-mini-dot ${type(n)}"></span><span>${esc(n.title)}<small>${esc(n.id)}</small></span><span aria-hidden="true">↗</span></button>`).join('') || '<p>No node additions, edits, or retirements at this step.</p>'}</div><p class="rch-footnote">Node size can encode degree—not decision quality. A citation is an association, not causal attribution.</p>`;
    } else {
      const n = g.nodes.find(n => n.id === selected) || state.previous?.nodes.find(n => n.id === selected);
      if (!n) {state.selected = null; inspector(); return;}
      const retired = !g.nodes.some(x => x.id === n.id);
      const before = state.previous?.nodes.find(x => x.id === n.id), path = C.ancestry(n.id, retired ? state.previous.nodes : g.nodes);
      const links = g.edges.filter(e => e.source === n.id || e.target === n.id);
      $('rchInspector').innerHTML = `<button type="button" id="rchClear">← All rules</button><div class="rch-eyebrow">${esc(n.id)}</div><h3>${esc(n.title)}</h3><div class="rch-chip-row"><span class="rch-chip">${esc(n.node_type || 'guideline')}</span><span class="rch-chip">${esc(retired ? 'retired at this step' : n.status || 'status not recorded')}</span></div><div class="rch-ancestry">${path.path.map(x => `<button type="button" data-inspect="${esc(x.id)}">${esc(short(x.title, 20))}</button>`).join('<span>›</span>')}</div>${path.cycle ? '<p class="rch-error">Cycle detected in parent metadata.</p>' : ''}<h4>${retired ? 'Retired rule text (previous version)' : 'Current rule text'}</h4><pre class="rch-rule">${esc(n.body || 'No rule body was recorded.')}</pre>${before && before.body !== n.body ? `<details open><summary>Previous wording · ${esc(frame().before_version)}</summary><pre class="rch-rule rch-before">${esc(before.body)}</pre></details>` : ''}<details><summary>Explicit relations (${links.length})</summary>${links.map(e => `<p class="rch-relation">${esc(e.type)}<br><button type="button" data-inspect="${esc(e.source === n.id ? e.target : e.source)}">${esc(e.source === n.id ? e.target : e.source)}</button>${e.synthetic ? ' (synthetic)' : ''}</p>`).join('')}</details><p class="rch-footnote">Source: <code>${esc(n.source || `${state.area}/${frame().version}/${n.id}.md`)}</code></p>`;
    }
    $('rchInspector').querySelectorAll('[data-inspect]').forEach(el => el.addEventListener('click', () => {state.selected = el.dataset.inspect; drawGraph(); inspector();}));
    $('rchClear')?.addEventListener('click', () => {state.selected = null; drawGraph(); inspector();});
  }
  function renderEvidence() {
    const run = state.run, c = cycle(), row = c?.metrics?.test?.system;
    const corrected = C.fromCounts(row, classes());
    const metrics = corrected || row || {};
    const coverage = corrected?.coverage ?? (C.finite(row?.n) && C.finite(row?.n_abstained) && row.n + row.n_abstained > 0 ? row.n / (row.n + row.n_abstained) : null);
    const blocks = [ ['FPR · macro', pct(metrics.macro_fpr), 'one-vs-rest'], ['FNR · macro', pct(metrics.macro_fnr), 'one-vs-rest'], ['Decision coverage', pct(coverage), 'non-abstaining fraction'], ['Scored decisions', number(metrics.n), 'abstentions excluded'] ];
    $('rchStats').innerHTML = blocks.map(([label, value, note]) => `<div><span>${label}</span><strong>${value}</strong><small>${note}</small></div>`).join('');
    const f = frame(), baseline = f.status === 'baseline';
    $('rchGateStatus').textContent = baseline ? 'BASELINE' : String(f.status || 'SNAPSHOT').toUpperCase();
    $('rchGateStatus').className = `rch-chip ${f.status === 'accepted' ? 'rch-add' : ['rejected','skipped'].includes(f.status) ? 'rch-remove' : ''}`;
    $('rchEvidenceTitle').textContent = run ? `k=${f.k} · ${f.status === 'accepted' || baseline ? 'evaluated policy' : 'candidate metrics; incumbent graph retained'}` : 'Snapshot catalog · not an experiment lineage';
    $('rchEvidenceNote').textContent = run ? `${corrected ? 'Recomputed from saved confusion counts.' : 'Stored development-validation measurements; unverified legacy F1 is not promoted here.'} These are not an untouched test result.` : 'Select a recorded run for cycle-level metrics. Adjacent catalog versions are not assumed to be parent and child.';
    const cfg = run?.config || {}, audit = run?.split_audit || {};
    const overlap = audit.train_gate_overlap;
    $('rchAudit').innerHTML = `<div><span class="rch-eyebrow">EXPERIMENT CONTRACT</span><h4>${run ? `Run ${esc(cfg.run_number ?? '—')} · seed ${esc(cfg.seed ?? '—')}` : 'A policy is a hypothesis.'}</h4><p>${run ? `${esc(cfg.strategy ?? 'strategy not recorded')} · ${esc(cfg.gate_mode ?? 'gate mode not recorded')}` : 'The lab below configures judges, drafts bounded edits, evaluates candidates, and records the gate.'}</p><p class="rch-footnote">${run ? `${esc((cfg.judge_models || []).join(' · '))}` : 'No model calls are made by this evidence viewer.'}</p></div><div><span class="rch-eyebrow">SPLIT AUDIT</span><h4>${overlap === null || overlap === undefined ? 'Not established' : overlap ? `${overlap} overlapping IDs` : '0 overlapping recorded IDs'}</h4><p>${esc(audit.scope || 'No experimental split identifiers are attached to a standalone snapshot.')}</p></div><div><span class="rch-eyebrow">GATE EVIDENCE</span><h4>${c ? `${number(c.n_misaligned)} misalignments` : 'No cycle selected'}</h4><p>${f.status === 'accepted' ? 'The record accepted this update. Acceptance is not a generalization guarantee.' : f.status === 'baseline' ? 'Reference policy before any edit in this run.' : run ? 'This step did not promote a new policy. Its candidate score does not replace the incumbent.' : 'No gate is inferred from a version name.'}</p>${c?.gate ? `<details><summary>Inspect saved gate record</summary><pre class="rch-rule">${esc(JSON.stringify(c.gate, null, 2))}</pre></details>` : ''}</div>`;
    const warnings = [...(run?.warnings || []), ...(state.graph?.warnings || [])];
    message(warnings.length ? warnings.join(' ') : `Recorded artifacts · ${run ? 'explicit within-run lineage' : 'catalog only'} · graph position and size are structural, not quality scores.`);
  }
  function render() {
    drawGraph(); inspector(); renderEvidence();
    $('rchPrev').disabled = state.index === 0; $('rchNext').disabled = state.index >= state.frames.length - 1;
    $('rchRange').disabled = state.frames.length < 2; $('rchPlay').disabled = state.frames.length < 2;
    $('rchRange').max = Math.max(0, state.frames.length - 1); $('rchRange').value = state.index;
    $('rchStep').textContent = `${state.index + 1} / ${state.frames.length}`;
    $('rchTimeline').innerHTML = state.frames.map((f, i) => `<button type="button" data-frame="${i}" aria-current="${i === state.index ? 'step' : 'false'}" class="${f.status === 'accepted' ? 'accepted' : ['rejected','skipped'].includes(f.status) ? 'rejected' : ''}"><small>${esc(f.k === null ? 'snapshot' : `k=${f.k}`)}</small><b>${esc(f.version)}</b><span>${esc(f.status || 'snapshot')}</span></button>`).join('');
    $('rchTimeline').querySelectorAll('[data-frame]').forEach(b => b.addEventListener('click', () => {pause(); $('rchFollow').checked = false; showFrame(Number(b.dataset.frame));}));
    const active = $('rchTimeline').querySelector('[aria-current="step"]');
    if (active) { const rail = $('rchTimeline'); if (active.offsetLeft < rail.scrollLeft || active.offsetLeft + active.offsetWidth > rail.scrollLeft + rail.clientWidth) rail.scrollLeft = Math.max(0, active.offsetLeft - rail.offsetLeft - 24); }
  }
  async function play() {
    if (state.playing) {pause(); return;}
    $('rchFollow').checked = false; state.playing = true; $('rchPlay').textContent = 'Ⅱ Pause';
    if (state.index === state.frames.length - 1) await showFrame(0);
    const step = async () => {
      if (!state.playing) return;
      if (state.index >= state.frames.length - 1) {pause(); return;}
      await showFrame(state.index + 1);
      if (state.playing) timer = setTimeout(step, 2000);
    };
    if (state.playing) timer = setTimeout(step, 2000);
  }
  function exportEvidence() {
    if (!state.graph) return;
    const out = { schema_version: 1, exported_at: new Date().toISOString(), origin: 'recorded',
      frame: frame(), graph: state.graph, comparison: state.previous, experiment: state.run,
      note: 'Read-only evidence export, not a model output or deployment approval.' };
    const url = URL.createObjectURL(new Blob([JSON.stringify(out, null, 2)], {type: 'application/json'}));
    const a = document.createElement('a'); a.href = url; a.download = `rush-evidence-${state.area}-${frame().version}.json`; a.click(); setTimeout(() => URL.revokeObjectURL(url), 1000);
  }
  function mount() {
    document.body.classList.add('rush-research');
    $('heroEyebrow').textContent = 'RUSH / POLICY LEARNING RESEARCH LAB';
    $('heroH1').textContent = 'Learn the rule. Test the generalization.';
    $('heroLede').textContent = 'Golden sets → disagreement → bounded policy edits → measured decisions. Inspect the mechanism, not just the final score.';
    const names = {loop:'Experiment lab',summary:'Image evidence',adjudicate:'SME review',benchmarks:'Replication bench',about:'Methods & hypotheses'};
    document.querySelectorAll('#viewSwitcher [data-view]').forEach(b => {if (names[b.dataset.view]) b.textContent = names[b.dataset.view];});
    const section = document.createElement('section'); section.id = 'researchWorkbench'; section.className = 'rch-workbench';
    section.setAttribute('aria-label', 'Policy learning research workbench');
    section.innerHTML = `<header class="rch-workbench-head"><div><span class="rch-eyebrow">01 / POLICY DYNAMICS</span><h2>The experiment, made inspectable.</h2></div><div class="rch-source"><span class="rch-recorded"><i></i> Recorded evidence</span><label>Experiment / snapshots<select id="rchSeries" aria-label="Recorded experiment or snapshot catalog"></select></label><button type="button" id="rchRefresh" title="Refresh saved evidence; no model calls">↻ Refresh</button></div></header><p id="rchRootNote" class="rch-root-note" hidden></p>
      <div class="rch-toolbar"><div class="rch-segment" aria-label="Graph layout"><button type="button" data-layout="network" aria-pressed="true">Knowledge graph</button><button type="button" data-layout="hierarchy" aria-pressed="false">Hierarchy</button></div><label class="rch-search"><span aria-hidden="true">⌕</span><input id="rchSearch" type="search" placeholder="Find a rule, boundary, or phrase…" aria-label="Search policy nodes"/></label><label>Size<select id="rchSize"><option value="uniform">Node type</option><option value="degree">Graph degree</option></select></label><label><input id="rchRelations" type="checkbox" checked/> Cross-links</label><label><input id="rchLabels" type="checkbox" checked/> Labels</label></div>
      <div id="rchStage" class="rch-stage" aria-busy="true"><div class="rch-canvas"><div class="rch-canvas-head"><div><h3 id="rchGraphHeading">Policy knowledge graph</h3><code id="rchVersion">Reading artifacts…</code></div><span id="rchMatch" class="rch-footnote"></span></div><svg id="rchSvg" viewBox="0 0 950 580" aria-label="Interactive recorded policy knowledge graph" role="group"></svg><div id="rchEmpty" class="rch-empty"><h3>Reading recorded policy evidence…</h3></div><div class="rch-canvas-bottom"><div id="rchDelta" class="rch-chip-row"></div><div class="rch-camera-controls"><button type="button" id="rchZoomOut" aria-label="Zoom out">−</button><button type="button" id="rchZoomIn" aria-label="Zoom in">+</button><button type="button" id="rchFit">Fit</button><button type="button" id="rchFull" aria-label="Toggle fullscreen graph">⛶</button></div></div><div class="rch-legend"><span><i class="root"></i>Policy root</span><span><i class="rule"></i>Rule / criterion</span><span><i class="boundary"></i>Boundary</span><span><i class="exception"></i>Exception / negative</span><span>+ addition · Δ edit · dashed retirement</span></div></div><aside id="rchInspector" class="rch-inspector" aria-label="Policy node inspector"></aside></div>
      <div class="rch-playback"><div><span class="rch-eyebrow">EXPLICIT LINEAGE</span><span id="rchStep"></span></div><div class="rch-play-actions"><label><input id="rchFollow" type="checkbox"/> Follow saved updates</label><button type="button" id="rchPrev" aria-label="Previous recorded step">←</button><button type="button" id="rchPlay">▶ Replay</button><button type="button" id="rchNext" aria-label="Next recorded step">→</button></div></div><div id="rchTimeline" class="rch-timeline" aria-label="Recorded policy development steps"></div><input id="rchRange" class="rch-range" type="range" min="0" max="0" value="0" aria-label="Recorded policy step"/>
      <div class="rch-evidence-head"><div><span id="rchGateStatus" class="rch-chip">BASELINE</span><strong id="rchEvidenceTitle"></strong></div><button type="button" id="rchExport" disabled>Export evidence ↗</button></div><div id="rchStats" class="rch-stats"></div><p id="rchEvidenceNote" class="rch-footnote"></p><div id="rchAudit" class="rch-audit"></div><p id="rchStatus" class="rch-status" role="status" aria-live="polite"></p></section>`;
    const experiment = $('experiment'); experiment.insertBefore(section, experiment.querySelector('.experiment-config'));
    const intro = experiment.querySelector('.section-head'); if (intro) intro.hidden = true;
    const oldGraph = $('policyEvolution');
    if (oldGraph) {const detail = document.createElement('details'); detail.className = 'rch-original'; detail.innerHTML = '<summary>Original policy inspector & node-change audit</summary>'; oldGraph.before(detail); detail.append(oldGraph);}
    const notice = document.createElement('p'); notice.className = 'rch-method-note'; notice.textContent = 'Measurement note: the repeatedly consulted test split is development validation. Historical scores are retained as recorded; pre-correction macro-F1 requires re-scoring before comparison. Interpret every conditional score with its decision coverage.'; $('experimentChart')?.before(notice);
    const nav = document.createElement('nav'); nav.className = 'rch-jumps'; nav.setAttribute('aria-label', 'Research panels');
    nav.innerHTML = '<span>RESEARCH BENCH</span><a href="#researchWorkbench">Policy dynamics</a><a href="#rchConfig">Configure experiment</a><a href="#experimentChart">Learning curves</a><a href="#experimentJudgeTable">Judge diagnostics</a><a href="#experimentLedger">Gate ledger</a><a href="#experimentConfusion">Confusion</a>';
    section.before(nav); experiment.querySelector('.experiment-config')?.setAttribute('id', 'rchConfig');
    const sandbox = document.createElement('a'); sandbox.className = 'nav-pill'; sandbox.id = 'rchSandbox'; sandbox.href = `studio.html?demo=${area() === 'MNIST_Digits' ? 'mnist' : 'genai'}`; sandbox.textContent = 'Shadow path sandbox ↗'; sandbox.title = 'Separate illustrative decision programs—not production execution traces'; document.querySelector('.topbar-action-links')?.prepend(sandbox);
    $('rchRefresh').addEventListener('click', () => {cache.clear(); refresh();});
    $('rchSeries').addEventListener('change', e => selectSeries(e.target.value));
    $('rchSearch').addEventListener('input', drawGraph);
    ['rchSize','rchRelations','rchLabels'].forEach(id => $(id).addEventListener('change', drawGraph));
    document.querySelectorAll('[data-layout]').forEach(b => b.addEventListener('click', () => {state.layout = b.dataset.layout; state.points.clear(); document.querySelectorAll('[data-layout]').forEach(x => x.setAttribute('aria-pressed', String(x === b))); drawGraph();}));
    $('rchPlay').addEventListener('click', play);
    $('rchRange').addEventListener('input', e => {pause(); $('rchFollow').checked = false; showFrame(Number(e.target.value));});
    $('rchPrev').addEventListener('click', () => {pause(); $('rchFollow').checked = false; showFrame(state.index - 1);});
    $('rchNext').addEventListener('click', () => {pause(); $('rchFollow').checked = false; showFrame(state.index + 1);});
    $('rchExport').addEventListener('click', exportEvidence);
    $('rchZoomIn').addEventListener('click', () => {pan.scale = Math.min(4, pan.scale * 1.2); camera();});
    $('rchZoomOut').addEventListener('click', () => {pan.scale = Math.max(.4, pan.scale / 1.2); camera();});
    $('rchFit').addEventListener('click', () => {pan = {x:0,y:0,scale:1}; camera();});
    $('rchFull').addEventListener('click', async () => {try {if (document.fullscreenElement) await document.exitFullscreen(); else await $('rchStage').requestFullscreen();} catch (_) {message('Fullscreen is not available in this browser.');}});
    $('rchSvg').addEventListener('pointerdown', e => {if (!e.target.closest('[data-rch-node]')) {pointer = {x:e.clientX,y:e.clientY,panX:pan.x,panY:pan.y}; $('rchSvg').setPointerCapture(e.pointerId);}});
    $('rchSvg').addEventListener('pointermove', e => {if (pointer) {const rect = $('rchSvg').getBoundingClientRect(), unit = 950 / rect.width; pan.x = pointer.panX + (e.clientX - pointer.x) * unit; pan.y = pointer.panY + (e.clientY - pointer.y) * unit; camera();}});
    ['pointerup','pointercancel','lostpointercapture'].forEach(name => $('rchSvg').addEventListener(name, () => {pointer = null;}));
    $('rchSvg').addEventListener('wheel', e => {if (!e.ctrlKey && !e.metaKey) return; e.preventDefault(); pan.scale = Math.max(.4, Math.min(4, pan.scale * (e.deltaY > 0 ? .92 : 1.08))); camera();}, {passive:false});
    $('experimentSelect')?.addEventListener('change', e => {if (!state.external && state.series.some(s => s.id === e.target.value) && state.sourceId !== e.target.value) {$('rchSeries').value = e.target.value; selectSeries(e.target.value);}});
    const route = () => {const hash = location.hash.slice(1); const target = hash.startsWith('method-') ? 'about' : ['researchWorkbench','rchConfig','experimentChart','experimentJudgeTable','experimentLedger','experimentConfusion'].includes(hash) ? 'loop' : null; if (target) document.querySelector(`#viewSwitcher [data-view="${target}"]`)?.click();};
    window.addEventListener('hashchange', route); route();
    window.addEventListener('rush-policy-accepted', () => {cache.clear(); if ($('rchFollow').checked) refresh();});
    document.addEventListener('visibilitychange', () => {if (document.hidden) pause();});
    const polling = setInterval(() => {if (!document.hidden && document.body.classList.contains('view-loop') && $('rchFollow').checked && !state.playing) {cache.clear(); refresh();}}, 8000);
    window.addEventListener('pagehide', () => {clearInterval(polling); pause(); controller?.abort(); cancelAnimationFrame(animation);});
    if (typeof window.rushApiOnReady === 'function') window.rushApiOnReady(() => {
      $('heroH1').textContent = 'Learn the rule. Test the generalization.';
      $('heroLede').textContent = 'Golden sets → disagreement → bounded policy edits → measured decisions. Inspect the mechanism, not just the final score.';
    });
    refresh();
  }
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', mount, {once:true}); else mount();
})();
