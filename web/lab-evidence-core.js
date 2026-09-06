/* Pure contracts for the native experiment evidence view. */
(function(root, make) { const api=make(); if(typeof module==='object'&&module.exports)module.exports=api;else root.RushEvidence=api; })(globalThis, () => {
  'use strict';
  const finite = v => typeof v === 'number' && Number.isFinite(v);
  const pct = v => finite(v) ? `${(100*v).toFixed(1)}%` : '—';
  const version = (v, area) => { if(typeof v!=='string')return null;const p=v.startsWith(area+'.')?v.slice(area.length+1):v;return /^v\d+\.\d+$/.test(p)?p:null; };
  function frames(run) {
    if(!run||!Array.isArray(run.cycles))throw new Error('Invalid native experiment response');
    let current=version(run.base_version,run.area);
    if(!current)throw new Error('Missing baseline version');
    const out=[],seen=new Set();let last=-1;
    for(const c of run.cycles) {
      if(!Number.isInteger(c.k)||c.k<0||seen.has(c.k)||c.k<=last)throw new Error('Duplicate or unordered cycle');
      seen.add(c.k);last=c.k;
      if(c.status==='open')break;
      const before=version(c.generator_before,run.area);
      if(before&&before!==current)throw new Error(`Broken lineage at k=${c.k}`);
      const previous=current;
      if(c.status==='accepted') {
        const after=version(c.new_version,run.area)||version(c.generator_after,run.area);
        if(!after||after===current)throw new Error(`Missing accepted version at k=${c.k}`);
        current=after;
      } else {
        const after=version(c.generator_after,run.area);
        if(after&&after!==current)throw new Error(`Unaccepted transition at k=${c.k}`);
      }
      out.push({k:c.k,version:current,previous,cycle:c});
    }
    return out;
  }
  function evidence(cycle) {
    return (cycle?.edit_summary||[]).filter(e=>e&&typeof e.path==='string').map(e=>({
      id:e.path.split('/').pop().replace(/\.md$/,''), path:e.path, change:String(e.change||'modified')
    }));
  }
  function metrics(cycle) {
    const m=cycle?.metrics?.test?.system||{};
    const coverage=finite(m.n)&&finite(m.n_abstained)&&m.n>=0&&m.n_abstained>=0&&m.n+m.n_abstained>0?m.n/(m.n+m.n_abstained):null;
    return {fpr:m.macro_fpr,fnr:m.macro_fnr,n:m.n,coverage};
  }
  function targets(vote) {
    const raw=vote?.policy_node_ids||vote?.cited_node_ids||vote?.policy_citations||vote?.citations||[];
    return (Array.isArray(raw)?raw:[]).map(x=>typeof x==='string'?x:x?.node_id||x?.id).filter(Boolean);
  }
  return {finite,pct,version,frames,evidence,metrics,targets};
});
