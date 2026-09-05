const test = require('node:test');
const assert = require('node:assert/strict');
const C = require('../web/studio-core.js');
const F = require('../web/studio-fixtures.js');
const clone = x => JSON.parse(JSON.stringify(x));
const fixture = (demo='genai') => {
  const f=F.build(demo),g=f.frames[5];
  return {...f,context:{policy_area:g.area,policy_version:g.version,policy_node_ids:g.nodes.map(n=>n.id)}};
};
for (const demo of ['genai','mnist']) {
  test(`${demo}: every scripted graph has explicit, valid edges`,()=>{
    for(const g of F.build(demo).frames){const n=C.normalizeGraph(g);assert.equal(n.warnings.length,0);assert.ok(n.nodes.length>0);}
  });
  test(`${demo}: rejected shortcut leaves policy content unchanged`,()=>{
    const f=F.build(demo);assert.deepEqual(C.graphDiff(f.frames[2],f.frames[3]),{added:[],changed:[],removed:[]});
  });
  test(`${demo}: all supplied scenarios terminate reproducibly`,()=>{
    const f=fixture(demo);
    for(const s of f.scenarios){const a=C.evaluateProgram(f.program,s.facts,f.context);assert.deepEqual(a,C.evaluateProgram(f.program,s.facts,f.context));assert.equal(a.mode,'shadow');}
  });
  test(`${demo}: missing evidence is unknown, never false`,()=>{
    const f=fixture(demo),r=C.evaluateProgram(f.program,{},f.context);assert.equal(r.action,'sme_review');assert.equal(r.trace[0].outcome,'unknown');
  });
}
test('absent metadata never implies a not-generated verdict',()=>{
  const f=fixture();assert.equal(C.evaluateProgram(f.program,f.scenarios[3].facts,f.context).action,'request_provenance');
});
test('numeric version order and deduplication',()=>assert.deepEqual(C.naturalVersions(['v10.1',{version:'v2.3'},'v2.3','v2.12']),['v2.3','v2.12','v10.1']));
test('graph normalization preserves semantic edge types and removes duplicate edges',()=>{
  const g=C.normalizeGraph({nodes:[{id:'a'},{id:'b'}],edges:[{source:'a',target:'b',type:'subtype_of'},{source_node_id:'a',target_node_id:'b',edge_type:'subtype_of'},{source:'a',target:'b',type:'constrains'}]});assert.equal(g.edges.length,2);
});
test('orphan nodes stay visible without fabricated edges',()=>{
  const g=C.normalizeGraph({nodes:[{id:'root'},{id:'orphan'}],edges:[{source:'root',target:'missing'}]});assert.equal(g.nodes.length,2);assert.equal(g.edges.length,0);assert.equal(g.warnings.length,1);
});
test('duplicate node identities fail closed',()=>assert.throws(()=>C.normalizeGraph({nodes:[{id:'a'},{id:'a'}],edges:[]})));
test('diff detects additions, content changes and removals',()=>{
  assert.deepEqual(C.graphDiff({nodes:[{id:'a',body:'old'},{id:'b'}]},{nodes:[{id:'a',body:'new'},{id:'c'}]}),{added:['c'],changed:['a'],removed:['b']});
});
test('stale version and cross-area pins are rejected',()=>{
  const f=fixture();for(const key of ['policy_version','policy_area'])assert.throws(()=>C.evaluateProgram(f.program,{}, {...f.context,[key]:'wrong'}),/pin mismatch/);
});
test('rule references must resolve against the pinned graph',()=>{
  const f=fixture();assert.throws(()=>C.evaluateProgram(f.program,{}, {...f.context,policy_node_ids:[]}),/missing policy node/);
});
test('cycles fail validation before evaluating any facts',()=>{
  const f=fixture(),p=clone(f.program);p.nodes[0].next.false='first';assert.throws(()=>C.validateProgram(p),/acyclic/);
});
test('dangling paths fail validation',()=>{
  const p=clone(fixture().program);p.nodes[0].next.true='missing';assert.throws(()=>C.validateProgram(p),/Dangling/);
});
test('unknown evidence cannot silently approve',()=>{
  const p=clone(fixture().program);p.nodes[0].next.unknown='generated';assert.throws(()=>C.validateProgram(p),/Unknown evidence/);
});
test('unreachable and duplicate decision nodes fail validation',()=>{
  let p=clone(fixture().program);p.nodes.push({id:'extra',kind:'action',action:'sme_review'});assert.throws(()=>C.validateProgram(p),/Unreachable/);
  p=clone(fixture().program);p.nodes.push(clone(p.nodes[0]));assert.throws(()=>C.validateProgram(p),/Duplicate/);
});
test('numeric and boolean types are never coerced',()=>{
  const base={field:'x',op:'eq',value:true,sources:['sme']};
  for(const v of [1,'true',null,NaN,Infinity])assert.equal(C.predicate(base,{x:{value:v,source:'sme'}}),'unknown');
  assert.equal(C.predicate(base,{x:{value:false,source:'sme'}}),'false');
});
test('only allowlisted sources and confidence thresholds are trusted',()=>{
  const f=fixture(),when=f.program.nodes[0].when;
  for(const obs of [{value:true,source:'model',confidence:1},{value:true,source:'sme',confidence:.4},{value:true,source:'sme',confidence:Infinity},{value:true,source:'sme'}])assert.equal(C.predicate(when,{[when.field]:obs}),'unknown');
});
test('inherited observation fields do not satisfy predicates',()=>{
  const when={field:'x',op:'eq',value:true,sources:['sme']};
  assert.equal(C.predicate(when,Object.create({x:{value:true,source:'sme'}})),'unknown');
  assert.equal(C.predicate(when,{x:Object.create({value:true,source:'sme'})}),'unknown');
});
test('predicate operators are explicit, never executable code',()=>{
  const p=clone(fixture().program);p.nodes[0].when.op='eval';assert.throws(()=>C.validateProgram(p),/Unsupported operator/);
});
test('numeric predicates reject string numbers',()=>{
  const when={field:'n',op:'gte',value:2,sources:['system']};
  assert.equal(C.predicate(when,{n:{value:'3',source:'system'}}),'unknown');assert.equal(C.predicate(when,{n:{value:3,source:'system'}}),'true');
});
test('set membership distinguishes types',()=>{
  const when={field:'n',op:'in',value:[1,2],sources:['sme']};
  assert.equal(C.predicate(when,{n:{value:'1',source:'sme'}}),'unknown');assert.equal(C.predicate(when,{n:{value:2,source:'sme'}}),'true');
});
test('programs cannot be switched to production by changing a flag',()=>{
  const p=clone(fixture().program);p.mode='production';assert.throws(()=>C.validateProgram(p),/shadow/);
});
