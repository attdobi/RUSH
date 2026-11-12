import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { webcrypto } from 'node:crypto';
import { DemoClient } from '../web/labeling/demo.js';

const fixture = JSON.parse(await readFile(new URL('../web/labeling/fixtures.json', import.meta.url), 'utf8'));
if (!globalThis.crypto) Object.defineProperty(globalThis, 'crypto', {value:webcrypto});
const storage = new Map();
globalThis.localStorage = {
  getItem: key => storage.get(key) ?? null,
  setItem: (key,value) => storage.set(key,value),
  removeItem: key => storage.delete(key),
};
globalThis.fetch = async url => {
  assert.equal(url, 'fixtures.json');
  return {ok:true, json:async () => structuredClone(fixture)};
};

const client = await new DemoClient().init();
const initial = await client.bootstrap();
assert.equal(initial.epoch,42);
assert.equal(initial.stats.total,fixture.items.length);
assert.equal(initial.mode,'browser-demo');
assert.deepEqual(initial.graph,fixture.graph);
assert.ok((await client.items()).items.every(item=>!item.route?.trace));
const oldManifest = await client.exportEpoch(initial.epoch);
assert.deepEqual(oldManifest.graph,fixture.graph);
assert.equal(oldManifest.graph_hash,fixture.graph.source_sha256);
assert.equal(oldManifest.graph_version,fixture.graph.version);
const blindTraceSource=fixture.items.find(item=>item.route?.trace&&!item.seed_independent_labels);
if(blindTraceSource) assert.ok(!(await client.item(blindTraceSource.id)).route.trace);
const blindSource = fixture.items.find(item => item.type==='assistant' && !item.seed_independent_labels && item.split!=='holdout');
let blind = await client.item(blindSource.id);
assert.equal(blind.guidance_visible,false);
assert.ok(blind.targets.every(target => target.ai===null && target.gold===null));
const firstLabels = Object.fromEntries(blind.targets.map(target => [target.id,target.choices[0]]));
const independent = {base_revision:blind.revision,phase:'independent',decision:'save',labels:firstLabels,rationale:'Independent target-by-target review.',policy_refs:[]};
blind = await client.review(blind.id,independent,'independent-1');
assert.equal(blind.guidance_visible,true);
assert.deepEqual(Object.keys(blind.independent_labels).sort(),['input','output']);
assert.deepEqual(blind.independent_labels,firstLabels);
const replay = await client.review(blind.id,independent,'independent-1');
assert.equal(replay.revision,blind.revision);
await assert.rejects(client.review(blind.id,{...independent,rationale:'Changed'},'independent-1'),error=>error.status===409);
await assert.rejects(client.review(blind.id,independent,'stale-1'),error=>error.status===409);

let selected = await client.item('turn-1042');
assert.equal(selected.guidance_visible,true);
const labels = Object.fromEntries(selected.targets.map(target=>[target.id,target.choices[0]]));
selected=await client.review(selected.id,{base_revision:selected.revision,phase:'adjudication',decision:'approve',labels,rationale:'Resolved both target labels.',policy_refs:[]},'approve-1');
assert.equal(selected.status,'approved');
const epoch=await client.publish({expected_epoch:42,note:'Reviewed corrections'},'publish-1');
assert.equal(epoch.epoch,43);
assert.equal(epoch.status,'local_published');
assert.deepEqual(await client.exportEpoch(42),oldManifest);
const snapshot=await client.exportEpoch(43);
const publishedTurn=snapshot.items.find(item=>item.id==='turn-1042');
assert.deepEqual(publishedTurn.route.trace,fixture.items.find(item=>item.id==='turn-1042').route.trace);
assert.equal(publishedTurn.conversation_id,fixture.items.find(item=>item.id==='turn-1042').conversation_id||fixture.items.find(item=>item.id==='turn-1042').content.conversation_id||null);
for(const protectedItem of fixture.items.filter(item=>item.split==='holdout')) assert.ok(!snapshot.items.some(item=>item.id===protectedItem.id));
const rerun=await client.rerun({epoch:43,note:'Pinned handoff'},'rerun-1');
assert.equal(rerun.status,'awaiting_connector');
assert.equal(rerun.digest,snapshot.digest);
assert.equal(rerun.epoch,43);
const restored=await new DemoClient().init();
assert.equal((await restored.bootstrap()).epoch,43);
assert.deepEqual(await restored.exportEpoch(42),oldManifest);

// A different reviewer's first pass cannot revive or restage published gold.
const reviewerId=client.fixtures.viewer.id;
client.fixtures.viewer.id='another-policy-reviewer';
let otherView=await client.item('turn-1042');
assert.equal(otherView.guidance_visible,false);
assert.ok(!otherView.route.trace);
assert.ok(otherView.events.every(event=>!('rationale' in event)&&!('policy_refs' in event)&&!('labels' in event)&&!('before' in event)&&!('after' in event)));
otherView=await client.review(otherView.id,{base_revision:otherView.revision,phase:'independent',decision:'save',labels,rationale:'A second independent review.',policy_refs:[]},'second-reviewer-first');
assert.equal(otherView.status,'published');
client.fixtures.viewer.id=reviewerId;

// Route correction quarantines current gold but never rewrites the historic trace.
selected=await client.item('turn-1042');
const recordedTrace=structuredClone(selected.route.trace);
selected=await client.route(selected.id,{base_revision:selected.revision,nodes:['HAR-ROOT','HAR-3'],reason:'Add quoted-content context.'},'route-correction');
assert.equal(selected.gold_eligible,false);
assert.equal(selected.route.current_route_changed,true);
assert.equal(selected.route.trace_stale,true);
assert.deepEqual(selected.route.trace,recordedTrace);
assert.deepEqual(await client.exportEpoch(43),snapshot);
assert.ok(!(await client.manifest(44)).items.some(item=>item.id===selected.id));

// Saving a newer draft revokes staged approval and next-epoch eligibility.
selected=await client.review(selected.id,{base_revision:selected.revision,phase:'adjudication',decision:'approve',labels,rationale:'Restaged after checking route.',policy_refs:[]},'restage-1');
assert.ok(selected.targets.every(target=>target.staged_label));
selected=await client.review(selected.id,{base_revision:selected.revision,phase:'adjudication',decision:'save',labels,rationale:'Reopened a policy boundary.',policy_refs:[]},'reopen-1');
assert.equal(selected.status,'open');
assert.equal(selected.gold_eligible,false);
assert.ok(selected.targets.every(target=>target.staged_label===null));
assert.ok(!(await client.manifest(44)).items.some(item=>item.id===selected.id));

const protectedSource=fixture.items.find(item=>item.split==='holdout');
let protectedItem=await client.item(protectedSource.id);
if(!protectedItem.guidance_visible) protectedItem=await client.review(protectedItem.id,{base_revision:protectedItem.revision,phase:'independent',decision:'save',labels:Object.fromEntries(protectedItem.targets.map(target=>[target.id,target.choices[0]])),rationale:'Independent evaluation review.',policy_refs:[]},'holdout-first');
await assert.rejects(client.review(protectedItem.id,{base_revision:protectedItem.revision,phase:'adjudication',decision:'approve',labels:Object.fromEntries(protectedItem.targets.map(target=>[target.id,target.choices[0]])),rationale:'Attempted optimization publication.',policy_refs:[]},'holdout-approve'),error=>error.status===403);
const events=await client.events();
assert.ok(events.events.length>=5);
assert.ok(events.events.every(event=>!('labels' in event)&&!('before' in event)&&!('after' in event)));
assert.ok(events.events.every(event=>!('rationale' in event)&&!('policy_refs' in event)));
client.fixtures.viewer.role='ops';
await assert.rejects(client.rerun({epoch:43,note:'Unauthorized'},'ops-run'),error=>error.status===403);
console.log('PASS browser demo: blind review and trace redaction, separate targets, idempotency, stale revisions, immutable graph-pinned epochs, route quarantine, approval revocation, persistence, protected holdout, metadata audit and role-gated pinned handoff');
