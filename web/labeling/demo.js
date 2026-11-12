/* Browser-local demo of the same review lifecycle as the Python API.
 * Fixtures, scores and policies are synthetic. This module never calls RUSH.
 */
const STORAGE_KEY = 'rush-labeling-workspace-v4';
const copy = value => structuredClone(value);
const now = () => new Date().toISOString();
const uid = () => crypto.randomUUID();
function fail(message, code = 'validation_error', status = 422) { const error = new Error(message); error.code = code; error.status = status; throw error; }
async function digest(value) {
  const bytes = new TextEncoder().encode(JSON.stringify(value));
  const hash = await crypto.subtle.digest('SHA-256', bytes);
  return Array.from(new Uint8Array(hash)).map(x => x.toString(16).padStart(2,'0')).join('');
}

export class DemoClient {
  async init() {
    const response = await fetch('fixtures.json');
    if (!response.ok) throw new Error('The example dataset could not be loaded.');
    this.fixtures = await response.json();
    this.fingerprint = await digest(this.fixtures);
    try { this.data = JSON.parse(localStorage.getItem(STORAGE_KEY)); } catch { this.data = null; }
    if (!this.data || this.data.fingerprint !== this.fingerprint) {
      const items = this.fixtures.items.map(source => {
        const item = copy(source);
        item.revision ||= 1;
        item.status ||= 'open';
        item.assignee ||= null;
        item.events ||= [];
        item.gold_eligible = item.split !== 'holdout';
        item.independent = item.seed_independent_labels ? {[this.fixtures.viewer.id]: copy(item.seed_independent_labels)} : {};
        if (item.seed_independent_labels) {
          item.events.push({id:uid(),item_id:item.id,type:'independent',phase:'independent',created_at:item.created_at,actor:copy(this.fixtures.viewer),rationale:'Independent review completed for this illustrative adjudication example.',labels:copy(item.seed_independent_labels),ai_visible:false,policy_version:item.policy_version});
        }
        delete item.seed_independent_labels;
        for (const target of item.targets) target.staged_label = null;
        return item;
      });
      this.data = {fingerprint:this.fingerprint,epoch:this.fixtures.initial_epoch,items,events:[],epochs:[],manifests:{},runs:[],idempotency:{}};
      const manifest = await this.manifest(this.data.epoch);
      this.data.manifests[this.data.epoch] = manifest;
      this.data.epochs = [{epoch:this.data.epoch,created_at:manifest.created_at,created_by:'Example dataset',item_count:manifest.items.length,changed_count:0,digest:manifest.digest,status:'local_published'}];
      this.save();
    }
    return this;
  }
  save() {
    try { localStorage.setItem(STORAGE_KEY,JSON.stringify(this.data)); }
    catch { fail('Browser storage is full or unavailable. Export your current epoch before continuing.','storage_unavailable',507); }
  }
  get viewer() { return this.fixtures.viewer; }
  getItem(id) { const item = this.data.items.find(x => x.id === id); if (!item) fail('This item no longer exists.','not_found',404); return item; }
  checkRevision(item,revision) { if(item.revision !== revision) fail('This item changed since you opened it. Refresh the item and review the latest version.','stale_revision',409); }
  role(...allowed) { if(!allowed.includes(this.viewer.role)) fail('Your reviewer role cannot perform this action.','forbidden',403); }
  visible(item) {
    const result=copy(item);
    result.independent_labels=copy(item.independent[this.viewer.id] || null);
    result.guidance_visible=!!result.independent_labels;
    delete result.independent;
    if(!result.guidance_visible) {
      for(const target of result.targets) { target.ai=null; target.gold=null; target.staged_label=null; }
      if(result.route){delete result.route.trace;delete result.route.override_reason;}
      result.events=result.events.map(event => { const safe=copy(event); for(const key of ['labels','before','after','rationale','comment','policy_refs'])delete safe[key]; return safe; });
    }
    return result;
  }
  summary(item) {
    const result=copy(item);
    result.target_count=result.targets.length;
    delete result.targets;delete result.events;delete result.independent;
    if(result.route){delete result.route.trace;delete result.route.override_reason;}
    return result;
  }
  async bootstrap() {
    const items=this.data.items;
    return copy({viewer:this.viewer,mode:'browser-demo',epoch:this.data.epoch,
      stats:{total:items.length,open:items.filter(x=>['open','in_review'].includes(x.status)).length,escalated:items.filter(x=>x.status==='escalated').length,approved:items.filter(x=>x.status==='approved').length,published:items.filter(x=>x.status==='published').length,gold_items:items.filter(x=>x.gold_eligible&&x.split!=='holdout'&&x.targets.every(t=>t.gold?.label&&t.gold.label!=='Uncertain')).length},
      categories:this.fixtures.categories,projects:this.fixtures.projects,policies:this.fixtures.policies,graph:this.fixtures.graph||null,
      epochs:this.data.epochs,runs:this.data.runs,integrations:{gds:'local',apo:'outbox'}});
  }
  async items(filters={}) {
    let items=this.data.items.filter(item => {
      for(const key of ['type','category','status','project']) if(filters[key]&&item[key]!==filters[key]) return false;
      return !filters.q || `${item.id} ${item.title} ${item.description} ${item.category} ${item.project}`.toLowerCase().includes(filters.q.toLowerCase());
    }).sort((a,b)=>(b.priority?.score||0)-(a.priority?.score||0));
    return {items:items.map(item=>this.summary(item)),total:items.length};
  }
  async item(id) { return this.visible(this.getItem(id)); }
  async mutate(operation,payload,key,callback) {
    const request=JSON.stringify({operation,payload});
    if(key&&this.data.idempotency[key]) {
      const old=this.data.idempotency[key];
      if(old.request!==request) fail('This retry key was already used for a different change.','idempotency_conflict',409);
      return copy(old.result);
    }
    const before=copy(this.data);
    try {
      const result=await callback();
      if(key)this.data.idempotency[key]={request,result:copy(result)};
      this.save();return copy(result);
    } catch(error) { this.data=before; throw error; }
  }
  event(item,type,fields={}) {
    const event={id:uid(),item_id:item?.id||null,type,created_at:now(),actor:copy(this.viewer),...fields};
    if(item){item.events.push(event);item.revision++;}
    else this.data.events.push(event);
    return event;
  }
  async claim(id,payload,key) {
    return this.mutate(`claim:${id}`,payload,key,()=>{
      const item=this.getItem(id);this.checkRevision(item,payload.base_revision);
      if(item.assignee&&item.assignee!==this.viewer.id)fail('This item is assigned to another reviewer.','already_claimed',409);
      item.assignee=this.viewer.id;
      this.event(item,'claim',{rationale:'Review claimed'});
      return this.visible(item);
    });
  }
  async review(id,payload,key) {
    return this.mutate(`review:${id}`,payload,key,()=>{
      const item=this.getItem(id);this.checkRevision(item,payload.base_revision);
      if(item.assignee&&item.assignee!==this.viewer.id)fail('This item is assigned to another reviewer.','already_claimed',409);
      if(!payload.rationale?.trim())fail('Add a short rationale for your decision.');
      for(const target of item.targets)if(!target.choices.includes(payload.labels?.[target.id]))fail(`Choose a valid label for ${target.name}.`);
      if(Object.keys(payload.labels||{}).some(id=>!item.targets.some(t=>t.id===id)))fail('Unknown review target.');
      const validRefs=new Set(this.fixtures.policies.map(x=>x.id));
      if((payload.policy_refs||[]).some(id=>!validRefs.has(id)))fail('Unknown policy reference.');
      if(payload.phase==='independent') {
        if(payload.decision!=='save')fail('Save your independent review before adjudication.');
        if(item.independent[this.viewer.id])fail('Your first pass is already recorded.','already_reviewed',409);
        item.independent[this.viewer.id]=copy(payload.labels);
        item.assignee=null;
        this.event(item,'independent',{phase:'independent',ai_visible:false,labels:payload.labels,rationale:payload.rationale,policy_refs:payload.policy_refs,policy_version:item.policy_version});
      } else if(payload.phase==='adjudication') {
        this.role('sme','policy');
        if(!item.independent[this.viewer.id])fail('Complete an independent first pass before adjudication.','first_pass_required',409);
        if(payload.decision==='approve') {
          if(item.split==='holdout')fail('Protected evaluation items cannot enter an optimization epoch.','protected_split',403);
          if(Object.values(payload.labels).includes('Uncertain'))fail('Uncertain decisions need Policy review before becoming gold.');
          for(const target of item.targets)target.staged_label=payload.labels[target.id];
          item.status='approved';delete item.escalation_owner;item.assignee=null;
        } else if(payload.decision==='escalate') {
          item.status='escalated';item.escalation_owner='Policy · '+(this.fixtures.categories.find(c=>c.id===item.category)?.name||item.category);
          item.gold_eligible=false;item.assignee=null;
          for(const target of item.targets)target.staged_label=null;
        } else if(payload.decision==='save') {
          item.status='open';item.gold_eligible=false;item.assignee=this.viewer.id;
          for(const target of item.targets)target.staged_label=null;
        } else fail('Unknown adjudication action.');
        this.event(item,'adjudication',{phase:'adjudication',decision:payload.decision,ai_visible:true,labels:payload.labels,rationale:payload.rationale,policy_refs:payload.policy_refs,policy_version:item.policy_version});
      } else fail('Unknown review phase.');
      return this.visible(item);
    });
  }
  async route(id,payload,key) {
    return this.mutate(`route:${id}`,payload,key,()=>{
      const item=this.getItem(id);this.checkRevision(item,payload.base_revision);
      if(!payload.reason?.trim())fail('Explain why this policy route needs to change.');
      const ids=new Set(this.fixtures.policies.map(x=>x.id));
      if(!Array.isArray(payload.nodes)||payload.nodes.some(x=>!ids.has(x)))fail('Select valid policy nodes.');
      const before=copy(item.route);
      item.route={...item.route,nodes:payload.nodes,fallback:payload.nodes.length===0,current_route_changed:true,trace_stale:true};
      item.gold_eligible=false;item.status='open';item.assignee=null;
      for(const target of item.targets)target.staged_label=null;
      this.event(item,'route',{rationale:payload.reason,before,after:copy(item.route)});
      return this.visible(item);
    });
  }
  async manifest(epoch) {
    const rows=this.data.items.filter(item=>item.split!=='holdout'&&item.gold_eligible&&item.targets.every(t=>t.gold?.label&&t.gold.label!=='Uncertain')).map(item=>({
      id:item.id,project:item.project,type:item.type,category:item.category,policy_version:item.policy_version,content_version:item.content_version||'1',split:item.split,content:copy(item.content),source:copy(item.source),route:copy(item.route||{}),conversation_id:item.conversation_id||item.content?.conversation_id||null,group_id:item.group_id||item.conversation_id||item.content?.conversation_id||null,
      targets:item.targets.map(t=>({id:t.id,category:t.category||item.category,label:t.gold.label,revision:t.gold.revision}))
    }));
    const manifest={epoch,policy_versions:[...new Set(rows.map(x=>x.policy_version))],created_at:now(),items:rows};
    if(this.fixtures.graph)Object.assign(manifest,{graph:copy(this.fixtures.graph),graph_id:this.fixtures.graph.graph_id,graph_version:this.fixtures.graph.version,graph_hash:this.fixtures.graph.source_sha256});
    manifest.digest=await digest(manifest);return manifest;
  }
  async publish(payload,key) {
    return this.mutate('publish',payload,key,async()=>{
      this.role('policy');
      if(!payload.note?.trim())fail('Add a publication note.');
      if(payload.expected_epoch!==this.data.epoch)fail('A newer epoch exists. Refresh before publishing.','stale_epoch',409);
      const changed=this.data.items.filter(item=>item.status==='approved'&&item.split!=='holdout');
      if(!changed.length)fail('Approve at least one review before publishing.','no_changes',409);
      const epoch=this.data.epoch+1;
      for(const item of changed) {
        for(const target of item.targets) {
          if(!target.staged_label||target.staged_label==='Uncertain')fail('An approved item contains an unresolved label.');
          target.gold={label:target.staged_label,revision:(target.gold?.revision||0)+1};target.staged_label=null;
        }
        item.status='published';item.gold_eligible=true;item.assignee=null;
        this.event(item,'published',{epoch,rationale:payload.note||'Published approved local gold',labels:Object.fromEntries(item.targets.map(t=>[t.id,t.gold.label])),policy_version:item.policy_version});
      }
      const manifest=await this.manifest(epoch);this.data.manifests[epoch]=manifest;this.data.epoch=epoch;
      const record={epoch,created_at:manifest.created_at,created_by:this.viewer.name,item_count:manifest.items.length,changed_count:changed.length,digest:manifest.digest,status:'local_published'};
      this.data.epochs.unshift(record);this.event(null,'epoch',{epoch,rationale:payload.note||'Published approved local gold'});
      return record;
    });
  }
  async exportEpoch(epoch) {
    const manifest=this.data.manifests[epoch];if(!manifest)fail('That epoch was not found.','not_found',404);return copy(manifest);
  }
  async rerun(payload,key) {
    return this.mutate('rerun',payload,key,()=>{
      this.role('policy');
      if(!payload.note?.trim())fail('Add a note for the APO handoff.');
      const manifest=this.data.manifests[payload.epoch];if(!manifest)fail('Select a published epoch first.','not_found',404);
      if(!manifest.items.length)fail('The epoch has no optimization-eligible gold items.','empty_epoch',409);
      const run={id:`apo-${uid().slice(0,8)}`,epoch:payload.epoch,digest:manifest.digest,created_at:now(),status:'awaiting_connector',note:payload.note||''};
      this.data.runs.unshift(run);this.event(null,'rerun',{epoch:run.epoch,run_id:run.id,rationale:'APO handoff queued; production connector required.'});return run;
    });
  }
  async events() {
    const events=[...this.data.events,...this.data.items.flatMap(item=>item.events)].sort((a,b)=>b.created_at.localeCompare(a.created_at));
    return {events:events.map(event=>{const safe=copy(event);for(const key of ['labels','before','after','rationale','comment','policy_refs'])delete safe[key];return safe;})};
  }
  async importItems(items,key) {
    return this.mutate('import',items,key,()=>{
      this.role('policy');if(!Array.isArray(items)||!items.length||items.length>200)fail('Supply between 1 and 200 items.');
      let imported=0,skipped=0;
      const categories=new Set(this.fixtures.categories.map(x=>x.id));
      const nodes=new Set(this.fixtures.policies.map(x=>x.id));
      for(const source of items) {
        if(!/^[A-Za-z0-9_-]{1,100}$/.test(source.id||'')||!categories.has(source.category)||!['image','pin','assistant'].includes(source.type)||!Array.isArray(source.targets)||!source.targets.length)fail('An imported item has invalid identity, category, type or targets.');
        if(!['train','holdout'].includes(source.split||'train'))fail('Invalid dataset split.');
        if(!this.fixtures.policies.some(p=>p.category===source.category&&p.version===source.policy_version))fail('Policy version does not match the current policy catalog.');
        const ids=source.targets.map(t=>t.id);
        if(new Set(ids).size!==ids.length||source.targets.some(t=>!Array.isArray(t.choices)||!t.choices.length||!t.id))fail('Target IDs and label choices must be valid.');
        if(source.type==='assistant'&&(!ids.includes('input')||!ids.includes('output')))fail('Assistant items require input and output targets.');
        if((source.route?.nodes||[]).some(id=>!nodes.has(id)))fail('Unknown policy route.');
        const images=[source.content?.image,...(source.content?.messages||[]).map(message=>message.image)].filter(Boolean);
        for(const image of images) {
          const local=/^assets\/[a-zA-Z0-9_.\/-]+$/.test(image)&&!image.split('/').some(p=>p==='..'||p==='.');
          let remote=false;try{const url=new URL(image);remote=url.protocol==='https:'&&!url.username&&!url.password;}catch{}
          if(!local&&!remote)fail('Use a safe local asset or HTTPS image reference.');
        }
        for(const target of source.targets) {
          if((target.ai?.policy_refs||[]).some(id=>!nodes.has(id)))fail('Unknown AI policy reference.');
          if(target.category&&!categories.has(target.category))fail('Unknown target category.');
        }
        if(this.data.items.some(x=>x.id===source.id)){skipped++;continue;}
        const item=copy(source);item.status='open';item.revision=1;item.assignee=null;item.events=[];item.independent={};item.created_at=now();item.split ||= 'train';item.gold_eligible=false;delete item.seed_independent_labels;
        for(const target of item.targets){target.staged_label=null;target.gold=null;}
        this.data.items.push(item);this.event(item,'import',{rationale:'New candidate imported for independent labeling'});imported++;
      }
      return {imported,skipped};
    });
  }
}
