"""Chromium component integration checks with explicit API/legacy-DOM fixtures.

These do not run the full legacy JS app, a live server, models, or a database.
  python tests/research-browser-smoke.py [--screenshots /tmp/rush-previews]
"""
import argparse
import json
from pathlib import Path
import shutil
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
ID = 'exp-20260707T004341-b9c4f0'
HTML = '''<!doctype html><html><head><meta charset="utf-8"/><meta name="viewport" content="width=device-width, initial-scale=1"/>
<style>body{margin:0}button,a,input,select{font:inherit}button,select{padding:8px;border:1px solid #334a5c;background:#142538;color:#d3e7f4}a{color:#9bc8e8}.topbar{display:flex;align-items:center;justify-content:space-between}.brand{display:flex;align-items:center;gap:9px}.mark{padding:4px 10px}.demo-selector,.view-switcher,.topbar-action-links{display:flex;gap:5px}.section{padding-top:24px}.api-panel{padding:20px;border:1px solid #344a5e}.run-controls{display:flex;flex-wrap:wrap;gap:18px;margin:18px 0}.run-controls label{display:grid;gap:6px}.view-loop #about{display:none}.view-about #experiment{display:none}input{background:#0b1724;border:1px solid #3a5164;color:#deebf5;padding:7px}h3{font-size:16px}.hero-copy{max-width:900px}@media(max-width:760px){.topbar{flex-wrap:wrap}.demo-selector{flex-wrap:wrap}}</style></head>
<body class="view-loop"><header class="hero"><nav class="topbar"><div class="brand"><span class="mark">R</span><b>RUSH</b></div><div class="demo-selector"><button type="button">GenAI classification</button><button type="button">MNIST</button></div><div id="viewSwitcher" class="view-switcher">VIEW_BUTTONS</div><div class="topbar-actions"><div class="topbar-action-links"><a class="nav-pill" href="#">GitHub ↗</a></div></div></nav><section class="hero-grid"><div class="hero-copy"><p id="heroEyebrow" class="eyebrow"></p><h1 id="heroH1"></h1><p id="heroLede" class="lede"></p></div></section></header>
<main><section id="experiment" data-view="loop" class="section"><div class="section-head"><h2>Original lab</h2></div><div class="experiment-config api-panel"><fieldset id="runTriggerModels"><legend>Judge panel — API contract fixture; no live models</legend><label><input type="checkbox" checked class="model-select-input" value="fixture/judge-a"/>Fixture judge A</label><label><input type="checkbox" checked class="model-select-input" value="fixture/judge-b"/>Fixture judge B</label></fieldset><div class="run-controls">CONTROLS</div><button id="experimentStart">Start experiment (fixture only)</button></div>
<div class="run-controls"><label>Native run selector<select id="experimentSelect"><option value="">Choose a run</option><option value="exp-20260707T004341-b9c4f0">API test fixture</option></select></label></div><div id="experimentChart"><h3>Learning curves</h3><p>Existing learning-curve component is preserved in the actual lab; not executed by this test harness.</p></div><div id="experimentJudgeTable"><h3>Judge diagnostics</h3></div><div id="experimentLedger"><h3>Gate ledger</h3></div><div id="policyEvolution"><div id="policyGraphSvgWrap">Original graph component</div><div id="experimentPolicyChanges"></div></div><div id="experimentConfusion"><h3>Confusion matrix</h3></div></section><section id="about"><div id="aboutContent"></div></section></main></body></html>'''
HTML = HTML.replace('VIEW_BUTTONS',''.join(f'<button class="view-switcher-option" data-view="{x}" aria-pressed="{str(x=="loop").lower()}">{x}</button>' for x in ['loop','summary','adjudicate','benchmarks','about']))
HTML = HTML.replace('CONTROLS',''.join(f'<label>{name}<input id="{id}" value="{value}"/></label>' for id,name,value in [('experimentKMax','Cycles k','5'),('experimentBatchN','Training N','20'),('experimentTestN','Validation T','100'),('experimentMaxChanges','Max changes','3'),('experimentStrategy','Anchor strategy','random_misalignment')]))


def fixture(demo='genai'):
    area = 'Generative_AI' if demo == 'genai' else 'MNIST_Digits'
    prefix = 'GA' if demo == 'genai' else 'MD'
    families = [('anatomy','Anatomy & objects'),('scene','Scene geometry'),('texture','Surface texture'),('text','Text & symbols'),('provenance','Provenance evidence'),('negative','Hard negatives')]
    if demo=='mnist':families=[(f'digit.{i}',f'Digit {i}') for i in range(10)]
    root={'id':prefix+'.root','title':'Image-generation policy' if demo=='genai' else 'MNIST digit policy','node_type':'root','body':'TEST FIXTURE: policy root; not a production decision.'}
    nodes=[root]+[{'id':prefix+'.'+id,'parent':root['id'],'title':name,'node_type':'category','body':'TEST FIXTURE: a general criterion with positive and negative controls.'} for id,name in families]
    for i, name in enumerate(['Object continuity','Occlusion boundary','Physical contact','Light consistency','Reflections','Compression artifacts','Glyph consistency','Raster boundary','Source verification','Missing evidence']):
        parent=nodes[1+i%len(families)]['id'];nodes.append({'id':parent+'.rule'+str(i),'parent':parent,'title':name,'node_type':'boundary' if i%3==0 else 'rule','body':'TEST FIXTURE: inspect the supplied criterion, preserve uncertainty, and evaluate against held-out controls.'})
    snapshots={}
    for v, extra in [('v0.1',0),('v1.1',3),('v1.3',6)]:
        current=[dict(n) for n in nodes]
        for i in range(extra):
            parent=nodes[1+i%len(families)]['id'];current.append({'id':parent+'.new'+str(i),'parent':parent,'title':['Boundary exception','Source counterexample','Generalized criterion','Contact ambiguity','Reference dispute','Shifted source'][i],'node_type':'exception' if i%2 else 'boundary','body':'TEST FIXTURE: a rule addition, not a memorized row.'})
        if extra:current[1]['body']='TEST FIXTURE: generalized wording after reviewing a counterexample. <img src=x onerror="window.pwned=true">'
        edges=[{'source':n['id'],'target':n['parent'],'type':'subtype_of'} for n in current if n.get('parent')]
        edges += [{'source':current[2]['id'],'target':current[4]['id'],'type':'confused_with'}]
        snapshots[v]={'origin':'recorded','area':area,'version':v,'nodes':current,'edges':edges}
    frames=[{'k':0,'version':'v0.1','before_version':None,'status':'baseline','title':'k=0 · baseline'}, {'k':1,'version':'v1.1','before_version':'v0.1','status':'accepted','title':'k=1 · accepted'}, {'k':2,'version':'v1.1','before_version':'v1.1','status':'skipped','title':'k=2 · skipped'}, {'k':3,'version':'v1.3','before_version':'v1.1','status':'accepted','title':'k=3 · accepted'}]
    run={'origin':'recorded','config':{'run_number':'TEST FIXTURE','seed':13,'strategy':'random_misalignment','gate_mode':'metric_only','judge_models':['fixture/judge-a','fixture/judge-b']},'frames':frames,'split_audit':{'train_gate_overlap':0,'scope':'Test fixture identifiers only; not production data.'},'cycles':[{'k':f['k'],'n_misaligned':7,'metrics':{'test':{'system':{'n':100,'n_abstained':4,'macro_fpr':.12-.02*f['k'],'macro_fnr':.21-.03*f['k']}}},'gate':{'status':f['status'],'reason':'Explicit API test fixture, not a measured result.'}} for f in frames]}
    return {'area':area,'snapshots':snapshots,'run':run,'history':{'series':[{'id':'catalog','title':'Snapshot catalog','lineage':False,'frames':[{'version':v,'status':'snapshot'} for v in snapshots]},{'id':ID,'title':'TEST FIXTURE · explicit lineage','lineage':True,'frames':frames}]}}


def boot(browser, *, width=1440, demo='genai', offline=False, external=False, reduced=False):
    page=browser.new_page(viewport={'width':width,'height':1080},reduced_motion='reduce' if reduced else 'no-preference')
    errors=[];page.on('pageerror',lambda e:errors.append(str(e)))
    page.set_content(HTML)
    page.add_style_tag(content=(ROOT/'web/research.css').read_text())
    page.evaluate('''arg => {
      window.fixture=arg.data;window.failSnapshot=false;window.offline=arg.offline;window.calls=[];
      window.rushActiveDemo=()=>({id:arg.demo,policyGraph:{area:arg.data.area}});
      document.querySelectorAll('#viewSwitcher [data-view]').forEach(b=>b.addEventListener('click',()=>{document.body.classList.toggle('view-about',b.dataset.view==='about');document.body.classList.toggle('view-loop',b.dataset.view!=='about');}));
      window.fetch=async (url,options={})=>{
        window.calls.push({url,method:options.method||'GET'});
        if(window.offline)throw new Error('Offline API contract fixture');
        const u=new URL(url,'https://fixture.test');
        if(u.pathname.endsWith('/history'))return {ok:true,json:async()=>({...fixture.history,external_evidence:arg.external})};
        if(u.pathname.endsWith('/research-run'))return {ok:true,json:async()=>fixture.run};
        const v=u.searchParams.get('version');
        if(window.failSnapshot&&v==='v1.3')return {ok:false,status:404};
        if(v==='v0.1')await new Promise(r=>setTimeout(r,40));
        return {ok:true,json:async()=>fixture.snapshots[v]};
      };
    }''',{'data':fixture(demo),'demo':demo,'offline':offline,'external':external})
    for name in ['research-core.js','about.js','research.js']:page.add_script_tag(content=(ROOT/'web'/name).read_text())
    page.wait_for_selector('#rchStatus');page.wait_for_function("document.querySelector('#rchStage').getAttribute('aria-busy')==='false'")
    return page,errors


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--screenshots');args=parser.parse_args()
    passed=[]
    with sync_playwright() as p:
        browser=p.chromium.launch(executable_path=shutil.which('chromium'),headless=True,args=['--no-sandbox'])
        page,errors=boot(browser)
        assert page.locator('#rchVersion').inner_text().endswith('v1.3');assert page.locator('#experimentStart').count()==1
        passed.append('research loads recorded lineage and preserves native configuration controls')
        assert page.evaluate('window.scrollY')==0;passed.append('initial graph load does not scroll away from the graph')
        assert '+3 added' in page.locator('#rchDelta').inner_text();passed.append('accepted step highlights actual node additions')
        page.locator('[data-frame="2"]').click();page.wait_for_function("document.querySelector('#rchGateStatus').textContent==='SKIPPED'")
        assert page.locator('#rchVersion').inner_text().endswith('v1.1');assert 'incumbent graph retained' in page.locator('#rchEvidenceTitle').inner_text();passed.append('rejected candidate metrics cannot replace incumbent graph')
        page.locator('[data-frame="1"]').click();page.wait_for_function("document.querySelector('#rchGateStatus').textContent==='ACCEPTED'")
        page.locator('[data-rch-node="GA.anatomy"]').press('Enter');assert 'generalized wording' in page.locator('#rchInspector').inner_text();assert page.locator('#rchInspector img').count()==0;assert page.evaluate('window.pwned') is None;passed.append('keyboard rule inspection and escaped evidence text')
        page.locator('[data-layout="hierarchy"]').click();assert page.locator('#rchGraphHeading').inner_text()=='Policy hierarchy';passed.append('hierarchy layout switch')
        page.locator('#rchSearch').fill('Source verification');assert '1 matching' in page.locator('#rchMatch').inner_text();passed.append('body/title search without losing the graph')
        page.locator('#rchSearch').fill('');page.locator('#rchClear').click();page.locator('[data-layout="network"]').click();page.locator('[data-frame="3"]').click();page.wait_for_function("document.querySelector('#rchVersion').textContent.endsWith('v1.3')")
        with page.expect_download() as item:page.locator('#rchExport').click()
        data=json.loads(Path(item.value.path()).read_text());assert data['origin']=='recorded' and data['frame']['version']=='v1.3';passed.append('downloadable version-pinned evidence bundle')
        assert page.locator('#experimentSelect').input_value()==ID;passed.append('same-root native run selection is synchronized')
        page.locator('[data-view="about"]').click();assert page.locator('#methodInterval').inner_text().endswith('1.26%]');passed.append('methods notebook and computed Wilson interval')
        page.locator('#methodFP').fill('500');assert 'Enter integer' in page.locator('#methodInterval').inner_text();passed.append('invalid uncertainty inputs rejected')
        page.locator('#methodSeeds').fill('13,13');page.locator('#methodExport').click();assert 'distinct' in page.locator('#methodPlanStatus').inner_text();passed.append('duplicate study seeds rejected')
        page.locator('#methodSeeds').fill('13,37,71')
        with page.expect_download() as item:page.locator('#methodExport').click()
        plan=json.loads(Path(item.value.path()).read_text());assert plan['execution_authorized'] is False and len(plan['seeds'])==3;passed.append('draft ablation export does not authorize model execution')
        page.locator('#viewSwitcher [data-view="loop"]').click();page.evaluate('window.scrollTo(0,0)');page.wait_for_timeout(1700)
        if args.screenshots:
            out=Path(args.screenshots);out.mkdir(parents=True,exist_ok=True)
            page.screenshot(path=str(out/'research-workbench-fixture.png'),full_page=True)
            page.locator('[data-view="about"]').click();page.screenshot(path=str(out/'research-methods-fixture.png'),full_page=True)
        assert all(c['method']=='GET' and c['url'].startswith('/api/studio/') for c in page.evaluate('calls'));assert not errors,errors;passed.append('read-only network contract and zero component JS errors')
        page.close()
        race,errs=boot(browser)
        race.evaluate("document.querySelector('[data-frame=\"0\"]').click();document.querySelector('[data-frame=\"3\"]').click()")
        race.wait_for_timeout(180);assert race.locator('#rchVersion').inner_text().endswith('v1.3');passed.append('latest frame wins a delayed overlapping snapshot request')
        race.evaluate("fixture.run.cycles.forEach(c=>c.metrics={})")
        race.locator('#rchRefresh').click();race.wait_for_function("document.querySelector('#rchStage').getAttribute('aria-busy')==='false'")
        assert race.locator('#rchStats strong').all_text_contents()==['—','—','—','—'];passed.append('missing measurements never become zero-valued success')
        race.evaluate('window.failSnapshot=true');race.locator('#rchRefresh').click();race.wait_for_selector('#rchEmpty',state='visible')
        assert race.locator('.rch-node').count()==0 and race.locator('#rchExport').is_disabled();assert race.locator('#rchVersion').inner_text()=='Evidence unavailable';assert not errs;passed.append('failed refresh clears stale graph, metrics, version and export');race.close()
        pinned,errs=boot(browser)
        pinned.evaluate("fixture.snapshots['v1.3'].version='v9.9'")
        pinned.locator('#rchRefresh').click();pinned.wait_for_selector('#rchEmpty',state='visible')
        assert pinned.locator('.rch-node').count()==0 and pinned.locator('#rchExport').is_disabled();assert 'policy pin' in pinned.locator('#rchStatus').inner_text();assert not errs;passed.append('mismatched snapshot identity fails closed');pinned.close()
        retired,errs=boot(browser)
        retired.evaluate("fixture.snapshots['v1.3'].nodes=fixture.snapshots['v1.3'].nodes.filter(n=>n.id!=='GA.anatomy.new0');fixture.snapshots['v1.3'].edges=fixture.snapshots['v1.3'].edges.filter(e=>e.source!=='GA.anatomy.new0'&&e.target!=='GA.anatomy.new0')")
        retired.locator('#rchRefresh').click();retired.wait_for_function("document.querySelector('#rchStage').getAttribute('aria-busy')==='false'")
        retired.locator('[data-rch-node="GA.anatomy.new0"]').press('Enter')
        assert 'Retired rule text (previous version)' in retired.locator('#rchInspector').inner_text();assert 'retired at this step' in retired.locator('#rchInspector').inner_text();assert not errs;passed.append('retired rules are not mislabeled as current policy');retired.close()
        off,errs=boot(browser,offline=True);assert off.locator('#rchEmpty').is_visible();assert off.locator('.rch-node').count()==0;assert off.locator('#experimentStart').is_visible();assert not errs;passed.append('offline errors never fall back to synthetic results');off.close()
        ext,errs=boot(browser,external=True);assert ext.locator('#rchRootNote').is_visible();assert ext.locator('#experimentSelect').input_value()=='';assert not errs;passed.append('external evidence root cannot redirect native lab selections');ext.close()
        mobile,errs=boot(browser,width=390,demo='mnist',reduced=True)
        assert 'MNIST_Digits' in mobile.locator('#rchVersion').inner_text();assert mobile.evaluate('document.documentElement.scrollWidth <= window.innerWidth+1');assert not errs,errs
        if args.screenshots:mobile.screenshot(path=str(Path(args.screenshots)/'research-mobile-fixture.png'),full_page=True)
        passed.append('MNIST, reduced-motion rendering, and mobile overflow');mobile.close();browser.close()
    for item in passed:print('PASS',item)
    print(f'{len(passed)} browser checks passed (component/fixture integration, not live deployment).')

if __name__=='__main__':main()
