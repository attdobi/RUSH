"""In-memory Chromium smoke tests; no network, API keys, data labels or server needed.

Install Playwright and Chromium, then:
  python tests/studio-browser-smoke.py [--chromium /path/to/chromium]
The API responses are explicit fixtures. This is not a live deployment test.
"""
import argparse
import json
from pathlib import Path
import re
import shutil
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1] / 'web'


def boot(browser, *, recorded=False, width=1440):
    page = browser.new_page(viewport={'width':width, 'height':1000})
    errors = []
    page.on('pageerror', lambda error: errors.append(str(error)))
    html = re.sub(r'<script[^>]*src=[^>]*></script>', '', (ROOT/'index.html').read_text())
    html = re.sub(r'<link[^>]*rel="stylesheet"[^>]*>', '', html)
    page.set_content(html)
    page.add_style_tag(content=(ROOT/'studio.css').read_text())
    page.evaluate('''() => {
      window.calls = [];
      window.fetch = async (url, options={}) => {
        calls.push({url,method:options.method||'GET'});
        throw new Error('Offline API fixture');
      };
    }''')
    if recorded:
        page.evaluate('''() => {
          window.failSnapshot=false;
          window.fetch = async (url, options={}) => {
            calls.push({url,method:options.method||'GET'});
            const u=new URL(url,'https://example.test');
            if(u.pathname.endsWith('/history')) return {ok:true,json:async()=>({series:[{id:'run-1',title:'Test recorded run',lineage:true,frames:[{version:'v0.1',title:'Baseline',status:'baseline'},{version:'v1.1',title:'Accepted',status:'accepted'},{version:'v1.2',title:'Later',status:'accepted'}]}]})};
            const v=u.searchParams.get('version');
            if(window.failSnapshot&&v==='v1.2') return {ok:false,status:404};
            // Delay the second snapshot to exercise latest-request-wins behavior.
            if(v==='v1.1') await new Promise(r=>setTimeout(r,70));
            return {ok:true,json:async()=>({origin:'recorded',area:'Generative_AI',version:v,nodes:[{id:'GA.root',node_type:'root',title:'Recorded intent',body:v==='v0.1'?'original rule':'changed rule',content_hash:v},{id:'GA.a',parent:'GA.root',title:'Boundary <script>bad()</script>',body:'untrusted <img src=x onerror=bad()>'}],edges:[{source:'GA.a',target:'GA.root',type:'subtype_of'}]})};
          };
        }''')
    for name in ['studio-core.js','studio-fixtures.js','about.js','studio.js']:
        page.add_script_tag(content=(ROOT/name).read_text())
    page.wait_for_selector('.graph-node')
    return page, errors


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--chromium',default=shutil.which('chromium'))
    args=parser.parse_args()
    passed=[]
    with sync_playwright() as p:
        browser=p.chromium.launch(executable_path=args.chromium,headless=True,args=['--no-sandbox'])
        page,errors=boot(browser)
        assert 'ILLUSTRATIVE' in page.locator('#sourceBadge').inner_text()
        assert 'unavailable' in page.locator('#sourceNote').inner_text()
        passed.append('offline provenance is explicit')
        start=page.locator('.graph-node').count()
        page.locator('[data-frame="4"]').click()
        assert page.locator('.graph-node').count()>start
        passed.append('timeline growth and content changes render')
        page.locator('[data-node="GA.root"]').focus()
        page.keyboard.press('Enter')
        assert page.locator('#clearSelection').is_visible()
        assert 'GA.root' in page.locator('#inspectorContent').inner_text()
        passed.append('keyboard node inspection')
        page.locator('#pathsTab').click()
        page.locator('[data-scenario="2"]').click()
        assert page.locator('.trace-outcome').inner_text()=='sme review'
        passed.append('missing evidence follows an explicit review path')
        page.locator('[data-scenario="3"]').click()
        assert page.locator('.trace-outcome').inner_text()=='request provenance'
        passed.append('absence of evidence never becomes a negative GenAI label')
        page.locator('#demoSelect').select_option('mnist')
        page.wait_for_selector('[data-node="MD.root"]')
        page.locator('#pathsTab').click()
        page.locator('[data-scenario="0"]').click()
        assert page.locator('.trace-outcome').inner_text()=='suggest digit 9'
        page.locator('[data-scenario="3"]').click()
        assert page.locator('.trace-outcome').inner_text()=='sme review'
        passed.append('MNIST switching and low-confidence review')
        page.locator('#knowledgeTab').click()
        for width in [375,390,768,1440]:
            page.set_viewport_size({'width':width,'height':900})
            assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
        passed.append('no horizontal overflow at four viewport widths')
        page.evaluate("location.hash='about'")
        page.wait_for_selector('#aboutView',state='visible')
        assert page.locator('#aboutContent a[href^="https://arxiv.org"]').count()>=5
        passed.append('method route and primary-source references')
        assert not errors, errors
        assert all(c['method']=='GET' for c in page.evaluate('calls'))
        passed.append('offline UI has no JavaScript errors or write requests')
        page.close()
        page,errors=boot(browser,recorded=True)
        assert 'RECORDED' in page.locator('#sourceBadge').inner_text()
        page.locator('[data-frame="1"]').click()
        page.wait_for_function("document.querySelector('#graphVersion').textContent.includes('v1.1')")
        assert '1 changed' in page.locator('#deltaNote').inner_text()
        passed.append('recorded snapshot API contract and body diff')
        page.locator('[data-node="GA.a"]').click()
        assert page.locator('#inspectorContent img').count()==0
        passed.append('untrusted rule text is escaped')
        page.locator('#pathsTab').click()
        assert 'No executable program' in page.locator('#graphEmpty').inner_text()
        passed.append('recorded policies are never silently made executable')
        page.locator('#knowledgeTab').click()
        page.evaluate('window.failSnapshot=true')
        page.locator('[data-frame="2"]').click()
        page.wait_for_selector('#graphEmpty',state='visible')
        assert page.locator('#graphSvg .graph-node').count()==0
        assert page.locator('#exportTrace').is_disabled()
        passed.append('failed snapshot never leaves stale evidence visible')
        assert not errors, errors
        assert all(c['method']=='GET' for c in page.evaluate('calls'))
        passed.append('recorded UI has no JavaScript errors or write requests')
        browser.close()
    print(json.dumps({'passed':len(passed),'checks':passed},indent=2))


if __name__=='__main__': main()
