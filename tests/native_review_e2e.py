"""Exercise the COMPLETE native app against real committed repository artifacts.

No page.set_content, no fake experiment responses, no model calls. D3 may be
served from a local npm installation to avoid a CDN dependency in CI.
"""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import threading
from urllib.error import HTTPError
from urllib.request import Request,urlopen

from playwright.sync_api import sync_playwright

ROOT=Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0,str(ROOT))
from pipeline.web.server import create_server
spec=importlib.util.spec_from_file_location('preview',ROOT/'scripts/rush_preview_server.py')
P=importlib.util.module_from_spec(spec);spec.loader.exec_module(P)


def git_blob(path):
    data=path.read_bytes();return hashlib.sha1(f'blob {len(data)}\0'.encode()+data).hexdigest()


def main():
    assert git_blob(ROOT/'web/about.js')=='013af218b8a8b053aaea5263fc03684ddd8c7a24','Original About source was not preserved'
    original=create_server(repo_root=ROOT,port=0)
    threading.Thread(target=original.serve_forever,daemon=True).start()
    source=f'http://127.0.0.1:{original.server_port}'
    preview=P.create_preview(ROOT,source,0)
    threading.Thread(target=preview.serve_forever,daemon=True).start()
    origin=f'http://127.0.0.1:{preview.server_port}'
    out=ROOT/'test-results/native-review';out.mkdir(parents=True,exist_ok=True)
    checks=[]
    try:
        with sync_playwright() as pw:
            browser=pw.chromium.launch(headless=True,args=['--no-sandbox'])
            for demo,area in [('genai','Generative_AI'),('mnist','MNIST_Digits')]:
                page=browser.new_page(viewport={'width':1512,'height':1050})
                errors=[];page.on('pageerror',lambda e:errors.append(str(e)))
                d3=os.environ.get('D3_JS')
                if d3:
                    page.route('https://cdn.jsdelivr.net/npm/d3@7.9.0/dist/d3.min.js',lambda route:route.fulfill(path=d3,content_type='application/javascript'))
                page.goto(f'{origin}/?demo={demo}#loop',wait_until='domcontentloaded')
                page.wait_for_selector('#policyGraphSvg .policy-node',timeout=90000)
                assert page.locator('#researchWorkbench').count()==0,'Disconnected replacement still mounted'
                assert page.locator('#evToolbar').count()==1
                assert page.locator('#experimentStart').count()==1
                selected=page.locator('#experimentSelect').input_value()
                if selected:
                    page.wait_for_function("document.querySelector('#evSource').textContent.startsWith('Native API')",timeout=30000)
                    with urlopen(f'{source}/api/experiments/{selected}') as r:data=json.load(r)
                    assert data['area']==area
                    chips=page.locator('#experimentKgCycles [data-kg-k]')
                    if chips.count()>1:
                        chips.first.click()
                        page.wait_for_function("document.querySelector('#evFrame').textContent.startsWith('k=0')")
                        chips.last.click()
                        page.wait_for_timeout(1200)
                # Snapshot/render APIs are proxied, not substituted by fixtures.
                with urlopen(f'{source}/api/policy/graph?area={area}') as r:native=json.load(r)
                with urlopen(f'{origin}/api/policy/graph?area={area}') as r:proxied=json.load(r)
                assert native==proxied
                page.locator('#policyEvolution').scroll_into_view_if_needed()
                page.screenshot(path=str(out/f'{demo}-actual-artifacts.png'))
                page.locator('#viewSwitcher [data-view="about"]').click()
                page.wait_for_selector('#aboutContent .about-arch',state='visible',timeout=30000)
                assert len(page.locator('#aboutContent').inner_text())>20000
                assert page.locator('#researchAddendum').count()==1
                page.screenshot(path=str(out/f'{demo}-restored-about.png'))
                assert not errors,errors
                checks.append({'demo':demo,'experiment':selected,'native_nodes':len(native['nodes']),
                    'full_app':True,'api_mocks':False,'page_errors':errors})
                page.close()
            browser.close()
        try:
            urlopen(Request(origin+'/api/experiments/start',data=b'{}',method='POST'))
            raise AssertionError('Preview forwarded a write')
        except HTTPError as e:assert e.code==405
        result={'about_original_sha':'013af218b8a8b053aaea5263fc03684ddd8c7a24','checks':checks,
                'scope':'Full app, real native server, committed artifacts. Not private host, SQL connection or model execution.'}
        (out/'results.json').write_text(json.dumps(result,indent=2))
        print(json.dumps(result,indent=2))
    finally:
        preview.shutdown();original.shutdown();preview.server_close();original.server_close()

if __name__=='__main__':main()
