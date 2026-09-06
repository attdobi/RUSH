"""Unit/integration checks for the loopback read-only bridge; no providers."""
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen

ROOT=Path(__file__).resolve().parents[1]
def module(name,path):
    spec=importlib.util.spec_from_file_location(name,ROOT/path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
P=module('native_preview','scripts/rush_preview_server.py')
A=module('native_audit','scripts/rush_source_audit.py')
S=module('native_shell','pipeline/web/research_shell.py')

class Contracts(unittest.TestCase):
    def test_shell_preserves_about_and_controls(self):
        html='<head></head><body><section id="experiment"></section><div id="aboutContent"></div><script src="about.js"></script></body>'
        out=S.enhance_lab_html(html)
        self.assertIn('<script src="about.js"></script>',out)
        self.assertIn('lab-evidence.js',out)
        self.assertNotIn('src="research.js',out)
        self.assertEqual(out,S.enhance_lab_html(out))

    def test_origin_is_not_an_open_proxy(self):
        for url in ('file:///etc/passwd','http://user:secret@localhost','http://localhost/path','http://localhost?url=evil','//evil'):
            with self.assertRaises(ValueError):P.validate_origin(url)
        self.assertEqual(P.validate_origin('http://127.0.0.1:8766/'),'http://127.0.0.1:8766')

    def test_db_config_precedence(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);(root/'.env').write_text('RUSH_DB_URL="postgresql://person:secret@db/test"\n')
            with patch.dict(os.environ,{},clear=True):self.assertEqual(A.configured_url(root)[1],'RUSH_DB_URL in checkout .env')
            with patch.dict(os.environ,{'RUSH_DB_URL':'postgresql:///override'},clear=True):self.assertEqual(A.configured_url(root)[0],'postgresql:///override')
            self.assertNotIn('secret',json.dumps(A.audit_files(root)))

    def test_default_is_not_reported_as_connected(self):
        with tempfile.TemporaryDirectory() as t,patch.dict(os.environ,{},clear=True):
            self.assertIn('not proof',A.configured_url(Path(t))[1])

class Bridge(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.calls=[]
        class Upstream(BaseHTTPRequestHandler):
            def do_GET(self):
                cls.calls.append((self.path,self.headers.get('Authorization'),self.headers.get('Cookie')))
                if self.path.startswith('/api/thumbnail?'):
                    self.send_response(302);self.send_header('Location','/data/images/test.png?v=known');self.end_headers();return
                if self.path=='/data/images/test.png?v=known':
                    self.send_response(200);self.send_header('Content-Type','image/png');self.end_headers();self.wfile.write(b'known-image-bytes');return
                if self.path=='/api/redirect':self.send_response(302);self.send_header('Location','http://example.invalid/');self.end_headers();return
                if self.path=='/api/missing':self.send_error(404);return
                data=json.dumps({'path':self.path}).encode();self.send_response(200);self.send_header('Content-Type','application/json');self.end_headers();self.wfile.write(data)
            def log_message(self,*args):pass
        cls.upstream=ThreadingHTTPServer(('127.0.0.1',0),Upstream)
        cls.thread=threading.Thread(target=cls.upstream.serve_forever,daemon=True);cls.thread.start()
        cls.preview=P.create_preview(ROOT,f'http://127.0.0.1:{cls.upstream.server_port}',0)
        cls.thread2=threading.Thread(target=cls.preview.serve_forever,daemon=True);cls.thread2.start()
        cls.origin=f'http://127.0.0.1:{cls.preview.server_port}'
    @classmethod
    def tearDownClass(cls):
        cls.preview.shutdown();cls.upstream.shutdown();cls.preview.server_close();cls.upstream.server_close()
    def get(self,path):
        with urlopen(self.origin+path,timeout=5) as r:return json.load(r)
    def test_preserves_native_paths_and_queries(self):
        path='/api/policy/graph?area=MNIST_Digits&version=v0.1'
        self.assertEqual(self.get(path)['path'],path)
    def test_experiments_proposals_and_media_same_origin(self):
        for p in ('/api/experiments','/api/policy/proposals/demo','/data/runs/r/scoring/misalignment.json','/policy-graph/MNIST_Digits/v0.1/MD.root.md'):
            self.assertEqual(self.get(p)['path'],p)
    def test_native_thumbnail_redirect_returns_image_bytes(self):
        with urlopen(self.origin+'/api/thumbnail?path=data/image.png') as r:
            self.assertEqual(r.headers['Content-Type'],'image/png')
            self.assertEqual(r.read(),b'known-image-bytes')
            self.assertTrue(r.geturl().startswith(self.origin+'/data/'))

    def test_redirect_targets_remain_same_origin_and_static(self):
        origin='http://127.0.0.1:8766'
        for target in ('https://evil.test/data/a.png','/api/experiments/start',
                       '//evil.test/data/a.png','/data/%2e%2e/secret','/data/.env',
                       'http://user:secret@127.0.0.1:8766/data/a.png'):
            self.assertIsNone(P.media_redirect(origin,'/api/thumbnail',target))
        self.assertEqual(P.media_redirect(origin,'/api/thumbnail','/data/a.png?v=2'),'/data/a.png?v=2')

    def test_all_write_verbs_blocked(self):
        for method in ('POST','PUT','PATCH','DELETE'):
            before=len(self.calls)
            with self.assertRaises(HTTPError) as caught:urlopen(Request(self.origin+'/api/experiments/start',data=b'{}',method=method))
            self.assertEqual(caught.exception.code,405);self.assertEqual(len(self.calls),before)
    def test_credentials_not_forwarded(self):
        req=Request(self.origin+'/api/experiments',headers={'Authorization':'secret','Cookie':'secret'})
        with urlopen(req) as r:r.read()
        self.assertEqual(self.calls[-1][1:],(None,None))
    def test_redirect_not_followed(self):
        with self.assertRaises(HTTPError) as caught:self.get('/api/redirect')
        self.assertEqual(caught.exception.code,502)
    def test_status_not_disguised_as_empty_history(self):
        with self.assertRaises(HTTPError) as caught:self.get('/api/missing')
        self.assertEqual(caught.exception.code,404)
    def test_cross_site_blocked(self):
        with self.assertRaises(HTTPError) as caught:urlopen(Request(self.origin+'/api/experiments',headers={'Sec-Fetch-Site':'cross-site'}))
        self.assertEqual(caught.exception.code,403)
    def test_paths_do_not_escape(self):
        for p in ('/../README.md','/%2e%2e/README.md','/.env','/api/%2e%2e/private'):
            with self.assertRaises(HTTPError) as caught:urlopen(self.origin+p)
            self.assertIn(caught.exception.code,(400,404))
    def test_source_record_explicit_no_database_claim(self):
        self.assertEqual(self.get('/__review__/source')['database_access'],'none; upstream owns persistence')

if __name__=='__main__':unittest.main()
