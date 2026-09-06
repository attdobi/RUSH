#!/usr/bin/env python3
"""Preview this branch against a working RUSH HTTP server. GET-only; no DB setup.

python3 scripts/rush_preview_server.py --upstream http://127.0.0.1:8766
All native experiment, graph, proposal and thumbnail reads use the SAME upstream.
No assumption about ~/RUSH, no copying datasets, no provider credentials required.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import unquote, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

ROOT = Path(__file__).resolve().parents[1]
MAX_RESPONSE = 32 * 1024 * 1024
PROXIED = ('/api/', '/data/', '/policy-graph/', '/download/')


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def validate_origin(value: str) -> str:
    p = urlsplit(value)
    if (p.scheme not in ('http', 'https') or not p.hostname or p.username or p.password
            or p.path not in ('', '/') or p.query or p.fragment):
        raise ValueError('Upstream must be an http(s) origin without credentials, path or query')
    return value.rstrip('/')


def enhanced_index(root: Path, name: str) -> bytes:
    spec = importlib.util.spec_from_file_location('preview_shell', root/'pipeline/web/research_shell.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    html = module.enhance_lab_html((root/'web'/name).read_text(encoding='utf-8'))
    html = html.replace('</head>', '<script>window.RUSH_REVIEW_ONLY=true;</script></head>')
    return html.encode('utf-8')


def create_preview(root: Path, upstream: str, port: int = 8767):
    root = root.resolve()
    upstream = validate_origin(upstream)
    opener = build_opener(NoRedirect())

    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(root/'web'), **kwargs)

        def end_headers(self):
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            super().end_headers()

        def list_directory(self, path):
            self.send_error(404)
            return None

        def respond(self, status, data, content_type='application/json'):
            self.send_response(status)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            if self.command != 'HEAD':
                self.wfile.write(data)

        def do_GET(self):
            if self.headers.get('Sec-Fetch-Site') == 'cross-site':
                self.send_error(403, 'Cross-site access to the local preview is disabled')
                return
            parts = urlsplit(self.path)
            path = unquote(parts.path)
            if parts.scheme or parts.netloc or '\\' in path or any(p in ('.', '..') for p in path.split('/')):
                self.send_error(400)
                return
            if path == '/__review__/source':
                self.respond(200, json.dumps({'mode':'read_only_preview', 'upstream':upstream,
                    'experiment_api':'/api/experiments', 'graph_api':'/api/policy/graph',
                    'database_access':'none; upstream owns persistence'}).encode())
                return
            if path.startswith(PROXIED):
                # No browser-supplied destination, cookies, authorization, or redirects.
                req = Request(upstream + self.path, headers={'Accept':self.headers.get('Accept', '*/*')}, method='GET')
                try:
                    with opener.open(req, timeout=20) as response:
                        data = response.read(MAX_RESPONSE + 1)
                        if len(data) > MAX_RESPONSE:
                            raise ValueError('Response limit exceeded')
                        self.respond(response.status, data, response.headers.get('Content-Type', 'application/octet-stream'))
                except HTTPError as exc:
                    status = exc.code if not 300 <= exc.code < 400 else 502
                    data = json.dumps({'error':'upstream_http_error', 'status':exc.code, 'path':parts.path}).encode()
                    self.respond(status, data)
                except (URLError, OSError, ValueError):
                    self.respond(502, json.dumps({'error':'upstream_unavailable', 'path':parts.path,
                        'hint':'Start the original RUSH server, or set --upstream to its working origin.'}).encode())
                return
            relative = path.removeprefix('/web/').lstrip('/')
            if not relative or relative in ('index.html', 'lab.html', 'studio.html'):
                name = relative or 'index.html'
                try:
                    self.respond(200, enhanced_index(root, name), 'text/html; charset=utf-8')
                except (OSError, ValueError):
                    self.send_error(404)
                return
            target = (root/'web'/relative).resolve()
            if not target.is_relative_to((root/'web').resolve()) or any(p.startswith('.') for p in Path(relative).parts):
                self.send_error(404)
                return
            if target.suffix not in ('.js', '.css', '.svg', '.png', '.ico') or not target.is_file():
                self.send_error(404)
                return
            self.path = '/' + relative
            if self.command == 'HEAD':
                super().do_HEAD()
            else:
                super().do_GET()

        def do_HEAD(self):
            self.do_GET()

        def reject_write(self):
            self.send_error(405, 'Read-only review: use the original RUSH application for writes')
        do_POST = do_PUT = do_PATCH = do_DELETE = reject_write

    server = ThreadingHTTPServer(('127.0.0.1', port), Handler)
    server.daemon_threads = True
    return server


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--upstream', default='http://127.0.0.1:8766')
    parser.add_argument('--port', type=int, default=8767)
    args = parser.parse_args()
    server = create_preview(ROOT, args.upstream, args.port)
    print(f'Read-only preview: http://127.0.0.1:{server.server_port}/', flush=True)
    print(f'All data reads use: {validate_origin(args.upstream)}; all writes are blocked.', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()

if __name__ == '__main__':
    main()
