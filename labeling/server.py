"""Run the standalone, localhost-only labeling API with its static frontend.

    python -m labeling.server --port 8090 --seed-demo

RUSH_LABELING_TOKENS optionally contains a JSON object mapping opaque bearer
tokens to {id, name, role}. Configured token mode never falls back to a demo user.
"""
from __future__ import annotations

import argparse
from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import hmac
import json
import mimetypes
import os
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlsplit

from .store import Actor, LabelingError, LabelingStore, canonical, require


API = "/api/labeling"
MAX_BODY = 2 * 1024 * 1024


def load_tokens(value: dict | None = None):
    if value is None:
        raw = os.environ.get("RUSH_LABELING_TOKENS")
        if raw is None:
            return None
        try:
            value = json.loads(raw)
        except ValueError as exc:
            raise ValueError("RUSH_LABELING_TOKENS must be a valid JSON object") from exc
    if not isinstance(value, dict) or not value:
        raise ValueError("Token mode requires a nonempty token-to-reviewer mapping")
    actors = {}
    for token, identity in value.items():
        if not isinstance(token, str) or not token.strip() or not isinstance(identity, dict):
            raise ValueError("Each configured bearer token must identify a reviewer")
        if set(identity) != {"id", "name", "role"}:
            raise ValueError("Token identities must contain exactly id, name and role")
        actors[token] = Actor(**identity)
    return actors


class LabelingRequestHandler(BaseHTTPRequestHandler):
    server_version = "RUSHLabeling/1.0"

    def __init__(self, *args, store: LabelingStore, web_root: Path, tokens, **kwargs):
        self.store = store
        self.web_root = web_root.resolve()
        self.tokens = tokens
        super().__init__(*args, **kwargs)

    def _send(self, status, value, *, content_type="application/json; charset=utf-8", head=False):
        raw = canonical(value).encode() if content_type.startswith("application/json") else value
        if isinstance(raw, str): raw = raw.encode()
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "same-origin")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Content-Security-Policy", "default-src 'self'; img-src 'self' https: data:; style-src 'self' 'unsafe-inline'; script-src 'self'; connect-src 'self'; font-src 'self'; frame-ancestors 'none'; base-uri 'self'; form-action 'self'")
        self.end_headers()
        if not head: self.wfile.write(raw)

    def _guard_host(self):
        try:
            host = urlsplit("http://" + self.headers.get("Host", ""))
            valid = host.hostname in {"127.0.0.1", "localhost"} and host.port == self.server.server_address[1] and not host.username
        except ValueError:
            valid = False
        require(valid, 403, "invalid_host", "This service accepts only its local origin")

    def _guard_origin(self):
        origin = self.headers.get("Origin")
        if origin:
            expected = "http://" + self.headers.get("Host", "")
            require(origin.rstrip("/") == expected, 403, "cross_origin", "Cross-origin mutations are not allowed")
        require(self.headers.get("Sec-Fetch-Site", "same-origin") not in {"cross-site"}, 403, "cross_origin", "Cross-site mutations are not allowed")

    def _actor(self):
        if self.tokens is None:
            return self.store.demo_actor()
        authorization = self.headers.get("Authorization", "")
        require(authorization.lower().startswith("bearer "), 401, "authentication_required", "A configured reviewer bearer token is required")
        supplied = authorization[7:].strip()
        for token, actor in self.tokens.items():
            if hmac.compare_digest(supplied, token): return actor
        raise LabelingError(401, "invalid_token", "The reviewer token is not recognized")

    def _body(self):
        require(self.headers.get_content_type() == "application/json", 415, "content_type", "Send an application/json request body")
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            raise LabelingError(400, "invalid_length", "Invalid Content-Length")
        require(0 < length <= MAX_BODY, 413, "body_size", "Request body is empty or exceeds 2 MiB")
        try:
            def invalid_constant(value): raise ValueError("Non-finite JSON value")
            result = json.loads(self.rfile.read(length), parse_constant=invalid_constant)
        except (ValueError, UnicodeDecodeError):
            raise LabelingError(400, "invalid_json", "The request body must contain valid JSON")
        require(isinstance(result, dict), 422, "validation_error", "Request body must be an object")
        return result

    def _api_get(self, path, query, actor):
        suffix = path[len(API):].strip("/")
        if suffix == "bootstrap":
            return self.store.bootstrap(actor, mode="authenticated" if self.tokens else "local-demo")
        if suffix == "items":
            filters = {k:v[0] for k,v in parse_qs(query).items() if k in {"q", "type", "category", "status", "project"}}
            return self.store.list_items(actor, filters)
        if suffix == "events": return self.store.events(actor)
        parts = suffix.split("/")
        if len(parts) == 2 and parts[0] == "items":
            return self.store.get_item(parts[1], actor)
        if len(parts) == 3 and parts[0] == "epochs" and parts[2] == "export":
            require(parts[1].isdigit(), 404, "not_found", "Gold epoch not found")
            return self.store.export_epoch(int(parts[1]), actor)
        raise LabelingError(404, "not_found", "API route not found")

    def _api_post(self, path, actor, body):
        suffix = path[len(API):].strip("/")
        key = self.headers.get("Idempotency-Key")
        if suffix == "epochs": return self.store.publish_epoch(actor, body, idempotency_key=key)
        if suffix == "reruns": return self.store.request_rerun(actor, body, idempotency_key=key)
        if suffix == "items/import": return self.store.import_items(actor, body, idempotency_key=key)
        parts = suffix.split("/")
        if len(parts) == 3 and parts[0] == "items":
            methods = {"claim": self.store.claim_item, "reviews": self.store.review_item, "route": self.store.route_item}
            require(parts[2] in methods, 404, "not_found", "API route not found")
            return methods[parts[2]](parts[1], actor, body, idempotency_key=key)
        raise LabelingError(404, "not_found", "API route not found")

    def _static(self, path, head=False):
        for prefix in ("/web/labeling/", "/labeling/"):
            if path.startswith(prefix):
                path = "/" + path[len(prefix):]; break
        if path in {"/", "/labeling", "/web/labeling"}: path = "/index.html"
        if path == "/runtime.js":
            runtime = {"mode": "api", "baseUrl": API, "authRequired": self.tokens is not None}
            return self._send(200, "window.RUSH_LABELING = " + canonical(runtime) + ";\n", content_type="application/javascript; charset=utf-8", head=head)
        target = (self.web_root / path.lstrip("/")).resolve()
        require(target.is_relative_to(self.web_root), 404, "not_found", "File not found")
        require(target.name != "fixtures.json" and not any(p.startswith(".") for p in target.relative_to(self.web_root).parts), 404, "not_found", "File not found")
        require(target.is_file() and target.suffix.lower() in {".html", ".css", ".js", ".json", ".svg", ".png", ".jpg", ".jpeg", ".webp", ".gif", ".ico", ".woff", ".woff2"}, 404, "not_found", "File not found")
        mime = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
        if mime in {"text/html", "text/css", "text/javascript", "application/javascript"}: mime += "; charset=utf-8"
        self._send(200, target.read_bytes(), content_type=mime, head=head)

    def _handle(self, method):
        try:
            self._guard_host()
            parsed = urlsplit(self.path); path = unquote(parsed.path)
            require(path.startswith("/"), 400, "invalid_path", "Invalid request path")
            is_api = path == API or path.startswith(API + "/")
            if is_api:
                actor = self._actor()
                if method == "GET": result = self._api_get(path, parsed.query, actor)
                elif method == "POST":
                    self._guard_origin(); result = self._api_post(path, actor, self._body())
                else: raise LabelingError(405, "method_not_allowed", "Method not allowed")
                self._send(200, result)
            elif method in {"GET", "HEAD"}: self._static(path, head=method == "HEAD")
            else: raise LabelingError(405, "method_not_allowed", "Method not allowed")
        except LabelingError as exc:
            self._send(exc.status, exc.response(), head=method == "HEAD")
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception:
            self.log_error("Unhandled labeling request error")
            self._send(500, {"error": {"code": "internal_error", "message": "The request could not be completed"}})

    def do_GET(self): self._handle("GET")
    def do_POST(self): self._handle("POST")
    def do_HEAD(self): self._handle("HEAD")
    def do_OPTIONS(self): self._send(405, {"error": {"code": "method_not_allowed", "message": "Cross-origin API access is disabled"}})


def create_server(*, host="127.0.0.1", port=8090, repo_root: str | Path,
                  db_path: str | Path | None = None, seed_demo=False,
                  tokens: dict | None = None) -> ThreadingHTTPServer:
    if host != "127.0.0.1":
        raise ValueError("RUSH labeling server binds only to 127.0.0.1")
    root = Path(repo_root).resolve()
    actor_tokens = load_tokens(tokens)
    store = LabelingStore(db_path or root / "data" / "labeling" / "labeling.sqlite3")
    if seed_demo:
        fixture = root / "web" / "labeling" / "fixtures.json"
        store.seed_fixture(json.loads(fixture.read_text(encoding="utf-8")))
    handler = partial(LabelingRequestHandler, store=store, web_root=root / "web" / "labeling", tokens=actor_tokens)
    server = ThreadingHTTPServer((host, port), handler)
    server.daemon_threads = True
    server.labeling_store = store
    return server


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8090)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--db", type=Path)
    parser.add_argument("--seed-demo", action="store_true")
    args = parser.parse_args(argv)
    server = create_server(host=args.host, port=args.port, repo_root=args.repo_root, db_path=args.db, seed_demo=args.seed_demo)
    identity = "configured reviewer tokens" if os.environ.get("RUSH_LABELING_TOKENS") is not None else "explicit local demo reviewer"
    print(f"RUSH labeling: http://127.0.0.1:{server.server_address[1]} ({identity})", flush=True)
    print("Gold is local. APO requests wait in an outbox for a configured connector.", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
