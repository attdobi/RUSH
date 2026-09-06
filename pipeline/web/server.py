"""Stdlib HTTP server for the local RUSH web UI/API."""
from __future__ import annotations

from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import re
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from ._safety import APIError, safe_static_path, utcnow_iso, whitelisted_static_prefix
from .build_id import get_build_id
from .handlers_runs import handle_api, send_api_error
from .run_registry import RunRegistry
from .studio import dispatch as dispatch_studio
from .research import dispatch as dispatch_research
from .research_shell import enhance_lab_html

SERVER_VERSION = "rush-web-server-v1"
_LOCAL_ASSET_RE = re.compile(
    r"(<(?:script|link)\b[^>]*?\b(?:src|href)=)([\"'])([^\"']+)(\2)", re.IGNORECASE,
)


def _with_build_version(url: str, build_id: str) -> str:
    if not url or "://" in url or url.startswith("//") or url.startswith("#"):
        return url
    parts = urlsplit(url)
    pairs = [(key, value) for key, value in parse_qsl(parts.query, keep_blank_values=True) if key != "v"]
    pairs.append(("v", build_id))
    return urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(pairs), parts.fragment))


def _rewrite_index_html(raw: bytes, build_id: str) -> bytes:
    """Enhance the original lab and version every local script/link reference."""
    html = enhance_lab_html(raw.decode("utf-8"))
    meta = f'<meta name="build-id" content="{build_id}">'
    if 'name="build-id"' not in html and "name='build-id'" not in html:
        charset = '<meta charset="utf-8" />'
        if charset in html:
            html = html.replace(charset, f"{charset}\n  {meta}", 1)

    def replace(match: re.Match[str]) -> str:
        prefix, quote, url, suffix = match.groups()
        return f"{prefix}{quote}{_with_build_version(url, build_id)}{suffix}"

    return _LOCAL_ASSET_RE.sub(replace, html).encode("utf-8")


class RushWebRequestHandler(SimpleHTTPRequestHandler):
    """Serve web-root static files and dispatch ``/api/*`` to JSON handlers."""
    server_version_name = SERVER_VERSION

    def __init__(self, *args: Any, repo_root: Path, registry: RunRegistry, started_at: str, **kwargs: Any) -> None:
        self.repo_root = repo_root.resolve()
        self.web_root = self.repo_root / "web"
        self.registry = registry
        self.started_at = started_at
        super().__init__(*args, directory=str(self.web_root), **kwargs)

    def version_string(self) -> str:
        return self.server_version_name

    def translate_path(self, path: str) -> str:
        return str(safe_static_path(self.repo_root, self.web_root, path))

    def list_directory(self, path: str):
        self.send_error(404, "not_found")
        return None

    def end_headers(self) -> None:
        if not self.path.startswith("/api/") and not self.path.startswith("/download"):
            cache_control = getattr(self, "_cache_control_override", None)
            if cache_control is None:
                try:
                    is_whitelisted = whitelisted_static_prefix(self.path) is not None
                except APIError:
                    is_whitelisted = True
                cache_control = "no-store" if is_whitelisted else "public, max-age=300"
            self.send_header("Cache-Control", cache_control)
        super().end_headers()

    def _handle_download(self, *, method: str) -> bool:
        if not (self.path.startswith("/download/") or self.path == "/download"):
            return False
        try:
            from .download import handle_download_request
        except ImportError:
            pass
        else:
            if handle_download_request(self, method=method):
                return True
        self.send_error(404)
        return True

    def _is_index_request(self) -> bool:
        return urlsplit(self.path).path in {"/", "/index.html", "/web/", "/web/index.html", "/lab.html", "/web/lab.html", "/studio.html", "/web/studio.html"}

    def _send_rewritten_index(self, *, head_only: bool = False) -> None:
        path = urlsplit(self.path).path
        filename = "studio.html" if path.endswith("/studio.html") else "lab.html" if path.endswith("/lab.html") else "index.html"
        index_path = self.web_root / filename
        if not index_path.is_file():
            self.send_error(404, "not_found")
            return
        raw = _rewrite_index_html(index_path.read_bytes(), get_build_id())
        self._cache_control_override = "no-cache, must-revalidate"
        try:
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            if not head_only:
                self.wfile.write(raw)
        finally:
            if hasattr(self, "_cache_control_override"):
                delattr(self, "_cache_control_override")

    def _send_studio(self) -> None:
        # The operator may select an external READ-ONLY evidence root. All lab
        # write handlers and model jobs remain bound to self.repo_root.
        source_root = Path(os.environ.get("RUSH_STUDIO_DATA_ROOT") or self.repo_root).expanduser().resolve()
        dispatch = dispatch_research if urlsplit(self.path).path == "/api/studio/research-run" else dispatch_studio
        status, payload = dispatch(source_root, self.path)
        payload = {**payload, "external_evidence": source_root != self.repo_root}
        raw = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self) -> None:
        if self._handle_download(method="GET"):
            return
        if self.path.startswith("/api/studio/"):
            self._send_studio()
            return
        if self.path.startswith("/api/"):
            handle_api(self, self.registry, method="GET")
            return
        if self._is_index_request():
            self._send_rewritten_index()
            return
        try:
            super().do_GET()
        except APIError as exc:
            send_api_error(self, exc)

    def do_POST(self) -> None:
        if self.path.startswith("/api/studio/"):
            self.send_error(405, "The policy research viewer is read-only")
            return
        if self.path.startswith("/api/"):
            handle_api(self, self.registry, method="POST")
            return
        self.send_error(405, "Method not allowed")

    def do_HEAD(self) -> None:
        if self._handle_download(method="HEAD"):
            return
        if self.path.startswith("/api/"):
            self.send_error(405, "Method not allowed")
            return
        if self._is_index_request():
            self._send_rewritten_index(head_only=True)
            return
        try:
            super().do_HEAD()
        except APIError as exc:
            send_api_error(self, exc)


def create_server(*, host: str = "127.0.0.1", port: int = 8766, repo_root: str | Path, registry: RunRegistry | None = None) -> ThreadingHTTPServer:
    """Create a localhost-only ThreadingHTTPServer."""
    if host != "127.0.0.1":
        raise ValueError("RUSH web server refuses to bind anything except 127.0.0.1")
    root = Path(repo_root).resolve()
    active_registry = registry or RunRegistry(root)
    handler_cls = partial(RushWebRequestHandler, repo_root=root, registry=active_registry, started_at=utcnow_iso())
    return ThreadingHTTPServer((host, port), handler_cls)
