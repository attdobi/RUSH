"""Stdlib HTTP server for the local RUSH web UI/API."""
from __future__ import annotations

from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from ._safety import APIError, safe_static_path, utcnow_iso
from .handlers_runs import handle_api, send_api_error
from .run_registry import RunRegistry

SERVER_VERSION = "rush-web-server-v1"


class RushWebRequestHandler(SimpleHTTPRequestHandler):
    """Serve repo-root static files and dispatch ``/api/*`` to JSON handlers."""

    server_version_name = SERVER_VERSION

    def __init__(
        self,
        *args: Any,
        repo_root: Path,
        registry: RunRegistry,
        started_at: str,
        **kwargs: Any,
    ) -> None:
        self.repo_root = repo_root.resolve()
        self.registry = registry
        self.started_at = started_at
        super().__init__(*args, directory=str(self.repo_root), **kwargs)

    def version_string(self) -> str:  # pragma: no cover - cosmetic header only
        return self.server_version_name

    def translate_path(self, path: str) -> str:
        return str(safe_static_path(self.repo_root, path))

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        if self.path.startswith("/api/"):
            handle_api(self, self.registry, method="GET")
            return
        try:
            super().do_GET()
        except APIError as exc:
            send_api_error(self, exc)

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        if self.path.startswith("/api/"):
            handle_api(self, self.registry, method="POST")
            return
        self.send_error(405, "Method not allowed")

    def do_HEAD(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        if self.path.startswith("/api/"):
            self.send_error(405, "Method not allowed")
            return
        try:
            super().do_HEAD()
        except APIError as exc:
            send_api_error(self, exc)


def create_server(
    *,
    host: str = "127.0.0.1",
    port: int = 8766,
    repo_root: str | Path,
    registry: RunRegistry | None = None,
) -> ThreadingHTTPServer:
    """Create a localhost-only ThreadingHTTPServer."""
    if host != "127.0.0.1":
        raise ValueError("RUSH web server refuses to bind anything except 127.0.0.1")
    root = Path(repo_root).resolve()
    active_registry = registry or RunRegistry(root)
    started_at = utcnow_iso()
    handler_cls = partial(
        RushWebRequestHandler,
        repo_root=root,
        registry=active_registry,
        started_at=started_at,
    )
    return ThreadingHTTPServer((host, port), handler_cls)
