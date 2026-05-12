"""Password-protected repository zip download endpoint for the RUSH web UI."""
from __future__ import annotations

import base64
from fnmatch import fnmatch
import hmac
from io import BytesIO
import json
import os
from pathlib import Path
import subprocess
from urllib.parse import urlsplit
from zipfile import ZIP_DEFLATED, ZipFile

_JSON_CONTENT_TYPE = "application/json; charset=utf-8"
_AUTH_REALM = 'Basic realm="RUSH download"'
_EXCLUDED_COMPONENTS = {
    ".git",
    ".venv",
    ".pytest_cache",
    "__pycache__",
    "node_modules",
    ".attila-notes",
}
_EXCLUDED_PATH_GLOBS = ("data/images", "data/images/*")
_EXCLUDED_FILENAME_GLOBS = ("*.pyc", ".DS_Store", ".env", ".env.*")


def handle_download_request(handler, *, method: str) -> bool:
    """Handle /download/* paths. Return True if a response was sent."""
    path = urlsplit(handler.path).path
    if path != "/download/rush.zip":
        return False

    user = os.environ.get("RUSH_DOWNLOAD_USER") or ""
    password = os.environ.get("RUSH_DOWNLOAD_PASS") or ""
    if not user or not password:
        _send_json(
            handler,
            503,
            {
                "error": {
                    "code": "download_disabled",
                    "message": "download endpoint not configured",
                }
            },
            method=method,
        )
        return True

    credentials = _parse_basic_auth(handler.headers.get("Authorization"))
    if credentials is None:
        _send_auth_required(handler, method=method)
        return True

    supplied_user, supplied_password = credentials
    expected_user = user.encode("utf-8")
    expected_password = password.encode("utf-8")
    user_ok = hmac.compare_digest(supplied_user.encode("utf-8"), expected_user)
    password_ok = hmac.compare_digest(supplied_password.encode("utf-8"), expected_password)
    if not (user_ok and password_ok):
        _send_auth_required(handler, method=method)
        return True

    repo_root = Path(handler.repo_root).resolve()
    short_sha = _short_sha(repo_root)
    archive = _build_zip(repo_root, short_sha)

    handler.send_response(200)
    handler.send_header("Content-Type", "application/zip")
    handler.send_header(
        "Content-Disposition", f'attachment; filename="rush-{short_sha}.zip"'
    )
    handler.send_header("Cache-Control", "no-store")
    handler.send_header("Content-Length", str(len(archive)))
    handler.end_headers()
    if method != "HEAD":
        handler.wfile.write(archive)
    return True


def _send_json(handler, status: int, payload: dict, *, method: str) -> None:
    raw = json.dumps(payload, sort_keys=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", _JSON_CONTENT_TYPE)
    handler.send_header("Content-Length", str(len(raw)))
    handler.send_header("Cache-Control", "no-store")
    handler.end_headers()
    if method != "HEAD":
        handler.wfile.write(raw)


def _send_auth_required(handler, *, method: str) -> None:
    raw = json.dumps(
        {"error": {"code": "auth_required", "message": "basic auth required"}},
        sort_keys=False,
    ).encode("utf-8")
    handler.send_response(401)
    handler.send_header("WWW-Authenticate", _AUTH_REALM)
    handler.send_header("Content-Type", _JSON_CONTENT_TYPE)
    handler.send_header("Content-Length", str(len(raw)))
    handler.send_header("Cache-Control", "no-store")
    handler.end_headers()
    if method != "HEAD":
        handler.wfile.write(raw)


def _parse_basic_auth(header: str | None) -> tuple[str, str] | None:
    if not header:
        return None
    scheme, sep, token = header.partition(" ")
    if sep != " " or scheme.lower() != "basic" or not token.strip():
        return None
    try:
        decoded = base64.b64decode(token.strip(), validate=True).decode("utf-8")
    except (ValueError, UnicodeDecodeError):
        return None
    username, sep, password = decoded.partition(":")
    if sep != ":":
        return None
    return username, password


def _short_sha(repo_root: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    sha = result.stdout.strip()
    return sha[:8] if sha else "unknown"


def _build_zip(repo_root: Path, short_sha: str) -> bytes:
    prefix = f"rush-{short_sha}"
    buffer = BytesIO()
    with ZipFile(buffer, "w", compression=ZIP_DEFLATED) as archive:
        for dirpath, dirnames, filenames in os.walk(repo_root):
            current = Path(dirpath)
            rel_dir = current.relative_to(repo_root)
            dirnames[:] = [
                dirname
                for dirname in dirnames
                if not _should_exclude(rel_dir / dirname, is_dir=True)
            ]
            for filename in filenames:
                rel_path = rel_dir / filename
                if _should_exclude(rel_path, is_dir=False):
                    continue
                archive.write(current / filename, arcname=f"{prefix}/{rel_path.as_posix()}")
    return buffer.getvalue()


def _should_exclude(rel_path: Path, *, is_dir: bool) -> bool:
    parts = rel_path.parts
    if any(part in _EXCLUDED_COMPONENTS or part == "cache" for part in parts):
        return True
    rel_posix = rel_path.as_posix()
    if any(fnmatch(rel_posix, pattern) for pattern in _EXCLUDED_PATH_GLOBS):
        return True
    filename = parts[-1] if parts else ""
    if any(fnmatch(filename, pattern) for pattern in _EXCLUDED_FILENAME_GLOBS):
        return True
    if is_dir and rel_posix == "data/images":
        return True
    return False
