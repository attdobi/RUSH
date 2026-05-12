from __future__ import annotations

import base64
from io import BytesIO
import http.client
import json
import threading
from pathlib import Path
from zipfile import ZipFile

import pytest

from pipeline.web.server import create_server


@pytest.fixture
def download_repo(tmp_path: Path) -> Path:
    (tmp_path / "web").mkdir(parents=True)
    (tmp_path / "web" / "index.html").write_text("<html>rush</html>\n", encoding="utf-8")
    (tmp_path / ".env").write_text("SECRET=do-not-ship\n", encoding="utf-8")
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")
    (tmp_path / "pipeline" / "web").mkdir(parents=True)
    (tmp_path / "pipeline" / "web" / "__init__.py").write_text("\n", encoding="utf-8")
    (tmp_path / "data" / "images").mkdir(parents=True)
    (tmp_path / "data" / "images" / "foo.jpg").write_bytes(b"fake image")
    (tmp_path / "data" / "runs").mkdir(parents=True)
    (tmp_path / "data" / "runs" / "keepme.json").write_text("{}\n", encoding="utf-8")
    (tmp_path / "some_file.py").write_text("print('keep me')\n", encoding="utf-8")
    (tmp_path / "__pycache__").mkdir()
    (tmp_path / "__pycache__" / "x.pyc").write_bytes(b"compiled")
    return tmp_path


@pytest.fixture
def web_server(download_repo: Path):
    server = create_server(host="127.0.0.1", port=0, repo_root=download_repo)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        thread.join(timeout=2)
        server.server_close()


def _basic(user: str, password: str) -> str:
    token = base64.b64encode(f"{user}:{password}".encode("utf-8")).decode("ascii")
    return f"Basic {token}"


def _request(server, path: str = "/download/rush.zip", *, auth: str | None = None):
    host, port = server.server_address
    conn = http.client.HTTPConnection(host, port, timeout=2)
    headers = {"Authorization": auth} if auth is not None else {}
    conn.request("GET", path, headers=headers)
    response = conn.getresponse()
    body = response.read()
    result = response.status, dict(response.getheaders()), body
    conn.close()
    return result


@pytest.fixture
def configured_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RUSH_DOWNLOAD_USER", "rush-user")
    monkeypatch.setenv("RUSH_DOWNLOAD_PASS", "rush-pass")


def test_download_without_credentials_requires_basic_auth(web_server, configured_env) -> None:
    status, headers, body = _request(web_server)

    assert status == 401
    assert headers["WWW-Authenticate"] == 'Basic realm="RUSH download"'
    assert json.loads(body.decode("utf-8"))["error"]["code"] == "auth_required"


@pytest.mark.parametrize("auth", [_basic("xxxx-user", "rush-pass"), _basic("nope", "rush-pass")])
def test_download_with_wrong_user_requires_basic_auth(web_server, configured_env, auth: str) -> None:
    status, headers, body = _request(web_server, auth=auth)

    assert status == 401
    assert headers["WWW-Authenticate"] == 'Basic realm="RUSH download"'
    assert json.loads(body.decode("utf-8"))["error"]["code"] == "auth_required"


def test_download_with_wrong_pass_requires_basic_auth(web_server, configured_env) -> None:
    status, headers, body = _request(web_server, auth=_basic("rush-user", "wrong-pass"))

    assert status == 401
    assert headers["WWW-Authenticate"] == 'Basic realm="RUSH download"'
    assert json.loads(body.decode("utf-8"))["error"]["code"] == "auth_required"


def test_download_with_correct_credentials_returns_filtered_zip(web_server, configured_env) -> None:
    status, headers, body = _request(web_server, auth=_basic("rush-user", "rush-pass"))

    assert status == 200
    assert headers["Content-Type"] == "application/zip"
    assert "rush-" in headers["Content-Disposition"]
    assert ".zip" in headers["Content-Disposition"]

    with ZipFile(BytesIO(body)) as archive:
        names = archive.namelist()

    assert any(name.endswith("/data/runs/keepme.json") for name in names)
    assert any(name.endswith("/some_file.py") for name in names)
    assert not any(name.endswith("/.env") for name in names)
    assert not any(name.endswith("/.git/HEAD") for name in names)
    assert not any(name.endswith("/data/images/foo.jpg") for name in names)
    assert not any(name.endswith("/__pycache__/x.pyc") for name in names)


def test_download_env_unset_returns_disabled(web_server, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("RUSH_DOWNLOAD_USER", raising=False)
    monkeypatch.delenv("RUSH_DOWNLOAD_PASS", raising=False)

    status, _headers, body = _request(web_server, auth=_basic("rush-user", "rush-pass"))

    assert status == 503
    assert json.loads(body.decode("utf-8"))["error"]["code"] == "download_disabled"
