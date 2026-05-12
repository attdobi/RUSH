from __future__ import annotations

import http.client
import json
import threading
from pathlib import Path

import pytest

from pipeline.web.server import create_server


@pytest.fixture
def secure_web_server(tmp_path: Path):
    (tmp_path / "web").mkdir(parents=True)
    (tmp_path / "web" / "index.html").write_text(
        "<!doctype html><html><head><title>RUSH</title></head><body>ok</body></html>",
        encoding="utf-8",
    )
    (tmp_path / "data" / "runs").mkdir(parents=True)
    (tmp_path / "data" / "runs" / "index.json").write_text("{}", encoding="utf-8")
    (tmp_path / "policy-graph" / "Generative_AI" / "v0.1").mkdir(parents=True)
    (tmp_path / "policy-graph" / "Generative_AI" / "v0.1" / "root.md").write_text(
        "# Root", encoding="utf-8"
    )
    (tmp_path / "docs" / "visuals").mkdir(parents=True)
    (tmp_path / "docs" / "visuals" / "foo.svg").write_text(
        "<svg xmlns='http://www.w3.org/2000/svg'></svg>", encoding="utf-8"
    )
    (tmp_path / "tests").mkdir(parents=True)
    (tmp_path / "tests" / "test_web_server.py").write_text("secret", encoding="utf-8")
    (tmp_path / "pipeline" / "web").mkdir(parents=True)
    (tmp_path / "pipeline" / "web" / "server.py").write_text("secret", encoding="utf-8")

    server = create_server(host="127.0.0.1", port=0, repo_root=tmp_path)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        thread.join(timeout=2)
        server.server_close()


def _request(server, method: str, path: str) -> tuple[int, bytes, dict[str, str]]:
    host, port = server.server_address
    conn = http.client.HTTPConnection(host, port, timeout=2)
    conn.request(method, path)
    response = conn.getresponse()
    body = response.read()
    headers = {key.lower(): value for key, value in response.getheaders()}
    conn.close()
    return response.status, body, headers


def test_root_serves_web_index(secure_web_server) -> None:
    status, body, _headers = _request(secure_web_server, "GET", "/")

    assert status == 200
    assert b"<title>RUSH" in body


@pytest.mark.parametrize(
    "path",
    [
        "/.env",
        "/.git/HEAD",
        "/web/.git/HEAD",
        "/data",
        "/data/",
        "/data/runs/",
        "/tests/test_web_server.py",
        "/pipeline/web/server.py",
    ],
)
def test_forbidden_static_paths_return_404(secure_web_server, path: str) -> None:
    status, _body, _headers = _request(secure_web_server, "GET", path)

    assert status == 404


def test_whitelisted_data_file_served(secure_web_server) -> None:
    status, body, _headers = _request(secure_web_server, "GET", "/data/runs/index.json")

    assert status == 200
    assert json.loads(body.decode("utf-8")) == {}


def test_api_health_still_works(secure_web_server) -> None:
    status, body, _headers = _request(secure_web_server, "GET", "/api/health")

    assert status == 200
    assert json.loads(body.decode("utf-8"))["status"] == "ok"


def test_whitelisted_policy_graph_file_served(secure_web_server) -> None:
    status, body, _headers = _request(
        secure_web_server, "GET", "/policy-graph/Generative_AI/v0.1/root.md"
    )

    assert status == 200
    assert body == b"# Root"


def test_whitelisted_docs_visual_served(secure_web_server) -> None:
    status, body, _headers = _request(secure_web_server, "GET", "/docs/visuals/foo.svg")

    assert status == 200
    assert body.startswith(b"<svg")


def test_head_root_has_no_body(secure_web_server) -> None:
    status, body, _headers = _request(secure_web_server, "HEAD", "/")

    assert status == 200
    assert body == b""
