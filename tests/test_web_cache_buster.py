from __future__ import annotations

import http.client
import json
import re
import threading
from pathlib import Path

import pytest

from pipeline.web.build_id import get_build_id
from pipeline.web.server import create_server

_BUILD_ID_RE = re.compile(r"^\d{8}-\d{4}-[0-9a-f]{4}$")
_ASSET_RE = re.compile(
    r"<(script|link)\b[^>]*?\b(?:src|href)=([\"'])([^\"']+)\2",
    re.IGNORECASE,
)


@pytest.fixture
def cache_buster_repo(tmp_path: Path) -> Path:
    web = tmp_path / "web"
    web.mkdir(parents=True)
    (web / "index.html").write_text(
        """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>RUSH</title>
  <link rel="stylesheet" href="styles.css?v=web-ux-x4-pass3" />
</head>
<body>
  <script src="genai-sampler.js"></script>
  <script src="app.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/d3@7.9.0/dist/d3.min.js"></script>
</body>
</html>
""",
        encoding="utf-8",
    )
    (web / "styles.css").write_text("body{}", encoding="utf-8")
    (web / "app.js").write_text("console.log('app')", encoding="utf-8")
    (web / "genai-sampler.js").write_text("console.log('sampler')", encoding="utf-8")

    source = (
        tmp_path
        / "data/images/genai-classification/source-datasets/wfir/not_ai_generated/00046.jpg"
    )
    source.parent.mkdir(parents=True)
    source.write_bytes(b"fake source jpg")
    thumb = (
        tmp_path
        / "data/images/genai-classification/derived/thumbnails/wfir/not_ai_generated/00046.jpg"
    )
    thumb.parent.mkdir(parents=True)
    thumb.write_bytes(b"fake thumb jpg")
    return tmp_path


@pytest.fixture
def web_server(cache_buster_repo: Path):
    server = create_server(host="127.0.0.1", port=0, repo_root=cache_buster_repo)
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


def _build_id_from_api(server) -> str:
    status, body, _headers = _request(server, "GET", "/api/build-id")
    assert status == 200
    payload = json.loads(body.decode("utf-8"))
    return payload["build_id"]


def test_build_id_stable_within_process() -> None:
    assert get_build_id() == get_build_id()


def test_build_id_format() -> None:
    assert _BUILD_ID_RE.match(get_build_id())


def test_api_build_id_endpoint(web_server) -> None:
    status, body, _headers = _request(web_server, "GET", "/api/build-id")

    assert status == 200
    payload = json.loads(body.decode("utf-8"))
    assert _BUILD_ID_RE.match(payload["build_id"])
    assert payload["started_at"]


def test_index_html_has_meta_build_id(web_server) -> None:
    build_id = _build_id_from_api(web_server)

    status, body, headers = _request(web_server, "GET", "/web/index.html")

    assert status == 200
    assert headers["cache-control"] == "no-cache, must-revalidate"
    html = body.decode("utf-8")
    assert f'<meta name="build-id" content="{build_id}">' in html


def test_index_html_local_scripts_versioned(web_server) -> None:
    build_id = _build_id_from_api(web_server)
    status, body, _headers = _request(web_server, "GET", "/web/index.html")
    assert status == 200
    html = body.decode("utf-8")

    local_urls = [
        url
        for _tag, _quote, url in _ASSET_RE.findall(html)
        if "://" not in url and not url.startswith("//")
    ]

    assert local_urls
    assert all(f"v={build_id}" in url for url in local_urls)
    assert f"styles.css?v={build_id}" in local_urls


def test_web_prefixed_static_assets_load(web_server) -> None:
    for path, content_type in [
        ("/web/styles.css", "text/css"),
        ("/web/app.js", "text/javascript"),
    ]:
        status, body, headers = _request(web_server, "GET", path)
        assert status == 200
        assert body
        assert headers["content-type"].startswith(content_type)


def test_thumbnail_redirect_has_version_param(web_server) -> None:
    build_id = _build_id_from_api(web_server)
    path = "data/images/genai-classification/source-datasets/wfir/not_ai_generated/00046.jpg"

    status, _body, headers = _request(web_server, "GET", f"/api/thumbnail?path={path}")

    assert status == 302
    assert headers["location"].endswith(f"?v={build_id}")


def test_cdn_scripts_unmodified(web_server) -> None:
    status, body, _headers = _request(web_server, "GET", "/web/index.html")

    assert status == 200
    html = body.decode("utf-8")
    assert '<script src="https://cdn.jsdelivr.net/npm/d3@7.9.0/dist/d3.min.js"></script>' in html
