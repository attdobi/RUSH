#!/usr/bin/env python3
"""Run the local RUSH web server."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.web.server import create_server  # noqa: E402


def _load_dotenv(env_path: Path) -> int:
    """Tiny stdlib .env loader. Used so the launchd-managed server (which
    inherits a minimal env) and its labeling subprocesses can see the
    provider API keys persisted in ``<repo>/.env``. Existing process env
    vars win; values may be wrapped in single or double quotes.
    """
    if not env_path.is_file():
        return 0
    loaded = 0
    for raw in env_path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if not key or key in os.environ:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        os.environ[key] = value
        loaded += 1
    return loaded


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    loaded = _load_dotenv(args.repo_root / ".env")
    if loaded:
        print(f"[env] loaded {loaded} keys from {args.repo_root / '.env'}", flush=True)
    server = create_server(host=args.host, port=args.port, repo_root=args.repo_root)
    addr, port = server.server_address
    print(f"RUSH web server listening on http://{addr}:{port}/web/", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down RUSH web server", flush=True)
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
