"""Safety and request-validation helpers for the local RUSH web server."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Any
from urllib.parse import unquote, urlsplit

from pipeline.providers.registry import MODEL_REGISTRY
from pipeline.web.demo_area import normalize_policy_area

_ALLOWED_SPLITS = {"dev_golden", "holdout", "all"}
_ALLOWED_MODES = {"cold_start", "warm_start"}
_ALLOWED_REASONING_EFFORTS = {"high", "xhigh"}
_POLICY_VERSION_RE = re.compile(r"^v\d+(\.\d+)?$")
DEFAULT_BATCH_SIZE = 20

_STATIC_PREFIXES: tuple[tuple[str, str], ...] = (
    # /api/thumbnail 302-redirects into one of these two directories depending on
    # whether a derived thumbnail exists; both must be reachable as static GETs.
    (
        "/data/images/genai-classification/derived/thumbnails/",
        "data/images/genai-classification/derived/thumbnails",
    ),
    (
        "/data/images/genai-classification/source-datasets/",
        "data/images/genai-classification/source-datasets",
    ),
    (
        "/data/images/genai-classification/sample/",
        "data/images/genai-classification/sample",
    ),
    (
        "/data/images/genai-classification/manifests/",
        "data/images/genai-classification/manifests",
    ),
    (
        "/data/images/mnist-classification/manifests/",
        "data/images/mnist-classification/manifests",
    ),
    (
        "/data/images/mnist-classification/derived/thumbnails/",
        "data/images/mnist-classification/derived/thumbnails",
    ),
    (
        "/data/images/mnist-classification/source-datasets/",
        "data/images/mnist-classification/source-datasets",
    ),
    ("/data/runs/", "data/runs"),
    ("/policy-graph/", "policy-graph"),
    ("/docs/visuals/", "docs/visuals"),
    ("/schemas/", "schemas"),
)


class APIError(Exception):
    """Exception that maps cleanly onto the JSON error envelope."""

    def __init__(
        self,
        status: int,
        code: str,
        message: str,
        *,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message
        self.details = details or {}

    def to_payload(self) -> dict[str, Any]:
        error: dict[str, Any] = {"code": self.code, "message": self.message}
        if self.details:
            error["details"] = self.details
        return {"error": error}


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def ensure_repo_relative(repo_root: Path, candidate: Path) -> Path:
    """Resolve ``candidate`` and require it to remain under ``repo_root``."""
    root = repo_root.resolve()
    resolved = candidate.resolve()
    if not resolved.is_relative_to(root):
        raise APIError(404, "not_found", "path is outside the repository root")
    return resolved


def _ensure_under_root(root: Path, candidate: Path) -> Path:
    """Resolve ``candidate`` and require it to remain under ``root``."""
    resolved_root = root.resolve()
    resolved = candidate.resolve()
    if not resolved.is_relative_to(resolved_root):
        raise APIError(404, "not_found", "path is outside the allowed static root")
    return resolved


def _normalized_url_path(request_path: str) -> tuple[str, list[str]]:
    parsed_path = unquote(urlsplit(request_path).path)
    if "\x00" in parsed_path:
        raise APIError(400, "bad_path", "path contains a NUL byte")
    parts = [part for part in parsed_path.split("/") if part]
    if any(part.startswith(".") for part in parts):
        raise APIError(404, "not_found", "dotfile access is not allowed")
    normalized = "/" + "/".join(parts)
    if parsed_path.endswith("/") and normalized != "/":
        normalized += "/"
    return normalized, parts


def whitelisted_static_prefix(request_path: str) -> str | None:
    """Return the URL whitelist prefix matching ``request_path``, if any."""
    normalized, _parts = _normalized_url_path(request_path)
    for url_prefix, _repo_rel in _STATIC_PREFIXES:
        if normalized.startswith(url_prefix):
            return url_prefix
    return None


def safe_static_path(
    repo_root: Path,
    web_root_or_request_path: Path | str,
    request_path: str | None = None,
) -> Path:
    """Translate a URL path to an allowed read-only static file path safely.

    Generic static URLs are rooted at ``web_root``. Only explicit read-only URL
    prefixes may resolve to repository directories outside ``web_root``.
    """
    if request_path is None:
        web_root = Path(repo_root) / "web"
        request_path = str(web_root_or_request_path)
    else:
        web_root = Path(web_root_or_request_path)

    repo = repo_root.resolve()
    web = web_root.resolve()
    normalized, parts = _normalized_url_path(request_path)

    for url_prefix, repo_rel in _STATIC_PREFIXES:
        if normalized.startswith(url_prefix):
            suffix = normalized.removeprefix(url_prefix).strip("/")
            suffix_parts = [part for part in suffix.split("/") if part]
            allowed_root = (repo / repo_rel).resolve()
            candidate = allowed_root.joinpath(*suffix_parts) if suffix_parts else allowed_root
            return _ensure_under_root(allowed_root, candidate)

    # Browsers resolve relative assets from /web/ and /web/index.html as /web/*.
    # Treat that route prefix as the web root instead of 404ing CSS/JS assets.
    if parts and parts[0] == "web":
        parts = parts[1:]

    rel = Path(*parts) if parts else Path(".")
    return _ensure_under_root(web, web / rel)


def read_json_body(handler, *, max_bytes: int = 64 * 1024) -> dict[str, Any]:
    raw_len = handler.headers.get("Content-Length", "0")
    try:
        length = int(raw_len)
    except ValueError as exc:
        raise APIError(400, "bad_content_length", "Content-Length must be an integer") from exc
    if length < 0 or length > max_bytes:
        raise APIError(413, "body_too_large", "request body is too large")
    raw = handler.rfile.read(length) if length else b"{}"
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise APIError(400, "bad_json", "request body must be valid JSON") from exc
    if not isinstance(payload, dict):
        raise APIError(400, "bad_request", "request JSON must be an object")
    return payload


def _require_bool_true(payload: dict[str, Any], name: str, message: str) -> None:
    if payload.get(name) is not True:
        raise APIError(400, "validation_error", message, details={"field": name})


def _validate_local_reasoning(payload: dict[str, Any]) -> dict[str, bool]:
    raw = payload.get("local_reasoning")
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise APIError(
            400,
            "validation_error",
            "local_reasoning must be an object mapping local model ids to booleans",
            details={"field": "local_reasoning"},
        )

    out: dict[str, bool] = {}
    for model_id, enabled in raw.items():
        if not isinstance(model_id, str) or not model_id.startswith("local/"):
            raise APIError(
                400,
                "validation_error",
                "local_reasoning keys must be local model ids",
                details={"field": "local_reasoning", "model_id": model_id},
            )
        reg_spec = MODEL_REGISTRY.get(model_id)
        if reg_spec is None or reg_spec.provider != "local":
            raise APIError(
                400,
                "unknown_model_id",
                f"unknown local model_id: {model_id}",
                details={"field": "local_reasoning", "model_id": model_id},
            )
        if not isinstance(enabled, bool):
            raise APIError(
                400,
                "validation_error",
                "local_reasoning values must be booleans",
                details={"field": "local_reasoning", "model_id": model_id},
            )
        out[model_id] = enabled
    return out


def validate_start_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Validate and normalize ``POST /api/runs/start`` JSON."""
    raw_demo = payload.get("demo")
    if raw_demo is not None and not isinstance(raw_demo, str):
        raise APIError(
            400,
            "validation_error",
            "demo must be a string when provided",
            details={"field": "demo"},
        )
    raw_area = payload.get("area")
    if raw_area is not None and not isinstance(raw_area, str):
        raise APIError(
            400,
            "validation_error",
            "area must be a string when provided",
            details={"field": "area"},
        )
    try:
        area = normalize_policy_area(raw_area, demo=raw_demo)
    except ValueError as exc:
        raise APIError(
            400,
            "validation_error",
            str(exc),
            details={"field": "area"},
        ) from exc

    raw_models = payload.get("models")
    if not isinstance(raw_models, list) or not raw_models:
        raise APIError(400, "validation_error", "models must be a non-empty list")
    models: list[str] = []
    for model in raw_models:
        if not isinstance(model, str) or not model.strip():
            raise APIError(400, "validation_error", "models must contain non-empty strings")
        model_id = model.strip()
        if model_id not in MODEL_REGISTRY:
            raise APIError(
                400,
                "unknown_model_id",
                f"unknown model_id: {model_id}",
                details={"model_id": model_id},
            )
        if model_id not in models:
            models.append(model_id)

    split = payload.get("split", "all")
    if split not in _ALLOWED_SPLITS:
        raise APIError(
            400,
            "validation_error",
            "split must be one of: all, dev_golden, holdout",
            details={"field": "split"},
        )

    mode = payload.get("mode", "cold_start")
    if mode not in _ALLOWED_MODES:
        raise APIError(
            400,
            "validation_error",
            "mode must be one of: cold_start, warm_start",
            details={"field": "mode"},
        )

    reasoning_effort = payload.get("reasoning_effort")
    if reasoning_effort is not None and (
        not isinstance(reasoning_effort, str)
        or reasoning_effort not in _ALLOWED_REASONING_EFFORTS
    ):
        raise APIError(
            400,
            "validation_error",
            "reasoning_effort must be one of: high, xhigh",
            details={"field": "reasoning_effort"},
        )

    local_reasoning = _validate_local_reasoning(payload)

    policy_version = payload.get("policy_version", "v0.1")
    if not isinstance(policy_version, str) or not _POLICY_VERSION_RE.match(policy_version):
        raise APIError(
            400,
            "validation_error",
            "policy_version must match ^v\\d+(\\.\\d+)?$",
            details={"field": "policy_version"},
        )

    _require_bool_true(
        payload,
        "allow_spend",
        "allow_spend: true is required before starting a live run",
    )
    if split in {"holdout", "all"}:
        _require_bool_true(
            payload,
            "allow_holdout",
            "allow_holdout: true is required before labeling holdout/testing records",
        )

    concurrency = payload.get("concurrency", 1)
    if not isinstance(concurrency, int) or isinstance(concurrency, bool) or not (1 <= concurrency <= 4):
        raise APIError(
            400,
            "validation_error",
            "concurrency must be an integer in [1, 4]",
            details={"field": "concurrency"},
        )

    batch_size = payload.get("batch_size", DEFAULT_BATCH_SIZE)
    if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < 1:
        raise APIError(
            400,
            "validation_error",
            "batch_size must be a positive integer",
            details={"field": "batch_size"},
        )

    has_limit = payload.get("limit") is not None
    has_sample_ids = payload.get("sample_ids") is not None
    if has_limit == has_sample_ids:
        raise APIError(
            400,
            "validation_error",
            "provide exactly one of limit or sample_ids",
            details={"fields": ["limit", "sample_ids"]},
        )

    limit = payload.get("limit")
    if limit is not None and (
        not isinstance(limit, int) or isinstance(limit, bool) or limit < 1
    ):
        raise APIError(
            400,
            "validation_error",
            "limit must be a positive integer",
            details={"field": "limit"},
        )

    sample_ids: str | None = None
    raw_sample_ids = payload.get("sample_ids")
    if raw_sample_ids is not None:
        if isinstance(raw_sample_ids, str):
            ids = [s.strip() for s in raw_sample_ids.split(",") if s.strip()]
        elif isinstance(raw_sample_ids, list):
            ids = []
            for item in raw_sample_ids:
                if not isinstance(item, str) or not item.strip():
                    raise APIError(
                        400,
                        "validation_error",
                        "sample_ids list must contain non-empty strings",
                    )
                ids.append(item.strip())
        else:
            raise APIError(
                400,
                "validation_error",
                "sample_ids must be a CSV string or list of strings",
                details={"field": "sample_ids"},
            )
        if not ids:
            raise APIError(400, "validation_error", "sample_ids must not be empty")
        sample_ids = ",".join(ids)

    return {
        "models": models,
        "demo": raw_demo.strip() if isinstance(raw_demo, str) and raw_demo.strip() else None,
        "area": area,
        "split": split,
        "limit": limit,
        "sample_ids": sample_ids,
        "policy_version": policy_version,
        "mode": mode,
        "reasoning_effort": reasoning_effort,
        "local_reasoning": local_reasoning,
        "allow_spend": True,
        "allow_holdout": payload.get("allow_holdout") is True,
        "concurrency": concurrency,
        "batch_size": batch_size,
    }
