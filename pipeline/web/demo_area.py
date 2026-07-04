"""Demo/policy-area helpers shared by local web API handlers."""
from __future__ import annotations

from typing import Any

DEFAULT_POLICY_AREA = "Generative_AI"
MNIST_POLICY_AREA = "MNIST_Digits"
ALLOWED_POLICY_AREAS = {DEFAULT_POLICY_AREA, MNIST_POLICY_AREA}

_DEMO_TO_AREA = {
    "genai": DEFAULT_POLICY_AREA,
    "generative_ai": DEFAULT_POLICY_AREA,
    "mnist": MNIST_POLICY_AREA,
    "mnist_digits": MNIST_POLICY_AREA,
}


def normalize_policy_area(
    area: str | None = None,
    *,
    demo: str | None = None,
    default: str = DEFAULT_POLICY_AREA,
) -> str:
    """Return a validated policy area from explicit area or demo alias."""
    raw = area if area not in {None, ""} else None
    if raw is None and demo not in {None, ""}:
        raw = _DEMO_TO_AREA.get(str(demo).strip().lower(), str(demo).strip())
    selected = str(raw or default).strip()
    if selected not in ALLOWED_POLICY_AREAS:
        raise ValueError(f"unsupported policy area: {selected!r}")
    return selected


def first_query_value(query: dict[str, list[str]], key: str) -> str | None:
    values = query.get(key)
    if not values:
        return None
    value = values[0]
    return value if value != "" else None


def area_from_query(query: dict[str, list[str]]) -> str:
    return normalize_policy_area(
        first_query_value(query, "area"),
        demo=first_query_value(query, "demo"),
    )


def policy_version_matches_area(version: Any, area: str) -> bool:
    """Return whether a manifest/scoring policy version belongs to ``area``.

    Historical GenAI runs sometimes stored bare versions such as ``v0.1``.
    Treat those as GenAI for back-compat; MNIST must use the explicit
    ``MNIST_Digits.`` prefix.
    """
    if version is None:
        return area == DEFAULT_POLICY_AREA
    text = str(version)
    if text.startswith(f"{area}."):
        return True
    return area == DEFAULT_POLICY_AREA and text.startswith("v")
