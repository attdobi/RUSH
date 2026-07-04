"""Scaffold for promoting confusion edges into boundary nodes.

MNIST starts with digit nodes that record ``confused_with`` edges such as
``MD.digit.1`` <-> ``MD.digit.7``. Those edges should remain as graph
relationships. At iterate time, RUSH can also materialize a boundary node such
as ``MD.boundary.1x7`` to hold boundary example images and to give
``is_boundary_between=["1", "7"]`` a concrete policy target.

This module only enumerates proposed node descriptors. Full materialization
(writing markdown/frontmatter, attaching examples, and exposing UI controls) is
left to the iterate-time workflow.
"""
from __future__ import annotations

import re
from typing import Any

_DIGIT_NODE_RE = re.compile(r"^MD\.digit\.(\d+)$")


def _digit_from_node_id(node_id: Any) -> int | None:
    match = _DIGIT_NODE_RE.fullmatch(str(node_id or "").strip())
    if not match:
        return None
    return int(match.group(1))


def _edge_type(edge: dict) -> str:
    return str(edge.get("edge_type") or edge.get("type") or "").strip()


def _source_node_id(edge: dict) -> Any:
    return (
        edge.get("source_node_id")
        or edge.get("source")
        or edge.get("from")
        or edge.get("node_id")
        or edge.get("id")
    )


def _target_node_id(edge: dict) -> Any:
    return edge.get("target_node_id") or edge.get("target") or edge.get("to")


def enumerate_boundary_promotions(edges: list[dict]) -> list[dict]:
    """Return proposed MNIST boundary nodes from ``confused_with`` edges.

    Accepts canonical ``edges.json`` records
    (``source_node_id``/``target_node_id``/``edge_type``) and frontmatter-style
    edge records (``type``/``to``) when a source is included on the dict via
    ``source_node_id``, ``source``, ``from``, ``node_id``, or ``id``.
    """
    pairs: set[tuple[int, int]] = set()
    for edge in edges:
        if _edge_type(edge) != "confused_with":
            continue
        source = _digit_from_node_id(_source_node_id(edge))
        target = _digit_from_node_id(_target_node_id(edge))
        if source is None or target is None or source == target:
            continue
        pairs.add(tuple(sorted((source, target))))

    promotions: list[dict[str, Any]] = []
    for a, b in sorted(pairs):
        promotions.append(
            {
                "node_id": f"MD.boundary.{a}x{b}",
                "digits": [a, b],
                "boundary_of": [f"MD.digit.{a}", f"MD.digit.{b}"],
                "node_type": "boundary",
            }
        )
    return promotions


__all__ = ["enumerate_boundary_promotions"]

