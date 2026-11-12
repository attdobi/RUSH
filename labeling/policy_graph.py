"""Read-only policy graph and recorded-route contracts.

The directory reader adapts the materialized ``manifest.json`` / Markdown /
``edges.json`` format of attdobi/d-ai-trader's policy_graph package, inspected
at commit 02e35305c506a381e1f1edf499fb907e0628d5f0. It never imports or executes
that repository, its router, code-node bodies, or trading functions.

The upstream manifest's source_sha256 is a map of prompt-field hashes. Our
source_sha256 instead pins one complete normalized graph; upstream hashes and
native node versions remain explicitly named provenance.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any


NODE_TYPES = frozenset("root template field section rule lesson entry reminder identity note code data ltm ticker concept factor".split())
EDGE_TYPES = frozenset("subtype_of exception_to boundary_with confused_with clarifies example_of negative_example_of includes related_to cites overlaps constrains enforced_by triggers".split())
DIGEST_FIELDS = ("schema", "graph_id", "version", "root_ids", "nodes", "edges")
_HASH = re.compile(r"[0-9a-f]{64}\Z")


class GraphValidationError(ValueError):
    """A graph or trace cannot be tied safely to the configured policy."""


def _check(condition: Any, message: str) -> None:
    if not condition:
        raise GraphValidationError(message)


def _text(value: Any, name: str) -> str:
    _check(isinstance(value, str) and bool(value.strip()), f"{name} must be a nonempty string")
    return value


def _json(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise GraphValidationError("Graph records must contain finite JSON values") from exc


def _copy(value: Any) -> Any:
    return json.loads(_json(value))


def _probability(value: Any, name: str, *, nullable: bool = False) -> None:
    _check((nullable and value is None) or (isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and 0 <= value <= 1), f"{name} must be a finite number between zero and one")


def graph_digest(graph: dict) -> str:
    """Hash the six canonical graph fields, including verbatim node bodies."""
    _check(isinstance(graph, dict) and all(k in graph for k in DIGEST_FIELDS), "Missing canonical graph fields")
    return hashlib.sha256(_json({k: graph[k] for k in DIGEST_FIELDS}).encode("utf-8")).hexdigest()


def normalize_policy_graph(raw: dict) -> dict:
    """Validate a normalized graph without reordering or rewriting its content.

    IDs are domain-independent: RUSH's HAR-2 and trader's CA/DA IDs are both
    valid. Directed edge types retain their upstream semantics. Unknown node
    metadata is preserved inside the digest, not interpreted as executable code.
    """
    _check(isinstance(raw, dict), "Policy graph must be an object")
    graph = _copy(raw)
    _check(graph.get("schema") == 1 and not isinstance(graph.get("schema"), bool), "Unsupported policy graph schema")
    _text(graph.get("graph_id"), "graph_id")
    version = _text(graph.get("version"), "graph version")
    nodes = graph.get("nodes")
    _check(isinstance(nodes, list) and bool(nodes), "Graph needs a node catalog")
    catalog = {}
    for node in nodes:
        _check(isinstance(node, dict), "Graph nodes must be full objects")
        node_id = _text(node.get("id"), "node id")
        _check(node_id not in catalog, "Graph node IDs must be unique")
        _check(isinstance(node.get("node_type"), str) and node["node_type"] in NODE_TYPES, f"Unknown node type for {node_id}")
        _check(node.get("version") == version, f"Node {node_id} uses a different graph version")
        _check(isinstance(node.get("body"), str), f"Node {node_id} needs a verbatim body")
        digest = hashlib.sha256(node["body"].encode("utf-8")).hexdigest()
        _check(node.get("body_sha256") == digest, f"Node {node_id} body hash does not match")
        _check("locked" not in node or isinstance(node["locked"], bool), "Node locked must be boolean")
        catalog[node_id] = node
    roots = graph.get("root_ids")
    _check(isinstance(roots, list) and bool(roots) and all(isinstance(x, str) and x in catalog for x in roots), "Graph roots must identify catalog nodes")
    _check(len(roots) == len(set(roots)), "Graph roots cannot repeat")
    for node in nodes:
        parent = node.get("parent")
        _check(parent is None or isinstance(parent, str) and parent in catalog and parent != node["id"], "Node parent must identify another catalog node")
    edges = graph.get("edges")
    _check(isinstance(edges, list), "Graph edges must be an array")
    seen = set()
    for edge in edges:
        _check(isinstance(edge, dict), "Graph edges must be objects")
        source, target, kind = edge.get("source_node_id"), edge.get("target_node_id"), edge.get("edge_type")
        _check(isinstance(source, str) and source in catalog and isinstance(target, str) and target in catalog, "Graph edge references an unknown node")
        _check(isinstance(kind, str) and kind in EDGE_TYPES, "Graph edge has an unsupported type")
        _check(edge.get("version") == version, "Graph edge uses a different policy version")
        _probability(edge.get("confidence"), "Edge confidence", nullable=True)
        key = (source, target, kind)
        _check(key not in seen, "Graph edge triples cannot repeat")
        seen.add(key)
    calculated = graph_digest(graph)
    _check(graph.get("source_sha256", calculated) == calculated, "Policy graph source_sha256 does not match its content")
    graph["source_sha256"] = calculated
    return graph


def validate_route_trace(raw: dict, graph: dict | None, *, policy_version: str, router_version: str | None = None) -> dict:
    """Validate recorded evidence without discarding routing audit anomalies.

    Shadow selection can differ from the context actually served. Citations
    and missing pinned context are observations to preserve and flag, not
    reasons to reject an otherwise identifiable trace. Retained score mass
    is not measured recall.
    """
    _check(isinstance(raw, dict), "route.trace must be an object")
    _check(isinstance(graph, dict), "A recorded route needs a configured policy graph")
    trace = _copy(raw)
    for key in ("run_id", "router_version", "requested_mode", "effective_mode", "source_kind"):
        _text(trace.get(key), key)
    _check(trace.get("policy_version") == policy_version == graph.get("version"), "Route trace policy version does not match the item and graph")
    _check(trace.get("graph_hash") == graph.get("source_sha256"), "Route trace graph hash does not match the configured snapshot")
    if router_version is not None:
        _check(trace["router_version"] == router_version, "Route trace router version does not match its route")
    catalog = {node["id"]: node for node in graph["nodes"]}
    sets = {}
    for key in ("pinned_ids", "selected_ids", "served_ids", "cited_ids"):
        ids = trace.get(key)
        _check(isinstance(ids, list) and all(isinstance(x, str) and x in catalog for x in ids), f"{key} contains unknown graph nodes")
        _check(len(ids) == len(set(ids)), f"{key} must not repeat nodes")
        sets[key] = set(ids)
    # Recompute diagnostics rather than trusting imported diagnostic fields.
    trace["selected_not_served_ids"] = sorted(sets["selected_ids"] - sets["served_ids"])
    trace["unserved_cited_ids"] = sorted(sets["cited_ids"] - sets["served_ids"])
    trace["missing_pinned_ids"] = sorted(sets["pinned_ids"] - sets["served_ids"])
    _check(trace.get("fallback_reason") is None or isinstance(trace["fallback_reason"], str), "fallback_reason must be text or null")
    for key in ("recall_target", "retained_score_mass"):
        _probability(trace.get(key), key, nullable=True)
    decisions = trace.get("decisions")
    _check(isinstance(decisions, list), "Route trace decisions must be an array")
    seen = set()
    for decision in decisions:
        _check(isinstance(decision, dict), "Each route decision must be an object")
        node_id = decision.get("node_id")
        _check(isinstance(node_id, str) and node_id in catalog and node_id not in seen, "Route decisions require distinct known node IDs")
        seen.add(node_id)
        _text(decision.get("kind"), "decision kind")
        _text(decision.get("choice"), "decision choice")
        _text(decision.get("why"), "decision explanation")
        for key in ("p", "confidence"):
            _probability(decision.get(key), f"Decision {key}", nullable=True)
        _check(isinstance(decision.get("pinned"), bool) and decision["pinned"] == (node_id in sets["pinned_ids"]), "Decision pin flag must match pinned_ids")
    return trace


# Small parser for the upstream writer's deliberately restricted frontmatter.
# JSON-quoted scalars preserve escapes. This is not a general YAML interpreter.
def _parts(value: str) -> list[str]:
    result, buffer, quote, escaped = [], "", None, False
    for char in value:
        if quote:
            buffer += char
            if escaped:
                escaped = False
            elif char == "\\" and quote == '"':
                escaped = True
            elif char == quote:
                quote = None
        elif char in ('"', "'"):
            buffer += char; quote = char
        elif char == ",":
            result.append(buffer.strip()); buffer = ""
        else:
            buffer += char
    _check(quote is None, "Unclosed frontmatter string")
    if buffer.strip(): result.append(buffer.strip())
    return result


def _scalar(raw: str) -> Any:
    value = raw.strip()
    if value.startswith('"'):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError as exc:
            raise GraphValidationError("Invalid frontmatter JSON string") from exc
        _check(isinstance(parsed, str), "Expected a frontmatter string")
        return parsed
    if value.startswith("'") and value.endswith("'"):
        return value[1:-1]
    if value in {"null", "~"}: return None
    if value == "true": return True
    if value == "false": return False
    if re.fullmatch(r"-?\d+", value): return int(value)
    if re.fullmatch(r"-?\d+\.\d+", value): return float(value)
    _check(value not in {"|", ">"}, "Multiline YAML scalars are not supported by the upstream writer")
    return value


def _inline_map(raw: str) -> dict:
    _check(raw.startswith("{") and raw.endswith("}"), "Expected an inline frontmatter map")
    result = {}
    for part in _parts(raw[1:-1]):
        key, sep, value = part.partition(":")
        _check(bool(sep) and bool(key.strip()) and key.strip() not in result, "Invalid frontmatter map key")
        result[key.strip()] = _scalar(value)
    return result


def read_node_bytes(data: bytes) -> dict:
    """Parse one upstream node, preserving every byte of its UTF-8 body."""
    _check(data.startswith(b"---\n"), "Node file must start with the upstream opening fence")
    end = data.find(b"\n---\n", 4)
    _check(end >= 0, "Closing frontmatter fence is missing")
    try:
        frontmatter, body = data[4:end].decode("utf-8"), data[end + 5:].decode("utf-8")
    except UnicodeDecodeError as exc:
        raise GraphValidationError("Nodes must use UTF-8") from exc
    fields, list_key = {}, None
    for line in frontmatter.split("\n"):
        if not line.strip(): continue
        if list_key and line.startswith("  - "):
            fields[list_key].append(_inline_map(line[4:].strip())); continue
        list_key = None
        match = re.fullmatch(r"([A-Za-z_][\w-]*):(.*)", line)
        _check(match is not None, "Unsupported nested or invalid frontmatter line")
        key, raw = match.group(1), match.group(2).strip()
        _check(key not in fields, "Frontmatter keys cannot repeat")
        if key == "edges" and not raw:
            fields[key] = []; list_key = key
        elif raw.startswith("[") and raw.endswith("]"):
            fields[key] = [_scalar(x) for x in _parts(raw[1:-1])]
        elif raw.startswith("{") and raw.endswith("}"):
            fields[key] = _inline_map(raw)
        else:
            fields[key] = _scalar(raw)
    _check(fields.get("body_sha256") == hashlib.sha256(body.encode("utf-8")).hexdigest(), "Native node body_sha256 does not match verbatim body")
    fields["body"] = body
    return fields


def load_version_directory(path: str | Path, *, graph_id: str | None = None) -> dict:
    """Read one materialized trader version, with bounded code/LTM overlays.

    Only local JSON/Markdown files are read. Manifest overlay paths must resolve
    under the sibling ``_code`` or ``_ltm`` directory and must exist; missing
    overlays are an error rather than silently incomplete provenance. Symlink
    escapes and arbitrary absolute overlay paths are rejected.
    """
    version_dir = Path(path).resolve(strict=True)
    _check(version_dir.is_dir(), "Version path must be a directory")

    def read_local(file: Path, root: Path) -> bytes:
        resolved = file.resolve(strict=True)
        _check(resolved.is_relative_to(root) and resolved.is_file(), "Graph file escapes its allowed directory")
        _check(resolved.stat().st_size <= 16 * 1024 * 1024, "Graph file exceeds the read-only adapter limit")
        return resolved.read_bytes()

    def read_json(file: Path) -> Any:
        try:
            return json.loads(read_local(file, version_dir).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise GraphValidationError(f"Invalid JSON in {file.name}") from exc

    manifest = read_json(version_dir / "manifest.json")
    _check(isinstance(manifest, dict) and manifest.get("schema") == 1, "Unsupported native manifest")
    agent = _text(manifest.get("agent_type"), "manifest agent_type")
    config = _text(manifest.get("config_hash"), "manifest config_hash")
    number = manifest.get("version")
    _check(isinstance(number, int) and not isinstance(number, bool) and number >= 1, "Native manifest version must be a positive integer")
    version = f"{agent}.{config}.v{number}"
    nodes = {}

    def read_nodes(directory: Path, *, overlay: str | None = None) -> None:
        for file in sorted(directory.glob("*.md")):
            node = read_node_bytes(read_local(file, directory))
            node_id = _text(node.get("id"), "native node id")
            if not overlay:
                _check(file.name == node_id + ".md", "Native filename must match node ID")
                _check(node.get("version") == version, "Native node version does not match manifest")
            else:
                expected = f"{overlay}@{(manifest.get(overlay) or {}).get('sha')}"
                _check(node.get("version") == expected, "Overlay node version does not match pinned manifest SHA")
                node["upstream_node_version"] = node["version"]
                node["version"] = version
            _check(node_id not in nodes, "Duplicate node across native graph and overlays")
            nodes[node_id] = node

    read_nodes(version_dir)
    for overlay in ("code", "ltm"):
        spec = manifest.get(overlay) or {}
        _check(isinstance(spec, dict), "Native overlay metadata must be an object")
        link = spec.get("dir")
        if link is not None:
            _text(link, f"{overlay}.dir")
            _check(not Path(link).is_absolute(), "Absolute overlay paths are not accepted")
            allowed = version_dir.parent / f"_{overlay}"
            _check(not allowed.is_symlink(), "Overlay roots cannot be symlink escapes")
            directory = (version_dir / link).resolve(strict=True)
            allowed = allowed.resolve(strict=True)
            _check(directory.is_relative_to(allowed) and directory != allowed and directory.is_dir(), "Overlay path must stay under its sibling overlay directory")
            read_nodes(directory, overlay=overlay)
    records = read_json(version_dir / "edges.json") if (version_dir / "edges.json").exists() else []
    _check(isinstance(records, list), "Native edges.json must contain an array")
    edges, seen = [], set()

    def add_edge(edge: dict) -> None:
        key = (edge.get("source_node_id"), edge.get("target_node_id"), edge.get("edge_type"))
        _check(all(isinstance(x, str) for x in key), "Native edge requires source, target and type")
        if key not in seen:
            seen.add(key); edges.append(edge)

    for record in records:
        _check(isinstance(record, dict), "Invalid native edge record")
        edge = {"source_node_id": record.get("source_node_id") or record.get("source"),
                "target_node_id": record.get("target_node_id") or record.get("target"),
                "edge_type": record.get("edge_type") or record.get("type"),
                "confidence": record.get("confidence", 1.0), "provenance": record.get("provenance", "derived"),
                "version": record.get("version") or version}
        if record.get("via") is not None: edge["via"] = record["via"]
        add_edge(edge)
    for node in nodes.values():
        authored = node.get("edges", []) or []
        _check(isinstance(authored, list), "Native authored edges must be an array")
        for record in authored:
            _check(isinstance(record, dict), "Native authored edge must be an object")
            kind, target = record.get("type") or record.get("edge_type"), record.get("to") or record.get("target")
            if not kind or not target or kind == "subtype_of": continue
            edge = {"source_node_id": node["id"], "target_node_id": target, "edge_type": kind,
                    "confidence": record.get("confidence", 1.0), "provenance": record.get("provenance", "authored"), "version": version}
            if record.get("via") is not None: edge["via"] = record["via"]
            add_edge(edge)
    compile_order = manifest.get("compile_order", {})
    _check(isinstance(compile_order, dict) and all(isinstance(ids, list) and all(isinstance(x, str) and x in nodes for x in ids) for ids in compile_order.values()), "Native compile_order references missing nodes")
    upstream_hashes = manifest.get("source_sha256", {})
    _check(isinstance(upstream_hashes, dict) and all(value is None or isinstance(value, str) and bool(_HASH.fullmatch(value)) for value in upstream_hashes.values()), "Native source_sha256 must be a field-to-hash map")
    graph = {"schema": 1, "graph_id": graph_id or f"{agent}.{config}", "version": version,
             "root_ids": [manifest.get("root_id")], "nodes": sorted(nodes.values(), key=lambda n: n["id"]),
             "edges": sorted(edges, key=lambda e: (e["source_node_id"], e["target_node_id"], e["edge_type"])),
             "upstream_source_sha256": upstream_hashes, "upstream_manifest": manifest,
             "provenance": "Read-only d-ai-trader materialized policy graph; no router execution"}
    return normalize_policy_graph(graph)


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize a local d-ai-trader materialized policy graph without running its code")
    parser.add_argument("version_directory", type=Path)
    parser.add_argument("--graph-id")
    args = parser.parse_args()
    print(json.dumps(load_version_directory(args.version_directory, graph_id=args.graph_id), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
