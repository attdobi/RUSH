#!/usr/bin/env python3
"""Validate the RUSH foundation scaffold without external dependencies.

This intentionally implements only lightweight contract checks. Full JSON Schema
validation can be added later, but the foundation validator should already catch
broken graph references, missing schema files, and misleading seed metrics.
"""
from __future__ import annotations

import json
from pathlib import Path
import re
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
GRAPH = ROOT / "policy-graph" / "Generative_AI" / "v0.1"
SCHEMAS = ROOT / "schemas"
SEED = ROOT / "data" / "seed"

REQUIRED_FILES = [
    ROOT / "README.md",
    ROOT / "web" / "index.html",
    ROOT / "web" / "styles.css",
    ROOT / "web" / "app.js",
    ROOT / "docs" / "visuals" / "rush-system.svg",
    ROOT / "docs" / "visuals" / "policy-evolution.svg",
]

REQUIRED_SCHEMAS = [
    "arbiter-decision.schema.json",
    "decision-quality.schema.json",
    "export-record.schema.json",
    "image-record.schema.json",
    "label-tier-record.schema.json",
    "label-vote.schema.json",
    "llm-output-format.schema.json",
    "metric-snapshot.schema.json",
    "policy-edge.schema.json",
    "policy-node.schema.json",
    "policy-patch.schema.json",
    "sme-review.schema.json",
    "split-assignment.schema.json",
]

VALID_SPLITS = {
    "development",
    "validation",
    "locked_holdout",
    "boundary_holdout",
    "production_sentinel_random",
    "adaptive_boundary_batch",
}
VALID_LABELS = {"gen_ai", "not_gen_ai", "abstain", "positive", "negative"}
VALID_TIERS = {"provisional", "silver", "gold", "platinum", "deprecated", "superseded"}

SCHEMA_REQUIRED_FIELD_EXPECTATIONS = {
    "decision-quality.schema.json": {
        "labeler_id",
        "labeler_type",
        "policy_graph_version",
        "ground_truth_source",
        "metrics",
        "timestamp",
    },
    "llm-output-format.schema.json": {
        "label",
        "l2_label",
        "justification",
        "confidence",
        "difficulty",
        "is_boundary",
    },
    "metric-snapshot.schema.json": {
        "metric_snapshot_id",
        "policy_graph_version",
        "truth_label_tiers",
        "split",
        "status",
        "denominators",
        "metrics",
    },
}


def parse_scalar(value: str) -> Any:
    value = value.strip().strip('"')
    if value in {"null", "None", "~"}:
        return None
    if value in {"true", "false"}:
        return value == "true"
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value


def parse_frontmatter(path: Path) -> tuple[dict[str, Any], list[str]]:
    text = path.read_text(encoding="utf-8")
    if not text.startswith("---\n"):
        raise ValueError(f"{path.relative_to(ROOT)} is missing YAML frontmatter")
    block = text.split("---", 2)[1]
    data: dict[str, Any] = {}
    for raw in block.splitlines():
        if not raw.strip() or raw.startswith(" ") or raw.startswith("  -"):
            continue
        if ":" in raw:
            key, value = raw.split(":", 1)
            data[key.strip()] = parse_scalar(value)
    edge_refs = re.findall(r"\bto:\s*([^,}\s]+)", block)
    return data, edge_refs


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def require_fields(errors: list[str], obj: dict[str, Any], fields: list[str], context: str) -> None:
    for field in fields:
        if field not in obj:
            errors.append(f"{context} missing required field {field}")


def validate_schema_files(errors: list[str]) -> None:
    for name in REQUIRED_SCHEMAS:
        path = SCHEMAS / name
        if not path.exists():
            errors.append(f"missing required schema: schemas/{name}")
            continue
        try:
            schema = load_json(path)
        except Exception as exc:  # noqa: BLE001 - validation report
            errors.append(f"schema does not parse as JSON: schemas/{name}: {exc}")
            continue
        if schema.get("type") != "object":
            errors.append(f"schemas/{name} must declare type=object")
        if not isinstance(schema.get("title"), str) or not schema["title"]:
            errors.append(f"schemas/{name} missing title")
        if not isinstance(schema.get("properties"), dict) or not schema["properties"]:
            errors.append(f"schemas/{name} missing properties object")
        required = schema.get("required")
        if not isinstance(required, list) or not required:
            errors.append(f"schemas/{name} missing required field list")
        expected_required = SCHEMA_REQUIRED_FIELD_EXPECTATIONS.get(name)
        if expected_required and not expected_required.issubset(set(required or [])):
            missing = sorted(expected_required - set(required or []))
            errors.append(f"schemas/{name} missing required fields from checklist: {missing}")


def validate_graph(errors: list[str]) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    node_meta: dict[str, dict[str, Any]] = {}
    frontmatter_edge_refs: dict[str, list[str]] = {}

    for md in sorted(GRAPH.glob("*.md")):
        try:
            fm, edge_refs = parse_frontmatter(md)
        except Exception as exc:  # noqa: BLE001 - validation report
            errors.append(str(exc))
            continue
        for key in ["id", "version", "title", "area", "node_type", "polarity", "coverage_target"]:
            if key not in fm or (key != "coverage_target" and fm[key] == ""):
                errors.append(f"{md.relative_to(ROOT)} missing frontmatter key {key}")
        node_id = fm.get("id")
        if not isinstance(node_id, str) or not node_id:
            continue
        if node_id in node_meta:
            errors.append(f"duplicate node id: {node_id}")
        node_meta[node_id] = {**fm, "path": str(md.relative_to(ROOT))}
        frontmatter_edge_refs[node_id] = edge_refs

    root_ids = [nid for nid, fm in node_meta.items() if fm.get("node_type") == "root"]
    if root_ids != ["GA.root"]:
        errors.append(f"expected exactly one root node GA.root with node_type=root; found {root_ids or 'none'}")
    if "GA.root" in node_meta and node_meta["GA.root"].get("parent") is not None:
        errors.append("GA.root must have parent: null")

    for node_id, fm in node_meta.items():
        parent = fm.get("parent")
        if node_id == "GA.root":
            continue
        if not isinstance(parent, str) or parent not in node_meta:
            errors.append(f"{node_id} has missing/unknown parent {parent!r}")
        if fm.get("node_type") == "root":
            errors.append(f"non-root node {node_id} incorrectly declares node_type=root")

    for node_id, refs in frontmatter_edge_refs.items():
        for ref in refs:
            if ref == "ROOT" or ref not in node_meta:
                errors.append(f"{node_id} frontmatter edge references unknown node {ref}")

    try:
        edges = load_json(GRAPH / "edges.json")
    except Exception as exc:  # noqa: BLE001 - validation report
        errors.append(f"policy edge manifest does not parse: {exc}")
        edges = []
    if not isinstance(edges, list):
        errors.append("policy edge manifest must be a JSON array")
        edges = []

    seen_edges: set[tuple[str, str, str]] = set()
    for idx, edge in enumerate(edges):
        if not isinstance(edge, dict):
            errors.append(f"edge #{idx} is not an object")
            continue
        require_fields(errors, edge, ["source_node_id", "target_node_id", "edge_type", "confidence", "provenance", "version"], f"edge #{idx}")
        source = edge.get("source_node_id")
        target = edge.get("target_node_id")
        edge_type = edge.get("edge_type")
        for field, ref in [("source_node_id", source), ("target_node_id", target)]:
            if ref == "ROOT" or ref not in node_meta:
                errors.append(f"edge #{idx} {field} references unknown node {ref}")
        key = (str(source), str(target), str(edge_type))
        if key in seen_edges:
            errors.append(f"duplicate edge: {key}")
        seen_edges.add(key)

    # Parent-chain reachability catches orphaned nodes even when the edge manifest exists.
    for node_id in node_meta:
        if node_id == "GA.root":
            continue
        visited: set[str] = set()
        current = node_id
        while current != "GA.root":
            if current in visited:
                errors.append(f"parent cycle detected at {node_id}")
                break
            visited.add(current)
            parent = node_meta.get(current, {}).get("parent")
            if parent not in node_meta:
                errors.append(f"orphan node not reachable to GA.root: {node_id}")
                break
            current = parent

    return node_meta, edges


def validate_seed_data(errors: list[str], node_ids: set[str]) -> tuple[int, int]:
    image_records = load_json(SEED / "image-records.json")
    if not isinstance(image_records, list):
        errors.append("data/seed/image-records.json must be an array")
        image_records = []
    image_ids: set[str] = set()
    for idx, image in enumerate(image_records):
        require_fields(errors, image, ["image_id", "source", "content_hash", "ingestion_version", "split"], f"image #{idx}")
        image_id = image.get("image_id")
        if image_id in image_ids:
            errors.append(f"duplicate image_id: {image_id}")
        if isinstance(image_id, str):
            image_ids.add(image_id)
        if image.get("split") not in VALID_SPLITS:
            errors.append(f"image {image_id} has invalid split {image.get('split')}")

    label_records = load_json(SEED / "label-records.json")
    if not isinstance(label_records, list):
        errors.append("data/seed/label-records.json must be an array")
        label_records = []
    for idx, label in enumerate(label_records):
        require_fields(errors, label, ["run_id", "image_id", "labeler_type", "label", "node_ids", "confidence", "justification", "policy_graph_version"], f"label #{idx}")
        image_id = label.get("image_id")
        if image_id not in image_ids:
            errors.append(f"label references unknown image {image_id}")
        if label.get("label") not in VALID_LABELS:
            errors.append(f"label for {image_id} has invalid label {label.get('label')}")
        if label.get("label_tier") and label.get("label_tier") not in VALID_TIERS:
            errors.append(f"label for {image_id} has invalid tier {label.get('label_tier')}")
        confidence = label.get("confidence")
        if not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1:
            errors.append(f"label for {image_id} has invalid confidence {confidence}")
        for node in label.get("node_ids", []):
            if node not in node_ids:
                errors.append(f"label references unknown node {node}")

    suggestions = load_json(SEED / "policy-suggestions.json")
    if not isinstance(suggestions, list):
        errors.append("data/seed/policy-suggestions.json must be an array")
        suggestions = []
    for idx, suggestion in enumerate(suggestions):
        require_fields(errors, suggestion, ["patch_id", "status", "suggestion_type", "target_nodes", "rationale", "proposed_diff"], f"suggestion #{idx}")
        for node in suggestion.get("target_nodes", []):
            if node not in node_ids:
                errors.append(f"suggestion references unknown node {node}")
        for diff in suggestion.get("proposed_diff", []):
            parent = diff.get("parent") if isinstance(diff, dict) else None
            if parent and parent not in node_ids:
                errors.append(f"suggestion {suggestion.get('patch_id')} references unknown parent {parent}")
            if isinstance(diff, dict) and diff.get("op") != "add_node":
                node_id = diff.get("node_id")
                if node_id and node_id not in node_ids:
                    errors.append(f"suggestion {suggestion.get('patch_id')} references unknown node_id {node_id}")

    metrics = load_json(SEED / "metrics.json")
    require_fields(errors, metrics, ["metric_snapshot_id", "policy_graph_version", "truth_label_tiers", "split", "status", "denominators", "metrics"], "metrics snapshot")
    if metrics.get("status") in {"not_enough_data", "mock_only"}:
        for name, value in metrics.get("metrics", {}).items():
            if value is not None:
                errors.append(f"mock/not_enough_data metric {name} must be null, got {value}")
    denominators = metrics.get("denominators", {})
    for field in ["truth_n", "prediction_n", "paired_truth_prediction_n", "positive_truth_n", "negative_truth_n"]:
        value = denominators.get(field)
        if not isinstance(value, int) or value < 0:
            errors.append(f"metrics denominators.{field} must be a non-negative integer")

    return len(image_ids), len(label_records)


def main() -> int:
    errors: list[str] = []
    for path in REQUIRED_FILES:
        if not path.exists():
            errors.append(f"missing required file: {path.relative_to(ROOT)}")

    validate_schema_files(errors)
    node_meta, edges = validate_graph(errors)
    image_count, label_count = validate_seed_data(errors, set(node_meta))

    if errors:
        print("RUSH foundation validation failed:", file=sys.stderr)
        for err in errors:
            print(f"- {err}", file=sys.stderr)
        return 1
    print(
        "RUSH foundation validation passed: "
        f"{len(node_meta)} policy nodes, {len(edges)} edges, "
        f"{image_count} image records, {label_count} label records, "
        f"{len(REQUIRED_SCHEMAS)} schemas."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
