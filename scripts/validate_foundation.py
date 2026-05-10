#!/usr/bin/env python3
"""Validate the RUSH foundation scaffold without external dependencies.

This validator intentionally stays stdlib-only. It performs lightweight contract
checks for schema presence/shape, policy graph frontmatter and edge invariants,
seed-data cross-references, split-leakage risks, and placeholder metric hygiene.
"""
from __future__ import annotations

from collections import defaultdict
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
    ROOT / "web" / "genai-sampler.js",
    ROOT / "web" / "app.js",
    ROOT / "docs" / "visuals" / "rush-system.svg",
    ROOT / "docs" / "visuals" / "policy-evolution.svg",
]

REQUIRED_SCHEMAS = [
    "arbiter-decision.schema.json",
    "consensus-record.schema.json",
    "decision-quality.schema.json",
    "export-record.schema.json",
    "image-record.schema.json",
    "label-hierarchy.schema.json",
    "label-tier-record.schema.json",
    "label-vote.schema.json",
    "llm-output.schema.json",
    "metric-snapshot.schema.json",
    "policy-edge.schema.json",
    "policy-node.schema.json",
    "policy-patch.schema.json",
    "sme-rereview-sample.schema.json",
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
HOLDOUT_SPLITS = {"locked_holdout", "boundary_holdout"}
NON_HOLDOUT_SPLITS = {"development", "validation"}
VALID_LABELS = {"gen_ai", "not_gen_ai", "abstain", "positive", "negative"}
VALID_TIERS = {"provisional", "silver", "gold", "platinum", "deprecated", "superseded"}
VALID_EDGE_TYPES = {
    "subtype_of",
    "exception_to",
    "boundary_with",
    "confused_with",
    "clarifies",
    "example_of",
    "negative_example_of",
}
DECISION_QUALITY_METRICS = [
    "accuracy",
    "f1",
    "precision",
    "recall",
    "fpr",
    "fnr",
    "positive_proportion",
    "n",
    "informedness",
]


def parse_scalar(value: str) -> Any:
    value = value.strip()
    if value in {"", "null", "None", "~"}:
        return None if value != "" else ""
    if value in {"[]", "[ ]"}:
        return []
    if value in {"{}", "{ }"}:
        return {}
    if value in {"true", "false"}:
        return value == "true"
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        return value[1:-1]
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value


def parse_frontmatter(path: Path) -> tuple[dict[str, Any], list[str], str]:
    text = path.read_text(encoding="utf-8")
    if not text.startswith("---\n"):
        raise ValueError(f"{path.relative_to(ROOT)} is missing YAML frontmatter")
    parts = text.split("---", 2)
    if len(parts) < 3:
        raise ValueError(f"{path.relative_to(ROOT)} has unterminated YAML frontmatter")
    block = parts[1]
    body = parts[2].strip()
    data: dict[str, Any] = {}
    active_key: str | None = None

    for raw in block.splitlines():
        if not raw.strip():
            continue
        if raw.startswith("  -"):
            if active_key is None:
                continue
            if not isinstance(data.get(active_key), list):
                data[active_key] = []
            item = raw.split("-", 1)[1].strip()
            data[active_key].append(item)
            continue
        if raw.startswith("  "):
            if active_key is None:
                continue
            if not isinstance(data.get(active_key), dict):
                data[active_key] = {}
            if ":" in raw:
                key, value = raw.strip().split(":", 1)
                data[active_key][key.strip()] = parse_scalar(value)
            continue
        if ":" in raw:
            key, value = raw.split(":", 1)
            active_key = key.strip()
            data[active_key] = parse_scalar(value)

    edge_refs = re.findall(r"\bto:\s*([^,}\s]+)", block)
    return data, edge_refs, body


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def require_fields(errors: list[str], obj: dict[str, Any], fields: list[str], context: str) -> None:
    for field in fields:
        if field not in obj:
            errors.append(f"{context} missing required field {field}")


def enum_values(schema: dict[str, Any], field: str) -> set[Any]:
    props = schema.get("properties", {})
    values = props.get(field, {}).get("enum", [])
    return set(values) if isinstance(values, list) else set()


def validate_schema_files(errors: list[str]) -> list[str]:
    schema_names = sorted(path.name for path in SCHEMAS.glob("*.schema.json"))
    for name in REQUIRED_SCHEMAS:
        if name not in schema_names:
            errors.append(f"missing required schema: schemas/{name}")

    for name in schema_names:
        path = SCHEMAS / name
        try:
            schema = load_json(path)
        except Exception as exc:  # noqa: BLE001 - validation report
            errors.append(f"schema does not parse as JSON: schemas/{name}: {exc}")
            continue
        if schema.get("$schema") != "https://json-schema.org/draft/2020-12/schema":
            errors.append(f"schemas/{name} must use JSON Schema draft 2020-12")
        if schema.get("type") != "object":
            errors.append(f"schemas/{name} must declare type=object")
        if not isinstance(schema.get("title"), str) or not schema["title"]:
            errors.append(f"schemas/{name} missing title")
        if not isinstance(schema.get("properties"), dict) or not schema["properties"]:
            errors.append(f"schemas/{name} missing properties object")
        if name in REQUIRED_SCHEMAS and (not isinstance(schema.get("required"), list) or not schema["required"]):
            errors.append(f"schemas/{name} missing required field list")

    return schema_names


def validate_schema_expectations(errors: list[str]) -> None:
    metric_schema = load_json(SCHEMAS / "metric-snapshot.schema.json")
    if "denominators" not in metric_schema.get("required", []):
        errors.append("metric-snapshot schema must require denominators")
    metric_props = metric_schema.get("properties", {}).get("metrics", {}).get("properties", {})
    for field in ["macro_f1", "calibration_ece", "graph_location_accuracy"]:
        if field not in metric_props:
            errors.append(f"metric-snapshot schema metrics missing {field}")
    graph_health_props = metric_schema.get("properties", {}).get("graph_health", {}).get("properties", {})
    for field in ["consensus_audit_error_rate", "sme_override_rate", "per_node_difficulty", "per_node_coverage"]:
        if field not in graph_health_props:
            errors.append(f"metric-snapshot schema graph_health missing {field}")
    ci_props = metric_schema.get("properties", {}).get("confidence_intervals", {}).get("properties", {})
    if set(ci_props.get("method", {}).get("enum", [])) != {"wilson", "clopper_pearson", "bootstrap", "none"}:
        errors.append("metric-snapshot schema confidence_intervals.method enum is incomplete")
    if not metric_schema.get("properties", {}).get("split", {}).get("enum"):
        errors.append("metric-snapshot schema split must be constrained to an enum")

    label_vote_props = load_json(SCHEMAS / "label-vote.schema.json").get("properties", {})
    for field in ["l0_label", "l2_label", "is_boundary", "difficulty"]:
        if field not in label_vote_props:
            errors.append(f"label-vote schema missing {field}")

    decision_quality = load_json(SCHEMAS / "decision-quality.schema.json")
    for field in ["policy_graph_version", "ground_truth_tier", "labelers"]:
        if field not in decision_quality.get("required", []):
            errors.append(f"decision-quality schema must require {field}")
    labeler_props = decision_quality.get("properties", {}).get("labelers", {}).get("items", {}).get("properties", {})
    if set(labeler_props.get("labeler_type", {}).get("enum", [])) != {"llm", "ensemble", "human"}:
        errors.append("decision-quality schema labeler_type enum must be llm/ensemble/human")
    dq_metrics = decision_quality.get("$defs", {}).get("decision_quality_metrics", {})
    for field in DECISION_QUALITY_METRICS:
        if field not in dq_metrics.get("required", []):
            errors.append(f"decision-quality schema metrics must require {field}")


def validate_graph(errors: list[str]) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    policy_node_schema = load_json(SCHEMAS / "policy-node.schema.json")
    required_node_keys = policy_node_schema.get("required", [])
    node_type_enum = enum_values(policy_node_schema, "node_type")
    polarity_enum = enum_values(policy_node_schema, "polarity")
    status_enum = enum_values(policy_node_schema, "status")

    node_meta: dict[str, dict[str, Any]] = {}
    node_bodies: dict[str, str] = {}
    frontmatter_edge_refs: dict[str, list[str]] = {}

    for md in sorted(GRAPH.glob("*.md")):
        try:
            fm, edge_refs, body = parse_frontmatter(md)
        except Exception as exc:  # noqa: BLE001 - validation report
            errors.append(str(exc))
            continue
        for key in required_node_keys:
            if key not in fm or (key != "coverage_target" and fm[key] == ""):
                errors.append(f"{md.relative_to(ROOT)} missing frontmatter key {key}")
        if fm.get("node_type") not in node_type_enum:
            errors.append(f"{md.relative_to(ROOT)} has invalid node_type {fm.get('node_type')!r}")
        if fm.get("polarity") not in polarity_enum:
            errors.append(f"{md.relative_to(ROOT)} has invalid polarity {fm.get('polarity')!r}")
        if "status" in fm and fm.get("status") not in status_enum:
            errors.append(f"{md.relative_to(ROOT)} has invalid status {fm.get('status')!r}")
        if not isinstance(fm.get("coverage_target"), dict):
            errors.append(f"{md.relative_to(ROOT)} coverage_target must be an object")

        node_id = fm.get("id")
        if not isinstance(node_id, str) or not node_id:
            continue
        if node_id in node_meta:
            errors.append(f"duplicate node id: {node_id}")
        node_meta[node_id] = {**fm, "path": str(md.relative_to(ROOT))}
        node_bodies[node_id] = body
        frontmatter_edge_refs[node_id] = edge_refs

    roots_by_area: dict[str, list[str]] = defaultdict(list)
    for node_id, fm in node_meta.items():
        if fm.get("node_type") == "root":
            roots_by_area[str(fm.get("area"))].append(node_id)
    for area, root_ids in sorted(roots_by_area.items()):
        if len(root_ids) != 1:
            errors.append(f"area {area} must have exactly one root node; found {root_ids}")
    areas_with_nodes = {str(fm.get("area")) for fm in node_meta.values()}
    for area in sorted(areas_with_nodes - set(roots_by_area)):
        errors.append(f"area {area} must have exactly one root node; found none")
    if "GA.root" in node_meta and node_meta["GA.root"].get("parent") is not None:
        errors.append("GA.root must have parent: null")

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
    subtype_sources: set[str] = set()
    subtype_targets: set[str] = set()
    exception_sources: set[str] = set()
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
        if edge_type not in VALID_EDGE_TYPES:
            errors.append(f"edge #{idx} has invalid edge_type {edge_type!r}")
        if edge_type == "subtype_of":
            if isinstance(source, str):
                subtype_sources.add(source)
            if isinstance(target, str):
                subtype_targets.add(target)
        if edge_type == "exception_to" and isinstance(source, str):
            exception_sources.add(source)
        key = (str(source), str(target), str(edge_type))
        if key in seen_edges:
            errors.append(f"duplicate edge: {key}")
        seen_edges.add(key)

    root_nodes = {node_id for node_id, fm in node_meta.items() if fm.get("node_type") == "root"}
    for node_id, fm in node_meta.items():
        if node_id in root_nodes:
            continue
        if node_id in subtype_sources:
            continue
        if fm.get("node_type") == "exception" and node_id in exception_sources:
            continue
        errors.append(f"orphan node: {node_id} is not the source of any subtype_of edge")

    for node_id, fm in node_meta.items():
        if node_id in root_nodes:
            continue
        parent = fm.get("parent")
        if not isinstance(parent, str) or parent not in node_meta:
            errors.append(f"{node_id} has missing/unknown parent {parent!r}")
            continue
        visited: set[str] = set()
        current = node_id
        while current not in root_nodes:
            if current in visited:
                errors.append(f"parent cycle detected at {node_id}")
                break
            visited.add(current)
            parent = node_meta.get(current, {}).get("parent")
            if parent not in node_meta:
                errors.append(f"orphan node not reachable to a root: {node_id}")
                break
            current = parent

    leaf_nodes = set(node_meta) - subtype_targets
    decision_section = re.compile(r"^##\s+(decision rule|boundary|positive criteria|negative rule|hard negatives?)\b", re.IGNORECASE | re.MULTILINE)
    for node_id in sorted(leaf_nodes):
        if not node_bodies.get(node_id, "").strip():
            errors.append(f"leaf node {node_id} must have non-empty body text")
        elif not decision_section.search(node_bodies[node_id]):
            errors.append(f"leaf node {node_id} body must contain a decision rule or boundary section")

    return node_meta, edges


def validate_static_web(errors: list[str]) -> None:
    index_path = ROOT / "web" / "index.html"
    try:
        index_html = index_path.read_text(encoding="utf-8")
    except Exception as exc:  # noqa: BLE001 - validation report
        errors.append(f"web/index.html does not parse as text: {exc}")
        return

    scripts = re.findall(r'<script\s+[^>]*src="([^"]+)"', index_html)
    for script in ["genai-sampler.js", "app.js"]:
        if script not in scripts:
            errors.append(f"web/index.html must load {script}")
    if "genai-sampler.js" in scripts and "app.js" in scripts:
        if scripts.index("genai-sampler.js") > scripts.index("app.js"):
            errors.append("web/index.html must load genai-sampler.js before app.js")
    for script in scripts:
        if script.startswith(("http://", "https://", "//")):
            continue
        script_path = (ROOT / "web" / script.split("?", 1)[0]).resolve()
        if ROOT.resolve() not in script_path.parents and script_path != ROOT.resolve():
            errors.append(f"web script escapes repo root: {script}")
        elif not script_path.exists():
            errors.append(f"web script missing: web/{script}")

    sampler_text = (ROOT / "web" / "genai-sampler.js").read_text(encoding="utf-8")
    if "RushGenaiSampler" not in sampler_text or "runDemoReset" not in sampler_text:
        errors.append("web/genai-sampler.js missing expected RushGenaiSampler.runDemoReset contract")
    if "applyOverride" not in sampler_text and "applyHumanOverrides" not in sampler_text:
        errors.append("web/genai-sampler.js should expose an override helper")
    app_text = (ROOT / "web" / "app.js").read_text(encoding="utf-8")
    for token in ["runSamplerDemo", "LLM labeling comes next", "SME/human override"]:
        if token not in app_text:
            errors.append(f"web/app.js missing sampler demo UI token {token}")


def validate_decision_quality(errors: list[str]) -> None:
    path = SEED / "decision-quality.json"
    if not path.exists():
        errors.append("missing seed data: data/seed/decision-quality.json")
        return
    data = load_json(path)
    require_fields(errors, data, ["policy_graph_version", "ground_truth_tier", "labelers"], "decision-quality snapshot")
    labelers = data.get("labelers", [])
    if not isinstance(labelers, list):
        errors.append("decision-quality labelers must be an array")
        return
    for idx, labeler in enumerate(labelers):
        if not isinstance(labeler, dict):
            errors.append(f"decision-quality labeler #{idx} is not an object")
            continue
        require_fields(errors, labeler, ["labeler_id", "labeler_type", "metrics"], f"decision-quality labeler #{idx}")
        if labeler.get("labeler_type") not in {"llm", "ensemble", "human"}:
            errors.append(f"decision-quality labeler {labeler.get('labeler_id')} has invalid type {labeler.get('labeler_type')}")
        metrics = labeler.get("metrics", {})
        if not isinstance(metrics, dict):
            errors.append(f"decision-quality labeler {labeler.get('labeler_id')} metrics must be an object")
            continue
        require_fields(errors, metrics, DECISION_QUALITY_METRICS, f"decision-quality labeler {labeler.get('labeler_id')} metrics")
        for name, value in metrics.items():
            if name == "n":
                if not isinstance(value, int) or value < 0:
                    errors.append(f"decision-quality labeler {labeler.get('labeler_id')} metrics.n must be a non-negative integer")
            elif value is not None and not isinstance(value, (int, float)):
                errors.append(f"decision-quality labeler {labeler.get('labeler_id')} metric {name} must be number or null")
    evolution = data.get("evolution", [])
    if evolution is not None and not isinstance(evolution, list):
        errors.append("decision-quality evolution must be an array")


def validate_seed_data(errors: list[str], node_ids: set[str]) -> tuple[int, int]:
    image_records = load_json(SEED / "image-records.json")
    if not isinstance(image_records, list):
        errors.append("data/seed/image-records.json must be an array")
        image_records = []
    image_ids: set[str] = set()
    splits_by_dedupe_cluster: dict[str, set[str]] = defaultdict(set)
    for idx, image in enumerate(image_records):
        require_fields(errors, image, ["image_id", "source", "content_hash", "ingestion_version", "split"], f"image #{idx}")
        image_id = image.get("image_id")
        if image_id in image_ids:
            errors.append(f"duplicate image_id: {image_id}")
        if isinstance(image_id, str):
            image_ids.add(image_id)
        split = image.get("split")
        if split not in VALID_SPLITS:
            errors.append(f"image {image_id} has invalid split {split}")
        cluster_id = image.get("dedupe_cluster_id")
        if isinstance(cluster_id, str) and cluster_id:
            splits_by_dedupe_cluster[cluster_id].add(str(split))

    for cluster_id, splits in sorted(splits_by_dedupe_cluster.items()):
        if splits & HOLDOUT_SPLITS and splits & NON_HOLDOUT_SPLITS:
            errors.append(
                f"split leakage risk: dedupe_cluster_id {cluster_id} spans holdout and non-holdout splits {sorted(splits)}"
            )

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
    if metrics.get("split") not in VALID_SPLITS:
        errors.append(f"metrics snapshot has invalid split {metrics.get('split')}")
    if metrics.get("status") in {"not_enough_data", "mock_only"} or metrics.get("mock_only") is True:
        for name, value in metrics.get("metrics", {}).items():
            if value is not None:
                errors.append(f"mock/not_enough_data metric {name} must be null, got {value}")
    metric_values = metrics.get("metrics", {})
    for field in ["accuracy", "precision", "recall", "fpr", "positive_proportion", "informedness", "macro_f1", "calibration_ece", "graph_location_accuracy"]:
        if field not in metric_values:
            errors.append(f"metrics snapshot metrics missing {field}")
    denominators = metrics.get("denominators", {})
    if not isinstance(denominators, dict):
        errors.append("metrics denominators must be an object")
    else:
        for field in ["n_total", "n_positive", "n_negative", "n_abstain"]:
            value = denominators.get(field)
            if not isinstance(value, int) or value < 0:
                errors.append(f"metrics denominators.{field} must be a non-negative integer")
    confidence_intervals = metrics.get("confidence_intervals", {})
    if confidence_intervals.get("method") != "none":
        errors.append("placeholder metrics confidence_intervals.method must be none")
    graph_health = metrics.get("graph_health", {})
    for field in ["node_count", "edge_count", "coverage", "gray_zone_mass", "orphan_nodes", "consensus_audit_error_rate", "sme_override_rate", "per_node_difficulty", "per_node_coverage"]:
        if field not in graph_health:
            errors.append(f"metrics graph_health missing {field}")

    validate_decision_quality(errors)
    return len(image_ids), len(label_records)


def main() -> int:
    errors: list[str] = []
    for path in REQUIRED_FILES:
        if not path.exists():
            errors.append(f"missing required file: {path.relative_to(ROOT)}")

    schema_names = validate_schema_files(errors)
    if not errors:
        validate_schema_expectations(errors)
    validate_static_web(errors)
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
        f"{len(schema_names)} schemas found."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
