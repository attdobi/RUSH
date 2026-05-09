#!/usr/bin/env python3
"""Validate the RUSH foundation scaffold without external dependencies."""
from __future__ import annotations
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
GRAPH = ROOT / "policy-graph" / "Generative_AI" / "v0.1"
REQUIRED = [
    ROOT / "README.md",
    ROOT / "web" / "index.html",
    ROOT / "web" / "styles.css",
    ROOT / "web" / "app.js",
    ROOT / "docs" / "visuals" / "rush-system.svg",
    ROOT / "docs" / "visuals" / "policy-evolution.svg",
]


def parse_frontmatter(path: Path) -> dict[str, str]:
    text = path.read_text(encoding="utf-8")
    if not text.startswith("---\n"):
        raise ValueError(f"{path} is missing YAML frontmatter")
    block = text.split("---", 2)[1]
    data: dict[str, str] = {}
    for raw in block.splitlines():
        if not raw.strip() or raw.startswith(" ") or raw.startswith("  -"):
            continue
        if ":" in raw:
            key, value = raw.split(":", 1)
            data[key.strip()] = value.strip().strip('"')
    return data


def load_json(path: Path):
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def main() -> int:
    errors: list[str] = []
    for path in REQUIRED:
        if not path.exists():
            errors.append(f"missing required file: {path.relative_to(ROOT)}")

    node_ids: set[str] = set()
    for md in sorted(GRAPH.glob("*.md")):
        try:
            fm = parse_frontmatter(md)
        except Exception as exc:  # noqa: BLE001 - validation report
            errors.append(str(exc))
            continue
        for key in ["id", "version", "title", "area", "node_type", "polarity"]:
            if key not in fm or not fm[key]:
                errors.append(f"{md.relative_to(ROOT)} missing frontmatter key {key}")
        node_id = fm.get("id")
        if node_id in node_ids:
            errors.append(f"duplicate node id: {node_id}")
        if node_id:
            node_ids.add(node_id)

    edges = load_json(GRAPH / "edges.json")
    for edge in edges:
        for field in ["source_node_id", "target_node_id"]:
            ref = edge[field]
            if ref != "ROOT" and ref not in node_ids:
                errors.append(f"edge references unknown node {ref}: {edge}")

    image_records = load_json(ROOT / "data" / "seed" / "image-records.json")
    image_ids = {item["image_id"] for item in image_records}
    label_records = load_json(ROOT / "data" / "seed" / "label-records.json")
    for label in label_records:
        if label["image_id"] not in image_ids:
            errors.append(f"label references unknown image {label['image_id']}")
        for node in label.get("node_ids", []):
            if node not in node_ids:
                errors.append(f"label references unknown node {node}")
    suggestions = load_json(ROOT / "data" / "seed" / "policy-suggestions.json")
    for suggestion in suggestions:
        for node in suggestion.get("target_nodes", []):
            if node not in node_ids:
                errors.append(f"suggestion references unknown node {node}")

    for schema in (ROOT / "schemas").glob("*.json"):
        load_json(schema)
    load_json(ROOT / "data" / "seed" / "metrics.json")

    if errors:
        print("RUSH foundation validation failed:", file=sys.stderr)
        for err in errors:
            print(f"- {err}", file=sys.stderr)
        return 1
    print(f"RUSH foundation validation passed: {len(node_ids)} policy nodes, {len(edges)} edges, {len(image_ids)} image records, {len(label_records)} label records.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
