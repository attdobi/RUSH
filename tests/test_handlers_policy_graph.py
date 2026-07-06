from __future__ import annotations

import json
from pathlib import Path

from pipeline.web.handlers_policy import (
    _frontmatter_edges,
    _parse_frontmatter,
    handle_policy_graph,
    handle_policy_versions,
)
from pipeline.policy_iterator import strip_leading_markers


def _write_node(path: Path, *, node_id: str, title: str, node_type: str, polarity: str, parent: str = "GA.root") -> None:
    path.write_text(
        f"""---
id: {node_id}
title: {title}
node_type: {node_type}
parent: {parent}
polarity: {polarity}
status: draft
edges:
  - {{type: subtype_of, to: GA.root}}
---
# {title}
""",
        encoding="utf-8",
    )


def test_strip_leading_markers_removes_echoed_bundle_comments() -> None:
    # Single, doubled, and no-marker cases.
    assert strip_leading_markers("<!-- GA.root.md -->\n---\nid: GA.root\n---\n") == "---\nid: GA.root\n---\n"
    assert strip_leading_markers("<!-- a.md -->\n<!-- a.md -->\n---\nid: x\n---\n") == "---\nid: x\n---\n"
    clean = "---\nid: GA.root\n---\n# body\n"
    assert strip_leading_markers(clean) == clean


def test_frontmatter_parsing_tolerates_echoed_marker() -> None:
    # A draft LLM echoing the bundle's <!-- name.md --> marker into a node must
    # not knock the node's frontmatter/edges out (regression: untyped/unparented).
    node = (
        "<!-- GA.visual_artifacts.eyes.md -->\n"
        "---\n"
        "id: GA.visual_artifacts.eyes\n"
        "node_type: category\n"
        "parent: GA.root\n"
        "edges:\n"
        "  - {type: boundary_with, to: GA.boundary.photo_editing}\n"
        "---\n"
        "# Eyes\n"
    )
    meta, _ = _parse_frontmatter(node)
    assert meta["id"] == "GA.visual_artifacts.eyes"
    assert meta["node_type"] == "category"
    assert meta["parent"] == "GA.root"
    edges = _frontmatter_edges(node, "GA.visual_artifacts.eyes")
    assert any(e.get("edge_type") == "boundary_with" for e in edges)


def test_handle_policy_graph_reads_nodes_edges_and_versions(tmp_path: Path) -> None:
    graph_dir = tmp_path / "policy-graph" / "Generative_AI"
    v01 = graph_dir / "v0.1"
    v02 = graph_dir / "v0.2"
    v01.mkdir(parents=True)
    v02.mkdir()
    _write_node(v01 / "GA.root.md", node_id="GA.root", title="Root title", node_type="root", polarity="mixed", parent="null")
    _write_node(v01 / "GA.visual_artifacts.text_symbols.md", node_id="GA.visual_artifacts.text_symbols", title="Garbled text", node_type="category", polarity="positive")
    _write_node(v01 / "GA.boundary.photo_editing.md", node_id="GA.boundary.photo_editing", title="Photo editing", node_type="boundary", polarity="negative")
    _write_node(v02 / "GA.root.md", node_id="GA.root", title="Root v2", node_type="root", polarity="mixed", parent="null")
    (v02 / "edges.json").write_text("[]\n", encoding="utf-8")
    (v01 / "edges.json").write_text(
        json.dumps([
            {
                "source_node_id": "GA.visual_artifacts.text_symbols",
                "target_node_id": "GA.root",
                "edge_type": "subtype_of",
                "confidence": 1.0,
            }
        ]),
        encoding="utf-8",
    )

    status, payload = handle_policy_graph(tmp_path, "v0.1")

    assert status == 200
    assert payload["version"] == "v0.1"
    assert payload["title"] == "Cold-start GenAI policy v0.1"
    assert payload["available_versions"] == ["v0.1", "v0.2"]
    assert [node["id"] for node in payload["nodes"]][:1] == ["GA.root"]
    root = payload["nodes"][0]
    assert root["node_type"] == "root"
    assert root["polarity"] == "mixed"
    assert root["parent"] is None
    boundary = next(node for node in payload["nodes"] if node["id"] == "GA.boundary.photo_editing")
    assert boundary["node_type"] == "boundary"
    assert boundary["status"] == "draft"
    assert payload["edges"] == [
        {
            "source_node_id": "GA.visual_artifacts.text_symbols",
            "target_node_id": "GA.root",
            "edge_type": "subtype_of",
            "confidence": 1.0,
            "source": "GA.visual_artifacts.text_symbols",
            "target": "GA.root",
        }
    ]


def test_handle_policy_graph_defaults_to_current_version(tmp_path: Path) -> None:
    graph_dir = tmp_path / "policy-graph" / "Generative_AI"
    v01 = graph_dir / "v0.1"
    v02 = graph_dir / "v0.2"
    v01.mkdir(parents=True)
    v02.mkdir()
    _write_node(v01 / "GA.root.md", node_id="GA.root", title="Root v1", node_type="root", polarity="mixed", parent="null")
    _write_node(v02 / "GA.root.md", node_id="GA.root", title="Root v2", node_type="root", polarity="mixed", parent="null")

    status, payload = handle_policy_graph(tmp_path, None)

    assert status == 200
    assert payload["version"] == "v0.2"
    assert payload["nodes"][0]["title"] == "Root v2"


def test_handle_policy_graph_rejects_unknown_version(tmp_path: Path) -> None:
    graph_dir = tmp_path / "policy-graph" / "Generative_AI" / "v0.1"
    graph_dir.mkdir(parents=True)
    _write_node(graph_dir / "GA.root.md", node_id="GA.root", title="Root", node_type="root", polarity="mixed", parent="null")

    status, payload = handle_policy_graph(tmp_path, "../../etc")

    assert status == 404
    assert "unknown policy version" in payload["error"]


def test_handle_policy_versions_filters_mnist_area(tmp_path: Path) -> None:
    genai = tmp_path / "policy-graph" / "Generative_AI"
    mnist = tmp_path / "policy-graph" / "MNIST_Digits"
    (genai / "v0.1").mkdir(parents=True)
    (genai / "v0.3").mkdir()
    (mnist / "v0.1").mkdir(parents=True)
    _write_node(genai / "v0.1" / "GA.root.md", node_id="GA.root", title="GA Root", node_type="root", polarity="mixed", parent="null")
    _write_node(genai / "v0.3" / "GA.root.md", node_id="GA.root", title="GA Root", node_type="root", polarity="mixed", parent="null")
    _write_node(mnist / "v0.1" / "MD.root.md", node_id="MD.root", title="MNIST Root", node_type="root", polarity="mixed", parent="null")

    status, payload = handle_policy_versions(tmp_path, area="MNIST_Digits")

    assert status == 200
    assert payload["domain"] == "MNIST_Digits"
    assert [item["version"] for item in payload["versions"]] == ["v0.1"]
    assert payload["current"] == "v0.1"


def test_handle_policy_graph_reads_mnist_root_and_confusion_edges(tmp_path: Path) -> None:
    graph_dir = tmp_path / "policy-graph" / "MNIST_Digits" / "v0.1"
    graph_dir.mkdir(parents=True)
    _write_node(
        graph_dir / "MD.root.md",
        node_id="MD.root",
        title="MNIST Root",
        node_type="root",
        polarity="mixed",
        parent="null",
    )
    _write_node(
        graph_dir / "MD.digit.0.md",
        node_id="MD.digit.0",
        title="Digit 0",
        node_type="digit_class",
        polarity="positive",
        parent="MD.root",
    )
    (graph_dir / "MD.digit.0.md").write_text(
        """---
id: MD.digit.0
title: Digit 0
node_type: digit_class
parent: MD.root
polarity: positive
status: draft
edges:
  - {type: subtype_of, to: MD.root}
  - {type: confused_with, to: MD.digit.6}
---
# Digit 0
""",
        encoding="utf-8",
    )
    (graph_dir / "MD.digit.6.md").write_text(
        """---
id: MD.digit.6
title: Digit 6
node_type: digit_class
parent: MD.root
polarity: positive
status: draft
edges:
  - {type: subtype_of, to: MD.root}
  - {type: confused_with, to: MD.digit.0}
---
# Digit 6
""",
        encoding="utf-8",
    )
    (graph_dir / "edges.json").write_text(
        json.dumps(
            [
                {
                    "source_node_id": "MD.digit.0",
                    "target_node_id": "MD.root",
                    "edge_type": "subtype_of",
                    "confidence": 1.0,
                }
            ]
        ),
        encoding="utf-8",
    )

    status, payload = handle_policy_graph(tmp_path, "v0.1", area="MNIST_Digits")

    assert status == 200
    assert payload["area"] == "MNIST_Digits"
    assert payload["title"] == "Cold-start MNIST digit policy v0.1"
    assert payload["nodes"][0]["id"] == "MD.root"
    assert "GA.root" not in {node["id"] for node in payload["nodes"]}
    assert any(edge["edge_type"] == "confused_with" for edge in payload["edges"])
