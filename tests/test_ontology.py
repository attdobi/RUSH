from __future__ import annotations

import json
import re
from pathlib import Path

from pipeline.policy_graph.boundary_promotion import enumerate_boundary_promotions
from pipeline.providers.base import coerce_label_fields
from pipeline.providers.ontology import get_ontology
from pipeline.scoring.decision_quality_multiclass import (
    compute_decision_quality_multiclass,
)
from pipeline.scoring.tasks import MNIST_MULTICLASS


ROOT = Path(__file__).resolve().parents[1]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n")


def _mnist_frontmatter_edges() -> list[dict]:
    edges: list[dict] = []
    graph_dir = ROOT / "policy-graph" / "MNIST_Digits" / "v0.1"
    for node_path in sorted(graph_dir.glob("MD.digit.*.md")):
        text = node_path.read_text(encoding="utf-8")
        node_match = re.search(r"^id:\s*(MD\.digit\.\d+)\s*$", text, re.MULTILINE)
        if not node_match:
            continue
        source = node_match.group(1)
        for edge_type, target in re.findall(
            r"\{\s*type:\s*([^,}]+),\s*to:\s*([^}]+)\s*\}",
            text,
        ):
            edges.append(
                {
                    "source_node_id": source,
                    "type": edge_type.strip(),
                    "to": target.strip(),
                }
            )
    return edges


def test_get_ontology_genai_and_mnist_contracts():
    genai = get_ontology("Generative_AI")
    assert genai.area == "Generative_AI"
    assert genai.l1_classes == ("gen_ai", "not_gen_ai")
    assert genai.label_enum == (
        "gen_ai",
        "not_gen_ai",
        "violative",
        "non_violative",
    )
    assert genai.scoring_task == "genai_binary"
    assert genai.require_boundary_between is False

    mnist = get_ontology("MNIST_Digits")
    assert mnist.area == "MNIST_Digits"
    assert mnist.l1_classes == tuple(str(d) for d in range(10))
    assert mnist.label_enum == tuple(str(d) for d in range(10))
    assert mnist.scoring_task == "mnist_multiclass"
    assert mnist.require_boundary_between is True


def test_response_schema_label_enums_offer_no_abstain():
    # No-abstain contract (Attila 2026-07-09): the provider-facing schema
    # must never OFFER abstain as a choice. "abstain" survives only as the
    # internal parse/transport-failure sentinel (Ontology.abstain_label).
    mnist = get_ontology("MNIST_Digits")
    assert mnist.response_schema["properties"]["label"]["enum"] == [
        str(d) for d in range(10)
    ]
    genai = get_ontology("Generative_AI")
    assert "abstain" not in genai.response_schema["properties"]["label"]["enum"]
    assert mnist.abstain_label == "abstain"
    assert genai.abstain_label == "abstain"


def test_coerce_label_fields_boundary_between_mnist_validation():
    mnist = get_ontology("MNIST_Digits")

    valid = coerce_label_fields(
        {"is_boundary": True, "is_boundary_between": ["1", "7"]},
        ontology=mnist,
    )
    assert valid["is_boundary_between"] == ["1", "7"]
    assert valid["boundary_between_invalid"] is False

    invalid = coerce_label_fields(
        {"is_boundary": True, "is_boundary_between": ["1"]},
        ontology=mnist,
    )
    assert invalid["is_boundary_between"] == ["1"]
    assert invalid["boundary_between_invalid"] is True

    not_boundary = coerce_label_fields(
        {"is_boundary": False, "is_boundary_between": ["1", "7"]},
        ontology=mnist,
    )
    assert not_boundary["is_boundary_between"] == []
    assert not_boundary["boundary_between_invalid"] is False


def test_coerce_label_fields_genai_default_does_not_require_boundary_pair():
    out = coerce_label_fields(
        {"is_boundary": True, "is_boundary_between": ["GA.boundary.cgi"]}
    )
    assert out["is_boundary_between"] == ["GA.boundary.cgi"]
    assert out["boundary_between_invalid"] is False


def test_multiclass_scoring_falls_back_from_l2_digit_label(tmp_path: Path):
    manifest = tmp_path / "manifest.jsonl"
    votes = tmp_path / "label_votes.jsonl"
    _write_jsonl(
        manifest,
        [
            {
                "sample_id": "img_7",
                "label": "7",
                "label_int": 7,
                "truth_tier": "gold",
                "split": "dev_golden",
                "repo_rel_path": "data/images/mnist/7.png",
            }
        ],
    )
    _write_jsonl(
        votes,
        [
            {
                "run_id": "r1",
                "image_id": "img_7",
                "labeler_id": "legacy",
                "model_id": "demo/legacy",
                "label": "legacy_l1_missing",
                "l2_label": "MD.digit.7",
            }
        ],
    )

    snapshot = compute_decision_quality_multiclass(
        votes,
        manifest,
        task=MNIST_MULTICLASS,
        policy_graph_version="MNIST_Digits.v0.1",
        ground_truth_tier=("gold",),
    )
    metrics = snapshot["labelers"][0]["metrics"]
    assert metrics["accuracy"] == 1.0
    assert metrics["confusion_matrix"]["7"]["7"] == 1


def test_policy_node_schema_accepts_boundary_and_digit_class():
    schema_path = ROOT / "schemas" / "policy-node.schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    enum = schema["properties"]["node_type"]["enum"]
    assert "boundary" in enum
    assert "digit_class" in enum

    base = {
        "id": "MD.boundary.1x7",
        "version": "MNIST_Digits.v0.1",
        "title": "Digit 1 vs 7 Boundary",
        "area": "MNIST_Digits",
        "node_type": "boundary",
        "polarity": "mixed",
        "coverage_target": {},
    }
    try:
        import jsonschema  # type: ignore
    except Exception:
        return

    validator = jsonschema.Draft202012Validator(schema)
    validator.validate(base)
    validator.validate({**base, "id": "MD.digit.1", "node_type": "digit_class"})


def test_enumerate_boundary_promotions_from_mnist_confusion_edges():
    promotions = enumerate_boundary_promotions(_mnist_frontmatter_edges())
    by_id = {item["node_id"]: item for item in promotions}

    assert by_id["MD.boundary.1x7"] == {
        "node_id": "MD.boundary.1x7",
        "digits": [1, 7],
        "boundary_of": ["MD.digit.1", "MD.digit.7"],
        "node_type": "boundary",
    }
    assert "MD.boundary.3x5" in by_id
    assert "MD.boundary.4x9" in by_id


def test_enumerate_boundary_promotions_accepts_edges_json_shape_and_dedupes():
    promotions = enumerate_boundary_promotions(
        [
            {
                "source_node_id": "MD.digit.7",
                "target_node_id": "MD.digit.1",
                "edge_type": "confused_with",
            },
            {
                "source_node_id": "MD.digit.1",
                "target_node_id": "MD.digit.7",
                "edge_type": "confused_with",
            },
            {
                "source_node_id": "MD.digit.1",
                "target_node_id": "MD.root",
                "edge_type": "subtype_of",
            },
        ]
    )
    assert promotions == [
        {
            "node_id": "MD.boundary.1x7",
            "digits": [1, 7],
            "boundary_of": ["MD.digit.1", "MD.digit.7"],
            "node_type": "boundary",
        }
    ]

