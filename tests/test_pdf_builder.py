"""Smoke tests for pipeline.pdf.policy_pdf.

These tests are entirely offline and self-contained: they build PDFs from
synthetic markdown bundles in tmpdirs and against the real policy graph
under ``policy-graph/Generative_AI/v0.1``.
"""
from __future__ import annotations

from pathlib import Path
import json
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.pdf import (  # noqa: E402
    BuildResult,
    PolicyPdfError,
    build_policy_pdf,
    collect_policy_image_examples,
    iter_policy_markdown,
    parse_frontmatter,
)
from pipeline.pdf.policy_pdf import _inline_format  # noqa: E402
from scripts.build_policy_pdf import main as build_policy_pdf_cli  # noqa: E402


SAMPLE_ROOT_MD = """\
---
id: GA.root
version: Generative_AI.v0.1
title: Generative AI Image Classification
node_type: root
parent: null
polarity: mixed
status: draft
edges:
  - {type: clarifies, to: GA.boundary.low_quality_uncertain}
canonical_examples: []
---
# Generative AI Image Classification

## Decision rule
Classify an image as `gen_ai` when visible evidence supports it.

- bullet one
- bullet two with [[GA.boundary.photo_editing|conventional edits]]
- bullet three

```
code block line 1
code block line 2
```
"""

SAMPLE_LEAF_MD = """\
---
id: GA.visual_artifacts.text_symbols
version: Generative_AI.v0.1
title: Garbled text and symbols
node_type: leaf
parent: GA.root
polarity: positive
status: draft
---
# Garbled text and symbols

Look for **malformed** typography or *pseudo-logos*.

1. ordered one
2. ordered two
"""


def _write_bundle(dirpath: Path) -> None:
    (dirpath / "GA.root.md").write_text(SAMPLE_ROOT_MD, encoding="utf-8")
    (dirpath / "GA.visual_artifacts.text_symbols.md").write_text(
        SAMPLE_LEAF_MD, encoding="utf-8"
    )


def test_parse_frontmatter_extracts_top_level_scalars() -> None:
    meta, body = parse_frontmatter(SAMPLE_ROOT_MD)
    assert meta["id"] == "GA.root"
    assert meta["title"] == "Generative AI Image Classification"
    assert meta["version"] == "Generative_AI.v0.1"
    # nested list values (edges) skipped, body present
    assert "edges" in meta  # key tracked but value blank
    assert meta["edges"] == ""
    assert body.startswith("# Generative AI Image Classification")


def test_parse_frontmatter_no_header_returns_full_body() -> None:
    text = "# No frontmatter\nbody\n"
    meta, body = parse_frontmatter(text)
    assert meta == {}
    assert body == text


def test_iter_policy_markdown_orders_root_first(tmp_path: Path) -> None:
    _write_bundle(tmp_path)
    files = iter_policy_markdown(tmp_path)
    assert [p.name for p in files] == [
        "GA.root.md",
        "GA.visual_artifacts.text_symbols.md",
    ]


def test_iter_policy_markdown_rejects_missing_dir(tmp_path: Path) -> None:
    with pytest.raises(PolicyPdfError):
        iter_policy_markdown(tmp_path / "does-not-exist")


def test_iter_policy_markdown_rejects_empty_dir(tmp_path: Path) -> None:
    with pytest.raises(PolicyPdfError):
        iter_policy_markdown(tmp_path)


def test_inline_format_handles_wiki_links_bold_italic_code() -> None:
    out = _inline_format("see [[GA.root|root]] and **bold** *italic* `code`")
    assert "<b>bold</b>" in out
    assert "<i>italic</i>" in out
    assert "Courier" in out
    assert "root (GA.root)" in out


def test_inline_format_escapes_xml() -> None:
    out = _inline_format("a & b < c > d")
    assert "&amp;" in out
    assert "&lt;" in out
    assert "&gt;" in out


def test_build_policy_pdf_smoke_synthetic(tmp_path: Path) -> None:
    src = tmp_path / "v0.1"
    src.mkdir()
    _write_bundle(src)
    out = tmp_path / "out" / "policy.pdf"

    result = build_policy_pdf(src, out)

    assert isinstance(result, BuildResult)
    assert out.exists()
    assert out.stat().st_size > 1024  # non-trivial PDF
    assert result.file_count == 2
    assert result.page_count >= 2  # cover + at least one content page
    assert result.policy_graph_version == "v0.1"
    assert result.sources[0].name == "GA.root.md"
    # PDF magic number sanity check
    with out.open("rb") as fh:
        assert fh.read(4) == b"%PDF"


def test_build_policy_pdf_against_real_policy_graph(tmp_path: Path) -> None:
    real_src = ROOT / "policy-graph" / "Generative_AI" / "v0.1"
    if not real_src.exists():
        pytest.skip("real policy graph not present in this checkout")
    out = tmp_path / "policy.pdf"

    result = build_policy_pdf(real_src, out)

    assert out.exists()
    assert result.file_count >= 10  # we currently ship 13 nodes
    assert result.page_count >= result.file_count  # at least one page per doc
    assert result.policy_graph_version == "v0.1"
    assert result.sources[0].name.endswith(".root.md")


def test_build_policy_pdf_creates_parent_dir(tmp_path: Path) -> None:
    src = tmp_path / "v0.1"
    src.mkdir()
    _write_bundle(src)
    nested = tmp_path / "a" / "b" / "c" / "policy.pdf"

    result = build_policy_pdf(src, nested)

    assert nested.exists()
    assert result.output_path == nested


def test_build_policy_pdf_version_override(tmp_path: Path) -> None:
    src = tmp_path / "v0.1"
    src.mkdir()
    _write_bundle(src)
    out = tmp_path / "out.pdf"

    result = build_policy_pdf(src, out, policy_graph_version="v0.1-test")

    assert result.policy_graph_version == "v0.1-test"


def _write_tiny_manifest_with_images(repo: Path) -> Path:
    from PIL import Image

    source_dir = repo / "data" / "images" / "genai-classification" / "source-datasets" / "unit" / "ai_generated"
    source_dir.mkdir(parents=True)
    rows = []
    for idx, color in enumerate(((220, 80, 40), (20, 120, 200), (80, 190, 90)), start=1):
        image_path = source_dir / f"img_{idx}.jpg"
        Image.new("RGB", (96 + idx, 64 + idx), color).save(image_path)
        rows.append(
            {
                "sample_id": f"dev_golden_{idx:04d}",
                "repo_rel_path": str(image_path.relative_to(repo)),
                "split": "dev_golden",
                "label": "ai_generated",
                "dataset": "unit",
                "sha256": f"test-{idx}",
                "sampling_version": "unit-test",
                "node_ids": ["GA.visual_artifacts.text_symbols"],
            }
        )
    manifest = repo / "data" / "images" / "genai-classification" / "manifests" / "combined_labels.jsonl"
    manifest.parent.mkdir(parents=True)
    manifest.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    return manifest


def test_collect_policy_image_examples_uses_golden_manifest(tmp_path: Path) -> None:
    repo = tmp_path
    src = repo / "policy-graph" / "Generative_AI" / "v0.1"
    src.mkdir(parents=True)
    _write_bundle(src)
    manifest = _write_tiny_manifest_with_images(repo)

    examples = collect_policy_image_examples(manifest, repo_root=repo)

    node_examples = examples["GA.visual_artifacts.text_symbols"]
    assert [example.image_id for example in node_examples] == [
        "dev_golden_0001",
        "dev_golden_0002",
        "dev_golden_0003",
    ]
    assert all(example.media_path.exists() for example in node_examples)


def test_build_policy_pdf_with_image_examples_embeds_xobjects(tmp_path: Path) -> None:
    repo = tmp_path
    src = repo / "policy-graph" / "Generative_AI" / "v0.1"
    src.mkdir(parents=True)
    _write_bundle(src)
    manifest = _write_tiny_manifest_with_images(repo)
    out = tmp_path / "out" / "policy.pdf"

    result = build_policy_pdf(src, out, examples_root=manifest, examples_per_node=3)
    pdf_bytes = out.read_bytes()

    assert out.exists()
    assert result.file_count == 2
    assert result.page_count >= 2
    assert out.stat().st_size > 1024
    assert b"/Subtype /Image" in pdf_bytes or b"/XObject" in pdf_bytes


def test_build_policy_pdf_cli_no_examples_omits_image_xobjects(tmp_path: Path) -> None:
    repo = tmp_path
    src = repo / "policy-graph" / "Generative_AI" / "v0.1"
    src.mkdir(parents=True)
    _write_bundle(src)
    manifest = _write_tiny_manifest_with_images(repo)
    out = tmp_path / "out" / "policy-no-examples.pdf"

    rc = build_policy_pdf_cli(
        [
            "--source",
            str(src),
            "--output",
            str(out),
            "--examples-root",
            str(manifest),
            "--no-examples",
        ]
    )

    assert rc == 0
    assert out.exists()
    assert b"/Subtype /Image" not in out.read_bytes()
