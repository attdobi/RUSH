"""Deterministic structural digest ("compressed policy") + per-judge render.

The policy-rendering × judge-scale axis (Attila 2026-07-09, after the qwen
probe: 0/6 under the full ~25k-char bundle, 8/8 under a two-line prompt).
The digest must be a PROJECTION — boilerplate sections dropped whole, every
decision rule / node id / edge kept byte-for-byte — and per-judge selection
must pin which render each judge's requests carry.
"""
from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from pipeline.manifest import SampleRecord
from pipeline.policy_iterator import load_policy_markdown
from pipeline.policy_render import (
    compress_policy_markdown,
    parse_compressed_models,
)
from pipeline.runner import DeterministicFakeClient, run_labeling

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("area", ["Generative_AI", "MNIST_Digits"])
def test_digest_is_a_deterministic_projection(area: str) -> None:
    bundle = load_policy_markdown(REPO_ROOT / "policy-graph" / area / "v0.1")
    digest = compress_policy_markdown(bundle)
    # Deterministic: same bytes in -> same bytes out.
    assert digest == compress_policy_markdown(bundle)
    # Meaningfully smaller, but never empty.
    assert 0 < len(digest) < len(bundle) * 0.85
    # Boilerplate sections dropped whole — matched on the EXACT heading, so
    # hybrid headings that carry decision content (e.g. MNIST's
    # "## Hard negatives / confusions") survive: safety over size.
    from pipeline.policy_render import BODY_DROP_SECTIONS
    digest_headings = {
        line[3:].strip().lower()
        for line in digest.splitlines() if line.startswith("## ")
    }
    assert not digest_headings & BODY_DROP_SECTIONS
    # Judge-facing content survives: node ids, graph shape, decision rules.
    assert "id: " in digest
    assert "edges:" in digest
    # Curation-only frontmatter dropped.
    assert "coverage_target" not in digest
    assert "source_anchors" not in digest
    # Every kept line is verbatim from the source (projection, not paraphrase).
    bundle_lines = set(bundle.splitlines())
    kept_body = [ln for ln in digest.splitlines()
                 if ln and not ln.startswith("<!--") and ln != "---"]
    assert all(ln in bundle_lines for ln in kept_body)


def test_digest_keeps_unknown_sections() -> None:
    # Drafter-minted sections with novel headings must survive: safety over
    # size — only the known-boilerplate list is dropped.
    node = (
        "<!-- GA.test.md -->\n---\nid: GA.test\ntitle: T\nstatus: draft\n---\n"
        "# T\n\n## Decision rule\nKeep me.\n\n## Novel drafter section\n"
        "Also keep me.\n\n## Why this node exists\nDrop me.\n"
    )
    digest = compress_policy_markdown(node)
    assert "Keep me." in digest
    assert "## Novel drafter section" in digest
    assert "Also keep me." in digest
    assert "Drop me." not in digest
    assert "status: draft" not in digest  # curation frontmatter dropped
    assert "id: GA.test" in digest


def test_parse_compressed_models() -> None:
    assert parse_compressed_models(None) == frozenset()
    assert parse_compressed_models("") == frozenset()
    assert parse_compressed_models("a, b ,a") == frozenset({"a", "b"})


_SAMPLES = [
    SampleRecord(
        sample_id=f"dev_golden_{idx:04d}",
        repo_rel_path=f"data/images/test/dev_golden_{idx:04d}.jpg",
        split="dev_golden",
        sme_label_raw="ai_generated" if idx % 2 else "not_ai_generated",
        sme_label="gen_ai" if idx % 2 else "not_gen_ai",
        dataset="test",
        sha256=f"{idx:064x}"[-64:],
        sampling_version="test-sampling-v1",
    )
    for idx in range(1, 4)
]


class _PolicyRecordingClient(DeterministicFakeClient):
    """Records the policy_markdown each request carried."""

    def __init__(self, model_id: str, seen: dict[str, set[int]]) -> None:
        super().__init__(model_id)
        self._seen = seen

    def label(self, request):  # type: ignore[override]
        self._seen.setdefault(request.model_id, set()).add(
            len(request.policy_markdown)
        )
        return super().label(request)


def test_run_labeling_per_judge_render_and_manifest_record() -> None:
    seen: dict[str, set[int]] = {}

    def factory(spec):
        return _PolicyRecordingClient(spec.model_id, seen)

    with TemporaryDirectory() as tmp:
        summary = run_labeling(
            models=["openai/gpt-5.5", "google/gemini-3.1-flash-lite"],
            split="dev_golden",
            samples=_SAMPLES,
            runs_root=Path(tmp),
            client_factory=factory,
            batch_size=1,
            compressed_models={"google/gemini-3.1-flash-lite"},
        )
        assert summary.completed_calls == 6
        full_sizes = seen["openai/gpt-5.5"]
        digest_sizes = seen["google/gemini-3.1-flash-lite"]
        assert len(full_sizes) == 1 and len(digest_sizes) == 1
        assert max(digest_sizes) < max(full_sizes)  # digest is smaller

        manifest = json.loads(
            (Path(tmp) / summary.run_id / "run_manifest.json").read_text()
        )
        assert manifest["compressed_policy_models"] == [
            "google/gemini-3.1-flash-lite"
        ]
        chars = manifest["policy_render_chars"]
        assert chars["compressed"] == max(digest_sizes)
        assert chars["full"] == max(full_sizes)


def test_run_labeling_rejects_unknown_compressed_model() -> None:
    with TemporaryDirectory() as tmp:
        with pytest.raises(ValueError, match="compressed_models"):
            run_labeling(
                models=["openai/gpt-5.5"],
                split="dev_golden",
                samples=_SAMPLES,
                runs_root=Path(tmp),
                compressed_models={"local/qwen2.5-vl-7b"},
            )
