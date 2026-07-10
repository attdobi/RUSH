"""Deterministic structural digest of a policy bundle ("compressed policy").

Why this exists (measured, 2026-07-09): the policy bundle is prepended to
EVERY judge call — it is the judge's whole context. Under the full ~25k-char
GenAI v0.1 bundle, qwen2.5-vl-7b collapsed to the policy's default branch on
110/110 live calls (0/6 generated images detected); under a two-line prompt it
went 8/8 on the same images. A 26B judge (gemma) kept discriminating under the
full bundle. Prompt mass × judge capacity is therefore a first-class knob:
small judges can label under a compressed render while frontier judges keep
the full policy (the reference semantics).

Design contract — a PROJECTION, never a paraphrase:
  * Byte-deterministic: the same bundle always digests to the same bytes, so
    (policy version, render mode) still pins the exact prompt — reproducible
    runs, stable prompt-cache prefixes, no LLM in the loop and therefore no
    compression-fidelity audit.
  * Whole known-boilerplate sections are DROPPED (rationale, SME workflow,
    dataset-curation notes); every kept section survives verbatim. Unknown
    sections — e.g. drafter-minted ones — are kept: safety over size.
  * Node ids, titles, graph shape (parent/edges) and every decision rule
    survive, so citations, l2_label and boundary semantics keep working.

The bundle format is ``load_policy_markdown``'s: per node, a
``<!-- name.md -->`` marker, YAML frontmatter between ``---`` fences, then a
markdown body with ``##`` sections.
"""
from __future__ import annotations

import re

# Frontmatter keys a JUDGE needs: identity, title, and graph shape (parent +
# edges carry the boundary/confusion semantics the prompts reference).
# Everything else — version/area (constant per bundle), status, coverage
# targets, source anchors, canonical example bookkeeping — is curation
# metadata, dropped.
FRONTMATTER_KEEP_KEYS: tuple[str, ...] = (
    "id",
    "title",
    "node_type",
    "parent",
    "polarity",
    "edges",
)

# Body sections that never carry decision semantics for a judge: rationale
# ("Why this node exists"), human-workflow notes ("SME review expectations"),
# and dataset-curation guidance ("Hard negatives" = what images to COLLECT).
# Matched on the lowercased ## heading text.
BODY_DROP_SECTIONS: frozenset[str] = frozenset(
    {
        "why this node exists",
        "sme review expectations",
        "hard negatives",
        "canonical examples",
    }
)

_NODE_MARKER_RE = re.compile(r"^<!-- (?P<name>[^>]+\.md) -->$", re.MULTILINE)
_SECTION_SPLIT_RE = re.compile(r"^(?=## )", re.MULTILINE)


def _compress_frontmatter(frontmatter: str) -> str:
    """Keep only ``FRONTMATTER_KEEP_KEYS`` (with their indented/list
    continuation lines), preserving original order and bytes of kept lines."""
    kept: list[str] = []
    keeping = False
    for line in frontmatter.splitlines():
        top_level = re.match(r"^([A-Za-z_][\w-]*):", line)
        if top_level:
            keeping = top_level.group(1) in FRONTMATTER_KEEP_KEYS
        if keeping:
            kept.append(line)
    return "\n".join(kept)


def _compress_body(body: str) -> str:
    """Drop ``BODY_DROP_SECTIONS``; keep every other section verbatim."""
    parts = _SECTION_SPLIT_RE.split(body)
    kept: list[str] = []
    for part in parts:
        if part.startswith("## "):
            heading = part.splitlines()[0][3:].strip().lower()
            if heading in BODY_DROP_SECTIONS:
                continue
        kept.append(part.rstrip() + "\n")
    return "".join(kept)


def _compress_node(text: str) -> str:
    """Digest one node file's text (frontmatter + body)."""
    stripped = text.lstrip("\n")
    if stripped.startswith("---"):
        fence_end = stripped.find("\n---", 3)
        if fence_end != -1:
            frontmatter = stripped[4:fence_end]
            body = stripped[fence_end + 4 :].lstrip("\n")
            fm = _compress_frontmatter(frontmatter)
            return f"---\n{fm}\n---\n{_compress_body(body)}"
    return _compress_body(stripped)


def compress_policy_markdown(bundle: str) -> str:
    """Digest a full ``load_policy_markdown`` bundle. Pure and deterministic."""
    matches = list(_NODE_MARKER_RE.finditer(bundle))
    if not matches:
        return _compress_node(bundle).strip() + "\n"
    chunks: list[str] = []
    for i, match in enumerate(matches):
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(bundle)
        chunks.append(f"\n\n{match.group(0)}\n")
        chunks.append(_compress_node(bundle[start:end]).rstrip() + "\n")
    return "".join(chunks).strip() + "\n"


def parse_compressed_models(raw: str | None) -> frozenset[str]:
    """Parse a comma-separated ``--compressed-models`` value (CLI surface)."""
    if not raw:
        return frozenset()
    return frozenset(part.strip() for part in raw.split(",") if part.strip())


__all__ = [
    "BODY_DROP_SECTIONS",
    "FRONTMATTER_KEEP_KEYS",
    "compress_policy_markdown",
    "parse_compressed_models",
]
