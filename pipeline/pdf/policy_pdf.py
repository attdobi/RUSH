"""Build a single bound PDF from policy-graph Markdown files.

Design notes:
- Stays self-contained: only requires `reportlab`. No network, no shell-out.
- Sorts input files for determinism (root first, then alphabetical).
- Strips YAML frontmatter, renders headings/paragraphs/list items in a
  reportlab document. Markdown formatting is intentionally minimal — the
  goal is a faithful, readable, bound copy of the policy bundle, not a
  perfect Markdown renderer.
- Returns a structured BuildResult so callers (CLI, tests, web glue) can
  surface counts and sizes without re-reading the artifact.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
import re
from typing import Iterable, Sequence

try:
    from reportlab.lib.enums import TA_LEFT
    from reportlab.lib.pagesizes import LETTER
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import inch
    from reportlab.platypus import (
        Image as RLImage,
        KeepTogether,
        Paragraph,
        PageBreak,
        SimpleDocTemplate,
        Spacer,
        Table,
        TableStyle,
    )
except Exception as exc:  # pragma: no cover - import-time guard
    raise ImportError(
        "reportlab is required for pipeline.pdf. Install with `pip install reportlab`."
    ) from exc
from .node_examples import (
    PolicyImageExample,
    collect_policy_image_examples as _collect_policy_image_examples,
    prepare_thumbnail_bytes,
)



class PolicyPdfError(RuntimeError):
    """Raised when the PDF cannot be built (missing inputs, IO, etc.)."""


@dataclass
class BuildResult:
    """Summary returned by :func:`build_policy_pdf`."""

    output_path: Path
    source_dir: Path
    policy_graph_version: str
    file_count: int
    page_count: int
    byte_size: int
    sources: list[Path] = field(default_factory=list)


_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)
_WIKI_LINK_RE = re.compile(r"\[\[([^\]|]+)\|([^\]]+)\]\]")
_BARE_WIKI_LINK_RE = re.compile(r"\[\[([^\]]+)\]\]")
_INLINE_CODE_RE = re.compile(r"`([^`]+)`")
_BOLD_RE = re.compile(r"\*\*([^*]+)\*\*")
_ITALIC_RE = re.compile(r"(?<!\*)\*([^*]+)\*(?!\*)")


def parse_frontmatter(text: str) -> tuple[dict[str, str], str]:
    """Extract a flat YAML-ish frontmatter dict + the body.

    We don't pull in PyYAML for a single header block. The policy markdowns
    use simple `key: value` pairs at the top level (nested values like
    ``coverage_target`` are intentionally ignored — only top-level scalars
    are returned).
    """
    match = _FRONTMATTER_RE.match(text)
    if not match:
        return {}, text
    block = match.group(1)
    body = text[match.end() :]
    meta: dict[str, str] = {}
    for line in block.splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        if line.startswith(" ") or line.startswith("\t"):
            # nested under previous key — skip for the simple parser
            continue
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        key = key.strip()
        value = value.strip()
        if not key or value.startswith("[") or value.startswith("{"):
            # skip list/object scalars
            meta.setdefault(key, "")
            continue
        # strip surrounding quotes
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        meta[key] = value
    return meta, body


def iter_policy_markdown(source_dir: Path) -> list[Path]:
    """Return policy markdown files in deterministic order.

    The root file (``GA.root.md``) is placed first; the rest are sorted
    alphabetically. Hidden files and non-``.md`` files are skipped.
    """
    if not source_dir.exists() or not source_dir.is_dir():
        raise PolicyPdfError(f"policy source directory does not exist: {source_dir}")
    files = sorted(
        p for p in source_dir.iterdir()
        if p.is_file() and p.suffix == ".md" and not p.name.startswith(".")
    )
    if not files:
        raise PolicyPdfError(f"no .md policy files found in {source_dir}")
    root = [p for p in files if p.name.endswith(".root.md")]
    rest = [p for p in files if not p.name.endswith(".root.md")]
    return root + rest


def _escape_xml(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def _inline_format(line: str) -> str:
    """Apply minimal inline Markdown -> reportlab mini-XML."""
    text = _escape_xml(line)
    # Wiki-style links: [[target|alias]] -> alias (target)
    text = _WIKI_LINK_RE.sub(lambda m: f"{m.group(2)} ({m.group(1)})", text)
    text = _BARE_WIKI_LINK_RE.sub(lambda m: m.group(1), text)
    text = _BOLD_RE.sub(r"<b>\1</b>", text)
    text = _ITALIC_RE.sub(r"<i>\1</i>", text)
    text = _INLINE_CODE_RE.sub(r'<font face="Courier">\1</font>', text)
    return text


def _styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    title = ParagraphStyle(
        "RushTitle",
        parent=base["Title"],
        fontSize=22,
        leading=26,
        spaceAfter=14,
    )
    h1 = ParagraphStyle(
        "RushH1",
        parent=base["Heading1"],
        fontSize=18,
        leading=22,
        spaceBefore=10,
        spaceAfter=8,
    )
    h2 = ParagraphStyle(
        "RushH2",
        parent=base["Heading2"],
        fontSize=14,
        leading=18,
        spaceBefore=8,
        spaceAfter=6,
    )
    h3 = ParagraphStyle(
        "RushH3",
        parent=base["Heading3"],
        fontSize=12,
        leading=15,
        spaceBefore=6,
        spaceAfter=4,
    )
    body = ParagraphStyle(
        "RushBody",
        parent=base["BodyText"],
        fontSize=10,
        leading=14,
        alignment=TA_LEFT,
        spaceAfter=6,
    )
    code = ParagraphStyle(
        "RushCode",
        parent=base["Code"],
        fontSize=9,
        leading=12,
        spaceAfter=6,
    )
    meta = ParagraphStyle(
        "RushMeta",
        parent=base["BodyText"],
        fontSize=9,
        leading=12,
        textColor="#555555",
        spaceAfter=10,
    )
    bullet = ParagraphStyle(
        "RushBullet",
        parent=body,
        leftIndent=18,
        bulletIndent=6,
        spaceAfter=2,
    )
    return {
        "title": title,
        "h1": h1,
        "h2": h2,
        "h3": h3,
        "body": body,
        "code": code,
        "meta": meta,
        "bullet": bullet,
    }


def _md_to_flowables(body: str, styles: dict[str, ParagraphStyle]) -> list:
    """Convert minimal Markdown body to a flat list of flowables."""
    flowables: list = []
    in_code = False
    code_lines: list[str] = []
    paragraph: list[str] = []

    def flush_paragraph() -> None:
        if paragraph:
            joined = " ".join(paragraph).strip()
            if joined:
                flowables.append(Paragraph(_inline_format(joined), styles["body"]))
            paragraph.clear()

    def flush_code() -> None:
        if code_lines:
            block = _escape_xml("\n".join(code_lines))
            block_xml = block.replace("\n", "<br/>")
            flowables.append(
                Paragraph(f'<font face="Courier">{block_xml}</font>', styles["code"])
            )
            code_lines.clear()

    for raw in body.splitlines():
        line = raw.rstrip()
        stripped = line.lstrip()
        if stripped.startswith("```"):
            flush_paragraph()
            if in_code:
                flush_code()
                in_code = False
            else:
                in_code = True
            continue
        if in_code:
            code_lines.append(line)
            continue
        if not stripped:
            flush_paragraph()
            continue
        if stripped.startswith("### "):
            flush_paragraph()
            flowables.append(Paragraph(_inline_format(stripped[4:].strip()), styles["h3"]))
            continue
        if stripped.startswith("## "):
            flush_paragraph()
            flowables.append(Paragraph(_inline_format(stripped[3:].strip()), styles["h2"]))
            continue
        if stripped.startswith("# "):
            flush_paragraph()
            flowables.append(Paragraph(_inline_format(stripped[2:].strip()), styles["h1"]))
            continue
        if stripped.startswith(("- ", "* ", "+ ")):
            flush_paragraph()
            item = stripped[2:].strip()
            flowables.append(
                Paragraph(_inline_format(item), styles["bullet"], bulletText="•")
            )
            continue
        # ordered list
        m = re.match(r"^(\d+)\.\s+(.*)", stripped)
        if m:
            flush_paragraph()
            flowables.append(
                Paragraph(
                    _inline_format(m.group(2)),
                    styles["bullet"],
                    bulletText=f"{m.group(1)}.",
                )
            )
            continue
        paragraph.append(stripped)

    flush_paragraph()
    if in_code:
        flush_code()
    return flowables


def _meta_lines(meta: dict[str, str]) -> str:
    keys = ("id", "version", "node_type", "parent", "polarity", "status")
    pairs = [f"<b>{k}</b>: {_escape_xml(meta[k])}" for k in keys if meta.get(k)]
    return " &nbsp;·&nbsp; ".join(pairs)


def _repo_root_for_source(source_dir: Path) -> Path | None:
    """Infer the checkout root for a policy source directory, if possible."""
    resolved = source_dir.resolve()
    for candidate in (resolved, *resolved.parents):
        if (candidate / "policy-graph").is_dir() and (candidate / "data").is_dir():
            return candidate
    return None


def _example_caption(example: PolicyImageExample) -> str:
    parts = [f"<b>{_escape_xml(example.image_id)}</b>"]
    if example.label:
        parts.append(_escape_xml(example.label))
    if example.tier:
        parts.append(_escape_xml(example.tier))
    if example.split:
        parts.append(_escape_xml(example.split))
    if example.confidence is not None:
        parts.append(f"conf {example.confidence:.2f}")
    return " · ".join(parts)


def _image_example_flowables(
    node_id: str,
    examples_by_node: dict[str, list[PolicyImageExample]],
    styles: dict[str, ParagraphStyle],
) -> list:
    examples = examples_by_node.get(node_id, [])
    if not examples:
        return []

    cells: list[list] = []
    image_size = 1.35 * inch
    for example in examples:
        thumbnail = prepare_thumbnail_bytes(example.media_path, max_px=200)
        if thumbnail is None:
            continue
        cells.append([
            RLImage(
                BytesIO(thumbnail),
                width=image_size,
                height=image_size,
                kind="proportional",
            ),
            Paragraph(_example_caption(example), styles["meta"]),
        ])

    flow: list = []
    if cells:
        flow.append(Paragraph("Image examples", styles["h3"]))
        rows = [cells[i : i + 3] for i in range(0, len(cells), 3)]
        for row in rows:
            row.extend([""] * (3 - len(row)))
        table = Table(rows, colWidths=[1.75 * inch] * 3, hAlign="LEFT")
        table.setStyle(TableStyle([
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 0),
            ("RIGHTPADDING", (0, 0), (-1, -1), 8),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ]))
        flow.append(KeepTogether([table, Spacer(1, 0.05 * inch)]))
    return flow


def _cover_flowables(
    styles: dict[str, ParagraphStyle],
    source_dir: Path,
    files: Sequence[Path],
    policy_graph_version: str,
) -> list:
    flow: list = [
        Paragraph("RUSH bound policy bundle", styles["title"]),
        Paragraph(
            f"Policy graph version: <b>{_escape_xml(policy_graph_version)}</b>",
            styles["body"],
        ),
        Paragraph(
            f"Source: {_escape_xml(str(source_dir))}",
            styles["meta"],
        ),
        Paragraph(
            "This PDF is generated on demand from the on-disk policy markdown bundle. "
            "It is a build artifact (not committed to the repo) and is intended for SME "
            "review and offline reading. The web UI links to the most recent build under "
            "the run output directory.",
            styles["body"],
        ),
        Spacer(1, 0.15 * inch),
        Paragraph("Contents", styles["h2"]),
    ]
    for path in files:
        flow.append(Paragraph(_escape_xml(path.name), styles["bullet"], bulletText="•"))
    flow.append(PageBreak())
    return flow


def build_policy_pdf(
    source_dir: Path | str,
    output_path: Path | str,
    *,
    policy_graph_version: str | None = None,
    examples_root: Path | str | None = None,
    examples_per_node: int = 3,
    include_examples: bool = True,
) -> BuildResult:
    """Build a bound PDF from all `.md` files in *source_dir*.

    Args:
        source_dir: Directory containing policy markdown files
            (e.g. ``policy-graph/Generative_AI/v0.1``).
        output_path: Destination ``.pdf`` path. Parent directories are created.
        policy_graph_version: Optional override; defaults to the parent
            directory name of *source_dir* (``v0.1``).
        examples_root: Optional data/manifest/run root for image examples.
            Defaults to the checkout ``data/`` directory when it can be inferred.
        examples_per_node: Maximum thumbnails to render under each node.
        include_examples: Set false to skip image discovery/rendering entirely.

    Returns:
        :class:`BuildResult` describing the artifact.
    """
    source_dir = Path(source_dir)
    output_path = Path(output_path)
    files = iter_policy_markdown(source_dir)
    version = policy_graph_version or source_dir.name

    output_path.parent.mkdir(parents=True, exist_ok=True)

    styles = _styles()
    repo_root = _repo_root_for_source(source_dir)
    resolved_examples_root = Path(examples_root) if examples_root is not None else (repo_root / "data" if repo_root else None)
    examples_by_node = (
        _collect_policy_image_examples(
            resolved_examples_root,
            repo_root=repo_root,
            max_per_node=examples_per_node,
        )
        if include_examples and examples_per_node > 0
        else {}
    )
    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=LETTER,
        title=f"RUSH Policy Bundle {version}",
        author="RUSH",
        leftMargin=0.75 * inch,
        rightMargin=0.75 * inch,
        topMargin=0.75 * inch,
        bottomMargin=0.75 * inch,
    )

    story: list = _cover_flowables(styles, source_dir, files, version)
    for idx, path in enumerate(files):
        text = path.read_text(encoding="utf-8")
        meta, body = parse_frontmatter(text)
        title = meta.get("title") or meta.get("id") or path.stem
        story.append(Paragraph(_escape_xml(title), styles["h1"]))
        meta_line = _meta_lines(meta)
        if meta_line:
            story.append(Paragraph(meta_line, styles["meta"]))
        story.extend(_md_to_flowables(body, styles))
        node_id = meta.get("id") or path.stem
        story.extend(_image_example_flowables(node_id, examples_by_node, styles))
        if idx < len(files) - 1:
            story.append(PageBreak())

    page_counter = {"n": 0}

    def _on_page(_canvas, _doc) -> None:
        page_counter["n"] = max(page_counter["n"], _doc.page)

    doc.build(story, onFirstPage=_on_page, onLaterPages=_on_page)

    byte_size = output_path.stat().st_size
    return BuildResult(
        output_path=output_path,
        source_dir=source_dir,
        policy_graph_version=version,
        file_count=len(files),
        page_count=page_counter["n"],
        byte_size=byte_size,
        sources=list(files),
    )


def build_from_files(
    sources: Iterable[Path],
    output_path: Path | str,
    *,
    policy_graph_version: str,
    examples_root: Path | str | None = None,
    examples_per_node: int = 3,
    include_examples: bool = True,
) -> BuildResult:
    """Build a PDF from an explicit list of files (used by tests)."""
    sources = list(sources)
    if not sources:
        raise PolicyPdfError("build_from_files requires at least one source")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    styles = _styles()
    repo_root = _repo_root_for_source(sources[0].parent)
    resolved_examples_root = Path(examples_root) if examples_root is not None else (repo_root / "data" if repo_root else None)
    examples_by_node = (
        _collect_policy_image_examples(
            resolved_examples_root,
            repo_root=repo_root,
            max_per_node=examples_per_node,
        )
        if include_examples and examples_per_node > 0
        else {}
    )
    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=LETTER,
        title=f"RUSH Policy Bundle {policy_graph_version}",
        author="RUSH",
        leftMargin=0.75 * inch,
        rightMargin=0.75 * inch,
        topMargin=0.75 * inch,
        bottomMargin=0.75 * inch,
    )
    story: list = _cover_flowables(styles, sources[0].parent, sources, policy_graph_version)
    for idx, path in enumerate(sources):
        text = path.read_text(encoding="utf-8")
        meta, body = parse_frontmatter(text)
        title = meta.get("title") or meta.get("id") or path.stem
        story.append(Paragraph(_escape_xml(title), styles["h1"]))
        meta_line = _meta_lines(meta)
        if meta_line:
            story.append(Paragraph(meta_line, styles["meta"]))
        story.extend(_md_to_flowables(body, styles))
        node_id = meta.get("id") or path.stem
        story.extend(_image_example_flowables(node_id, examples_by_node, styles))
        if idx < len(sources) - 1:
            story.append(PageBreak())
    page_counter = {"n": 0}

    def _on_page(_canvas, _doc) -> None:
        page_counter["n"] = max(page_counter["n"], _doc.page)

    doc.build(story, onFirstPage=_on_page, onLaterPages=_on_page)
    return BuildResult(
        output_path=output_path,
        source_dir=sources[0].parent,
        policy_graph_version=policy_graph_version,
        file_count=len(sources),
        page_count=page_counter["n"],
        byte_size=output_path.stat().st_size,
        sources=list(sources),
    )
