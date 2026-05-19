"""Policy-node image example discovery for the bound PDF.

The PDF builder keeps rendering concerns in ``policy_pdf.py``. This module owns
only data discovery and thumbnail-byte preparation so the same example sourcing
can be reused by tests/CLI without growing the renderer.
"""
from __future__ import annotations

from dataclasses import dataclass
import csv
import json
from pathlib import Path
import re
from typing import Any, Iterable

from pipeline.labeling.image_prep import prepare_image_for_labeling
from pipeline.io_paths import REPO_ROOT
from pipeline.manifest import DEFAULT_SAMPLE_MANIFEST, load_records
from pipeline.thumbnails import thumbnail_rel_path_for_source


@dataclass(frozen=True)
class PolicyImageExample:
    """Resolved image example associated with a policy node."""

    node_id: str
    image_id: str
    media_path: Path
    label: str = ""
    confidence: float | None = None
    split: str = ""
    tier: str = ""
    source: str = ""
    justification: str = ""
    source_priority: int = 50


_NODE_ID_RE = re.compile(r"^GA\.[A-Za-z0-9_.-]+$")
_LIST_SPLIT_RE = re.compile(r"[,;|]\s*")
_IMAGE_FIELDS = (
    "media_path",
    "media_uri",
    "repo_rel_path",
    "thumbnail_path",
    "thumbnail_rel_path",
    "path",
    "file_path",
)
_GOLDEN_NAMES = (
    "combined_labels",
    "dev_golden",
    "development",
    "holdout",
    "locked_holdout",
)
_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".gif"}
_MAX_THUMBNAIL_BYTES = 80 * 1024


def _read_record_file(path: Path) -> list[dict[str, Any]]:
    """Read JSON array/object, JSONL, or CSV records from *path*."""
    if not path.exists() or not path.is_file():
        return []
    suffix = path.suffix.lower()
    try:
        if suffix == ".json":
            payload = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(payload, list):
                return [row for row in payload if isinstance(row, dict)]
            if isinstance(payload, dict):
                for key in ("records", "items", "images", "samples", "rows"):
                    rows = payload.get(key)
                    if isinstance(rows, list):
                        return [row for row in rows if isinstance(row, dict)]
                return [payload]
        if suffix == ".jsonl":
            rows: list[dict[str, Any]] = []
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                if isinstance(row, dict):
                    rows.append(row)
            return rows
        if suffix == ".csv":
            with path.open(newline="", encoding="utf-8") as fh:
                return [dict(row) for row in csv.DictReader(fh)]
    except (OSError, json.JSONDecodeError, csv.Error):
        return []
    return []


def _candidate_roots(examples_root: Path | None, repo_root: Path | None) -> list[Path]:
    roots: list[Path] = []
    for root in (examples_root, repo_root / "data" if repo_root is not None else None, repo_root):
        if root is None:
            continue
        path = Path(root)
        if path.is_file():
            path = path.parent
        if path not in roots:
            roots.append(path)
    return roots


def _manifest_candidates(examples_root: Path | None, repo_root: Path | None) -> list[Path]:
    candidates: list[Path] = []
    if examples_root is not None:
        root = Path(examples_root)
        if root.is_file():
            candidates.append(root)
        else:
            for base in (
                root,
                root / "images" / "genai-classification" / "manifests",
                root / "genai-classification" / "manifests",
                root / "data" / "images" / "genai-classification" / "manifests",
            ):
                for name in _GOLDEN_NAMES:
                    for suffix in (".jsonl", ".json", ".csv"):
                        candidates.append(base / f"{name}{suffix}")
    if repo_root is not None:
        candidates.append(repo_root / DEFAULT_SAMPLE_MANIFEST.relative_to(REPO_ROOT))
        manifest_dir = repo_root / "data" / "images" / "genai-classification" / "manifests"
        for name in _GOLDEN_NAMES:
            for suffix in (".jsonl", ".json", ".csv"):
                candidates.append(manifest_dir / f"{name}{suffix}")

    deduped: list[Path] = []
    seen: set[Path] = set()
    for path in candidates:
        try:
            key = path.resolve(strict=False)
        except OSError:
            key = path
        if key not in seen:
            seen.add(key)
            deduped.append(path)
    return deduped


def _seed_record_paths(examples_root: Path | None, repo_root: Path | None, basename: str) -> list[Path]:
    paths: list[Path] = []
    for root in _candidate_roots(examples_root, repo_root):
        paths.extend([
            root / "seed" / f"{basename}.json",
            root / "data" / "seed" / f"{basename}.json",
        ])
    return paths


def _run_roots(examples_root: Path | None, repo_root: Path | None) -> list[Path]:
    roots: list[Path] = []
    for root in _candidate_roots(examples_root, repo_root):
        for candidate in (root / "runs", root / "data" / "runs", root):
            if candidate.exists() and candidate.is_dir() and candidate not in roots:
                roots.append(candidate)
    return roots


def _scored_record_paths(examples_root: Path | None, repo_root: Path | None) -> list[Path]:
    paths: list[Path] = []
    seen: set[Path] = set()
    for root in _run_roots(examples_root, repo_root):
        direct = [root / "label_votes.jsonl"] if (root / "label_votes.jsonl").exists() else []
        for path in [*direct, *root.glob("*/label_votes.jsonl"), *root.glob("*/scoring/*.jsonl"), *root.glob("*/scoring/*.json")]:
            if path.is_file() and path not in seen:
                seen.add(path)
                paths.append(path)
    return paths


def _image_id(row: dict[str, Any]) -> str | None:
    for key in ("image_id", "sample_id", "id"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _confidence(row: dict[str, Any]) -> float | None:
    value = row.get("confidence")
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _flatten_node_value(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        for part in _LIST_SPLIT_RE.split(value.strip()):
            part = part.strip()
            if _NODE_ID_RE.match(part):
                yield part
        return
    if isinstance(value, dict):
        explicit = False
        for key in ("node_id", "node", "top_node", "l2_label", "policy_node", "policy_node_id"):
            if key in value:
                explicit = True
                yield from _flatten_node_value(value[key])
        if not explicit:
            for key, inner in value.items():
                if isinstance(key, str) and _NODE_ID_RE.match(key):
                    yield key
                yield from _flatten_node_value(inner)
        return
    if isinstance(value, list):
        for item in value:
            yield from _flatten_node_value(item)


def _node_ids_from_row(
    row: dict[str, Any],
    *,
    allow_label_fallback: bool = True,
) -> list[str]:
    nodes: set[str] = set()
    for key in (
        "node_assignments",
        "top_node",
        "node_ids",
        "policy_node_ids",
        "nodes",
        "l2_label",
        "policy_node",
        "policy_node_id",
    ):
        if key in row:
            nodes.update(_flatten_node_value(row[key]))

    if not nodes and allow_label_fallback:
        # Broad fallback for class-only golden manifests: useful for the root
        # and authentic-photo negative node without fabricating detailed leaf
        # evidence.
        label = str(row.get("label") or row.get("sme_label") or "").strip().lower()
        if label in {"ai_generated", "gen_ai", "positive", "1", "true"}:
            nodes.add("GA.root")
        elif label in {"not_ai_generated", "not_gen_ai", "negative", "0", "false"}:
            nodes.add("GA.negative.authentic_photo")
    return sorted(nodes)


def _resolve_media_path(
    row: dict[str, Any],
    *,
    examples_root: Path | None,
    repo_root: Path | None,
    manifest_path: Path | None = None,
) -> Path | None:
    raw: str | None = None
    for key in _IMAGE_FIELDS:
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            raw = value.strip()
            break
    if not raw or "://" in raw:
        return None

    candidate_roots = _candidate_roots(examples_root, repo_root)
    if manifest_path is not None:
        candidate_roots.insert(0, manifest_path.parent)

    candidates: list[Path] = []
    path = Path(raw)
    if path.is_absolute():
        candidates.append(path)
    else:
        for root in candidate_roots:
            candidates.extend([
                root / path,
                root / "data" / path,
                root / "images" / "genai-classification" / path,
            ])
        if repo_root is not None:
            candidates.extend([
                repo_root / path,
                repo_root / "data" / path,
                repo_root / "data" / "images" / "genai-classification" / path,
            ])
            # If source images are absent but generated thumbnails exist, use the
            # thumbnail path rather than duplicating path-mirroring logic here.
            try:
                thumb_rel = thumbnail_rel_path_for_source(path)
            except (ValueError, TypeError):
                thumb_rel = None
            if thumb_rel is not None:
                candidates.append(repo_root / thumb_rel)

    for candidate in candidates:
        try:
            resolved = candidate.resolve(strict=False)
        except OSError:
            continue
        if resolved.is_file() and resolved.suffix.lower() in _IMAGE_SUFFIXES:
            return resolved
    return None


def _make_example(
    row: dict[str, Any],
    *,
    node_id: str,
    media_path: Path,
    source: str,
    priority: int,
) -> PolicyImageExample | None:
    image_id = _image_id(row)
    if image_id is None:
        return None
    return PolicyImageExample(
        node_id=node_id,
        image_id=image_id,
        media_path=media_path,
        label=str(row.get("label") or row.get("sme_label") or ""),
        confidence=_confidence(row),
        split=str(row.get("split") or ""),
        tier=str(row.get("label_tier") or row.get("truth_tier") or ""),
        source=source,
        justification=str(row.get("justification") or row.get("assumption_note") or ""),
        source_priority=priority,
    )


def _sort_key(example: PolicyImageExample) -> tuple[int, int, int, float, str]:
    tier_rank = {"platinum": 0, "gold": 1, "gold_candidate": 2, "silver": 3, "provisional": 4}
    split_rank = {"development": 0, "dev_golden": 0, "validation": 1, "holdout": 3, "locked_holdout": 4}
    conf = example.confidence if example.confidence is not None else -1.0
    return (
        example.source_priority,
        tier_rank.get(example.tier, 99),
        split_rank.get(example.split, 99),
        -conf,
        example.image_id,
    )


def _load_golden_rows(
    examples_root: Path | None,
    repo_root: Path | None,
) -> list[tuple[dict[str, Any], Path | None]]:
    rows: list[tuple[dict[str, Any], Path | None]] = []
    for manifest in _manifest_candidates(examples_root, repo_root):
        if not manifest.exists():
            continue
        loaded_any = False
        # Prefer the typed manifest loader for the canonical combined_labels
        # format, then merge with raw rows so optional node assignment fields are
        # still visible to the PDF example picker.
        try:
            typed = load_records(manifest)
        except Exception:
            typed = []
        raw_by_id = {_image_id(row): row for row in _read_record_file(manifest)}
        for rec in typed:
            raw = dict(raw_by_id.get(rec.sample_id) or {})
            raw.update(
                {
                    "sample_id": rec.sample_id,
                    "repo_rel_path": rec.repo_rel_path,
                    "split": rec.split,
                    "label": rec.sme_label_raw,
                    "sme_label": rec.sme_label,
                    "dataset": rec.dataset,
                }
            )
            rows.append((raw, manifest))
            loaded_any = True
        if not loaded_any:
            rows.extend((row, manifest) for row in raw_by_id.values())
    return rows


def _load_seed_rows(
    examples_root: Path | None,
    repo_root: Path | None,
) -> list[dict[str, Any]]:
    image_rows: list[dict[str, Any]] = []
    label_rows: list[dict[str, Any]] = []
    for path in _seed_record_paths(examples_root, repo_root, "image-records"):
        image_rows.extend(_read_record_file(path))
    for path in _seed_record_paths(examples_root, repo_root, "label-records"):
        label_rows.extend(_read_record_file(path))

    image_by_id = {_image_id(row): row for row in image_rows if _image_id(row)}
    joined: list[dict[str, Any]] = []
    for label in label_rows:
        image = dict(image_by_id.get(_image_id(label)) or {})
        image.update(label)
        joined.append(image)
    # Some fixture manifests carry node_ids directly in image records.
    joined.extend(row for row in image_rows if _node_ids_from_row(row))
    return joined


def collect_policy_image_examples(
    examples_root: Path | str | None,
    *,
    repo_root: Path | str | None = None,
    max_per_node: int = 3,
) -> dict[str, list[PolicyImageExample]]:
    """Collect deterministic image examples keyed by policy node id.

    ``examples_root`` may be a repo ``data/`` directory, a repository root, a
    single manifest file, or a run directory. Scored run assignments are sorted
    before golden/seed examples for nodes where both are available.
    """
    if max_per_node <= 0:
        return {}

    root = Path(examples_root) if examples_root is not None else None
    repo = Path(repo_root) if repo_root is not None else None
    if repo is None and root is not None:
        for candidate in (root, *root.parents):
            if (candidate / "policy-graph").is_dir() and (candidate / "data").is_dir():
                repo = candidate
                break

    examples_by_node: dict[str, list[PolicyImageExample]] = {}

    def add_row(
        row: dict[str, Any],
        *,
        source: str,
        priority: int,
        manifest_path: Path | None = None,
        node_ids: list[str] | None = None,
    ) -> None:
        media_path = _resolve_media_path(row, examples_root=root, repo_root=repo, manifest_path=manifest_path)
        if media_path is None:
            return
        for node_id in (node_ids if node_ids is not None else _node_ids_from_row(row)):
            example = _make_example(row, node_id=node_id, media_path=media_path, source=source, priority=priority)
            if example is not None:
                examples_by_node.setdefault(node_id, []).append(example)

    # First source: the canonical golden/gold-set manifests.
    for row, manifest_path in _load_golden_rows(root, repo):
        add_row(row, source="golden_manifest", priority=20, manifest_path=manifest_path)

    # Also support committed seed image/label records as a tiny gold-set-shaped
    # fallback for clean clones and unit tests.
    for row in _load_seed_rows(root, repo):
        add_row(row, source="seed_records", priority=30)

    # Second source with preference: scored run/node assignments.
    image_lookup: dict[str, dict[str, Any]] = {}
    for row, manifest_path in _load_golden_rows(root, repo):
        image_id = _image_id(row)
        if image_id:
            copy = dict(row)
            copy.setdefault("_manifest_path", str(manifest_path) if manifest_path else "")
            image_lookup.setdefault(image_id, copy)
    for row in _load_seed_rows(root, repo):
        image_id = _image_id(row)
        if image_id:
            image_lookup.setdefault(image_id, row)

    for path in _scored_record_paths(root, repo):
        for row in _read_record_file(path):
            assigned_nodes = _node_ids_from_row(row, allow_label_fallback=False)
            if not assigned_nodes:
                continue
            image_id = _image_id(row)
            merged = dict(image_lookup.get(image_id or "") or {})
            merged.update(row)
            add_row(
                merged,
                source=f"scored:{path.parent.name}",
                priority=0,
                manifest_path=path,
                node_ids=assigned_nodes,
            )

    trimmed: dict[str, list[PolicyImageExample]] = {}
    for node_id, examples in examples_by_node.items():
        deduped: dict[str, PolicyImageExample] = {}
        for example in sorted(examples, key=_sort_key):
            deduped.setdefault(example.image_id, example)
        if deduped:
            trimmed[node_id] = list(deduped.values())[:max_per_node]
    return trimmed


def prepare_thumbnail_bytes(
    image_path: Path | str,
    *,
    max_px: int = 200,
    max_bytes: int = _MAX_THUMBNAIL_BYTES,
) -> bytes | None:
    """Return JPEG thumbnail bytes using the shared image-prep helper."""
    path = Path(image_path)
    for size, quality in ((max_px, 78), (max_px, 68), (180, 62), (160, 58)):
        try:
            prepared = prepare_image_for_labeling(path, max_size=(size, size), jpeg_quality=quality)
        except Exception:
            return None
        if prepared.byte_size <= max_bytes:
            return prepared.bytes_
        last = prepared.bytes_
    return last if len(last) <= max_bytes * 2 else None


__all__ = [
    "PolicyImageExample",
    "collect_policy_image_examples",
    "prepare_thumbnail_bytes",
]
