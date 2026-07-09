"""Shared thumbnail path helpers for demo source images.

Thumbnails use a human-inspectable path-mirroring scheme: a source image at
``data/images/<demo>-classification/source-datasets/<dataset>/<label>/foo.png``
maps to
``data/images/<demo>-classification/derived/thumbnails/<dataset>/<label>/foo.jpg``.

Roots are per-demo/area (clean separation): each demo owns its own
``source-datasets`` and ``derived/thumbnails`` trees. See
``SOURCE_THUMBNAIL_ROOTS`` / ``DEMO_THUMBNAIL_ROOTS``.
"""
from __future__ import annotations

from pathlib import Path, PurePosixPath

# --- Per-demo/area roots (clean separation) --------------------------------
GENAI_SOURCE_ROOT_REL = Path("data/images/genai-classification/source-datasets")
GENAI_THUMBNAIL_ROOT_REL = Path("data/images/genai-classification/derived/thumbnails")
GENAI_SAMPLE_ROOT_REL = Path("data/images/genai-classification/sample")
MNIST_SOURCE_ROOT_REL = Path("data/images/mnist-classification/source-datasets")
MNIST_THUMBNAIL_ROOT_REL = Path("data/images/mnist-classification/derived/thumbnails")

# Back-compat aliases (GenAI was the original single-demo default).
SOURCE_ROOT_REL = GENAI_SOURCE_ROOT_REL
THUMBNAIL_ROOT_REL = GENAI_THUMBNAIL_ROOT_REL

# Each demo owns its own derived/thumbnails tree that mirrors its source tree.
SOURCE_THUMBNAIL_ROOTS: tuple[tuple[Path, Path | None], ...] = (
    (GENAI_SOURCE_ROOT_REL, GENAI_THUMBNAIL_ROOT_REL),
    # The committed GenAI portable fixture: sparse clones run entirely off
    # these images (their manifest rows point HERE, not at source-datasets),
    # so anchor-evidence thumbnails must resolve them. No derived thumbnails
    # exist for the fixture — the (already small) originals serve as-is.
    (GENAI_SAMPLE_ROOT_REL, None),
    (MNIST_SOURCE_ROOT_REL, MNIST_THUMBNAIL_ROOT_REL),
)

# Demo/area -> (source_root, thumbnail_root) for demo-aware callers.
DEMO_THUMBNAIL_ROOTS: dict[str, tuple[Path, Path]] = {
    "genai": (GENAI_SOURCE_ROOT_REL, GENAI_THUMBNAIL_ROOT_REL),
    "mnist": (MNIST_SOURCE_ROOT_REL, MNIST_THUMBNAIL_ROOT_REL),
}

# All repo-relative roots a request path may legitimately point at: every
# source root plus every derived thumbnail root. Serving already-derived
# thumbnail paths keeps demos.js free to reference derived/thumbnails directly.
_THUMBNAIL_OUTPUT_ROOTS: tuple[Path, ...] = tuple(
    thumb for _src, thumb in SOURCE_THUMBNAIL_ROOTS if thumb is not None
)
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}


def _repo_rel_posix(value: str | Path) -> PurePosixPath:
    text = str(value).replace("\\", "/").strip()
    path = PurePosixPath(text)
    if not text or path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"invalid repo-relative image path: {value!r}")
    return path


def validate_source_repo_path(repo_root: Path | str, repo_rel_path: str | Path) -> Path:
    """Return a normalized source-image path under ``SOURCE_ROOT_REL``.

    Raises ``ValueError`` for absolute paths, traversal, non-source paths, and
    non-image suffixes. The returned path is repo-relative and safe to join to
    ``repo_root``.
    """
    rel_posix = _repo_rel_posix(repo_rel_path)
    allowed_roots: tuple[Path, ...] = tuple(
        root for root, _thumb in SOURCE_THUMBNAIL_ROOTS
    ) + _THUMBNAIL_OUTPUT_ROOTS
    allowed_root: Path | None = None
    for candidate_root in allowed_roots:
        try:
            rel_posix.relative_to(PurePosixPath(candidate_root.as_posix()))
        except ValueError:
            continue
        allowed_root = candidate_root
        break
    if allowed_root is None:
        roots = ", ".join(root.as_posix() for root in allowed_roots)
        raise ValueError(f"image path must be under one of: {roots}")
    if rel_posix.suffix.lower() not in IMAGE_SUFFIXES:
        raise ValueError(f"unsupported image suffix: {rel_posix.suffix!r}")

    root = Path(repo_root).resolve()
    candidate = (root / Path(*rel_posix.parts)).resolve(strict=False)
    resolved_root = (root / allowed_root).resolve(strict=False)
    if resolved_root != candidate and resolved_root not in candidate.parents:
        raise ValueError(f"image path must be under {allowed_root.as_posix()}")
    return Path(*rel_posix.parts)


def thumbnail_rel_path_for_source(
    source_repo_rel_path: str | Path,
    *,
    source_root_rel: Path | str = SOURCE_ROOT_REL,
    output_root_rel: Path | str = THUMBNAIL_ROOT_REL,
) -> Path:
    """Map a source repo-relative image path to its thumbnail repo-relative path.

    If the input is already under a derived thumbnail root, it is returned
    unchanged (it is already a thumbnail path).
    """
    source_rel = Path(str(_repo_rel_posix(source_repo_rel_path)))
    for thumbnail_root in _THUMBNAIL_OUTPUT_ROOTS:
        try:
            source_rel.relative_to(thumbnail_root)
        except ValueError:
            continue
        return source_rel
    source_root = Path(source_root_rel)
    output_root = Path(output_root_rel)
    for candidate_source_root, candidate_output_root in SOURCE_THUMBNAIL_ROOTS:
        try:
            source_rel.relative_to(candidate_source_root)
        except ValueError:
            continue
        if candidate_output_root is None:
            return source_rel
        source_root = candidate_source_root
        output_root = candidate_output_root
        break
    rel_inside_source = source_rel.relative_to(source_root)
    return (output_root / rel_inside_source).with_suffix(".jpg")


__all__ = [
    "IMAGE_SUFFIXES",
    "SOURCE_ROOT_REL",
    "SOURCE_THUMBNAIL_ROOTS",
    "THUMBNAIL_ROOT_REL",
    "GENAI_SOURCE_ROOT_REL",
    "GENAI_THUMBNAIL_ROOT_REL",
    "GENAI_SAMPLE_ROOT_REL",
    "MNIST_SOURCE_ROOT_REL",
    "MNIST_THUMBNAIL_ROOT_REL",
    "DEMO_THUMBNAIL_ROOTS",
    "thumbnail_rel_path_for_source",
    "validate_source_repo_path",
]
