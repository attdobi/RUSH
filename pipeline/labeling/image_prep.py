"""Shared image downsampler for every LLM provider client.

Every provider client (OpenAI, Anthropic, Gemini) MUST route image bytes
through :func:`prepare_image` before building a request payload. This is the
single chokepoint for:

* deterministic resizing (longest edge ≤ 1024 px, LANCZOS, aspect-preserved);
* deterministic re-encoding to JPEG (quality ~85, RGB, no EXIF);
* a content-addressable sha256 of the bytes that actually go on the wire;
* persistable metadata that X2 can drop into every label record so we can
  prove which exact pixels each model saw.

Token-cost guidance (informational, do not gate on these):

* OpenAI vision (``detail: "high"``) at 1024² ≈ 765 tokens per image.
* Anthropic Claude vision at 768² ≈ 600 tokens per image.
* Google Gemini multimodal at 768² ≈ 1032 tokens per image.

We standardize on a longest-edge cap of 1024 px to keep the same bytes/hash
across providers; per-provider sub-resizing is intentionally NOT done here.
If a provider needs a tighter cap later, add a parameter — do not introduce a
second helper. One downsampler, one hash, one truth.
"""

from __future__ import annotations

import base64
import hashlib
import io
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final

from PIL import Image
from PIL.Image import Resampling

# Public constants. Tuned per the bulk-labeling-v1 plan; bump deliberately and
# bump prompt_version + label-vote records together if these change.
MAX_LONGEST_EDGE_PX: Final[int] = 1024
JPEG_QUALITY: Final[int] = 85
OUTPUT_MIME_TYPE: Final[str] = "image/jpeg"
OUTPUT_FORMAT: Final[str] = "JPEG"


@dataclass(frozen=True)
class PreparedImage:
    """A downsampled image ready to ship to an LLM provider.

    Attributes:
        bytes_: The downsampled JPEG payload that providers must send. This
            is also the byte string that ``sha256`` is computed over.
        sha256: Hex sha256 of ``bytes_``. This is the canonical image
            identifier used in label records (X2 persists it per vote).
        width: Pixel width of the downsampled image.
        height: Pixel height of the downsampled image.
        byte_size: ``len(bytes_)`` — convenience for callers/loggers.
        mime_type: Always ``image/jpeg`` (we standardize on JPEG).
        source_path: Original on-disk path if the input came from a file;
            ``None`` when prepared from an in-memory blob.
    """

    bytes_: bytes
    sha256: str
    width: int
    height: int
    byte_size: int
    mime_type: str
    source_path: str | None

    # Convenience helpers --------------------------------------------------

    def to_base64(self) -> str:
        """Base64-encode the downsampled bytes (no data: URL prefix)."""
        return base64.b64encode(self.bytes_).decode("ascii")

    def to_data_url(self) -> str:
        """``data:image/jpeg;base64,...`` URL form for OpenAI image_url blocks."""
        return f"data:{self.mime_type};base64,{self.to_base64()}"

    def metadata(self) -> dict[str, object]:
        """Persistable metadata block (no image bytes)."""
        meta = asdict(self)
        meta.pop("bytes_", None)
        return meta


def prepare_image(image_path: str | Path) -> PreparedImage:
    """Load ``image_path`` and return a :class:`PreparedImage`.

    Reads the file, downsamples with LANCZOS so the longest edge is
    ≤ ``MAX_LONGEST_EDGE_PX``, re-encodes as JPEG (quality
    ``JPEG_QUALITY``), hashes the result, and returns metadata + bytes.

    Args:
        image_path: Absolute or workspace-relative path to a readable image.

    Returns:
        A :class:`PreparedImage` carrying the downsampled bytes and metadata.

    Raises:
        FileNotFoundError: If ``image_path`` does not exist.
        OSError: If PIL cannot decode the file.
    """
    path = Path(image_path)
    if not path.is_file():
        raise FileNotFoundError(f"image not found: {path}")
    with path.open("rb") as fh:
        raw = fh.read()
    return prepare_image_bytes(raw, source_path=str(path))


def prepare_image_bytes(
    raw_bytes: bytes,
    *,
    source_path: str | None = None,
    max_size: tuple[int, int] = (MAX_LONGEST_EDGE_PX, MAX_LONGEST_EDGE_PX),
    jpeg_quality: int = JPEG_QUALITY,
) -> PreparedImage:
    """Same as :func:`prepare_image` but takes an in-memory blob.

    Useful for tests and for any future code path that already has bytes in
    hand (e.g. fetched from object storage). ``source_path`` is metadata
    only — never read from.
    """
    if not raw_bytes:
        raise ValueError("raw_bytes is empty")

    with Image.open(io.BytesIO(raw_bytes)) as img:
        # Convert to RGB so JPEG encoding is well-defined for PNG/RGBA inputs.
        # PIL's ``thumbnail`` mutates in-place and preserves aspect ratio.
        img = img.convert("RGB")
        img.thumbnail(max_size, resample=Resampling.LANCZOS)
        out = io.BytesIO()
        img.save(out, format=OUTPUT_FORMAT, quality=jpeg_quality, optimize=True)
        downsampled = out.getvalue()
        width, height = img.size

    sha = hashlib.sha256(downsampled).hexdigest()
    return PreparedImage(
        bytes_=downsampled,
        sha256=sha,
        width=width,
        height=height,
        byte_size=len(downsampled),
        mime_type=OUTPUT_MIME_TYPE,
        source_path=source_path,
    )


def prepare_image_for_labeling(
    image_path: str | Path,
    *,
    max_size: tuple[int, int] = (MAX_LONGEST_EDGE_PX, MAX_LONGEST_EDGE_PX),
    jpeg_quality: int = JPEG_QUALITY,
) -> PreparedImage:
    """Canonical entry point for provider clients.

    Thin wrapper that exposes ``max_size`` and ``jpeg_quality`` knobs for
    parity with the spec in ``docs/EXECUTION-PLAN-bulk-labeling-v1.md``
    (Pista correction). Provider clients (OpenAI, Anthropic, Gemini) MUST
    call this function — never read or encode original image bytes
    themselves.
    """
    path = Path(image_path)
    if not path.is_file():
        raise FileNotFoundError(f"image not found: {path}")
    with path.open("rb") as fh:
        raw = fh.read()
    return prepare_image_bytes(
        raw,
        source_path=str(path),
        max_size=max_size,
        jpeg_quality=jpeg_quality,
    )


__all__ = [
    "MAX_LONGEST_EDGE_PX",
    "JPEG_QUALITY",
    "OUTPUT_MIME_TYPE",
    "OUTPUT_FORMAT",
    "PreparedImage",
    "prepare_image",
    "prepare_image_bytes",
    "prepare_image_for_labeling",
]
