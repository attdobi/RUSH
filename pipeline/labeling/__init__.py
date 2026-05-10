"""Shared labeling helpers used by every provider client.

The :mod:`pipeline.labeling.image_prep` module is the single source of truth
for image downsampling. Provider clients MUST route every image through
:func:`pipeline.labeling.image_prep.prepare_image` before building a request
payload. This keeps token cost predictable across providers, ensures the same
bytes are hashed/recorded everywhere, and prevents accidental shipping of
multi-megabyte originals to LLM APIs.
"""

from pipeline.labeling.image_prep import (
    PreparedImage,
    prepare_image,
    prepare_image_bytes,
    prepare_image_for_labeling,
)

__all__ = [
    "PreparedImage",
    "prepare_image",
    "prepare_image_bytes",
    "prepare_image_for_labeling",
]
