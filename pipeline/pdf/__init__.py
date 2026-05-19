"""Policy PDF builder for RUSH.

Generates a single bound PDF from a directory of policy-graph Markdown files.
The PDF is built from the on-disk policy version directory at request time;
no PDFs are committed to the repo (build artifact).
"""

from .node_examples import PolicyImageExample, collect_policy_image_examples
from .policy_pdf import (
    BuildResult,
    PolicyPdfError,
    build_policy_pdf,
    iter_policy_markdown,
    parse_frontmatter,
)

__all__ = [
    "BuildResult",
    "PolicyImageExample",
    "PolicyPdfError",
    "build_policy_pdf",
    "collect_policy_image_examples",
    "iter_policy_markdown",
    "parse_frontmatter",
]
