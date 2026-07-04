"""Per-project (per demo-area) labeling ONTOLOGY.

Single source of truth per demo area. Each :class:`Ontology` declares the L1
classes (measured for accuracy), the L2 semantics, the boundary semantics, the
provider-facing response schema, and the domain-specific SYSTEM + USER prompt
fragments. The prompt/schema builder in every provider client selects by area
so the three providers stay strictly comparable *within* a project.

Areas (see :mod:`pipeline.web.demo_area`):
  * ``Generative_AI`` — regression-sensitive baseline; preserves the exact
    GenAI prompt copy + schema shipped before ontologies existed.
  * ``MNIST_Digits`` — multiclass digit labeling; the digit is the L1 ``label``.

Backward compatibility: :func:`get_ontology` defaults to GenAI, and the GenAI
ontology re-exports the historical strings from :mod:`pipeline.providers._prompts`
verbatim, so existing callers/tests are unchanged.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any

from pipeline.providers._prompts import (
    LABELING_RESPONSE_KEYS,
    LABELING_RESPONSE_SCHEMA,
    LABELING_SYSTEM_PROMPT,
    LABELING_USER_INSTRUCTIONS,
)
from pipeline.web.demo_area import (
    DEFAULT_POLICY_AREA,
    MNIST_POLICY_AREA,
    normalize_policy_area,
)

# The 10 MNIST digit classes (L1) as strings, ordered.
MNIST_DIGITS: tuple[str, ...] = tuple(str(d) for d in range(10))


@dataclass(frozen=True)
class Ontology:
    """Per-area declaration driving prompt + schema + scoring selection."""

    area: str
    l1_classes: tuple[str, ...]
    label_enum: tuple[str, ...]
    scoring_task: str
    l2_semantics: str
    boundary_semantics: str
    require_boundary_between: bool
    system_prompt: str
    user_instructions: str
    response_keys: tuple[str, ...]
    response_schema: dict[str, Any]
    abstain_label: str = "abstain"

    def schema_copy(self) -> dict[str, Any]:
        """Deep copy of the response schema (providers mutate their own copy)."""
        return copy.deepcopy(self.response_schema)


# ---------------------------------------------------------------------------
# Shared optional field: is_boundary_between
# ---------------------------------------------------------------------------
# An array of exactly TWO L1 class ids naming the boundary pair. Required when
# is_boundary=true for areas with require_boundary_between (MNIST); optional for
# GenAI (L2-level parity). Validation lives in
# :func:`pipeline.providers.base.coerce_label_fields`.
_IS_BOUNDARY_BETWEEN_SCHEMA: dict[str, Any] = {
    "type": "array",
    "items": {"type": "string"},
    "maxItems": 2,
    "description": (
        "Exactly two L1 class ids naming the boundary pair, e.g. [\"1\",\"7\"]. "
        "Required when is_boundary=true (MNIST); empty/absent otherwise."
    ),
}


# ---------------------------------------------------------------------------
# GenAI ontology (baseline — preserve behavior exactly)
# ---------------------------------------------------------------------------

_GENAI_SCHEMA = copy.deepcopy(LABELING_RESPONSE_SCHEMA)
# is_boundary_between is OPTIONAL for GenAI (parity, not required): add the
# property but do NOT add it to `required`.
_GENAI_SCHEMA["properties"]["is_boundary_between"] = copy.deepcopy(
    _IS_BOUNDARY_BETWEEN_SCHEMA
)

GENAI_ONTOLOGY = Ontology(
    area=DEFAULT_POLICY_AREA,
    l1_classes=("gen_ai", "not_gen_ai"),
    label_enum=("gen_ai", "not_gen_ai", "abstain", "violative", "non_violative"),
    scoring_task="genai_binary",
    l2_semantics=(
        "policy subcategory node id applied as the primary classification "
        "(e.g. GA.visual_artifacts.anatomy.hands)"
    ),
    boundary_semantics=(
        "true when the image lives on a documented boundary/exception node "
        "(CGI/game render, edited photo, low-quality uncertain)"
    ),
    require_boundary_between=False,
    system_prompt=LABELING_SYSTEM_PROMPT,
    user_instructions=LABELING_USER_INSTRUCTIONS,
    response_keys=LABELING_RESPONSE_KEYS,
    response_schema=_GENAI_SCHEMA,
)


# ---------------------------------------------------------------------------
# MNIST ontology (new)
# ---------------------------------------------------------------------------

MNIST_RESPONSE_KEYS: tuple[str, ...] = (
    "label",
    "l2_label",
    "justification",
    "policy_citations",
    "policy_quotes",
    "confidence",
    "difficulty",
    "is_boundary",
    "is_boundary_between",
)

_MNIST_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "label": {
            "type": "string",
            "enum": list(MNIST_DIGITS) + ["abstain"],
        },
        "l2_label": {"type": "string"},
        "justification": {"type": "string", "min_length": 10},
        "policy_citations": {"type": "array", "items": {"type": "string"}},
        "policy_quotes": {
            "type": "array",
            "items": {"type": "string", "max_length": 600},
            "max_items": 6,
        },
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "difficulty": {"type": "string", "enum": ["high", "medium", "low"]},
        "is_boundary": {"type": "boolean"},
        "is_boundary_between": copy.deepcopy(_IS_BOUNDARY_BETWEEN_SCHEMA),
    },
    # is_boundary_between is conditionally required (only when is_boundary=true);
    # JSON-schema `required` lists the always-present keys. The exactly-two rule
    # is enforced in coerce_label_fields (providers vary in if/then support).
    "required": list(MNIST_RESPONSE_KEYS[:-1]),
    "property_ordering": list(MNIST_RESPONSE_KEYS),
}

MNIST_SYSTEM_PROMPT: str = (
    "You are RUSH's policy-graph handwritten-digit labeler.\n"
    "\n"
    "Your job is to classify a single image of ONE handwritten digit (0-9) "
    "using ONLY the supplied policy document — the markdown bundle delimited "
    "by [POLICY DOCUMENT] markers. Treat that bundle as authoritative; do not "
    "invent rules, do not rely on metadata or filenames. Judge by STROKE "
    "SHAPE and TOPOLOGY: count and orientation of strokes, presence/absence of "
    "loops and enclosed regions, horizontal bars, bowls, and where strokes "
    "cross. If the mark is unreadable or the policy is silent, abstain.\n"
    "\n"
    "REASONING vs OUTPUT.\n"
    "Think hard internally. Your VISIBLE output is short and dense JSON.\n"
    "\n"
    "OUTPUT FORMAT — STRICT.\n"
    "Return EXACTLY one JSON object. No prose, no markdown fences. Keys:\n"
    "  label               the digit you read: one of \"0\"..\"9\", or "
    "\"abstain\". THIS is the primary classification (accuracy is measured "
    "here).\n"
    "  l2_label            the policy node id you applied (e.g. "
    "\"MD.digit.7\"); empty string only if label=abstain.\n"
    "  justification       a precise, dense argument grounded in the policy. "
    "HARD CAP: <= 300 words (~400 tokens, ≈1500 chars). Name the stroke/topology features and "
    "the policy nodes you invoked. Substance over length.\n"
    "  policy_citations    array of policy node ids referenced (e.g. "
    "[\"MD.digit.1\", \"MD.digit.7\"]). MUST include the l2_label node when "
    "label is not abstain. Use [] only when label=abstain.\n"
    "  policy_quotes       array of short verbatim snippets (≤240 chars each) "
    "lifted directly from the policy markdown. 1-6 entries.\n"
    "  confidence          number in [0, 1]. Reserve >0.9 for unambiguous "
    "digits.\n"
    "  difficulty          one of high, medium, low.\n"
    "  is_boundary         true if the mark sits on a documented confusion "
    "boundary — it plausibly reads as either of TWO digits (classic pairs: "
    "1↔7, 4↔9, 3↔5, 3↔8, 5↔6, 7↔9, 2↔7). Otherwise false.\n"
    "  is_boundary_between when is_boundary=true, an array of EXACTLY TWO "
    "digit ids naming the confusion pair, e.g. [\"1\",\"7\"]; when "
    "is_boundary=false, use [] (empty).\n"
    "\n"
    "QUALITY BAR.\n"
    "- Decide the single most likely digit for `label` even when you also mark "
    "  a boundary pair — `label` is your best single call; `is_boundary_between`"
    "  records the competing alternative.\n"
    "- Quote, do not paraphrase, when populating policy_quotes.\n"
    "- Never cite a node you didn't use.\n"
    "\n"
    "FAILURE MODE — ABSTAIN.\n"
    "If the image is unreadable, empty, contains multiple digits, or the "
    "policy does not cover it, set label=abstain, l2_label=\"\", "
    "is_boundary_between=[], and explain why. abstain is legitimate; bad "
    "guesses are not."
)

MNIST_USER_INSTRUCTIONS: str = (
    "Classify the single handwritten digit in this image against the policy "
    "document below.\n"
    "\n"
    "Return ONE JSON object matching the schema in the system prompt. Keep the "
    "justification to a HARD CAP of <= 300 words (~400 tokens). Put the "
    "digit you read in `label` (\"0\"..\"9\" or \"abstain\"). Put the policy "
    "node id in `l2_label` (MD.digit.N). Cite policy nodes (policy_citations) "
    "and include verbatim policy quotes (policy_quotes).\n"
    "\n"
    "Workflow inside your reasoning (hidden — only the final JSON is "
    "returned):\n"
    "  1. Trace the strokes: count them, note orientation, loops, enclosed "
    "     regions, horizontal bars, bowls, crossings.\n"
    "  2. Map those features to the digit policy node that governs them.\n"
    "  3. Check the documented confusion pairs; if the mark plausibly reads as "
    "     two digits, set is_boundary=true and name the pair in "
    "     is_boundary_between.\n"
    "  4. Commit to the single most likely digit as `label`.\n"
    "\n"
    "Do NOT include any text outside the JSON object. Do NOT wrap the JSON in "
    "markdown fences."
)

MNIST_ONTOLOGY = Ontology(
    area=MNIST_POLICY_AREA,
    l1_classes=MNIST_DIGITS,
    label_enum=MNIST_DIGITS + ("abstain",),
    scoring_task="mnist_multiclass",
    l2_semantics="policy digit node id applied (MD.digit.N)",
    boundary_semantics=(
        "true when the mark sits on a documented digit-confusion boundary "
        "(reads plausibly as either of two digits)"
    ),
    require_boundary_between=True,
    system_prompt=MNIST_SYSTEM_PROMPT,
    user_instructions=MNIST_USER_INSTRUCTIONS,
    response_keys=MNIST_RESPONSE_KEYS,
    response_schema=_MNIST_SCHEMA,
)


# ---------------------------------------------------------------------------
# Registry + selection
# ---------------------------------------------------------------------------

_ONTOLOGIES: dict[str, Ontology] = {
    GENAI_ONTOLOGY.area: GENAI_ONTOLOGY,
    MNIST_ONTOLOGY.area: MNIST_ONTOLOGY,
}


def get_ontology(area: str | None = None, *, demo: str | None = None) -> Ontology:
    """Return the :class:`Ontology` for a demo area (or demo alias).

    Defaults to the GenAI baseline when ``area``/``demo`` are unset, preserving
    backward compatibility for callers that never learned about ontologies.
    """
    resolved = normalize_policy_area(area, demo=demo)
    try:
        return _ONTOLOGIES[resolved]
    except KeyError as exc:  # pragma: no cover - normalize guards the set
        raise ValueError(f"no ontology registered for area {resolved!r}") from exc


def available_ontologies() -> tuple[str, ...]:
    return tuple(sorted(_ONTOLOGIES))


__all__ = [
    "Ontology",
    "MNIST_DIGITS",
    "GENAI_ONTOLOGY",
    "MNIST_ONTOLOGY",
    "MNIST_SYSTEM_PROMPT",
    "MNIST_USER_INSTRUCTIONS",
    "MNIST_RESPONSE_KEYS",
    "get_ontology",
    "available_ontologies",
]
