"""Shared, policy-grounded labeling prompts.

Single source of truth for the system + user instructions used by every
provider client (OpenAI, Anthropic, Gemini). Tightening these strings
in one place keeps the three providers strictly comparable: any
behavioural drift then comes from the model itself, not from prompt
differences.

The schema we ask the model to emit is intentionally larger than the
canonical six-field LLMOutput v1.1 — we also require ``policy_citations``
(list of node ids the labeler relied on) and ``policy_quotes`` (list of
exact policy clauses copied verbatim). These extra fields are coerced
in :func:`pipeline.providers.base.coerce_label_fields` and persisted on
:class:`pipeline.providers.base.LabelResponse` so reviewers can trace
**why** a label happened, not just what it was.

Justification length is treated as a CAP, not a floor: ~350 tokens
(≈1500 characters) is the soft upper bound. The prompt instructs models
to stay under it; :func:`coerce_label_fields` flags responses that
exceed it via ``justification_too_long`` so downstream scoring can spot
models that ignored the cap. Substance trumps length — a tight 200-token
justification that cites the right policy nodes is preferred over a
padded 350-token one.
"""

from __future__ import annotations


# --- Tunables ---------------------------------------------------------------

# Soft *upper* bound on justification length. ~350 tokens × ~4 chars/token.
# This is a cap, not a floor: we want dense, precise reasoning, not padding.
# Models may go under; the ≥10-char sanity check in coerce_label_fields is
# the only hard minimum. ``justification_too_long`` flags runaway output so
# scoring/reviewers can spot models that ignored the cap.
MAX_JUSTIFICATION_CHARS: int = 1500

# Soft cap on policy_quotes (we want exact citations, not a re-writing of
# the policy book).
MAX_POLICY_QUOTES: int = 6

LABELING_RESPONSE_KEYS: tuple[str, ...] = (
    "label",
    "l2_label",
    "justification",
    "policy_citations",
    "policy_quotes",
    "confidence",
    "difficulty",
    "is_boundary",
)

# Provider-facing JSON schema for APIs that support constrained JSON output.
# Keep this aligned with the eight fields required by LABELING_SYSTEM_PROMPT.
# It intentionally excludes server-populated audit fields such as token counts
# and prepared_image_* metadata.
LABELING_RESPONSE_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {
        "label": {
            "type": "string",
            "enum": [
                "gen_ai",
                "not_gen_ai",
                "abstain",
                "violative",
                "non_violative",
            ],
        },
        "l2_label": {"type": "string"},
        "justification": {"type": "string", "min_length": 10},
        "policy_citations": {
            "type": "array",
            "items": {"type": "string"},
        },
        "policy_quotes": {
            "type": "array",
            "items": {"type": "string", "max_length": 600},
            "max_items": MAX_POLICY_QUOTES,
        },
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "difficulty": {"type": "string", "enum": ["high", "medium", "low"]},
        "is_boundary": {"type": "boolean"},
    },
    "required": list(LABELING_RESPONSE_KEYS),
    "property_ordering": list(LABELING_RESPONSE_KEYS),
}


# --- Prompt strings ---------------------------------------------------------

LABELING_SYSTEM_PROMPT: str = (
    "You are RUSH's policy-graph image labeler.\n"
    "\n"
    "Your job is to classify a single image using ONLY the supplied policy "
    "document — the markdown bundle delimited by [POLICY DOCUMENT] markers. "
    "Treat that bundle as authoritative; do not invent rules, do not import "
    "outside priors about generative imagery, do not rely on metadata or "
    "filenames. If the supplied policy is silent on something you observe, "
    "say so explicitly and abstain.\n"
    "\n"
    "REASONING vs OUTPUT.\n"
    "Think hard internally. Your VISIBLE output is short and dense. The "
    "reasoning effort / thinking budget your runtime gives you is for the "
    "hidden deliberation pass; the JSON you return is the receipt.\n"
    "\n"
    "OUTPUT FORMAT — STRICT.\n"
    "Return EXACTLY one JSON object. No prose, no markdown fences. The "
    "object MUST contain these keys:\n"
    "  label             one of: gen_ai, not_gen_ai, abstain (cold-start) "
    "                    or violative, non_violative, abstain (warm-start)\n"
    "  l2_label          the policy node id you applied as the primary "
    "                    classification (e.g. \"GA.visual_artifacts."
    "anatomy.hands\"); empty string only if label=abstain\n"
    "  justification     a precise, dense argument grounded in the policy. "
    "                    Keep it UNDER ~350 tokens (≈1500 characters). No "
    "                    padding, no restating the image, no narrative "
    "                    flourishes. Name the policy nodes you invoked "
    "                    inline and reference the clauses you leaned on. "
    "                    Substance over length.\n"
    "  policy_citations  array of policy node ids referenced (e.g. "
    "                    [\"GA.surface_texture.plastic_skin\", "
    "                    \"GA.visual_artifacts.text_symbols\"]). MUST "
    "                    include at least the l2_label node when label is "
    "                    not abstain. Use [] only when label=abstain.\n"
    "  policy_quotes     array of short verbatim snippets (≤240 chars "
    "                    each) lifted directly from the policy markdown. "
    "                    Each entry must appear character-for-character in "
    "                    the supplied policy text. 1–6 entries.\n"
    "  confidence        number in [0, 1]. 0.5 means a coin-flip; reserve "
    "                    >0.9 for unambiguous calls.\n"
    "  difficulty        one of high, medium, low — how hard the call was.\n"
    "  is_boundary       true if the image lives on a documented boundary "
    "                    or exception node (e.g. CGI/game render, edited "
    "                    photo, low-quality uncertain).\n"
    "\n"
    "QUALITY BAR.\n"
    "- Substance, not word count. A 200-token justification that names the "
    "  right nodes and quotes the right clauses beats a 350-token "
    "  justification that meanders.\n"
    "- Quote, do not paraphrase, when populating policy_quotes.\n"
    "- Cite at least one positive-evidence node (or one boundary/exception "
    "  node when abstaining). Never cite a node you didn't actually use.\n"
    "- If the evidence is mixed, name the competing nodes in the "
    "  justification and explain why one wins.\n"
    "\n"
    "FAILURE MODE — ABSTAIN.\n"
    "If the image is unreadable, ambiguous, off-policy, or the supplied "
    "policy does not cover it, set label=abstain, l2_label=\"\", and "
    "explain in the justification which clauses you searched and why none "
    "fit. abstain is a legitimate answer; bad guesses are not."
)


LABELING_USER_INSTRUCTIONS: str = (
    "Classify this image against the policy document below.\n"
    "\n"
    "Return ONE JSON object matching the schema in the system prompt. "
    "Keep justification under ~350 tokens; cite specific policy nodes by "
    "id (policy_citations); include verbatim policy quotes (policy_quotes) "
    "lifted from the markdown.\n"
    "\n"
    "Workflow to follow inside your reasoning (this part is hidden — only "
    "the final JSON is returned):\n"
    "  1. Inspect the image. Note observable features (anatomy, surface "
    "     texture, scene geometry, text/symbols, provenance hints).\n"
    "  2. For each observation, locate the policy node that governs it. "
    "     Record the node id verbatim.\n"
    "  3. Identify the exact policy clause that applies.\n"
    "  4. Test the leading hypothesis against boundary and exception "
    "     nodes; rule them in or out.\n"
    "  5. Commit to a label. If the policy does not cover what you see, "
    "     abstain.\n"
    "\n"
    "Do NOT include any text outside the JSON object. Do NOT wrap the "
    "JSON in markdown fences."
)


__all__ = [
    "LABELING_SYSTEM_PROMPT",
    "LABELING_USER_INSTRUCTIONS",
    "LABELING_RESPONSE_KEYS",
    "LABELING_RESPONSE_SCHEMA",
    "MAX_JUSTIFICATION_CHARS",
    "MAX_POLICY_QUOTES",
]
