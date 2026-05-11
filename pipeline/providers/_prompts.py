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

The minimum justification length (1500 characters / roughly 350 tokens)
is enforced as a *soft* request in the prompt and as a *hard* validation
hook in :func:`coerce_label_fields`. Models that under-deliver still
parse successfully — but the run record flags ``justification_too_short``
so downstream scoring can downweight them.
"""

from __future__ import annotations


# --- Tunables ---------------------------------------------------------------

# Roughly 350 tokens at ~4.3 chars/token. We enforce a character floor
# because token counts vary per tokenizer and we don't want to embed a
# provider-specific tokenizer here.
MIN_JUSTIFICATION_CHARS: int = 1500

# Soft cap on policy_quotes (we want exact citations, not a re-writing of
# the policy book).
MAX_POLICY_QUOTES: int = 6


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
    "OUTPUT FORMAT — STRICT.\n"
    "Return EXACTLY one JSON object. No prose, no markdown fences. The "
    "object MUST contain these keys:\n"
    "  label             one of: gen_ai, not_gen_ai, abstain (cold-start) "
    "                    or violative, non_violative, abstain (warm-start)\n"
    "  l2_label          the policy node id you applied as the primary "
    "                    classification (e.g. \"GA.visual_artifacts."
    "anatomy.hands\"); empty string only if label=abstain\n"
    "  justification     a coherent argument grounded in the policy. MUST "
    "                    be at least ~350 tokens (≈1500+ characters). MUST "
    "                    name the policy nodes you invoked and quote the "
    "                    exact policy clauses you leaned on. Walk the "
    "                    reader through what you see in the image, why the "
    "                    policy applies (or does not), and how alternative "
    "                    policy nodes were considered and rejected.\n"
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
    "- A justification under ~1500 characters is a failure. If you are "
    "  about to emit a one-sentence justification, you are wrong. Re-read "
    "  the policy, find the relevant nodes, cite them.\n"
    "- Quote, do not paraphrase, when populating policy_quotes.\n"
    "- Cite at least one positive-evidence node (or one boundary/exception "
    "  node when abstaining). Never cite a node you didn't actually use.\n"
    "- If the evidence is mixed, walk through both sides in the "
    "  justification before committing to a label.\n"
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
    "Return ONE JSON object matching the schema in the system prompt. The "
    "justification must run at least ~350 tokens, must cite specific "
    "policy nodes by id (policy_citations), and must include verbatim "
    "policy quotes (policy_quotes) lifted from the markdown.\n"
    "\n"
    "Workflow you must follow inside your reasoning:\n"
    "  1. Inspect the image. Note observable features (anatomy, surface "
    "     texture, scene geometry, text/symbols, provenance hints).\n"
    "  2. For each observation, locate the policy node that governs it. "
    "     Record the node id verbatim.\n"
    "  3. Quote the exact policy clause that applies.\n"
    "  4. Test the leading hypothesis against boundary and exception "
    "     nodes; rule them in or out explicitly.\n"
    "  5. Commit to a label only after that walkthrough. If the policy "
    "     does not cover what you see, abstain.\n"
    "\n"
    "Do NOT include any text outside the JSON object. Do NOT wrap the "
    "JSON in markdown fences."
)


__all__ = [
    "LABELING_SYSTEM_PROMPT",
    "LABELING_USER_INSTRUCTIONS",
    "MIN_JUSTIFICATION_CHARS",
    "MAX_POLICY_QUOTES",
]
