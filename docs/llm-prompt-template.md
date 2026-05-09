# LLM Prompt Template for Policy-Graph Labelers

This template defines the structured contract for LLM labelers evaluating an image against a bound policy document and policy-graph context pack.

## 1. Input specification

Each labeling request MUST include:

- **Image**: the image to classify.
- **Policy document**: the authoritative policy text for this label version, supplied as `.pdf` or `.md`.
- **Graph context pack**: the relevant policy-graph excerpts for the image, including:
  - node IDs, titles, and node types;
  - decision rules and positive/negative criteria;
  - canonical examples when available;
  - exception, boundary, and confused-with warnings;
  - relevant source anchors into the policy document.

The labeler must treat the policy document as authoritative. The graph context pack helps choose the correct L2 placement, but it does not replace the policy text.

## 2. Output specification: six-field JSON

Return exactly one JSON object with these six fields:

```json
{
  "label": "violative|non_violative|abstain",
  "l2_label": "GA.visual_artifacts.anatomy.hands",
  "justification": "Six fingers visible on left hand, violating anatomical plausibility per policy §4.1",
  "confidence": 0.82,
  "difficulty": "high|medium|low",
  "is_boundary": true
}
```

No markdown, prose, or extra keys should be emitted outside the JSON object in production labeling runs.

## 3. Field explanations

- `label` — L0 binary policy decision. `violative` = policy violation detected. `non_violative` = no violation. `abstain` = insufficient evidence.
- `l2_label` — L2 policy-graph node ID where the image belongs. Must reference a valid node from the supplied graph context pack.
- `justification` — must cite specific policy text, criteria, or source anchors. "Looks AI-generated" is NOT sufficient.
- `confidence` — number from 0 to 1. Reflects certainty about both the L0 label and L2 graph placement.
- `difficulty` — labeler's self-assessment. `high` = boundary case or ambiguous. `medium` = some uncertainty. `low` = clear case.
- `is_boundary` — `true` if the image is a hard positive, hard negative, or sits at an edge between nodes. This is the MOST IMPORTANT signal for policy ambiguity reduction.

If evidence is insufficient, prefer `label: "abstain"`, choose the best applicable low-quality or uncertainty node for `l2_label`, set lower confidence, and explain what evidence is missing.

## 4. Consensus model

Use three independent labeler outputs for consensus analysis:

- **Full agreement** — 3/3 agreement on label and placement indicates an easy case and clear policy region.
- **Lack of consensus** — disagreement signals that policy may need clarification, non-experts may need guidance, or an expert judgment may be wrong.
- **Majority-vote disagreements** — cases with 2/3 agreement flag missing, underspecified, or ambiguous policy regions for SME review and policy-diff proposals.

Consensus is not only a quality gate; it is a discovery mechanism for policy ambiguity.

## 5. Example prompt skeleton

```text
You are a policy-graph image labeler. Classify the image using ONLY the supplied policy document and graph context pack.

[IMAGE]
{{image}}

[POLICY DOCUMENT]
{{policy_document_pdf_or_markdown}}

[GRAPH CONTEXT PACK]
Relevant nodes:
{{node_context}}

Decision rules:
{{decision_rules}}

Canonical examples:
{{canonical_examples}}

Boundary and exception warnings:
{{boundary_warnings}}

Instructions:
1. Decide the L0 policy label: violative, non_violative, or abstain.
2. Select the most specific valid L2 policy-graph node ID.
3. Cite specific policy text or criteria in the justification.
4. Estimate confidence from 0 to 1 for both label and placement.
5. Mark is_boundary=true for hard positives, hard negatives, exceptions, or ambiguous edge cases.
6. If evidence is insufficient, abstain. Do not guess.

Return exactly this JSON shape and no extra text:
{
  "label": "violative|non_violative|abstain",
  "l2_label": "<valid policy graph node id>",
  "justification": "<specific policy-grounded rationale>",
  "confidence": <number from 0 to 1>,
  "difficulty": "high|medium|low",
  "is_boundary": <true|false>
}
```

## 6. Reference

See `schemas/llm-output.schema.json` for the formal output contract.
