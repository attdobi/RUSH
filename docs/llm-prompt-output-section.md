# LLM labeling prompt — output format section

This Markdown section is included in the LLM prompt after the policy document and canonical examples. It defines the structured output the model must return.

---

## Output format

You must return a JSON object with exactly these fields:

```json
{
  "label": "<L0 decision>",
  "l2_label": "<subcategory node ID from the policy graph>",
  "justification": "<grounded justification citing specific policy sections>",
  "confidence": <float 0.0 to 1.0>,
  "difficulty": "<high | medium | low>",
  "is_boundary": <true | false>
}
```

### Field definitions

**label** — Your binary policy decision.
- For GenAI classification: `"gen_ai"` or `"not_gen_ai"`.
- For warm-start policies: `"violative"` or `"non_violative"`.

**l2_label** — The most specific subcategory node from the policy graph that applies. Use the `node_id` exactly as defined in the graph (e.g., `"GA.visual_artifacts.anatomy.hands"`). If the image is negative, use the most relevant boundary or negative node.

**justification** — Explain your decision by citing:
- Which policy section or criterion led to the decision.
- What visual evidence in the image supports or contradicts the label.
- If near a boundary, explain why you chose one side.
- Reference specific policy text, not vague summaries.

**confidence** — Your confidence in the label, from 0.0 (no confidence) to 1.0 (certain).
- Low confidence near boundaries is expected and useful — do not inflate.
- Confidence below 0.5 should typically accompany `"difficulty": "high"`.

**difficulty** — Your assessment of how hard this case is:
- `"low"` — Clear positive or clear negative. Policy unambiguous.
- `"medium"` — Requires some judgment but policy provides guidance.
- `"high"` — Near a decision boundary, policy is silent or ambiguous, or multiple subcategories could apply.

**is_boundary** — Set to `true` if the case sits near a decision boundary:
- Between positive and negative (hard positive or hard negative).
- Between two subcategories.
- Where the policy is silent, contradictory, or where you would want SME clarification.

### Important

- Do not guess or hallucinate policy references. If the policy does not cover the case, set `"difficulty": "high"`, `"is_boundary": true`, and explain what is missing in justification.
- Every field is required. Do not omit any field.
- `l2_label` must be a valid node_id from the policy graph provided in context.
