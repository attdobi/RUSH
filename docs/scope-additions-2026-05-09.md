# RUSH scope additions — 2026-05-09

Attila's additions to the foundation scope, captured during initial build session.

## 1. Warm-start label hierarchy

For warm-start policies (existing policy document available), labels are fixed:

| Level | Name | Values | When it changes |
| --- | --- | --- | --- |
| L0 | Decision | violative / non_violative | Never (binary policy decision) |
| L1 | Enforcement | ignore / hide / deactivate | Only with policy owner approval |
| L2 | Subcategory | {set from policy graph nodes} | Only when SME accepts policy changes |

- Labels are **tied to the policy document** (.pdf or .md) which is fed into the LLM prompt alongside the image.
- For cold-start (GenAI pilot): L0 = gen_ai / not_gen_ai. L1 is deferred. L2 = subcategory from the graph.
- New subcategories only enter L2 when the SME approves the policy change that creates them.

## 2. LLM output format

Every LLM labeler must return structured output in this format:

```json
{
  "label": "violative",
  "l2_label": "GA.visual_artifacts.anatomy.hands",
  "justification": "Six fingers visible on left hand; policy §3.1 criterion 1 (digit-count violation).",
  "confidence": 0.82,
  "difficulty": "high",
  "is_boundary": true
}
```

| Field | Type | Purpose |
| --- | --- | --- |
| `label` | L0 enum | Binary policy decision |
| `l2_label` | node_id from policy graph | Subcategory placement |
| `justification` | string | Grounded in policy text with section references |
| `confidence` | float 0–1 | Model self-assessed confidence |
| `difficulty` | high / medium / low | Flags hard positives and negatives |
| `is_boundary` | boolean | Whether the case sits near a decision boundary |

`is_boundary` and `difficulty` are the **critical signals** for identifying where policy ambiguity lives. Hard positives and hard negatives at boundaries are the most important regions for reducing ambiguity.

A `.md` section in the prompt should describe this output format to the LLM, along with the policy document and any cached canonical examples.

## 3. Consensus-based ambiguity detection

- **Full consensus** (all labelers agree) → label is likely easy, policy is clear for this region.
- **Lack of consensus** → one of three causes:
  1. Policy needs clarification or is missing coverage for this case.
  2. Non-experts need more guidance and examples (context pack improvement).
  3. The expert was wrong (SME re-review needed).
- Majority vote of LLMs, non-experts, or LLM-SME misalignment on an image → **flag areas of ambiguity or missing policy areas**.

## 4. SME re-review sampling

At each iteration:
- Flag a sample of GDS images that would be **most beneficial** for an SME to re-review.
- Priority: high difficulty, boundary cases, LLM-SME misalignment, consensus audit cases, cases where overturning would have largest metric impact.
- If the SME overturns labels → **recompute all decision quality metrics** against the updated ground truth.
- The SME is the ultimate arbiter of truth, but high-reasoning model justifications grounded in policy can surface legitimate inconsistencies.

## 5. Decision Quality tab (new web UI tab)

A dedicated tab showing decision quality **given the policy version**:

**Rows (labelers):**
- GPT 5.4
- GPT 5.5
- GPT 5.5-high
- Gemini 3.1 Pro
- Majority vote (ensemble)
- Non-expert human

**Columns (metrics):**
- Accuracy
- F1
- Precision
- Recall
- FPR
- FNR
- Positive proportion
- N (images evaluated)
- Informedness

**Ground truth:** SME latest label for each image.

**Evolution chart:** at the bottom, showing metric evolution across policy versions (x-axis = policy version, y-axis = metric value, one line per labeler or selectable metric).

## 6. Next phase

Once the foundation is solid: image gathering and LLM labeling via API calls.
