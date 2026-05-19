# Architecture addendum — bulk labeling v1

> Companion to `EXECUTION-PLAN-bulk-labeling-v1.md`. Owner: X4.
> Status: in-flight (web + PDF surfaces wired; runner/scoring landed by X1–X3).

## 1. Data flow at a glance

```
                     +------------------------------+
                     |  policy-graph/Generative_AI/ |
                     |  v0.1/*.md   (SME-curated)   |
                     +---------------+--------------+
                                     |
                +--------------------+----------------------+
                |                                           |
                v                                           v
   pipeline.pdf.policy_pdf                       pipeline.providers.*
   (build_policy_pdf)                            (LabelClient ABC)
                |                                           |
                v                                           v
        web/policy.pdf  <--------- web/index.html --------> data/runs/<run_id>/
        data/runs/<run_id>/policy.pdf                       ├── label_votes.jsonl
                                                            ├── llm_outputs.jsonl
                                                            ├── errors.jsonl
                                                            ├── scoring/{decision_quality,misalignment,borderline}.json
                                                            ├── policy_patches.jsonl
                                                            ├── policy.pdf  (per-run snapshot)
                                                            └── web/{summary,borderline,misalignment}.json
```

The web UI fetches `data/runs/index.json` to discover runs, then loads
`data/runs/<run_id>/web/{summary,borderline,misalignment}.json` for the
selected run. None of those JSON files contain image bytes.

Batch labeling is real, not cosmetic: the runner groups same-model images into
logical provider batches (`batch_size`, default 20) and records both
`batch_size` and `effective_batches` in `run_manifest.json`. OpenAI uses one
multi-image provider request per batch; providers without a native multi-image
method use the compatibility `batch_label()` fallback, preserving input order
under the same per-provider concurrency semaphore.

## 2. Provider image preparation contract (visible in web copy)

Provider calls **always** downsample images before submission via a shared
helper (X1 owns the helper; runner calls it; web only consumes the audit
metadata). The contract surfaced in the UI is:

- Longest edge ≤ **1024 px**.
- JPEG quality ≈ **85**.
- Original image bytes are **never** embedded in any persisted JSON output.
- When present, optional audit metadata is recorded per image:
  ```json
  {
    "prepared_image": {
      "sha256": "<hex of the prepared bytes>",
      "width": 1024,
      "height": 768,
      "byte_size": 187432,
      "mime_type": "image/jpeg",
      "longest_edge": 1024,
      "jpeg_quality": 85
    }
  }
  ```
- The web Misalignment Worklist and Borderline Inspector both look for
  `prepared_image` on each row/item and surface a one-line summary so SMEs
  can verify the exact bytes the providers saw. Missing metadata renders
  as a blank line — no fake values, no crashes.

## 3. PDF builder

- Module: `pipeline/pdf/policy_pdf.py`.
- CLI:
  ```bash
  # demo build (committed link target)
  python scripts/build_policy_pdf.py \
      --source policy-graph/Generative_AI/v0.1 \
      --output web/policy.pdf

  # per-run snapshot (recommended once runner exists)
  python scripts/build_policy_pdf.py \
      --source policy-graph/Generative_AI/v0.1 \
      --output data/runs/<run_id>/policy.pdf
  ```
- The PDF is a build artifact; nothing is committed.
- Determinism: input files sorted (root first, then alphabetical). YAML
  frontmatter is parsed for the cover page metadata; only top-level scalars
  are used (we do not pull in PyYAML for one header block).
- Tests: `tests/test_pdf_builder.py` (11 cases, fully offline).

## 4. Web surfaces (X4)

| Section | Purpose | Source JSON |
|---------|---------|-------------|
| Hero CTA | Download bound policy PDF | `web/policy.pdf` (or per-run) |
| Runs picker | Select a run | `data/runs/index.json` |
| Run summary | Run id, models, image count, policy/prompt versions | `data/runs/<run_id>/web/summary.json` |
| Borderline Inspector | Items grouped by L0 (`gen_ai`/`not_gen_ai`/`abstain`) | `data/runs/<run_id>/web/borderline.json` |
| Misalignment Worklist | Per-image SME-vs-models table with patch links | `data/runs/<run_id>/web/misalignment.json` |

All three JSON shapes are tolerant: missing fields render an empty-state
row, never a crash. Thumbnails are path-based (no embedded bytes).

### Expected JSON shapes (consumed by app.js)

```jsonc
// data/runs/index.json
{ "runs": [ { "run_id": "20260510T172300-a1b2c3d4", "started_at": "2026-05-10T17:23:00Z" } ] }

// data/runs/<run_id>/web/summary.json
{
  "run_id": "...",
  "started_at": "...",
  "models": ["openai/gpt-5.5", "..."],
  "image_count": 100,
  "split": "dev_golden",
  "policy_graph_version": "v0.1",
  "prompt_version": "v1.0"
}

// data/runs/<run_id>/web/borderline.json
{
  "groups": [
    {
      "l0": "gen_ai",
      "items": [
        {
          "image_id": "...",
          "reason": "low confidence across models",
          "confidence": 0.51,
          "difficulty": "medium",
          "prepared_image": { "...": "see §2" }
        }
      ]
    }
  ]
}

// data/runs/<run_id>/web/misalignment.json
{
  "models": ["openai/gpt-5.5", "anthropic/claude-opus-4-6", "google/gemini-3.1-pro-preview"],
  "rows": [
    {
      "image_id": "...",
      "image_path": "data/images/.../foo.jpg",
      "sme_truth": "gen_ai",
      "model_labels": { "openai/gpt-5.5": "not_gen_ai", "...": "..." },
      "agreement": "split",
      "disagreement_reason": "model missed plastic-skin signal",
      "policy_patch_id": "PP-2026-05-10-007",
      "policy_patch_url": "...",
      "prepared_image": { "...": "see §2" }
    }
  ]
}
```

## 5. How to run locally

```bash
# 1. virtualenv + deps (X1 will own requirements.txt; for now, manual)
python3 -m venv .venv && source .venv/bin/activate
pip install reportlab pytest

# 2. tests (offline)
python -m pytest tests/ -v

# 3. baseline foundation validator
python scripts/validate_foundation.py

# 4. build the bound policy PDF
python scripts/build_policy_pdf.py \
    --source policy-graph/Generative_AI/v0.1 \
    --output web/policy.pdf

# 5. open the demo (any static server)
cd web && python3 -m http.server 8000
# then visit http://localhost:8000
```

The web UI works without any runs present — Borderline / Misalignment
sections render an empty state with the exact CLI to run next.

## 6. Out of scope for this addendum

- Provider client implementations (X1).
- Runner orchestration + persistence (X2).
- Scoring / policy-iterator logic (X3).
- The shared image-preparation helper itself (X1) — this addendum only
  documents the contract the web UI relies on.
