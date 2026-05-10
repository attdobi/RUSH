# Runbook — Bulk Labeling Pipeline v1 (X2)

> Operator guide for the `scripts/run_bulk_labeling.py` entry point.
> Branch: `feat/bulk-labeling-v1`. Pista is the only operator authorized to
> dispatch real provider calls; engineers wire it up but **do not** spend.

---

## 0. TL;DR

```bash
# Clean, offline dry run (no provider calls, no spend, no secrets needed):
python scripts/run_bulk_labeling.py \
    --models openai/gpt-5.5 \
    --split dev_golden --limit 5

# Print the plan only (no disk writes):
python scripts/run_bulk_labeling.py \
    --models openai/gpt-5.5,anthropic/claude-opus-4-6,google/gemini-3.1-pro-preview \
    --split dev_golden --plan-only
```

Outputs land under `data/runs/<run_id>/` (gitignored).

---

## 1. Prerequisites

* Python 3.11+ (developed on 3.14).
* `jsonschema` (4.x) — already required by `scripts/validate_foundation.py`.
* For real provider calls (X1's slice): `openai`, `anthropic`, `google-genai`,
  `Pillow`, plus secrets in `.env`. Engineers should never run live mode.

Verify schemas + foundation are sane before any run:

```bash
python scripts/validate_foundation.py
```

---

## 2. Image preparation (shared, every provider)

**Every provider call uses one shared helper**:
`pipeline/labeling/image_prep.py::prepare_image_for_labeling` (X1).

Cap: longest edge ≤ **1024 px**, re-encoded as **JPEG quality ≈ 85**, MIME
`image/jpeg`. The helper returns the prepared bytes plus this audit metadata,
which the runner persists into both `llm_outputs.jsonl` and `label_votes.jsonl`:

| Field                       | Type   | Purpose                                            |
|-----------------------------|--------|----------------------------------------------------|
| `prepared_image_sha256`     | string | SHA256 of the JPEG bytes actually sent             |
| `prepared_image_width`      | int    | Pixel width of the downsampled JPEG                |
| `prepared_image_height`     | int    | Pixel height of the downsampled JPEG               |
| `prepared_image_mime_type`  | string | Typically `image/jpeg`                             |
| `prepared_image_byte_size`  | int    | Byte length of the JPEG payload                    |

Why: deterministic, cross-provider audit of cost (token counts scale with
image size) and quality (every provider sees byte-identical input).

Run-level summary lives in `run_manifest.json::image_prep`:

```json
{
  "image_prep": {
    "longest_edge_px": 1024,
    "format": "JPEG",
    "quality": 85,
    "helper_module": "pipeline.labeling.image_prep"
  }
}
```

---

## 3. Environment variables

Loaded by `pipeline.providers.auth` (X1) from `.env`. **Never** pass via CLI
args. **Never** echo to logs.

| Variable           | Purpose                                |
|--------------------|----------------------------------------|
| `OPENAI_API_KEY`   | OpenAI Responses API                   |
| `ANTHROPIC_API_KEY`| Anthropic Messages API                 |
| `GEMINI_API_KEY`   | Google Generative AI                   |
| `RUSH_DOTENV_PATH` | Override path to `.env` (CI-friendly)  |

The runner itself is secret-free; it never touches these vars directly.

---

## 4. Inputs

* **Sample manifest:** `data/images/genai-classification/manifests/combined_labels.jsonl`
  (200 rows: 100 dev_golden + 100 holdout, balanced).
* **Policy bundle:** all `*.md` under `policy-graph/Generative_AI/v0.1/` are
  concatenated (in lexical order) and passed as `LabelRequest.policy_markdown`.
* **Split picker:** `--split dev_golden | holdout | all`. Holdout is hard-gated
  behind `--allow-holdout`; the runner refuses without it.

---

## 5. Outputs

```
data/runs/<run_id>/
├── run_manifest.json        # validated against schemas/run-manifest.schema.json
├── label_votes.jsonl        # validated against schemas/label-vote.schema.json
├── llm_outputs.jsonl        # each row: {image_id, model_id, recorded_at, output{...}}
├── errors.jsonl             # structured failures; never contains secrets
├── scoring/                 # X3 fills these (decision_quality / misalignment / borderline)
├── policy_patches.jsonl     # X3 writes; validated against policy-patch.schema.json
├── policy.pdf               # X4 builds on demand
└── web/                     # X3 exports; X4 reads
```

`run_id` format: `YYYYMMDDTHHMMSS-xxxxxxxx` (UTC + 8 hex).

---

## 6. Smoke test recipe (PISTA ONLY for live)

| Stage                | Command                                                                                          | Who    | Expected                          |
|----------------------|--------------------------------------------------------------------------------------------------|--------|-----------------------------------|
| Schemas + scaffold   | `python scripts/validate_foundation.py`                                                          | anyone | `passed`                          |
| Unit tests           | `python -m unittest discover tests -v` *(or `pytest tests/ -v` once X1 pins it)*                 | anyone | green                             |
| **Plan-only**        | `python scripts/run_bulk_labeling.py --models openai/gpt-5.5 --split dev_golden --limit 5 --plan-only` | anyone | JSON plan with `n_calls=5`        |
| Dry run 1×1          | `python scripts/run_bulk_labeling.py --models openai/gpt-5.5 --split dev_golden --limit 1`       | anyone | 1 row each in `label_votes.jsonl` and `llm_outputs.jsonl` |
| Dry run 5×3          | `python scripts/run_bulk_labeling.py --models openai/gpt-5.5,anthropic/claude-opus-4-6,google/gemini-3.1-pro-preview --split dev_golden --limit 5` | anyone | 15 rows each                      |
| **Live 1×1×1**       | add `--live --allow-spend`                                                                       | Pista  | requires X1's providers           |
| Full N=100 dev_golden | `--limit 100 --concurrency 4 --live --allow-spend`                                              | Pista  | gated on Attila approval          |
| Holdout              | `--split holdout --allow-holdout --live --allow-spend`                                          | Pista  | locked; never engineers           |

---

## 7. Hard guardrails baked in

* **No secrets in code or logs.** Keys load via `pipeline.providers.auth`
  from `.env` (X1).
* **No image bytes in JSON.** The runner never serialises image bytes; only
  the prepared-image audit metadata. `raw_provider_payload` is scrubbed by
  `pipeline.persistence._strip_image_bytes` as defence in depth.
* **Schema validation before write.** Every label vote, llm output, and run
  manifest is validated against its JSON Schema. Failures route to
  `errors.jsonl` with a structured reason; the run keeps going.
* **Determinism.** Sample ids and (sample, model) pairs are sorted before
  dispatch. Per-provider concurrency is capped at `--concurrency`.
* **Holdout lock.** `--split holdout` is rejected without `--allow-holdout`.
* **Dry-run by default.** Live calls require both `--live` AND `--allow-spend`.

---

## 8. Recovery / re-runs

A failed dry run is safe to re-run; each call mints a fresh `run_id` and
writes into a fresh directory. There's no in-place mutation of prior runs.

To clean local artefacts:

```bash
rm -rf data/runs/
```

(Already gitignored — nothing committed.)

---

## 9. Open hand-offs

* **X1:** publish `pipeline.providers.{base,registry,auth,retries,openai_client,anthropic_client,gemini_client}` and `pipeline.labeling.image_prep`.
* **X3:** read `data/runs/<run_id>/{label_votes.jsonl,llm_outputs.jsonl}`, write under `scoring/` and `web/`.
* **X4:** consume `web/summary.json`, render the PDF CTA, surface
  `prepared_image_*` audit metadata on misalignment / borderline detail cards.
