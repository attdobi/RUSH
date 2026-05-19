# RUSH Local Web API Contract (v1)

**Status:** authored by Theo (tech lead) on 2026-05-10 for the
`feat/web-runner-policy-insights` wave. All engineers code to this contract.

**Server:** `scripts/rush_web_server.py` — stdlib `ThreadingHTTPServer`,
binds **127.0.0.1 only**. Serves repo root as static files (so existing
`/web/*` URLs work). Routes any path starting with `/api/` to JSON handlers.

**Conventions**
- All requests/responses are JSON (UTF-8). Static is unchanged.
- Errors: `{"error": {"code": "...", "message": "...", "details": {...}}}` + HTTP 4xx/5xx.
- Timestamps: ISO-8601 UTC (`2026-05-10T22:30:00Z`).
- `run_id` matches `data/runs/<run_id>` directory name; regex `^[0-9]{8}T[0-9]{6}-[0-9a-f]{8}$`.
- `policy_version` regex `^v[0-9]+(\.[0-9]+)?$` (e.g. `v0.1`, `v0.2`).
- `model_id` MUST be in `pipeline.providers.registry.MODEL_REGISTRY`.
- No live provider calls in tests. Use injected fakes.

---

## 1. Health & catalog (X1)

### `GET /api/health`
Response:
```json
{
  "status": "ok",
  "server_version": "rush-web-server-v1",
  "started_at": "2026-05-10T22:30:00Z",
  "repo_root": "/Users/sacsimoto/GitHub/RUSH"
}
```

### `GET /api/runs`
Lists discovered runs, newest first.
```json
{
  "runs": [
    {
      "run_id": "20260510T230535-6a71939d",
      "started_at": "2026-05-10T23:05:35Z",
      "finished_at": "2026-05-10T23:08:12Z",
      "split": "dev_golden",
      "policy_graph_version": "v0.1",
      "models": ["anthropic/claude-opus-4-6", "google/gemini-3.1-pro-preview", "openai/gpt-5.5"],
      "totals": {"expected_calls": 30, "completed_calls": 30, "errored_calls": 0},
      "scoring_done": true,
      "running": false
    }
  ]
}
```

---

## 2. Run lifecycle (X1)

### `POST /api/runs/start`
Body (all fields validated):
```json
{
  "models": ["openai/gpt-5.5", "anthropic/claude-opus-4-6"],
  "split": "dev_golden",
  "limit": 10,
  "sample_ids": null,
  "policy_version": "v0.1",
  "mode": "cold_start",
  "allow_spend": true,
  "concurrency": 1
}
```
- Either `limit` OR `sample_ids` (CSV-style list) — not both.
- `allow_spend: true` is required for any non-dry-run job.
- `mode` is recorded in the job manifest; runner currently treats both modes
  identically (no warm-start logic in `run_bulk_labeling.py` yet — record the
  intent and pass through).
Response:
```json
{
  "run_id": "20260510T230600-abcdef12",
  "status_url": "/api/runs/20260510T230600-abcdef12/status",
  "log_url": "/api/runs/20260510T230600-abcdef12/log"
}
```
- Implementation: `subprocess.Popen` with explicit argv (NO shell=True).
  Pre-allocate the run_id by importing `pipeline.manifest.new_run_id()` (or
  by mirroring the format) and pass `--run-id <id>` only if the runner
  supports it; otherwise spawn the runner and capture the auto-generated
  run_id from its stdout JSON. **Tip:** simplest path is to spawn with
  `python -u scripts/run_bulk_labeling.py …`, pipe stdout into a temp
  buffer/log, and once the JSON summary line is emitted at the end, parse
  `run_id`. For live progress before completion, X1 should return a
  preliminary job-id and let `/status` resolve the run_id once the runner
  has created `data/runs/<run_id>/run_manifest.json`.
- Persist job state to `data/runs/_jobs/<job_id>.json` with: pid,
  argv, started_at, model list, etc.

### `GET /api/runs/<run_id>/status`
```json
{
  "run_id": "20260510T230600-abcdef12",
  "running": true,
  "started_at": "2026-05-10T23:06:00Z",
  "finished_at": null,
  "expected_calls": 30,
  "completed_calls": 12,
  "errored_calls": 0,
  "progress": 0.4,
  "scoring_done": false,
  "log_tail": ["...last ~40 lines..."]
}
```
Reads progress from `label_votes.jsonl` line count vs `run_manifest.json`'s
`totals.expected_calls`.

### `POST /api/runs/<run_id>/score`
Triggers `scripts/score_labels.py --run-id <id>` if not already done.
Returns 200 with summary or 409 if already scored.

---

## 3. Decision quality + insights (X2)

### `GET /api/decision-quality`
Query: `?run_id=<id>` and/or `?policy_version=<v>` (both optional).
- No params → aggregate across all scored runs, grouped by run_id and
  policy_version, ordered by run started_at.
```json
{
  "runs": [
    {
      "run_id": "...",
      "started_at": "...",
      "policy_graph_version": "Generative_AI.v0.1",
      "n_images": 10,
      "labelers": [
        {"labeler_id": "openai/gpt-5.5", "labeler_type": "llm",
         "metrics": {"accuracy": 0.7, "precision": 0.8, "recall": 0.6,
                     "f1": 0.69, "fpr": 0.1, "fnr": 0.4, "n": 10}}
      ],
      "majority_vote": {"accuracy": 0.8, "n": 10},
      "consensus_summary": {"unanimous": 6, "split": 4},
      "boundary_rate": 0.3
    }
  ],
  "policy_versions": ["Generative_AI.v0.1"]
}
```

### `GET /api/insights`
Query: `?run_id=<id>` (required).
```json
{
  "run_id": "...",
  "majority_wrong": [
    {"image_id": "...", "sme_truth": "ai_generated",
     "majority_label": "not_ai_generated",
     "votes": [{"labeler_id":"...","label":"..."}, ...]}
  ],
  "model_disagreement": [
    {"image_id": "...",
     "votes": [{"labeler_id":"openai/gpt-5.5","label":"gen_ai"},
               {"labeler_id":"google/gemini-3.1-pro-preview","label":"not_gen_ai"},
               {"labeler_id":"anthropic/claude-opus-4-6","label":"abstain"}]}
  ],
  "boundary_concentration": [
    {"l0_bucket": "abstain", "n_images": 4,
     "top_l2_nodes": ["GA.boundary.low_quality_uncertain"]}
  ],
  "consistent_pair_disagreement": [
    {"pair": ["openai/gpt-5.5", "google/gemini-3.1-pro-preview"],
     "n_disagreements": 3, "fraction": 0.3}
  ]
}
```

---

## 4. Policy versions + proposals (X3)

### `GET /api/policy/versions`
```json
{
  "domain": "Generative_AI",
  "versions": [
    {"version": "v0.1", "files": 13, "path": "policy-graph/Generative_AI/v0.1"}
  ],
  "current": "v0.1"
}
```

### `POST /api/policy/propose-diff`
Body:
```json
{
  "run_id": "20260510T230535-6a71939d",
  "base_version": "v0.1",
  "model_id": "anthropic/claude-opus-4-7"
}
```
- **Default model: `anthropic/claude-opus-4-7`.** Only also accept
  `openai/gpt-5.5` (high reasoning) when explicitly requested.
- Calls Claude through the Anthropic SDK (lazy import).
- DOES NOT write a new policy version. Stores the proposal under
  `data/policy_proposals/<proposal_id>/`:
  - `proposal.json` — metadata, base_version, model_id, run_id, created_at,
    status: "pending", per-file change list.
  - `proposed/<filename>.md` — proposed file contents (full file, not a
    diff string — diffs are computed on-the-fly server-side).
  - `prompt.json` — for reproducibility.
Response:
```json
{
  "proposal_id": "20260510T231500-cafef00d",
  "base_version": "v0.1",
  "model_id": "anthropic/claude-opus-4-7",
  "files_changed": ["GA.boundary.low_quality_uncertain.md", "GA.root.md"],
  "files_added": ["GA.boundary.over_smoothed_skin.md"],
  "files_removed": [],
  "status": "pending"
}
```

### `GET /api/policy/proposals`
Lists proposals with status (pending/accepted/rejected).

### `GET /api/policy/proposals/<proposal_id>`
Returns proposal metadata + per-file diff in unified-diff text:
```json
{
  "proposal_id": "...",
  "base_version": "v0.1",
  "status": "pending",
  "diffs": [
    {"path": "GA.root.md", "change": "modified",
     "unified_diff": "--- a/GA.root.md\n+++ b/GA.root.md\n@@ -3,4 +3,5 @@\n line\n-old\n+new\n",
     "before": "...", "after": "..."}
  ]
}
```

### `POST /api/policy/proposals/<proposal_id>/accept`
- Computes next version (e.g. `v0.1` → `v0.2`).
- Creates `policy-graph/Generative_AI/v0.2/`.
- Copies the **base version** files unchanged, then overlays the accepted
  proposed files (and accepted additions/removals).
- Writes `proposal.json.status = "accepted"` and
  `accepted_into_version: "v0.2"`.
- Returns `{"new_version": "v0.2", "path": "..."}`.

### `POST /api/policy/proposals/<proposal_id>/reject`
- Moves the proposal directory under
  `data/policy_proposals/_archive/<proposal_id>/` and updates status.

## 4.5. Policy growth (X1 — Phase 2c)

Two additive proposal endpoints. Both write into the existing
`data/policy_proposals/<proposal_id>/` shape, so the existing
`GET /api/policy/proposals/<id>`, `POST .../accept`, and `POST .../reject`
endpoints work unchanged. New additive metadata fields in `proposal.json`:
`kind` (`"cold_start"` | `"grow_batch"` | absent for legacy `propose_diff`
proposals), and for `grow_batch` proposals also `batch_index` and `batch`.

### `POST /api/policy/cold-start`
Seed a brand-new policy graph from a free-form task description. There is
no base version; the proposal contains only `files_added`. Accepting the
proposal creates `policy-graph/Generative_AI/v0.1/` (or the next free
`vN.M`).

Body:
```json
{
  "task_description": "Classify whether an image is AI-generated...",
  "domain": "Generative_AI",
  "model_id": "openai/gpt-5.5"
}
```
- `task_description` (required, non-empty string, truncated to 2000 chars
  in stored metadata).
- `domain` optional, defaults to `"Generative_AI"`. Only that domain is
  supported today; anything else returns 400.
- `model_id` optional; defaults to `DEFAULT_POLICY_MODEL` (`openai/gpt-5.5`).
  Must be a member of `ALLOWED_POLICY_MODELS`.

Response 200:
```json
{
  "proposal_id": "20260518T230000-abc12345",
  "kind": "cold_start",
  "domain": "Generative_AI",
  "base_version": null,
  "task_description": "Classify whether an image is AI-generated...",
  "model_id": "openai/gpt-5.5",
  "created_at": "2026-05-18T23:00:00Z",
  "status": "pending",
  "files_changed": [],
  "files_added": ["GA.root.md", "GA.visual_artifacts.md"],
  "files_removed": []
}
```
Errors: 400 (invalid/empty `task_description`, unsupported `domain`),
422 (`status: "parse_error"` — LLM returned malformed JSON, the proposal
is still persisted under `data/policy_proposals/<id>/raw_response.txt`),
500 other.

### `POST /api/policy/grow-batch`
Grow an existing policy graph from a stratified 50/50 batch of
SME-labeled misclassifications.

Body:
```json
{
  "run_id": "20260518T180000-abcdef01",
  "base_version": "v0.1",
  "batch_index": 0,
  "batch_size": 50,
  "model_id": "openai/gpt-5.5"
}
```
- `run_id` (required, non-empty string).
- `base_version` (required, matches `^v\d+\.\d+$`).
- `batch_index` (required int, `>= 0`).
- `batch_size` (required int, `>= 2`).
- `model_id` optional; same allowed set as `cold-start`.

**Stratification:** the misalignment records are split by `sme_truth` into
positives (`"gen_ai"`) and negatives (`"not_gen_ai"`), each sorted by
`image_id` for reproducibility. With `half = batch_size // 2`, the handler
takes `positives[batch_index*half : (batch_index+1)*half]` and the same
slice of negatives. If one class is exhausted, the remainder is filled
from the other class's leftover rows (no wrap, no repeat). `batch.batch_size_actual`
reports the actual rows assembled, which may be less than
`batch_size_requested`.

Response 200:
```json
{
  "proposal_id": "20260518T231500-def67890",
  "kind": "grow_batch",
  "base_version": "v0.1",
  "batch_index": 0,
  "batch": {
    "batch_size_requested": 50,
    "batch_size_actual": 48,
    "n_positives": 24,
    "n_negatives": 24,
    "sme_truth_positive_label": "gen_ai",
    "sme_truth_negative_label": "not_gen_ai"
  },
  "run_id": "20260518T180000-abcdef01",
  "model_id": "openai/gpt-5.5",
  "created_at": "2026-05-18T23:15:00Z",
  "status": "pending",
  "files_changed": ["GA.visual_artifacts.md"],
  "files_added": ["GA.visual_artifacts.eyes.md"],
  "files_removed": []
}
```
Errors: 400 (invalid body, missing required fields, malformed version,
batch bounds violated), 404 (unknown `run_id` / `base_version` /
missing `data/runs/<run_id>/scoring/misalignment.json` with no auto-score
fallback), 422 (`status: "parse_error"`), 500 other.

### `POST /api/policy/build-pdf`
Body: `{"version": "v0.1", "model_id": "anthropic/claude-opus-4-7"}`
- Calls `scripts/build_policy_pdf.py` to produce
  `policy-graph/Generative_AI/<version>/policy.pdf` (or `web/policy.pdf`).
- The model param is reserved for future LLM-assisted formatting; for now
  the PDF is built deterministically from MD files. **Critical: never
  overwrites MD files.**

---

## 5. Frontend usage

- Static fetch paths from `/web/`: keep using `../data/...` for compatibility
  with both plain `python -m http.server` and `rush_web_server.py`.
- API detection: `fetch('/api/health')` once at init; if 200, expose
  run-trigger UI, otherwise hide it with a "local server not running" hint.
- Image src: prefer `'../' + repo_rel_path` (works under both servers since
  `/web/` is one level below repo root).

---

## 6. Safety & validation rules

- Bind 127.0.0.1 only. Refuse to start if `--bind` is anything else.
- Validate every path against `repo_root` after `Path.resolve()` to block
  traversal.
- Reject `policy_version` not present under `policy-graph/Generative_AI/`.
- No secrets in logs. The runner subprocess inherits env from the parent;
  do not echo env values.
- All subprocess.Popen calls: `shell=False`, explicit argv list, capture
  stdout/stderr to a per-run log file.
- All proposal writes go under `data/policy_proposals/`; never modify
  `policy-graph/` from `/api/policy/propose-diff`.
