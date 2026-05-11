# Batch API architecture for RUSH labeling

## 1. Current state

RUSH currently does synchronous, per-image calls. The web registry starts `scripts/run_bulk_labeling.py` as a subprocess (`pipeline/web/run_registry.py:134-203`), which resolves live clients through `_resolve_factory(use_live)` -> `build_client(spec.model_id, reasoning_effort=...)` (`scripts/run_bulk_labeling.py:84-105`; `pipeline/providers/registry.py:167-215`). `run_labeling(...)` builds deterministic `(sample, model)` pairs (`pipeline/runner.py:393-479`), then `_process(pair)` builds exactly one `LabelRequest` and calls `response = client.label(request)` (`pipeline/runner.py:508-522`). With `concurrency > 1`, the same one-request function is submitted to a `ThreadPoolExecutor` (`pipeline/runner.py:573-590`). Provider clients implement only `LabelClient.label(request) -> LabelResponse` (`pipeline/providers/base.py:371-394`): OpenAI calls `client.chat.completions.create(**api_params)` (`pipeline/providers/openai_client.py:205-222`), Anthropic calls `client.messages.create(**api_params)` (`pipeline/providers/anthropic_client.py:192-209`), and Gemini calls `client.models.generate_content(**api_params)` (`pipeline/providers/gemini_client.py:204-221`). So a 15-image run is 15 independent sync API calls per selected model, not one provider batch.

## 2. Provider batch capabilities

### OpenAI Files + Batches (`/v1/batches`)

Official guide: <https://developers.openai.com/api/docs/guides/batch>. OpenAI says the Batch API gives “50% cost discount compared to synchronous APIs” and “Each batch completes within 24 hours.” Flow: write JSONL, upload with `/v1/files` using `purpose="batch"`, create `/v1/batches` with `input_file_id`, `endpoint`, and `completion_window="24h"`, poll, then download `output_file_id`/`error_file_id`. Each JSONL line is `{custom_id, method:"POST", url:"/v1/chat/completions", body:{...}}`; the body is the same non-streaming Chat Completions request RUSH already builds. For RUSH, use `/v1/chat/completions` for GPT-5.5 vision because the current OpenAI client already sends text plus `image_url` data-URL content (`pipeline/providers/openai_client.py:8-9,113-127`) and batch preserves the endpoint body. Current docs list 50,000 requests and 200 MB input file per batch (the design should treat this as the source of truth if it differs from older 100 MB notes).

```python
jsonl = [to_openai_line(req) for req in label_requests]
file = openai.files.create(file=jsonl, purpose="batch")
batch = openai.batches.create(input_file_id=file.id, endpoint="/v1/chat/completions", completion_window="24h")
while batch.status not in {"completed", "failed", "expired", "cancelled"}:
    sleep(backoff.next()); batch = openai.batches.retrieve(batch.id)
outputs = openai.files.content(batch.output_file_id)
errors = openai.files.content(batch.error_file_id) if batch.error_file_id else ""
return parse_openai_jsonl(outputs, errors)
```

### Anthropic Message Batches

Official guide: <https://platform.claude.com/docs/en/build-with-claude/batch-processing>. Anthropic documents `POST /v1/messages/batches`, polling until `processing_status: ended`, then retrieving results from the batch results endpoint/results URL. It is GA in current docs (older integrations used `anthropic-beta: message-batches-2024-09-24`). The guide says costs are reduced by 50%; batches expire after 24 hours; each Message Batch is limited to 100,000 requests or 256 MB. Each request is `{custom_id, params}` where `params` is the normal Messages API body. Current docs say “Any request that you can make to the Messages API can be included in a batch,” including “Vision,” so Claude Opus 4.6 vision is compatible with the existing Anthropic request shape (`pipeline/providers/anthropic_client.py:108-114`) and registry model (`pipeline/providers/registry.py:100-108`).

```python
batch = anthropic.messages.batches.create(requests=[to_anthropic_request(r) for r in label_requests])
while batch.processing_status != "ended":
    sleep(backoff.next())
    batch = anthropic.messages.batches.retrieve(batch.id)
if any(batch.request_counts.failed, batch.request_counts.expired):
    note_partial_failures(batch.request_counts)
stream = anthropic.messages.batches.results(batch.id)
return parse_anthropic_results(stream)
```

### Google Gemini batch mode

Official Gemini API guide: <https://ai.google.dev/gemini-api/docs/batch-api>. Google says the Gemini Batch API processes requests asynchronously at “50% of the standard cost” with a 24-hour target turnaround. There are two viable paths: Vertex AI Batch Prediction (<https://docs.cloud.google.com/vertex-ai/generative-ai/docs/multimodal/batch-prediction-gemini>) for enterprise BigQuery/GCS workflows, or the simpler Gemini API Files+Batches path. Use the simpler path first: upload a JSONL file through the File API; each line is `{key, request}` where `request` is a `GenerateContentRequest`; create `client.batches.create(model="gemini-3.1-pro-preview", src=uploaded_file.name, ...)`; poll `client.batches.get(name=...)`; if succeeded, download `batch_job.dest.file_name`. Gemini docs explicitly say JSONL can reference uploaded files for multimodal input, and Vertex docs list Gemini 3.1 Pro as batch-capable, so the current Gemini 3.1 Pro vision workload is compatible.

```python
uploaded = gemini.files.upload(file="rush-batch.jsonl", mime_type="jsonl")
job = gemini.batches.create(model="gemini-3.1-pro-preview", src=uploaded.name, config={"display_name": run_id})
while job.state.name not in TERMINAL_STATES:
    sleep(backoff.next())
    job = gemini.batches.get(name=job.name)
if job.state.name != "JOB_STATE_SUCCEEDED":
    raise BatchFailed(job.error)
blob = gemini.files.download(file=job.dest.file_name)
return parse_gemini_jsonl(blob)
```

## 3. Architectural design — RUSH integration

### 3a. Sync vs async lifecycle

Keep the current synchronous path for small jobs, but add async states to `_jobs/*.json` and make the runner/registry resumable. State machine:

`queued -> running -> scoring -> completed` for sync runs.

`queued -> batch_queued -> batch_processing -> batch_complete -> scoring -> completed` for batch runs, with `failed`, `batch_failed`, and `scoring_failed` terminal/error branches.

Extend the existing state file (`pipeline/web/run_registry.py:158-174`) rather than creating a parallel store:

```json
{
  "job_id": "job-...",
  "run_id": "20260511T...",
  "status": "batch_processing",
  "mode": "batch",
  "batch": {
    "provider": "openai",
    "provider_batch_id": "batch_abc",
    "input_file_id": "file_abc",
    "output_file_id": null,
    "error_file_id": null,
    "submitted_at": "...",
    "last_polled_at": "...",
    "request_counts": {"total": 15, "completed": 8, "failed": 0},
    "next_poll_at": "...",
    "attempt": 1
  }
}
```

### 3b. Sync fallback for small runs

Add `RUSH_BATCH_MIN_IMAGES=10` (default 10) in provider config/env. If `len(samples) * len(models) < threshold`, call `label()` exactly as today. At or above threshold, group by provider/model and submit batches.

### 3c. Cost & SLA surfacing in UI

The launch preview should show estimated sync cost, estimated batch cost, savings, and chosen path. Highlight the 50% batch discount. Batch job pages should show queued/processing elapsed time, provider SLA (“target within 24h”), and progress if exposed: OpenAI/Anthropic `request_counts`; Gemini state/file result metadata varies.

### 3d. Polling strategy

Do not tight-loop. Use exponential backoff: 30s, 60s, 2m, 5m, capped at 10m. `RunRegistry` should start a background poller on web-server boot that scans `_jobs/*.json` for `batch_queued`/`batch_processing`, reloads provider batch IDs, polls once when `next_poll_at <= now`, and persists new state before ingesting outputs.

### 3e. Provider abstraction

Extend `LabelClient` with optional batch capability; default implementations can raise `NotImplementedError` so unsupported providers stay sync.

```python
@dataclass(frozen=True)
class BatchHandle:
    provider: str
    model_id: str
    provider_batch_id: str
    metadata: dict[str, Any]

@dataclass(frozen=True)
class BatchStatus:
    state: Literal["pending", "processing", "ready", "failed"]
    request_counts: dict[str, int]
    error: str | None = None

class LabelClient(ABC):
    def label(self, request: LabelRequest) -> LabelResponse: ...
    def submit_batch(self, requests: list[LabelRequest]) -> BatchHandle: ...
    def poll_batch(self, handle: BatchHandle) -> BatchStatus: ...
    def fetch_batch(self, handle: BatchHandle) -> list[LabelResponse]: ...
```

### 3f. Error handling

Partial failures are normal: map each provider per-request error back by `custom_id`/`key` to the existing `errors.jsonl` schema with `stage="provider_call"`, `image_id`, `model_id`, `reason`, and attempts where available. Whole-batch failures (upload failure, provider outage, expired batch) retry once with exponential backoff, then persist `status="batch_failed"` and a clear `batch_error`.

## 4. Implementation plan

Phase 1 (P0): OpenAI batches first. Tests: JSONL shape, submit/poll mocks, output/error parsing, partial failures, web restart resumption. Estimate 1-2 engineer-days.

Phase 2: Anthropic Message Batches with the same tests plus request-count progress mapping. Estimate 1-2 engineer-days.

Phase 3: Gemini Files+Batches, with Vertex AI documented as future/enterprise. Tests: file JSONL shape, terminal state mapping, result file parsing. Estimate 1-2 engineer-days.

## 5. Open questions for Attila

- Default threshold: 10 images/calls, or 20 to avoid async overhead?
- Should the web UI keep polling the run page, or introduce email/notification later? Recommendation: page polls in background; RUSH has no email today.
- Should old sync runs be re-runnable as batch? Recommendation: yes, opt-in from run detail.
- Should policy iteration cost reports show batch savings alongside realized spend?
- Do we require one batch per provider/model, or can same-provider models be submitted separately but displayed as one logical RUSH run?

## 6. References

- OpenAI Batch API guide: <https://developers.openai.com/api/docs/guides/batch>
- Anthropic Message Batches guide: <https://platform.claude.com/docs/en/build-with-claude/batch-processing>
- Anthropic create/retrieve API: <https://platform.claude.com/docs/en/api/creating-message-batches>
- Gemini API Batch API: <https://ai.google.dev/gemini-api/docs/batch-api>
- Vertex AI Gemini batch inference: <https://docs.cloud.google.com/vertex-ai/generative-ai/docs/multimodal/batch-prediction-gemini>
