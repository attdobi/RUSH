# OpenAI labeling source review

Reviewed local source repo: `/Users/sacsimoto/GitHub/d-ai-trader`.

## Credential handling

- Source repo `.env` exists and is not tracked by git.
- RUSH `.env` has been copied locally from the source repo and is ignored by RUSH `.gitignore`.
- Keep both `.env` files local-only. Do not add `.env`, `.env.*`, generated manifests with credentials, or raw labeling outputs that might contain sensitive metadata.

## Useful patterns for the future labeling client

- `config.py` loads `OPENAI_API_KEY` from environment / `.env`, masks optional debug logging, and sets `openai.api_key`.
- `config.py::PromptManager.ask_openai(...)` already handles:
  - GPT-5-style reasoning parameters.
  - Chat Completions calls.
  - Vision payloads by base64-encoding image files into `image_url` message parts.
  - JSON response coercion for selected agents.
  - A retry path when `reasoning_effort` is rejected.
- `main.py::get_openai_summary(...)` shows the current screenshot/image filtering flow before calling `PromptManager.ask_openai(...)`.
- `feedback_agent.py` has a smaller helper pattern for assembling API params and retrying without `reasoning_effort` if needed.
- `dashboard_server.py` contains a minimal `OpenAI()` client example, but RUSH should probably use a dedicated labeling module rather than importing trader app code.

## RUSH recommendation

When we build the GenAI labeling pass, copy the useful ideas rather than coupling to `d-ai-trader`:

1. A small RUSH-local OpenAI client wrapper.
2. Explicit image size/type validation before API calls.
3. Structured JSON output tied to `schemas/label-record.schema.json`.
4. Policy-node evidence requirements in the prompt.
5. No-submit/no-mutation default for labeling runs until SME review gates are defined.
