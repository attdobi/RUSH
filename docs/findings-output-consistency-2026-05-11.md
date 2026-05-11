# Output verbosity / token-cap consistency audit — 2026-05-11

## Summary

The three Phase-1 labeling providers are **not** currently comparable. A shared prompt module exists (`pipeline/providers/_prompts.py`), but it is dead code for provider request construction: OpenAI, Anthropic, and Gemini each maintain local prompt forks. Those local prompts ask for the older six-field JSON shape, while the shared prompt asks for the newer eight-field shape with `policy_citations` and `policy_quotes`.

Token caps also diverge. OpenAI has a very large `max_completion_tokens` cap, Anthropic has per-model output caps, and Gemini has no explicit output cap. This explains why the same image can produce materially different visible verbosity across providers.

## Per-provider current state

| Provider/model | System prompt source | User prompt source | Requested schema | Output cap | Reasoning / thinking knob | Temperature |
|---|---|---|---|---|---|---|
| `openai/gpt-5.5` | `pipeline/providers/openai_client.py:48` local `DEFAULT_SYSTEM_PROMPT` | `pipeline/providers/openai_client.py:55` local `USER_INSTRUCTIONS` | **6 fields**: `label`, `l2_label`, `justification`, `confidence`, `difficulty`, `is_boundary` | `max_completion_tokens: 24000` in `pipeline/providers/registry.py:62`; Chat Completions cap includes hidden reasoning + visible output | `reasoning_effort: xhigh` in `pipeline/providers/registry.py:59` | Omitted by `resolve_temperature()` for `gpt-5.5*` (`pipeline/providers/_config.py:23-27`) because custom temperature is unsupported |
| `anthropic/claude-opus-4-6` | `pipeline/providers/anthropic_client.py:43` local `DEFAULT_SYSTEM_PROMPT` | `pipeline/providers/anthropic_client.py:50` local `USER_INSTRUCTIONS` | **6 fields**: same legacy shape | `max_tokens: 2048` in `pipeline/providers/registry.py:98`; Anthropic `max_tokens` is visible output cap | None configured | `0.1` via `LABELING_TEMPERATURE` / `resolve_temperature()` (`pipeline/providers/_config.py:9`, `:27`) |
| `anthropic/claude-opus-4-7` | `pipeline/providers/anthropic_client.py:43` local `DEFAULT_SYSTEM_PROMPT` | `pipeline/providers/anthropic_client.py:50` local `USER_INSTRUCTIONS` | **6 fields**: same legacy shape | `max_tokens: 4096` in `pipeline/providers/registry.py:107`; Anthropic `max_tokens` is visible output cap | `thinking_budget_tokens: 32000` in `pipeline/providers/registry.py:108` | Effective request temperature is `1` because Anthropic extended thinking requires it (`pipeline/providers/anthropic_client.py:141-147`) |
| `google/gemini-3.1-pro-preview` | `pipeline/providers/gemini_client.py:46` local `DEFAULT_SYSTEM_PROMPT` embedded in user content | `pipeline/providers/gemini_client.py:53` local `USER_INSTRUCTIONS` | **6 fields**: same legacy shape | **No explicit cap**; registry only sets `thinking_budget_tokens: -1` in `pipeline/providers/registry.py:90` | `thinking_budget_tokens: -1` (unlimited) in `pipeline/providers/registry.py:90` | `0.1` via `LABELING_TEMPERATURE` / `resolve_temperature()` (`pipeline/providers/_config.py:9`, `:27`) |
| Shared prompt module | `pipeline/providers/_prompts.py:45` `LABELING_SYSTEM_PROMPT` | `pipeline/providers/_prompts.py:108` `LABELING_USER_INSTRUCTIONS` | **8 fields**: six legacy fields + `policy_citations`, `policy_quotes` | Soft justification cap only: `MAX_JUSTIFICATION_CHARS = 1500` (`pipeline/providers/_prompts.py:35`) | Prompt says hidden reasoning should be used for deliberation, not visible output | N/A |

## Explicit diffs

| Area | OpenAI | Anthropic | Gemini | Impact |
|---|---|---|---|---|
| Prompt source | Local fork in `openai_client.py` | Local fork in `anthropic_client.py` | Local fork in `gemini_client.py` | `_prompts.py` is not the single source of truth despite its docstring claim. |
| Schema requested | Legacy six-field JSON | Legacy six-field JSON | Legacy six-field JSON | Providers are not being asked for the canonical eight-field policy-grounded output. |
| Shared policy trace fields | Not requested | Not requested | Not requested | `policy_citations` / `policy_quotes` may be absent or provider-dependent even though `coerce_label_fields()` can handle them. |
| Visible output cap | No clean visible cap; `max_completion_tokens=24000` includes hidden reasoning + visible output | `max_tokens=2048` or `4096` | No `max_output_tokens` | OpenAI can ramble, Anthropic is bounded, Gemini has no consistent ceiling. |
| Reasoning budget | `reasoning_effort=xhigh` / variants | Optional `thinking_budget_tokens` on Opus 4.7 | `thinking_budget_tokens=-1` unlimited | Hidden reasoning budgets are not harmonized, and OpenAI's only configured cap mixes hidden + visible tokens. |
| Temperature | Omitted for GPT-5.5 | Usually `0.1`; Opus 4.7 thinking forces `1` | `0.1` | Mostly standardized by `_config.py`, with provider-required exceptions. |

## Root cause

1. `pipeline/providers/_prompts.py` exists and claims to be the shared prompt source for OpenAI, Anthropic, and Gemini, but provider clients do not import it for request construction.
2. Each provider client has a local prompt fork asking for a legacy six-field JSON object.
3. The canonical shared prompt asks for an eight-field output (`policy_citations` and `policy_quotes` included), and `pipeline/providers/base.py:263` already normalizes those fields.
4. Output caps are configured per provider with different semantics and values, not against one visible-output budget.

## Fix plan

1. Wire `openai_client.py`, `anthropic_client.py`, and `gemini_client.py` to `pipeline.providers._prompts`.
   - Keep local exported aliases (`DEFAULT_SYSTEM_PROMPT`, `DEFAULT_USER_PROMPT`, and current `USER_INSTRUCTIONS`) so existing imports/tests continue to work.
   - Make all request builders use the canonical eight-field system and user instructions.
2. Normalize visible-output caps around a **2000-token target**.
   - Anthropic: set `max_tokens=2000` for Phase-1 Opus models; extended-thinking budget remains separate.
   - Gemini: add `max_output_tokens=2000` to registry params and rely on existing `extra_params` plumbing into `config`.
   - OpenAI: this client uses Chat Completions (`client.chat.completions.create`), where the implemented cap is `max_completion_tokens`; there is no currently plumbed separate visible-output cap. To make the bound less permissive while preserving high reasoning headroom, set reasoning variants to `max_completion_tokens=10000` (roughly 8000 reasoning + 2000 visible target) and keep the shared prompt's ~350-token justification soft cap as the visible verbosity control.
3. Update prompt/cap tests that pinned old wording or old cap values.
4. Run a small dev-golden sample if auth/spend allows, then append post-fix token-count verification to this note.
