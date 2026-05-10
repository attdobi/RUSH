"""RUSH bulk-labeling pipeline.

Subpackages:
    providers: LLM provider clients (OpenAI, Anthropic, Gemini) + auth/retries/registry.
    labeling:  Shared labeling helpers (image preprocessing, etc.).

Owner of providers/ + labeling/image_prep + tests/test_providers_smoke.py: X1.
See docs/EXECUTION-PLAN-bulk-labeling-v1.md for the full slice ownership map.
"""

__all__: list[str] = []
