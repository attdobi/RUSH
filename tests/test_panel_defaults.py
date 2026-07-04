from __future__ import annotations

import re
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]

EXPECTED_DEFAULT_CHECKED = {
    "openai/gpt-5.5-low",
    "anthropic/claude-sonnet-5-low",
    "google/gemini-3.5-flash",
    "local/gemma-4-26b-a4b-qat",
}

UNCHECKED_REASONING_VARIANTS = {
    "openai/gpt-5.5-xhigh",
    "openai/gpt-5.5-high",
    "openai/gpt-5.5-medium",
    "openai/gpt-5.4-mini-xhigh",
    "openai/gpt-5.4-mini-high",
    "openai/gpt-5.4-mini-medium",
    "anthropic/claude-sonnet-5-medium",
    "anthropic/claude-haiku-4-5-medium",
}

NEW_PANEL_IDS = {
    "openai/gpt-5.5-medium",
    "openai/gpt-5.4-mini-medium",
    "anthropic/claude-sonnet-5-low",
    "anthropic/claude-sonnet-5-medium",
    "anthropic/claude-haiku-4-5-low",
    "anthropic/claude-haiku-4-5-medium",
}


def _extract_model_groups() -> str:
    text = (_REPO_ROOT / "web" / "run-trigger.js").read_text(encoding="utf-8")
    start = text.index("const MODEL_GROUPS = [")
    body = text[start:]
    depth = 0
    for i, ch in enumerate(body):
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                return body[body.index("[") : i + 1]
    raise AssertionError("Could not parse MODEL_GROUPS")


def _panel_entries() -> dict[str, bool]:
    body = _extract_model_groups()
    entry_re = re.compile(
        r"\{\s*id:\s*'([^']+)'\s*,\s*checked:\s*(true|false)\b[^}]*\}"
    )
    return {model_id: checked == "true" for model_id, checked in entry_re.findall(body)}


def test_run_trigger_default_panel_is_diverse_low_reasoning() -> None:
    entries = _panel_entries()

    assert {model_id for model_id, checked in entries.items() if checked} == EXPECTED_DEFAULT_CHECKED
    assert NEW_PANEL_IDS <= set(entries)
    assert "anthropic/claude-sonnet-5" not in entries
    assert "anthropic/claude-haiku-4-5" not in entries

    missing_variants = UNCHECKED_REASONING_VARIANTS - set(entries)
    assert not missing_variants
    assert {model_id for model_id in UNCHECKED_REASONING_VARIANTS if entries[model_id]} == set()
