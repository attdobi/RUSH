from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import anthropic

from pipeline.providers.anthropic_chat import policy_chat_callable


class _FakeMessages:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        return SimpleNamespace(
            content=[
                SimpleNamespace(type="text", text='{"files": []}'),
            ],
            usage=SimpleNamespace(
                input_tokens=40,
                output_tokens=90,
                cache_read_input_tokens=5000,
                cache_creation_input_tokens=200,
            ),
        )


class _FakeAnthropic:
    instances: list["_FakeAnthropic"] = []

    def __init__(self) -> None:
        self.messages = _FakeMessages()
        self.__class__.instances.append(self)


def test_policy_chat_callable_maps_messages_without_live_call(monkeypatch: Any) -> None:
    _FakeAnthropic.instances.clear()
    monkeypatch.setattr(anthropic, "Anthropic", _FakeAnthropic)

    chat = policy_chat_callable("anthropic/claude-opus-4-7")
    assert _FakeAnthropic.instances == []  # factory is lazy

    out = chat(
        [
            {"role": "system", "content": "system prompt"},
            {"role": "user", "content": "draft a policy"},
            {"role": "assistant", "content": "previous assistant text"},
        ],
        model_id="anthropic/claude-opus-4-7",
        reasoning_effort="high",
    )

    assert out == '{"files": []}'
    assert len(_FakeAnthropic.instances) == 1
    call = _FakeAnthropic.instances[0].messages.calls[0]
    assert call["model"] == "claude-opus-4-7"
    assert call["max_tokens"] == 8000
    # System goes as a block with an ephemeral prompt-cache breakpoint so
    # repeat drafter/gate calls re-read the shared prefix at ~0.1x.
    assert call["system"] == [
        {"type": "text", "text": "system prompt", "cache_control": {"type": "ephemeral"}}
    ]
    assert call["messages"] == [
        {"role": "user", "content": "draft a policy"},
        {"role": "assistant", "content": "previous assistant text"},
    ]


def test_anthropic_usage_sink_carries_cache_read_and_write(monkeypatch: Any) -> None:
    """Anthropic bills cache reads/writes OUTSIDE input_tokens — the sink must
    carry both or the drafter/gate ledger under-counts real spend."""
    _FakeAnthropic.instances.clear()
    monkeypatch.setattr(anthropic, "Anthropic", _FakeAnthropic)

    sink: list[dict[str, Any]] = []
    chat = policy_chat_callable("anthropic/claude-opus-4-7", usage_sink=sink)
    chat([{"role": "user", "content": "draft"}], model_id="anthropic/claude-opus-4-7")

    assert sink == [
        {
            "model_id": "anthropic/claude-opus-4-7",
            "input_tokens": 40,
            "output_tokens": 90,
            "cached_input_tokens": 5000,
            "cache_creation_input_tokens": 200,
        }
    ]
