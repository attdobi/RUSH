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
            ]
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
    assert call["max_tokens"] == 2000
    assert call["system"] == "system prompt"
    assert call["messages"] == [
        {"role": "user", "content": "draft a policy"},
        {"role": "assistant", "content": "previous assistant text"},
    ]
