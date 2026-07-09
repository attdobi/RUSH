from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import openai

from pipeline.providers import auth
from pipeline.providers.openai_chat import policy_chat_callable


class _FakeCompletions:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content='{"files": []}'))],
            usage=SimpleNamespace(
                prompt_tokens=123,
                completion_tokens=45,
                prompt_tokens_details=SimpleNamespace(cached_tokens=100),
            ),
        )


class _FakeChat:
    def __init__(self) -> None:
        self.completions = _FakeCompletions()


class _FakeOpenAI:
    instances: list["_FakeOpenAI"] = []

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.chat = _FakeChat()
        self.__class__.instances.append(self)


def test_openai_policy_chat_callable_maps_json_request_without_live_call(monkeypatch: Any) -> None:
    _FakeOpenAI.instances.clear()
    auth.reset_for_tests()
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(openai, "OpenAI", _FakeOpenAI)

    chat = policy_chat_callable("openai/gpt-5.5")
    assert _FakeOpenAI.instances == []  # factory is lazy

    messages = [
        {"role": "system", "content": "system prompt"},
        {"role": "user", "content": "draft a policy"},
    ]
    out = chat(messages, model_id="openai/gpt-5.5", reasoning_effort="high")

    assert out == '{"files": []}'
    assert len(_FakeOpenAI.instances) == 1
    assert _FakeOpenAI.instances[0].kwargs["api_key"] == "test-key"
    call = _FakeOpenAI.instances[0].chat.completions.calls[0]
    assert call["model"] == "gpt-5.5"
    assert call["messages"] == messages
    assert call["response_format"] == {"type": "json_object"}
    assert call["reasoning_effort"] == "high"
    assert call["max_completion_tokens"] == 10000


def test_openai_usage_sink_carries_cached_tokens(monkeypatch: Any) -> None:
    """The drafter/gate cost ledger needs the cached share to discount it."""
    _FakeOpenAI.instances.clear()
    auth.reset_for_tests()
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(openai, "OpenAI", _FakeOpenAI)

    sink: list[dict[str, Any]] = []
    chat = policy_chat_callable("openai/gpt-5.5", usage_sink=sink)
    chat([{"role": "user", "content": "draft"}], model_id="openai/gpt-5.5")

    assert sink == [
        {
            "model_id": "openai/gpt-5.5",
            "input_tokens": 123,
            "output_tokens": 45,
            "cached_input_tokens": 100,
        }
    ]
