from __future__ import annotations

import pytest

from pipeline.runner import (
    LOCAL_MODEL_MAX_CONCURRENCY,
    LOCAL_PROVIDER_TAG,
    _sem_key_and_size,
)


@pytest.mark.parametrize("concurrency", [2, 4, 8])
def test_local_models_key_by_model_id_and_serialize_per_card(concurrency: int) -> None:
    """Each local model runs on its own GPU card, so distinct local models get
    DISTINCT semaphore keys (parallel across cards), each capped to
    LOCAL_MODEL_MAX_CONCURRENCY (serial per card)."""
    key_qwen, size_qwen = _sem_key_and_size(
        LOCAL_PROVIDER_TAG, "local/qwen3.6-27b", concurrency
    )
    key_gemma, size_gemma = _sem_key_and_size(
        LOCAL_PROVIDER_TAG, "local/gemma-4-26b-a4b-qat", concurrency
    )

    # Distinct keys -> separate semaphores -> parallel across the two cards.
    assert key_qwen != key_gemma
    assert key_qwen == "local/qwen3.6-27b"
    assert key_gemma == "local/gemma-4-26b-a4b-qat"

    # Each local model serializes its own calls.
    assert size_qwen == LOCAL_MODEL_MAX_CONCURRENCY == 1
    assert size_gemma == LOCAL_MODEL_MAX_CONCURRENCY == 1


@pytest.mark.parametrize("provider", ["openai", "anthropic", "gemini"])
@pytest.mark.parametrize("concurrency", [2, 4, 8])
def test_hosted_providers_key_by_provider_shared_across_models(
    provider: str, concurrency: int
) -> None:
    """Hosted providers key by provider (shared API rate limit): all models of
    a provider share one semaphore sized to the global concurrency."""
    key_a, size_a = _sem_key_and_size(provider, f"{provider}/model-a", concurrency)
    key_b, size_b = _sem_key_and_size(provider, f"{provider}/model-b", concurrency)

    # Same key across models of the same provider -> shared semaphore.
    assert key_a == key_b == provider
    assert size_a == size_b == concurrency
