"""LLM provider clients for the RUSH bulk-labeling pipeline.

Public surface:

* :class:`pipeline.providers.base.LabelClient` — abstract base class.
* :class:`pipeline.providers.base.LabelRequest` /
  :class:`pipeline.providers.base.LabelResponse` — in-process contracts.
* :data:`pipeline.providers.registry.MODEL_REGISTRY` — model_id → provider.
* :func:`pipeline.providers.registry.build_client` — factory by model_id.

Provider implementations (lazy SDK imports — safe to import this package
without any LLM SDK installed):

* :class:`pipeline.providers.openai_client.OpenAIClient`
* :class:`pipeline.providers.anthropic_client.AnthropicClient`
* :class:`pipeline.providers.gemini_client.GeminiClient`

Every client routes image bytes through
:func:`pipeline.labeling.image_prep.prepare_image` so the same downsampled
JPEG (and the same sha256) is what every provider sees and what we persist.
"""

from pipeline.providers.base import (
    LabelClient,
    LabelRequest,
    LabelResponse,
    ProviderError,
    ProviderRateLimitError,
)
from pipeline.providers.registry import (
    MODEL_REGISTRY,
    ModelSpec,
    build_client,
    list_models,
)

__all__ = [
    "LabelClient",
    "LabelRequest",
    "LabelResponse",
    "ProviderError",
    "ProviderRateLimitError",
    "MODEL_REGISTRY",
    "ModelSpec",
    "build_client",
    "list_models",
]
