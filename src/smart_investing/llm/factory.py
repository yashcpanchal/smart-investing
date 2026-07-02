"""Provider selection — the plug-and-play switch.

`get_llm()` picks the provider from config, so swapping Gemini for Claude is a
`.env` change, not a code change:

- LLM_PROVIDER=anthropic | gemini | none   (default "auto")
- LLM_MODEL=<override the provider's default model>          (optional)
- "auto": Anthropic if ANTHROPIC_API_KEY is set (the intended upgrade path),
  else Gemini if GEMINI_API_KEY is set, else None (deterministic fallbacks).
"""

from __future__ import annotations

from smart_investing.config import settings
from smart_investing.llm.anthropic import AnthropicClient
from smart_investing.llm.base import LLMClient
from smart_investing.llm.gemini import GeminiClient


def get_llm() -> LLMClient | None:
    provider = (settings.llm_provider or "auto").strip().lower()
    model = settings.llm_model or None
    if provider in ("none", "off", "disabled"):
        return None
    client: LLMClient
    if provider == "anthropic":
        client = AnthropicClient(model=model)
    elif provider == "gemini":
        client = GeminiClient(model=model)
    elif settings.anthropic_api_key:
        client = AnthropicClient(model=model)
    else:
        client = GeminiClient(model=model)
    return client if client.available else None
