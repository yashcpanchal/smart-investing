"""LLM layer: provider-agnostic contract + Gemini/Anthropic providers.

Type against `LLMClient`; get a configured instance from `get_llm()`.
Swapping providers is a .env change (see llm/factory.py).
"""

from smart_investing.llm.anthropic import AnthropicClient
from smart_investing.llm.base import (
    ChatMessage,
    LLMClient,
    LLMResponse,
    ToolCall,
    ToolResult,
    ToolSpec,
)
from smart_investing.llm.compiler import compile_spec
from smart_investing.llm.factory import get_llm
from smart_investing.llm.gemini import GeminiClient

__all__ = [
    "AnthropicClient",
    "ChatMessage",
    "GeminiClient",
    "LLMClient",
    "LLMResponse",
    "ToolCall",
    "ToolResult",
    "ToolSpec",
    "compile_spec",
    "get_llm",
]
