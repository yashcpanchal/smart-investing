"""LLM layer (Gemini): prompt parsing + (later) relation extraction."""

from smart_investing.llm.compiler import compile_spec
from smart_investing.llm.gemini import GeminiClient, get_llm

__all__ = ["GeminiClient", "get_llm", "compile_spec"]
