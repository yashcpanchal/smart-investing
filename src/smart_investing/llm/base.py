"""Provider-agnostic LLM contract.

Every provider (Gemini today, Anthropic/Claude drop-in) implements `LLMClient`.
The rest of the system types against this module only — swapping providers is a
config change (`LLM_PROVIDER` / the relevant API key in `.env`), never a code
change.

The core primitive is `chat()`: a message history + optional tool schemas in,
text and/or tool calls out. `complete*` are one-shot conveniences built on it.
Grounded search is provider-specific (Google Search on Gemini, web_search on
Anthropic); the base falls back to an ungrounded completion with no sources.
"""

from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import httpx

_RETRY_STATUS = {429, 500, 502, 503, 504, 529}  # transient / rate-limit / overloaded


@dataclass
class ToolSpec:
    """One callable tool, described provider-neutrally (JSON-Schema params)."""

    name: str
    description: str
    parameters: dict  # JSON schema: {"type": "object", "properties": {...}, ...}


@dataclass
class ToolCall:
    """A tool invocation the model asked for."""

    name: str
    arguments: dict
    id: str = ""  # provider call id (Anthropic requires it round-tripped)


@dataclass
class ToolResult:
    """The outcome of executing one ToolCall, to feed back into the loop."""

    call: ToolCall
    content: Any  # dict preferred; strings are wrapped by providers as needed


@dataclass
class ChatMessage:
    """One turn. role: "user" | "assistant" | "tool" (tool = results batch)."""

    role: str
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)  # assistant turns
    tool_results: list[ToolResult] = field(default_factory=list)  # tool turns


@dataclass
class LLMResponse:
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    sources: list[dict] = field(default_factory=list)  # [{title, uri}] when grounded
    stop_reason: str = ""


class LLMClient(ABC):
    """Common surface + shared HTTP plumbing for all providers."""

    provider: str = "base"

    def __init__(self, api_key: str | None, model: str, timeout: float = 20.0) -> None:
        # None -> provider falls back to its configured key upstream; explicit ""
        # forces the deterministic no-LLM path (tests / offline mode).
        self.api_key = api_key
        self.model = model
        # Short timeout: this is an interactive app. A slow/unavailable call should
        # fall back to the deterministic path fast, not block a conversational turn.
        self._client = httpx.Client(timeout=timeout)

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    # ------------------------------------------------------------- primitives
    @abstractmethod
    def chat(
        self,
        messages: list[ChatMessage],
        *,
        system: str | None = None,
        tools: list[ToolSpec] | None = None,
        json_mode: bool = False,
        temperature: float = 0.2,
    ) -> LLMResponse:
        """One model turn over a history, optionally offering tools."""

    # ----------------------------------------------------------- conveniences
    def complete(
        self,
        prompt: str,
        system: str | None = None,
        json_mode: bool = False,
        temperature: float = 0.2,
    ) -> str:
        return self.chat(
            [ChatMessage(role="user", content=prompt)],
            system=system,
            json_mode=json_mode,
            temperature=temperature,
        ).text

    def complete_json(self, prompt: str, system: str | None = None, temperature: float = 0.1) -> dict:
        text = self.complete(prompt, system=system, json_mode=True, temperature=temperature)
        return loads_lenient(text)

    def complete_grounded(
        self, prompt: str, system: str | None = None, temperature: float = 0.3
    ) -> tuple[str, list[dict]]:
        """Answer grounded in live web search where the provider supports it.
        Returns (text, sources); base falls back to an ungrounded answer."""
        return self.complete(prompt, system=system, temperature=temperature), []

    # ------------------------------------------------------------- plumbing
    def _post_retry(
        self,
        url: str,
        *,
        body: dict,
        params: dict | None = None,
        headers: dict | None = None,
    ) -> dict:
        """POST with fail-fast retry; returns the parsed JSON response.

        At most 3 attempts with short backoff (0.5s, 1s) — on a rate-limited tier
        a sustained 429 drops to the caller's deterministic fallback in ~1.5s
        rather than hanging the turn. Transient blips still get a couple retries."""
        last: httpx.Response | None = None
        for attempt in range(3):
            try:
                last = self._client.post(url, params=params, headers=headers, json=body)
            except httpx.TransportError:
                if attempt < 2:
                    time.sleep(0.5 * (attempt + 1))
                    continue
                raise
            if last.status_code in _RETRY_STATUS and attempt < 2:
                time.sleep(0.5 * (attempt + 1))
                continue
            break
        assert last is not None
        last.raise_for_status()
        return last.json()


def loads_lenient(text: str) -> dict:
    """Parse model JSON output, tolerating ``` fences and stray prose."""
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        t = text.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        start, end = t.find("{"), t.rfind("}")
        if start != -1 and end != -1:
            return json.loads(t[start : end + 1])
        raise
