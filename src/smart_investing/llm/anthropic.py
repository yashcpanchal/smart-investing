"""Anthropic (Claude) provider — the drop-in upgrade path from Gemini.

Implements the same `LLMClient` contract (REST via httpx, matching the repo's
no-SDK-dependency pattern and the shared fail-fast retry that keeps a slow or
unavailable LLM from blocking a conversational turn). Selected automatically by
the factory when ANTHROPIC_API_KEY is set, or forced with LLM_PROVIDER=anthropic.

Provider notes (Claude Messages API):
- Default model is claude-opus-4-8. Sampling params (temperature/top_p) are
  REMOVED on Opus 4.7+ and return 400 — so this client ignores the interface's
  `temperature` knob; behavior is steered by prompting instead.
- "Thinking" is omitted (off by default on Opus 4.8) for interactive latency,
  mirroring the Gemini client's thinkingBudget=0.
- Grounded answers use the web_search server tool; a safety-classifier
  `refusal` stop reason returns empty output, which drops the caller onto the
  deterministic fallback like any other LLM failure.
"""

from __future__ import annotations

import json

from smart_investing.config import settings
from smart_investing.llm.base import (
    ChatMessage,
    LLMClient,
    LLMResponse,
    ToolCall,
    ToolSpec,
)

ANTHROPIC_BASE = "https://api.anthropic.com/v1"
ANTHROPIC_VERSION = "2023-06-01"
DEFAULT_MODEL = "claude-opus-4-8"
_MAX_TOKENS = 4096  # short structured outputs; thinking is off


class AnthropicClient(LLMClient):
    provider = "anthropic"

    def __init__(self, api_key: str | None = None, model: str | None = None) -> None:
        # None -> configured key; explicit "" -> deterministic no-LLM path (tests).
        key = settings.anthropic_api_key if api_key is None else api_key
        super().__init__(api_key=key, model=model or DEFAULT_MODEL, timeout=30.0)

    def _messages(self, body: dict) -> dict:
        return self._post_retry(
            f"{ANTHROPIC_BASE}/messages",
            body=body,
            headers={"x-api-key": self.api_key, "anthropic-version": ANTHROPIC_VERSION},
        )

    def chat(
        self,
        messages: list[ChatMessage],
        *,
        system: str | None = None,
        tools: list[ToolSpec] | None = None,
        json_mode: bool = False,
        temperature: float = 0.2,  # ignored — removed on Opus 4.7+ (400 if sent)
    ) -> LLMResponse:
        body: dict = {
            "model": self.model,
            "max_tokens": _MAX_TOKENS,
            "messages": [_to_message(m) for m in messages],
        }
        if json_mode:
            # No JSON mime-type knob without a schema; instruct + parse leniently
            # (complete_json already runs loads_lenient on the text).
            system = ((system + "\n\n") if system else "") + (
                "Respond with ONLY a valid JSON object — no prose, no code fences."
            )
        if system:
            body["system"] = system
        if tools:
            body["tools"] = [
                {"name": t.name, "description": t.description, "input_schema": t.parameters} for t in tools
            ]
        return _parse_response(self._messages(body))

    def complete_grounded(
        self, prompt: str, system: str | None = None, temperature: float = 0.3
    ) -> tuple[str, list[dict]]:
        """Answer grounded in live web search (Anthropic server-side tool).
        Returns (text, sources) with sources as [{title, uri}]."""
        body: dict = {
            "model": self.model,
            "max_tokens": _MAX_TOKENS,
            "messages": [{"role": "user", "content": prompt}],
            "tools": [{"type": "web_search_20260209", "name": "web_search", "max_uses": 3}],
        }
        if system:
            body["system"] = system
        data = self._messages(body)
        if data.get("stop_reason") == "refusal":
            return "", []
        text_bits: list[str] = []
        sources: list[dict] = []
        seen: set[str] = set()
        for block in data.get("content", []):
            btype = block.get("type")
            if btype == "text":
                text_bits.append(block.get("text", ""))
                for cit in block.get("citations") or []:
                    url = cit.get("url")
                    if url and url not in seen:
                        seen.add(url)
                        sources.append({"title": cit.get("title", ""), "uri": url})
            elif btype == "web_search_tool_result":
                content = block.get("content")
                if isinstance(content, list):  # error results come back as an object
                    for r in content:
                        url = r.get("url")
                        if url and url not in seen:
                            seen.add(url)
                            sources.append({"title": r.get("title", ""), "uri": url})
        return "".join(text_bits).strip(), sources


def _to_message(m: ChatMessage) -> dict:
    """Map a provider-neutral message onto Claude's messages format."""
    if m.role == "assistant":
        blocks: list[dict] = [{"type": "text", "text": m.content}] if m.content else []
        blocks += [
            {"type": "tool_use", "id": c.id or f"toolu_{i}", "name": c.name, "input": c.arguments}
            for i, c in enumerate(m.tool_calls)
        ]
        return {"role": "assistant", "content": blocks or [{"type": "text", "text": ""}]}
    if m.role == "tool":
        # All results for the preceding assistant turn go in ONE user message.
        return {"role": "user", "content": [
            {
                "type": "tool_result",
                "tool_use_id": r.call.id,
                "content": r.content if isinstance(r.content, str) else json.dumps(r.content),
            }
            for r in m.tool_results
        ]}
    return {"role": "user", "content": m.content}


def _parse_response(data: dict) -> LLMResponse:
    stop = data.get("stop_reason") or ""
    if stop == "refusal":  # safety classifiers declined; treat as no output
        return LLMResponse(stop_reason=stop)
    text_bits: list[str] = []
    calls: list[ToolCall] = []
    for block in data.get("content", []):
        btype = block.get("type")
        if btype == "text":
            text_bits.append(block.get("text", ""))
        elif btype == "tool_use":
            calls.append(ToolCall(name=block.get("name", ""), arguments=block.get("input") or {}, id=block.get("id", "")))
    return LLMResponse(text="".join(text_bits).strip(), tool_calls=calls, stop_reason=stop)
