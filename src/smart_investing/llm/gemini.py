"""Gemini provider (REST via httpx — no SDK dependency).

Implements the provider-agnostic `LLMClient` contract, including native
function-calling (`chat` with tools) and Google-Search-grounded answers.
Kept lean and low-volume to respect the free tier.
"""

from __future__ import annotations

from smart_investing.config import settings
from smart_investing.llm.base import (
    ChatMessage,
    LLMClient,
    LLMResponse,
    ToolCall,
    ToolSpec,
)

GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta"
DEFAULT_MODEL = "gemini-2.5-flash"


class GeminiClient(LLMClient):
    provider = "gemini"

    def __init__(self, api_key: str | None = None, model: str | None = None) -> None:
        # None -> fall back to configured key; explicit "" -> force the deterministic
        # no-LLM path (used by tests and offline mode). Don't let "" leak the env key.
        key = settings.gemini_api_key if api_key is None else api_key
        super().__init__(api_key=key, model=model or DEFAULT_MODEL)

    def _generate(self, body: dict) -> dict:
        url = f"{GEMINI_BASE}/models/{self.model}:generateContent"
        return self._post_retry(url, body=body, params={"key": self.api_key})

    def chat(
        self,
        messages: list[ChatMessage],
        *,
        system: str | None = None,
        tools: list[ToolSpec] | None = None,
        json_mode: bool = False,
        temperature: float = 0.2,
    ) -> LLMResponse:
        body: dict = {
            "contents": [_to_content(m) for m in messages],
            # Disable "thinking" — for our short structured prompts it only adds
            # multi-second latency (and burns the free-tier rate limit) with no gain.
            "generationConfig": {"temperature": temperature, "thinkingConfig": {"thinkingBudget": 0}},
        }
        if system:
            body["systemInstruction"] = {"parts": [{"text": system}]}
        if tools:
            body["tools"] = [{"functionDeclarations": [_to_decl(t) for t in tools]}]
        elif json_mode:  # responseMimeType and tools are mutually exclusive on Gemini
            body["generationConfig"]["responseMimeType"] = "application/json"
        data = self._generate(body)
        return _parse_response(data)

    def complete_grounded(
        self, prompt: str, system: str | None = None, temperature: float = 0.3
    ) -> tuple[str, list[dict]]:
        """Answer grounded in live Google Search. Returns (text, sources) where
        sources is [{title, uri}]. Used for current industry/market context.
        Grounding needs "thinking" on, so we don't disable it here."""
        body: dict = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "tools": [{"google_search": {}}],
            "generationConfig": {"temperature": temperature},
        }
        if system:
            body["systemInstruction"] = {"parts": [{"text": system}]}
        data = self._generate(body)
        try:
            cand = data["candidates"][0]
            text = cand["content"]["parts"][0]["text"]
        except (KeyError, IndexError):
            return "", []
        sources: list[dict] = []
        for ch in cand.get("groundingMetadata", {}).get("groundingChunks", []):
            web = ch.get("web") or {}
            if web.get("uri"):
                sources.append({"title": web.get("title", ""), "uri": web["uri"]})
        return text, sources


def _to_decl(t: ToolSpec) -> dict:
    decl = {"name": t.name, "description": t.description}
    # Gemini rejects declarations with an empty properties object — omit instead.
    if t.parameters.get("properties"):
        decl["parameters"] = t.parameters
    return decl


def _to_content(m: ChatMessage) -> dict:
    """Map a provider-neutral message onto Gemini's contents format."""
    if m.role == "assistant":
        parts: list[dict] = [{"text": m.content}] if m.content else []
        parts += [{"functionCall": {"name": c.name, "args": c.arguments}} for c in m.tool_calls]
        return {"role": "model", "parts": parts or [{"text": ""}]}
    if m.role == "tool":
        return {"role": "user", "parts": [
            {"functionResponse": {
                "name": r.call.name,
                # Gemini requires an object; wrap scalars/strings.
                "response": r.content if isinstance(r.content, dict) else {"result": r.content},
            }}
            for r in m.tool_results
        ]}
    return {"role": "user", "parts": [{"text": m.content}]}


def _parse_response(data: dict) -> LLMResponse:
    try:
        cand = data["candidates"][0]
        parts = cand["content"]["parts"]
    except (KeyError, IndexError):
        return LLMResponse()
    text_bits: list[str] = []
    calls: list[ToolCall] = []
    for i, part in enumerate(parts):
        if "text" in part:
            text_bits.append(part["text"])
        elif "functionCall" in part:
            fc = part["functionCall"]
            # Gemini doesn't issue call ids; synthesize one so results round-trip
            # cleanly (and so a Claude-format replay of this history stays valid).
            calls.append(ToolCall(name=fc.get("name", ""), arguments=fc.get("args") or {}, id=f"call_{i}"))
    return LLMResponse(
        text="".join(text_bits).strip(),
        tool_calls=calls,
        stop_reason=cand.get("finishReason", ""),
    )


def get_llm():
    """Back-compat alias — the provider-aware factory lives in llm.factory."""
    from smart_investing.llm.factory import get_llm as _factory_get_llm

    return _factory_get_llm()
