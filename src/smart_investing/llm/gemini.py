"""Minimal Gemini client (REST via httpx — no SDK dependency).

Used for the LLM-driven steps only (prompt -> StrategySpec, and later
supplier/customer relation extraction). Everything else in the system is
deterministic. Kept lean and low-volume to respect the free tier.
"""

from __future__ import annotations

import json
import time

import httpx

from smart_investing.config import settings

GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta"
_RETRY_STATUS = {429, 500, 502, 503, 504}  # transient / rate-limit


class GeminiClient:
    def __init__(self, api_key: str | None = None, model: str = "gemini-2.5-flash") -> None:
        # None -> fall back to configured key; explicit "" -> force the deterministic
        # no-LLM path (used by tests and offline mode). Don't let "" leak the env key.
        self.api_key = settings.gemini_api_key if api_key is None else api_key
        self.model = model
        # Short timeout: this is an interactive app. A slow/unavailable call should
        # fall back to the deterministic path fast, not block a conversational turn.
        self._client = httpx.Client(timeout=20.0)

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    def _generate(self, body: dict) -> dict:
        """POST generateContent with fail-fast retry; returns the parsed response.

        At most 3 attempts with short backoff (0.5s, 1s) — on the free tier a
        sustained 429 drops to the caller's deterministic fallback in ~1.5s rather
        than hanging the turn. Transient blips still get a couple of retries."""
        url = f"{GEMINI_BASE}/models/{self.model}:generateContent"
        last: httpx.Response | None = None
        for attempt in range(3):
            try:
                last = self._client.post(url, params={"key": self.api_key}, json=body)
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

    def complete(
        self,
        prompt: str,
        system: str | None = None,
        json_mode: bool = False,
        temperature: float = 0.2,
    ) -> str:
        body: dict = {
            "contents": [{"parts": [{"text": prompt}]}],
            # Disable "thinking" — for our short structured prompts it only adds
            # multi-second latency (and burns the free-tier rate limit) with no gain.
            "generationConfig": {"temperature": temperature, "thinkingConfig": {"thinkingBudget": 0}},
        }
        if system:
            body["systemInstruction"] = {"parts": [{"text": system}]}
        if json_mode:
            body["generationConfig"]["responseMimeType"] = "application/json"
        data = self._generate(body)
        try:
            return data["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError):
            return ""

    def complete_json(self, prompt: str, system: str | None = None, temperature: float = 0.1) -> dict:
        text = self.complete(prompt, system=system, json_mode=True, temperature=temperature)
        return _loads_lenient(text)

    def complete_grounded(
        self, prompt: str, system: str | None = None, temperature: float = 0.3
    ) -> tuple[str, list[dict]]:
        """Answer grounded in live Google Search. Returns (text, sources) where
        sources is [{title, uri}]. Used for current industry/market context.
        Grounding needs "thinking" on, so we don't disable it here."""
        body: dict = {
            "contents": [{"parts": [{"text": prompt}]}],
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


def _loads_lenient(text: str) -> dict:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # strip ``` fences / stray prose, grab the outermost object
        t = text.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        start, end = t.find("{"), t.rfind("}")
        if start != -1 and end != -1:
            return json.loads(t[start : end + 1])
        raise


def get_llm() -> GeminiClient | None:
    client = GeminiClient()
    return client if client.available else None
