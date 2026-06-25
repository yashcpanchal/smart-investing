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
        self._client = httpx.Client(timeout=60.0)

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    def complete(
        self,
        prompt: str,
        system: str | None = None,
        json_mode: bool = False,
        temperature: float = 0.2,
    ) -> str:
        url = f"{GEMINI_BASE}/models/{self.model}:generateContent"
        body: dict = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": temperature},
        }
        if system:
            body["systemInstruction"] = {"parts": [{"text": system}]}
        if json_mode:
            body["generationConfig"]["responseMimeType"] = "application/json"

        last: httpx.Response | None = None
        for attempt in range(5):
            try:
                last = self._client.post(url, params={"key": self.api_key}, json=body)
            except httpx.TransportError:
                if attempt < 4:
                    time.sleep(min(2**attempt, 8))
                    continue
                raise
            if last.status_code in _RETRY_STATUS and attempt < 4:
                time.sleep(min(2**attempt, 8))  # 1s,2s,4s,8s backoff on 429/5xx
                continue
            break
        assert last is not None
        last.raise_for_status()
        data = last.json()
        try:
            return data["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError):
            return ""

    def complete_json(self, prompt: str, system: str | None = None, temperature: float = 0.1) -> dict:
        text = self.complete(prompt, system=system, json_mode=True, temperature=temperature)
        return _loads_lenient(text)


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
