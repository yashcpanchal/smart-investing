"""Prompt -> StrategySpec.

Uses Gemini to parse a messy natural-language thesis into a strict, structured
spec. Falls back to a deterministic keyword parser when no key is set or the call
fails — so the pipeline always produces a usable spec. The LLM is a *parser*
here, never the source of tickers (that's the retrieval engine's job).
"""

from __future__ import annotations

from smart_investing.domain.types import Objective, RiskParams, StrategySpec
from smart_investing.llm.gemini import GeminiClient, get_llm

_SYSTEM = (
    "You are a financial strategy parser. Convert a retail investor's natural-language "
    "investment thesis into a STRICT JSON strategy spec. Output ONLY JSON, no prose. "
    "Never invent tickers; only echo tickers the user explicitly names."
)

_TEMPLATE = """Investor thesis:
\"\"\"{prompt}\"\"\"

Return ONLY a JSON object with these fields:
- "themes": array of 1-5 short search phrases (e.g. ["nuclear energy","uranium mining"])
- "include_symbols": array of US tickers the user explicitly named (else [])
- "exclude_symbols": array of US tickers/companies to avoid (else [])
- "include_indirect": boolean (include supply-chain/indirect beneficiaries; default true)
- "objective": one of "max_sharpe", "min_vol", "target_vol"
- "concentration_cap": number 0.1-1.0 (max weight per stock; default 0.3)
- "target_volatility": number or null (annual vol target; e.g. "low risk" -> 0.15)"""


def _fallback_spec(prompt: str) -> StrategySpec:
    p = prompt.lower()
    objective = Objective.MAX_SHARPE
    target_vol = None
    cap = 0.30
    if any(w in p for w in ("low risk", "lower risk", "less risk", "reduce risk",
                            "conservative", "safe", "low volatility", "stable")):
        objective, target_vol = Objective.TARGET_VOL, 0.15
    if any(w in p for w in ("diversif", "spread out", "broad", "balanced")):
        cap = 0.20
    include_indirect = not any(w in p for w in ("only direct", "no supply chain", "pure play", "pure-play"))
    return StrategySpec(
        raw_prompt=prompt,
        themes=[prompt],
        objective=objective,
        include_indirect=include_indirect,
        risk=RiskParams(concentration_cap=cap, target_volatility=target_vol),
    )


def _coerce_float(value, default: float | None) -> float | None:
    try:
        return float(value) if value is not None else default
    except (TypeError, ValueError):
        return default


def _spec_from_dict(prompt: str, d: dict) -> StrategySpec:
    themes = [str(t) for t in (d.get("themes") or [prompt])][:5]
    try:
        objective = Objective(str(d.get("objective", "max_sharpe")))
    except ValueError:
        objective = Objective.MAX_SHARPE
    cap = _coerce_float(d.get("concentration_cap"), 0.30) or 0.30
    cap = min(max(cap, 0.05), 1.0)
    return StrategySpec(
        raw_prompt=prompt,
        themes=themes,
        include_symbols=[str(s).upper() for s in (d.get("include_symbols") or [])],
        exclude_symbols=[str(s).upper() for s in (d.get("exclude_symbols") or [])],
        include_indirect=bool(d.get("include_indirect", True)),
        objective=objective,
        risk=RiskParams(concentration_cap=cap, target_volatility=_coerce_float(d.get("target_volatility"), None)),
    )


def compile_spec(prompt: str, llm: GeminiClient | None = None) -> StrategySpec:
    llm = llm if llm is not None else get_llm()
    if llm and llm.available:
        try:
            data = llm.complete_json(_TEMPLATE.format(prompt=prompt), system=_SYSTEM)
            return _spec_from_dict(prompt, data)
        except Exception:
            pass
    return _fallback_spec(prompt)
