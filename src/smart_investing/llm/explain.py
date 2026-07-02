"""Pipeline outputs -> plain-English Explanation ("explain our findings").

The structured sections are written deterministically from the REAL numbers the
pipeline produced, so nothing can drift or hallucinate. The LLM is handed those
same facts and asked only to write a warm one-line `summary`; if it's absent or
fails, a templated summary is used. This is what turns a terse rationale into a
conversation the user can actually follow.
"""

from __future__ import annotations

from smart_investing.domain.types import (
    AssetUniverse,
    BacktestResult,
    Explanation,
    HoldingExplanation,
    OptimizationResult,
    StrategySpec,
    ValidationResult,
)
from smart_investing.llm.base import LLMClient

_OBJ_WORDS = {
    "max_sharpe": "the best return per unit of risk (max-Sharpe)",
    "min_vol": "the steadiest possible mix (minimum volatility)",
    "target_vol": "a target risk level you chose",
}

_SYSTEM = (
    "You are an investing co-pilot summarizing a portfolio you just built for a "
    "retail user. Warm, plain English, no jargon, no hype, no markdown. "
    "Use ONLY the facts given, never invent numbers. Output ONLY JSON."
)


def _pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def _build_holdings(universe: AssetUniverse, held: dict[str, float]) -> list[HoldingExplanation]:
    """Per-name rows with deterministic weight/role/relevance and a templated
    `why`; the LLM later overwrites `why` with a qualitative reason (one call)."""
    by_symbol = {a.symbol: a for a in universe.assets}
    rows: list[HoldingExplanation] = []
    for sym, w in sorted(held.items(), key=lambda kv: -kv[1]):
        a = by_symbol.get(sym)
        direct = (a.degree == 1) if a else True
        rel = a.scores.get("relevance", a.scores.get("graph_proximity", 0.0)) if a else 0.0
        why = (
            f"Direct match to the thesis (relevance {rel * 100:.0f}%)."
            if direct
            else f"Supply-chain exposure surfaced from filings — {a.rationale if a else 'indirect link'}."
        )
        rows.append(
            HoldingExplanation(
                symbol=sym,
                name=(a.name if a else sym),
                weight=round(w, 4),
                role=("direct" if direct else "supply-chain"),
                relevance=round(rel, 4),
                why=why,
            )
        )
    return rows


def _summary_fallback(spec: StrategySpec, opt: OptimizationResult, n: int) -> str:
    theme = ", ".join(spec.themes[:2]) or "your thesis"
    return (
        f"I built a {n}-name portfolio around {theme}, tuned for "
        f"{_OBJ_WORDS.get(spec.objective.value, 'a good risk/return balance')}. "
        f"Projected ~{_pct(opt.expected_return)} return at ~{_pct(opt.volatility)} risk "
        f"(Sharpe {opt.sharpe:.2f})."
    )


def build_explanation(
    *,
    spec: StrategySpec,
    universe: AssetUniverse,
    opt: OptimizationResult,
    target_weights: dict[str, float],
    backtest: BacktestResult | None,
    validation: ValidationResult,
    effective_cap: float,
    n_orders: int,
    lookback: str,
    price_source: str,
    llm: LLMClient | None = None,
) -> Explanation:
    held = {s: w for s, w in target_weights.items() if w > 0.005}
    n_held = len(held)
    direct = [a for a in universe.assets if a.degree == 1]
    indirect = [a for a in universe.assets if a.degree == 2]
    top = sorted(held.items(), key=lambda kv: -kv[1])[:3]
    top_str = ", ".join(f"{s} ({_pct(w)})" for s, w in top)

    understood = (
        f"You asked for {', '.join(spec.themes) or 'a custom thesis'}. "
        f"I read that as objective '{spec.objective.value}'"
        + (f", targeting ~{_pct(spec.risk.target_volatility)} volatility" if spec.risk.target_volatility else "")
        + (", including supply-chain names" if spec.include_indirect else ", pure-play only")
        + (f", excluding {', '.join(spec.exclude_symbols)}" if spec.exclude_symbols else "")
        + "."
    )

    selection = (
        f"I searched real SEC filings and kept {len(direct)} direct match"
        f"{'es' if len(direct) != 1 else ''}"
        + (
            f" plus {len(indirect)} indirect supply-chain name{'s' if len(indirect) != 1 else ''} "
            "(surfaced from co-mentions in those filings)"
            if indirect
            else ""
        )
        + f". After pricing, {n_held} of them earned a real weight."
    )

    cap_clause = (
        f"No single name exceeds {_pct(effective_cap)}"
        + (
            f" (I relaxed your cap from {_pct(spec.risk.concentration_cap)} because the universe was small)"
            if effective_cap > spec.risk.concentration_cap + 1e-9
            else ""
        )
        + "."
    )
    construction = (
        f"Weights come from mean-variance optimization aiming for "
        f"{_OBJ_WORDS.get(spec.objective.value, 'a strong risk/return balance')}. "
        f"Your biggest positions are {top_str or 'spread evenly'}. {cap_clause}"
    )

    if validation.ok:
        risk_note = (
            f"Every order cleared the circuit breaker, a hard non-AI safety gate that checks "
            f"buying power, position caps and price sanity before anything executes. "
            f"Projected risk is ~{_pct(opt.volatility)} annual volatility."
        )
    else:
        reasons = "; ".join(v.message for v in validation.fatal[:3]) or "a safety check failed"
        risk_note = (
            f"The circuit breaker BLOCKED execution ({reasons}). Nothing will trade until that's resolved; "
            "this is the deterministic safety gate doing its job."
        )

    data_note = (
        f"Risk, correlations and the backtest use {lookback} of {price_source} price history"
        + (
            f". Backtested CAGR over that window was {_pct(backtest.cagr)} "
            f"with a {_pct(abs(backtest.max_drawdown))} max drawdown"
            if backtest
            else ""
        )
        + "."
    )

    highlights = [
        f"{n_held} holdings | top weight {_pct(top[0][1]) if top else '-'}",
        f"Return {_pct(opt.expected_return)} | risk {_pct(opt.volatility)} | Sharpe {opt.sharpe:.2f}",
        f"{len(direct)} direct + {len(indirect)} supply-chain candidates",
        ("Circuit breaker: PASS" if validation.ok else f"Circuit breaker: BLOCKED ({len(validation.fatal)} fatal)"),
    ]

    holdings = _build_holdings(universe, held)

    # ONE LLM call writes both the warm summary and a per-name qualitative `why`
    # (no numbers, so nothing can drift). Keeps each conversational turn fast.
    summary = _summary_fallback(spec, opt, n_held)
    if llm is not None and getattr(llm, "available", False):
        theme = ", ".join(spec.themes) or "the thesis"
        facts = {
            "theme": theme,
            "holdings": n_held,
            "expected_return": round(opt.expected_return, 4),
            "volatility": round(opt.volatility, 4),
            "sharpe": round(opt.sharpe, 3),
            "blocked": not validation.ok,
            "companies": [{"symbol": h.symbol, "name": h.name, "role": h.role} for h in holdings],
        }
        prompt = (
            "You built this portfolio. Using ONLY these facts, return JSON with:\n"
            '  "summary": a warm 1-2 sentence headline (<=45 words), and\n'
            '  "reasons": a map {TICKER: "<=16-word concrete reason it fits the theme — what it '
            "does / where it sits in the supply chain, no numbers, no hype\"}.\n"
            f"FACTS: {facts}"
        )
        try:
            data = llm.complete_json(prompt, system=_SYSTEM)
            s = str(data.get("summary") or "").strip()
            if s:
                summary = s
            reasons = data.get("reasons") or {}
            for h in holdings:
                txt = str(reasons.get(h.symbol) or "").strip()
                if txt:
                    h.why = txt
        except Exception:
            pass

    return Explanation(
        summary=summary,
        understood=understood,
        selection=selection,
        construction=construction,
        risk_note=risk_note,
        data_note=data_note,
        highlights=highlights,
        holdings=holdings,
    )
