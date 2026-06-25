"""End-to-end orchestrator: prompt -> Proposal.

Ties the whole pipeline together:
  prompt --(LLM)--> StrategySpec --(clarify answers)--> --(retrieval)--> AssetUniverse
        --(prices)--> --(MPT)--> weights --(diff engine)--> orders
        --(circuit breaker)--> validated Proposal --(explain)--> findings.
"""

from __future__ import annotations

from smart_investing.data import load_prices, synthetic_prices
from smart_investing.domain.types import (
    AccountState,
    Objective,
    Proposal,
    RiskParams,
    StrategySpec,
)
from smart_investing.execution.planner import plan_orders
from smart_investing.llm.compiler import compile_spec
from smart_investing.llm.explain import build_explanation
from smart_investing.quant import backtest_constant_weights, optimize
from smart_investing.retrieval.universe import build_universe
from smart_investing.risk import validate

# Look-back window -> (yfinance period, synthetic day count).
_LOOKBACKS = {"1y": ("1y", 252), "2y": ("2y", 504), "3y": ("3y", 756), "5y": ("5y", 1260)}


def _prices_for(symbols: list[str], live: bool = True, lookback: str = "2y"):
    period, n_days = _LOOKBACKS.get(lookback, _LOOKBACKS["2y"])
    if live and len(symbols) >= 2:
        try:
            px = load_prices(symbols, period=period)
            if px.shape[1] >= 2 and px.shape[0] > 60:
                return px, "yfinance (live)"
        except Exception:
            pass
    return synthetic_prices(symbols or ["AAA", "BBB"], n_days=n_days, seed=11), "synthetic (offline)"


def _apply_answers(spec: StrategySpec, answers: dict | None) -> StrategySpec:
    """Map the conversational follow-up answers onto the structured spec. These
    OVERRIDE the LLM's parse because they're the user's explicit, deliberate choice."""
    if not answers:
        return spec
    risk: RiskParams = spec.risk.model_copy()
    objective = spec.objective

    appetite = answers.get("risk")
    if appetite == "low":
        objective = Objective.TARGET_VOL
        risk.target_volatility = risk.target_volatility or 0.12
        risk.concentration_cap = min(risk.concentration_cap, 0.20)
    elif appetite == "high":
        objective = Objective.MAX_SHARPE
        risk.target_volatility = None
        risk.concentration_cap = max(risk.concentration_cap, 0.40)
    elif appetite == "balanced":
        objective = Objective.MAX_SHARPE

    breadth = answers.get("breadth")
    if breadth == "focused":
        risk.concentration_cap = max(risk.concentration_cap, 0.35)
    elif breadth == "diversified":
        risk.concentration_cap = min(risk.concentration_cap, 0.18)

    include_indirect = spec.include_indirect
    if answers.get("supply_chain") == "no":
        include_indirect = False
    elif answers.get("supply_chain") == "yes":
        include_indirect = True

    extra_excludes = answers.get("exclude_symbols") or []
    excludes = list({*spec.exclude_symbols, *(str(s).upper() for s in extra_excludes)})

    return spec.model_copy(
        update={
            "risk": risk,
            "objective": objective,
            "include_indirect": include_indirect,
            "exclude_symbols": excludes,
        }
    )


def _holdings_for(answers: dict | None) -> int:
    """WE decide how many names to cast for — the user never sets this. Breadth and
    risk appetite nudge it; the optimizer then chooses the final non-zero holdings."""
    answers = answers or {}
    base = {"focused": 8, "balanced": 12, "diversified": 18}.get(answers.get("breadth", "balanced"), 12)
    if answers.get("risk") == "low":  # lower risk benefits from a wider spread
        base += 3
    return max(6, min(base, 24))


def compile_strategy(
    prompt: str,
    store,
    *,
    top_k: int | None = None,
    initial_cash: float = 10_000.0,
    live: bool = True,
    embedder=None,
    llm=None,
    account: AccountState | None = None,
    answers: dict | None = None,
    lookback: str = "2y",
) -> Proposal:
    spec = _apply_answers(compile_spec(prompt, llm=llm), answers)
    k = top_k if top_k is not None else _holdings_for(answers)
    universe = build_universe(prompt, store, spec, top_k=k, embedder=embedder, min_relevance=0.12)

    px, source = _prices_for(universe.symbols, live=live, lookback=lookback)
    priced = list(px.columns)
    # keep only names we can actually price/trade
    kept = [a for a in universe.assets if a.symbol in priced]
    if kept:
        universe.assets = kept

    opt = optimize(px, objective=spec.objective, risk=spec.risk)
    last_prices = {s: float(px[s].iloc[-1]) for s in priced}
    account = account or AccountState(cash=initial_cash)
    investable = account.cash if account.cash > 0 else initial_cash
    orders = plan_orders(opt.weights, account, last_prices, investable=investable)
    # Validate against the EFFECTIVE cap (relaxed to 1/n when the universe is too
    # small to honor the requested cap) so the breaker matches what the optimizer
    # can actually achieve, instead of blocking every feasible allocation.
    eff_cap = max(spec.risk.concentration_cap, 1.0 / max(len(priced), 1))
    val_risk = spec.risk.model_copy(update={"concentration_cap": eff_cap})
    vr = validate(orders, account, last_prices, val_risk)
    backtest = backtest_constant_weights(px, opt.weights)

    cap_note = (
        f" (cap relaxed to {eff_cap:.0%} - only {len(priced)} priced names)"
        if eff_cap > spec.risk.concentration_cap + 1e-9
        else ""
    )
    rationale = (
        f"Theme [{', '.join(spec.themes)}] -> {len(priced)} priced names ({source}); "
        f"objective {spec.objective.value}, cap {spec.risk.concentration_cap:.0%}{cap_note}; "
        f"{len(orders)} orders; circuit breaker {'PASS' if vr.ok else 'BLOCK'}"
        + ("" if vr.ok else f" ({len(vr.fatal)} fatal)")
    )

    explanation = build_explanation(
        spec=spec,
        universe=universe,
        opt=opt,
        target_weights=opt.weights,
        backtest=backtest,
        validation=vr,
        effective_cap=eff_cap,
        n_orders=len(orders),
        lookback=lookback,
        price_source=source,
        llm=llm,
    )

    return Proposal(
        spec=spec,
        universe=universe,
        optimization=opt,
        target_weights=opt.weights,
        trades=orders if vr.ok else [],
        backtest=backtest,
        rationale=rationale,
        explanation=explanation,
        blocked=not vr.ok,
        violations=vr.violations,
        lookback=lookback,
        price_source=source,
    )
