"""End-to-end orchestrator: prompt -> Proposal.

Ties the whole pipeline together:
  prompt --(LLM)--> StrategySpec --(retrieval)--> AssetUniverse
        --(prices)--> --(MPT)--> weights --(diff engine)--> orders
        --(circuit breaker)--> validated Proposal.
"""

from __future__ import annotations

from smart_investing.data import load_prices, synthetic_prices
from smart_investing.domain.types import AccountState, Proposal
from smart_investing.execution.planner import plan_orders
from smart_investing.llm.compiler import compile_spec
from smart_investing.quant import backtest_constant_weights, optimize
from smart_investing.retrieval.universe import build_universe
from smart_investing.risk import validate


def _prices_for(symbols: list[str], live: bool = True):
    if live and len(symbols) >= 2:
        try:
            px = load_prices(symbols, period="2y")
            if px.shape[1] >= 2 and px.shape[0] > 60:
                return px, "yfinance (live)"
        except Exception:
            pass
    return synthetic_prices(symbols or ["AAA", "BBB"], n_days=504, seed=11), "synthetic (offline)"


def compile_strategy(
    prompt: str,
    store,
    *,
    top_k: int = 12,
    initial_cash: float = 10_000.0,
    live: bool = True,
    embedder=None,
    llm=None,
    account: AccountState | None = None,
) -> Proposal:
    spec = compile_spec(prompt, llm=llm)
    universe = build_universe(prompt, store, spec, top_k=top_k, embedder=embedder, min_relevance=0.12)

    px, source = _prices_for(universe.symbols, live=live)
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
        f" (cap relaxed to {eff_cap:.0%} — only {len(priced)} priced names)"
        if eff_cap > spec.risk.concentration_cap + 1e-9
        else ""
    )
    rationale = (
        f"Theme [{', '.join(spec.themes)}] -> {len(priced)} priced names ({source}); "
        f"objective {spec.objective.value}, cap {spec.risk.concentration_cap:.0%}{cap_note}; "
        f"{len(orders)} orders; circuit breaker {'PASS' if vr.ok else 'BLOCK'}"
        + ("" if vr.ok else f" ({len(vr.fatal)} fatal)")
    )
    return Proposal(
        spec=spec,
        universe=universe,
        optimization=opt,
        target_weights=opt.weights,
        trades=orders if vr.ok else [],
        backtest=backtest,
        rationale=rationale,
    )
