"""Deterministic, non-LLM execution guardrail.

This is the hard gate every order array passes through before it can reach a
broker. It cannot be reasoned around by an LLM — it is plain arithmetic and it
either passes or it halts. Checks run sanity-first (so malformed input can't
poison the notional math), then short, cash, concentration, breadth, and PDT.
"""

from __future__ import annotations

import math
from collections import defaultdict

from smart_investing.domain.types import (
    AccountState,
    Order,
    OrderSide,
    OrderType,
    RiskParams,
    Severity,
    ValidationResult,
    Violation,
)

PDT_EQUITY_THRESHOLD = 25_000.0
PDT_MAX_DAY_TRADES = 3


def validate(
    orders: list[Order],
    account: AccountState,
    prices: dict[str, float],
    risk: RiskParams | None = None,
    *,
    pdt_equity_threshold: float = PDT_EQUITY_THRESHOLD,
) -> ValidationResult:
    risk = risk or RiskParams()
    v: list[Violation] = []

    def order_price(o: Order) -> float | None:
        if o.est_price is not None:
            return o.est_price
        return prices.get(o.symbol)

    # ---- 1. Sanity (short-circuit: bad input makes all later math unsafe) ----
    sane = True
    for o in orders:
        px = order_price(o)
        if not o.symbol or not isinstance(o.symbol, str):
            v.append(Violation(code="bad_symbol", message="empty/invalid symbol"))
            sane = False
        if o.quantity is None or not math.isfinite(o.quantity) or o.quantity <= 0:
            v.append(Violation(code="bad_quantity", message=f"{o.symbol}: non-positive/NaN quantity {o.quantity}"))
            sane = False
        if px is None or not math.isfinite(px) or px <= 0:
            v.append(Violation(code="bad_price", message=f"{o.symbol}: missing/invalid price"))
            sane = False
        if o.order_type == OrderType.LIMIT and (o.limit_price is None or o.limit_price <= 0):
            v.append(Violation(code="bad_limit", message=f"{o.symbol}: limit order needs positive limit_price"))
            sane = False
    if not sane:
        return ValidationResult(ok=False, violations=v)

    def val_price(sym: str) -> float | None:
        p = prices.get(sym)
        if p:
            return p
        if sym in account.positions:
            return account.positions[sym].avg_cost
        return None

    # ---- 2. Aggregate same-symbol orders to a net share delta + cash flows ----
    net_delta: dict[str, float] = defaultdict(float)
    buy_cost = 0.0
    sell_proceeds = 0.0
    for o in orders:
        px = order_price(o)  # guaranteed valid by sanity pass
        if o.side == OrderSide.BUY:
            net_delta[o.symbol] += o.quantity
            buy_cost += o.quantity * px
        else:
            net_delta[o.symbol] -= o.quantity
            sell_proceeds += o.quantity * px

    # ---- 3. No-short: post-trade net position per symbol >= 0 ----
    if not risk.allow_short:
        for sym, delta in net_delta.items():
            held = account.positions[sym].quantity if sym in account.positions else 0.0
            if held + delta < -1e-9:
                v.append(Violation(code="short_sell", message=f"{sym}: sells exceed held shares ({held:.4f}{delta:+.4f} < 0)"))

    # ---- 4. Cash sufficiency (good-faith; assumes same-day sell settlement) ----
    equity = account.equity(prices)
    buffer = risk.cash_buffer * equity
    available = account.cash + sell_proceeds - buffer
    if buy_cost > available + 1e-6:
        v.append(Violation(
            code="insufficient_cash",
            message=(f"buys ${buy_cost:,.2f} exceed available ${available:,.2f} "
                     f"(cash ${account.cash:,.2f} + sells ${sell_proceeds:,.2f} - buffer ${buffer:,.2f})"),
        ))
    if sell_proceeds > 0:
        v.append(Violation(
            code="settlement_assumption",
            message="sell proceeds credited same-day; confirm settlement before live trading",
            severity=Severity.WARN,
        ))

    # ---- 5. Post-trade concentration & breadth ----
    projected: dict[str, float] = {s: p.quantity for s, p in account.positions.items()}
    for sym, delta in net_delta.items():
        projected[sym] = projected.get(sym, 0.0) + delta
    projected = {s: q for s, q in projected.items() if q > 1e-9}

    proj_value = {s: q * val_price(s) for s, q in projected.items() if val_price(s)}
    cash_after = account.cash - buy_cost + sell_proceeds
    equity_after = cash_after + sum(proj_value.values())
    if equity_after > 0:
        for sym, value in proj_value.items():
            wt = value / equity_after
            if wt > risk.concentration_cap + 1e-6:
                v.append(Violation(
                    code="concentration",
                    message=f"{sym}: post-trade weight {wt:.1%} exceeds cap {risk.concentration_cap:.0%}",
                ))
    if len(projected) > risk.max_positions:
        v.append(Violation(
            code="max_positions",
            message=f"{len(projected)} positions exceed max {risk.max_positions}",
        ))

    # ---- 6. Pattern Day Trader protection ----
    if equity < pdt_equity_threshold and account.day_trades_5d >= PDT_MAX_DAY_TRADES:
        v.append(Violation(
            code="pdt_block",
            message=(f"equity ${equity:,.0f} < ${pdt_equity_threshold:,.0f} and "
                     f"{account.day_trades_5d} day-trades in 5d (limit {PDT_MAX_DAY_TRADES})"),
        ))

    ok = not any(x.severity == Severity.FATAL for x in v)
    return ValidationResult(ok=ok, violations=v)
