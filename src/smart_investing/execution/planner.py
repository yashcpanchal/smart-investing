"""The diff engine: compute the orders that move current holdings → target weights.

Used both for the initial deployment (all cash → target) and for every
bi-weekly rebalance (current positions → new target). Sells are emitted before
buys so freed cash can fund purchases.
"""

from __future__ import annotations

from smart_investing.domain.types import AccountState, Order, OrderSide


def plan_orders(
    target_weights: dict[str, float],
    account: AccountState,
    prices: dict[str, float],
    *,
    investable: float | None = None,
    min_trade_notional: float = 1.0,
    allow_fractional: bool = True,
) -> list[Order]:
    """Generate buy/sell orders bridging current → target allocation.

    `investable` is the dollar value to spread across the target (defaults to
    current account equity). Trades smaller than `min_trade_notional` are skipped
    to avoid churn.
    """
    equity = account.equity(prices) if investable is None else investable

    current_value = {
        s: p.quantity * prices.get(s, p.avg_cost) for s, p in account.positions.items()
    }
    tw_sum = sum(max(0.0, w) for w in target_weights.values()) or 1.0

    symbols = set(target_weights) | set(account.positions)
    orders: list[Order] = []
    for s in sorted(symbols):
        px = prices.get(s)
        if px is None or px <= 0:
            continue
        target_value = (max(0.0, target_weights.get(s, 0.0)) / tw_sum) * equity
        delta_value = target_value - current_value.get(s, 0.0)
        if abs(delta_value) < min_trade_notional:
            continue

        qty = delta_value / px
        if not allow_fractional:
            qty = float(int(qty))  # truncate toward zero
            if qty == 0:
                continue

        if qty > 0:
            orders.append(Order(symbol=s, side=OrderSide.BUY, quantity=abs(qty), est_price=px))
        else:
            held = account.positions[s].quantity if s in account.positions else 0.0
            sell_qty = min(abs(qty), held)
            if sell_qty <= 1e-9:
                continue
            orders.append(Order(symbol=s, side=OrderSide.SELL, quantity=sell_qty, est_price=px))

    # Sells first so proceeds are available to fund buys.
    orders.sort(key=lambda o: 0 if o.side == OrderSide.SELL else 1)
    return orders
