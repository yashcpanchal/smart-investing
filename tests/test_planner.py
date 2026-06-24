from __future__ import annotations

from smart_investing.domain.types import AccountState, OrderSide, Position
from smart_investing.execution.planner import plan_orders

PRICES = {"AAA": 100.0, "BBB": 50.0, "CCC": 25.0}


def test_initial_deploy_all_buys():
    acct = AccountState(cash=10_000.0)
    target = {"AAA": 0.5, "BBB": 0.5}
    orders = plan_orders(target, acct, PRICES, investable=10_000.0)
    assert all(o.side == OrderSide.BUY for o in orders)
    notional = sum(o.quantity * o.est_price for o in orders)
    assert abs(notional - 10_000.0) < 1.0


def test_rebalance_emits_sells_before_buys():
    # Hold 100% AAA, want 50/50 AAA/BBB.
    acct = AccountState(
        cash=0.0,
        positions={"AAA": Position(symbol="AAA", quantity=100, avg_cost=100.0)},  # $10k
    )
    target = {"AAA": 0.5, "BBB": 0.5}
    orders = plan_orders(target, acct, PRICES)
    assert orders[0].side == OrderSide.SELL  # sells first
    assert any(o.symbol == "BBB" and o.side == OrderSide.BUY for o in orders)
    assert any(o.symbol == "AAA" and o.side == OrderSide.SELL for o in orders)


def test_skips_tiny_trades():
    acct = AccountState(cash=10_000.0)
    # target weight so small the trade is sub-$1
    target = {"AAA": 0.99999, "BBB": 0.00001}
    orders = plan_orders(target, acct, PRICES, investable=10_000.0, min_trade_notional=1.0)
    assert all(o.symbol != "BBB" for o in orders)
