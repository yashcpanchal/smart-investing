from __future__ import annotations

import math

from smart_investing.domain.types import (
    AccountState,
    Order,
    OrderSide,
    Position,
    RiskParams,
)
from smart_investing.risk import validate

PRICES = {"AAA": 100.0, "BBB": 50.0, "CCC": 25.0}


def _empty_account(cash=10_000.0, **kw):
    return AccountState(cash=cash, **kw)


def test_valid_buy_passes():
    acct = _empty_account()
    orders = [Order(symbol="AAA", side=OrderSide.BUY, quantity=30, est_price=100.0)]  # $3k of $10k
    res = validate(orders, acct, PRICES, RiskParams(concentration_cap=0.40))
    assert res.ok
    assert not res.fatal


def test_overspend_blocked():
    acct = _empty_account(cash=1_000.0)
    orders = [Order(symbol="AAA", side=OrderSide.BUY, quantity=50, est_price=100.0)]  # $5k > $1k
    res = validate(orders, acct, PRICES)
    assert not res.ok
    assert any(v.code == "insufficient_cash" for v in res.fatal)


def test_short_sell_blocked():
    acct = _empty_account()  # holds nothing
    orders = [Order(symbol="AAA", side=OrderSide.SELL, quantity=10, est_price=100.0)]
    res = validate(orders, acct, PRICES)
    assert not res.ok
    assert any(v.code == "short_sell" for v in res.fatal)


def test_concentration_blocked():
    acct = _empty_account(cash=10_000.0)
    # $5k into one name on $10k equity = 50% > 30% cap
    orders = [Order(symbol="AAA", side=OrderSide.BUY, quantity=50, est_price=100.0)]
    res = validate(orders, acct, PRICES, RiskParams(concentration_cap=0.30))
    assert not res.ok
    assert any(v.code == "concentration" for v in res.fatal)


def test_nan_quantity_short_circuits():
    acct = _empty_account()
    orders = [Order.model_construct(symbol="AAA", side=OrderSide.BUY, quantity=math.nan, est_price=100.0)]
    res = validate(orders, acct, PRICES)
    assert not res.ok
    assert any(v.code == "bad_quantity" for v in res.fatal)


def test_bad_price_blocked():
    acct = _empty_account()
    orders = [Order(symbol="ZZZ", side=OrderSide.BUY, quantity=1, est_price=-5.0)]
    res = validate(orders, acct, {"ZZZ": -5.0})
    assert not res.ok
    assert any(v.code == "bad_price" for v in res.fatal)


def test_pdt_block():
    acct = _empty_account(cash=5_000.0, day_trades_5d=3)  # < $25k and at the limit
    orders = [Order(symbol="AAA", side=OrderSide.BUY, quantity=10, est_price=100.0)]
    res = validate(orders, acct, PRICES)
    assert not res.ok
    assert any(v.code == "pdt_block" for v in res.fatal)


def test_max_positions_block():
    acct = _empty_account(cash=10_000.0)
    orders = [
        Order(symbol="AAA", side=OrderSide.BUY, quantity=10, est_price=100.0),
        Order(symbol="BBB", side=OrderSide.BUY, quantity=10, est_price=50.0),
    ]
    res = validate(orders, acct, PRICES, RiskParams(concentration_cap=0.9, max_positions=1))
    assert any(v.code == "max_positions" for v in res.fatal)


def test_rebalance_sell_funds_buy():
    # hold $6k of AAA, want to rotate $3k into BBB; sells fund buys.
    acct = AccountState(cash=0.0, positions={"AAA": Position(symbol="AAA", quantity=60, avg_cost=100.0)})
    orders = [
        Order(symbol="AAA", side=OrderSide.SELL, quantity=30, est_price=100.0),  # +$3k
        Order(symbol="BBB", side=OrderSide.BUY, quantity=60, est_price=50.0),  # -$3k
    ]
    res = validate(orders, acct, PRICES, RiskParams(concentration_cap=0.6))
    assert res.ok
    assert any(v.code == "settlement_assumption" for v in res.warnings)


def test_est_price_spoof_cannot_bypass_cash():
    # Attacker understates cost via est_price; breaker must use the trusted feed.
    acct = _empty_account(cash=1_000.0)
    orders = [Order(symbol="AAA", side=OrderSide.BUY, quantity=50, est_price=1.0)]  # claims $50
    res = validate(orders, acct, {"AAA": 100.0})  # real cost $5,000
    assert not res.ok
    assert any(v.code == "insufficient_cash" for v in res.fatal)


def test_symbol_absent_from_feed_rejected():
    # A symbol carried only by est_price (not in the trusted feed) must be rejected,
    # not silently dropped from the concentration denominator.
    acct = _empty_account(cash=10_000.0)
    orders = [Order(symbol="ZZZ", side=OrderSide.BUY, quantity=10, est_price=5.0)]
    res = validate(orders, acct, {"AAA": 100.0})
    assert not res.ok
    assert any(v.code == "bad_price" for v in res.fatal)
