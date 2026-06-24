from __future__ import annotations

from smart_investing.broker.paper import PaperBroker
from smart_investing.domain.types import Order, OrderSide, OrderStatus


def test_buy_updates_cash_and_position():
    b = PaperBroker(cash=10_000.0, prices={"AAA": 100.0})
    r = b.place_order(Order(symbol="AAA", side=OrderSide.BUY, quantity=10, est_price=100.0))
    assert r.status == OrderStatus.FILLED
    acct = b.get_account_state()
    assert abs(acct.cash - 9_000.0) < 1e-6
    assert acct.positions["AAA"].quantity == 10
    assert abs(acct.positions["AAA"].avg_cost - 100.0) < 1e-6


def test_second_buy_averages_cost():
    b = PaperBroker(cash=10_000.0, prices={"AAA": 100.0})
    b.place_order(Order(symbol="AAA", side=OrderSide.BUY, quantity=10, est_price=100.0))
    b.set_prices({"AAA": 200.0})
    b.place_order(Order(symbol="AAA", side=OrderSide.BUY, quantity=10, est_price=200.0))
    pos = b.get_positions()["AAA"]
    assert pos.quantity == 20
    assert abs(pos.avg_cost - 150.0) < 1e-6  # (1000 + 2000) / 20


def test_sell_realizes_pnl_and_frees_cash():
    b = PaperBroker(cash=10_000.0, prices={"AAA": 100.0})
    b.place_order(Order(symbol="AAA", side=OrderSide.BUY, quantity=10, est_price=100.0))
    b.set_prices({"AAA": 120.0})
    r = b.place_order(Order(symbol="AAA", side=OrderSide.SELL, quantity=10, est_price=120.0))
    assert r.status == OrderStatus.FILLED
    assert abs(r.realized_pnl - 200.0) < 1e-6  # (120-100)*10
    assert abs(b.realized_pnl - 200.0) < 1e-6
    acct = b.get_account_state()
    assert abs(acct.cash - 10_200.0) < 1e-6
    assert "AAA" not in acct.positions  # fully sold -> removed


def test_partial_sell_keeps_avg_cost():
    b = PaperBroker(cash=10_000.0, prices={"AAA": 100.0})
    b.place_order(Order(symbol="AAA", side=OrderSide.BUY, quantity=10, est_price=100.0))
    b.set_prices({"AAA": 130.0})
    b.place_order(Order(symbol="AAA", side=OrderSide.SELL, quantity=4, est_price=130.0))
    pos = b.get_positions()["AAA"]
    assert abs(pos.quantity - 6) < 1e-9
    assert abs(pos.avg_cost - 100.0) < 1e-6  # unchanged on sells


def test_oversell_rejected():
    b = PaperBroker(cash=10_000.0, prices={"AAA": 100.0})
    r = b.place_order(Order(symbol="AAA", side=OrderSide.SELL, quantity=5, est_price=100.0))
    assert r.status == OrderStatus.REJECTED


def test_insufficient_cash_rejected():
    b = PaperBroker(cash=100.0, prices={"AAA": 100.0})
    r = b.place_order(Order(symbol="AAA", side=OrderSide.BUY, quantity=10, est_price=100.0))
    assert r.status == OrderStatus.REJECTED


def test_order_status_lookup():
    b = PaperBroker(cash=10_000.0, prices={"AAA": 100.0})
    o = Order(symbol="AAA", side=OrderSide.BUY, quantity=1, est_price=100.0)
    b.place_order(o)
    assert b.get_order_status(o.id) is not None
    assert b.get_order_status(o.id).broker_order_id is not None
