"""In-memory paper broker: simulates fills, tracks cost basis and realized P&L.

Defense-in-depth: it re-checks cash/shares even though the circuit breaker
already validated — a broker must never trust its caller. Money is rounded to
cents to avoid float drift accumulating across many trades.
"""

from __future__ import annotations

from datetime import UTC, datetime

from smart_investing.broker.base import BrokerAdapter
from smart_investing.domain.types import (
    AccountState,
    Order,
    OrderResult,
    OrderSide,
    OrderStatus,
    OrderType,
    Position,
)


def _cents(x: float) -> float:
    return round(x, 2)


def _now() -> str:
    return datetime.now(UTC).isoformat()


class PaperBroker(BrokerAdapter):
    def __init__(
        self,
        cash: float = 10_000.0,
        prices: dict[str, float] | None = None,
        slippage_bps: float = 0.0,
        day_trades_5d: int = 0,
    ) -> None:
        self._cash = float(cash)
        self._positions: dict[str, Position] = {}
        self._prices: dict[str, float] = dict(prices or {})
        self._slippage = slippage_bps / 1e4
        self._orders: dict[str, OrderResult] = {}
        self._day_trades_5d = day_trades_5d
        self.realized_pnl = 0.0

    # ---- market data ----
    def set_prices(self, prices: dict[str, float]) -> None:
        self._prices.update(prices)

    def get_prices(self, symbols: list[str]) -> dict[str, float]:
        return {s: self._prices[s] for s in symbols if s in self._prices}

    # ---- account ----
    def get_positions(self) -> dict[str, Position]:
        return dict(self._positions)

    def get_buying_power(self) -> float:
        return _cents(self._cash)

    def get_account_state(self) -> AccountState:
        return AccountState(
            cash=_cents(self._cash),
            positions=dict(self._positions),
            buying_power=_cents(self._cash),
            day_trades_5d=self._day_trades_5d,
        )

    def load_state(self, account: AccountState, realized_pnl: float = 0.0) -> None:
        """Rehydrate broker state from a persisted account (used on API restart)."""
        self._cash = float(account.cash)
        self._positions = dict(account.positions)
        self._day_trades_5d = account.day_trades_5d
        self.realized_pnl = realized_pnl

    # ---- execution ----
    def _fill_price(self, order: Order) -> float | None:
        if order.order_type == OrderType.LIMIT and order.limit_price:
            base = order.limit_price
        else:
            base = self._prices.get(order.symbol)
        if base is None or base <= 0:
            return None
        # slippage adverse to the trader
        return base * (1 + self._slippage) if order.side == OrderSide.BUY else base * (1 - self._slippage)

    def place_order(self, order: Order) -> OrderResult:
        px = self._fill_price(order)
        if px is None or px <= 0:
            return self._record(OrderResult(order=order, status=OrderStatus.REJECTED, message=f"no price for {order.symbol}"))

        if order.side == OrderSide.BUY:
            cost = order.quantity * px
            if cost > self._cash + 1e-6:
                return self._record(OrderResult(
                    order=order, status=OrderStatus.REJECTED,
                    message=f"insufficient cash: need ${cost:,.2f}, have ${self._cash:,.2f}",
                ))
            self._cash = _cents(self._cash - cost)
            pos = self._positions.get(order.symbol)
            if pos:
                new_qty = pos.quantity + order.quantity
                new_avg = (pos.cost_basis + cost) / new_qty
                self._positions[order.symbol] = Position(symbol=order.symbol, quantity=new_qty, avg_cost=new_avg)
            else:
                self._positions[order.symbol] = Position(symbol=order.symbol, quantity=order.quantity, avg_cost=px)
            return self._record(OrderResult(
                order=order, status=OrderStatus.FILLED, filled_quantity=order.quantity,
                filled_price=px, broker_order_id=f"paper-{order.id[:8]}", filled_at=_now(),
            ))

        # SELL
        pos = self._positions.get(order.symbol)
        held = pos.quantity if pos else 0.0
        if order.quantity > held + 1e-9:
            return self._record(OrderResult(
                order=order, status=OrderStatus.REJECTED,
                message=f"cannot sell {order.quantity}, hold {held}",
            ))
        proceeds = order.quantity * px
        realized = (px - pos.avg_cost) * order.quantity  # avg_cost unchanged on sells
        self.realized_pnl += realized
        self._cash = _cents(self._cash + proceeds)
        remaining = held - order.quantity
        if remaining <= 1e-9:
            del self._positions[order.symbol]
        else:
            self._positions[order.symbol] = Position(symbol=order.symbol, quantity=remaining, avg_cost=pos.avg_cost)
        return self._record(OrderResult(
            order=order, status=OrderStatus.FILLED, filled_quantity=order.quantity,
            filled_price=px, realized_pnl=realized, broker_order_id=f"paper-{order.id[:8]}", filled_at=_now(),
        ))

    def get_order_status(self, order_id: str) -> OrderResult | None:
        return self._orders.get(order_id)

    def _record(self, result: OrderResult) -> OrderResult:
        self._orders[result.order.id] = result
        return result
