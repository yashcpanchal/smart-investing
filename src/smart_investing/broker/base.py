"""Broker abstraction. Everything executes through this interface.

`PaperBroker` (no real money) is the default. `RobinhoodMCPBroker` (Phase 11)
drops in behind the same interface, talking to the Robinhood Agentic MCP. The
extra methods (order id, status, cancel) exist now so that swap needs no changes
upstream — the diff engine and scheduler already speak in order ids.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from smart_investing.domain.types import AccountState, Order, OrderResult, Position


class BrokerAdapter(ABC):
    @abstractmethod
    def get_account_state(self) -> AccountState:
        """Cash, positions, buying power, day-trade count."""

    @abstractmethod
    def get_buying_power(self) -> float: ...

    @abstractmethod
    def get_positions(self) -> dict[str, Position]: ...

    @abstractmethod
    def get_prices(self, symbols: list[str]) -> dict[str, float]:
        """Latest known price per symbol (may be stale for a real broker)."""

    @abstractmethod
    def place_order(self, order: Order) -> OrderResult:
        """Submit an order. `order.id` is the client idempotency key."""

    @abstractmethod
    def get_order_status(self, order_id: str) -> OrderResult | None: ...

    def cancel_order(self, order_id: str) -> bool:  # optional; default no-op
        return False
