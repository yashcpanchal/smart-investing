"""Execute a validated Proposal against a broker (shared by the API approve
endpoint and the autonomous rebalancer)."""

from __future__ import annotations

from smart_investing.broker.base import BrokerAdapter
from smart_investing.domain.types import OrderResult, Proposal


def execute_proposal(proposal: Proposal, broker: BrokerAdapter) -> list[OrderResult]:
    if not proposal.trades:
        return []
    if hasattr(broker, "set_prices"):
        broker.set_prices({o.symbol: o.est_price for o in proposal.trades if o.est_price})
    return [broker.place_order(o) for o in proposal.trades]
