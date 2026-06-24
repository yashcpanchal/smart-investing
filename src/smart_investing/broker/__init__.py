"""Broker adapters (paper now; Robinhood Agentic MCP later)."""

from smart_investing.broker.base import BrokerAdapter
from smart_investing.broker.paper import PaperBroker

__all__ = ["BrokerAdapter", "PaperBroker"]
