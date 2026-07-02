"""Broker adapters (paper now; Robinhood Agentic MCP scaffold for Phase 11)."""

from smart_investing.broker.base import BrokerAdapter
from smart_investing.broker.paper import PaperBroker
from smart_investing.broker.robinhood import RobinhoodMCPBroker

__all__ = ["BrokerAdapter", "PaperBroker", "RobinhoodMCPBroker"]
