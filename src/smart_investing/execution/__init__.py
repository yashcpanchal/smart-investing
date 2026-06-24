"""Execution: planning (diff engine) and executing proposals against a broker."""

from smart_investing.execution.executor import execute_proposal
from smart_investing.execution.planner import plan_orders

__all__ = ["plan_orders", "execute_proposal"]
