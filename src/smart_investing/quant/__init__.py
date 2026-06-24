"""Deterministic quant engine: optimization, backtest, simulation, metrics."""

from smart_investing.quant.backtest import backtest_constant_weights
from smart_investing.quant.montecarlo import simulate_terminal
from smart_investing.quant.optimizer import (
    efficient_frontier,
    max_sharpe,
    min_variance,
    optimize,
)

__all__ = [
    "optimize",
    "efficient_frontier",
    "max_sharpe",
    "min_variance",
    "backtest_constant_weights",
    "simulate_terminal",
]
