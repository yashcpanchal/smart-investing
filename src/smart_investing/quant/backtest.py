"""Historical backtest of a fixed target allocation.

Constant-weight (daily-rebalanced) backtest: portfolio return each day is the
weighted sum of asset returns. This matches the assumption MPT optimizes under
and is the standard way to show a candidate allocation's historical behavior.
True drift / periodic-rebalance variants can be added later.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from smart_investing.domain.types import BacktestResult
from smart_investing.quant import metrics
from smart_investing.quant.returns import daily_returns


def backtest_constant_weights(
    prices: pd.DataFrame, weights: dict[str, float], initial: float = 10_000.0
) -> BacktestResult:
    rets = daily_returns(prices)
    symbols = [s for s in weights if s in rets.columns]
    w = np.array([weights[s] for s in symbols], dtype=float)
    if w.sum() > 0:
        w = w / w.sum()

    port = rets[symbols].to_numpy() @ w
    equity = initial * np.cumprod(1.0 + port)
    equity = np.insert(equity, 0, initial)

    total = float(equity[-1] / initial - 1.0)
    start = str(prices.index[0].date()) if len(prices) else ""
    end = str(prices.index[-1].date()) if len(prices) else ""

    return BacktestResult(
        total_return=total,
        cagr=metrics.cagr(equity),
        volatility=metrics.annualized_volatility(port),
        sharpe=metrics.sharpe_ratio(port),
        max_drawdown=metrics.max_drawdown(equity),
        start=start,
        end=end,
        n_days=len(prices),
    )
