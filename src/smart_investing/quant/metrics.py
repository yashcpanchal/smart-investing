"""Performance metrics on a return series / equity curve."""

from __future__ import annotations

import numpy as np

TRADING_DAYS = 252


def annualized_volatility(daily_rets) -> float:
    arr = np.asarray(daily_rets, dtype=float)
    if arr.size < 2:
        return 0.0
    return float(np.std(arr, ddof=1) * np.sqrt(TRADING_DAYS))


def sharpe_ratio(daily_rets, rf_annual: float = 0.04) -> float:
    arr = np.asarray(daily_rets, dtype=float)
    if arr.size < 2:
        return 0.0
    excess = arr - rf_annual / TRADING_DAYS
    sd = np.std(excess, ddof=1)
    if sd <= 0:
        return 0.0
    return float(np.mean(excess) / sd * np.sqrt(TRADING_DAYS))


def max_drawdown(equity_curve) -> float:
    """Most negative peak-to-trough return (<= 0)."""
    eq = np.asarray(equity_curve, dtype=float)
    if eq.size == 0:
        return 0.0
    peak = np.maximum.accumulate(eq)
    dd = (eq - peak) / peak
    return float(dd.min())


def cagr(equity_curve, periods_per_year: int = TRADING_DAYS) -> float:
    eq = np.asarray(equity_curve, dtype=float)
    if eq.size < 2 or eq[0] <= 0:
        return 0.0
    years = eq.size / periods_per_year
    if years <= 0:
        return 0.0
    return float((eq[-1] / eq[0]) ** (1.0 / years) - 1.0)
