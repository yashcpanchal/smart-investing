"""Return/covariance estimation from a price history."""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def daily_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Simple daily returns; first row (NaN from pct_change) dropped."""
    return prices.pct_change().dropna(how="any")


def annualized_mean(returns: pd.DataFrame) -> pd.Series:
    return returns.mean() * TRADING_DAYS


def annualized_cov(returns: pd.DataFrame) -> pd.DataFrame:
    return returns.cov() * TRADING_DAYS


def clean_cov(sigma: np.ndarray, eps_scale: float = 1e-8) -> np.ndarray:
    """Symmetrize and add a tiny ridge so the matrix is genuinely PSD.

    Ridge is scaled to the matrix trace so it stabilizes near-singular Σ
    (e.g. highly correlated assets) without distorting the optimization.
    """
    sigma = np.asarray(sigma, dtype=float)
    sigma = 0.5 * (sigma + sigma.T)
    n = sigma.shape[0]
    if n == 0:
        return sigma
    ridge = eps_scale * (np.trace(sigma) / n)
    return sigma + ridge * np.eye(n)
