"""Price-history adapters.

`load_prices` pulls real adjusted-close history from yfinance (network).
`synthetic_prices` generates deterministic correlated GBM data so the quant
engine and its tests run fully offline and reproducibly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def load_prices(
    symbols: list[str],
    start: str | None = None,
    end: str | None = None,
    period: str = "2y",
    interval: str = "1d",
    auto_adjust: bool = True,
) -> pd.DataFrame:
    """Download adjusted-close prices (dates x symbols) from yfinance.

    Network call. Forward-fills gaps and drops rows missing any symbol so the
    returned frame is rectangular and ready for covariance estimation.
    """
    import yfinance as yf

    symbols = list(symbols)
    data = yf.download(
        symbols,
        start=start,
        end=end,
        period=None if start else period,
        interval=interval,
        auto_adjust=auto_adjust,
        progress=False,
    )
    if isinstance(data.columns, pd.MultiIndex):
        close = data["Close"].copy()
    else:
        # single symbol -> flat columns
        close = data[["Close"]].copy()
        close.columns = symbols
    close = close.ffill().dropna(how="any")
    cols = [s for s in symbols if s in close.columns]
    return close[cols]


def synthetic_prices(
    symbols: list[str],
    n_days: int = 504,
    seed: int = 0,
    start_price: float = 100.0,
    mu_annual: float = 0.10,
    sigma_annual: float = 0.25,
    corr: float = 0.30,
) -> pd.DataFrame:
    """Deterministic correlated geometric-Brownian-motion prices.

    Used for offline tests/demos. Assets share a constant pairwise correlation
    with mild per-asset dispersion in drift/vol so the covariance matrix is
    non-degenerate and the optimizer has a real allocation problem to solve.
    """
    rng = np.random.default_rng(seed)
    n = len(symbols)
    cmat = np.full((n, n), corr)
    np.fill_diagonal(cmat, 1.0)
    chol = np.linalg.cholesky(cmat)

    dt = 1.0 / TRADING_DAYS
    mus = mu_annual * (1.0 + 0.3 * (rng.random(n) - 0.5))
    sigmas = sigma_annual * (1.0 + 0.5 * (rng.random(n) - 0.5))

    z = rng.standard_normal((n_days, n)) @ chol.T
    daily = (mus - 0.5 * sigmas**2) * dt + sigmas * np.sqrt(dt) * z
    log_prices = np.log(start_price) + np.cumsum(daily, axis=0)
    prices = np.exp(log_prices)

    idx = pd.bdate_range(end=pd.Timestamp("2026-06-01"), periods=n_days)
    return pd.DataFrame(prices, columns=list(symbols), index=idx)
