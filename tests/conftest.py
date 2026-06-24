from __future__ import annotations

import pytest

from smart_investing.data import synthetic_prices

TICKERS = ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF"]


@pytest.fixture
def prices_df():
    return synthetic_prices(TICKERS, n_days=504, seed=42)


@pytest.fixture
def last_prices(prices_df):
    return {s: float(prices_df[s].iloc[-1]) for s in prices_df.columns}
