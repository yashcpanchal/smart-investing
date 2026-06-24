from __future__ import annotations

import numpy as np

from smart_investing.data import synthetic_prices
from smart_investing.domain.types import Objective, RiskParams
from smart_investing.quant import optimize


def _check_simplex(weights, cap):
    vals = list(weights.values())
    assert all(w >= -1e-6 for w in vals), "no negative weights (long-only)"
    assert all(w <= cap + 1e-6 for w in vals), "respects concentration cap"
    assert abs(sum(vals) - 1.0) < 1e-4, "weights sum to 1"


def test_max_sharpe_valid_simplex(prices_df):
    res = optimize(prices_df, objective=Objective.MAX_SHARPE, risk=RiskParams(concentration_cap=0.30))
    _check_simplex(res.weights, 0.30)
    assert res.volatility > 0
    assert len(res.frontier) > 0


def test_min_vol_is_lowest_variance(prices_df):
    mv = optimize(prices_df, objective=Objective.MIN_VOL, risk=RiskParams(concentration_cap=0.40))
    ms = optimize(prices_df, objective=Objective.MAX_SHARPE, risk=RiskParams(concentration_cap=0.40))
    _check_simplex(mv.weights, 0.40)
    # global min-variance portfolio must not have higher vol than any other
    assert mv.volatility <= ms.volatility + 1e-6


def test_max_sharpe_beats_min_vol_on_sharpe(prices_df):
    mv = optimize(prices_df, objective=Objective.MIN_VOL, risk=RiskParams(concentration_cap=0.40))
    ms = optimize(prices_df, objective=Objective.MAX_SHARPE, risk=RiskParams(concentration_cap=0.40))
    assert ms.sharpe >= mv.sharpe - 1e-6


def test_target_vol_respected(prices_df):
    res = optimize(
        prices_df,
        objective=Objective.TARGET_VOL,
        risk=RiskParams(concentration_cap=0.40, target_volatility=0.18),
    )
    assert res.volatility <= 0.18 + 1e-2


def test_concentration_cap_binds(prices_df):
    res = optimize(prices_df, objective=Objective.MAX_SHARPE, risk=RiskParams(concentration_cap=0.20))
    assert max(res.weights.values()) <= 0.20 + 1e-6


def test_frontier_volatility_increases_with_return(prices_df):
    res = optimize(prices_df, objective=Objective.MAX_SHARPE, risk=RiskParams(concentration_cap=0.40))
    fr = res.frontier
    assert fr[-1].volatility >= fr[0].volatility - 1e-6
    assert fr[-1].expected_return >= fr[0].expected_return - 1e-6


def test_degenerate_identical_assets_does_not_crash():
    # perfectly correlated, identical assets -> singular covariance
    px = synthetic_prices(["X", "Y", "Z"], n_days=300, seed=1, corr=0.999999)
    res = optimize(px, objective=Objective.MAX_SHARPE, risk=RiskParams(concentration_cap=0.5))
    assert abs(sum(res.weights.values()) - 1.0) < 1e-3
    assert np.isfinite(res.volatility)
