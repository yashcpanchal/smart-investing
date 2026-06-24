from __future__ import annotations

from smart_investing.quant import metrics


def test_max_drawdown_known():
    eq = [100.0, 120.0, 60.0, 80.0]  # peak 120 -> trough 60 = -50%
    assert abs(metrics.max_drawdown(eq) - (-0.5)) < 1e-9


def test_max_drawdown_monotonic_up_is_zero():
    assert metrics.max_drawdown([100, 110, 121, 133]) == 0.0


def test_cagr_sign():
    assert metrics.cagr([100, 110, 121, 133, 146]) > 0
    assert metrics.cagr([100, 90, 81, 73]) < 0
    assert metrics.cagr([100, 100, 100]) == 0.0


def test_sharpe_sign():
    up = [0.001] * 252  # steady positive daily returns
    down = [-0.001] * 252
    assert metrics.sharpe_ratio(up, rf_annual=0.0) > 0
    assert metrics.sharpe_ratio(down, rf_annual=0.0) < 0


def test_volatility_nonnegative():
    assert metrics.annualized_volatility([0.01, -0.02, 0.015, -0.005]) >= 0
