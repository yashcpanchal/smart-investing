"""Mean-variance portfolio optimization (Modern Portfolio Theory).

The LLM picks the *assets*; this module picks the *weights*, deterministically.

Objectives (all long-only with a per-asset concentration cap):
  - min_vol     : minimize wᵀΣw
  - max_sharpe  : maximize (μ-rf)ᵀw / sqrt(wᵀΣw) via the Charnes-Cooper transform
  - target_vol  : maximize μᵀw s.t. wᵀΣw ≤ σ*²

All solves go through a solver-fallback chain and degrade to min-variance (then
equal-weight) rather than ever raising.
"""

from __future__ import annotations

import cvxpy as cp
import numpy as np
import pandas as pd

from smart_investing.domain.types import FrontierPoint, Objective, OptimizationResult, RiskParams
from smart_investing.quant._convex import clip_norm, equal_weight, psd, solve_problem
from smart_investing.quant.returns import (
    annualized_cov,
    annualized_mean,
    clean_cov,
    daily_returns,
)


# --------------------------------------------------------------------------- #
# Low-level weight solvers (operate on numpy μ, Σ)
# --------------------------------------------------------------------------- #
def min_variance(sigma: np.ndarray, cap: float) -> np.ndarray:
    n = sigma.shape[0]
    w = cp.Variable(n)
    cons = [cp.sum(w) == 1, w >= 0, w <= cap]
    prob = cp.Problem(cp.Minimize(cp.quad_form(w, psd(sigma))), cons)
    if not solve_problem(prob) or w.value is None:
        return equal_weight(n)
    return clip_norm(w.value)


def max_sharpe(mu: np.ndarray, sigma: np.ndarray, cap: float, rf: float) -> np.ndarray:
    """Charnes-Cooper: minimize yᵀΣy s.t. (μ-rf)ᵀy=1, sum(y)=κ, y≤cap·κ; w=y/κ."""
    mu = np.asarray(mu, dtype=float)
    n = len(mu)
    excess = mu - rf
    if np.max(excess) <= 0:  # no asset beats the risk-free rate → just minimize risk
        return min_variance(sigma, cap)
    y = cp.Variable(n)
    k = cp.Variable(nonneg=True)
    cons = [excess @ y == 1, cp.sum(y) == k, y >= 0, y <= cap * k]
    prob = cp.Problem(cp.Minimize(cp.quad_form(y, psd(sigma))), cons)
    if not solve_problem(prob) or y.value is None or k.value is None or k.value < 1e-8:
        return min_variance(sigma, cap)
    return clip_norm(y.value / k.value)


def max_return_target_vol(mu: np.ndarray, sigma: np.ndarray, cap: float, target_vol: float) -> np.ndarray:
    mu = np.asarray(mu, dtype=float)
    n = len(mu)
    w = cp.Variable(n)
    cons = [cp.sum(w) == 1, w >= 0, w <= cap, cp.quad_form(w, psd(sigma)) <= target_vol**2]
    prob = cp.Problem(cp.Maximize(mu @ w), cons)
    if not solve_problem(prob) or w.value is None:
        return min_variance(sigma, cap)
    return clip_norm(w.value)


def _min_var_at_return(mu: np.ndarray, sigma: np.ndarray, cap: float, target: float) -> np.ndarray | None:
    n = len(mu)
    w = cp.Variable(n)
    cons = [cp.sum(w) == 1, w >= 0, w <= cap, np.asarray(mu) @ w >= target]
    prob = cp.Problem(cp.Minimize(cp.quad_form(w, psd(sigma))), cons)
    if not solve_problem(prob) or w.value is None:
        return None
    return clip_norm(w.value)


def _max_feasible_return(mu: np.ndarray, sigma: np.ndarray, cap: float) -> float:
    """Greatest return achievable under sum=1, 0≤w≤cap (NOT max(μ), which is
    infeasible once a cap forces diversification)."""
    mu = np.asarray(mu, dtype=float)
    n = len(mu)
    w = cp.Variable(n)
    prob = cp.Problem(cp.Maximize(mu @ w), [cp.sum(w) == 1, w >= 0, w <= cap])
    if not solve_problem(prob) or w.value is None:
        return float(np.max(mu))
    return float(mu @ clip_norm(w.value))


# --------------------------------------------------------------------------- #
# Efficient frontier
# --------------------------------------------------------------------------- #
def efficient_frontier(
    mu: np.ndarray, sigma: np.ndarray, cap: float, rf: float, n_points: int = 20
) -> list[FrontierPoint]:
    mu = np.asarray(mu, dtype=float)
    sig = clean_cov(sigma)
    w_mv = min_variance(sigma, cap)
    r_min = float(mu @ w_mv)
    r_max = _max_feasible_return(mu, sigma, cap)
    if r_max <= r_min:
        r_max = r_min + 1e-4
    points: list[FrontierPoint] = []
    for target in np.linspace(r_min, r_max, n_points):
        w = _min_var_at_return(mu, sigma, cap, float(target))
        if w is None:
            continue
        vol = float(np.sqrt(max(w @ sig @ w, 0.0)))
        ret = float(mu @ w)
        sharpe = (ret - rf) / vol if vol > 1e-12 else 0.0
        points.append(FrontierPoint(volatility=vol, expected_return=ret, sharpe=sharpe))
    return points


# --------------------------------------------------------------------------- #
# Top-level entry: prices -> OptimizationResult
# --------------------------------------------------------------------------- #
def optimize(
    prices: pd.DataFrame,
    *,
    objective: Objective = Objective.MAX_SHARPE,
    risk: RiskParams | None = None,
    rf: float = 0.04,
    with_frontier: bool = True,
) -> OptimizationResult:
    risk = risk or RiskParams()
    symbols = list(prices.columns)
    rets = daily_returns(prices)
    mu = annualized_mean(rets).reindex(symbols).to_numpy()
    sigma = annualized_cov(rets).reindex(index=symbols, columns=symbols).to_numpy()
    cap = risk.concentration_cap

    if objective == Objective.MIN_VOL:
        w = min_variance(sigma, cap)
    elif objective == Objective.TARGET_VOL and risk.target_volatility:
        w = max_return_target_vol(mu, sigma, cap, risk.target_volatility)
    else:
        objective = Objective.MAX_SHARPE
        w = max_sharpe(mu, sigma, cap, rf)

    weights = {s: float(wi) for s, wi in zip(symbols, w, strict=True)}
    exp_ret = float(mu @ w)
    vol = float(np.sqrt(max(w @ clean_cov(sigma) @ w, 0.0)))
    sharpe = (exp_ret - rf) / vol if vol > 1e-12 else 0.0
    frontier = efficient_frontier(mu, sigma, cap, rf) if with_frontier else []

    return OptimizationResult(
        weights=weights,
        expected_return=exp_ret,
        volatility=vol,
        sharpe=sharpe,
        objective=objective,
        frontier=frontier,
    )
