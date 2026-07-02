"""Mean-variance portfolio optimization (Modern Portfolio Theory).

The LLM picks the *assets*; this module picks the *weights*, deterministically.

Objectives (all long-only with a per-asset concentration cap):
  - min_vol     : minimize wᵀΣw
  - max_sharpe  : maximize (μ-rf)ᵀw / sqrt(wᵀΣw) via the Charnes-Cooper transform
  - target_vol  : maximize μᵀw s.t. wᵀΣw ≤ σ*²

Every solver output is projected onto the capped simplex, so the result always
sums to 1 and respects the cap — even on infeasible caps (relaxed to 1/n),
single assets, or degenerate covariances. Solves degrade to min-variance, then
equal-weight, rather than ever raising or returning NaN.
"""

from __future__ import annotations

import cvxpy as cp
import numpy as np
import pandas as pd

from smart_investing.domain.types import FrontierPoint, Objective, OptimizationResult, RiskParams
from smart_investing.quant._convex import (
    effective_cap,
    equal_weight,
    project_capped_simplex,
    psd,
    solve_problem,
)
from smart_investing.quant.returns import annualized_cov, annualized_mean, clean_cov, daily_returns


# --------------------------------------------------------------------------- #
# Low-level weight solvers (operate on numpy μ, Σ)
# --------------------------------------------------------------------------- #
def min_variance(sigma: np.ndarray, cap: float) -> np.ndarray:
    n = sigma.shape[0]
    cap_eff = effective_cap(n, cap)
    w = cp.Variable(n)
    cons = [cp.sum(w) == 1, w >= 0, w <= cap_eff]
    prob = cp.Problem(cp.Minimize(cp.quad_form(w, psd(sigma))), cons)
    if not solve_problem(prob) or w.value is None:
        return project_capped_simplex(equal_weight(n), cap)
    return project_capped_simplex(w.value, cap)


def max_sharpe(mu: np.ndarray, sigma: np.ndarray, cap: float, rf: float) -> np.ndarray:
    """Charnes-Cooper: minimize yᵀΣy s.t. (μ-rf)ᵀy=1, sum(y)=κ, y≤cap·κ; w=y/κ."""
    mu = np.asarray(mu, dtype=float)
    n = len(mu)
    cap_eff = effective_cap(n, cap)
    excess = mu - rf
    if np.max(excess) <= 0:  # nothing beats the risk-free rate → minimize risk
        return min_variance(sigma, cap)
    y = cp.Variable(n)
    k = cp.Variable(nonneg=True)
    cons = [excess @ y == 1, cp.sum(y) == k, y >= 0, y <= cap_eff * k]
    prob = cp.Problem(cp.Minimize(cp.quad_form(y, psd(sigma))), cons)
    if not solve_problem(prob) or y.value is None or k.value is None or k.value < 1e-8:
        return min_variance(sigma, cap)
    return project_capped_simplex(y.value / k.value, cap)


def max_return_target_vol(mu: np.ndarray, sigma: np.ndarray, cap: float, target_vol: float) -> np.ndarray:
    mu = np.asarray(mu, dtype=float)
    n = len(mu)
    cap_eff = effective_cap(n, cap)
    w = cp.Variable(n)
    cons = [cp.sum(w) == 1, w >= 0, w <= cap_eff, cp.quad_form(w, psd(sigma)) <= target_vol**2]
    prob = cp.Problem(cp.Maximize(mu @ w), cons)
    if not solve_problem(prob) or w.value is None:
        return min_variance(sigma, cap)
    return project_capped_simplex(w.value, cap)


def _min_var_at_return(mu: np.ndarray, sigma: np.ndarray, cap: float, target: float) -> np.ndarray | None:
    n = len(mu)
    cap_eff = effective_cap(n, cap)
    w = cp.Variable(n)
    cons = [cp.sum(w) == 1, w >= 0, w <= cap_eff, np.asarray(mu) @ w >= target]
    prob = cp.Problem(cp.Minimize(cp.quad_form(w, psd(sigma))), cons)
    if not solve_problem(prob) or w.value is None:
        return None
    return project_capped_simplex(w.value, cap)


def _max_feasible_return(mu: np.ndarray, sigma: np.ndarray, cap: float) -> float:
    """Greatest return achievable under sum=1, 0≤w≤cap. Falls back to the
    equal-weight return (always feasible) — NOT max(μ), which is unreachable
    once a cap forces diversification."""
    mu = np.asarray(mu, dtype=float)
    n = len(mu)
    cap_eff = effective_cap(n, cap)
    w = cp.Variable(n)
    prob = cp.Problem(cp.Maximize(mu @ w), [cp.sum(w) == 1, w >= 0, w <= cap_eff])
    if not solve_problem(prob) or w.value is None:
        return float(mu @ equal_weight(n))
    return float(mu @ project_capped_simplex(w.value, cap))


# --------------------------------------------------------------------------- #
# Efficient frontier
# --------------------------------------------------------------------------- #
def _point(mu, sig, w, rf) -> FrontierPoint:
    vol = float(np.sqrt(max(w @ sig @ w, 0.0)))
    ret = float(mu @ w)
    sharpe = (ret - rf) / vol if vol > 1e-12 else 0.0
    return FrontierPoint(volatility=vol, expected_return=ret, sharpe=sharpe)


def efficient_frontier(
    mu: np.ndarray, sigma: np.ndarray, cap: float, rf: float, n_points: int = 20
) -> list[FrontierPoint]:
    mu = np.asarray(mu, dtype=float)
    sig = clean_cov(sigma)
    w_mv = min_variance(sigma, cap)
    r_min = float(mu @ w_mv)
    r_max = _max_feasible_return(mu, sigma, cap)

    points: list[FrontierPoint] = [_point(mu, sig, w_mv, rf)]
    if r_max > r_min + 1e-9:
        for target in np.linspace(r_min, r_max, n_points):
            w = _min_var_at_return(mu, sigma, cap, float(target))
            if w is not None:
                points.append(_point(mu, sig, w, rf))
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
    n = len(symbols)
    cap = risk.concentration_cap
    rets = daily_returns(prices)

    # Degenerate input (too little history, NaN stats): equal-weight, zero stats.
    if n == 0 or len(rets) < 2:
        w = project_capped_simplex(equal_weight(n), cap)
        return OptimizationResult(
            weights={s: float(wi) for s, wi in zip(symbols, w, strict=True)},
            expected_return=0.0, volatility=0.0, sharpe=0.0, objective=objective, frontier=[],
        )

    mu = annualized_mean(rets).reindex(symbols).to_numpy()
    sigma = annualized_cov(rets).reindex(index=symbols, columns=symbols).to_numpy()
    if not (np.all(np.isfinite(mu)) and np.all(np.isfinite(sigma))):
        w = project_capped_simplex(equal_weight(n), cap)
        return OptimizationResult(
            weights={s: float(wi) for s, wi in zip(symbols, w, strict=True)},
            expected_return=0.0, volatility=0.0, sharpe=0.0, objective=objective, frontier=[],
        )

    if objective == Objective.MIN_VOL:
        w = min_variance(sigma, cap)
    elif objective == Objective.TARGET_VOL and risk.target_volatility:
        w = max_return_target_vol(mu, sigma, cap, risk.target_volatility)
    else:
        objective = Objective.MAX_SHARPE
        w = max_sharpe(mu, sigma, cap, rf)

    weights = {s: float(wi) for s, wi in zip(symbols, w, strict=True)}
    exp_ret = float(mu @ w)
    var = float(w @ clean_cov(sigma) @ w)
    vol = float(np.sqrt(var)) if np.isfinite(var) and var > 0 else 0.0
    sharpe = (exp_ret - rf) / vol if vol > 1e-12 else 0.0
    frontier = efficient_frontier(mu, sigma, cap, rf) if with_frontier else []

    return OptimizationResult(
        weights=weights, expected_return=exp_ret, volatility=vol, sharpe=sharpe,
        objective=objective, frontier=frontier,
    )
