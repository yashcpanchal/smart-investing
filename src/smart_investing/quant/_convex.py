"""Shared cvxpy helpers: solver fallback, PSD wrapping, weight cleanup."""

from __future__ import annotations

import cvxpy as cp
import numpy as np

from smart_investing.quant.returns import clean_cov

# CLARABEL and SCS ship with cvxpy and support numpy 2.x. ECOS/OSQP are tried if
# present but may not be installed — each solve is guarded so a missing solver
# degrades gracefully instead of raising.
_SOLVER_ORDER = ["CLARABEL", "SCS", "ECOS", "OSQP"]


def solve_problem(prob: cp.Problem) -> bool:
    for name in _SOLVER_ORDER:
        solver = getattr(cp, name, None)
        if solver is None:
            continue
        try:
            prob.solve(solver=solver)
        except Exception:
            continue
        if prob.status in ("optimal", "optimal_inaccurate") and prob.value is not None:
            return True
    return False


def psd(sigma: np.ndarray) -> cp.expressions.expression.Expression:
    """PSD-wrapped, cleaned covariance for use in cp.quad_form."""
    return cp.psd_wrap(clean_cov(sigma))


def clip_norm(w: np.ndarray) -> np.ndarray:
    """Clip tiny negatives from solver noise and renormalize to sum 1."""
    w = np.clip(np.asarray(w, dtype=float), 0.0, None)
    s = w.sum()
    if s <= 0:
        n = len(w)
        return np.full(n, 1.0 / n) if n else w
    return w / s


def equal_weight(n: int) -> np.ndarray:
    return np.full(n, 1.0 / n) if n else np.array([])
