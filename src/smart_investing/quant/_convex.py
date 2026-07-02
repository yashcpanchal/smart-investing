"""Shared cvxpy helpers: solver fallback, PSD wrapping, weight projection.

`cap*n < 1` makes the long-only capped simplex {sum=1, 0<=w<=cap} infeasible.
`effective_cap` relaxes the cap to 1/n in that case so optimization stays
feasible, and `project_capped_simplex` guarantees the returned weights actually
sum to 1 and respect the cap (no solver-noise overshoot, no fallback violation).
"""

from __future__ import annotations

import cvxpy as cp
import numpy as np

from smart_investing.quant.returns import clean_cov

# CLARABEL and SCS ship with cvxpy and support numpy 2.x. ECOS/OSQP are tried if
# present but may not be installed — each solve is guarded.
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
    return cp.psd_wrap(clean_cov(sigma))


def effective_cap(n: int, cap: float) -> float:
    """A per-asset cap below 1/n is infeasible; relax it to 1/n."""
    if n <= 0:
        return cap
    return max(cap, 1.0 / n)


def equal_weight(n: int) -> np.ndarray:
    return np.full(n, 1.0 / n) if n else np.array([])


def project_capped_simplex(w, cap: float) -> np.ndarray:
    """Project weights onto {w >= 0, sum w = 1, w <= cap} via water-filling.

    Guarantees the result respects the cap and sums to 1 (cap is relaxed to 1/n
    if the caller passed an infeasible value). Used on every solver output so
    numerical noise or a fallback can never emit a cap-violating allocation.
    """
    w = np.clip(np.asarray(w, dtype=float), 0.0, None)
    n = len(w)
    if n == 0:
        return w
    cap = effective_cap(n, cap)
    s = w.sum()
    w = np.full(n, 1.0 / n) if s <= 0 else w / s

    for _ in range(100):
        over = w > cap + 1e-12
        if not over.any():
            break
        excess = float((w[over] - cap).sum())
        w[over] = cap
        under = ~over
        under_mass = float(w[under].sum())
        if under_mass <= 0:
            w[:] = cap  # everything capped (cap*n ~= 1)
            break
        w[under] += excess * (w[under] / under_mass)
    return w


def clip_norm(w) -> np.ndarray:
    """Clip tiny negatives and renormalize (no cap). For internal/frontier use."""
    w = np.clip(np.asarray(w, dtype=float), 0.0, None)
    s = w.sum()
    if s <= 0:
        n = len(w)
        return np.full(n, 1.0 / n) if n else w
    return w / s
