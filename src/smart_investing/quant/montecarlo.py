"""Monte-Carlo forward simulation of a portfolio's terminal return.

numpy implementation (fast enough for interactive use). A JAX-accelerated path
(jax.vmap over scenarios) can drop in later behind the same signature when the
`accel` extra is installed — the numpy version is always the fallback.
"""

from __future__ import annotations

import numpy as np

from smart_investing.quant.returns import TRADING_DAYS, clean_cov


def simulate_terminal(
    mu,
    sigma,
    weights,
    horizon_days: int = TRADING_DAYS,
    n_sims: int = 10_000,
    seed: int = 0,
) -> dict[str, float]:
    """Distribution of the portfolio's terminal return over `horizon_days`.

    Collapses the multivariate problem to the 1-D portfolio return
    (mean wᵀμ, variance wᵀΣw), samples daily normals, and compounds.
    """
    w = np.asarray(weights, dtype=float)
    mu = np.asarray(mu, dtype=float)
    port_mu_daily = float(mu @ w) / TRADING_DAYS
    port_var_daily = float(w @ clean_cov(sigma) @ w) / TRADING_DAYS
    sd = np.sqrt(max(port_var_daily, 0.0))

    rng = np.random.default_rng(seed)
    daily = rng.normal(port_mu_daily, sd, size=(n_sims, horizon_days))
    terminal = np.prod(1.0 + daily, axis=1) - 1.0

    return {
        "mean": float(terminal.mean()),
        "p05": float(np.percentile(terminal, 5)),
        "p50": float(np.percentile(terminal, 50)),
        "p95": float(np.percentile(terminal, 95)),
        "prob_loss": float((terminal < 0).mean()),
        "var_95": float(-np.percentile(terminal, 5)),
    }
