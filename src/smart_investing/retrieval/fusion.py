"""Rank fusion utilities."""

from __future__ import annotations

import numpy as np


def reciprocal_rank_fusion(rankings: list[list[int]], k: int = 60) -> dict[int, float]:
    """Combine several best-first rankings (lists of item indices) into one
    score per item: sum of 1/(k + rank). Robust to incomparable score scales
    (dense cosine vs BM25), which is why it beats naive score addition."""
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, idx in enumerate(ranking):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank + 1)
    return scores


def argsort_desc(values: np.ndarray) -> list[int]:
    return list(np.argsort(-values))
