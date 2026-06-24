"""Hybrid retrieval index: dense (embeddings) + sparse (BM25), fused by RRF."""

from __future__ import annotations

import numpy as np

from smart_investing.retrieval.bm25 import BM25
from smart_investing.retrieval.embeddings import get_embedder
from smart_investing.retrieval.fusion import argsort_desc, reciprocal_rank_fusion


class RetrievalIndex:
    def __init__(self, embedder=None) -> None:
        self.embedder = embedder if embedder is not None else get_embedder()
        self.tickers: list[str] = []
        self.texts: list[str] = []
        self.doc_vectors: np.ndarray | None = None
        self.bm25 = BM25()

    def build(self, items: list[tuple[str, str]]) -> RetrievalIndex:
        """items: one (ticker, text) per company."""
        self.tickers = [t for t, _ in items]
        self.texts = [x for _, x in items]
        if self.texts:
            self.embedder.fit(self.texts)
            self.doc_vectors = self.embedder.embed(self.texts)
            self.bm25.fit(self.texts)
        return self

    def query(self, q: str, top_k: int = 25) -> list[dict]:
        if not self.tickers:
            return []
        if self.doc_vectors is not None and self.doc_vectors.shape[1] > 0:
            dense = self.doc_vectors @ self.embedder.embed([q])[0]
        else:
            dense = np.zeros(len(self.tickers))
        sparse = self.bm25.scores(q)
        # k=20 (not 60): sharpens the gap between strong and weak ranks so BM25
        # keyword overlap can't drag off-theme names into the top.
        fused = reciprocal_rank_fusion([argsort_desc(dense), argsort_desc(sparse)], k=20)
        ranked = sorted(fused.items(), key=lambda kv: -kv[1])[:top_k]
        return [
            {
                "ticker": self.tickers[i],
                "rrf": float(s),
                "dense": float(dense[i]),
                "bm25": float(sparse[i]),
            }
            for i, s in ranked
        ]
