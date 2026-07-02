"""Hybrid retrieval index: dense (embeddings) + sparse (BM25), fused by RRF.

Dense retrieval is CHUNK-level: each company's text is split into overlapping
word windows (see chunk.py) and every chunk is embedded, so a theme buried deep
in a 100k-char risk-factors section is still visible to the embedder (whose
sequence window covers only ~1 chunk). A company's dense score is the MAX
cosine over its chunks — same scale as a whole-document cosine, so downstream
relevance gates keep their semantics. BM25 stays whole-document: as a bag of
words it already sees the full text, and re-fitting Okapi stats at chunk scale
would change its behavior for no recall gain.
"""

from __future__ import annotations

import logging

import numpy as np

from smart_investing.retrieval.bm25 import BM25
from smart_investing.retrieval.chunk import chunk_text
from smart_investing.retrieval.embeddings import get_embedder
from smart_investing.retrieval.fusion import argsort_desc, reciprocal_rank_fusion

logger = logging.getLogger(__name__)

# Above this, brute-force numpy is still fine but worth a breadcrumb in logs.
_LARGE_CHUNK_COUNT = 20_000


class RetrievalIndex:
    def __init__(
        self,
        embedder=None,
        *,
        chunk_target_words: int = 400,
        chunk_overlap_words: int = 80,
    ) -> None:
        self.embedder = embedder if embedder is not None else get_embedder()
        self.chunk_target_words = chunk_target_words
        self.chunk_overlap_words = chunk_overlap_words
        self.tickers: list[str] = []
        self.texts: list[str] = []
        self.chunk_vectors: np.ndarray | None = None  # (n_chunks, dim)
        self.chunk_owner: np.ndarray | None = None  # (n_chunks,) index into tickers
        self.bm25 = BM25()

    def build(self, items: list[tuple[str, str]]) -> RetrievalIndex:
        """items: one (ticker, text) per company."""
        self.tickers = [t for t, _ in items]
        self.texts = [x for _, x in items]
        chunks: list[str] = []
        owners: list[int] = []
        for i, text in enumerate(self.texts):
            for piece in chunk_text(
                text,
                target_words=self.chunk_target_words,
                overlap_words=self.chunk_overlap_words,
            ):
                chunks.append(piece)
                owners.append(i)
        if self.texts:
            self.bm25.fit(self.texts)  # sparse stays whole-document
        if chunks:
            if len(chunks) > _LARGE_CHUNK_COUNT:
                logger.info(
                    "RetrievalIndex: embedding %d chunks for %d companies "
                    "(brute-force dense scoring)",
                    len(chunks),
                    len(self.tickers),
                )
            self.embedder.fit(chunks)
            self.chunk_vectors = self.embedder.embed(chunks)  # one batched call
            self.chunk_owner = np.asarray(owners, dtype=int)
        return self

    def _dense_scores(self, q: str) -> np.ndarray:
        """Per-ticker dense score = max cosine over the ticker's chunks."""
        n = len(self.tickers)
        if (
            self.chunk_vectors is None
            or self.chunk_owner is None
            or self.chunk_vectors.shape[1] == 0
        ):
            return np.zeros(n)
        sims = self.chunk_vectors @ self.embedder.embed([q])[0]
        dense = np.full(n, -np.inf)
        np.maximum.at(dense, self.chunk_owner, sims)
        dense[np.isneginf(dense)] = 0.0  # tickers with no chunks (empty text)
        return dense

    def query(self, q: str, top_k: int = 25) -> list[dict]:
        if not self.tickers:
            return []
        dense = self._dense_scores(q)
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
