"""Document embedders.

`get_embedder()` returns a SentenceTransformer embedder when available (better
semantics), else a dependency-free TF-IDF embedder so retrieval always runs and
tests stay offline. Both expose the same `.fit(corpus).embed(texts) -> matrix`
interface and return L2-normalized rows (so dot product == cosine).
"""

from __future__ import annotations

import math
from collections import Counter

import numpy as np

from smart_investing.retrieval.text import tokenize


class TfidfEmbedder:
    """Numpy-only TF-IDF embedder. No torch, deterministic, offline."""

    def __init__(self, max_features: int = 4096) -> None:
        self.max_features = max_features
        self.vocab: dict[str, int] = {}
        self.idf: np.ndarray | None = None

    @property
    def name(self) -> str:
        return "tfidf"

    def fit(self, docs: list[str]) -> TfidfEmbedder:
        df: Counter[str] = Counter()
        for d in docs:
            df.update(set(tokenize(d)))
        common = [t for t, _ in df.most_common(self.max_features)]
        self.vocab = {t: i for i, t in enumerate(common)}
        n = max(1, len(docs))
        self.idf = np.zeros(len(self.vocab))
        for t, i in self.vocab.items():
            self.idf[i] = math.log((1 + n) / (1 + df[t])) + 1.0
        return self

    def embed(self, texts: list[str]) -> np.ndarray:
        dim = len(self.vocab)
        mat = np.zeros((len(texts), dim))
        if dim == 0 or self.idf is None:
            return mat
        for r, txt in enumerate(texts):
            counts = Counter(tokenize(txt))
            for t, c in counts.items():
                j = self.vocab.get(t)
                if j is not None:
                    mat[r, j] = c
            mat[r] *= self.idf
            norm = np.linalg.norm(mat[r])
            if norm > 0:
                mat[r] /= norm
        return mat


class SentenceTransformerEmbedder:
    """Wraps a local sentence-transformers model (downloaded once, then cached)."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)
        self.model_name = model_name

    @property
    def name(self) -> str:
        return f"st:{self.model_name}"

    def fit(self, docs: list[str]) -> SentenceTransformerEmbedder:
        return self  # no fitting needed

    def embed(self, texts: list[str]) -> np.ndarray:
        vecs = self.model.encode(
            list(texts), normalize_embeddings=True, show_progress_bar=False
        )
        return np.asarray(vecs, dtype=float)


def get_embedder(prefer_transformer: bool = True, model_name: str = "all-MiniLM-L6-v2"):
    """Best available embedder. Falls back to TF-IDF if transformers/model
    aren't usable (no torch, offline, download blocked)."""
    if prefer_transformer:
        try:
            return SentenceTransformerEmbedder(model_name)
        except Exception:
            pass
    return TfidfEmbedder()
