"""Okapi BM25 — sparse lexical ranking. Hand-rolled (no dependency) so the
sparse half of hybrid search always works and is testable offline."""

from __future__ import annotations

import math
from collections import Counter

import numpy as np

from smart_investing.retrieval.text import tokenize


class BM25:
    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.docs: list[list[str]] = []
        self.tf: list[Counter] = []
        self.doc_len: list[int] = []
        self.avgdl = 0.0
        self.idf: dict[str, float] = {}
        self.n = 0

    def fit(self, corpus: list[str]) -> BM25:
        self.docs = [tokenize(d) for d in corpus]
        self.n = len(self.docs)
        self.tf = [Counter(d) for d in self.docs]
        self.doc_len = [len(d) for d in self.docs]
        self.avgdl = (sum(self.doc_len) / self.n) if self.n else 0.0
        df: Counter[str] = Counter()
        for d in self.docs:
            df.update(set(d))
        self.idf = {t: math.log(1 + (self.n - f + 0.5) / (f + 0.5)) for t, f in df.items()}
        return self

    def scores(self, query: str) -> np.ndarray:
        q = tokenize(query)
        out = np.zeros(self.n)
        if self.n == 0 or self.avgdl == 0:
            return out
        for i in range(self.n):
            tf, dl = self.tf[i], self.doc_len[i]
            s = 0.0
            for t in q:
                f = tf.get(t, 0)
                if not f:
                    continue
                idf = self.idf.get(t, 0.0)
                s += idf * (f * (self.k1 + 1)) / (f + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
            out[i] = s
        return out
