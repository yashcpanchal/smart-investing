"""Chunk-level dense retrieval: the truncation fix.

Production embedders (MiniLM, 256-token window) see only the head of a document;
per-ticker filing text is 35k-150k+ chars. These tests prove that chunking makes
facts buried deep in a document visible to dense retrieval, and that the index's
public contract (return shape, determinism, store ordering) holds.
"""

from __future__ import annotations

from smart_investing.data.store import Store
from smart_investing.retrieval.chunk import chunk_text
from smart_investing.retrieval.embeddings import TfidfEmbedder
from smart_investing.retrieval.index import RetrievalIndex

# ---------------------------------------------------------------------------
# chunk_text
# ---------------------------------------------------------------------------


def test_chunk_empty_and_short():
    assert chunk_text("") == []
    assert chunk_text("   ") == []
    assert chunk_text("one two three") == ["one two three"]  # under target: one chunk


def test_chunk_windows_and_overlap():
    words = [f"w{i}" for i in range(1000)]
    chunks = chunk_text(" ".join(words), target_words=400, overlap_words=80)
    assert len(chunks) > 1
    # every word survives chunking
    seen = set()
    for c in chunks:
        seen.update(c.split())
    assert seen == set(words)
    # consecutive chunks share exactly the overlap
    first, second = chunks[0].split(), chunks[1].split()
    assert first[-80:] == second[:80]
    assert len(first) == 400


def test_chunk_tiny_tail_folded_into_previous():
    # 410 words with target 400/overlap 80 -> tail would add only 10 new words
    words = [f"w{i}" for i in range(410)]
    chunks = chunk_text(" ".join(words), target_words=400, overlap_words=80, min_tail_words=50)
    assert len(chunks) == 1 or all(len(c.split()) >= 50 + 80 for c in chunks[1:])
    # last word must still be present
    assert chunks[-1].split()[-1] == "w409"


def test_chunk_deterministic():
    text = " ".join(f"tok{i % 97}" for i in range(3000))
    assert chunk_text(text) == chunk_text(text)


# ---------------------------------------------------------------------------
# RetrievalIndex with chunk-level dense scoring
# ---------------------------------------------------------------------------


class TruncatingTfidf(TfidfEmbedder):
    """Simulates a bounded-sequence-length dense model (MiniLM sees ~256
    tokens): embeds only the first `max_words` words of each input."""

    def __init__(self, max_words: int = 256, **kw) -> None:
        super().__init__(**kw)
        self.max_words = max_words

    def embed(self, texts):
        return super().embed([" ".join(t.split()[: self.max_words]) for t in texts])


FILLER = " ".join(f"filler{i}" for i in range(1500))
DEEP_FACT = (
    "We fabricate photonic qubit interconnect transceivers for zettascale "
    "quantum networking datacenters. "
) * 4
# distinctive fact starts well past word 400 (past any single embedder window)
LONG_DOC = FILLER + " " + DEEP_FACT.strip()
OTHER_DOC = "We sell packaged snacks and beverages to grocery retail chains nationwide."

ITEMS = [("DEEP", LONG_DOC), ("FOOD", OTHER_DOC)]
QUERY = "photonic qubit interconnect quantum networking"


def test_chunking_defeats_embedder_truncation():
    assert len(FILLER.split()) >= 400  # the fact truly sits past the first window

    # WITH chunking (default): deep fact lands in its own embedded chunk.
    chunked = RetrievalIndex(embedder=TruncatingTfidf()).build(ITEMS)
    hit = {h["ticker"]: h for h in chunked.query(QUERY)}
    assert hit["DEEP"]["dense"] > 0.2

    # WITHOUT chunking (whole doc as one chunk): the truncating embedder never
    # sees the fact -> dense score ~0. This is the old, broken behavior.
    unchunked = RetrievalIndex(
        embedder=TruncatingTfidf(), chunk_target_words=10**9, chunk_overlap_words=0
    ).build(ITEMS)
    old = {h["ticker"]: h for h in unchunked.query(QUERY)}
    assert old["DEEP"]["dense"] < 0.05
    assert hit["DEEP"]["dense"] > old["DEEP"]["dense"] + 0.15


def test_query_return_shape_and_rank_order():
    idx = RetrievalIndex(embedder=TfidfEmbedder()).build(ITEMS)
    hits = idx.query(QUERY, top_k=25)
    assert hits, "expected results"
    for h in hits:
        assert set(h) == {"ticker", "rrf", "dense", "bm25"}
        assert isinstance(h["ticker"], str)
        for key in ("rrf", "dense", "bm25"):
            assert isinstance(h[key], float)
    rrfs = [h["rrf"] for h in hits]
    assert rrfs == sorted(rrfs, reverse=True)
    assert hits[0]["ticker"] == "DEEP"  # on-theme name ranks first


def test_build_and_query_deterministic():
    a = RetrievalIndex(embedder=TfidfEmbedder()).build(ITEMS).query(QUERY)
    b = RetrievalIndex(embedder=TfidfEmbedder()).build(ITEMS).query(QUERY)
    assert a == b


def test_empty_text_company_scores_zero():
    idx = RetrievalIndex(embedder=TfidfEmbedder()).build([("BLANK", ""), *ITEMS])
    hits = {h["ticker"]: h for h in idx.query(QUERY)}
    if "BLANK" in hits:
        assert hits["BLANK"]["dense"] == 0.0


def test_empty_index_query():
    assert RetrievalIndex(embedder=TfidfEmbedder()).build([]).query("anything") == []


# ---------------------------------------------------------------------------
# Store.documents() deterministic ordering
# ---------------------------------------------------------------------------


def test_store_documents_deterministic_section_order():
    s = Store(":memory:")
    s.upsert_company("AAA", 1, "Aaa Inc")
    # insert deliberately out of priority order
    s.upsert_document("AAA", 1, "acc-2", "full", "fallback text")
    s.upsert_document("AAA", 1, "acc-1", "risk_factors", "risk text")
    s.upsert_document("AAA", 1, "acc-1", "business", "business text")
    rows = s.documents()
    sections = [r[1] for r in rows]
    assert sections == ["business", "risk_factors", "full"]
    assert s.documents() == rows  # stable across calls
    s.close()
