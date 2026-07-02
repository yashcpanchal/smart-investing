"""Deterministic word-window chunking for dense retrieval.

Production dense embedders (e.g. MiniLM, max_seq_length=256 tokens) see only the
first ~1-1.5k characters of a document, but per-ticker filing text runs 35k-150k+
characters — without chunking, dense retrieval silently ignores ~99% of the
corpus. Chunks are overlapping word windows so a fact never straddles a hard
boundary unseen; everything here is plain string splitting (no deps, no
tokenizer downloads) and fully deterministic.
"""

from __future__ import annotations


def chunk_text(
    text: str,
    target_words: int = 400,
    overlap_words: int = 80,
    min_tail_words: int = 50,
) -> list[str]:
    """Split `text` into overlapping windows of ~`target_words` words.

    Consecutive chunks share `overlap_words` words. If the final window would
    add fewer than `min_tail_words` new words, it is folded into the previous
    chunk instead (no near-duplicate tail chunk).
    """
    if target_words <= 0:
        raise ValueError("target_words must be positive")
    if not 0 <= overlap_words < target_words:
        raise ValueError("overlap_words must satisfy 0 <= overlap < target_words")
    words = text.split()
    n = len(words)
    if n == 0:
        return []
    if n <= target_words:
        return [" ".join(words)]

    chunks: list[str] = []
    start = 0
    while start < n:
        end = min(start + target_words, n)
        if 0 < n - end < min_tail_words:  # tiny tail -> extend this chunk to the end
            end = n
        chunks.append(" ".join(words[start:end]))
        if end >= n:
            break
        start = end - overlap_words
    return chunks
