"""Relationship graph for indirect ("supply-chain") discovery.

Baseline edges are deterministic *co-mentions*: company B is linked from company
A if B's name appears in A's filing text. This surfaces non-obvious connected
names that pure semantic search misses (the uranium miner behind "nuclear").
Phase 6 layers LLM-extracted supplier/customer/competitor edges on top via
`add_edge(..., rel=...)` — the graph structure is already in place.
"""

from __future__ import annotations

import re
from collections import defaultdict, deque

_SUFFIXES = re.compile(
    r"\b(inc|incorporated|corp|corporation|company|co|ltd|limited|plc|holdings?|"
    r"group|the|technologies|technology|systems|international|industries)\b"
)


def name_keys(title: str) -> list[str]:
    """Distinctive match key(s) for co-mention detection.

    Multi-word names -> the leading BIGRAM (e.g. "advanced micro"), which almost
    never collides; single-word names -> that token if long enough. Using the
    bigram avoids the false positives a single generic token causes ("advanced",
    "micro", "energy" matching unrelated filings)."""
    t = re.sub(r"[^a-z0-9 ]", " ", title.lower())
    t = _SUFFIXES.sub(" ", t)
    toks = [w for w in t.split() if len(w) >= 4]
    if len(toks) >= 2:
        return [f"{toks[0]} {toks[1]}"]
    if toks and len(toks[0]) >= 5:
        return [toks[0]]
    return []


class RelationGraph:
    def __init__(self) -> None:
        self.adj: dict[str, dict[str, dict]] = defaultdict(dict)

    def add_edge(self, a: str, b: str, rel: str = "related", weight: float = 1.0) -> None:
        if a == b:
            return
        self.adj[a][b] = {"type": rel, "weight": weight}
        self.adj.setdefault(b, {})

    def neighbors(self, node: str) -> dict[str, dict]:
        return self.adj.get(node, {})

    def expand(self, seeds: list[str], hops: int = 1) -> dict[str, int]:
        """BFS from seeds; returns {node: distance} including seeds (distance 0)."""
        dist = {s: 0 for s in seeds}
        frontier = deque(seeds)
        while frontier:
            node = frontier.popleft()
            d = dist[node]
            if d >= hops:
                continue
            for nb in self.adj.get(node, {}):
                if nb not in dist:
                    dist[nb] = d + 1
                    frontier.append(nb)
        return dist


def build_comention_graph(
    companies: list[tuple[str, str]], texts: dict[str, str]
) -> RelationGraph:
    """Edge A→B when B's core name appears in A's filing text.

    companies: list of (ticker, title); texts: {ticker: filing_text}.
    """
    keys = {t: name_keys(title) for t, title in companies}
    keys = {t: k for t, k in keys.items() if k}
    g = RelationGraph()
    for ticker, text in texts.items():
        low = text.lower()
        for other, other_keys in keys.items():
            if other == ticker:
                continue
            if any(re.search(rf"\b{re.escape(k)}\b", low) for k in other_keys):
                g.add_edge(ticker, other, rel="co_mention", weight=1.0)
    return g
