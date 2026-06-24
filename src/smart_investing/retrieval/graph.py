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


def core_name(title: str) -> str:
    """Most distinctive token of a company name (for co-mention matching)."""
    t = re.sub(r"[^a-z0-9 ]", " ", title.lower())
    t = _SUFFIXES.sub(" ", t)
    toks = [w for w in t.split() if len(w) >= 4]
    return max(toks, key=len) if toks else ""


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
    names = {t: core_name(title) for t, title in companies}
    names = {t: n for t, n in names.items() if n}
    g = RelationGraph()
    for ticker, text in texts.items():
        low = text.lower()
        for other, core in names.items():
            if other == ticker:
                continue
            if re.search(rf"\b{re.escape(core)}\b", low):
                g.add_edge(ticker, other, rel="co_mention", weight=1.0)
    return g
