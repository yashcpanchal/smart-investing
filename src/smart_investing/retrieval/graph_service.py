"""Interactive supply-chain graph over the corpus.

Powers the explorable canvas and the "go deeper on X" conversation action. Built
once from the store and cached:

  - a hybrid retrieval index  -> theme search + per-node relevance scoring
  - a merged relationship graph: curated supply-chain edges + persisted
    LLM-extracted edges + co-mention edges (bidirectional fallback)

Every node is a corpus company (a US-listed filer we ingested), so everything
shown is tradeable. Edge `direction` is from the *expanded* node's view:
  upstream   = a supplier that feeds this node   (walk "back to the manufacturer")
  downstream = a customer this node feeds
  peer       = a competitor at the same layer
  related    = a co-mention with no typed relationship yet
"""

from __future__ import annotations

from collections import defaultdict

from smart_investing.retrieval.curated import curated_edges
from smart_investing.retrieval.graph import build_comention_graph
from smart_investing.retrieval.index import RetrievalIndex

# how strongly to trust each edge origin when deduping a node-pair
_ORIGIN_RANK = {"curated": 3, "llm": 2, "comention": 1}
_DIR_RANK = {"upstream": 3, "downstream": 3, "peer": 2, "related": 1}


class GraphService:
    def __init__(self, store, embedder=None) -> None:
        self.store = store
        self.embedder = embedder
        self.titles: dict[str, str] = {}
        self.texts: dict[str, str] = {}
        self.index: RetrievalIndex | None = None
        # adj[node] -> list of (other, direction, rel, weight, origin)
        self.adj: dict[str, list[tuple[str, str, str, float, str]]] = defaultdict(list)
        self._relevance_cache: dict[str, dict[str, float]] = {}
        self._built = False

    # ------------------------------------------------------------------ build
    def build(self) -> GraphService:
        self.titles = {t: title for t, _cik, title, _sic in self.store.companies()}
        texts: dict[str, str] = {}
        for ticker, _section, text in self.store.documents():
            texts[ticker] = (texts.get(ticker, "") + " " + text).strip()
        self.texts = texts
        known = set(self.titles)

        items = [(t, texts.get(t, self.titles.get(t, t))) for t in self.titles]
        self.index = RetrievalIndex(embedder=self.embedder).build(items)

        # 1) typed edges: curated seed map + any persisted LLM/curated relations
        typed: list[tuple[str, str, str, float, str]] = [
            (s, d, r, w, "curated") for s, d, r, w in curated_edges(known)
        ]
        for src, dst, rel, weight, origin in self.store.relations():
            if src in known and dst in known:
                typed.append((src, dst, rel, weight, origin))

        seen_pairs: set[tuple[str, str, str]] = set()
        for src, dst, rel, weight, origin in typed:
            key = (src, dst, rel)
            if key in seen_pairs:
                continue
            seen_pairs.add(key)
            if rel == "supplies":
                self.adj[dst].append((src, "upstream", rel, weight, origin))
                self.adj[src].append((dst, "downstream", rel, weight, origin))
            else:  # competes / partner -> symmetric peers
                self.adj[src].append((dst, "peer", rel, weight, origin))
                self.adj[dst].append((src, "peer", rel, weight, origin))

        # 2) co-mention edges as a 'related' fallback (bidirectional)
        cograph = build_comention_graph([(t, self.titles.get(t, "")) for t in texts], texts)
        for node in known:
            for other in cograph.neighbors(node):
                if other in known:
                    self.adj[node].append((other, "related", "co_mention", 1.0, "comention"))
                    self.adj[other].append((node, "related", "co_mention", 1.0, "comention"))

        self._built = True
        return self

    def _ensure(self) -> None:
        if not self._built:
            self.build()

    def retrieval(self) -> tuple[RetrievalIndex, dict]:
        """The cached corpus index + {titles, texts}, for reuse by the optimizer
        pipeline so it doesn't re-embed every filing on each rebuild."""
        self._ensure()
        return self.index, {"titles": self.titles, "texts": self.texts}

    # -------------------------------------------------------------- relevance
    def relevance(self, theme: str) -> dict[str, float]:
        """{ticker: cosine to theme}, 0..1-ish, cached per theme string."""
        self._ensure()
        theme = (theme or "").strip()
        if not theme or self.index is None:
            return {}
        if theme not in self._relevance_cache:
            hits = self.index.query(theme, top_k=len(self.titles) or 1)
            self._relevance_cache[theme] = {h["ticker"]: max(0.0, h["dense"]) for h in hits}
        return self._relevance_cache[theme]

    def node_view(self, ticker: str, theme: str = "") -> dict:
        rel = self.relevance(theme)
        return {
            "symbol": ticker,
            "name": self.titles.get(ticker, ticker),
            "tradeable": ticker in self.titles,
            "relevance": round(rel.get(ticker, 0.0), 4),
        }

    # ----------------------------------------------------------------- search
    def search(self, prompt: str, top_k: int = 8, *, gate: float = 0.4) -> list[dict]:
        """Theme -> seed nodes (direct matches), highest relevance first."""
        self._ensure()
        if self.index is None:
            return []
        hits = self.index.query(prompt, top_k=max(top_k * 3, 12))
        top = max((h["dense"] for h in hits), default=0.0)
        floor = gate * top
        out = []
        for h in hits:
            if h["dense"] < floor:
                continue
            out.append(
                {
                    "symbol": h["ticker"],
                    "name": self.titles.get(h["ticker"], h["ticker"]),
                    "tradeable": True,
                    "relevance": round(h["dense"], 4),
                }
            )
            if len(out) >= top_k:
                break
        return out

    # -------------------------------------------------------------- neighbors
    def neighbors(self, ticker: str, theme: str = "", *, limit: int = 12) -> list[dict]:
        """Connections of `ticker`, deduped to the strongest edge per neighbor and
        ranked by (typed before related, then theme relevance, then weight)."""
        self._ensure()
        ticker = ticker.upper()
        rel = self.relevance(theme)

        best: dict[str, tuple[str, str, float, str]] = {}  # other -> (direction, rel, weight, origin)
        for other, direction, edge_rel, weight, origin in self.adj.get(ticker, []):
            if other == ticker:
                continue
            cur = best.get(other)
            cand = (direction, edge_rel, weight, origin)
            if cur is None or _score_edge(cand) > _score_edge(cur):
                best[other] = cand

        rows = []
        for other, (direction, edge_rel, weight, origin) in best.items():
            rows.append(
                {
                    "symbol": other,
                    "name": self.titles.get(other, other),
                    "direction": direction,
                    "rel": edge_rel,
                    "weight": round(weight, 3),
                    "origin": origin,
                    "tradeable": other in self.titles,
                    "relevance": round(rel.get(other, 0.0), 4),
                }
            )
        rows.sort(key=lambda r: (_DIR_RANK.get(r["direction"], 0), r["relevance"], r["weight"]), reverse=True)
        return rows[:limit]


def _score_edge(e: tuple[str, str, float, str]) -> tuple[int, int, float]:
    direction, _rel, weight, origin = e
    return (_DIR_RANK.get(direction, 0), _ORIGIN_RANK.get(origin, 0), weight)
