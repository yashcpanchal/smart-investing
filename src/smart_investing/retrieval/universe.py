"""Theme prompt -> ranked AssetUniverse.

The "search": hybrid retrieval (dense + sparse) finds the direct thematic
matches (degree 1); graph expansion pulls in indirect supply-chain names
(degree 2); a configurable fusion ranks them. This is what makes the product
better than asking an LLM for tickers — it retrieves over a real, current corpus
and surfaces non-obvious connected names, all inspectable.
"""

from __future__ import annotations

from smart_investing.domain.types import AssetUniverse, SourceWeights, StrategySpec, UniverseAsset
from smart_investing.retrieval.graph import build_comention_graph
from smart_investing.retrieval.index import RetrievalIndex


def build_index_from_store(store, embedder=None) -> tuple[RetrievalIndex, dict]:
    """One combined (business + risk) document per ticker -> hybrid index."""
    titles = {t: title for t, _cik, title, _sic in store.companies()}
    texts: dict[str, str] = {}
    for ticker, _section, text in store.documents():
        texts[ticker] = (texts.get(ticker, "") + " " + text).strip()
    items = [(t, texts[t]) for t in texts]
    index = RetrievalIndex(embedder=embedder).build(items)
    return index, {"titles": titles, "texts": texts}


def _composite(asset: UniverseAsset, source_weights: SourceWeights | None = None) -> float:
    """Rank score: retrieval (direct) / graph proximity (indirect), blended with
    the smart-money signals via spec.source_weights (the Phase 5b fusion). The
    blend only REORDERS names already admitted by the relevance gate — with
    zero weights (or no smart-money data) it reduces exactly to the base score."""
    if asset.degree == 1:
        base = asset.scores.get("retrieval", 0.0)
    else:
        base = 0.5 * asset.scores.get("graph_proximity", 0.0)
    if source_weights is None:
        return base
    return (
        base
        + source_weights.sec_13f * asset.scores.get("smart_money_13f", 0.0)
        + source_weights.insider * asset.scores.get("smart_money_insider", 0.0)
    )


def build_universe(
    prompt: str,
    store,
    spec: StrategySpec | None = None,
    *,
    top_k: int = 15,
    embedder=None,
    relevance_gate: float = 0.5,
    min_relevance: float = 0.0,
    index: RetrievalIndex | None = None,
    meta: dict | None = None,
    smart_money: dict[str, dict[str, float]] | None = None,
) -> AssetUniverse:
    """relevance_gate: keep a direct hit only if its cosine >= gate * top cosine
    (drops bottom-half noise, embedder-agnostic). min_relevance: optional
    absolute cosine floor for the 'everything is weak' case (set per embedder).

    index/meta: a prebuilt, cached retrieval index (+ {titles, texts}) to reuse.
    Embedding the whole corpus is the slow step; passing a shared index keeps each
    conversational rebuild fast instead of re-embedding every filing each turn."""
    spec = spec or StrategySpec(raw_prompt=prompt, themes=[prompt])
    # Retrieve on the (LLM-)extracted themes, not the raw chat message. A verbose
    # prompt ("I want to invest in quantum and its related supply chain") flattens
    # the embedding similarities and lets off-theme names (e.g. consumer staples)
    # slip past the relevance gate; the clean themes give a sharp on/off-theme break.
    query = " ".join(spec.themes).strip() or prompt
    if index is None or meta is None:
        index, meta = build_index_from_store(store, embedder=embedder)
    titles, texts = meta["titles"], meta["texts"]
    exclude = {s.upper() for s in spec.exclude_symbols}

    hits = index.query(query, top_k=top_k * 3)
    max_rrf = max((h["rrf"] for h in hits), default=1.0) or 1.0
    top_dense = max((h["dense"] for h in hits if h["ticker"] not in exclude), default=0.0)
    floor = max(min_relevance, relevance_gate * top_dense)

    assets: list[UniverseAsset] = []
    seen: set[str] = set()

    # Degree 1 — direct matches passing the relevance gate. `relevance` is the
    # RAW dense cosine (true on/off-theme separation); `retrieval` is the RRF rank.
    for h in hits:
        t = h["ticker"]
        if t in exclude or t in seen:
            continue
        if h["dense"] < floor:  # gate out weak/off-theme names
            continue
        seen.add(t)
        assets.append(
            UniverseAsset(
                symbol=t,
                name=titles.get(t, ""),
                degree=1,
                rationale="direct thematic match",
                scores={
                    "relevance": round(h["dense"], 4),
                    "retrieval": round(h["rrf"] / max_rrf, 4),
                    "bm25": round(h["bm25"], 4),
                },
            )
        )

    # Degree 2 — indirect supply-chain co-mentions
    if spec.include_indirect and spec.max_indirect_hops > 0:
        graph = build_comention_graph([(t, titles.get(t, "")) for t in texts], texts)
        seeds = [h["ticker"] for h in hits[: max(3, top_k // 2)]]
        for t, dist in graph.expand(seeds, hops=spec.max_indirect_hops).items():
            if dist == 0 or t in seen or t in exclude or t not in texts:
                continue
            seen.add(t)
            assets.append(
                UniverseAsset(
                    symbol=t,
                    name=titles.get(t, ""),
                    degree=2,
                    rationale=f"indirect: supply-chain co-mention ({dist} hop)",
                    scores={"graph_proximity": round(1.0 / (1 + dist), 4)},
                )
            )

    # User-pinned symbols always included
    for s in spec.include_symbols:
        s = s.upper()
        if s in seen or s in exclude:
            continue
        seen.add(s)
        assets.append(UniverseAsset(symbol=s, name=titles.get(s, ""), degree=1, rationale="user-specified"))

    # Smart-money sub-scores on every asset (0.0 when no data). They influence
    # ONLY the ordering below — never the relevance gate above, so smart money
    # can reorder on-theme names but can't re-admit an off-theme one.
    for a in assets:
        sm = (smart_money or {}).get(a.symbol, {})
        a.scores["smart_money_13f"] = round(sm.get("smart_money_13f", 0.0), 4)
        a.scores["smart_money_insider"] = round(sm.get("smart_money_insider", 0.0), 4)

    assets.sort(key=lambda a: (a.degree, -_composite(a, spec.source_weights)))
    return AssetUniverse(theme=query, assets=assets[:top_k])
