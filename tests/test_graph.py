"""Supply-chain graph: curated edge expansion, the service (search + directional
neighbor expansion), and the HTTP endpoints. Uses TfidfEmbedder (no downloads)."""

from __future__ import annotations

import pytest

from smart_investing.data.store import Store
from smart_investing.retrieval.curated import curated_edges
from smart_investing.retrieval.embeddings import TfidfEmbedder
from smart_investing.retrieval.graph_service import GraphService

# A slice of the real corpus that exercises the curated map across two chains:
# the AI/semis chain and the nuclear power/fuel chain.
ROWS = [
    ("NVDA", "NVIDIA Corp", "gpus and accelerators for ai data centers and gaming."),
    ("AMD", "Advanced Micro Devices Inc", "cpus and gpus for data centers, competes with nvidia and intel."),
    ("INTC", "Intel Corp", "cpus and foundry; semiconductor manufacturing."),
    ("MU", "Micron Technology Inc", "memory and storage chips for data centers and gpus."),
    ("AMAT", "Applied Materials Inc", "semiconductor fabrication equipment for chipmakers."),
    ("MSFT", "Microsoft Corp", "cloud azure hyperscale data centers and ai services."),
    ("CEG", "Constellation Energy Corp", "nuclear power generation for the grid and data centers."),
    ("VST", "Vistra Corp", "power generation including nuclear for data centers."),
    ("SMR", "NuScale Power Corp", "small modular nuclear reactors."),
    ("LEU", "Centrus Energy Corp", "uranium enrichment and nuclear fuel."),
    ("CCJ", "Cameco Corp", "uranium mining and fuel supply."),
]


def _store() -> Store:
    s = Store(":memory:")
    for t, title, text in ROWS:
        s.upsert_company(t, 1, title)
        s.upsert_document(t, 1, f"acc-{t}", "business", text)
    return s


def _graph() -> GraphService:
    return GraphService(_store(), embedder=TfidfEmbedder()).build()


# ------------------------------------------------------------------ curated
def test_curated_edges_filter_to_known_universe():
    known = {"NVDA", "MU", "AMD", "MSFT"}
    edges = curated_edges(known)
    assert edges  # some edges form within this slice
    for src, dst, rel, _weight in edges:
        assert src in known and dst in known
        assert rel in {"supplies", "competes"}
    # MU -> NVDA is a known 'supplies' edge; NVDA/AMD are competitors
    assert ("MU", "NVDA", "supplies", 1.0) in edges
    assert any({src, dst} == {"NVDA", "AMD"} and rel == "competes" for src, dst, rel, _ in edges)


# ------------------------------------------------------------------ service
def test_neighbors_label_direction_from_expanded_node():
    g = _graph()
    by_symbol = {n["symbol"]: n for n in g.neighbors("NVDA", theme="ai data center gpus")}
    assert by_symbol["MU"]["direction"] == "upstream"  # memory feeds the GPU
    assert by_symbol["MSFT"]["direction"] == "downstream"  # cloud buys the GPU
    assert by_symbol["AMD"]["direction"] == "peer"  # rival
    assert by_symbol["MU"]["rel"] == "supplies"


def test_walk_back_up_the_nuclear_fuel_chain():
    g = _graph()
    # CEG (utility) <- SMR/LEU upstream; LEU <- CCJ upstream (miner feeds enrichment)
    ceg_up = {n["symbol"] for n in g.neighbors("CEG") if n["direction"] == "upstream"}
    assert {"SMR", "LEU"} & ceg_up
    leu_up = {n["symbol"] for n in g.neighbors("LEU") if n["direction"] == "upstream"}
    assert "CCJ" in leu_up


def test_search_returns_relevant_seed_nodes():
    g = _graph()
    nodes = g.search("nuclear power and uranium", top_k=6)
    syms = {n["symbol"] for n in nodes}
    assert syms  # found something
    assert syms & {"CEG", "VST", "SMR", "LEU", "CCJ"}
    assert all(n["tradeable"] for n in nodes)


def test_neighbors_deduped_one_row_per_symbol():
    g = _graph()
    nbrs = g.neighbors("NVDA")
    syms = [n["symbol"] for n in nbrs]
    assert len(syms) == len(set(syms))


# ---------------------------------------------------------------- endpoints
def test_graph_endpoints():
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from smart_investing.api.app import create_app
    from smart_investing.broker.paper import PaperBroker
    from smart_investing.llm.gemini import GeminiClient
    from smart_investing.persistence.repo import StateRepo

    app = create_app(
        store=_store(),
        repo=StateRepo(":memory:"),
        broker=PaperBroker(cash=10_000.0),
        llm=GeminiClient(api_key=""),
        embedder=TfidfEmbedder(),
    )
    c = TestClient(app)

    r = c.get("/api/graph/search", params={"theme": "ai data center gpus"})
    assert r.status_code == 200
    assert r.json()["nodes"]

    r = c.get("/api/graph/neighbors", params={"node": "NVDA", "theme": "ai data center"})
    assert r.status_code == 200
    body = r.json()
    assert body["node"]["symbol"] == "NVDA"
    assert any(n["symbol"] == "MU" and n["direction"] == "upstream" for n in body["neighbors"])
