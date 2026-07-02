from __future__ import annotations

from smart_investing.data.store import Store
from smart_investing.domain.types import StrategySpec
from smart_investing.retrieval import build_universe
from smart_investing.retrieval.embeddings import TfidfEmbedder

ROWS = [
    ("QBIT", "Quantum Computing Inc",
     "We build superconducting quantum computing processors and qubit control "
     "systems. Our quantum processors require CryoCool cooling hardware."),
    ("COOL", "CryoCool Systems Inc",
     "We manufacture cryogenic cooling and dilution refrigerators used by quantum "
     "computing and superconducting research laboratories."),
    ("NUKE", "Atomic Power Inc",
     "We operate small modular nuclear reactors and sell uranium fuel assemblies "
     "to electric utilities."),
    ("FOOD", "Tasty Foods Inc",
     "We produce packaged snacks and beverages distributed to retail grocery "
     "chains across the country."),
]


def _store() -> Store:
    s = Store(":memory:")
    for ticker, title, text in ROWS:
        s.upsert_company(ticker, 1, title)
        s.upsert_document(ticker, 1, f"acc-{ticker}", "business", text)
    return s


def test_direct_match_ranks_above_irrelevant():
    spec = StrategySpec(themes=["quantum computing"], include_indirect=False)
    u = build_universe("quantum computing processors", _store(), spec, embedder=TfidfEmbedder())
    syms = u.symbols
    assert "QBIT" in syms
    if "FOOD" in syms:
        assert syms.index("QBIT") < syms.index("FOOD")


def test_indirect_comention_surfaces_supplier():
    spec = StrategySpec(include_indirect=True, max_indirect_hops=1)
    u = build_universe("quantum computing", _store(), spec, embedder=TfidfEmbedder())
    # CryoCool is co-mentioned in QBIT's filing -> must appear in the universe.
    assert "COOL" in u.symbols


def test_exclude_symbol_removed():
    spec = StrategySpec(exclude_symbols=["FOOD"], include_indirect=False)
    u = build_universe("snacks beverages grocery retail", _store(), spec, embedder=TfidfEmbedder())
    assert "FOOD" not in u.symbols


def test_degrees_assigned():
    spec = StrategySpec(include_indirect=True, max_indirect_hops=1)
    u = build_universe("quantum computing", _store(), spec, embedder=TfidfEmbedder())
    assert all(a.degree in (1, 2) for a in u.assets)
    assert any(a.degree == 1 for a in u.assets)
