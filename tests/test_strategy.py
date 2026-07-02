from __future__ import annotations

from smart_investing.data.store import Store
from smart_investing.llm.gemini import GeminiClient
from smart_investing.retrieval.embeddings import TfidfEmbedder
from smart_investing.strategy import compile_strategy

ROWS = [
    ("QBIT", "Quantum Computing Inc", "superconducting quantum computing processors and qubit control."),
    ("COOL", "CryoCool Systems Inc", "cryogenic cooling and dilution refrigerators for quantum computing labs."),
    ("NUKE", "Atomic Power Inc", "small modular nuclear reactors and uranium fuel assemblies."),
    ("FOOD", "Tasty Foods Inc", "packaged snacks and beverages for grocery retail."),
]


def _store() -> Store:
    s = Store(":memory:")
    for t, title, text in ROWS:
        s.upsert_company(t, 1, title)
        s.upsert_document(t, 1, f"acc-{t}", "business", text)
    return s


def test_compile_strategy_offline_end_to_end():
    # Unavailable LLM -> deterministic spec; live=False -> synthetic prices.
    proposal = compile_strategy(
        "quantum computing",
        _store(),
        live=False,
        llm=GeminiClient(api_key=""),
        embedder=TfidfEmbedder(),
        top_k=4,
        initial_cash=10_000.0,
    )
    assert proposal.universe.symbols
    assert proposal.target_weights
    assert abs(sum(proposal.target_weights.values()) - 1.0) < 1e-2
    assert proposal.optimization.volatility >= 0
    assert "circuit breaker" in proposal.rationale.lower()
