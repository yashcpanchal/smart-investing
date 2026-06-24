from __future__ import annotations

from smart_investing.broker.paper import PaperBroker
from smart_investing.data.store import Store
from smart_investing.llm.gemini import GeminiClient
from smart_investing.persistence.repo import StateRepo
from smart_investing.retrieval.embeddings import TfidfEmbedder
from smart_investing.scheduler import rebalance_once

ROWS = [
    ("QBIT", "Quantum Computing Inc", "superconducting quantum computing processors and qubits."),
    ("COOL", "CryoCool Systems Inc", "cryogenic cooling for quantum computing labs."),
    ("NUKE", "Atomic Power Inc", "small modular nuclear reactors and uranium fuel."),
    ("FOOD", "Tasty Foods Inc", "packaged snacks and beverages for grocery retail."),
]


def _store() -> Store:
    s = Store(":memory:")
    for t, title, text in ROWS:
        s.upsert_company(t, 1, title)
        s.upsert_document(t, 1, f"acc-{t}", "business", text)
    return s


def test_autonomous_rebalance_executes_and_persists():
    broker = PaperBroker(cash=10_000.0)
    repo = StateRepo(":memory:")
    proposal, executed = rebalance_once(
        "quantum computing",
        _store(),
        broker,
        repo,
        autonomous=True,
        live=False,
        llm=GeminiClient(api_key=""),
        embedder=TfidfEmbedder(),
        top_k=4,
    )
    assert proposal.target_weights
    assert len(executed) >= 1
    # portfolio snapshot was persisted and the broker holds positions
    assert broker.get_account_state().positions
    assert repo.get_portfolio() is not None
    assert any(e["event"] == "autonomous_rebalance" for e in repo.audit_log())


def test_manual_rebalance_does_not_execute():
    broker = PaperBroker(cash=10_000.0)
    proposal, executed = rebalance_once(
        "quantum computing",
        _store(),
        broker,
        autonomous=False,
        live=False,
        llm=GeminiClient(api_key=""),
        embedder=TfidfEmbedder(),
        top_k=4,
    )
    assert proposal.target_weights
    assert executed == []
    assert not broker.get_account_state().positions  # nothing executed
