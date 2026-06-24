from __future__ import annotations

import pytest

from smart_investing.broker.paper import PaperBroker
from smart_investing.data.store import Store
from smart_investing.llm.gemini import GeminiClient
from smart_investing.persistence.repo import StateRepo
from smart_investing.retrieval.embeddings import TfidfEmbedder

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from smart_investing.api.app import create_app  # noqa: E402

ROWS = [
    ("QBIT", "Quantum Computing Inc", "superconducting quantum computing processors and qubits."),
    ("COOL", "CryoCool Systems Inc", "cryogenic cooling for quantum computing research labs."),
    ("NUKE", "Atomic Power Inc", "small modular nuclear reactors and uranium fuel."),
    ("FOOD", "Tasty Foods Inc", "packaged snacks and beverages for grocery retail."),
]


def _client() -> TestClient:
    store = Store(":memory:")
    for t, title, text in ROWS:
        store.upsert_company(t, 1, title)
        store.upsert_document(t, 1, f"acc-{t}", "business", text)
    app = create_app(
        store=store,
        repo=StateRepo(":memory:"),
        broker=PaperBroker(cash=10_000.0),
        llm=GeminiClient(api_key=""),  # force deterministic, no network
        embedder=TfidfEmbedder(),  # no model download in tests
    )
    return TestClient(app)


def test_health():
    r = _client().get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_compile_approve_portfolio_flow():
    c = _client()
    r = c.post("/api/compile", json={"prompt": "quantum computing", "live": False, "top_k": 4})
    assert r.status_code == 200
    pid = r.json()["id"]
    assert r.json()["target_weights"]

    approve = c.post(f"/api/proposals/{pid}/approve")
    assert approve.status_code == 200
    body = approve.json()
    assert body["filled"] >= 1

    port = c.get("/api/portfolio")
    assert port.status_code == 200
    assert port.json()["account"]["positions"]


def test_missing_proposal_404():
    assert _client().get("/api/proposals/nope").status_code == 404


def test_rebalance_returns_new_proposal():
    c = _client()
    pid = c.post("/api/compile", json={"prompt": "nuclear", "live": False, "top_k": 4}).json()["id"]
    r = c.post(f"/api/rebalance/{pid}")
    assert r.status_code == 200
    assert r.json()["id"] != pid
