"""Company profile + industry briefing. Network (yfinance/Gemini) is avoided by
pre-seeding the info cache and forcing the deterministic LLM path."""

from __future__ import annotations

import pytest

from smart_investing import research
from smart_investing.data.store import Store
from smart_investing.llm.gemini import GeminiClient
from smart_investing.research import company_profile, industry_brief
from smart_investing.retrieval.embeddings import TfidfEmbedder
from smart_investing.retrieval.graph_service import GraphService

_NVDA_INFO = {
    "shortName": "NVIDIA Corp",
    "sector": "Technology",
    "industry": "Semiconductors",
    "marketCap": 4.8e12,
    "trailingPE": 30.0,
    "revenueGrowth": 0.85,
    "profitMargins": 0.63,
    "longBusinessSummary": "NVIDIA designs GPUs for AI data centers. It is a large semiconductor company. More text.",
}

ROWS = [
    ("NVDA", "NVIDIA Corp", "gpus and accelerators for ai data centers"),
    ("MU", "Micron Technology", "memory chips for gpus and data centers"),
    ("MSFT", "Microsoft", "cloud hyperscale data centers and ai"),
    ("AMD", "Advanced Micro Devices", "cpus and gpus for data centers"),
]


def _graph() -> GraphService:
    s = Store(":memory:")
    for t, title, text in ROWS:
        s.upsert_company(t, 1, title)
        s.upsert_document(t, 1, f"a-{t}", "business", text)
    return GraphService(s, embedder=TfidfEmbedder()).build()


def test_company_profile_facts_and_metrics():
    research._INFO_CACHE["NVDA"] = _NVDA_INFO
    p = company_profile("NVDA", theme="AI data center", llm=GeminiClient(api_key=""))
    assert p["symbol"] == "NVDA"
    assert p["name"] == "NVIDIA Corp"
    assert p["sector"] == "Technology"
    mcap = next(m for m in p["metrics"] if m["label"] == "Market cap")
    assert mcap["value"] == "$4.8T"
    assert p["theme_fit"]
    assert p["must_knows"]
    assert "NVIDIA" in p["summary"]


def test_company_profile_handles_missing_data():
    research._INFO_CACHE["ZZZZ"] = {}
    p = company_profile("ZZZZ", name="Zeta Corp", theme="x", llm=GeminiClient(api_key=""))
    assert p["symbol"] == "ZZZZ"
    assert p["name"] == "Zeta Corp"
    assert len(p["metrics"]) == 6  # still renders metric rows, with em-dash values
    assert any(m["value"] == "—" for m in p["metrics"])


def test_industry_brief_builds_supply_chain_layers():
    b = industry_brief("ai data center gpus", _graph(), llm=GeminiClient(api_key=""))
    assert b["supply_chain"]
    assert any(layer["layer"].startswith("Core") for layer in b["supply_chain"])
    # every player has a symbol + name
    for layer in b["supply_chain"]:
        for p in layer["players"]:
            assert p["symbol"] and "name" in p
    assert b["grounded"] is False  # no LLM -> deterministic fallback
    assert b["analysis"]


def test_stock_and_industry_endpoints():
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from smart_investing.api.app import create_app
    from smart_investing.broker.paper import PaperBroker
    from smart_investing.persistence.repo import StateRepo

    research._INFO_CACHE["NVDA"] = _NVDA_INFO
    s = Store(":memory:")
    for t, title, text in ROWS:
        s.upsert_company(t, 1, title)
        s.upsert_document(t, 1, f"a-{t}", "business", text)
    app = create_app(
        store=s,
        repo=StateRepo(":memory:"),
        broker=PaperBroker(cash=10_000.0),
        llm=GeminiClient(api_key=""),
        embedder=TfidfEmbedder(),
    )
    c = TestClient(app)

    r = c.get("/api/stock/NVDA", params={"theme": "ai data center"})
    assert r.status_code == 200
    assert r.json()["metrics"]
    assert r.json()["name"] == "NVIDIA Corp"

    r = c.get("/api/industry", params={"theme": "ai data center gpus"})
    assert r.status_code == 200
    assert r.json()["supply_chain"]
