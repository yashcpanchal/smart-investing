"""Conversational layer: session state, the keyword-fallback agent, and the
/api/chat loop (interpret -> apply -> rebuild -> reply). Deterministic path
(GeminiClient(api_key="") + live=False), no network."""

from __future__ import annotations

import pytest

from smart_investing.llm.agent import interpret
from smart_investing.session import Session, SessionManager

ROWS = [
    ("NVDA", "NVIDIA Corp", "gpus and accelerators for ai data centers and gaming."),
    ("AMD", "Advanced Micro Devices Inc", "cpus and gpus for data centers, competes with nvidia."),
    ("MU", "Micron Technology Inc", "memory and storage chips for data centers and gpus."),
    ("MSFT", "Microsoft Corp", "cloud azure hyperscale data centers and ai services."),
    ("KO", "Coca Cola Co", "beverages and snacks for retail."),
]


# ----------------------------------------------------------------- session
def test_session_pin_exclude_and_answers():
    s = Session(id="x")
    assert s.pin(["nvda", "MU"]) == ["NVDA", "MU"]
    assert s.exclude(["mu"]) == ["MU"]  # excluding a pinned name moves it out of pinned
    assert "MU" not in s.pinned and "MU" in s.excluded
    s.pin(["MU"])  # re-pinning un-excludes and restores it
    assert "MU" in s.pinned and "MU" not in s.excluded
    ca = s.compile_answers()
    assert "NVDA" in ca["include_symbols"]
    assert ca["risk"] == "balanced"


def test_session_manager_roundtrip():
    m = SessionManager()
    s = m.create()
    assert m.get(s.id) is s
    assert m.get("nope") is None
    assert m.get_or_create(None).id != s.id


# ------------------------------------------------------------- agent fallback
def _ctx(holdings_syms):
    return {"theme": "ai data center", "holdings": [{"symbol": s, "weight": 0.2} for s in holdings_syms]}


def test_agent_fallback_risk_down():
    out = interpret("can you make it safer please", _ctx(["NVDA"]), llm=None)
    assert {"op": "set_risk", "value": "low"} in out["actions"]


def test_agent_fallback_remove():
    out = interpret("drop KO", _ctx(["NVDA", "KO"]), llm=None)
    assert any(a["op"] == "remove" and "KO" in a["symbols"] for a in out["actions"])


def test_agent_fallback_expand_deeper():
    out = interpret("go deeper on NVDA, show me the suppliers", _ctx(["NVDA"]), llm=None)
    assert any(a["op"] == "expand" and a["symbol"] == "NVDA" for a in out["actions"])


def test_agent_fallback_diversify():
    out = interpret("spread it out across more names", _ctx(["NVDA"]), llm=None)
    assert {"op": "set_breadth", "value": "diversified"} in out["actions"]


# ---------------------------------------------------------------- /api/chat
def _client(llm=None):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from smart_investing.api.app import create_app
    from smart_investing.broker.paper import PaperBroker
    from smart_investing.data.store import Store
    from smart_investing.llm.gemini import GeminiClient
    from smart_investing.persistence.repo import StateRepo
    from smart_investing.retrieval.embeddings import TfidfEmbedder

    store = Store(":memory:")
    for t, title, text in ROWS:
        store.upsert_company(t, 1, title)
        store.upsert_document(t, 1, f"acc-{t}", "business", text)
    app = create_app(
        store=store,
        repo=StateRepo(":memory:"),
        broker=PaperBroker(cash=10_000.0),
        llm=llm if llm is not None else GeminiClient(api_key=""),
        embedder=TfidfEmbedder(),
    )
    return TestClient(app)


def test_chat_first_message_builds_then_refines():
    c = _client()
    # opening thesis -> builds a portfolio
    r = c.post("/api/chat", json={"message": "ai data center gpus", "live": False})
    assert r.status_code == 200
    b = r.json()
    sid = b["session_id"]
    assert b["rebuilt"] is True
    assert b["proposal"] is not None
    assert b["state"]["theme"]

    # refine: make it safer -> risk knob flips, rebuilds
    r2 = c.post("/api/chat", json={"message": "make it safer", "session_id": sid, "live": False}).json()
    assert r2["state"]["risk"] == "low"
    assert r2["rebuilt"] is True
    assert r2["proposal"]["spec"]["objective"] == "target_vol"

    # remove a name -> shows up in excluded and not in trades
    r3 = c.post("/api/chat", json={"message": "drop KO", "session_id": sid, "live": False}).json()
    assert "KO" in r3["state"]["excluded"]
    assert "KO" not in r3["proposal"]["target_weights"] or r3["proposal"]["target_weights"].get("KO", 0) == 0


def test_chat_expand_pins_supply_chain_neighbors():
    c = _client()
    sid = c.post("/api/chat", json={"message": "ai data center gpus", "live": False}).json()["session_id"]
    before = set(c.post("/api/chat", json={"message": "status", "session_id": sid, "live": False}).json()["state"]["pinned"])
    r = c.post("/api/chat", json={"message": "go deeper on NVDA suppliers", "session_id": sid, "live": False}).json()
    after = set(r["state"]["pinned"])
    assert after > before  # at least one neighbor pinned


def test_chat_agent_read_tools_run_against_real_objects():
    """A scripted LLM exercises every read-tool closure in /api/chat against the
    real Store/Graph/Proposal — catches signature drift the fallback path can't."""
    from smart_investing.llm.base import LLMClient, LLMResponse, ToolCall

    class ScriptedLLM(LLMClient):
        provider = "fake"

        def __init__(self):
            super().__init__(api_key="fake", model="fake-1")
            self.script = [
                LLMResponse(tool_calls=[
                    ToolCall(name="get_portfolio", arguments={}, id="c1"),
                    ToolCall(name="get_neighbors", arguments={"symbol": "NVDA"}, id="c2"),
                    ToolCall(name="search_companies", arguments={"query": "memory chips"}, id="c3"),
                ]),
                LLMResponse(text="NVDA leads; MU supplies the memory."),
            ]

        def chat(self, messages, *, system=None, tools=None, json_mode=False, temperature=0.2):
            if json_mode:  # compile_spec / explain paths -> use their fallbacks
                raise RuntimeError("no json in this fake")
            return self.script.pop(0) if self.script else LLMResponse(text="ok")

    c = _client(llm=ScriptedLLM())
    sid = c.post("/api/chat", json={"message": "ai data center gpus", "live": False}).json()["session_id"]
    r = c.post("/api/chat", json={"message": "why NVDA?", "session_id": sid, "live": False})
    assert r.status_code == 200
    b = r.json()
    assert set(b["researched"]) == {"get_portfolio", "get_neighbors", "search_companies"}
    assert b["reply"] == "NVDA leads; MU supplies the memory."
