"""Agent polish: SSE chat streaming, the web_research read tool, and the
grounded industry-brief path. Fully offline — scripted LLMs, Store(":memory:"),
TfidfEmbedder; no network, no API keys."""

from __future__ import annotations

import json

import pytest

from smart_investing.data.store import Store
from smart_investing.llm.base import LLMClient, LLMResponse, ToolCall
from smart_investing.research import industry_brief
from smart_investing.retrieval.embeddings import TfidfEmbedder
from smart_investing.retrieval.graph_service import GraphService

ROWS = [
    ("NVDA", "NVIDIA Corp", "gpus and accelerators for ai data centers and gaming."),
    ("AMD", "Advanced Micro Devices Inc", "cpus and gpus for data centers, competes with nvidia."),
    ("MU", "Micron Technology Inc", "memory and storage chips for data centers and gpus."),
    ("MSFT", "Microsoft Corp", "cloud azure hyperscale data centers and ai services."),
    ("KO", "Coca Cola Co", "beverages and snacks for retail."),
]


class ScriptedLLM(LLMClient):
    """Plays back a chat script; JSON-mode calls raise so compile/explain fall back."""

    provider = "fake"

    def __init__(self, script: list[LLMResponse]):
        super().__init__(api_key="fake", model="fake-1")
        self.script = list(script)

    def chat(self, messages, *, system=None, tools=None, json_mode=False, temperature=0.2):
        if json_mode:
            raise RuntimeError("no json in this fake")
        return self.script.pop(0) if self.script else LLMResponse(text="ok")


class GroundedLLM(ScriptedLLM):
    """ScriptedLLM whose complete_grounded returns a canned (text, sources)."""

    def __init__(self, script: list[LLMResponse], grounded_text: str, sources: list[dict]):
        super().__init__(script)
        self.grounded_text = grounded_text
        self.grounded_sources = sources
        self.grounded_queries: list[str] = []

    def complete_grounded(self, prompt, system=None, temperature=0.3):
        self.grounded_queries.append(prompt)
        return self.grounded_text, self.grounded_sources


def _read_script() -> list[LLMResponse]:
    return [
        LLMResponse(tool_calls=[ToolCall(name="get_portfolio", arguments={}, id="c1")]),
        LLMResponse(text="NVDA anchors the theme."),
    ]


def _client(llm):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from smart_investing.api.app import create_app
    from smart_investing.broker.paper import PaperBroker
    from smart_investing.persistence.repo import StateRepo

    store = Store(":memory:")
    for t, title, text in ROWS:
        store.upsert_company(t, 1, title)
        store.upsert_document(t, 1, f"acc-{t}", "business", text)
    app = create_app(
        store=store,
        repo=StateRepo(":memory:"),
        broker=PaperBroker(cash=10_000.0),
        llm=llm,
        embedder=TfidfEmbedder(),
    )
    return TestClient(app)


def _stream_events(client, body: dict) -> list[dict]:
    events: list[dict] = []
    with client.stream("POST", "/api/chat/stream", json=body) as r:
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/event-stream")
        for line in r.iter_lines():
            if line.startswith("data: "):
                events.append(json.loads(line[len("data: "):]))
    return events


def _scrub(x):
    """Drop volatile identifiers so two independent runs compare equal."""
    volatile = {"id", "session_id", "created_at", "generated_at"}
    if isinstance(x, dict):
        return {k: _scrub(v) for k, v in x.items() if k not in volatile}
    if isinstance(x, list):
        return [_scrub(v) for v in x]
    return x


# ------------------------------------------------------------ /api/chat/stream
def test_chat_stream_emits_tool_events_then_final_matching_plain_chat():
    # streamed run
    cs = _client(ScriptedLLM(_read_script()))
    sid = cs.post("/api/chat", json={"message": "ai data center gpus", "live": False}).json()["session_id"]
    events = _stream_events(cs, {"message": "why NVDA?", "session_id": sid, "live": False})

    types = [e["type"] for e in events]
    assert types[-1] == "final"
    assert "tool_call" in types and "tool_result" in types
    assert types.index("tool_call") < types.index("final")
    call = next(e for e in events if e["type"] == "tool_call")
    assert call["tool"] == "get_portfolio" and call["args"] == {}

    final = events[-1]
    assert final["reply"] == "NVDA anchors the theme."
    assert final["researched"][0]["tool"] == "get_portfolio"

    # identical setup + script through plain /api/chat -> identical payload
    cp = _client(ScriptedLLM(_read_script()))
    sid2 = cp.post("/api/chat", json={"message": "ai data center gpus", "live": False}).json()["session_id"]
    plain = cp.post("/api/chat", json={"message": "why NVDA?", "session_id": sid2, "live": False}).json()
    final_payload = {k: v for k, v in final.items() if k != "type"}
    assert _scrub(final_payload) == _scrub(plain)


def test_chat_stream_first_turn_builds_and_finals():
    c = _client(ScriptedLLM([]))
    events = _stream_events(c, {"message": "ai data center gpus", "live": False})
    final = events[-1]
    assert final["type"] == "final"
    assert final["rebuilt"] is True
    assert final["proposal"] is not None
    assert final["state"]["theme"] == "ai data center gpus"


# ------------------------------------------------------------- web_research
def test_web_research_helper_unavailable_llm():
    from smart_investing.api.app import _web_research
    from smart_investing.llm.gemini import GeminiClient

    expected = {"note": "web research unavailable (no LLM configured)", "results": []}
    assert _web_research(None, "nvidia news") == expected
    assert _web_research(GeminiClient(api_key=""), "nvidia news") == expected


def test_web_research_helper_grounded_llm():
    from smart_investing.api.app import _web_research

    llm = GroundedLLM([], "NVDA posted record data-center revenue.", [{"title": "t", "uri": "u"}])
    out = _web_research(llm, "nvidia earnings news")
    assert out == {"summary": "NVDA posted record data-center revenue.",
                   "sources": [{"title": "t", "uri": "u"}]}
    assert llm.grounded_queries == ["nvidia earnings news"]


def test_web_research_tool_runs_through_chat():
    llm = GroundedLLM(
        [
            LLMResponse(tool_calls=[ToolCall(name="web_research",
                                             arguments={"query": "nvidia news this week"}, id="c1")]),
            LLMResponse(text="Fresh news says demand is strong."),
        ],
        "Record earnings this quarter.",
        [{"title": "src", "uri": "https://example.com"}],
    )
    c = _client(llm)
    sid = c.post("/api/chat", json={"message": "ai data center gpus", "live": False}).json()["session_id"]
    b = c.post("/api/chat", json={"message": "any nvidia news?", "session_id": sid, "live": False}).json()
    assert b["reply"] == "Fresh news says demand is strong."
    (entry,) = b["researched"]
    assert entry["tool"] == "web_research"
    assert entry["args"] == {"query": "nvidia news this week"}
    assert "Record earnings" in entry["preview"]


# --------------------------------------------- grounded industry_brief (bug fix)
def _graph() -> GraphService:
    s = Store(":memory:")
    for t, title, text in ROWS:
        s.upsert_company(t, 1, title)
        s.upsert_document(t, 1, f"a-{t}", "business", text)
    return GraphService(s, embedder=TfidfEmbedder()).build()


def test_industry_brief_grounded_path_parses_payload():
    """Regression: the grounded branch imported a nonexistent name and silently
    fell back to grounded=False on every call."""
    payload = json.dumps({
        "state": "Accelerator demand is outpacing supply.",
        "tailwinds": ["hyperscaler capex", "sovereign AI buildouts"],
        "risks": ["export controls", "power constraints"],
        "outlook": "Supply stays tight into next year.",
    })
    llm = GroundedLLM([], payload, [{"title": "brief", "uri": "https://example.com/b"}])
    b = industry_brief("ai data center gpus", _graph(), llm=llm)
    assert b["grounded"] is True
    assert b["analysis"] == "Accelerator demand is outpacing supply."
    assert b["tailwinds"] == ["hyperscaler capex", "sovereign AI buildouts"]
    assert b["risks"] == ["export controls", "power constraints"]
    assert b["outlook"] == "Supply stays tight into next year."
    assert b["sources"] == [{"title": "brief", "uri": "https://example.com/b"}]
    assert llm.grounded_queries and "ai data center gpus" in llm.grounded_queries[0]
