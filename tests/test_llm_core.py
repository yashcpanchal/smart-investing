"""Provider-agnostic LLM core: contract types, provider wire-format mapping,
factory selection, and the tool-use agent loop (scripted FakeLLM, no network)."""

from __future__ import annotations

import json

import pytest

from smart_investing.config import settings
from smart_investing.llm.agent import MUTATION_TOOLS, READ_TOOLS, run_agent
from smart_investing.llm.anthropic import AnthropicClient, _to_message
from smart_investing.llm.anthropic import _parse_response as parse_claude
from smart_investing.llm.base import (
    ChatMessage,
    LLMClient,
    LLMResponse,
    ToolCall,
    ToolResult,
    loads_lenient,
)
from smart_investing.llm.factory import get_llm
from smart_investing.llm.gemini import GeminiClient, _to_content, _to_decl
from smart_investing.llm.gemini import _parse_response as parse_gemini


# ----------------------------------------------------------------- fake LLM
class FakeLLM(LLMClient):
    """Plays back a script of LLMResponse objects; records what it was sent."""

    provider = "fake"

    def __init__(self, script: list[LLMResponse]):
        super().__init__(api_key="fake", model="fake-1")
        self.script = list(script)
        self.calls: list[dict] = []

    def chat(self, messages, *, system=None, tools=None, json_mode=False, temperature=0.2):
        self.calls.append({"messages": list(messages), "system": system, "tools": tools})
        return self.script.pop(0) if self.script else LLMResponse()


# ------------------------------------------------------------- base helpers
def test_loads_lenient_strips_fences_and_prose():
    assert loads_lenient('```json\n{"a": 1}\n```') == {"a": 1}
    assert loads_lenient('Sure! {"a": {"b": 2}} hope that helps') == {"a": {"b": 2}}
    with pytest.raises(json.JSONDecodeError):
        loads_lenient("no json here")


def test_complete_json_goes_through_chat():
    fake = FakeLLM([LLMResponse(text='{"x": 1}')])
    assert fake.complete_json("give me x") == {"x": 1}
    assert fake.calls[0]["messages"][0].content == "give me x"


# --------------------------------------------------------------- agent loop
def _ctx():
    return {"theme": "ai data center", "holdings": [{"symbol": "NVDA", "weight": 0.4}]}


def test_agent_executes_read_tool_then_replies():
    fake = FakeLLM([
        LLMResponse(tool_calls=[ToolCall(name="get_portfolio", arguments={}, id="c1")]),
        LLMResponse(text="NVDA is 40% because it dominates AI accelerators."),
    ])
    seen = {}
    out = run_agent(
        "why is NVDA so big?", _ctx(), fake,
        read_tools={"get_portfolio": lambda **kw: seen.setdefault("hit", True) and {"holdings": []} or {"holdings": []}},
    )
    assert seen.get("hit") is True
    assert out["reply"].startswith("NVDA is 40%")
    assert out["researched"] == [{"tool": "get_portfolio", "args": {}, "preview": "{'holdings': []}"}]
    assert out["actions"] == [{"op": "none"}]
    # round 2 must carry the assistant tool-call turn + the tool result turn
    roles = [m.role for m in fake.calls[1]["messages"]]
    assert roles == ["user", "assistant", "tool"]


def test_agent_queues_mutations_without_touching_state():
    fake = FakeLLM([
        LLMResponse(
            text="",
            tool_calls=[
                ToolCall(name="remove_symbols", arguments={"symbols": ["ko"]}, id="c1"),
                ToolCall(name="set_risk", arguments={"value": "low"}, id="c2"),
            ],
        ),
        LLMResponse(text="Dropped KO and dialed the risk down."),
    ])
    out = run_agent("drop KO and make it safer", _ctx(), fake)
    assert {"op": "remove", "symbols": ["KO"]} in out["actions"]
    assert {"op": "set_risk", "value": "low"} in out["actions"]
    assert out["reply"]


def test_agent_passes_history_and_context():
    fake = FakeLLM([LLMResponse(text="It held up well last year.")])
    history = [{"role": "user", "text": "build me an ai portfolio"},
               {"role": "assistant", "text": "Done — 8 names."}]
    run_agent("how did it do?", _ctx(), fake, history=history)
    msgs = fake.calls[0]["messages"]
    assert [m.content for m in msgs] == ["build me an ai portfolio", "Done — 8 names.", "how did it do?"]
    assert "ai data center" in fake.calls[0]["system"]  # context in system prompt


def test_agent_read_tool_error_feeds_back_not_raises():
    def boom(**kw):
        raise RuntimeError("yfinance down")
    fake = FakeLLM([
        LLMResponse(tool_calls=[ToolCall(name="get_stock_facts", arguments={"symbol": "MU"}, id="c1")]),
        LLMResponse(text="Couldn't fetch live facts, but MU is the memory play here."),
    ])
    out = run_agent("what does MU do?", _ctx(), fake, read_tools={"get_stock_facts": boom})
    assert out["reply"]


def test_agent_falls_back_when_llm_unavailable_or_errors():
    out = run_agent("make it safer", _ctx(), None)
    assert {"op": "set_risk", "value": "low"} in out["actions"]

    class Exploder(FakeLLM):
        def chat(self, *a, **k):
            raise RuntimeError("429")
    out2 = run_agent("make it safer", _ctx(), Exploder([]))
    assert {"op": "set_risk", "value": "low"} in out2["actions"]


def test_agent_last_round_withholds_tools():
    fake = FakeLLM([
        LLMResponse(tool_calls=[ToolCall(name="get_portfolio", arguments={}, id=f"c{i}")])
        for i in range(3)
    ] + [LLMResponse(text="done looking.")])
    out = run_agent("inspect everything", _ctx(), fake,
                    read_tools={"get_portfolio": lambda **kw: {"holdings": []}}, max_rounds=4)
    assert out["reply"] == "done looking."
    assert fake.calls[-1]["tools"] is None  # final round forces a reply


def test_agent_rich_trace_and_preview_cap():
    fake = FakeLLM([
        LLMResponse(tool_calls=[ToolCall(name="get_stock_facts", arguments={"symbol": "MU"}, id="c1")]),
        LLMResponse(text="MU is the memory play."),
    ])
    big = {"summary": "x" * 1000}
    out = run_agent("what does MU do?", _ctx(), fake, read_tools={"get_stock_facts": lambda **kw: big})
    (entry,) = out["researched"]
    assert entry["tool"] == "get_stock_facts"
    assert entry["args"] == {"symbol": "MU"}
    assert len(entry["preview"]) <= 200


def test_agent_on_event_ordering_and_shapes():
    fake = FakeLLM([
        LLMResponse(tool_calls=[
            ToolCall(name="get_portfolio", arguments={}, id="c1"),
            ToolCall(name="set_risk", arguments={"value": "low"}, id="c2"),
        ]),
        LLMResponse(text="Checked and dialed it down."),
    ])
    events: list[dict] = []
    out = run_agent(
        "make it safer after checking", _ctx(), fake,
        read_tools={"get_portfolio": lambda **kw: {"holdings": []}},
        on_event=events.append,
    )
    assert out["reply"]
    types = [e["type"] for e in events]
    assert types == ["round", "tool_call", "tool_result", "queued", "round"]
    assert events[0] == {"type": "round", "round": 0}
    assert events[1] == {"type": "tool_call", "tool": "get_portfolio", "args": {}}
    assert events[2]["tool"] == "get_portfolio" and "holdings" in events[2]["preview"]
    assert events[3] == {"type": "queued", "tool": "set_risk"}


def test_agent_on_event_exception_never_breaks_the_turn():
    def bad_listener(e: dict) -> None:
        raise RuntimeError("listener broke")
    fake = FakeLLM([
        LLMResponse(tool_calls=[ToolCall(name="get_portfolio", arguments={}, id="c1")]),
        LLMResponse(text="All good."),
    ])
    out = run_agent("check it", _ctx(), fake,
                    read_tools={"get_portfolio": lambda **kw: {"holdings": []}},
                    on_event=bad_listener)
    assert out["reply"] == "All good."
    assert out["researched"][0]["tool"] == "get_portfolio"


# --------------------------------------------------- gemini wire format
def test_gemini_tool_declaration_and_message_mapping():
    assert "parameters" not in _to_decl(READ_TOOLS["get_portfolio"])  # empty props omitted
    assert _to_decl(MUTATION_TOOLS["set_risk"])["parameters"]["required"] == ["value"]

    call = ToolCall(name="get_portfolio", arguments={}, id="call_0")
    asst = _to_content(ChatMessage(role="assistant", content="checking", tool_calls=[call]))
    assert asst["role"] == "model" and {"functionCall": {"name": "get_portfolio", "args": {}}} in asst["parts"]

    tool = _to_content(ChatMessage(role="tool", tool_results=[ToolResult(call=call, content={"ok": 1})]))
    assert tool["parts"][0]["functionResponse"]["response"] == {"ok": 1}
    # scalars get wrapped — Gemini requires an object
    tool2 = _to_content(ChatMessage(role="tool", tool_results=[ToolResult(call=call, content="fine")]))
    assert tool2["parts"][0]["functionResponse"]["response"] == {"result": "fine"}


def test_gemini_parses_function_calls_with_synthetic_ids():
    resp = parse_gemini({"candidates": [{"content": {"parts": [
        {"text": "let me check. "},
        {"functionCall": {"name": "get_stock_facts", "args": {"symbol": "MU"}}},
    ]}, "finishReason": "STOP"}]})
    assert resp.text == "let me check."
    assert resp.tool_calls[0].name == "get_stock_facts" and resp.tool_calls[0].id


# ------------------------------------------------- anthropic wire format
def test_anthropic_message_mapping_and_parse():
    call = ToolCall(name="get_neighbors", arguments={"symbol": "NVDA"}, id="toolu_1")
    asst = _to_message(ChatMessage(role="assistant", content="", tool_calls=[call]))
    assert asst["content"][0] == {"type": "tool_use", "id": "toolu_1", "name": "get_neighbors",
                                  "input": {"symbol": "NVDA"}}
    tool = _to_message(ChatMessage(role="tool", tool_results=[ToolResult(call=call, content={"n": []})]))
    assert tool["role"] == "user" and tool["content"][0]["tool_use_id"] == "toolu_1"

    resp = parse_claude({"stop_reason": "tool_use", "content": [
        {"type": "text", "text": "researching…"},
        {"type": "tool_use", "id": "toolu_2", "name": "get_portfolio", "input": {}},
    ]})
    assert resp.tool_calls[0].id == "toolu_2"


def test_anthropic_refusal_yields_empty_response():
    resp = parse_claude({"stop_reason": "refusal", "content": []})
    assert resp.text == "" and not resp.tool_calls and resp.stop_reason == "refusal"


# ------------------------------------------------------------------ factory
def test_factory_selection(monkeypatch):
    monkeypatch.setattr(settings, "llm_provider", "none")
    assert get_llm() is None

    monkeypatch.setattr(settings, "llm_provider", "auto")
    monkeypatch.setattr(settings, "anthropic_api_key", "sk-test")
    monkeypatch.setattr(settings, "llm_model", None)
    llm = get_llm()
    assert isinstance(llm, AnthropicClient) and llm.model == "claude-opus-4-8"

    monkeypatch.setattr(settings, "anthropic_api_key", None)
    monkeypatch.setattr(settings, "gemini_api_key", "g-test")
    monkeypatch.setattr(settings, "llm_model", "gemini-2.5-pro")
    llm2 = get_llm()
    assert isinstance(llm2, GeminiClient) and llm2.model == "gemini-2.5-pro"

    monkeypatch.setattr(settings, "gemini_api_key", None)
    assert get_llm() is None  # no keys -> deterministic mode

    monkeypatch.setattr(settings, "llm_provider", "anthropic")
    monkeypatch.setattr(settings, "anthropic_api_key", "sk-test")
    monkeypatch.setattr(settings, "llm_model", None)
    assert isinstance(get_llm(), AnthropicClient)
