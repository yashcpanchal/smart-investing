"""FastAPI surface over the engine (Phase 9) + rebalance trigger (Phase 8).

Single-user/paper for now. Holds three things: the corpus Store (search), the
StateRepo (persistence), and a PaperBroker (rehydrated from the last saved
portfolio). The frontend (Phase 10) talks to this; swapping PaperBroker for the
Robinhood MCP broker (Phase 11) needs no API change.
"""

from __future__ import annotations

import asyncio
import json
import queue
from collections.abc import AsyncIterator, Callable

from fastapi import FastAPI, HTTPException
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from smart_investing.broker.paper import PaperBroker
from smart_investing.data.smart_money import compute_smart_money_scores
from smart_investing.data.store import Store
from smart_investing.domain.types import Proposal
from smart_investing.execution.executor import execute_proposal
from smart_investing.llm.agent import run_agent
from smart_investing.llm.clarify import clarify as clarify_prompt
from smart_investing.llm.compiler import compile_spec
from smart_investing.llm.factory import get_llm
from smart_investing.persistence.repo import StateRepo
from smart_investing.research import company_profile, industry_brief
from smart_investing.retrieval.graph_service import GraphService
from smart_investing.session import Session, SessionManager
from smart_investing.strategy import compile_strategy


class ClarifyRequest(BaseModel):
    prompt: str


class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None
    live: bool = True


class KnobsRequest(BaseModel):
    """Direct-manipulation controls (the UI sliders) — no language in the loop.
    Extensible: today only source_weights, e.g. {"sec_13f": 0.7, "insider": 0.4}."""

    source_weights: dict[str, float] | None = None
    live: bool = True


class CompileRequest(BaseModel):
    prompt: str
    initial_cash: float = 10_000.0
    live: bool = True
    # Conversational follow-up answers (risk / breadth / supply_chain / ...).
    answers: dict | None = None
    # Price-history window used for risk + backtest ("1y" | "2y" | "3y" | "5y").
    lookback: str = "2y"
    # Optional retrieval breadth override; normally WE decide holding count.
    top_k: int | None = None


_KNOB_OPS = {"set_risk": "risk", "set_breadth": "breadth", "set_supply_chain": "supply_chain"}
_REBUILD_OPS = {"set_theme", "add", "remove", "expand", "set_lookback", "set_cash", "set_source_weights", *_KNOB_OPS}
_EXPAND_LIMIT = 4


def _apply_actions(session: Session, actions: list[dict], graph: GraphService) -> dict:
    """Mutate the session from interpreted actions. Returns a summary of changes."""
    added: list[str] = []
    removed: list[str] = []
    changed: list[str] = []
    for a in actions:
        op = a.get("op")
        if op == "set_theme" and a.get("value"):
            session.theme = str(a["value"]).strip()
            changed.append("theme")
        elif op == "add" and a.get("symbols"):
            added += session.pin([str(s) for s in a["symbols"]])
        elif op == "remove" and a.get("symbols"):
            removed += session.exclude([str(s) for s in a["symbols"]])
        elif op == "expand" and a.get("symbol"):
            direction = a.get("direction") or "all"
            nbrs = graph.neighbors(str(a["symbol"]).upper(), theme=session.theme, limit=_EXPAND_LIMIT * 3)
            picks = [n["symbol"] for n in nbrs if direction == "all" or n["direction"] == direction][:_EXPAND_LIMIT]
            added += session.pin(picks)
        elif op == "set_source_weights" and isinstance(a.get("value"), dict):
            for k, v in a["value"].items():
                if k not in session.source_weights:
                    continue
                try:
                    session.source_weights[k] = min(1.0, max(0.0, float(v)))
                except (TypeError, ValueError):
                    continue
            changed.append("source_weights")
        elif op in _KNOB_OPS and a.get("value"):
            session.answers[_KNOB_OPS[op]] = str(a["value"])
            changed.append(_KNOB_OPS[op])
        elif op == "set_lookback" and a.get("value"):
            session.lookback = str(a["value"])
            changed.append("lookback")
        elif op == "set_cash" and a.get("value"):
            try:
                session.cash = float(a["value"])
                changed.append("cash")
            except (TypeError, ValueError):
                pass
    return {"added": added, "removed": removed, "changed": changed}


def _template_reply(changes: dict, proposal: Proposal | None) -> str:
    """Fallback reply when the LLM didn't supply one — grounded in what changed."""
    bits: list[str] = []
    if changes["added"]:
        bits.append("added " + ", ".join(changes["added"]))
    if changes["removed"]:
        bits.append("dropped " + ", ".join(changes["removed"]))
    if "risk" in changes["changed"]:
        bits.append("retuned the risk")
    if "breadth" in changes["changed"]:
        bits.append("adjusted the spread")
    if "lookback" in changes["changed"]:
        bits.append("changed the look-back window")
    if "theme" in changes["changed"]:
        bits.append("updated the thesis")
    if "source_weights" in changes["changed"]:
        bits.append("reweighted the evidence signals")
    lead = ("I " + ", ".join(bits) + ". ") if bits else ""
    if proposal is not None:
        held = sum(1 for w in proposal.target_weights.values() if w > 0.005)
        o = proposal.optimization
        return (
            f"{lead}Rebuilt: {held} holdings, ~{o.expected_return * 100:.1f}% return at "
            f"~{o.volatility * 100:.1f}% risk (Sharpe {o.sharpe:.2f})."
        )
    return lead or "Got it — tell me how you'd like to shape the portfolio."


def _web_research(llm, query: str) -> dict:
    """Grounded live-web lookup for the agent's web_research read tool."""
    if llm is None or not getattr(llm, "available", False):
        return {"note": "web research unavailable (no LLM configured)", "results": []}
    text, sources = llm.complete_grounded(str(query))
    return {"summary": text, "sources": sources}


def create_app(store=None, repo=None, broker=None, llm=None, embedder=None, *, default_cash: float = 10_000.0) -> FastAPI:
    app = FastAPI(title="smart-investing API", version="0.1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    state = {
        "store": store, "repo": repo, "broker": broker, "llm": llm, "embedder": embedder,
        "graph": None, "smart_money": None,
    }
    sessions = SessionManager()

    def get_store() -> Store:
        if state["store"] is None:
            state["store"] = Store()
        return state["store"]

    def get_graph() -> GraphService:
        if state["graph"] is None:
            state["graph"] = GraphService(get_store(), embedder=state["embedder"]).build()
        return state["graph"]

    def get_smart_money() -> dict:
        """Per-process cache of the deterministic 13F/insider scores (same
        discipline as get_graph(): computed lazily once from the store)."""
        if state["smart_money"] is None:
            state["smart_money"] = compute_smart_money_scores(get_store())
        return state["smart_money"]

    def get_repo() -> StateRepo:
        if state["repo"] is None:
            state["repo"] = StateRepo()
        return state["repo"]

    def get_broker() -> PaperBroker:
        if state["broker"] is None:
            b = PaperBroker(cash=default_cash)
            snap = get_repo().get_portfolio()
            if snap:
                b.load_state(snap[0], snap[1])
            state["broker"] = b
        return state["broker"]

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "corpus_docs": get_store().count("documents")}

    @app.post("/api/clarify")
    def clarify_(req: ClarifyRequest) -> dict:
        """Conversational step: restate the thesis and return tailored follow-up
        questions BEFORE building anything."""
        return clarify_prompt(req.prompt, llm=state["llm"])

    @app.get("/api/graph/search")
    def graph_search(theme: str, top_k: int = 8) -> dict:
        """Theme -> seed nodes for the supply-chain canvas (direct matches)."""
        g = get_graph()
        return {"theme": theme, "nodes": g.search(theme, top_k=top_k)}

    @app.get("/api/graph/neighbors")
    def graph_neighbors(node: str, theme: str = "", limit: int = 12) -> dict:
        """Connections of a node: upstream suppliers, downstream customers, peers.
        This is the 'walk back to the manufacturer, and further' expansion."""
        g = get_graph()
        node = node.upper()
        return {"node": g.node_view(node, theme), "neighbors": g.neighbors(node, theme=theme, limit=limit)}

    @app.get("/api/stock/{symbol}")
    def stock(symbol: str, theme: str = "") -> dict:
        """Per-stock detail: company facts (yfinance) + LLM theme-fit & must-knows."""
        g = get_graph()
        name = g.titles.get(symbol.upper(), "")
        return company_profile(symbol, name=name, theme=theme, llm=state["llm"])

    @app.get("/api/industry")
    def industry(theme: str) -> dict:
        """Industry briefing: deterministic supply-chain layers + current-state
        analysis grounded in live search (when the LLM is available)."""
        return industry_brief(theme, get_graph(), llm=state["llm"])

    def _holdings_context(session: Session) -> list[dict]:
        p = session.proposal
        if p is None:
            return []
        return [
            {"symbol": a.symbol, "name": a.name, "weight": p.target_weights.get(a.symbol, 0.0)}
            for a in p.universe.assets
            if p.target_weights.get(a.symbol, 0.0) > 0.005
        ]

    def _read_tools(session: Session) -> dict:
        """Live research tools the agent can call mid-turn. Read-only by design —
        mutations flow through the action vocabulary + deterministic rebuild."""
        g = get_graph()

        def get_portfolio() -> dict:
            p = session.proposal
            if p is None:
                return {"holdings": [], "note": "no portfolio built yet"}
            o = p.optimization
            why = {h.symbol: h.why for h in (p.explanation.holdings if p.explanation else [])}
            return {
                "holdings": [
                    {"symbol": a.symbol, "name": a.name,
                     "weight": round(p.target_weights.get(a.symbol, 0.0), 4),
                     "why": why.get(a.symbol, "")}
                    for a in p.universe.assets
                    if p.target_weights.get(a.symbol, 0.0) > 0.005
                ],
                "expected_return": round(o.expected_return, 4),
                "volatility": round(o.volatility, 4),
                "sharpe": round(o.sharpe, 2),
                "blocked": p.blocked,
            }

        def get_stock_facts(symbol: str = "") -> dict:
            sym = str(symbol).upper()
            # Facts only — no nested LLM call inside the agent loop.
            return company_profile(sym, name=g.titles.get(sym, ""), theme=session.theme, llm=None)

        def get_neighbors(symbol: str = "", direction: str = "all") -> dict:
            sym = str(symbol).upper()
            nbrs = g.neighbors(sym, theme=session.theme, limit=10)
            if direction != "all":
                nbrs = [n for n in nbrs if n.get("direction") == direction]
            return {"node": sym, "neighbors": nbrs}

        def search_companies(query: str = "") -> dict:
            return {"matches": g.search(str(query), top_k=8)}

        def web_research(query: str = "") -> dict:
            return _web_research(state["llm"], query)

        return {"get_portfolio": get_portfolio, "get_stock_facts": get_stock_facts,
                "get_neighbors": get_neighbors, "search_companies": search_companies,
                "web_research": web_research}

    def _chat_turn(req: ChatRequest, on_event: Callable[[dict], None] | None = None) -> dict:
        """One full conversation turn: interpret freeform feedback -> apply ->
        rebuild -> reply. Shared by /api/chat and /api/chat/stream so the two
        endpoints cannot drift. `on_event` receives agent progress dicts."""
        session = sessions.get_or_create(req.session_id)
        session.messages.append({"role": "user", "text": req.message})

        first_turn = not session.theme
        # First turn: the message IS the thesis — no need to spend an LLM call
        # interpreting refine-intent. Later turns: the agent loop researches with
        # read tools and queues edits, with the conversation history in context.
        if first_turn:
            session.theme = req.message.strip()
            result: dict = {"actions": [], "reply": "", "researched": []}
            actions: list[dict] = []
        else:
            context = {**session.snapshot(), "holdings": _holdings_context(session)}
            result = run_agent(
                req.message, context, state["llm"],
                history=session.messages[:-1],  # current message passed separately
                read_tools=_read_tools(session),
                on_event=on_event,
            )
            actions = result["actions"]
            if any(a.get("op") == "set_theme" and a.get("value") for a in actions):
                session.base_spec = None  # theme changed -> re-parse on rebuild

        changes = _apply_actions(session, actions, get_graph())
        needs_rebuild = first_turn or any(a.get("op") in _REBUILD_OPS for a in actions)

        if needs_rebuild and session.theme:
            # Parse the thesis with the LLM ONCE per conversation; reuse the cached
            # base spec across refine turns (answers/pins are layered on each rebuild).
            if session.base_spec is None:
                session.base_spec = compile_spec(session.theme, llm=state["llm"])
            idx, meta = get_graph().retrieval()
            proposal = compile_strategy(
                session.theme,
                get_store(),
                live=req.live,
                initial_cash=session.cash,
                account=get_broker().get_account_state(),
                llm=state["llm"],
                embedder=state["embedder"],
                answers=session.compile_answers(),
                lookback=session.lookback,
                spec=session.base_spec,
                index=idx,
                meta=meta,
                smart_money=get_smart_money(),
            )
            get_repo().save_proposal(proposal)
            session.proposal = proposal
            # use the clean extracted themes for graph search (sharper than the raw message)
            if proposal.spec.themes:
                session.search_theme = " ".join(proposal.spec.themes)

        reply = result.get("reply") or _template_reply(changes, session.proposal)
        session.messages.append({"role": "assistant", "text": reply})
        return {
            "session_id": session.id,
            "reply": reply,
            "actions": actions,
            # read-tool trace: [{"tool", "args", "preview"}] per executed call
            "researched": result.get("researched", []),
            "added": changes["added"],
            "removed": changes["removed"],
            "rebuilt": needs_rebuild and bool(session.theme),
            "proposal": session.proposal,
            "state": session.snapshot(),
        }

    @app.post("/api/chat")
    def chat(req: ChatRequest) -> dict:
        """The conversation loop: interpret freeform feedback -> apply -> rebuild -> reply.

        First message is the thesis. Subsequent messages refine it ("go deeper on
        NVDA", "make it safer", "drop the consumer names", "why MU?")."""
        return _chat_turn(req)

    @app.post("/api/chat/stream")
    async def chat_stream(req: ChatRequest) -> StreamingResponse:
        """Same turn as /api/chat, streamed as server-sent events: agent progress
        (round / tool_call / tool_result / queued) while the turn runs on a worker
        thread, then a single {"type": "final", ...} frame carrying the exact
        /api/chat payload."""
        q: queue.Queue = queue.Queue()
        _DONE = object()

        def emit(event: dict) -> None:
            q.put(event)

        def worker() -> None:
            try:
                payload = _chat_turn(req, on_event=emit)
                q.put({"type": "final", **jsonable_encoder(payload)})
            except Exception as e:  # surfaced to the stream, not a 500 mid-stream
                q.put({"type": "error", "detail": str(e)[:500]})
            finally:
                q.put(_DONE)

        async def gen() -> AsyncIterator[str]:
            task = asyncio.ensure_future(run_in_threadpool(worker))
            try:
                while True:
                    item = await run_in_threadpool(q.get)
                    if item is _DONE:
                        break
                    yield f"data: {json.dumps(item, default=str)}\n\n"
            finally:
                await task

        return StreamingResponse(gen(), media_type="text/event-stream")

    @app.post("/api/session/{session_id}/knobs")
    def knobs(session_id: str, req: KnobsRequest) -> dict:
        """Direct knob manipulation (the sliders): apply the same deterministic
        action path chat uses, then ONE rebuild with the cached retrieval index.
        Same response shape as /api/chat minus reply/researched."""
        session = sessions.get(session_id)
        if session is None:
            raise HTTPException(404, "session not found")
        actions: list[dict] = []
        if req.source_weights is not None:
            actions.append({"op": "set_source_weights", "value": req.source_weights})
        changes = _apply_actions(session, actions, get_graph())

        rebuilt = bool(actions) and bool(session.theme)
        if rebuilt:
            if session.base_spec is None:
                session.base_spec = compile_spec(session.theme, llm=state["llm"])
            idx, meta = get_graph().retrieval()
            proposal = compile_strategy(
                session.theme,
                get_store(),
                live=req.live,
                initial_cash=session.cash,
                account=get_broker().get_account_state(),
                llm=state["llm"],
                embedder=state["embedder"],
                answers=session.compile_answers(),
                lookback=session.lookback,
                spec=session.base_spec,
                index=idx,
                meta=meta,
                smart_money=get_smart_money(),
            )
            get_repo().save_proposal(proposal)
            session.proposal = proposal

        return {
            "session_id": session.id,
            "actions": actions,
            "added": changes["added"],
            "removed": changes["removed"],
            "rebuilt": rebuilt,
            "proposal": session.proposal,
            "state": session.snapshot(),
        }

    @app.post("/api/compile")
    def compile_(req: CompileRequest) -> Proposal:
        idx, meta = get_graph().retrieval()
        proposal = compile_strategy(
            req.prompt,
            get_store(),
            top_k=req.top_k,
            live=req.live,
            initial_cash=req.initial_cash,
            account=get_broker().get_account_state(),
            llm=state["llm"],
            embedder=state["embedder"],
            answers=req.answers,
            lookback=req.lookback,
            index=idx,
            meta=meta,
        )
        get_repo().save_proposal(proposal)
        return proposal

    @app.get("/api/proposals")
    def list_proposals() -> list[dict]:
        return get_repo().list_proposals()

    @app.get("/api/proposals/{pid}")
    def get_proposal(pid: str) -> Proposal:
        p = get_repo().get_proposal(pid)
        if not p:
            raise HTTPException(404, "proposal not found")
        return p

    @app.post("/api/proposals/{pid}/approve")
    def approve(pid: str) -> dict:
        p = get_repo().get_proposal(pid)
        if not p:
            raise HTTPException(404, "proposal not found")
        if not p.trades:
            raise HTTPException(400, "proposal has no executable trades (was it blocked?)")
        b = get_broker()
        results = execute_proposal(p, b)
        acct = b.get_account_state()
        get_repo().save_portfolio(acct, b.realized_pnl)
        get_repo().save_execution(pid, acct)
        filled = sum(1 for r in results if r.status.value == "filled")
        return {
            "proposal_id": pid,
            "filled": filled,
            "rejected": len(results) - filled,
            "realized_pnl": b.realized_pnl,
            "account": acct.model_dump(),
        }

    @app.get("/api/portfolio")
    def portfolio() -> dict:
        b = get_broker()
        acct = b.get_account_state()
        prices = b.get_prices(list(acct.positions.keys()))
        return {
            "account": acct.model_dump(),
            "prices": prices,
            "equity": acct.equity(prices),
            "realized_pnl": b.realized_pnl,
        }

    @app.post("/api/rebalance/{pid}")
    def rebalance(pid: str) -> Proposal:
        """Re-run the saved strategy against the CURRENT portfolio -> new proposal
        with delta orders (the bi-weekly rebalance, triggered)."""
        p = get_repo().get_proposal(pid)
        if not p:
            raise HTTPException(404, "proposal not found")
        idx, meta = get_graph().retrieval()
        new = compile_strategy(
            p.spec.raw_prompt,
            get_store(),
            account=get_broker().get_account_state(),
            llm=state["llm"],
            embedder=state["embedder"],
            lookback=p.lookback,
            index=idx,
            meta=meta,
        )
        get_repo().save_proposal(new)
        get_repo().audit("rebalance", f"{pid}->{new.id}")
        return new

    @app.get("/api/audit")
    def audit() -> list[dict]:
        return get_repo().audit_log()

    return app


# The `si serve` entrypoint. Auto-wire the configured Gemini client (None-safe:
# get_llm() returns None when no key is set, keeping the deterministic fallback).
app = create_app(llm=get_llm())
