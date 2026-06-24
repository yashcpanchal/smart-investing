"""FastAPI surface over the engine (Phase 9) + rebalance trigger (Phase 8).

Single-user/paper for now. Holds three things: the corpus Store (search), the
StateRepo (persistence), and a PaperBroker (rehydrated from the last saved
portfolio). The frontend (Phase 10) talks to this; swapping PaperBroker for the
Robinhood MCP broker (Phase 11) needs no API change.
"""

from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from smart_investing.broker.paper import PaperBroker
from smart_investing.data.store import Store
from smart_investing.domain.types import Proposal
from smart_investing.execution.executor import execute_proposal
from smart_investing.persistence.repo import StateRepo
from smart_investing.strategy import compile_strategy


class CompileRequest(BaseModel):
    prompt: str
    initial_cash: float = 10_000.0
    top_k: int = 12
    live: bool = True


def create_app(store=None, repo=None, broker=None, llm=None, embedder=None, *, default_cash: float = 10_000.0) -> FastAPI:
    app = FastAPI(title="smart-investing API", version="0.1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    state = {"store": store, "repo": repo, "broker": broker, "llm": llm, "embedder": embedder}

    def get_store() -> Store:
        if state["store"] is None:
            state["store"] = Store()
        return state["store"]

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

    @app.post("/api/compile")
    def compile_(req: CompileRequest) -> Proposal:
        proposal = compile_strategy(
            req.prompt,
            get_store(),
            top_k=req.top_k,
            live=req.live,
            initial_cash=req.initial_cash,
            account=get_broker().get_account_state(),
            llm=state["llm"],
            embedder=state["embedder"],
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
        new = compile_strategy(
            p.spec.raw_prompt,
            get_store(),
            account=get_broker().get_account_state(),
            llm=state["llm"],
            embedder=state["embedder"],
        )
        get_repo().save_proposal(new)
        get_repo().audit("rebalance", f"{pid}->{new.id}")
        return new

    @app.get("/api/audit")
    def audit() -> list[dict]:
        return get_repo().audit_log()

    return app


app = create_app()
