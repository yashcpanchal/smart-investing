from __future__ import annotations

from smart_investing.domain.types import (
    AccountState,
    AssetUniverse,
    OptimizationResult,
    Position,
    Proposal,
    StrategySpec,
)
from smart_investing.persistence.repo import StateRepo


def _proposal() -> Proposal:
    return Proposal(
        spec=StrategySpec(raw_prompt="nuclear energy"),
        universe=AssetUniverse(theme="nuclear"),
        optimization=OptimizationResult(weights={"BWXT": 1.0}, expected_return=0.1, volatility=0.2, sharpe=0.5),
        target_weights={"BWXT": 1.0},
    )


def test_save_and_get_proposal_roundtrip():
    repo = StateRepo(":memory:")
    p = _proposal()
    repo.save_proposal(p)
    got = repo.get_proposal(p.id)
    assert got is not None
    assert got.spec.raw_prompt == "nuclear energy"
    assert got.target_weights == {"BWXT": 1.0}


def test_portfolio_snapshot_roundtrip():
    repo = StateRepo(":memory:")
    acct = AccountState(cash=5000.0, positions={"BWXT": Position(symbol="BWXT", quantity=10, avg_cost=100.0)})
    repo.save_portfolio(acct, realized_pnl=123.45)
    loaded = repo.get_portfolio()
    assert loaded is not None
    state, pnl = loaded
    assert state.cash == 5000.0
    assert state.positions["BWXT"].quantity == 10
    assert pnl == 123.45


def test_audit_log_records_events():
    repo = StateRepo(":memory:")
    repo.save_proposal(_proposal())
    log = repo.audit_log()
    assert any(e["event"] == "proposal_saved" for e in log)
