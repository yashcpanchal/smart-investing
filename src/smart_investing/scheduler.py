"""Autonomous rebalancing (Phase 8).

`rebalance_once` re-runs a saved strategy against the CURRENT portfolio and, in
autonomous mode (and only if the circuit breaker passed), executes the delta
trades. `RebalanceScheduler` fires it on a fixed interval. In production a cron
hitting POST /api/rebalance is equivalent; this is the in-process option.
"""

from __future__ import annotations

import threading
from collections.abc import Callable

from smart_investing.broker.base import BrokerAdapter
from smart_investing.domain.types import OrderResult, Proposal
from smart_investing.execution.executor import execute_proposal
from smart_investing.strategy import compile_strategy


def rebalance_once(
    prompt: str,
    store,
    broker: BrokerAdapter,
    repo=None,
    *,
    autonomous: bool = False,
    **compile_kwargs,
) -> tuple[Proposal, list[OrderResult]]:
    proposal = compile_strategy(prompt, store, account=broker.get_account_state(), **compile_kwargs)
    if repo is not None:
        repo.save_proposal(proposal)
    executed: list[OrderResult] = []
    if autonomous and proposal.trades:
        executed = execute_proposal(proposal, broker)
        if repo is not None:
            repo.save_portfolio(broker.get_account_state(), getattr(broker, "realized_pnl", 0.0))
            repo.save_execution(proposal.id, broker.get_account_state())
            repo.audit("autonomous_rebalance", proposal.id)
    return proposal, executed


class RebalanceScheduler:
    """Runs a job every `interval_seconds` on a daemon thread until stopped.
    (Map cadence_days -> seconds in production; small intervals for testing.)"""

    def __init__(self, interval_seconds: float, job: Callable[[], None]) -> None:
        self.interval = interval_seconds
        self.job = job
        self._timer: threading.Timer | None = None
        self._stop = threading.Event()

    def start(self) -> RebalanceScheduler:
        self._schedule()
        return self

    def _schedule(self) -> None:
        if self._stop.is_set():
            return
        self._timer = threading.Timer(self.interval, self._run)
        self._timer.daemon = True
        self._timer.start()

    def _run(self) -> None:
        try:
            self.job()
        except Exception:  # noqa: BLE001 — a failed cycle must not kill the scheduler
            pass
        self._schedule()

    def stop(self) -> None:
        self._stop.set()
        if self._timer is not None:
            self._timer.cancel()
