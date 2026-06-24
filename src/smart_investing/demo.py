"""Demo 1 — the full Phase 1-3 loop on paper money.

theme (hardcoded) -> optimize (max-Sharpe) -> plan orders -> circuit breaker ->
paper-execute -> show positions, P&L, and the efficient frontier summary.
"""

from __future__ import annotations

from rich.console import Console
from rich.table import Table

from smart_investing.broker.paper import PaperBroker
from smart_investing.data import load_prices, synthetic_prices
from smart_investing.domain.types import Objective, RiskParams
from smart_investing.execution.planner import plan_orders
from smart_investing.quant import backtest_constant_weights, optimize
from smart_investing.quant.montecarlo import simulate_terminal
from smart_investing.quant.returns import annualized_cov, annualized_mean, daily_returns
from smart_investing.risk import validate

console = Console()

DEFAULT_THEME = "AI & semiconductors"
DEFAULT_TICKERS = ["NVDA", "AMD", "AVGO", "TSM", "MU", "ASML", "ARM", "SMCI"]


def _get_prices(tickers: list[str], live: bool = True):
    if live:
        try:
            px = load_prices(tickers, period="2y")
            if px.shape[1] >= 2 and px.shape[0] > 60:
                return px, "yfinance (live)"
            console.print("[yellow]live data too sparse; falling back to synthetic[/]")
        except Exception as e:  # noqa: BLE001
            console.print(f"[yellow]yfinance unavailable ({e}); using synthetic data[/]")
    return synthetic_prices(tickers, n_days=504, seed=7), "synthetic (offline)"


def run_demo1(
    tickers: list[str] | None = None,
    theme: str = DEFAULT_THEME,
    initial_cash: float = 10_000.0,
    concentration_cap: float = 0.30,
    live: bool = True,
) -> dict:
    tickers = tickers or DEFAULT_TICKERS
    console.rule(f"[bold]Demo 1 — {theme}")
    px, source = _get_prices(tickers, live=live)
    syms = list(px.columns)
    console.print(f"Universe ({source}): {', '.join(syms)}\n")

    risk = RiskParams(concentration_cap=concentration_cap)
    opt = optimize(px, objective=Objective.MAX_SHARPE, risk=risk)
    bt = backtest_constant_weights(px, opt.weights)
    last_prices = {s: float(px[s].iloc[-1]) for s in syms}

    rets = daily_returns(px)
    mc = simulate_terminal(
        annualized_mean(rets).to_numpy(),
        annualized_cov(rets).to_numpy(),
        [opt.weights[s] for s in syms],
        n_sims=10_000,
    )

    t = Table(title="Optimized target (max-Sharpe, no-short, 30% cap)")
    t.add_column("Ticker")
    t.add_column("Weight", justify="right")
    t.add_column("Last price", justify="right")
    for s in syms:
        t.add_row(s, f"{opt.weights[s]:.1%}", f"${last_prices[s]:,.2f}")
    console.print(t)
    console.print(
        f"Expected return [bold]{opt.expected_return:.1%}[/]  "
        f"vol [bold]{opt.volatility:.1%}[/]  Sharpe [bold]{opt.sharpe:.2f}[/]  "
        f"(frontier: {len(opt.frontier)} pts)"
    )
    console.print(
        f"Backtest: total [bold]{bt.total_return:.1%}[/]  CAGR {bt.cagr:.1%}  "
        f"maxDD {bt.max_drawdown:.1%}  Sharpe {bt.sharpe:.2f}"
    )
    console.print(
        f"1y Monte-Carlo: median [bold]{mc['p50']:.1%}[/]  "
        f"5–95%: {mc['p05']:.1%}…{mc['p95']:.1%}  P(loss) {mc['prob_loss']:.1%}\n"
    )

    broker = PaperBroker(cash=initial_cash, prices=last_prices)
    account = broker.get_account_state()
    orders = plan_orders(opt.weights, account, last_prices, investable=initial_cash)
    vr = validate(orders, account, last_prices, risk)
    verdict = "[green]PASS[/]" if vr.ok else "[red]BLOCK[/]"
    console.print(f"Circuit breaker: {verdict}  ({len(orders)} orders, {len(vr.fatal)} fatal, {len(vr.warnings)} warn)")
    for x in vr.violations:
        console.print(f"  - {x.severity.value}: {x.message}")
    if not vr.ok:
        console.print("[red]Halted by circuit breaker — no orders executed.[/]")
        return {"executed": False}

    for o in orders:
        broker.place_order(o)

    acct = broker.get_account_state()
    eq = acct.equity(last_prices)
    pos_t = Table(title="\nPaper positions after execution")
    pos_t.add_column("Ticker")
    pos_t.add_column("Shares", justify="right")
    pos_t.add_column("Avg cost", justify="right")
    pos_t.add_column("Value", justify="right")
    pos_t.add_column("Weight", justify="right")
    for s, p in acct.positions.items():
        val = p.quantity * last_prices[s]
        pos_t.add_row(s, f"{p.quantity:.4f}", f"${p.avg_cost:,.2f}", f"${val:,.2f}", f"{val / eq:.1%}")
    console.print(pos_t)
    console.print(
        f"Cash ${acct.cash:,.2f}  Equity ${eq:,.2f}  Invested ${eq - acct.cash:,.2f}  "
        f"Realized P&L ${broker.realized_pnl:,.2f}"
    )
    console.rule("[bold green]Demo 1 complete — full loop ran on paper money")
    return {"executed": True, "weights": opt.weights, "equity": eq, "orders": len(orders)}


if __name__ == "__main__":
    run_demo1()
