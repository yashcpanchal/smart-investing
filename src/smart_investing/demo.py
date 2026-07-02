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


DEMO2_TICKERS = [
    # nuclear / uranium
    "CCJ", "LEU", "BWXT", "SMR", "OKLO", "UEC", "UUUU", "DNN", "NNE",
    # quantum / advanced compute
    "IONQ", "RGTI", "QBTS", "NVDA", "AMD", "AVGO", "MRVL", "ARM",
    # defense / aerospace
    "LMT", "RTX", "NOC", "LHX",
    # clean energy
    "FSLR", "ENPH", "NEE",
    # unrelated controls
    "KO", "WMT", "PG", "MCD",
]

DEFAULT_PROMPT = (
    "Invest in nuclear energy and uranium mining, include the supply chain, "
    "and keep it diversified"
)


def run_demo2(
    prompt: str = DEFAULT_PROMPT,
    initial_cash: float = 10_000.0,
    live: bool = True,
    reingest: bool = False,
) -> object:
    """Demo 2 — the full MVP loop: natural-language prompt -> universe -> optimized
    portfolio -> paper execution."""
    from smart_investing.data import Store, ingest_companies
    from smart_investing.llm.factory import get_llm
    from smart_investing.strategy import compile_strategy

    console.rule("[bold]Demo 2 — natural language -> portfolio")
    console.print(f"Prompt: [italic]{prompt}[/]\n")

    store = Store()
    if reingest or store.count("documents") < 20:
        console.print("Ingesting corpus from EDGAR (one-time, cached to data/)...")
        summary, store = ingest_companies(DEMO2_TICKERS, store=store)
        console.print(f"  ingested {len(summary['ok'])}, skipped {len(summary['skipped'])}\n")

    llm = get_llm()
    console.print(f"LLM: {'Gemini' if llm else 'deterministic fallback (no key)'}")
    proposal = compile_strategy(prompt, store, initial_cash=initial_cash, live=live, llm=llm)
    spec = proposal.spec
    line = f"Parsed themes {spec.themes}  objective={spec.objective.value} cap={spec.risk.concentration_cap:.0%}"
    if spec.risk.target_volatility:
        line += f" targetVol={spec.risk.target_volatility:.0%}"
    console.print(line)
    if spec.exclude_symbols:
        console.print(f"Excluding: {spec.exclude_symbols}")

    ut = Table(title="\nAsset universe (search results)")
    ut.add_column("Deg")
    ut.add_column("Ticker")
    ut.add_column("Weight", justify="right")
    ut.add_column("Relevance", justify="right")
    ut.add_column("Name")
    for a in proposal.universe.assets:
        w = proposal.target_weights.get(a.symbol, 0.0)
        rel = a.scores.get("relevance", a.scores.get("graph_proximity", 0.0))
        ut.add_row(str(a.degree), a.symbol, f"{w:.1%}", f"{rel:.2f}", a.name[:34])
    console.print(ut)

    opt, bt = proposal.optimization, proposal.backtest
    console.print(f"Expected {opt.expected_return:.1%}  vol {opt.volatility:.1%}  Sharpe {opt.sharpe:.2f}")
    if bt:
        console.print(f"Backtest total {bt.total_return:.1%}  CAGR {bt.cagr:.1%}  maxDD {bt.max_drawdown:.1%}")
    console.print(f"\n[bold]{proposal.rationale}[/]")

    if proposal.trades:
        prices = {o.symbol: o.est_price for o in proposal.trades if o.est_price}
        broker = PaperBroker(cash=initial_cash, prices=prices)
        for o in proposal.trades:
            broker.place_order(o)
        acct = broker.get_account_state()
        console.print(
            f"Paper-executed {len(proposal.trades)} orders. "
            f"Cash ${acct.cash:,.2f}  Equity ${acct.equity(prices):,.2f}"
        )
    console.rule("[bold green]Demo 2 complete — NL prompt to executed paper portfolio")
    return proposal


if __name__ == "__main__":
    run_demo1()

