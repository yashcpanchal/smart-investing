"""`si` command-line entry point."""

from __future__ import annotations

import typer

app = typer.Typer(help="smart-investing — thematic portfolio engine", add_completion=False)


@app.command()
def demo(
    offline: bool = typer.Option(False, help="Force synthetic data instead of yfinance."),
    cash: float = typer.Option(10_000.0, help="Starting paper cash."),
    cap: float = typer.Option(0.30, help="Per-asset concentration cap."),
) -> None:
    """Run Demo 1: theme -> optimize -> validate -> paper-execute."""
    from smart_investing.demo import run_demo1

    run_demo1(initial_cash=cash, concentration_cap=cap, live=not offline)


@app.command()
def optimize(
    tickers: str = typer.Argument(..., help="Comma-separated tickers, e.g. NVDA,AMD,AVGO"),
    objective: str = typer.Option("max_sharpe", help="max_sharpe | min_vol | target_vol"),
    cap: float = typer.Option(0.30, help="Per-asset concentration cap."),
    offline: bool = typer.Option(False, help="Use synthetic data."),
) -> None:
    """Optimize a portfolio for an explicit ticker list."""
    from rich.console import Console

    from smart_investing.data import load_prices, synthetic_prices
    from smart_investing.domain.types import Objective, RiskParams
    from smart_investing.quant import optimize as run_opt

    console = Console()
    syms = [t.strip().upper() for t in tickers.split(",") if t.strip()]
    px = synthetic_prices(syms) if offline else load_prices(syms, period="2y")
    res = run_opt(px, objective=Objective(objective), risk=RiskParams(concentration_cap=cap))
    for s, w in sorted(res.weights.items(), key=lambda kv: -kv[1]):
        console.print(f"{s:>6}  {w:6.1%}")
    console.print(
        f"\nExpected {res.expected_return:.1%}  vol {res.volatility:.1%}  Sharpe {res.sharpe:.2f}"
    )


@app.command()
def strategy(
    prompt: str = typer.Argument(..., help="Natural-language investment thesis."),
    cash: float = typer.Option(10_000.0, help="Starting paper cash."),
    offline: bool = typer.Option(False, help="Use synthetic prices."),
    reingest: bool = typer.Option(False, help="Re-ingest the demo corpus from EDGAR."),
) -> None:
    """Demo 2: compile a natural-language thesis into a paper-executed portfolio."""
    from smart_investing.demo import run_demo2

    run_demo2(prompt=prompt, initial_cash=cash, live=not offline, reingest=reingest)


@app.command()
def ingest(
    tickers: str = typer.Argument(..., help="Comma-separated tickers to ingest from EDGAR."),
    forms: str = typer.Option(
        "10-K,20-F,40-F",
        help="Comma-separated annual-report forms to try, in order "
        "(20-F/40-F cover foreign filers like ARM/CCJ).",
    ),
) -> None:
    """Pull latest annual-report Business/Risk sections from SEC EDGAR into the local store."""
    from rich.console import Console

    from smart_investing.data import ingest_companies

    # Avoid UnicodeEncodeError when stdout is redirected to a non-UTF-8 file on
    # Windows (rich falls back to the console's cp1252 codec); ASCII markers are safe.
    console = Console()
    syms = [t.strip().upper() for t in tickers.split(",") if t.strip()]
    form_list = [f.strip().upper() for f in forms.split(",") if f.strip()]
    summary, store = ingest_companies(syms, forms=form_list)
    for t, sizes in summary["ok"]:
        console.print(f"[green][OK][/] {t}: " + ", ".join(f"{k} {n:,}c" for k, n in sizes.items()))
    for t, why in summary["skipped"]:
        console.print(f"[yellow][--][/] {t}: {why}")
    for t, err in summary["errors"]:
        console.print(f"[red][XX][/] {t}: {err}")
    console.print(
        f"\nStore: {store.count('companies')} companies, "
        f"{store.count('filings')} filings, {store.count('documents')} documents"
    )
    store.close()


@app.command(name="smart-money")
def smart_money(
    per_ticker: int = typer.Option(8, help="Recent Form 4 filings pulled per corpus ticker."),
    skip_form4: bool = typer.Option(False, help="Skip Form 4 insider ingestion."),
    skip_13f: bool = typer.Option(False, help="Skip 13F institutional ingestion."),
) -> None:
    """Ingest 'smart money' signals from SEC EDGAR into the local store:
    Form 4 insider trades for every corpus ticker + latest 13F-HR holdings of
    the curated institutional managers. Run `si ingest` first."""
    from rich.console import Console

    from smart_investing.data.edgar import EdgarClient
    from smart_investing.data.managers import DEFAULT_MANAGERS
    from smart_investing.data.smart_money import ingest_13f, ingest_insider_trades
    from smart_investing.data.store import Store

    console = Console()
    store = Store()
    tickers = [row[0] for row in store.companies()]
    if not tickers:
        console.print("[yellow]Store has no companies — run `si ingest TICKERS` first.[/]")
        store.close()
        raise typer.Exit(1)

    client = EdgarClient()
    try:
        if not skip_form4:
            res = ingest_insider_trades(store, tickers, client=client, per_ticker_limit=per_ticker)
            for t, n in res["ok"]:
                console.print(f"[green][OK][/] {t}: {n} insider transactions")
            for t, why in res["errors"]:
                console.print(f"[red][XX][/] {t}: {why}")
        if not skip_13f:
            res = ingest_13f(store, client=client, managers=DEFAULT_MANAGERS)
            for name, matched, total in res["ok"]:
                console.print(f"[green][OK][/] {name}: {matched}/{total} holdings matched to corpus")
            for name, why in res["errors"]:
                console.print(f"[red][XX][/] {name}: {why}")
        console.print(
            f"\nStore: {store.count('insider_trades')} insider trades, "
            f"{store.count('inst_holdings')} institutional holdings"
        )
    except Exception as e:  # offline / EDGAR unreachable — fail politely
        console.print(f"[red]EDGAR unreachable ({str(e)[:120]}). Are you online?[/]")
        raise typer.Exit(1) from None
    finally:
        client.close()
        store.close()


@app.command()
def serve(host: str = "127.0.0.1", port: int = 8000) -> None:
    """Run the HTTP API (FastAPI). Ingest a corpus first with `si ingest`."""
    import uvicorn

    uvicorn.run("smart_investing.api.app:app", host=host, port=port)


if __name__ == "__main__":
    app()
