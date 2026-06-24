# Build progress log

Running log of the autonomous overnight build. Newest first.

## Session 1 (overnight, 2026-06-24)

### ✅ Done & pushed to `dev` (Phases 0–6, two demos working)

- **Phase 0** — repo scaffold, `uv` env, hatchling build, pydantic domain
  contracts, config via `.env`, CI workflow, `.gitattributes`.
- **Phase 1** — quant engine: yfinance + synthetic prices; cvxpy optimizer
  (max-Sharpe via Charnes-Cooper, min-vol, target-vol; no-short; concentration
  cap; capped-simplex projection; solver fallback); efficient frontier; backtest;
  Monte-Carlo; metrics.
- **Phase 2** — deterministic circuit breaker: sanity-first, no-short, cash,
  post-trade concentration, max-positions, PDT. Trusts only the market price
  feed (never caller `est_price`).
- **Phase 3** — `BrokerAdapter` + `PaperBroker` (cost-basis, realized P&L, cents
  rounding, order ids/status). Diff-engine planner.
- **🎯 Demo 1** — `python -m smart_investing.demo`: theme → optimize → validate →
  paper-execute, on live data.
- **Phase 4** — SEC EDGAR ingest (10-K Business/Risk sections, last-header
  heuristic, html.unescape), DuckDB store, `si ingest`. Verified on real filings.
- **Phase 5** — thematic retrieval: SentenceTransformer (all-MiniLM) + TF-IDF
  fallback, hand-rolled BM25, RRF fusion (k=20), relevance gate, co-mention
  graph (distinctive-bigram matching) → `build_universe`.
- **Phase 6** — LLM compiler: Gemini (REST, retry/backoff) parses prompt →
  StrategySpec; deterministic fallback. `compile_strategy` orchestrator → Proposal.
- **🎯 Demo 2 (MVP-alpha)** — `si strategy "<thesis>"`: NL prompt → Gemini spec →
  hybrid+graph universe → MPT → circuit breaker → paper execution. Verified:
  "nuclear + uranium, lower risk, diversified" → target_vol@20% cap, nuclear/
  defense portfolio, executed on paper.

**Tests:** 49 passing. **Lint:** ruff clean. **CI:** GitHub Actions.

### Reviewed by confirmation subagents (per founder request)
Domain contracts, optimizer math (Charnes-Cooper verified), risk/broker design,
then adversarial code audits + a retrieval-quality review. Folded in: order ids/
enums/buying_power; trusted-price circuit breaker (closed 2 bypasses);
effective-cap + capped-simplex projection (no cap-violating weights); short-
history/single-asset guards; raw-cosine relevance + gate; bigram co-mention.

### Known limitations / next
- **Phase 7** persistence (Postgres), **Phase 8** scheduler + autonomous
  rebalance, **Phase 9** FastAPI, **Phase 10** Next.js frontend (before
  Robinhood, per your call), **Phase 11** real Robinhood MCP.
- Retrieval 5b: chunk+max-pool embeddings (currently ~256-token truncation);
  foreign filers (40-F/20-F, e.g. CCJ); LLM-extracted supplier/customer edges.
- Thematic tilt: optimizer can drift to high-Sharpe defense names over volatile
  pure-play juniors (math is correct; product may want a relevance tilt).

### Notes for the morning
- Secrets in `.env` (gitignored): Gemini + GitHub token. **Please rotate both** —
  they passed through chat.
- All work on `dev`; `main` untouched. Review + merge at your leisure.
- `si demo` / `si strategy "..."` / `si ingest TICKERS` / `si optimize T1,T2`.
