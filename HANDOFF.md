# HANDOFF — resume here

Everything needed to pick this project up cold. (See also `ROADMAP.md`,
`PROGRESS.md`, `docs/robinhood_mcp.md`.)

## What this is
Natural-language thematic investing. A user types an investment thesis; the
system searches real SEC filings, builds a supply-chain-aware ticker universe,
optimizes weights with Modern Portfolio Theory, runs a hard non-LLM circuit
breaker, and executes on **paper money** (Robinhood Agentic MCP comes later).

**Pipeline:** prompt → Gemini parses a `StrategySpec` → hybrid retrieval
(sentence-transformer embeddings + BM25 + RRF) over EDGAR 10-Ks + co-mention
graph → cvxpy MPT optimizer (+ efficient frontier) → deterministic circuit
breaker → PaperBroker execution → persisted portfolio. FastAPI backend, Next.js
frontend.

## Status (2026-06-24)
- Branch **`dev`** (10 commits; `main` untouched), pushed, working tree clean.
- Latest commit: `7c69d9c`.
- **Phases 0–10 DONE** + Demo 1 + Demo 2; **Phase 11 (Robinhood MCP) scaffolded**
  (needs desktop OAuth). Phases 12–15 not started (need product decisions).
- ~58 tests passing, ruff clean, GitHub Actions CI.
- Holistic review verdict: **ship-ready for paper MVP.**

## Run it
```bash
cd C:\Users\dhruv\smart-investing
uv venv && uv pip install -e ".[dev,api,retrieval]"     # core + api + embeddings
si ingest CCJ,LEU,BWXT,SMR,OKLO,UEC,IONQ,RGTI,NVDA,AMD,LMT,NOC,KO,WMT   # build corpus (data/*.duckdb, gitignored)
si serve                                  # FastAPI → http://localhost:8000
cd web && npm install && npm run dev      # UI → http://localhost:3000
# headless one-shot:
si strategy "Invest in nuclear and uranium, lower risk, diversified"
si demo            # Demo 1 (hardcoded basket)
pytest -q          # tests
```
Optional Robinhood client: `uv pip install -e ".[robinhood]"`.

## Code map (`src/smart_investing/`)
- `domain/types.py` — all pydantic contracts (StrategySpec, AssetUniverse, Order,
  AccountState, Proposal, enums). Money is float (documented MVP tradeoff).
- `data/` — `prices.py` (yfinance + synthetic GBM), `edgar.py` (EDGAR client +
  section extraction), `store.py` (DuckDB), `ingest.py`.
- `quant/` — `optimizer.py` (max-Sharpe via Charnes-Cooper, min-vol, target-vol;
  `_convex.py` effective-cap + capped-simplex projection), `frontier`, `backtest`,
  `montecarlo`, `metrics`.
- `risk/circuit_breaker.py` — deterministic gate; values ONLY from trusted price
  feed (never caller est_price).
- `broker/` — `base.py` (BrokerAdapter), `paper.py` (PaperBroker), `robinhood.py`
  (Phase 11 scaffold).
- `retrieval/` — `embeddings.py` (SentenceTransformer + TF-IDF fallback, model
  cached), `bm25.py`, `fusion.py` (RRF k=20), `graph.py` (co-mention, distinctive
  bigram), `index.py`, `universe.py` (build_universe: direct + indirect, relevance
  gate).
- `llm/` — `gemini.py` (REST client, retry/backoff), `compiler.py` (prompt →
  StrategySpec + deterministic fallback).
- `strategy.py` — `compile_strategy` orchestrator → Proposal.
- `execution/` — `planner.py` (diff engine), `executor.py`.
- `scheduler.py` — autonomous `rebalance_once` + `RebalanceScheduler`.
- `persistence/repo.py` — DuckDB StateRepo (proposals, portfolio, audit).
- `api/app.py` — FastAPI (`create_app` DI). `web/` — Next.js 16 + recharts.

## Key decisions & gotchas
- **We are the agent HOST, not the MCP server** (Robinhood publishes the MCP;
  Gemini's PRD got this backwards). See `docs/robinhood_mcp.md`.
- "Scrappy" = **no paid services**; free-but-heavy tools (cvxpy, sentence-
  transformers, DuckDB) are fine. Only paid gap skipped: private/VC flow data
  (proxied by 13F/insider — not yet wired).
- Gemini key has an unusual `AQ.` prefix but works as an API key (`?key=` or
  `x-goog-api-key` header). `gemini-2.5-flash`. Keep calls low (free tier).
- Root `.gitignore` `lib/` was swallowing `web/src/lib/` — fixed (anchored `/lib/`).
- Git pushes authenticate via a PAT in `.git/.git-credentials` (repo-local helper);
  no GitHub remote auth otherwise.
- Optimizer relaxes the concentration cap to 1/n for tiny universes; the breaker
  and the proposal rationale reflect the effective cap.

## Secrets
`.env` (gitignored, NEVER committed) holds `GEMINI_API_KEY` and `GITHUB_TOKEN`.
**Both should be rotated** — they passed through a chat. `.env.example` shows the
shape. The app runtime only needs `GEMINI_API_KEY` (+ free EDGAR/yfinance).

## Pending / next
1. Rotate the two secrets.
2. Review `dev` → merge to `main`.
3. Phase 11: desktop-OAuth a Robinhood agentic account, `broker.list_tools()`,
   fill `_TOOL_MAP` + response parsing in `broker/robinhood.py`.
4. Polish: chunked embeddings (recall), foreign filers (40-F/20-F), source-weight
   configurator sliders, 13F/insider "smart money" scoring.
5. Phases 12 (auth) / 13 (compliance) / 14 (deploy) / 15 (scale).

## Working style (founder preference)
Run multiple confirmation subagents at each step; commit per phase to `dev` with
co-author trailer; paper-first, real money last.
