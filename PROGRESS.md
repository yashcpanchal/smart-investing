# Build progress log

Running log of the autonomous overnight build. Newest first.

## Session 1 (overnight, 2026-06-24) — Phases 0–10 + Phase 11 scaffold

A working MVP, end-to-end, on paper money — from a natural-language thesis to an
executed, persisted portfolio with a real web UI. **All on branch `dev`.**

### Pipeline (works end to end)
prompt → **Gemini** parses a StrategySpec → **hybrid search** (sentence-transformer
embeddings + BM25 + RRF) over **real SEC EDGAR filings** + **supply-chain graph**
for indirect names → **MPT optimizer** (cvxpy: max-Sharpe / min-vol / target-vol,
efficient frontier) → **deterministic circuit breaker** → **PaperBroker**
execution → **persisted** portfolio. Exposed via **FastAPI**, driven by a
**Next.js** UI (verified in a real browser via Playwright).

### Phases
- ✅ 0 scaffold/contracts/CI · 1 quant engine · 2 circuit breaker · 3 paper broker
- ✅ Demo 1 (theme→optimize→validate→execute, live data)
- ✅ 4 EDGAR ingest · 5 thematic retrieval · 6 LLM compiler (Gemini)
- ✅ Demo 2 (NL prompt → portfolio, live)
- ✅ 7 persistence · 8 autonomous rebalance + scheduler · 9 FastAPI
- ✅ 10 Next.js frontend (verified in-browser)
- ◐ 11 Robinhood MCP — **scaffolded**, needs desktop OAuth (see docs/robinhood_mcp.md)
- ☐ 12 auth/multi-tenancy · 13 safety/compliance · 14 deploy · 15 scale

### How to run
```
uv venv && uv pip install -e ".[dev,api,retrieval]"
si ingest CCJ,LEU,BWXT,SMR,OKLO,UEC,IONQ,RGTI,NVDA,AMD,LMT,NOC,KO,WMT   # corpus
si serve            # API on :8000
cd web && npm install && npm run dev    # UI on :3000
# or headless: si strategy "Invest in nuclear and uranium, lower risk, diversified"
```

### Reviewed by confirmation subagents throughout (per founder request)
Domain/optimizer/risk design pre-build; adversarial code audits (found+fixed:
trusted-price circuit breaker closing 2 bypasses; effective-cap + capped-simplex
projection; short-history/single-asset guards); retrieval-quality review (raw-
cosine relevance + gate, RRF k=20, distinctive-bigram co-mention); plus a final
holistic integration/security pass.

### Stats
~58 tests, ruff clean, CI (GitHub Actions). 9 commits on `dev`.

### ⚠️ For the morning
- **Rotate the secrets in `.env`** (Gemini key + GitHub PAT) — they passed
  through chat. (`.env` is gitignored; never committed — verified.)
- Everything is on `dev`; `main` untouched. Review and merge at your leisure.
- Next high-value polish: chunked embeddings (recall), foreign filers (40-F/20-F),
  source-weight sliders in the configurator, then Phase 11 OAuth.
