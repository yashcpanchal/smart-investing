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

## Status (2026-07-01 — provider-agnostic LLM core + real agent loop)
- Branch **`dev`**, pushed to origin (Yash is onboarding from this file).
- **LLM layer rebuilt for plug-and-play providers**: everything types against
  `llm/base.py::LLMClient` (chat with native tool-calling + complete/
  complete_json/complete_grounded). `GeminiClient` (function-calling added) and
  `AnthropicClient` (Claude Messages API, web_search grounding, refusal-safe)
  both implement it. `llm/factory.get_llm()` picks by env: **set
  ANTHROPIC_API_KEY and Claude takes over automatically** (or force with
  LLM_PROVIDER / override LLM_MODEL). Gemini remains the tested default.
- **The chat "agent" is now a real tool-use loop** (`llm/agent.py::run_agent`,
  bounded at 4 rounds): READ tools (get_portfolio incl. per-holding reasoning,
  get_stock_facts, get_neighbors, search_companies) execute live so it
  researches before answering; EDIT tools queue as the old action vocabulary
  and are applied deterministically with one rebuild per turn — the LLM still
  never touches money/math. Conversation history now reaches the model.
  /api/chat returns `researched`; ChatPanel renders the research trace.
  Live-verified against Gemini (function-calling roundtrip + grounded reply).
- Paper MVP (Phases 0–11) done; then a major UX rebuild turned the linear form
  into a **conversational, explorable, 5-view workspace**:
  - Corpus expanded ~28 → **72 US 10-K filers** (semis/equipment/datacenter/
    power/nuclear/defense) so supply chains are walkable.
  - **Explorable supply-chain graph** (walk upstream/downstream/peers).
  - **Conversation**: freeform chat drives build + refine; graph add/remove and
    edits route through one agent so all panels stay in sync.
  - **Per-holding reasoning**, a **per-stock detail panel** (yfinance facts), and
    an **industry-analysis panel** (supply-chain layers + Google-Search-grounded
    market commentary).
- ~87 tests passing; ruff + tsc + eslint + `next build` clean.
- **Gemini key was rotated** (new one in `.env`). NOTE: heavy build/testing
  exhausted the free-tier quota — LLM features fall back to deterministic text
  until quota resets, then the grounded analysis + nicer phrasing light up.

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
- `llm/` — `base.py` (**LLMClient contract**: `chat(messages, tools)` with
  provider-neutral ToolSpec/ToolCall/ChatMessage/LLMResponse, shared fail-fast
  retry, lenient JSON), `gemini.py` + `anthropic.py` (providers; both REST via
  httpx, no SDK deps), `factory.py` (`get_llm()` — LLM_PROVIDER/LLM_MODEL env
  selection, auto prefers Anthropic when keyed), `agent.py` (**tool-use loop**:
  read tools run live, mutations queue as actions; keyword fallback),
  `compiler.py` (prompt → StrategySpec + deterministic fallback), `explain.py`
  (grounded Explanation incl. per-holding `why`), `clarify.py`.
- `session.py` — in-memory `Session`/`SessionManager` (theme, knobs, pinned/
  excluded, cached base_spec, last proposal, transcript).
- `research.py` — `company_profile` (yfinance facts, cached) + `industry_brief`
  (deterministic supply-chain layers from the graph + grounded market analysis).
- `retrieval/graph_service.py` (cached index + merged graph), `curated.py`
  (hand-curated supplier/customer/competitor seed edges).
- `strategy.py` — `compile_strategy` orchestrator → Proposal (accepts a cached
  `spec` + shared `index`/`meta` to keep conversational rebuilds fast).
- `execution/` — `planner.py` (diff engine), `executor.py`.
- `scheduler.py` — autonomous `rebalance_once` + `RebalanceScheduler`.
- `persistence/repo.py` — DuckDB StateRepo (proposals, portfolio, audit).
- `api/app.py` — FastAPI (`create_app` DI). Endpoints: `/api/chat` (main loop),
  `/api/graph/{search,neighbors}`, `/api/stock/{sym}`, `/api/industry`,
  `/api/compile`, `/api/proposals/*`, `/api/rebalance`, `/api/clarify`.
  `si serve` wires the real Gemini client via `get_llm()`.
- `web/` — Next 16 / React 19 / Tailwind v4. Layout: persistent chat + tabbed
  workspace (Supply chain | Portfolio | Stocks | Industry). Components:
  ChatPanel, GraphCanvas (hand-rolled SVG force layout), PortfolioPanel,
  StocksPanel (per-stock tabs+dropdown), IndustryPanel, plus the recharts
  Allocation/Frontier charts and Findings.

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
1. Rotate GITHUB_TOKEN (Gemini key already rotated; add ANTHROPIC_API_KEY when
   we get one — the factory flips to Claude automatically).
2. Review `dev` → merge to `main`.
3. Phase 11: desktop-OAuth a Robinhood agentic account, `broker.list_tools()`,
   fill `_TOOL_MAP` + response parsing in `broker/robinhood.py`.
4. Agent polish: stream replies token-by-token, richer tool trace in the UI,
   let the agent call `complete_grounded` as a `web_research` read tool.
5. Polish: chunked embeddings (recall), foreign filers (40-F/20-F), source-weight
   configurator sliders, 13F/insider "smart money" scoring.
6. Phases 12 (auth) / 13 (compliance) / 14 (deploy) / 15 (scale).

## Working style (founder preference)
Run multiple confirmation subagents at each step; commit per phase to `dev` with
co-author trailer; paper-first, real money last.
