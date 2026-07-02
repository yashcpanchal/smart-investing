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

## Status (2026-07-02 — retrieval depth + smart money + streamed agent round)
- Branch **`dev`**, pushed to origin (Yash is onboarding from this file).
- **Three-stream round merged** (built by parallel worktree agents, reviewed by
  a 4-lens Opus panel + fix loop; 155 tests):
  - **Chunk-level dense retrieval** (`retrieval/chunk.py` + `index.py`): 10-Ks
    are now embedded as ~400-word chunks (80 overlap), per-ticker score =
    max-pooled chunk cosine. Fixes real truncation blindness — the MiniLM
    embedder saw only the first 256 tokens of ~35-150k-char filings; BM25 stays
    whole-doc; `query()` contract and the relevance-gate scale are unchanged.
    `store.documents()` now has a deterministic ORDER BY.
  - **Foreign filers**: ingest falls back 10-K → 20-F → 40-F (`si ingest
    --forms`); 20-F Item 4/3.D extraction (live-validated on ARM), 40-F AIF
    exhibit discovery via `EdgarClient.accession_index()` (live-validated on
    Cameco). CCJ/DNN/ARM stop being text-less phantom graph nodes — run
    `si ingest CCJ,DNN,ARM` to pull their filings in.
  - **Smart money** (`data/smart_money.py`, `data/managers.py`): 13F-HR
    holdings of 10 curated managers (CIKs live-verified) + Form 4 insider
    trades into new `inst_holdings`/`insider_trades` DuckDB tables (`si
    smart-money` ingests; per-manager pruning keeps exactly the latest 13F).
    `compute_smart_money_scores(store)` → deterministic 0..1 rank-normalized
    `smart_money_13f`/`smart_money_insider` (insider window anchored to max
    store date, never now()). Blended into universe ORDERING via the
    previously-dead `StrategySpec.source_weights` — the pure dense-cosine
    relevance gate is untouched, so smart money can reorder but never re-admit
    off-theme names. UI: "Signal weights" sliders in PortfolioPanel →
    `POST /api/session/{id}/knobs` (LLM-free path; chat can also drive it via
    the `set_source_weights` tool), smart-$ badge per holding.
  - **Agent polish**: `POST /api/chat/stream` (SSE) streams live tool-use
    progress; `researched` is now a rich trace `[{tool, args, preview}]`
    rendered as pills; new `web_research` READ tool (grounded search when an
    LLM key is present, graceful offline note otherwise); the client falls
    back to plain `/api/chat` ONLY when the stream provably never executed
    (`StreamUnavailableError`) — mid-stream failures surface as errors so a
    committed turn is never re-run. Fixed a latent bug where `industry_brief`
    imported a nonexistent `_loads_lenient` and silently un-grounded every
    grounded industry analysis since it shipped.
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
- **155 tests passing**; ruff + tsc + eslint + `next build` clean. All tests
  offline/deterministic (fake LLMs, in-memory stores, inline EDGAR fixtures).
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
  form-aware section extraction incl. 20-F/40-F + `accession_index`),
  `store.py` (DuckDB; + `inst_holdings`/`insider_trades`), `ingest.py`
  (multi-form fallback), `smart_money.py` (13F/Form 4 parse+ingest+scoring),
  `managers.py` (curated 13F filer CIKs).
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
1. Rotate GITHUB_TOKEN in `.env` (it is dead — 401; pushes/PRs use the
   repo-local `.git/.git-credentials` PAT). Add ANTHROPIC_API_KEY when we get
   one — the factory flips to Claude automatically.
2. Merge the open `dev` → `main` PR after Yash reviews.
3. Data refresh: `si ingest CCJ,DNN,ARM` (foreign filers now supported) and
   `si smart-money` (populates 13F/insider tables) against live EDGAR.
4. Phase 11: desktop-OAuth a Robinhood agentic account, `broker.list_tools()`,
   fill `_TOOL_MAP` + response parsing in `broker/robinhood.py`.
5. Agent: token-level streaming inside provider `chat()` (SSE plumbing +
   client state machine already exist); first-turn build progress events;
   live-test `web_research` grounding once quota/keys allow.
6. Smart money: news_sentiment/social weights have model+UI plumbing but no
   data source; CUSIP-based 13F matching (name-bigram only today); multi-
   quarter 13F history/deltas.
7. Phases 12 (auth) / 13 (compliance) / 14 (deploy) / 15 (scale).

## Working style (founder preference)
Run multiple confirmation subagents at each step; commit per phase to `dev` with
co-author trailer; paper-first, real money last.
