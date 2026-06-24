# smart-investing — Roadmap

Natural-language thematic investing: a user types a thesis ("invest in quantum +
nuclear, follow where private money is going"), and the system builds a
mathematically-optimized US-equity portfolio and autonomously rebalances it
through Robinhood's Agentic Trading (MCP).

**Architecture note (important):** Robinhood *publishes* the MCP server at
`https://agent.robinhood.com/mcp/trading`; the agent is the *client*. **We are the
agent host** — our backend holds the user's OAuth connection and calls Robinhood's
tools (read portfolio/buying-power; place orders only in the dedicated agentic
account). We do **not** deploy our own MCP onto their account. Agentic Trading
launched 2026-05-27, equities-only beta.

**Constraint:** "scrappy" = **no paid services**. Free-but-heavy tooling (JAX,
C++/Rust, local Spark, FAISS, Neo4j Community) is welcome. Free data only: SEC
EDGAR (10-K, 13F-HR, Form 4), yfinance, local embeddings. US equities only for MVP.

**Build principle:** everything sits behind a `BrokerAdapter`, defaulting to a
`PaperBroker` (no real money). Paper-first → we build and demo the entire loop,
including autonomous rebalancing, with zero real-money risk.

> Ordering note: **the frontend (Phase 10) ships before the real Robinhood
> integration (Phase 11)** — per founder decision.

---

## Stage A — Foundation
- [x] **Phase 0 — Scaffolding & contracts.** Monorepo layout, env/config, shared
  domain types (`StrategySpec`, `AssetUniverse`, `TargetWeights`, `Order`,
  `AccountState`, `Proposal`), test harness, CI.
  *Done when:* tests run green in CI; the type contracts every later phase imports exist.

## Stage B — Core Quant Engine (the differentiator)
- [x] **Phase 1 — Optimizer + backtester.** Price loader (yfinance→DuckDB),
  returns/covariance, cvxpy mean-variance (max-Sharpe / min-vol / target-vol,
  no-short, concentration cap), JAX (optional) Monte-Carlo + efficient frontier,
  backtester, metrics (Sharpe, vol, max drawdown, CAGR).
  *Done when:* `optimize(tickers, constraints)` → weights + frontier + backtest, deterministically.
- [x] **Phase 2 — Deterministic circuit breaker.** Non-LLM validator: Σ|w|≤1
  (no margin), per-asset concentration cap, cash sufficiency, PDT block, wash-sale
  flag, order sanity. Adversarial test suite.
  *Done when:* `validate(orders, account)` → pass/fail + reasons, all bad inputs caught.
- [x] **Phase 3 — Broker abstraction + PaperBroker.** `BrokerAdapter` interface,
  `PaperBroker` simulating fills, holdings ledger, cost-basis, P&L.
  *Done when:* full buy→hold→sell cycle runs on paper with correct P&L.
- [x] 🎯 **Demo 1 (internal prototype):** hardcoded theme → fixed tickers →
  optimize → validate → paper-execute → positions & P&L. End-to-end on fake money.

## Stage C — Data & Thematic Intelligence (the "indirect connections" moat)
- [ ] **Phase 4 — Data ingestion.** EDGAR client (10-K Item 1/1A, 13F-HR, Form 4),
  ticker↔CIK master, DuckDB storage; price-history store.
  *Done when:* ingest ~300–500 companies; query business text + filings.
- [ ] **Phase 5 — Thematic retrieval.** Local embeddings + FAISS/Chroma + BM25 +
  RRF hybrid fusion; LLM relation-extraction → knowledge graph
  (supplier/customer/competitor); graph traversal for indirect names; 13F/insider
  "smart-money" scoring; configurable fusion ranking (the slider weights).
  *Done when:* `theme prompt → ranked ticker universe` with rationale + prunable graph.
- [ ] **Phase 6 — Strategy compiler / orchestrator.** LLM prompt → structured
  `StrategySpec`; chain retrieval → optimizer → `Proposal`.
  *Done when:* `compile(prompt, config) → Proposal`.
- [ ] 🎯 **Demo 2 (MVP-alpha):** natural-language thesis → real universe →
  optimized portfolio → paper execution. The full loop on fake money.

## Stage D — State, Autonomy & API
- [ ] **Phase 7 — Persistence & state.** Users, strategies, positions, cost-basis,
  run history, immutable audit log.
  *Done when:* state survives restarts and is reloadable.
- [ ] **Phase 8 — Scheduler & autonomous rebalancing.** Bi-weekly re-ingest →
  re-optimize → diff engine (target vs current) → validator → propose/execute;
  manual-approval vs fully-autonomous modes; notifications.
  *Done when:* a scheduled rebalance fires, proposes, and executes on approval (paper).
- [ ] **Phase 9 — API layer (FastAPI).** REST + WebSocket over compiler, proposals,
  approvals, account state, history; OpenAPI docs.
  *Done when:* the whole backend is usable over HTTP.

## Stage E — Product & Real Money (frontend BEFORE Robinhood)
- [ ] **Phase 10 — Frontend (Next.js).** Connect/onboarding, prompt interface,
  anti-black-box configurator (source sliders, risk, indirect toggle),
  efficient-frontier + allocation charts, knowledge-graph network viz with
  branch-pruning, proposal review + one-click approve, notification center,
  portfolio/P&L dashboard. Runs against PaperBroker + the API.
  *Done when:* a non-technical user can go prompt → tune → review → approve → watch, in-browser.
- [ ] **Phase 11 — Real Robinhood MCP integration.** MCP client to
  `agent.robinhood.com/mcp/trading`, OAuth onboarding, `RobinhoodMCPBroker`
  implementing `BrokerAdapter`, reconcile RH preview/approve/push with our flow;
  feature-flag swap PaperBroker ↔ Robinhood.
  *Done when:* connect a real agentic account, read portfolio, place a real $1 trade.
- [ ] **Phase 12 — Auth & multi-tenancy.** User accounts, encrypted token storage,
  per-user isolation.
- [ ] 🎯 **Demo 3 (MVP-beta):** real (small) money, full UI, autonomous
  rebalancing, multiple users. First external testers.

## Stage F — Safety, Launch, Scale
- [ ] **Phase 13 — Safety, compliance & observability.** RIA/adviser gating,
  disclaimers/ToS, kill-switch, rate limits, logging/metrics/alerting, incident playbook.
- [ ] **Phase 14 — Deployment & beta launch.** Docker, hosting, managed DB,
  secrets, CI/CD, monitoring; closed beta.
- [ ] **Phase 15 — Scale & expand.** Spark full-corpus; options/crypto when RH GA;
  finance-tuned embeddings; learned ranking; advanced risk (Black-Litterman, CVaR);
  live sentiment; mobile; monetization.

---

### Critical path
`Phase 0 → 1 → 3 → (Demo 1) → 6 → (Demo 2) → 10 → 11 → (Demo 3)`

### Parallelizable
While one builds the quant engine (1–3), the other builds data + retrieval (4–5);
they meet at Phase 6. Frontend (10) starts against the API (9) as soon as endpoints stub out.
