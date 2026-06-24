# smart-investing

Natural-language thematic investing. Type an investment thesis ("invest in
quantum + nuclear, follow where private money is going") and the system builds a
mathematically-optimized US-equity portfolio and autonomously rebalances it
through Robinhood's Agentic Trading (MCP).

The LLM picks the *assets*; a deterministic engine picks the *weights* (Modern
Portfolio Theory), a hard non-LLM circuit breaker guards every order, and a
pluggable broker executes — **paper money by default**, real Robinhood later.

See [ROADMAP.md](ROADMAP.md) for the full phase plan (0 → finished product).

## Status

| Phase | What | State |
|------|------|-------|
| 0 | Scaffolding, domain contracts, config | ✅ |
| 1 | Quant engine (MPT optimizer, frontier, backtest, Monte-Carlo) | ✅ |
| 2 | Deterministic circuit breaker | ✅ |
| 3 | Broker abstraction + PaperBroker | ✅ |
| — | **Demo 1**: theme → optimize → validate → paper-execute | ✅ |
| 4+ | Data ingest, retrieval, LLM compiler, API, frontend, Robinhood | in progress |

## Quickstart

```bash
git clone https://github.com/yashcpanchal/smart-investing.git
cd smart-investing
uv venv && uv pip install -e ".[dev]"

# Demo 1 — full loop on paper money (uses live yfinance, falls back to synthetic)
.venv/Scripts/python -m smart_investing.demo      # Windows
# python -m smart_investing.demo                  # macOS/Linux

# Optimize an explicit basket
si optimize NVDA,AMD,AVGO,TSM --objective max_sharpe

# Tests
pytest -q
```

Copy `.env.example` → `.env` for optional keys (LLM, EDGAR user-agent). The engine
runs fully without them.

## Architecture (one paragraph)

Robinhood publishes the MCP server (`agent.robinhood.com/mcp/trading`); the agent
is the client. **We are the agent host** — our backend holds the user's OAuth
connection and calls Robinhood's tools. Our moat is everything around the agent:
supply-chain-aware universe construction, MPT math, persistent state + scheduled
rebalancing, and a deterministic guardrail. Everything executes behind a
`BrokerAdapter`, defaulting to `PaperBroker` so the whole loop runs with zero
real-money risk.

## License

Apache-2.0
