# Robinhood Agentic MCP integration (Phase 11)

## The model (important — Gemini's PRD got this backwards)

Robinhood **publishes** the MCP server at `https://agent.robinhood.com/mcp/trading`.
The agent is the **client**. So **we are the agent host**: our backend holds the
user's OAuth connection and *calls Robinhood's tools*. We do **not** deploy our
own MCP onto the user's account.

- Read access: all the user's accounts/positions/transactions.
- Trade access: **only** the dedicated **Agentic account** (separate funds).
- Robinhood adds its own safety: push notifications, preview/approve, instant
  disconnect. Our circuit breaker still runs *before* we submit anything.
- Beta: **US equities only** (matches our MVP scope).
- Onboarding is **desktop-OAuth only**.

## What's built

`src/smart_investing/broker/robinhood.py` — `RobinhoodMCPBroker(BrokerAdapter)`:
connection + tool-discovery scaffolding is real; it drops in for `PaperBroker`
with zero upstream changes (same interface). The tool-name → method mapping is a
**best guess** and must be confirmed live.

## Steps to finish (needs a human + a funded agentic account)

1. `uv pip install -e ".[robinhood]"` (adds the `mcp` client).
2. On a **desktop**, connect an agent to Robinhood and open an **Agentic account**
   (see Robinhood's "Agentic Trading overview"). Obtain the OAuth access token
   our backend will use.
3. Put the token where the broker can read it (env/secret store), construct
   `RobinhoodMCPBroker(access_token=...)`.
4. Run `broker.list_tools()` to get the **real** tool names + parameter schemas.
   Fill in `_TOOL_MAP` and implement the response parsing in:
   `get_account_state`, `get_buying_power`, `get_positions`, `place_order`,
   `get_order_status`.
5. Start with a **$1 paper-equivalent / smallest real trade** to validate the
   full path: our compile → circuit breaker → `place_order` → Robinhood fill →
   `get_positions` reconciliation.
6. Feature-flag the broker swap (PaperBroker ↔ RobinhoodMCPBroker) in the API's
   `get_broker()`.

## Safety / compliance gates before real users (Phase 13)

- Executing trades for *other* people or charging AUM → investment-adviser /
  broker territory. Fine for yourselves / paper / bring-your-own-agent; gate
  public onboarding behind real compliance review.
- Keep the deterministic circuit breaker in front of every order regardless of
  Robinhood's own checks (defense in depth).
