"""Robinhood Agentic MCP broker (Phase 11 scaffold).

Integration model (verified from Robinhood docs): Robinhood PUBLISHES the MCP
server at https://agent.robinhood.com/mcp/trading. WE are the client/agent host.
Our backend holds the user's OAuth connection and calls Robinhood's tools to read
the portfolio and place orders in the dedicated *agentic account*.

Status: connection + tool-discovery scaffolding is real; the tool-name -> method
mapping below is a BEST GUESS and MUST be confirmed against the live server once
authenticated (run `list_tools()` and fill in `_TOOL_MAP`). Onboarding is
desktop-OAuth only, so this can't be exercised headless — it's wired so that once
you authenticate and confirm names, it swaps in for PaperBroker with no upstream
changes.

Requires the optional `mcp` client: `uv pip install -e ".[robinhood]"`.
"""

from __future__ import annotations

import asyncio
from typing import Any

from smart_investing.broker.base import BrokerAdapter
from smart_investing.config import settings
from smart_investing.domain.types import AccountState, Order, OrderResult, Position

# Best-guess tool names — CONFIRM via list_tools() against the live MCP server.
_TOOL_MAP = {
    "account": "get_account",          # -> cash, buying power, account id
    "positions": "get_positions",      # -> held positions
    "buying_power": "get_buying_power",
    "place_order": "place_order",      # -> args: symbol, side, quantity/amount, type
    "order_status": "get_order_status",
}


class RobinhoodNotConfigured(RuntimeError):
    pass


class RobinhoodMCPBroker(BrokerAdapter):
    def __init__(self, access_token: str | None = None, mcp_url: str | None = None) -> None:
        self.mcp_url = mcp_url or settings.robinhood_mcp_url
        self.access_token = access_token  # OAuth token obtained via desktop onboarding
        self._tool_map = dict(_TOOL_MAP)

    # ---- low-level MCP plumbing ----
    def _run(self, coro):
        return asyncio.run(coro)

    async def _session(self):
        """Open an MCP client session to the Robinhood Trading MCP.

        Uses streamable-HTTP transport with the user's OAuth bearer token. The
        exact auth handshake (token vs full OAuth flow) is finalized during
        desktop onboarding; wire the obtained token here.
        """
        try:
            from mcp import ClientSession
            from mcp.client.streamable_http import streamablehttp_client
        except ImportError as e:  # pragma: no cover
            raise RobinhoodNotConfigured(
                "mcp client not installed — run: uv pip install -e \".[robinhood]\""
            ) from e
        if not self.access_token:
            raise RobinhoodNotConfigured("no OAuth access_token — complete desktop onboarding first")
        headers = {"Authorization": f"Bearer {self.access_token}"}
        return streamablehttp_client(self.mcp_url, headers=headers), ClientSession

    async def _alist_tools(self) -> list[dict]:
        transport_cm, ClientSession = await self._session()
        async with transport_cm as (read, write, *_), ClientSession(read, write) as session:
            await session.initialize()
            resp = await session.list_tools()
            return [{"name": t.name, "description": t.description} for t in resp.tools]

    async def _acall(self, tool: str, args: dict) -> Any:
        transport_cm, ClientSession = await self._session()
        async with transport_cm as (read, write, *_), ClientSession(read, write) as session:
            await session.initialize()
            return await session.call_tool(tool, args)

    def list_tools(self) -> list[dict]:
        """Discover the real tool names/params. RUN THIS FIRST after auth."""
        return self._run(self._alist_tools())

    def call_tool(self, tool: str, args: dict | None = None) -> Any:
        return self._run(self._acall(tool, args or {}))

    # ---- BrokerAdapter (parsing of responses needs confirmation vs live schema) ----
    def get_account_state(self) -> AccountState:
        raise NotImplementedError(
            "Map _TOOL_MAP['account'] response -> AccountState after confirming the "
            "live tool schema via list_tools()."
        )

    def get_buying_power(self) -> float:
        raise NotImplementedError("Map _TOOL_MAP['buying_power'] -> float after list_tools().")

    def get_positions(self) -> dict[str, Position]:
        raise NotImplementedError("Map _TOOL_MAP['positions'] -> dict[str,Position] after list_tools().")

    def get_prices(self, symbols: list[str]) -> dict[str, float]:
        # Robinhood MCP may expose quotes; otherwise reuse the yfinance adapter.
        from smart_investing.data import load_prices

        px = load_prices(symbols, period="5d")
        return {s: float(px[s].iloc[-1]) for s in px.columns}

    def place_order(self, order: Order) -> OrderResult:
        raise NotImplementedError(
            "Translate Order -> _TOOL_MAP['place_order'] args (symbol/side/quantity/type) and "
            "parse the response -> OrderResult after confirming the schema. NOTE: Robinhood "
            "already enforces its own safety + may require preview/approval."
        )

    def get_order_status(self, order_id: str) -> OrderResult | None:
        raise NotImplementedError("Map _TOOL_MAP['order_status'] -> OrderResult after list_tools().")
