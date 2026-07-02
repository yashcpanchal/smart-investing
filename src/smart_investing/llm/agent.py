"""Conversational agent: a real tool-use loop, not a one-shot intent mapper.

The LLM drives a bounded loop over two kinds of tools:

- READ tools (get_portfolio, get_stock_facts, get_neighbors, search_companies)
  execute live inside the loop, so the agent can actually research before it
  answers "why MU?" instead of guessing from a one-line context string.
- MUTATION tools (set_theme, add/remove symbols, expand, knobs, cash) are
  QUEUED as actions and returned to the caller — the LLM never touches money
  or math. Actions are applied deterministically and the portfolio is always
  rebuilt by the real pipeline, exactly as before.

Conversation history is passed through, so multi-turn context finally works.
A keyword fallback keeps the conversation alive when no LLM is configured or
it rate-limits.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable

from smart_investing.llm.base import ChatMessage, LLMClient, ToolResult, ToolSpec

MAX_ROUNDS = 4  # LLM turns per user message (tool rounds + the final reply)
_HISTORY_TURNS = 10  # prior messages shown to the model
_READ_RESULT_CAP = 2400  # chars of tool output fed back per call

_SYSTEM = (
    "You are the brain of a thematic-investing copilot. The user is refining a US-equity "
    "portfolio built from real SEC filings, a supply-chain graph, and a mean-variance "
    "optimizer. You have READ tools to research (portfolio details, stock facts, "
    "supply-chain neighbors, company search) and EDIT tools to change the strategy.\n"
    "Rules:\n"
    "- EDIT tools queue changes; the deterministic pipeline applies them and rebuilds the "
    "portfolio after you reply. You never compute weights, prices, or orders yourself.\n"
    "- Before answering questions about holdings ('why NVDA?', 'what does MU do?'), use the "
    "READ tools rather than guessing.\n"
    "- Never invent tickers that aren't plausible US equities. Prefer editing what exists "
    "over starting over.\n"
    "- Finish with a warm, concrete reply of one to three sentences. Plain text only."
)

# ------------------------------------------------------------- tool schemas
_SYMS = {"type": "array", "items": {"type": "string"}, "description": "Ticker symbols, e.g. ['NVDA','MU']"}

MUTATION_TOOLS: dict[str, ToolSpec] = {
    "set_theme": ToolSpec(
        "set_theme",
        "Replace the investment thesis with a new one. Only when the user pivots the whole idea.",
        {"type": "object", "properties": {"value": {"type": "string", "description": "The new thesis"}},
         "required": ["value"]},
    ),
    "add_symbols": ToolSpec(
        "add_symbols",
        "Pin specific tickers into the portfolio universe.",
        {"type": "object", "properties": {"symbols": _SYMS}, "required": ["symbols"]},
    ),
    "remove_symbols": ToolSpec(
        "remove_symbols",
        "Exclude tickers from the portfolio (user said drop/sell/without).",
        {"type": "object", "properties": {"symbols": _SYMS}, "required": ["symbols"]},
    ),
    "expand": ToolSpec(
        "expand",
        "Go deeper around one holding: pin its supply-chain neighbors (suppliers=upstream, "
        "customers=downstream, competitors=peer).",
        {"type": "object", "properties": {
            "symbol": {"type": "string"},
            "direction": {"type": "string", "enum": ["upstream", "downstream", "peer", "all"]},
        }, "required": ["symbol"]},
    ),
    "set_risk": ToolSpec(
        "set_risk", "Set the risk appetite.",
        {"type": "object", "properties": {"value": {"type": "string", "enum": ["low", "balanced", "high"]}},
         "required": ["value"]},
    ),
    "set_breadth": ToolSpec(
        "set_breadth", "Set concentration: focused (fewer names) vs diversified (more names).",
        {"type": "object", "properties": {"value": {"type": "string", "enum": ["focused", "balanced", "diversified"]}},
         "required": ["value"]},
    ),
    "set_supply_chain": ToolSpec(
        "set_supply_chain", "Include indirect supply-chain beneficiaries (yes) or direct plays only (no).",
        {"type": "object", "properties": {"value": {"type": "string", "enum": ["yes", "no"]}},
         "required": ["value"]},
    ),
    "set_lookback": ToolSpec(
        "set_lookback", "Set the price-history window used for risk and backtest.",
        {"type": "object", "properties": {"value": {"type": "string", "enum": ["1y", "2y", "3y", "5y"]}},
         "required": ["value"]},
    ),
    "set_cash": ToolSpec(
        "set_cash", "Set the cash amount to invest, in dollars.",
        {"type": "object", "properties": {"value": {"type": "number", "description": "Dollars, e.g. 25000"}},
         "required": ["value"]},
    ),
    "set_source_weights": ToolSpec(
        "set_source_weights",
        "Set how much the 'smart money' evidence signals influence universe ranking: "
        "sec_13f = institutional 13F accumulation, insider = recent insider open-market buying. "
        "Each 0..1 (0 = ignore the signal).",
        {"type": "object", "properties": {
            "sec_13f": {"type": "number", "description": "Weight of institutional 13F holdings, 0..1"},
            "insider": {"type": "number", "description": "Weight of insider buying, 0..1"},
        }},
    ),
}

READ_TOOLS: dict[str, ToolSpec] = {
    "get_portfolio": ToolSpec(
        "get_portfolio",
        "Current portfolio: holdings with weights, expected return/volatility/Sharpe, and "
        "per-holding reasoning. Use before answering questions about the portfolio.",
        {"type": "object", "properties": {}},
    ),
    "get_stock_facts": ToolSpec(
        "get_stock_facts",
        "Company facts for one ticker: name, sector, business summary, market cap, margins.",
        {"type": "object", "properties": {"symbol": {"type": "string"}}, "required": ["symbol"]},
    ),
    "get_neighbors": ToolSpec(
        "get_neighbors",
        "Supply-chain graph neighbors of a ticker: upstream suppliers, downstream customers, peers.",
        {"type": "object", "properties": {
            "symbol": {"type": "string"},
            "direction": {"type": "string", "enum": ["upstream", "downstream", "peer", "all"]},
        }, "required": ["symbol"]},
    ),
    "search_companies": ToolSpec(
        "search_companies",
        "Search the SEC-filing corpus for companies matching a theme or description.",
        {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
    ),
}

# tool name -> the action op vocabulary the API applies deterministically
_TOOL_TO_OP = {
    "set_theme": "set_theme", "add_symbols": "add", "remove_symbols": "remove",
    "expand": "expand", "set_risk": "set_risk", "set_breadth": "set_breadth",
    "set_supply_chain": "set_supply_chain", "set_lookback": "set_lookback", "set_cash": "set_cash",
    "set_source_weights": "set_source_weights",
}


def run_agent(
    message: str,
    context: dict,
    llm: LLMClient | None,
    *,
    history: list[dict] | None = None,
    read_tools: dict[str, Callable[..., dict]] | None = None,
    max_rounds: int = MAX_ROUNDS,
) -> dict:
    """One conversational turn. Returns {"actions": [...], "reply": str, "researched": [...]}.

    `history` is prior turns as [{role: "user"|"assistant", text: str}] (without the
    current message). `read_tools` maps READ tool names to callables taking the tool
    args as kwargs and returning a JSON-safe dict. LLM-first; keyword fallback."""
    if llm is None or not getattr(llm, "available", False) or not message.strip():
        return {**_fallback(message, context), "researched": []}

    tools = list(MUTATION_TOOLS.values()) + [READ_TOOLS[n] for n in (read_tools or {})]
    messages = [
        ChatMessage(role=m["role"], content=m["text"])
        for m in (history or [])[-_HISTORY_TURNS:]
        if m.get("role") in ("user", "assistant") and m.get("text")
    ]
    messages.append(ChatMessage(role="user", content=message))

    actions: list[dict] = []
    researched: list[str] = []
    try:
        for round_no in range(max_rounds):
            # Last round: no tools, so the model must produce the reply.
            offer = tools if round_no < max_rounds - 1 else None
            resp = llm.chat(messages, system=_system_prompt(context), tools=offer)
            if not resp.tool_calls:
                reply = resp.text.strip()
                if reply or actions:
                    return {"actions": actions or [{"op": "none"}], "reply": reply, "researched": researched}
                break  # empty turn -> fallback
            messages.append(ChatMessage(role="assistant", content=resp.text, tool_calls=resp.tool_calls))
            results = []
            for call in resp.tool_calls:
                if call.name in _TOOL_TO_OP:
                    actions.append(_to_action(call.name, call.arguments))
                    out: dict = {"status": "queued", "note": "applied after your reply; portfolio rebuilds automatically"}
                elif read_tools and call.name in read_tools:
                    researched.append(call.name)
                    out = _run_read_tool(read_tools[call.name], call.arguments)
                else:
                    out = {"error": f"unknown tool {call.name}"}
                results.append(ToolResult(call=call, content=out))
            messages.append(ChatMessage(role="tool", tool_results=results))
    except Exception:
        pass  # any provider hiccup -> deterministic path below
    if actions:  # model made edits but never phrased a reply
        return {"actions": actions, "reply": "", "researched": researched}
    return {**_fallback(message, context), "researched": researched}


def interpret(message: str, context: dict, llm: LLMClient | None) -> dict:
    """Back-compat single-shot entry point (no history / read tools)."""
    return run_agent(message, context, llm)


def _system_prompt(context: dict) -> str:
    holdings = context.get("holdings") or []
    hs = ", ".join(f"{h['symbol']} {h.get('weight', 0) * 100:.0f}%" for h in holdings[:12]) or "(none yet)"
    return _SYSTEM + (
        f"\n\nCURRENT STATE\nThesis: {context.get('theme') or '(not set)'}\n"
        f"Risk: {context.get('risk') or 'balanced'} | Breadth: {context.get('breadth') or 'balanced'} | "
        f"Supply-chain: {context.get('supply_chain') or 'yes'} | Look-back: {context.get('lookback') or '2y'} | "
        f"Cash: ${float(context.get('cash') or 10_000):,.0f}\nHoldings: {hs}"
    )


def _to_action(tool: str, args: dict) -> dict:
    op = _TOOL_TO_OP[tool]
    if op in ("add", "remove"):
        return {"op": op, "symbols": [str(s).upper() for s in (args.get("symbols") or [])]}
    if op == "expand":
        return {"op": op, "symbol": str(args.get("symbol", "")).upper(), "direction": args.get("direction") or "all"}
    if op == "set_source_weights":  # the weights ARE the args (no "value" wrapper)
        return {"op": op, "value": {k: args[k] for k in ("sec_13f", "insider") if args.get(k) is not None}}
    return {"op": op, "value": args.get("value")}


def _run_read_tool(fn: Callable[..., dict], args: dict) -> dict:
    try:
        out = fn(**{k: v for k, v in args.items() if isinstance(k, str)})
    except Exception as e:  # tool errors go back to the model, not up the stack
        return {"error": str(e)[:200]}
    # Cap what flows back into the context window.
    text = json.dumps(out, default=str)
    if len(text) > _READ_RESULT_CAP:
        return {"truncated": True, "data": text[:_READ_RESULT_CAP]}
    return out


# ------------------------------------------------------------- fallback path
_TICKER = re.compile(r"\b[A-Z]{1,5}\b")
_RISK_UP = ("aggressive", "riskier", "more risk", "punchy", "punchier", "growth", "go for it", "yolo", "spicy")
_RISK_DOWN = ("safer", "less risk", "lower risk", "conservative", "defensive", "play it safe", "cautious")
_WIDE = ("diversif", "spread", "more names", "broaden", "wider")
_NARROW = ("focus", "fewer", "concentrat", "tighter", "high conviction", "fewer names")


def _fallback(message: str, context: dict) -> dict:
    """Deterministic keyword interpreter for the common edits (no-LLM path)."""
    low = message.lower()
    known = {h["symbol"] for h in (context.get("holdings") or [])}
    actions: list[dict] = []

    # remove / drop
    m = re.search(r"\b(drop|remove|without|exclude|sell|get rid of)\b(.+)", low)
    if m:
        syms = [t for t in _TICKER.findall(message[m.start(2):]) if t not in {"I", "A"}]
        if syms:
            actions.append({"op": "remove", "symbols": syms})

    # go deeper / suppliers / upstream
    md = re.search(r"\b(deeper|suppliers?|upstream|behind|who makes|further back)\b", low)
    if md:
        cand = [t for t in _TICKER.findall(message) if t in known] or list(known)[:1]
        if cand:
            direction = "upstream" if re.search(r"upstream|suppliers?|behind|back|who makes", low) else "all"
            actions.append({"op": "expand", "symbol": cand[0], "direction": direction})
    elif re.search(r"\b(add|include|buy|pin)\b", low):
        syms = [t for t in _TICKER.findall(message) if t not in known and t not in {"I", "A"}]
        if syms:
            actions.append({"op": "add", "symbols": syms})

    if any(k in low for k in _RISK_UP):
        actions.append({"op": "set_risk", "value": "high"})
    elif any(k in low for k in _RISK_DOWN):
        actions.append({"op": "set_risk", "value": "low"})
    if any(k in low for k in _WIDE):
        actions.append({"op": "set_breadth", "value": "diversified"})
    elif any(k in low for k in _NARROW):
        actions.append({"op": "set_breadth", "value": "focused"})

    mc = re.search(r"\$?\s*([0-9][0-9,\.]{2,})\s*(k|dollars|cash)?", low)
    if mc and re.search(r"\b(cash|invest|budget|put in|start with)\b", low):
        raw = mc.group(1).replace(",", "")
        try:
            val = float(raw) * (1000 if mc.group(2) == "k" else 1)
            if val >= 100:
                actions.append({"op": "set_cash", "value": val})
        except ValueError:
            pass

    for win in ("1y", "2y", "3y", "5y"):
        if win in low or win.replace("y", " year") in low:
            actions.append({"op": "set_lookback", "value": win})
            break

    if not actions:
        actions = [{"op": "none"}]
    return {"actions": actions, "reply": ""}
