"""Conversational agent: a freeform message -> structured actions + a reply.

The LLM is the interpreter (it maps natural feedback like "make it punchier and
go deeper on the chip suppliers" onto our fixed action vocabulary), but it never
touches money or math — actions are applied deterministically by the caller and
the portfolio is always rebuilt by the real pipeline. A keyword fallback keeps
the conversation working when no LLM is configured or it rate-limits.

Action vocabulary (each action is one object):
  {"op": "set_theme", "value": "<new thesis>"}
  {"op": "add",        "symbols": ["NVDA", "MU"]}
  {"op": "remove",     "symbols": ["KO"]}
  {"op": "expand",     "symbol": "NVDA", "direction": "upstream|downstream|peer|all"}
  {"op": "set_risk",       "value": "low|balanced|high"}
  {"op": "set_breadth",    "value": "focused|balanced|diversified"}
  {"op": "set_supply_chain","value": "yes|no"}
  {"op": "set_lookback",   "value": "1y|2y|3y|5y"}
  {"op": "set_cash",       "value": 25000}
  {"op": "none"}      # a pure question / chit-chat; answer in `reply`, change nothing
"""

from __future__ import annotations

import re

from smart_investing.llm.gemini import GeminiClient

_OPS = {
    "set_theme", "add", "remove", "expand", "set_risk", "set_breadth",
    "set_supply_chain", "set_lookback", "set_cash", "none",
}

_SYSTEM = (
    "You are the brain of a thematic-investing copilot. The user is mid-conversation, "
    "refining a portfolio built from real SEC filings. Map their message onto the FIXED "
    "action vocabulary you are given and write a warm, concrete one-to-two sentence reply. "
    "Never invent tickers that aren't plausible US equities. Prefer editing what exists over "
    "starting over. Output ONLY JSON: {\"actions\": [...], \"reply\": \"...\"}."
)

_TEMPLATE = """CONTEXT
Current thesis: {theme}
Risk: {risk} | Breadth: {breadth} | Supply-chain: {supply_chain} | Look-back: {lookback} | Cash: ${cash:,.0f}
Current holdings: {holdings}
Available supply-chain neighbors to "go deeper" are fetched by the app when you emit an "expand" action.

ACTION VOCABULARY (emit zero or more; omit "rebuild" — the app rebuilds automatically after any change):
  set_theme(value) | add(symbols[]) | remove(symbols[]) | expand(symbol, direction)
  set_risk(low|balanced|high) | set_breadth(focused|balanced|diversified)
  set_supply_chain(yes|no) | set_lookback(1y|2y|3y|5y) | set_cash(number) | none

USER MESSAGE:
\"\"\"{message}\"\"\"

Return ONLY {{"actions": [...], "reply": "..."}}. If it's just a question (e.g. "why NVDA?"),
use actions:[{{"op":"none"}}] and answer it in `reply` using the context."""

_TICKER = re.compile(r"\b[A-Z]{1,5}\b")
_RISK_UP = ("aggressive", "riskier", "more risk", "punchy", "punchier", "growth", "go for it", "yolo", "spicy")
_RISK_DOWN = ("safer", "less risk", "lower risk", "conservative", "defensive", "play it safe", "cautious")
_WIDE = ("diversif", "spread", "more names", "broaden", "wider")
_NARROW = ("focus", "fewer", "concentrat", "tighter", "high conviction", "fewer names")


def interpret(message: str, context: dict, llm: GeminiClient | None) -> dict:
    """Returns {"actions": [ {op,...} ], "reply": str}. LLM-first, keyword fallback."""
    if llm is not None and getattr(llm, "available", False) and message.strip():
        try:
            data = llm.complete_json(_TEMPLATE.format(message=message, **_ctx(context)), system=_SYSTEM)
            actions = [a for a in data.get("actions", []) if isinstance(a, dict) and a.get("op") in _OPS]
            reply = str(data.get("reply") or "").strip()
            if actions or reply:
                return {"actions": actions or [{"op": "none"}], "reply": reply}
        except Exception:
            pass
    return _fallback(message, context)


def _ctx(context: dict) -> dict:
    holdings = context.get("holdings") or []
    hs = ", ".join(f"{h['symbol']} {h.get('weight', 0) * 100:.0f}%" for h in holdings[:12]) or "(none yet)"
    return {
        "theme": context.get("theme") or "(not set)",
        "risk": context.get("risk") or "balanced",
        "breadth": context.get("breadth") or "balanced",
        "supply_chain": context.get("supply_chain") or "yes",
        "lookback": context.get("lookback") or "2y",
        "cash": float(context.get("cash") or 10_000),
        "holdings": hs,
    }


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
