"""Company profiles and industry briefings for the detail/analysis panels.

Company facts come from yfinance (cached per ticker); the supply-chain narrative
is built deterministically from the relationship graph; the "current state of the
market" prose is Gemini grounded in live Google Search (with a clean fallback so
the panels still render when offline or rate-limited).
"""

from __future__ import annotations

from smart_investing.llm.base import LLMClient

_INFO_CACHE: dict[str, dict] = {}


# --------------------------------------------------------------------------- #
# Company profile (per-stock detail panel)
# --------------------------------------------------------------------------- #
def _yf_info(symbol: str) -> dict:
    if symbol in _INFO_CACHE:
        return _INFO_CACHE[symbol]
    info: dict = {}
    try:
        import yfinance as yf

        info = yf.Ticker(symbol).info or {}
    except Exception:
        info = {}
    _INFO_CACHE[symbol] = info
    return info


def _money(v) -> str:
    try:
        v = float(v)
    except (TypeError, ValueError):
        return "—"
    for unit, size in (("T", 1e12), ("B", 1e9), ("M", 1e6)):
        if abs(v) >= size:
            return f"${v / size:.1f}{unit}"
    return f"${v:,.0f}"


def _pct(v) -> str:
    try:
        return f"{float(v) * 100:.1f}%"
    except (TypeError, ValueError):
        return "—"


def _num(v) -> str:
    try:
        return f"{float(v):.1f}"
    except (TypeError, ValueError):
        return "—"


def _first_sentences(text: str, n: int = 2) -> str:
    text = (text or "").strip()
    if not text:
        return ""
    out, count = [], 0
    for chunk in text.replace("\n", " ").split(". "):
        out.append(chunk.strip())
        count += 1
        if count >= n:
            break
    s = ". ".join(out).strip()
    return s if s.endswith(".") else s + "."


def company_profile(symbol: str, *, name: str = "", theme: str = "", llm: LLMClient | None = None) -> dict:
    """Key must-know facts for one company, plus an LLM theme-fit + bullets."""
    symbol = symbol.upper()
    info = _yf_info(symbol)
    disp = info.get("shortName") or info.get("longName") or name or symbol
    sector = info.get("sector") or ""
    industry = info.get("industry") or ""
    summary = _first_sentences(info.get("longBusinessSummary", ""), 2)

    metrics = [
        {"label": "Market cap", "value": _money(info.get("marketCap"))},
        {"label": "P/E (ttm)", "value": _num(info.get("trailingPE"))},
        {"label": "Rev. growth", "value": _pct(info.get("revenueGrowth"))},
        {"label": "Profit margin", "value": _pct(info.get("profitMargins"))},
        {"label": "1Y return", "value": _pct(info.get("52WeekChange"))},
        {"label": "Beta", "value": _num(info.get("beta"))},
    ]

    # deterministic fallbacks
    theme_fit = (
        f"{disp} operates in {industry or sector or 'its sector'}"
        + (f", relevant to {theme}." if theme else ".")
    )
    must_knows = [b for b in [
        f"{sector}{' · ' + industry if industry else ''}" if sector or industry else "",
        f"Market cap {_money(info.get('marketCap'))}" if info.get("marketCap") else "",
        f"Revenue growing {_pct(info.get('revenueGrowth'))} YoY" if info.get("revenueGrowth") is not None else "",
    ] if b]

    if llm is not None and getattr(llm, "available", False):
        facts = {"name": disp, "sector": sector, "industry": industry, "summary": summary, "theme": theme}
        prompt = (
            "Given these facts about a public company, return JSON with:\n"
            f'  "theme_fit": one sentence on how it fits the theme "{theme}" (where it sits in the chain),\n'
            '  "must_knows": 3 short must-know bullets for an investor (business model, moat/position, '
            "key risk or catalyst). Concrete, no hype, no numbers you weren't given.\n"
            f"FACTS: {facts}"
        )
        try:
            data = llm.complete_json(prompt)
            tf = str(data.get("theme_fit") or "").strip()
            mk = [str(b).strip() for b in (data.get("must_knows") or []) if str(b).strip()]
            if tf:
                theme_fit = tf
            if mk:
                must_knows = mk[:4]
        except Exception:
            pass

    return {
        "symbol": symbol,
        "name": disp,
        "sector": sector,
        "industry": industry,
        "summary": summary,
        "metrics": metrics,
        "theme_fit": theme_fit,
        "must_knows": must_knows,
    }


# --------------------------------------------------------------------------- #
# Industry briefing (industry-analysis panel)
# --------------------------------------------------------------------------- #
def _supply_chain_layers(graph, theme: str) -> list[dict]:
    """Deterministic upstream -> core -> downstream view from the graph.
    This is the 'TSMC -> NVIDIA -> hyperscalers' structure, built from real edges."""
    seeds = graph.search(theme, top_k=5)
    core = [s["symbol"] for s in seeds]
    up: list[str] = []
    down: list[str] = []
    for sym in core:
        for n in graph.neighbors(sym, theme=theme, limit=8):
            if n["direction"] == "upstream" and n["symbol"] not in core:
                up.append(n["symbol"])
            elif n["direction"] == "downstream" and n["symbol"] not in core:
                down.append(n["symbol"])

    def _dedupe(seq: list[str], cap: int = 6) -> list[str]:
        seen, out = set(), []
        for s in seq:
            if s not in seen:
                seen.add(s)
                out.append(s)
            if len(out) >= cap:
                break
        return out

    names = lambda syms: [{"symbol": s, "name": graph.titles.get(s, s)} for s in syms]  # noqa: E731
    layers = []
    if up:
        layers.append({"layer": "Upstream — suppliers & enablers", "players": names(_dedupe(up))})
    layers.append({"layer": "Core — direct plays", "players": names(_dedupe(core))})
    if down:
        layers.append({"layer": "Downstream — customers & demand", "players": names(_dedupe(down))})
    return layers


_IND_SYSTEM = (
    "You are an equity research analyst writing a concise, balanced briefing for a "
    "retail investor. Be specific and current, cite what's actually happening. No hype."
)


def industry_brief(theme: str, graph, *, llm: LLMClient | None = None) -> dict:
    """Supply-chain structure (deterministic) + current-state analysis (grounded LLM)."""
    layers = _supply_chain_layers(graph, theme)
    chain_str = "  ->  ".join(
        f"{lyr['layer'].split(' — ')[0]}: " + ", ".join(p["symbol"] for p in lyr["players"]) for lyr in layers
    )

    analysis = (
        f"This briefing maps the {theme} supply chain from the companies in our corpus. "
        "Live market commentary is unavailable right now (set a Gemini key / quota) — "
        "the structure below is built from real relationships in SEC filings."
    )
    tailwinds: list[str] = []
    risks: list[str] = []
    outlook = ""
    sources: list[dict] = []
    grounded = False

    if llm is not None and getattr(llm, "available", False):
        prompt = (
            f"Theme: {theme}. The key public companies by supply-chain layer are:\n{chain_str}\n\n"
            "Using current information, write a briefing. Return ONLY JSON:\n"
            '  "state": 2-3 sentences on what is happening in this industry right now,\n'
            '  "tailwinds": 2-3 short bullets (demand drivers / catalysts),\n'
            '  "risks": 2-3 short bullets (what could go wrong),\n'
            '  "outlook": one forward-looking sentence.'
        )
        try:
            text, sources = llm.complete_grounded(prompt, system=_IND_SYSTEM)
            from smart_investing.llm.gemini import _loads_lenient

            data = _loads_lenient(text)
            analysis = str(data.get("state") or analysis).strip()
            tailwinds = [str(b).strip() for b in (data.get("tailwinds") or []) if str(b).strip()][:3]
            risks = [str(b).strip() for b in (data.get("risks") or []) if str(b).strip()][:3]
            outlook = str(data.get("outlook") or "").strip()
            grounded = True
        except Exception:
            grounded = False

    return {
        "theme": theme,
        "supply_chain": layers,
        "analysis": analysis,
        "tailwinds": tailwinds,
        "risks": risks,
        "outlook": outlook,
        "sources": sources,
        "grounded": grounded,
    }
