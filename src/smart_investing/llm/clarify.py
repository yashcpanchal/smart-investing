"""Prompt -> clarifying questions (the conversational follow-up step).

Before we build anything, we restate the thesis in plain English and ask a few
high-value follow-ups. The *wired* questions (risk appetite, breadth, supply
chain, look-back window) map to real spec knobs; the LLM only supplies the
conversational `interpretation` line and may add at most one tailored, free-text
question. Everything degrades to a deterministic default when no LLM is present.

Deliberately NOT asked: "how many stocks?" — the optimizer determines holding
count from the breadth/risk answers. That's our job, not the user's.
"""

from __future__ import annotations

from smart_investing.llm.base import LLMClient
from smart_investing.llm.factory import get_llm

# The fixed, always-wired follow-ups. `kind` drives the UI control; each option's
# `value` is what the API maps back onto the StrategySpec.
WIRED_QUESTIONS: list[dict] = [
    {
        "id": "risk",
        "question": "How much risk are you comfortable with?",
        "help": "Drives the optimization objective and how much any one name can take.",
        "kind": "single",
        "default": "balanced",
        "options": [
            {"label": "Play it safe", "value": "low", "hint": "minimize volatility, tighter caps"},
            {"label": "Balanced", "value": "balanced", "hint": "best risk-adjusted return (Sharpe)"},
            {"label": "Go for growth", "value": "high", "hint": "let winners run, looser caps"},
        ],
    },
    {
        "id": "breadth",
        "question": "Focused or diversified?",
        "help": "We decide the exact number of holdings - this just sets how wide to cast.",
        "kind": "single",
        "default": "balanced",
        "options": [
            {"label": "Focused", "value": "focused", "hint": "fewer, higher-conviction names"},
            {"label": "Balanced", "value": "balanced", "hint": "a sensible spread"},
            {"label": "Diversified", "value": "diversified", "hint": "spread risk across more names"},
        ],
    },
    {
        "id": "supply_chain",
        "question": "Include supply-chain & indirect beneficiaries?",
        "help": "We trace co-mentions in filings to surface non-obvious connected names.",
        "kind": "single",
        "default": "yes",
        "options": [
            {"label": "Yes, find the hidden names", "value": "yes", "hint": "adds 2nd-degree suppliers/customers"},
            {"label": "Pure-play only", "value": "no", "hint": "direct thematic matches only"},
        ],
    },
    {
        "id": "lookback",
        "question": "How far back should we analyze?",
        "help": "The price-history window used to estimate risk, correlations and the backtest. "
        "Longer is steadier; shorter is more responsive to the current regime.",
        "kind": "single",
        "default": "2y",
        "options": [
            {"label": "1 year", "value": "1y", "hint": "recent regime"},
            {"label": "2 years", "value": "2y", "hint": "recommended"},
            {"label": "3 years", "value": "3y", "hint": "steadier"},
            {"label": "5 years", "value": "5y", "hint": "through a full cycle"},
        ],
    },
]

_SYSTEM = (
    "You are a friendly investing co-pilot helping a retail user refine a thesis. "
    "Be warm, concrete and concise. Output ONLY JSON, no prose, no markdown fences."
)

_TEMPLATE = """The user typed this investment thesis:
\"\"\"{prompt}\"\"\"

Return ONLY a JSON object:
- "interpretation": ONE friendly sentence (<=28 words) restating what they seem to
  want, in plain English, as if you're confirming you understood. No tickers invented.
- "focus": a 2-5 word label for the theme (e.g. "Nuclear & uranium").

Example: {{"interpretation": "Got it — you want exposure to nuclear power and the uranium miners that feed it, leaning lower-risk and spread out.", "focus": "Nuclear & uranium"}}"""


def _fallback_interpretation(prompt: str) -> dict:
    """No-LLM read of the thesis. Picks out a short focus phrase and a risk lean
    from keywords so the restatement feels understood, not echoed."""
    p = prompt.strip()
    low = p.lower()
    lean = ""
    if any(w in low for w in ("low risk", "lower risk", "safe", "conservative", "stable")):
        lean = " I'll lean this lower-risk."
    elif any(w in low for w in ("aggressive", "growth", "high risk", "moonshot")):
        lean = " I'll lean this toward growth."
    if "diversif" in low or "spread" in low:
        lean += " Keeping it well spread."
    # focus = first clause, trimmed
    focus = p.split(",")[0].split(".")[0].strip()
    focus = (focus[:46] + "...") if len(focus) > 46 else focus
    return {
        "interpretation": f'Got it - you want exposure to "{focus}".{lean} '
        "Answer a couple of quick questions and I'll build the portfolio.",
        "focus": focus,
    }


def clarify(prompt: str, llm: LLMClient | None = None) -> dict:
    """Returns {"interpretation": str, "focus": str, "questions": [...]}."""
    llm = llm if llm is not None else get_llm()
    head = _fallback_interpretation(prompt)
    if llm and llm.available and prompt.strip():
        try:
            data = llm.complete_json(_TEMPLATE.format(prompt=prompt), system=_SYSTEM)
            interp = str(data.get("interpretation") or "").strip()
            focus = str(data.get("focus") or "").strip()
            if interp:
                head = {"interpretation": interp, "focus": focus or head["focus"]}
        except Exception:
            pass
    return {**head, "questions": WIRED_QUESTIONS}
