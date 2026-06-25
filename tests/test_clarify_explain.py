"""Tests for the conversational layer: clarify (restate + follow-ups), the
answer->spec mapping, and the grounded plain-English explanation.

All tests force the deterministic, no-network path with GeminiClient(api_key="")
so nothing here touches Gemini or yfinance.
"""

from __future__ import annotations

from smart_investing.data.store import Store
from smart_investing.domain.types import Objective
from smart_investing.llm.clarify import WIRED_QUESTIONS, clarify
from smart_investing.llm.compiler import _fallback_spec
from smart_investing.llm.gemini import GeminiClient
from smart_investing.retrieval.embeddings import TfidfEmbedder
from smart_investing.strategy import _apply_answers, _holdings_for, compile_strategy

ROWS = [
    ("QBIT", "Quantum Computing Inc", "superconducting quantum computing processors and qubit control."),
    ("COOL", "CryoCool Systems Inc", "cryogenic cooling and dilution refrigerators for quantum computing labs."),
    ("NUKE", "Atomic Power Inc", "small modular nuclear reactors and uranium fuel assemblies."),
    ("FOOD", "Tasty Foods Inc", "packaged snacks and beverages for grocery retail."),
]


def _store() -> Store:
    s = Store(":memory:")
    for t, title, text in ROWS:
        s.upsert_company(t, 1, title)
        s.upsert_document(t, 1, f"acc-{t}", "business", text)
    return s


# --------------------------------------------------------------------------- #
# clarify (deterministic fallback path)
# --------------------------------------------------------------------------- #
def test_clarify_returns_interpretation_and_wired_questions():
    out = clarify("Nuclear energy and uranium mining, lower risk, diversified", llm=GeminiClient(api_key=""))
    assert out["interpretation"]
    assert out["focus"]
    # Always returns exactly the four wired follow-ups, untouched.
    assert out["questions"] is WIRED_QUESTIONS
    assert {q["id"] for q in out["questions"]} == {"risk", "breadth", "supply_chain", "lookback"}


def test_clarify_fallback_reflects_risk_and_breadth_keywords():
    out = clarify("Nuclear power, keep it safe and well spread", llm=GeminiClient(api_key=""))
    interp = out["interpretation"]
    assert "lower-risk" in interp  # "safe" -> lower-risk lean
    assert "spread" in interp.lower()  # "spread" -> diversification note


def test_clarify_handles_empty_prompt_gracefully():
    out = clarify("", llm=GeminiClient(api_key=""))
    assert len(out["questions"]) == 4  # still offers the follow-ups


# --------------------------------------------------------------------------- #
# answer -> spec mapping (the user's explicit choices override the LLM parse)
# --------------------------------------------------------------------------- #
def test_apply_answers_low_risk_targets_vol_and_tightens_cap():
    spec = _fallback_spec("nuclear")  # default: MAX_SHARPE, cap 0.30, no target vol
    out = _apply_answers(spec, {"risk": "low"})
    assert out.objective == Objective.TARGET_VOL
    assert out.risk.target_volatility == 0.12
    assert out.risk.concentration_cap <= 0.20


def test_apply_answers_high_risk_loosens_cap_and_drops_target_vol():
    spec = _fallback_spec("nuclear")
    out = _apply_answers(spec, {"risk": "high"})
    assert out.objective == Objective.MAX_SHARPE
    assert out.risk.target_volatility is None
    assert out.risk.concentration_cap >= 0.40


def test_apply_answers_supply_chain_toggle_and_excludes_merge():
    spec = _fallback_spec("nuclear")
    out = _apply_answers(spec, {"supply_chain": "no", "exclude_symbols": ["tsla", "ko"]})
    assert out.include_indirect is False
    assert set(out.exclude_symbols) >= {"TSLA", "KO"}  # merged + upper-cased


def test_apply_answers_none_is_identity():
    spec = _fallback_spec("nuclear")
    assert _apply_answers(spec, None) is spec


def test_holdings_for_breadth_and_low_risk_bump():
    assert _holdings_for({"breadth": "focused"}) == 8
    assert _holdings_for({"breadth": "balanced"}) == 12
    assert _holdings_for({"breadth": "diversified"}) == 18
    # lower risk benefits from a wider spread (+3), still clamped to <= 24
    assert _holdings_for({"breadth": "diversified", "risk": "low"}) == 21
    assert _holdings_for(None) == 12  # sensible default


# --------------------------------------------------------------------------- #
# explanation is populated and grounded in the real pipeline numbers
# --------------------------------------------------------------------------- #
def test_compile_strategy_with_answers_produces_grounded_explanation():
    proposal = compile_strategy(
        "quantum computing",
        _store(),
        live=False,
        llm=GeminiClient(api_key=""),
        embedder=TfidfEmbedder(),
        answers={"risk": "low", "breadth": "diversified", "supply_chain": "no"},
        lookback="1y",
        initial_cash=10_000.0,
    )
    # answers flow through to the spec
    assert proposal.spec.objective == Objective.TARGET_VOL
    assert proposal.spec.include_indirect is False
    assert proposal.lookback == "1y"
    assert proposal.price_source == "synthetic (offline)"

    exp = proposal.explanation
    assert exp is not None
    # every structured section is filled (deterministic, never empty)
    for section in (exp.summary, exp.understood, exp.selection, exp.construction, exp.risk_note, exp.data_note):
        assert section.strip()
    assert len(exp.highlights) == 4
    # data_note must reflect the window we asked for
    assert "1y" in exp.data_note

    # per-holding reasoning is populated and grounded in the real weights
    assert exp.holdings
    held_syms = {s for s, w in proposal.target_weights.items() if w > 0.005}
    assert {h.symbol for h in exp.holdings} == held_syms
    for h in exp.holdings:
        assert h.weight > 0.005
        assert h.why.strip()
        assert h.role in {"direct", "supply-chain"}
    # sorted by weight, descending
    weights = [h.weight for h in exp.holdings]
    assert weights == sorted(weights, reverse=True)
