from __future__ import annotations

from smart_investing.domain.types import Objective
from smart_investing.llm.compiler import _fallback_spec, _spec_from_dict, compile_spec
from smart_investing.llm.gemini import GeminiClient


def test_fallback_low_risk_sets_target_vol():
    spec = _fallback_spec("I want something safe and low risk in utilities")
    assert spec.objective == Objective.TARGET_VOL
    assert spec.risk.target_volatility == 0.15


def test_fallback_pure_play_disables_indirect():
    spec = _fallback_spec("pure-play quantum computing only")
    assert spec.include_indirect is False


def test_spec_from_dict_maps_and_clamps():
    spec = _spec_from_dict(
        "x",
        {
            "themes": ["nuclear", "uranium", "a", "b", "c", "d", "e"],  # >5 -> trimmed
            "include_symbols": ["ccj", "leu"],
            "exclude_symbols": ["tsla"],
            "include_indirect": True,
            "objective": "min_vol",
            "concentration_cap": 5.0,  # out of range -> clamped to 1.0
            "target_volatility": 0.2,
        },
    )
    assert spec.objective == Objective.MIN_VOL
    assert len(spec.themes) == 5
    assert spec.include_symbols == ["CCJ", "LEU"]
    assert spec.exclude_symbols == ["TSLA"]
    assert spec.risk.concentration_cap == 1.0
    assert spec.risk.target_volatility == 0.2


def test_spec_from_dict_bad_objective_defaults():
    spec = _spec_from_dict("x", {"objective": "nonsense"})
    assert spec.objective == Objective.MAX_SHARPE


def test_compile_spec_without_llm_uses_fallback():
    # An unavailable client forces the deterministic path — no network call.
    spec = compile_spec("nuclear and quantum, conservative", llm=GeminiClient(api_key=""))
    assert spec.themes
    assert spec.objective == Objective.TARGET_VOL  # "conservative" -> low-risk fallback
