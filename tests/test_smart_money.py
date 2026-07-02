"""Smart-money stream: 13F/Form 4 parsing, issuer-name matching, deterministic
scoring, universe blending via source_weights, session/agent plumbing, the
/api/session/{id}/knobs endpoint, and explanation surfacing. Fully offline —
inline XML/JSON fixtures, Store(':memory:'), TF-IDF embedder."""

from __future__ import annotations

import pytest

from smart_investing.data.smart_money import (
    build_name_key_map,
    compute_smart_money_scores,
    infotable_candidates,
    match_issuer,
    parse_13f_infotable,
    parse_form4,
)
from smart_investing.data.store import Store
from smart_investing.domain.types import SourceWeights, StrategySpec
from smart_investing.llm.agent import MUTATION_TOOLS, _to_action
from smart_investing.retrieval.embeddings import TfidfEmbedder
from smart_investing.retrieval.universe import build_universe
from smart_investing.session import Session

# --------------------------------------------------------------- fixtures
FORM4_NAMESPACED = """<?xml version="1.0"?>
<ownershipDocument xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
  <schemaVersion>X0508</schemaVersion>
  <issuer>
    <issuerCik>0000320193</issuerCik>
    <issuerName>Apple Inc.</issuerName>
    <issuerTradingSymbol>aapl</issuerTradingSymbol>
  </issuer>
  <reportingOwner>
    <reportingOwnerId><rptOwnerName>Doe Jane</rptOwnerName></reportingOwnerId>
    <reportingOwnerRelationship>
      <isDirector>0</isDirector>
      <isOfficer>1</isOfficer>
      <isTenPercentOwner>false</isTenPercentOwner>
    </reportingOwnerRelationship>
  </reportingOwner>
  <nonDerivativeTable>
    <nonDerivativeTransaction>
      <transactionDate><value>2026-05-01</value></transactionDate>
      <transactionCoding><transactionCode>P</transactionCode></transactionCoding>
      <transactionAmounts>
        <transactionShares><value>1000</value></transactionShares>
        <transactionPricePerShare><value>150.25</value></transactionPricePerShare>
        <transactionAcquiredDisposedCode><value>A</value></transactionAcquiredDisposedCode>
      </transactionAmounts>
    </nonDerivativeTransaction>
    <nonDerivativeTransaction>
      <transactionDate><value>2026-05-02</value></transactionDate>
      <transactionCoding><transactionCode>S</transactionCode></transactionCoding>
      <transactionAmounts>
        <transactionShares><value>200</value></transactionShares>
        <transactionPricePerShare><value>152</value></transactionPricePerShare>
        <transactionAcquiredDisposedCode><value>D</value></transactionAcquiredDisposedCode>
      </transactionAmounts>
    </nonDerivativeTransaction>
  </nonDerivativeTable>
</ownershipDocument>
"""

INFOTABLE_NAMESPACED = """<?xml version="1.0"?>
<ns1:informationTable xmlns:ns1="http://www.sec.gov/edgar/document/thirteenf/informationtable">
  <ns1:infoTable>
    <ns1:nameOfIssuer>NVIDIA CORP</ns1:nameOfIssuer>
    <ns1:titleOfClass>COM</ns1:titleOfClass>
    <ns1:cusip>67066G104</ns1:cusip>
    <ns1:value>5000000000</ns1:value>
    <ns1:shrsOrPrnAmt><ns1:sshPrnamt>40000000</ns1:sshPrnamt><ns1:sshPrnamtType>SH</ns1:sshPrnamtType></ns1:shrsOrPrnAmt>
  </ns1:infoTable>
  <ns1:infoTable>
    <ns1:nameOfIssuer>OBSCURE UNMATCHED HOLDINGS LLC</ns1:nameOfIssuer>
    <ns1:titleOfClass>COM</ns1:titleOfClass>
    <ns1:cusip>000000000</ns1:cusip>
    <ns1:value>12345</ns1:value>
    <ns1:shrsOrPrnAmt><ns1:sshPrnamt>10</ns1:sshPrnamt><ns1:sshPrnamtType>SH</ns1:sshPrnamtType></ns1:shrsOrPrnAmt>
  </ns1:infoTable>
</ns1:informationTable>
"""

INFOTABLE_PLAIN = """<informationTable>
  <infoTable>
    <nameOfIssuer>Micron Technology Inc</nameOfIssuer>
    <cusip>595112103</cusip>
    <value>250000</value>
    <shrsOrPrnAmt><sshPrnamt>3000</sshPrnamt></shrsOrPrnAmt>
  </infoTable>
</informationTable>
"""

ROWS = [
    ("NVDA", 1045810, "NVIDIA Corp", "gpus and accelerators for ai data centers and gaming."),
    ("AMD", 2488, "Advanced Micro Devices Inc", "cpus and gpus for data centers, competes with nvidia."),
    ("MU", 723125, "Micron Technology Inc", "memory and storage chips for data centers and gpus."),
    ("MSFT", 789019, "Microsoft Corp", "cloud azure hyperscale data centers and ai services."),
    ("KO", 21344, "Coca Cola Co", "beverages and snacks for retail."),
]


def _store() -> Store:
    s = Store(":memory:")
    for t, cik, title, text in ROWS:
        s.upsert_company(t, cik, title)
        s.upsert_document(t, cik, f"acc-{t}", "business", text)
    return s


# ------------------------------------------------------------ XML parsing
def test_parse_form4_namespaced():
    d = parse_form4(FORM4_NAMESPACED)
    assert d["symbol"] == "AAPL"
    assert d["owner_name"] == "Doe Jane"
    assert d["is_officer"] is True
    assert d["is_director"] is False
    assert d["is_ten_pct_owner"] is False
    assert len(d["transactions"]) == 2
    buy, sell = d["transactions"]
    assert buy == {"date": "2026-05-01", "code": "P", "shares": 1000.0, "price": 150.25, "acquired_disposed": "A"}
    assert sell["code"] == "S" and sell["acquired_disposed"] == "D"


def test_parse_form4_namespace_stripped_variant():
    plain = FORM4_NAMESPACED.replace(' xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"', "")
    assert parse_form4(plain) == parse_form4(FORM4_NAMESPACED)


def test_parse_13f_infotable_namespaced_and_plain():
    rows = parse_13f_infotable(INFOTABLE_NAMESPACED)
    assert len(rows) == 2
    assert rows[0] == {
        "issuer_name": "NVIDIA CORP", "cusip": "67066G104",
        "value_usd": 5_000_000_000.0, "shares": 40_000_000.0,
    }
    plain = parse_13f_infotable(INFOTABLE_PLAIN)
    assert plain == [
        {"issuer_name": "Micron Technology Inc", "cusip": "595112103", "value_usd": 250_000.0, "shares": 3000.0}
    ]


def test_infotable_candidates_prefers_named_then_largest():
    index = {"directory": {"item": [
        {"name": "primary_doc.xml", "size": "3000"},
        {"name": "big_holdings.xml", "size": "900000"},
        {"name": "form13fInfoTable.xml", "size": "100"},
        {"name": "cover.htm", "size": "50"},
    ]}}
    assert infotable_candidates(index) == ["form13fInfoTable.xml", "big_holdings.xml"]


# ---------------------------------------------------------- name matching
def test_issuer_name_matching_maps_to_corpus_tickers():
    s = _store()
    keymap = build_name_key_map(s.companies())
    assert match_issuer("NVIDIA CORP", keymap) == "NVDA"
    assert match_issuer("MICRON TECHNOLOGY INC", keymap) == "MU"
    assert match_issuer("ADVANCED MICRO DEVICES", keymap) == "AMD"
    assert match_issuer("OBSCURE UNMATCHED HOLDINGS LLC", keymap) is None


# --------------------------------------------------------------- scoring
def _seed_insiders(s: Store) -> None:
    # NVDA: big net buying; MU: net selling; AMD: stale buy outside the window
    s.upsert_insider_trade("NVDA", 1, "a1", "Buyer Bob", True, False, False, "2026-06-01", "P", 10_000, 100.0, "A")
    s.upsert_insider_trade("MU", 2, "a2", "Seller Sue", True, False, False, "2026-05-20", "S", 5_000, 80.0, "D")
    s.upsert_insider_trade("AMD", 3, "a3", "Old Olly", False, True, False, "2025-01-01", "P", 99_999, 50.0, "A")


def test_scores_empty_store_is_empty_dict():
    assert compute_smart_money_scores(_store()) == {}


def test_scores_deterministic_and_window_from_store_not_now():
    s = _store()
    _seed_insiders(s)
    s.upsert_inst_holding(1067983, "Berkshire Hathaway", "NVDA", "67066G104", "NVIDIA CORP", 5e9, 4e7, "2026-03-31", "x1")
    s.upsert_inst_holding(1037389, "Renaissance", "MU", "595112103", "MICRON", 2e8, 3e6, "2026-03-31", "x2")
    a = compute_smart_money_scores(s)
    b = compute_smart_money_scores(s)
    assert a == b  # same input -> same scores
    # 13F: both held; NVDA has the larger aggregate value
    assert a["NVDA"]["smart_money_13f"] == 1.0
    assert a["MU"]["smart_money_13f"] == 0.0
    # insider: window anchored at max date in store (2026-06-01), so AMD's
    # 2025 buy is out of window entirely; MU is a net seller -> 0.0
    assert a["NVDA"]["smart_money_insider"] == 1.0
    assert a["MU"]["smart_money_insider"] == 0.0
    assert "smart_money_insider" not in a.get("AMD", {})


def test_scores_all_in_0_1():
    s = _store()
    _seed_insiders(s)
    for sub in compute_smart_money_scores(s).values():
        for v in sub.values():
            assert 0.0 <= v <= 1.0


# ------------------------------------------------------- universe blending
def _universe(s, sw: SourceWeights, smart_money=None):
    spec = StrategySpec(raw_prompt="ai data center gpus", themes=["ai data center gpus"], source_weights=sw)
    return build_universe(
        "ai data center gpus", s, spec, top_k=10, embedder=TfidfEmbedder(), smart_money=smart_money
    )


def test_universe_zero_weights_ranking_identical_to_baseline():
    s = _store()
    sm = {"MU": {"smart_money_13f": 1.0, "smart_money_insider": 1.0}}
    zero = SourceWeights(sec_13f=0.0, insider=0.0)
    base = _universe(s, zero, smart_money=None)
    blended = _universe(s, zero, smart_money=sm)
    assert [a.symbol for a in base.assets] == [a.symbol for a in blended.assets]
    # sub-scores present on every asset, defaulting to 0.0
    for a in blended.assets:
        assert "smart_money_13f" in a.scores and "smart_money_insider" in a.scores
    by = {a.symbol: a for a in blended.assets}
    assert by["MU"].scores["smart_money_insider"] == 1.0
    assert by["NVDA"].scores["smart_money_insider"] == 0.0


def test_universe_insider_weight_reorders_but_never_readmits_gated_names():
    s = _store()
    zero = SourceWeights(sec_13f=0.0, insider=0.0)
    base = _universe(s, zero)
    base_syms = {a.symbol for a in base.assets}
    assert "KO" not in base_syms  # off-theme, gated by relevance
    direct = [a.symbol for a in base.assets if a.degree == 1]
    assert len(direct) >= 2
    trailing = direct[-1]  # boost the weakest direct hit's insider score
    sm = {trailing: {"smart_money_insider": 1.0}, "KO": {"smart_money_insider": 1.0, "smart_money_13f": 1.0}}
    heavy = _universe(s, SourceWeights(sec_13f=0.0, insider=1.0), smart_money=sm)
    heavy_direct = [a.symbol for a in heavy.assets if a.degree == 1]
    assert heavy_direct[0] == trailing  # reordered to the top
    assert {a.symbol for a in heavy.assets} == base_syms  # same admitted set — KO still out


# --------------------------------------------------- session + agent plumbing
def test_session_snapshot_and_compile_answers_include_source_weights():
    sess = Session(id="x")
    snap = sess.snapshot()
    assert snap["source_weights"] == {"sec_13f": 0.5, "insider": 0.2, "news_sentiment": 0.2, "social": 0.1}
    assert sess.compile_answers()["source_weights"]["sec_13f"] == 0.5


def test_agent_set_source_weights_tool_maps_to_action():
    assert "set_source_weights" in MUTATION_TOOLS
    a = _to_action("set_source_weights", {"sec_13f": 0.8, "insider": 0.3})
    assert a == {"op": "set_source_weights", "value": {"sec_13f": 0.8, "insider": 0.3}}
    partial = _to_action("set_source_weights", {"insider": 0.9})
    assert partial == {"op": "set_source_weights", "value": {"insider": 0.9}}


# --------------------------------------------------------------- explanation
def test_explanation_surfaces_smart_money_clause_and_score():
    from smart_investing.domain.types import AssetUniverse, UniverseAsset
    from smart_investing.llm.explain import _build_holdings

    uni = AssetUniverse(theme="t", assets=[
        UniverseAsset(symbol="NVDA", name="NVIDIA Corp", degree=1,
                      scores={"relevance": 0.9, "smart_money_13f": 0.9, "smart_money_insider": 0.7}),
        UniverseAsset(symbol="AMD", name="AMD Inc", degree=1,
                      scores={"relevance": 0.8, "smart_money_13f": 0.2, "smart_money_insider": 0.0}),
    ])
    rows = {r.symbol: r for r in _build_holdings(uni, {"NVDA": 0.6, "AMD": 0.4})}
    assert rows["NVDA"].smart_money == 0.9
    assert "accumulating" in rows["NVDA"].why and "insiders have been buying" in rows["NVDA"].why
    assert rows["AMD"].smart_money is None
    assert "Smart money" not in rows["AMD"].why


# ------------------------------------------------------------ knobs endpoint
def _client():
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from smart_investing.api.app import create_app
    from smart_investing.broker.paper import PaperBroker
    from smart_investing.llm.gemini import GeminiClient
    from smart_investing.persistence.repo import StateRepo

    store = _store()
    _seed_insiders(store)
    app = create_app(
        store=store,
        repo=StateRepo(":memory:"),
        broker=PaperBroker(cash=10_000.0),
        llm=GeminiClient(api_key=""),
        embedder=TfidfEmbedder(),
    )
    return TestClient(app)


def test_knobs_endpoint_round_trip():
    c = _client()
    sid = c.post("/api/chat", json={"message": "ai data center gpus", "live": False}).json()["session_id"]

    r = c.post(f"/api/session/{sid}/knobs", json={"source_weights": {"sec_13f": 0.9, "insider": 0.65}, "live": False})
    assert r.status_code == 200
    b = r.json()
    assert b["session_id"] == sid
    assert b["rebuilt"] is True
    assert b["proposal"] is not None
    assert b["state"]["source_weights"]["sec_13f"] == 0.9
    assert b["state"]["source_weights"]["insider"] == 0.65
    # the rebuilt proposal's spec carries the weights (session -> answers -> spec)
    assert b["proposal"]["spec"]["source_weights"]["sec_13f"] == 0.9
    # chat_state from a later chat turn echoes them too
    r2 = c.post("/api/chat", json={"message": "status", "session_id": sid, "live": False}).json()
    assert r2["state"]["source_weights"]["insider"] == 0.65


def test_knobs_endpoint_clamps_and_404s():
    c = _client()
    assert c.post("/api/session/nope/knobs", json={"source_weights": {"sec_13f": 1.0}}).status_code == 404
    sid = c.post("/api/chat", json={"message": "ai data center gpus", "live": False}).json()["session_id"]
    b = c.post(f"/api/session/{sid}/knobs", json={"source_weights": {"sec_13f": 7.0, "insider": -3.0}, "live": False}).json()
    assert b["state"]["source_weights"]["sec_13f"] == 1.0
    assert b["state"]["source_weights"]["insider"] == 0.0


def test_knobs_no_weights_is_noop_without_rebuild():
    c = _client()
    sid = c.post("/api/chat", json={"message": "ai data center gpus", "live": False}).json()["session_id"]
    b = c.post(f"/api/session/{sid}/knobs", json={"live": False}).json()
    assert b["rebuilt"] is False
    assert b["proposal"] is not None  # previous proposal still returned


# ---------------------------------------------------------------- store CRUD
def test_store_upsert_and_read_roundtrip():
    s = Store(":memory:")
    s.upsert_inst_holding(1, "Mgr", "nvda", "CUSIP1", "NVIDIA CORP", 100.0, 10.0, "2026-03-31", "acc1")
    s.upsert_inst_holding(1, "Mgr", "nvda", "CUSIP1", "NVIDIA CORP", 200.0, 20.0, "2026-03-31", "acc1")  # replace
    rows = s.inst_holdings()
    assert len(rows) == 1 and rows[0][2] == "NVDA" and rows[0][5] == 200.0
    assert s.inst_holdings("NVDA") == rows and s.inst_holdings("MU") == []

    s.upsert_insider_trade("mu", 2, "a", "F", True, False, False, "2026-01-01", "P", 10, 5.0, "A")
    trades = s.insider_trades()
    assert len(trades) == 1 and trades[0][0] == "MU" and trades[0][4] is True
    assert s.insider_trades("MU") == trades
