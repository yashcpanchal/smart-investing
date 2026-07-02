from __future__ import annotations

from smart_investing.data.edgar import extract_sections, find_aif_document, html_to_text

SAMPLE_10K = """
<html><head><style>.x{color:red}</style></head><body>
<table><tr><td>Item 1. Business</td><td>3</td></tr>
<tr><td>Item 1A. Risk Factors</td><td>12</td></tr></table>
<p>Item 1. Business</p>
<p>We design and manufacture quantum computing hardware and the cryogenic cooling
systems that support it. Our customers include national laboratories.</p>
<p>Item 1A. Risk Factors</p>
<p>Our business is subject to supply chain disruptions in rare-earth materials
and dependence on a small number of foundry partners.</p>
<p>Item 1B. Unresolved Staff Comments</p>
<p>None.</p>
</body></html>
"""


def test_html_to_text_strips_tags_and_styles():
    text = html_to_text(SAMPLE_10K)
    assert "<" not in text
    assert "color:red" not in text  # style block removed
    assert "quantum computing hardware" in text


def test_extract_sections_finds_business_and_risk():
    sections = extract_sections(html_to_text(SAMPLE_10K))
    assert "business" in sections
    assert "quantum computing hardware" in sections["business"]
    # business span should stop before risk factors
    assert "rare-earth" not in sections["business"]
    assert "risk_factors" in sections
    assert "supply chain" in sections["risk_factors"]


def test_extract_sections_fallback_when_no_items():
    sections = extract_sections("just some plain text with no item headers at all")
    assert "full" in sections


# --------------------------------------------------------------------------- #
# 20-F (foreign private issuer, e.g. ARM). Structure distilled from ARM's live
# 20-F: a table of contents, cross-references with an em dash right after the
# item title, and the real body headers followed by plain content.
# --------------------------------------------------------------------------- #
_RISK_BODY = (
    "Demand for our semiconductor licensing products may decline materially. " * 30
)
_BIZ_BODY = (
    "We architect energy efficient processor designs licensed across mobile "
    "datacenter and automotive markets worldwide. " * 25
)
_MDA_BODY = "Revenue increased due to higher royalty rates across our licensing lines. "

SAMPLE_20F = f"""
<html><body>
<p>Item 3. Key Information 6 Item 4. Information on the Company 55
Item 4A. Unresolved Staff Comments 66
Item 5. Operating and Financial Review and Prospects 66</p>
<p>See the factors discussed in &#8220;Item 3. Key Information&#8212;D. Risk Factors&#8221;
and &#8220;Item 4. Information on the Company&#8212;B. Business Overview&#8221; of this
Annual Report.</p>
<p>Item 3. Key Information</p>
<p>A. [Reserved] B. Capitalization and indebtedness Not applicable.
C. Reasons for the offer and use of proceeds Not applicable.</p>
<p>D. Risk Factors</p>
<p>{_RISK_BODY}</p>
<p>Item 4. Information on the Company</p>
<p>A. History and development of the company</p>
<p>{_BIZ_BODY}</p>
<p>Item 4A. Unresolved Staff Comments None.</p>
<p>Item 5. Operating and Financial Review and Prospects</p>
<p>{_MDA_BODY}</p>
</body></html>
"""


def test_extract_20f_business_and_risk():
    sections = extract_sections(html_to_text(SAMPLE_20F), form="20-F")
    assert "business" in sections
    assert "energy efficient processor designs" in sections["business"]
    assert "royalty rates" not in sections["business"]  # stops before Item 5
    assert "risk_factors" in sections
    assert "licensing products may decline" in sections["risk_factors"]
    assert "History and development" not in sections["risk_factors"]  # stops at Item 4


def test_extract_20f_ignores_cross_references():
    # The dash-suffixed references near the top must not be chosen as section
    # starts (they would yield the wrong spans).
    sections = extract_sections(html_to_text(SAMPLE_20F), form="20-F")
    assert not sections["risk_factors"].startswith("and")
    assert sections["business"].startswith("A. History")


def test_extract_20f_fallback_to_full():
    sections = extract_sections("short text without any 20-F headers", form="20-F")
    assert sections == {"full": "short text without any 20-F headers"}


# --------------------------------------------------------------------------- #
# 40-F / Annual Information Form (Canadian MJDS filers, e.g. Cameco/Denison)
# --------------------------------------------------------------------------- #
_AIF_BIZ = (
    "Our uranium mining operations span exploration mining milling refining "
    "and conversion services for nuclear utilities. " * 30
)
_AIF_RISK = (
    "Volatility in uranium prices and regulatory decisions may materially "
    "affect our production plans and financial results. " * 30
)

SAMPLE_AIF = (
    "2025 ANNUAL INFORMATION FORM Contents Our business 9 Risk factors 114 "
    "GENERAL DEVELOPMENT OF THE BUSINESS "
    + _AIF_BIZ
    + " RISK FACTORS "
    + _AIF_RISK
    + " LEGAL PROCEEDINGS None material."
)


def test_extract_aif_sections():
    sections = extract_sections(SAMPLE_AIF, form="40-F")
    assert "business" in sections
    assert "uranium mining operations" in sections["business"]
    assert "risk_factors" in sections
    assert "Volatility in uranium prices" in sections["risk_factors"]
    assert "LEGAL PROCEEDINGS" not in sections["risk_factors"]


def test_extract_aif_cameco_house_style():
    # Cameco's AIF has no "Risk Factors" heading; it uses
    # "Risks that can affect our business" (verified live).
    text = (
        "Contents Risks that can affect our business 114 Legal proceedings 141 "
        "Some introductory text about the company and its uranium operations. "
        "Risks that can affect our business " + _AIF_RISK + " Legal proceedings None."
    )
    sections = extract_sections(text, form="40-F")
    assert "risk_factors" in sections
    assert "Volatility in uranium prices" in sections["risk_factors"]


def test_extract_aif_fallback_to_full():
    sections = extract_sections("a bare cover page with no AIF headings", form="40-F")
    assert "full" in sections


# --------------------------------------------------------------------------- #
# AIF exhibit discovery from an accession index.json (shape verified live
# against Cameco's 40-F accession).
# --------------------------------------------------------------------------- #
def _index(items: list[tuple[str, int]]) -> dict:
    return {"directory": {"item": [{"name": n, "size": s} for n, s in items]}}


CCJ_STYLE_INDEX = _index(
    [
        ("0001193125-26-116229-index.html", 0),
        ("d34605d40f.htm", 609_286),  # primary doc: thin cover wrapper
        ("d34605dex991.htm", 1_797_756),  # EX-99.1 = the AIF
        ("d34605dex992.htm", 2_249_087),  # EX-99.2 = MD&A (largest, NOT the AIF)
        ("d34605dex9910.htm", 6_283),  # certifications
        ("d34605dex995.htm", 4_347),
        ("FilingSummary.xml", 81_747),
        ("g34107g00k01.jpg", 325_035),
    ]
)


def test_find_aif_prefers_lowest_numbered_big_exhibit():
    # ex99.2 (MD&A) is larger, but 99.1 is the AIF by convention.
    assert find_aif_document(CCJ_STYLE_INDEX) == "d34605dex991.htm"


def test_find_aif_prefers_aif_in_name():
    idx = _index([("cover40f.htm", 500_000), ("denison-aif.htm", 90_000)])
    assert find_aif_document(idx) == "denison-aif.htm"


def test_find_aif_falls_back_to_largest_exhibit():
    idx = _index([("aex991.htm", 40_000), ("aex992.htm", 60_000)])
    assert find_aif_document(idx) == "aex992.htm"


def test_find_aif_none_when_no_htm():
    assert find_aif_document(_index([("data.xml", 10_000)])) is None
    assert find_aif_document({}) is None
