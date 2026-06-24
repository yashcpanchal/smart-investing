from __future__ import annotations

from smart_investing.data.edgar import extract_sections, html_to_text

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
