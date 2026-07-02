"""SEC EDGAR client — free, authoritative US-filing data (no API key).

Pulls ticker→CIK mapping, finds the latest 10-K, downloads it, and extracts the
Business (Item 1) and Risk Factors (Item 1A) sections — the text the thematic
retrieval engine embeds. Respects SEC fair-access: descriptive User-Agent and a
throttle so we stay well under 10 req/s.
"""

from __future__ import annotations

import html as _html
import re
import time

import httpx

from smart_investing.config import settings

TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
ARCHIVE_DOC_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{acc}/{doc}"
ACCESSION_INDEX_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{acc}/index.json"

# Annual-report forms we can extract thematic text from, tried in this order.
# 20-F: foreign private issuers (e.g. ARM); 40-F: Canadian MJDS filers
# (e.g. Cameco, Denison) whose real disclosure lives in an AIF exhibit.
ANNUAL_REPORT_FORMS = ("10-K", "20-F", "40-F")


class EdgarClient:
    def __init__(self, user_agent: str | None = None, min_interval: float = 0.15) -> None:
        self.user_agent = user_agent or settings.sec_user_agent
        self._client = httpx.Client(
            headers={"User-Agent": self.user_agent, "Accept-Encoding": "gzip, deflate"},
            timeout=30.0,
            follow_redirects=True,
        )
        self._min_interval = min_interval
        self._last = 0.0
        self._ticker_map: dict[str, dict] | None = None

    def _throttle(self) -> None:
        dt = time.monotonic() - self._last
        if dt < self._min_interval:
            time.sleep(self._min_interval - dt)
        self._last = time.monotonic()

    def _get(self, url: str) -> httpx.Response:
        self._throttle()
        r = self._client.get(url)
        r.raise_for_status()
        return r

    def company_tickers(self) -> dict[str, dict]:
        if self._ticker_map is None:
            data = self._get(TICKERS_URL).json()
            self._ticker_map = {
                row["ticker"].upper(): {"cik": int(row["cik_str"]), "title": row["title"]}
                for row in data.values()
            }
        return self._ticker_map

    def cik_for(self, ticker: str) -> dict | None:
        return self.company_tickers().get(ticker.upper())

    def submissions(self, cik: int) -> dict:
        return self._get(SUBMISSIONS_URL.format(cik=cik)).json()

    def latest_filing(self, cik: int, form: str = "10-K") -> dict | None:
        recent = self.submissions(cik).get("filings", {}).get("recent", {})
        forms = recent.get("form", [])
        for i, f in enumerate(forms):
            if f == form:
                return {
                    "accession": recent["accessionNumber"][i],
                    "primary_doc": recent["primaryDocument"][i],
                    "filing_date": recent["filingDate"][i],
                    "form": f,
                }
        return None

    def filing_html(self, cik: int, accession: str, primary_doc: str) -> str:
        url = ARCHIVE_DOC_URL.format(cik=cik, acc=accession.replace("-", ""), doc=primary_doc)
        return self._get(url).text

    def accession_index(self, cik: int, accession: str) -> dict:
        """File listing (index.json) for one accession — used to discover
        exhibits (e.g. the AIF inside a 40-F) beyond the primary document."""
        url = ACCESSION_INDEX_URL.format(cik=cik, acc=accession.replace("-", ""))
        return self._get(url).json()

    def close(self) -> None:
        self._client.close()


# --------------------------------------------------------------------------- #
# Pure parsing helpers (offline-testable)
# --------------------------------------------------------------------------- #
def html_to_text(html: str) -> str:
    text = re.sub(r"(?is)<(script|style)[^>]*>.*?</\1>", " ", html)
    text = re.sub(r"(?s)<[^>]+>", " ", text)
    text = _html.unescape(text)  # decode &#8217; &nbsp; &amp; etc.
    return re.sub(r"\s+", " ", text).strip()


def _section(text: str, start_pat: str, end_pats: list[str]) -> str:
    """Text from the LAST occurrence of a section header (the real body header,
    not the table-of-contents entry) up to the next boundary header after it."""
    starts = [m.end() for m in re.finditer(start_pat, text, re.I)]
    if not starts:
        return ""
    start = starts[-1]
    ends = [
        m.start()
        for ep in end_pats
        for m in re.finditer(ep, text, re.I)
        if m.start() > start
    ]
    end = min(ends) if ends else len(text)
    return text[start:end].strip()


def _section_any(
    text: str, start_pats: list[str], end_pats: list[str], min_chars: int = 1_000
) -> str:
    """Like _section, but tries start patterns in priority order and, within one
    pattern, takes the LATEST start whose span to the next boundary is at least
    `min_chars` — cross-references and table-of-contents entries yield tiny
    spans and get skipped, the real body header yields a long one."""
    for sp in start_pats:
        starts = [m.end() for m in re.finditer(sp, text, re.I)]
        for start in reversed(starts):
            ends = [
                m.start()
                for ep in end_pats
                for m in re.finditer(ep, text, re.I)
                if m.start() > start
            ]
            end = min(ends) if ends else len(text)
            if end - start >= min_chars:
                return text[start:end].strip()
    return ""


# Negative lookahead marking a CROSS-REFERENCE rather than a body header: in
# 20-F prose, references read `"Item 4. Information on the Company—B. Business
# Overview"` (em/en dash or closing quote right after the title), while the real
# body header is followed by plain content. Derived from ARM's live 20-F.
_REF = r"(?!\s*[—–\-.,”’])"

_20F_BUSINESS_START = r"item\s+4\.?\s+information\s+on\s+the\s+company" + _REF
_20F_BUSINESS_END = [
    r"item\s+4a\b" + _REF,
    r"item\s+5\.?\s+operating\s+and\s+financial\s+review\s+and\s+prospects" + _REF,
]
_20F_RISK_START = r"\bd\.\s*risk\s+factors" + _REF  # Item 3.D under "Key Information"
_20F_RISK_END = [r"item\s+4\.?\s+information\s+on\s+the\s+company" + _REF]

# Canadian AIF (NI 51-102F2) headings, plus Cameco's house style
# ("Risks that can affect our business"). Verified against Cameco's live AIF.
_AIF_RISK_STARTS = [
    r"risk\s+factors\b" + _REF,
    r"risks\s+that\s+can\s+affect\s+our\s+business",
]
_AIF_RISK_ENDS = [
    r"legal\s+proceedings",
    r"material\s+contracts",
    r"transfer\s+agent",
    r"investor\s+information",
    r"interests?\s+of\s+experts",
]
_AIF_BUSINESS_STARTS = [
    r"general\s+development\s+of\s+the\s+business",
    r"description\s+of\s+(?:the|our)\s+business",
]
_AIF_BUSINESS_ENDS = [
    r"risk\s+factors\b",
    r"risks\s+that\s+can\s+affect\s+our\s+business",
    r"legal\s+proceedings",
    r"dividends\b",
]


def _extract_10k(text: str) -> dict[str, str]:
    business = _section(text, r"item\s+1\b\.?\s*business", [r"item\s+1a\b"])
    risk = _section(text, r"item\s+1a\b\.?\s*risk\s+factors", [r"item\s+1b\b", r"item\s+2\b"])
    out: dict[str, str] = {}
    if business:
        out["business"] = business
    if risk:
        out["risk_factors"] = risk
    return out


def _extract_20f(text: str) -> dict[str, str]:
    business = _section_any(text, [_20F_BUSINESS_START], _20F_BUSINESS_END)
    risk = _section_any(text, [_20F_RISK_START], _20F_RISK_END)
    out: dict[str, str] = {}
    if business:
        out["business"] = business
    if risk:
        out["risk_factors"] = risk
    return out


def _extract_aif(text: str) -> dict[str, str]:
    """Sections from a 40-F's Annual Information Form exhibit (heading styles
    vary by issuer, so this is heuristic; caller falls back to raw text)."""
    business = _section_any(text, _AIF_BUSINESS_STARTS, _AIF_BUSINESS_ENDS, min_chars=2_000)
    risk = _section_any(text, _AIF_RISK_STARTS, _AIF_RISK_ENDS, min_chars=2_000)
    out: dict[str, str] = {}
    if business:
        out["business"] = business
    if risk:
        out["risk_factors"] = risk
    return out


def extract_sections(text: str, form: str = "10-K") -> dict[str, str]:
    """Best-effort extraction of the Business and Risk Factors sections for the
    given annual-report form. Always returns at least a 'full' fallback so the
    company still gets indexed — text in the store beats nothing."""
    form = form.upper()
    if form == "20-F":
        out = _extract_20f(text)
    elif form == "40-F":
        out = _extract_aif(text)
    else:
        out = _extract_10k(text)
    if not out:
        out["full"] = text[:20_000]  # fallback so the company still gets indexed
    return out


def find_aif_document(index_json: dict) -> str | None:
    """Pick the Annual Information Form exhibit from a 40-F accession's
    index.json. The 40-F primary document is usually a thin cover wrapper; the
    real disclosure is an EX-99.* exhibit. Preference order: a filename hinting
    at the AIF; else the lowest-numbered substantial (>=100 KB) EX-99 exhibit
    (convention: 99.1 is the AIF, tiny ones are certifications/consents); else
    the largest .htm exhibit."""
    items = index_json.get("directory", {}).get("item", [])
    htms: list[tuple[str, int]] = []
    for it in items:
        name = str(it.get("name", ""))
        if not name.lower().endswith((".htm", ".html")) or "index" in name.lower():
            continue
        try:
            size = int(it.get("size") or 0)
        except (TypeError, ValueError):
            size = 0
        htms.append((name, size))
    if not htms:
        return None
    for name, _size in htms:
        low = name.lower()
        if "aif" in low or "annualinformationform" in low:
            return name
    exhibits = []
    for name, size in htms:
        m = re.search(r"ex[-_.]?99[-_.]?(\d+)", name.lower())
        if m:
            exhibits.append((int(m.group(1)), name, size))
    big = sorted(e for e in exhibits if e[2] >= 100_000)
    if big:
        return big[0][1]
    if exhibits:
        return max(exhibits, key=lambda e: e[2])[1]
    return None
