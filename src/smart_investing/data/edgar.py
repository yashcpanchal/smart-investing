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


def extract_sections(text: str) -> dict[str, str]:
    """Best-effort extraction of Item 1 (Business) and Item 1A (Risk Factors)."""
    business = _section(text, r"item\s+1\b\.?\s*business", [r"item\s+1a\b"])
    risk = _section(text, r"item\s+1a\b\.?\s*risk\s+factors", [r"item\s+1b\b", r"item\s+2\b"])
    out: dict[str, str] = {}
    if business:
        out["business"] = business
    if risk:
        out["risk_factors"] = risk
    if not out:
        out["full"] = text[:20_000]  # fallback so the company still gets indexed
    return out
