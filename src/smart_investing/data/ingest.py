"""Ingest pipeline: ticker list -> EDGAR annual-report sections -> DuckDB store.

Tries 10-K first, then the foreign-filer forms (20-F for foreign private
issuers like ARM, 40-F for Canadian MJDS filers like Cameco/Denison) so those
companies get real text instead of an empty companies row. For 40-Fs the
primary document is a thin cover wrapper, so we fetch the Annual Information
Form exhibit discovered via the accession's file index.
"""

from __future__ import annotations

from collections.abc import Sequence

from smart_investing.data.edgar import (
    ANNUAL_REPORT_FORMS,
    EdgarClient,
    extract_sections,
    find_aif_document,
    html_to_text,
)
from smart_investing.data.store import Store


def _filing_text(client: EdgarClient, cik: int, filing: dict) -> str:
    """Fetch the text we extract sections from. For 40-Fs, prefer the AIF
    exhibit over the cover-page primary document; fall back to the primary
    document if exhibit discovery fails (text in store beats nothing)."""
    doc = filing["primary_doc"]
    if filing["form"].upper() == "40-F":
        try:
            index = client.accession_index(cik, filing["accession"])
            aif = find_aif_document(index)
            if aif:
                doc = aif
        except Exception:  # noqa: BLE001 — exhibit discovery is best-effort
            pass
    return html_to_text(client.filing_html(cik, filing["accession"], doc))


def ingest_companies(
    tickers: list[str],
    client: EdgarClient | None = None,
    store: Store | None = None,
    forms: Sequence[str] = ANNUAL_REPORT_FORMS,
) -> tuple[dict, Store]:
    """Fetch the latest annual report (first hit among `forms`, in order) for
    each ticker, extract Business/Risk sections, and upsert into the store.
    Returns (summary, store)."""
    client = client or EdgarClient()
    store = store or Store()
    summary: dict[str, list] = {"ok": [], "skipped": [], "errors": []}

    for raw in tickers:
        t = raw.strip().upper()
        try:
            info = client.cik_for(t)
            if not info:
                summary["skipped"].append((t, "no CIK"))
                continue
            cik = info["cik"]
            store.upsert_company(t, cik, info["title"])

            filing = None
            for form in forms:
                filing = client.latest_filing(cik, form=form)
                if filing:
                    break
            if not filing:
                summary["skipped"].append((t, f"no {'/'.join(forms)}"))
                continue
            store.upsert_filing(
                filing["accession"], filing["form"], t, cik,
                filing["filing_date"], filing["primary_doc"],
            )

            text = _filing_text(client, cik, filing)
            sections = extract_sections(text, form=filing["form"])
            for section, body in sections.items():
                store.upsert_document(t, cik, filing["accession"], section, body)
            summary["ok"].append((t, {k: len(v) for k, v in sections.items()}))
        except Exception as e:  # noqa: BLE001 — one bad ticker shouldn't halt a batch
            summary["errors"].append((t, str(e)))

    return summary, store
