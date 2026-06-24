"""Ingest pipeline: ticker list -> EDGAR 10-K sections -> DuckDB store."""

from __future__ import annotations

from smart_investing.data.edgar import EdgarClient, extract_sections, html_to_text
from smart_investing.data.store import Store


def ingest_companies(
    tickers: list[str],
    client: EdgarClient | None = None,
    store: Store | None = None,
    form: str = "10-K",
) -> tuple[dict, Store]:
    """Fetch the latest `form` for each ticker, extract Business/Risk sections,
    and upsert into the store. Returns (summary, store)."""
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

            filing = client.latest_filing(cik, form=form)
            if not filing:
                summary["skipped"].append((t, f"no {form}"))
                continue
            store.upsert_filing(
                filing["accession"], filing["form"], t, cik,
                filing["filing_date"], filing["primary_doc"],
            )

            html = client.filing_html(cik, filing["accession"], filing["primary_doc"])
            sections = extract_sections(html_to_text(html))
            for section, body in sections.items():
                store.upsert_document(t, cik, filing["accession"], section, body)
            summary["ok"].append((t, {k: len(v) for k, v in sections.items()}))
        except Exception as e:  # noqa: BLE001 — one bad ticker shouldn't halt a batch
            summary["errors"].append((t, str(e)))

    return summary, store
