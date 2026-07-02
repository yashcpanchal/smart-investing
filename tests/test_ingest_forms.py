"""Foreign-filer ingestion: form fallback (10-K -> 20-F -> 40-F) and the 40-F
AIF-exhibit path, exercised offline via an injected fake EDGAR client."""

from __future__ import annotations

from smart_investing.data.ingest import ingest_companies
from smart_investing.data.store import Store

_AIF_HTML = (
    "<html><body><p>GENERAL DEVELOPMENT OF THE BUSINESS</p><p>"
    + "Uranium exploration and mining operations across the Athabasca basin. " * 40
    + "</p><p>RISK FACTORS</p><p>"
    + "Commodity price volatility may affect results. " * 60
    + "</p><p>LEGAL PROCEEDINGS None.</p></body></html>"
)
_COVER_HTML = "<html><body><p>40-F cover page wrapper only.</p></body></html>"
_20F_HTML = "<html><body><p>Foreign issuer annual report narrative text.</p></body></html>"


class FakeClient:
    """Duck-typed stand-in for EdgarClient (ingest only calls these five)."""

    def __init__(self, filings: dict, htmls: dict, index: dict | None = None,
                 index_error: bool = False) -> None:
        self.filings = filings  # form -> filing dict
        self.htmls = htmls  # doc name -> html
        self.index = index or {}
        self.index_error = index_error
        self.form_requests: list[str] = []
        self.doc_requests: list[str] = []

    def cik_for(self, ticker: str) -> dict:
        return {"cik": 42, "title": f"{ticker} Corp"}

    def latest_filing(self, cik: int, form: str = "10-K") -> dict | None:
        self.form_requests.append(form)
        return self.filings.get(form)

    def filing_html(self, cik: int, accession: str, doc: str) -> str:
        self.doc_requests.append(doc)
        return self.htmls[doc]

    def accession_index(self, cik: int, accession: str) -> dict:
        if self.index_error:
            raise RuntimeError("index.json unavailable")
        return self.index


def _filing(form: str, primary: str) -> dict:
    return {
        "accession": "0000000000-26-000001",
        "primary_doc": primary,
        "filing_date": "2026-01-15",
        "form": form,
    }


def test_fallback_tries_forms_in_order_and_stores_real_form():
    client = FakeClient(
        filings={"20-F": _filing("20-F", "arm-20f.htm")},
        htmls={"arm-20f.htm": _20F_HTML},
    )
    store = Store(":memory:")
    summary, store = ingest_companies(["ARM"], client=client, store=store)

    assert client.form_requests == ["10-K", "20-F"]  # stopped at first hit
    assert [t for t, _ in summary["ok"]] == ["ARM"]
    form = store.con.execute("select form from filings where ticker='ARM'").fetchone()[0]
    assert form == "20-F"
    docs = store.documents()
    assert docs and all(r[0] == "ARM" for r in docs)
    store.close()


def test_40f_fetches_aif_exhibit_not_cover():
    index = {
        "directory": {
            "item": [
                {"name": "cover40f.htm", "size": 9_000},
                {"name": "denison-aif.htm", "size": 900_000},
            ]
        }
    }
    client = FakeClient(
        filings={"40-F": _filing("40-F", "cover40f.htm")},
        htmls={"cover40f.htm": _COVER_HTML, "denison-aif.htm": _AIF_HTML},
        index=index,
    )
    store = Store(":memory:")
    summary, store = ingest_companies(["DNN"], client=client, store=store)

    assert client.form_requests == ["10-K", "20-F", "40-F"]
    assert client.doc_requests == ["denison-aif.htm"]  # AIF, not the cover
    sections = {r[1]: r[2] for r in store.documents()}
    assert "business" in sections and "Athabasca" in sections["business"]
    assert "risk_factors" in sections and "volatility" in sections["risk_factors"]
    store.close()


def test_40f_falls_back_to_primary_doc_when_index_fails():
    client = FakeClient(
        filings={"40-F": _filing("40-F", "cover40f.htm")},
        htmls={"cover40f.htm": _COVER_HTML},
        index_error=True,
    )
    summary, store = ingest_companies(["CCJ"], client=client, store=Store(":memory:"))
    assert [t for t, _ in summary["ok"]] == ["CCJ"]
    sections = {r[1]: r[2] for r in store.documents()}
    assert "full" in sections  # cover text still lands in the store
    store.close()


def test_no_annual_report_at_all_is_skipped():
    client = FakeClient(filings={}, htmls={})
    summary, store = ingest_companies(["XYZ"], client=client, store=Store(":memory:"))
    assert summary["skipped"] == [("XYZ", "no 10-K/20-F/40-F")]
    assert store.count("documents") == 0
    store.close()


def test_custom_forms_list_respected():
    client = FakeClient(
        filings={"20-F": _filing("20-F", "x.htm")}, htmls={"x.htm": _20F_HTML}
    )
    summary, store = ingest_companies(
        ["ABC"], client=client, store=Store(":memory:"), forms=["10-K"]
    )
    assert client.form_requests == ["10-K"]
    assert summary["skipped"] == [("ABC", "no 10-K")]
    store.close()
