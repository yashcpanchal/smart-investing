"""DuckDB-backed local store for company filings and (later) embeddings.

Zero-cost, file-based, no server. Holds the ingested corpus the thematic
retrieval engine (Phase 5) queries.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import duckdb

from smart_investing.config import settings

_SCHEMA = """
create table if not exists companies (
    ticker varchar primary key,
    cik bigint,
    title varchar,
    sic_description varchar,
    updated_at varchar
);
create table if not exists filings (
    accession varchar,
    form varchar,
    ticker varchar,
    cik bigint,
    filing_date varchar,
    primary_doc varchar,
    primary key (accession, form)
);
create table if not exists documents (
    ticker varchar,
    cik bigint,
    accession varchar,
    section varchar,        -- 'business' | 'risk_factors' | 'full'
    text varchar,
    primary key (ticker, accession, section)
);
create table if not exists relations (
    src varchar,            -- supplier / upstream (for 'supplies'); either side for 'competes'
    dst varchar,            -- customer / downstream
    rel varchar,            -- 'supplies' | 'competes' | 'partner'
    weight double,
    origin varchar,         -- 'curated' | 'llm' | 'comention'
    primary key (src, dst, rel, origin)
);
"""


def default_db_path() -> str:
    Path(settings.data_dir).mkdir(parents=True, exist_ok=True)
    return str(Path(settings.data_dir) / "smart_investing.duckdb")


def _now() -> str:
    return datetime.now(UTC).isoformat()


class Store:
    def __init__(self, path: str | None = None) -> None:
        self.path = path or default_db_path()
        self.con = duckdb.connect(self.path)
        self.con.execute(_SCHEMA)

    # ---- writes ----
    def upsert_company(self, ticker: str, cik: int, title: str, sic_description: str = "") -> None:
        self.con.execute(
            "insert or replace into companies values (?,?,?,?,?)",
            [ticker.upper(), cik, title, sic_description, _now()],
        )

    def upsert_filing(
        self, accession: str, form: str, ticker: str, cik: int, filing_date: str, primary_doc: str
    ) -> None:
        self.con.execute(
            "insert or replace into filings values (?,?,?,?,?,?)",
            [accession, form, ticker.upper(), cik, filing_date, primary_doc],
        )

    def upsert_document(self, ticker: str, cik: int, accession: str, section: str, text: str) -> None:
        self.con.execute(
            "insert or replace into documents values (?,?,?,?,?)",
            [ticker.upper(), cik, accession, section, text],
        )

    def upsert_relation(self, src: str, dst: str, rel: str, weight: float, origin: str) -> None:
        self.con.execute(
            "insert or replace into relations values (?,?,?,?,?)",
            [src.upper(), dst.upper(), rel, weight, origin],
        )

    def replace_relations(self, origin: str, edges: list[tuple[str, str, str, float]]) -> None:
        """Atomically swap all edges of a given origin (e.g. re-running LLM extraction)."""
        self.con.execute("delete from relations where origin = ?", [origin])
        for src, dst, rel, weight in edges:
            self.upsert_relation(src, dst, rel, weight, origin)

    # ---- reads ----
    def companies(self) -> list[tuple]:
        return self.con.execute(
            "select ticker, cik, title, sic_description from companies order by ticker"
        ).fetchall()

    # Deterministic order: business first (the cleanest thematic signal), then
    # risk_factors, then any fallback 'full' text. Without an ORDER BY, DuckDB's
    # row order is arbitrary, which made per-ticker text concatenation (and thus
    # embeddings and retrieval ranks) nondeterministic across runs.
    _DOC_ORDER = (
        "order by ticker, "
        "case section when 'business' then 0 when 'risk_factors' then 1 else 2 end, "
        "accession"
    )

    def documents(self, section: str | None = None) -> list[tuple]:
        if section:
            return self.con.execute(
                f"select ticker, section, text from documents where section = ? {self._DOC_ORDER}",
                [section],
            ).fetchall()
        return self.con.execute(
            f"select ticker, section, text from documents {self._DOC_ORDER}"
        ).fetchall()

    def relations(self, origin: str | None = None) -> list[tuple]:
        """Returns (src, dst, rel, weight, origin)."""
        if origin:
            return self.con.execute(
                "select src, dst, rel, weight, origin from relations where origin = ?", [origin]
            ).fetchall()
        return self.con.execute("select src, dst, rel, weight, origin from relations").fetchall()

    def count(self, table: str) -> int:
        return self.con.execute(f"select count(*) from {table}").fetchone()[0]

    def close(self) -> None:
        self.con.close()

    def __enter__(self) -> Store:
        return self

    def __exit__(self, *exc) -> None:
        self.close()
