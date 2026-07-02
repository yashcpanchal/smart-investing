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
create table if not exists inst_holdings (
    manager_cik bigint,     -- 13F filer (institutional manager)
    manager_name varchar,
    ticker varchar,         -- matched corpus ticker (issuer)
    cusip varchar,
    issuer_name varchar,
    value_usd double,
    shares double,
    period_of_report varchar,
    accession varchar,
    primary key (manager_cik, accession, cusip)
);
create table if not exists insider_trades (
    ticker varchar,
    cik bigint,             -- issuer CIK
    accession varchar,
    filer_name varchar,
    is_officer boolean,
    is_director boolean,
    is_ten_pct_owner boolean,
    transaction_date varchar,
    transaction_code varchar,   -- 'P' open-market purchase | 'S' sale | ...
    shares double,
    price double,
    acquired_disposed varchar,  -- 'A' | 'D'
    primary key (accession, ticker, filer_name, transaction_date, transaction_code)
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

    def upsert_inst_holding(
        self,
        manager_cik: int,
        manager_name: str,
        ticker: str,
        cusip: str,
        issuer_name: str,
        value_usd: float,
        shares: float,
        period_of_report: str,
        accession: str,
    ) -> None:
        self.con.execute(
            "insert or replace into inst_holdings values (?,?,?,?,?,?,?,?,?)",
            [manager_cik, manager_name, ticker.upper(), cusip, issuer_name, value_usd, shares, period_of_report, accession],
        )

    def upsert_insider_trade(
        self,
        ticker: str,
        cik: int,
        accession: str,
        filer_name: str,
        is_officer: bool,
        is_director: bool,
        is_ten_pct_owner: bool,
        transaction_date: str,
        transaction_code: str,
        shares: float,
        price: float,
        acquired_disposed: str,
    ) -> None:
        self.con.execute(
            "insert or replace into insider_trades values (?,?,?,?,?,?,?,?,?,?,?,?)",
            [
                ticker.upper(),
                cik,
                accession,
                filer_name,
                is_officer,
                is_director,
                is_ten_pct_owner,
                transaction_date,
                transaction_code,
                shares,
                price,
                acquired_disposed,
            ],
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

    def documents(self, section: str | None = None) -> list[tuple]:
        if section:
            return self.con.execute(
                "select ticker, section, text from documents where section = ?", [section]
            ).fetchall()
        return self.con.execute("select ticker, section, text from documents").fetchall()

    def relations(self, origin: str | None = None) -> list[tuple]:
        """Returns (src, dst, rel, weight, origin)."""
        if origin:
            return self.con.execute(
                "select src, dst, rel, weight, origin from relations where origin = ?", [origin]
            ).fetchall()
        return self.con.execute("select src, dst, rel, weight, origin from relations").fetchall()

    def inst_holdings(self, ticker: str | None = None) -> list[tuple]:
        """Returns (manager_cik, manager_name, ticker, cusip, issuer_name,
        value_usd, shares, period_of_report, accession)."""
        base = (
            "select manager_cik, manager_name, ticker, cusip, issuer_name, "
            "value_usd, shares, period_of_report, accession from inst_holdings"
        )
        if ticker:
            return self.con.execute(base + " where ticker = ?", [ticker.upper()]).fetchall()
        return self.con.execute(base + " order by ticker").fetchall()

    def insider_trades(self, ticker: str | None = None) -> list[tuple]:
        """Returns (ticker, cik, accession, filer_name, is_officer, is_director,
        is_ten_pct_owner, transaction_date, transaction_code, shares, price,
        acquired_disposed)."""
        base = (
            "select ticker, cik, accession, filer_name, is_officer, is_director, "
            "is_ten_pct_owner, transaction_date, transaction_code, shares, price, "
            "acquired_disposed from insider_trades"
        )
        if ticker:
            return self.con.execute(base + " where ticker = ?", [ticker.upper()]).fetchall()
        return self.con.execute(base + " order by ticker, transaction_date").fetchall()

    def count(self, table: str) -> int:
        return self.con.execute(f"select count(*) from {table}").fetchone()[0]

    def close(self) -> None:
        self.con.close()

    def __enter__(self) -> Store:
        return self

    def __exit__(self, *exc) -> None:
        self.close()
