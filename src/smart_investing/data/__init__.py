"""Data adapters: price history (yfinance) and SEC EDGAR filings."""

from smart_investing.data.edgar import EdgarClient, extract_sections, html_to_text
from smart_investing.data.ingest import ingest_companies
from smart_investing.data.prices import load_prices, synthetic_prices
from smart_investing.data.store import Store

__all__ = [
    "load_prices",
    "synthetic_prices",
    "EdgarClient",
    "extract_sections",
    "html_to_text",
    "ingest_companies",
    "Store",
]
