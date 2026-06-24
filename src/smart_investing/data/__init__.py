"""Data adapters: price history (yfinance) and, later, EDGAR filings."""

from smart_investing.data.prices import load_prices, synthetic_prices

__all__ = ["load_prices", "synthetic_prices"]
