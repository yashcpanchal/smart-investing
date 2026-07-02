"""Curated 13F institutional managers ("smart money") we track.

These are the filer CIKs for well-known managers whose quarterly 13F-HR
holdings we aggregate into the `smart_money_13f` universe signal. Manager CIKs
are NOT in company_tickers.json (they're filers, not issuers) — every CIK below
was verified live against https://data.sec.gov/submissions/CIK{cik}.json
(name match + recent 13F-HR present) on 2026-07-02.
"""

from __future__ import annotations

# (display name, EDGAR filer CIK)
DEFAULT_MANAGERS: list[tuple[str, int]] = [
    ("Berkshire Hathaway", 1067983),
    ("Renaissance Technologies", 1037389),
    ("Bridgewater Associates", 1350694),
    ("Citadel Advisors", 1423053),
    ("ARK Investment Management", 1697748),
    ("Two Sigma Investments", 1179392),
    ("D. E. Shaw & Co", 1009207),
    ("Coatue Management", 1135730),
    ("Tiger Global Management", 1167483),
    ("Baillie Gifford", 1088875),
]
