""""Smart money" signals from free SEC EDGAR filings.

Two evidence sources, both authoritative and $0:

- 13F-HR: quarterly holdings of curated institutional managers (Berkshire,
  RenTech, ...). The holdings live in a separate "information table" XML inside
  the accession (NOT the primary document); issuers are identified by name +
  CUSIP (no tickers), so we match names against our corpus company titles.
- Form 4: insider transactions filed against the ISSUER's own CIK. The XML
  carries the trading symbol, the reporting owner's role flags, and the
  non-derivative transaction rows (code P = open-market purchase, S = sale).

Everything network-facing is thin; the parsers and the scorer are pure and
offline-testable. Scoring is deterministic: the insider window is anchored to
the max transaction date IN THE STORE, never wall-clock now().

"""

from __future__ import annotations

import math
import re
import xml.etree.ElementTree as ET
from datetime import date, timedelta

from smart_investing.data.managers import DEFAULT_MANAGERS
from smart_investing.retrieval.graph import name_keys


# --------------------------------------------------------------------------- #
# Pure XML parsing (offline-testable)
# --------------------------------------------------------------------------- #
def _strip_namespaces(xml_text: str) -> str:
    """Drop XML namespace machinery so one ElementTree path handles namespaced
    and namespace-free variants (both occur in the EDGAR wild)."""
    text = re.sub(r"<\?xml[^>]*\?>", "", xml_text)
    text = re.sub(r'\sxmlns(:[\w.-]+)?\s*=\s*"[^"]*"', "", text)
    text = re.sub(r'\s[\w.-]+:[\w.-]+\s*=\s*"[^"]*"', "", text)  # xsi:type etc.
    text = re.sub(r"<(/?)[\w.-]+:", r"<\1", text)  # tag prefixes: <ns1:infoTable>
    return text.strip()


def _text(el: ET.Element | None, path: str = "") -> str:
    if el is None:
        return ""
    node = el.find(path) if path else el
    return (node.text or "").strip() if node is not None else ""


def _num(s: str) -> float:
    try:
        return float(s.replace(",", ""))
    except (ValueError, AttributeError):
        return 0.0


def _flag(s: str) -> bool:
    return s.strip().lower() in ("1", "true", "yes")


def parse_form4(xml_text: str) -> dict:
    """Form 4 ownershipDocument -> issuer, reporting owner + flags, and the
    non-derivative transaction rows. Handles namespaced and plain XML."""
    root = ET.fromstring(_strip_namespaces(xml_text))
    issuer = root.find("issuer")
    owner = root.find("reportingOwner")
    rel = owner.find("reportingOwnerRelationship") if owner is not None else None

    transactions: list[dict] = []
    for tx in root.findall(".//nonDerivativeTransaction"):
        transactions.append(
            {
                "date": _text(tx, "transactionDate/value"),
                "code": _text(tx, "transactionCoding/transactionCode").upper(),
                "shares": _num(_text(tx, "transactionAmounts/transactionShares/value")),
                "price": _num(_text(tx, "transactionAmounts/transactionPricePerShare/value")),
                "acquired_disposed": _text(
                    tx, "transactionAmounts/transactionAcquiredDisposedCode/value"
                ).upper(),
            }
        )

    return {
        "symbol": _text(issuer, "issuerTradingSymbol").upper(),
        "issuer_cik": _text(issuer, "issuerCik").lstrip("0"),
        "owner_name": _text(owner, "reportingOwnerId/rptOwnerName"),
        "is_officer": _flag(_text(rel, "isOfficer")),
        "is_director": _flag(_text(rel, "isDirector")),
        "is_ten_pct_owner": _flag(_text(rel, "isTenPercentOwner")),
        "transactions": transactions,
    }


def parse_13f_infotable(xml_text: str) -> list[dict]:
    """13F information-table XML -> one row per holding: issuer name, CUSIP,
    reported dollar value, share count. Issuers carry NO ticker here."""
    root = ET.fromstring(_strip_namespaces(xml_text))
    rows: list[dict] = []
    tables = root.findall(".//infoTable")
    if not tables and root.tag == "infoTable":
        tables = [root]
    for t in tables:
        name = _text(t, "nameOfIssuer")
        if not name:
            continue
        rows.append(
            {
                "issuer_name": name,
                "cusip": _text(t, "cusip").upper(),
                "value_usd": _num(_text(t, "value")),
                "shares": _num(_text(t, "shrsOrPrnAmt/sshPrnamt")),
            }
        )
    return rows


# --------------------------------------------------------------------------- #
# Issuer-name -> corpus-ticker matching (13F rows have no tickers)
# --------------------------------------------------------------------------- #
def build_name_key_map(companies: list[tuple]) -> dict[str, str]:
    """store.companies() rows -> {normalized name key: ticker} (bigram keys,
    same normalization the co-mention graph uses)."""
    keymap: dict[str, str] = {}
    for row in companies:
        ticker, title = row[0], row[2] if len(row) > 2 else ""
        for k in name_keys(title or ""):
            keymap.setdefault(k, ticker)
    return keymap


def match_issuer(issuer_name: str, keymap: dict[str, str]) -> str | None:
    for k in name_keys(issuer_name or ""):
        if k in keymap:
            return keymap[k]
    return None


# --------------------------------------------------------------------------- #
# EDGAR fetch helpers (network)
# --------------------------------------------------------------------------- #
def filings_of_form(client, cik: int, form: str, limit: int | None = None) -> list[dict]:
    """ALL matching filings from filings.recent (latest_filing stops at the
    first; Form 4s need the list)."""
    recent = client.submissions(cik).get("filings", {}).get("recent", {})
    forms = recent.get("form", [])
    report_dates = recent.get("reportDate", [])
    out: list[dict] = []
    for i, f in enumerate(forms):
        if f != form:
            continue
        out.append(
            {
                "accession": recent["accessionNumber"][i],
                "primary_doc": recent["primaryDocument"][i],
                "filing_date": recent["filingDate"][i],
                "report_date": report_dates[i] if i < len(report_dates) else "",
            }
        )
        if limit is not None and len(out) >= limit:
            break
    return out


def fetch_accession_index(client, cik: int, accession: str) -> dict:
    """index.json of one accession's archive folder (lists every member file)."""
    return client.accession_index(cik, accession)


def infotable_candidates(index_json: dict) -> list[str]:
    """Likely info-table XML filenames, best first: names containing
    infotable/informationtable, then remaining XMLs by size (the info table is
    the big one; primary_doc.xml is the small cover page)."""
    items = index_json.get("directory", {}).get("item", [])
    xmls = [it for it in items if str(it.get("name", "")).lower().endswith(".xml")]
    named = [it["name"] for it in xmls if re.search(r"info(rmation)?table", it["name"], re.I)]
    rest = sorted(
        (it for it in xmls if it["name"] not in named and "primary_doc" not in it["name"].lower()),
        key=lambda it: -_num(str(it.get("size") or 0)),
    )
    return named + [it["name"] for it in rest]


# --------------------------------------------------------------------------- #
# Ingestion
# --------------------------------------------------------------------------- #
def _get_client(client):
    if client is not None:
        return client
    from smart_investing.data.edgar import EdgarClient

    return EdgarClient()


def ingest_insider_trades(store, tickers: list[str], client=None, per_ticker_limit: int = 8) -> dict:
    """Recent Form 4s for each corpus ticker -> insider_trades rows."""
    client = _get_client(client)
    ciks = {row[0]: row[1] for row in store.companies()}
    ok: list[tuple[str, int]] = []
    errors: list[tuple[str, str]] = []
    for ticker in tickers:
        t = ticker.upper()
        cik = ciks.get(t)
        if not cik:
            errors.append((t, "no CIK in store"))
            continue
        try:
            n = 0
            for filing in filings_of_form(client, int(cik), "4", limit=per_ticker_limit):
                doc = filing["primary_doc"].split("/")[-1]  # strip xsl render prefix
                if not doc.lower().endswith(".xml"):
                    continue
                parsed = parse_form4(client.filing_html(int(cik), filing["accession"], doc))
                sym = parsed["symbol"] or t
                for tx in parsed["transactions"]:
                    if not tx["date"] or not tx["code"]:
                        continue
                    store.upsert_insider_trade(
                        ticker=sym,
                        cik=int(cik),
                        accession=filing["accession"],
                        filer_name=parsed["owner_name"],
                        is_officer=parsed["is_officer"],
                        is_director=parsed["is_director"],
                        is_ten_pct_owner=parsed["is_ten_pct_owner"],
                        transaction_date=tx["date"],
                        transaction_code=tx["code"],
                        shares=tx["shares"],
                        price=tx["price"],
                        acquired_disposed=tx["acquired_disposed"],
                    )
                    n += 1
            ok.append((t, n))
        except Exception as e:  # network / parse hiccup: skip the name, keep going
            errors.append((t, str(e)[:120]))
    return {"ok": ok, "errors": errors}


def ingest_13f(store, client=None, managers: list[tuple[str, int]] | None = None) -> dict:
    """Latest 13F-HR per curated manager -> matched inst_holdings rows.
    Holdings whose issuer name doesn't match a corpus company are dropped."""
    client = _get_client(client)
    managers = managers if managers is not None else DEFAULT_MANAGERS
    keymap = build_name_key_map(store.companies())
    ok: list[tuple[str, int, int]] = []  # (manager, matched, total)
    errors: list[tuple[str, str]] = []
    for name, cik in managers:
        try:
            filings = filings_of_form(client, cik, "13F-HR", limit=1)
            if not filings:
                errors.append((name, "no 13F-HR found"))
                continue
            filing = filings[0]
            index = fetch_accession_index(client, cik, filing["accession"])
            rows: list[dict] = []
            for candidate in infotable_candidates(index):
                try:
                    rows = parse_13f_infotable(client.filing_html(cik, filing["accession"], candidate))
                except Exception:
                    rows = []
                if rows:
                    break
            # Replace, don't accumulate: drop this manager's prior-quarter rows
            # so the table always holds exactly one (latest) 13F per manager.
            # Only once the new filing parsed — a fetchable-but-unparseable info
            # table must not wipe the last good quarter.
            if rows:
                store.clear_inst_holdings(cik)
            matched = 0
            for row in rows:
                ticker = match_issuer(row["issuer_name"], keymap)
                if ticker is None:
                    continue
                store.upsert_inst_holding(
                    manager_cik=cik,
                    manager_name=name,
                    ticker=ticker,
                    cusip=row["cusip"],
                    issuer_name=row["issuer_name"],
                    value_usd=row["value_usd"],
                    shares=row["shares"],
                    period_of_report=filing.get("report_date", ""),
                    accession=filing["accession"],
                )
                matched += 1
            ok.append((name, matched, len(rows)))
        except Exception as e:
            errors.append((name, str(e)[:120]))
    return {"ok": ok, "errors": errors}


# --------------------------------------------------------------------------- #
# Deterministic scoring (no network, no wall-clock)
# --------------------------------------------------------------------------- #
INSIDER_WINDOW_DAYS = 180


def _rank_normalize(values: dict[str, float]) -> dict[str, float]:
    """Values -> 0..1 by rank (ties broken by ticker for determinism)."""
    if not values:
        return {}
    if len(values) == 1:
        return {t: 1.0 for t in values}
    ordered = sorted(values.items(), key=lambda kv: (kv[1], kv[0]))
    n = len(ordered) - 1
    return {t: round(i / n, 4) for i, (t, _v) in enumerate(ordered)}


def compute_smart_money_scores(store) -> dict[str, dict[str, float]]:
    """Store -> {ticker: {smart_money_13f, smart_money_insider}} (each 0..1,
    rank-normalized; keys absent when that signal has no data for the ticker).

    Deterministic: the insider trailing window is measured from the max
    transaction_date in the store — never datetime.now(). Empty tables -> {}.
    """
    out: dict[str, dict[str, float]] = {}

    # 13F: aggregate reported value across curated managers, log-scaled.
    agg: dict[str, float] = {}
    for row in store.inst_holdings():
        ticker, value = row[2], float(row[5] or 0.0)
        if value > 0:
            agg[ticker] = agg.get(ticker, 0.0) + value
    for t, s in _rank_normalize({t: math.log1p(v) for t, v in agg.items()}).items():
        out.setdefault(t, {})["smart_money_13f"] = s

    # Insider: net open-market buying (P notional minus S notional) over a
    # trailing window anchored at the newest trade in the store.
    trades = store.insider_trades()
    dates = [tr[7] for tr in trades if tr[7]]
    if dates:
        try:
            ref = date.fromisoformat(max(dates))
            cutoff = (ref - timedelta(days=INSIDER_WINDOW_DAYS)).isoformat()
        except ValueError:
            cutoff = ""
        net: dict[str, float] = {}
        for tr in trades:
            ticker, tx_date, code = tr[0], tr[7], (tr[8] or "").upper()
            if not tx_date or tx_date < cutoff:
                continue
            notional = float(tr[9] or 0.0) * float(tr[10] or 0.0)
            if code == "P":
                net[ticker] = net.get(ticker, 0.0) + notional
            elif code == "S":
                net[ticker] = net.get(ticker, 0.0) - notional
        ranked = _rank_normalize(net)
        for t, s in ranked.items():
            # net sellers score 0 — "insiders dumping" must never boost a name
            out.setdefault(t, {})["smart_money_insider"] = s if net[t] > 0 else 0.0
    return out
