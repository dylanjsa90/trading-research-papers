"""EDGAR filing metadata (8-K, 10-Q, 10-K) for the universe -> DATA_DIR/filings/<SYMBOL>.parquet.

available_at is the EDGAR acceptance time (§4.2), not the filing or report date. It comes from each filing's
SGML header, because the submissions JSON's acceptanceDateTime cannot be trusted: despite its 'Z' it is true
UTC for some filers and UTC plus the US/Eastern offset for others (found 2026-10-19 against index pages:
AAPL, MSFT and 31 more shifted; NFLX, GE, TSLA and 13 more not). SEC asks for at most 10 requests/s and a
declared User-Agent (§7).
"""

import re
import sys
from datetime import datetime

import pandas as pd
import requests

from trader import config
from trader.data import store, web

TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/{}"
FOLDER_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{folder}/"


def parse_header(text):
    """ACCEPTANCE-DATETIME from a filing's .hdr.sgml, a US/Eastern wall clock; None if absent."""
    m = re.search(r"<ACCEPTANCE-DATETIME>(\d{14})", text)
    if not m:
        return None
    return pd.Timestamp(datetime.strptime(m.group(1), "%Y%m%d%H%M%S")).tz_localize("America/New_York")


def filings_frame(symbol, cik, block, forms=config.FILING_FORMS):
    """Rows for the tracked forms from one column-oriented block (`filings.recent` or an older file)."""
    df = pd.DataFrame(
        {
            "accession": block["accessionNumber"],
            "form": block["form"],
            "filing_date": block["filingDate"],
            "report_date": block["reportDate"],
            "items": block["items"],
            "primary_document": block["primaryDocument"],
            "accepted_json": block["acceptanceDateTime"],
        }
    )
    df = df[df["form"].isin(forms)].reset_index(drop=True)
    folders = [FOLDER_URL.format(cik=cik, folder=acc.replace("-", "")) for acc in df["accession"]]
    df["url"] = [f + doc for f, doc in zip(folders, df["primary_document"])]
    df["header_url"] = [f + f"{acc}.hdr.sgml" for f, acc in zip(folders, df["accession"])]
    return df.assign(symbol=symbol, cik=cik)


def _acceptance(get_text, header_url, accepted_json):
    try:
        ts = parse_header(get_text(header_url))
    except requests.HTTPError:
        ts = None
    if ts is not None:
        return ts, "sgml-header"
    # Read literally as UTC, the JSON value is the later of its two possible meanings: late, never early.
    return pd.Timestamp(accepted_json), "json-latest"


def refresh(symbols=config.UNIVERSE, root=None, session=None, user_agent=None, limiter=None,
            start=config.HISTORY_START):  # fmt: skip
    """Fetch filings not yet stored, one header request each. Older submission files are read only on a
    symbol's first backfill; after that the `recent` block (the latest 1,000+ filings) covers any gap.
    Returns rows added."""
    session = session or requests.Session()
    headers = {"User-Agent": user_agent or config.require_env("SEC_USER_AGENT")}
    limiter = limiter or web.RateLimiter(per_second=8)
    get_json = lambda url: web.get_json(session, url, limiter, headers=headers)
    get_text = lambda url: web.fetch(session, url, limiter, headers=headers).text

    ciks = {row["ticker"]: row["cik_str"] for row in get_json(TICKERS_URL).values()}
    added = 0
    for symbol in symbols:
        if symbol not in ciks:
            print(f"filings: no CIK for {symbol}, skipped", file=sys.stderr)
            continue
        cik = ciks[symbol]
        stored = store.read_partition("filings", symbol, root)
        sub = get_json(SUBMISSIONS_URL.format(f"CIK{cik:010d}.json"))
        blocks = [sub["filings"]["recent"]]
        if stored.empty:
            older = [f for f in sub["filings"]["files"] if f["filingTo"] >= start.isoformat()]
            blocks += [get_json(SUBMISSIONS_URL.format(f["name"])) for f in older]
        df = pd.concat([filings_frame(symbol, cik, b) for b in blocks], ignore_index=True)
        df = df[df["filing_date"] >= start.isoformat()]
        if not stored.empty:
            df = df[~df["accession"].isin(stored["accession"])]
        if df.empty:
            continue
        stamps = [_acceptance(get_text, u, j) for u, j in zip(df["header_url"], df["accepted_json"])]
        df = df.assign(
            available_at=pd.to_datetime([ts for ts, _ in stamps], utc=True),
            accepted_source=[src for _, src in stamps],
        )
        added += store.upsert("filings", df, key=["accession"], partition="symbol", root=root)
    return added
