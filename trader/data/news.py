"""Alpaca (Benzinga) news for the universe -> DATA_DIR/news/<YYYY-MM>.parquet, via the v1beta1 REST endpoint
(alpaca-py's data clients target v2, §7).

available_at is created_at. When Benzinga edits an article, the stored row takes the latest text but keeps
created_at, so edited text can be known slightly earlier than it was written; headlines rarely change.
"""

import pandas as pd
import requests

from trader import config
from trader.data import store, web

NEWS_URL = "https://data.alpaca.markets/v1beta1/news"
FIELDS = ["id", "headline", "summary", "content", "author", "url", "source", "symbols", "created_at", "updated_at"]


def articles_frame(items):
    df = pd.DataFrame(items, columns=FIELDS)
    df["symbols"] = [list(dict.fromkeys(config.SYMBOL_ALIASES.get(s, s) for s in syms)) for syms in df["symbols"]]
    df["created_at"] = pd.to_datetime(df["created_at"], utc=True)
    df["updated_at"] = pd.to_datetime(df["updated_at"], utc=True)
    df["available_at"] = df["created_at"]
    df["month"] = df["created_at"].dt.strftime("%Y-%m")
    return df


def _iso(ts):
    return pd.Timestamp(ts).tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ")


def _pages(session, limiter, credentials, params):
    headers = {"APCA-API-KEY-ID": credentials[0], "APCA-API-SECRET-KEY": credentials[1]}
    params = {**params, "limit": 50, "sort": "asc", "include_content": "true"}
    while True:
        page = web.get_json(session, NEWS_URL, limiter, params=params, headers=headers)
        yield page.get("news", [])
        if not page.get("next_page_token"):
            return
        params = {**params, "page_token": page["next_page_token"]}


def refresh(symbols=config.UNIVERSE, root=None, session=None, credentials=None, limiter=None, start=None,
            end=None, flush_every=2000):  # fmt: skip
    """Fetch articles from the newest stored one (or `start`) to `end`, saving every `flush_every` articles so
    an interrupted backfill resumes where it stopped. Symbols that no stored article mentions yet (say, added
    to the universe later) are first backfilled from `start`. Returns articles added."""
    start = pd.Timestamp(start or config.HISTORY_START)
    start = start if start.tzinfo else start.tz_localize("UTC")
    end = pd.Timestamp(end or pd.Timestamp.now(tz="UTC"))
    newest = store.latest("news", "created_at", root)
    if newest is None:
        runs = [(symbols, start, end)]
    else:
        missing = [s for s in symbols if s not in store.list_items("news", "symbols", root)]
        runs = ([(missing, start, newest)] if missing else []) + [(symbols, newest, end)]
    session = session or requests.Session()
    limiter = limiter or web.RateLimiter(per_second=3)
    credentials = credentials or config.alpaca_credentials()

    def save(items):
        return store.upsert("news", articles_frame(items), key=["id"], partition="month", root=root)

    added = 0
    for run_symbols, since, until in runs:
        aliases = [old for old, new in config.SYMBOL_ALIASES.items() if new in run_symbols]
        params = {"symbols": ",".join([*run_symbols, *aliases]), "start": _iso(since), "end": _iso(until)}
        buffer = []
        for items in _pages(session, limiter, credentials, params):
            buffer += items
            if len(buffer) >= flush_every:
                added, buffer = added + save(buffer), []
        added += save(buffer) if buffer else 0
    return added


def count_day(day, session=None, credentials=None, limiter=None):
    """Articles across all symbols created on one New York calendar day (Alpaca documents 130+ a day)."""
    start = pd.Timestamp(day, tz="America/New_York")
    params = {"start": _iso(start), "end": _iso(start + pd.Timedelta(days=1))}
    pages = _pages(session or requests.Session(), limiter or web.RateLimiter(per_second=3),
                   credentials or config.alpaca_credentials(), params)  # fmt: skip
    return sum(len(items) for items in pages)
