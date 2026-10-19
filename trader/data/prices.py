"""Alpaca daily bars for the universe and SPY -> DATA_DIR/prices/<SYMBOL>.parquet.

Bars are adjusted for splits and dividends (`adjustment=all`), so returns are total returns. Alpaca restates
the whole adjusted history after each corporate action, so every refresh re-reads the last stored session and
re-fetches a symbol from scratch when that bar changed. Adjusted levels before the latest action are not what
was quoted at the time; use them for returns, not for price-level rules.

available_at is 16:00 New York on the session date. On early-close days that is later than the real close,
which is the safe direction.
"""

from collections import defaultdict
from datetime import UTC, date, datetime

import pandas as pd

from trader import config
from trader.data import store

SIP_DELAY = pd.Timedelta(minutes=16)  # the free plan cannot query the latest 15 minutes of SIP data
COLUMNS = ["symbol", "date", "open", "high", "low", "close", "volume", "trade_count", "vwap", "available_at"]


def bars_frame(raw):
    """Normalise alpaca-py's BarSet.df ((symbol, timestamp) index) to one row per symbol-session."""
    if raw.empty:
        return pd.DataFrame(columns=COLUMNS)
    df = raw.reset_index()
    stamp = df["timestamp"].dt.tz_convert("America/New_York")
    session = stamp.dt.normalize()
    if (stamp != session).any():
        raise ValueError("expected daily bars stamped at midnight New York; session dates would be wrong")
    df["date"] = session.dt.strftime("%Y-%m-%d")
    df["available_at"] = (session + pd.Timedelta(hours=16)).dt.tz_convert("UTC")
    return df[COLUMNS]


def alpaca_fetch(symbols, start, end):
    from alpaca.data.enums import Adjustment, DataFeed
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame

    client = StockHistoricalDataClient(*config.alpaca_credentials())
    request = StockBarsRequest(
        symbol_or_symbols=list(symbols),
        timeframe=TimeFrame.Day,
        start=datetime.combine(start, datetime.min.time(), tzinfo=UTC),
        end=end.to_pydatetime(),
        adjustment=Adjustment.ALL,
        feed=DataFeed.SIP,
    )
    return bars_frame(client.get_stock_bars(request).df)


def _restated(new, last):
    overlap = new[(new["symbol"] == last["symbol"]) & (new["date"] == last["date"])]
    return len(overlap) > 0 and abs(overlap["close"].iloc[0] / last["close"] - 1) > 1e-6


def refresh(symbols=(*config.UNIVERSE, config.REGIME_SYMBOL), fetch=alpaca_fetch, root=None,
            start=config.HISTORY_START, now=None):  # fmt: skip
    """Append completed sessions since each symbol's last stored one. Returns rows added."""
    end = pd.Timestamp(now or pd.Timestamp.now(tz="UTC")) - SIP_DELAY
    last = {}
    for symbol in symbols:
        stored = store.read_partition("prices", symbol, root)
        if not stored.empty:
            last[symbol] = stored.iloc[-1]
    groups = defaultdict(list)  # symbols that resume from the same session share one request
    for symbol in symbols:
        groups[date.fromisoformat(last[symbol]["date"]) if symbol in last else start].append(symbol)

    added = 0
    for since, group in groups.items():
        new = fetch(group, since, end)
        restated = [s for s in group if s in last and _restated(new, last[s])]
        if restated:
            new = pd.concat([new[~new["symbol"].isin(restated)], fetch(restated, start, end)], ignore_index=True)
        new = new[new["available_at"] <= end]  # only sessions whose close the fetch window fully covers
        if len(new):
            added += store.upsert("prices", new, key=["symbol", "date"], partition="symbol", root=root)
    return added
