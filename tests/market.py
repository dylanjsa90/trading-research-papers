"""Synthetic prices and tags for strategy and backtest tests."""

import numpy as np
import pandas as pd

from trader.config import CoreStrategy

SMALL = CoreStrategy(trend_window=5, momentum_window=10, momentum_skip=2, max_names=2, vol_window=5,
                     target_vol=0.15, regime_window=5, rebalance_every=5)  # fmt: skip


def close_time(day):
    return pd.Timestamp(day).tz_localize("America/New_York") + pd.Timedelta(hours=16)


def prices(days=60, symbols=("AAA", "BBB", "CCC", "SPY"), seed=0, start="2024-01-02"):
    """Daily bars as stored in the prices table: one row per symbol-session, available at the 16:00 close."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, periods=days)
    rows = []
    for i, symbol in enumerate(symbols):
        close = 100 * np.exp(np.cumsum(rng.normal(0.002 * (i - 1), 0.02, days)))
        opens = close * (1 + rng.normal(0, 0.005, days))
        rows += [{"symbol": symbol, "date": d.strftime("%Y-%m-%d"), "open": o, "close": c, "volume": 1e7,
                  "available_at": close_time(d).tz_convert("UTC")} for d, o, c in zip(dates, opens, close)]  # fmt: skip
    return pd.DataFrame(rows)


def tags(rows):
    """Tag rows from (doc_id, symbols, sentiment, surprise, available_at, n_tickers) tuples."""
    df = pd.DataFrame(rows, columns=["doc_id", "symbols", "sentiment", "surprise", "available_at", "n_tickers"])
    df["available_at"] = pd.to_datetime(df["available_at"], utc=True)
    return df.assign(event_type="other", seconds=1.0)
