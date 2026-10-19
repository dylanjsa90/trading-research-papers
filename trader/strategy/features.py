"""Features per session and symbol, each computed only from data available at that session's 16:00 close.

Price features are backward-looking rolling windows over adjusted bars, which are fine for returns, momentum
and volatility (§4.2: adjusted levels are not what was quoted). LLM features come from tags, each assigned to
the first close at or after its available_at, so news after 16:00 counts from the next session.
"""

import numpy as np
import pandas as pd

from trader import config


def panels(prices):
    """Stored price rows -> {"open", "close", "volume": DataFrame(session date x symbol)}."""
    df = prices.assign(date=pd.to_datetime(prices["date"]))
    return {
        f: df.pivot(index="date", columns="symbol", values=f).sort_index() for f in ("open", "close", "volume")
    }


def price_features(p, core=config.CORE, regime_symbol=config.REGIME_SYMBOL):
    close = p["close"]
    spy = close[regime_symbol]
    return {
        "close": close,
        "returns": np.log(close).diff(),
        "trend": close > close.rolling(core.trend_window).mean(),
        "momentum": close.shift(core.momentum_skip) / close.shift(core.momentum_window) - 1,
        "vol": np.log(close).diff().rolling(core.vol_window).std() * np.sqrt(252),
        "dollar_volume": (close * p["volume"]).rolling(20).mean(),
        "regime": spy > spy.rolling(core.regime_window).mean(),
    }


def session_closes(sessions):
    return (pd.DatetimeIndex(sessions).tz_localize("America/New_York") + pd.Timedelta(hours=16)).tz_convert("UTC")


def tag_events(tags, sessions, universe=config.UNIVERSE, overlay=config.OVERLAY):
    """Tags -> one row per (tag, universe symbol), dated by the first session close at or after available_at.
    Roundups (many publisher tickers) are dropped; tags after the last close are not usable yet."""
    i = session_closes(sessions).searchsorted(tags["available_at"], side="left")
    usable = (i < len(sessions)) & (tags["n_tickers"] < overlay.roundup_tickers)
    events = tags[usable].assign(date=pd.DatetimeIndex(sessions)[i[usable]]).explode("symbols")
    events = events[events["symbols"].isin(set(universe))].rename(columns={"symbols": "symbol"})
    cols = ["doc_id", "date", "symbol", "sentiment", "surprise", "event_type", "available_at"]
    return events[cols].sort_values(["date", "symbol", "doc_id"], ignore_index=True)


def _daily(events, sessions, symbols, values, aggfunc):
    if events.empty:
        return pd.DataFrame(0.0, index=sessions, columns=symbols)
    table = events.pivot_table(index="date", columns="symbol", values=values, aggfunc=aggfunc)
    return table.reindex(index=sessions, columns=symbols).fillna(0.0)


def llm_features(events, sessions, symbols, overlay=config.OVERLAY):
    """Per session x symbol: exponentially decayed mean sentiment (shrunk toward 0 by a neutral prior), a fresh
    negative-event flag, the count of fresh high-surprise tags, and the tags dated to each session."""
    sessions = pd.DatetimeIndex(sessions)
    total = _daily(events, sessions, symbols, "sentiment", "sum").to_numpy()
    count = _daily(events, sessions, symbols, "sentiment", "count").to_numpy()
    keep = 0.5 ** (1 / overlay.half_life)
    num, den = np.zeros_like(total), np.zeros_like(total)
    for t in range(len(sessions)):
        num[t] = total[t] + (keep * num[t - 1] if t else 0)
        den[t] = count[t] + (keep * den[t - 1] if t else 0)

    def fresh(subset):  # how many such tags in the last negative_window sessions
        return _daily(subset, sessions, symbols, "sentiment", "count").rolling(overlay.negative_window, 1).sum()

    negative = events[
        (events["sentiment"] <= overlay.negative_sentiment) & events["surprise"].isin(["med", "high"])
    ]
    return {
        "sentiment": pd.DataFrame(num / (den + overlay.prior_weight), index=sessions, columns=symbols),
        "negative": fresh(negative) > 0,
        "high_surprise": fresh(events[events["surprise"] == "high"]),
        "count": pd.DataFrame(count, index=sessions, columns=symbols),  # tags per session, for coverage
    }
