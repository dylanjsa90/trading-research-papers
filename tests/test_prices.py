from datetime import date

import pandas as pd

from trader.data import prices, store


def _alpaca_df(rows):
    """Shaped like alpaca-py's BarSet.df: (symbol, timestamp) index; daily bars stamped at midnight New York."""
    df = pd.DataFrame(rows, columns=["symbol", "timestamp", "close"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.assign(open=df.close, high=df.close, low=df.close, volume=1e6, trade_count=1000, vwap=df.close)
    return df.set_index(["symbol", "timestamp"])


def test_bars_frame_stamps_each_session_at_the_new_york_close():
    raw = _alpaca_df([("AAPL", "2024-07-01T04:00:00Z", 216.75), ("AAPL", "2024-01-02T05:00:00Z", 185.64)])
    got = prices.bars_frame(raw).set_index("date")["available_at"]
    assert got["2024-07-01"] == pd.Timestamp("2024-07-01 20:00", tz="UTC")  # 16:00 EDT
    assert got["2024-01-02"] == pd.Timestamp("2024-01-02 21:00", tz="UTC")  # 16:00 EST


class FakeBars:
    """Stands in for Alpaca: serves bars from a mutable 'truth' table and records each request."""

    def __init__(self, closes):
        self.closes = dict(closes)  # (symbol, "YYYY-MM-DD") -> close
        self.calls = []

    def __call__(self, symbols, start, end):
        self.calls.append((tuple(symbols), start))
        rows = [
            (sym, pd.Timestamp(day, tz="America/New_York").tz_convert("UTC"), close)
            for (sym, day), close in sorted(self.closes.items())
            if sym in symbols and date.fromisoformat(day) >= start
            and pd.Timestamp(day, tz="America/New_York") <= end  # Alpaca returns a bar once its day begins
        ]  # fmt: skip
        return prices.bars_frame(_alpaca_df(rows))


def _stored_closes(root):
    df = store.as_of("prices", pd.Timestamp("2030-01-01", tz="UTC"), root=root)
    return {(s, d): c for s, d, c in zip(df["symbol"], df["date"], df["close"])}


AFTER = lambda day: pd.Timestamp(f"{day} 17:00", tz="America/New_York")


def test_incremental_refresh_resumes_from_last_stored_session(tmp_path):
    fake = FakeBars({("AAPL", "2024-03-04"): 175.0, ("MSFT", "2024-03-04"): 415.0})
    prices.refresh(["AAPL", "MSFT"], fetch=fake, root=tmp_path, start=date(2024, 3, 1), now=AFTER("2024-03-04"))
    fake.closes |= {("AAPL", "2024-03-05"): 170.0, ("MSFT", "2024-03-05"): 403.0}

    added = prices.refresh(["AAPL", "MSFT"], fetch=fake, root=tmp_path, start=date(2024, 3, 1),
                           now=AFTER("2024-03-05"))  # fmt: skip

    assert added == 2
    assert fake.calls[-1] == (("AAPL", "MSFT"), date(2024, 3, 4))  # one request, overlapping the last session
    assert len(_stored_closes(tmp_path)) == 4


def test_restated_history_triggers_a_full_refetch_of_that_symbol(tmp_path):
    fake = FakeBars({("AAPL", "2024-03-04"): 175.0, ("AAPL", "2024-03-05"): 170.0, ("MSFT", "2024-03-05"): 403.0})
    prices.refresh(["AAPL", "MSFT"], fetch=fake, root=tmp_path, start=date(2024, 3, 1), now=AFTER("2024-03-05"))
    # A dividend adjustment rescales all of AAPL's earlier closes; MSFT is untouched.
    fake.closes |= {("AAPL", "2024-03-04"): 174.0, ("AAPL", "2024-03-05"): 169.0, ("AAPL", "2024-03-06"): 171.0}

    prices.refresh(["AAPL", "MSFT"], fetch=fake, root=tmp_path, start=date(2024, 3, 1), now=AFTER("2024-03-06"))

    assert (("AAPL",), date(2024, 3, 1)) in fake.calls
    assert (("MSFT",), date(2024, 3, 1)) not in fake.calls[1:]
    got = _stored_closes(tmp_path)
    assert [got[("AAPL", d)] for d in ("2024-03-04", "2024-03-05", "2024-03-06")] == [174.0, 169.0, 171.0]


def test_session_inside_the_sip_delay_is_not_stored(tmp_path):
    fake = FakeBars({("AAPL", "2024-03-04"): 175.0})
    ten_past_close = pd.Timestamp("2024-03-04 16:10", tz="America/New_York")
    assert prices.refresh(["AAPL"], fetch=fake, root=tmp_path, start=date(2024, 3, 1), now=ten_past_close) == 0
    later = pd.Timestamp("2024-03-04 16:20", tz="America/New_York")
    assert prices.refresh(["AAPL"], fetch=fake, root=tmp_path, start=date(2024, 3, 1), now=later) == 1
