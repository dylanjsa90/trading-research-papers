import numpy as np
import pandas as pd
import pytest

from tests import market
from trader.config import LlmOverlay
from trader.strategy import features


def test_price_features_match_hand_computed_values():
    p = features.panels(market.prices())
    f = features.price_features(p, core=market.SMALL)
    t, c, spy = p["close"].index[30], p["close"]["AAA"], p["close"]["SPY"]

    assert f["momentum"].loc[t, "AAA"] == pytest.approx(c.iloc[28] / c.iloc[20] - 1)  # skip 2, look back 10
    assert f["trend"].loc[t, "AAA"] == (c.iloc[30] > c.iloc[26:31].mean())
    assert f["vol"].loc[t, "AAA"] == pytest.approx(np.log(c).diff().iloc[26:31].std() * np.sqrt(252))
    assert f["regime"].loc[t] == (spy.iloc[30] > spy.iloc[26:31].mean())
    assert not f["trend"].iloc[:4].any().any()  # no verdict before the window fills


def test_price_features_at_t_ignore_everything_after_t():
    rows = market.prices()
    t = sorted(rows["date"].unique())[40]
    planted = rows.copy()
    planted.loc[planted["date"] > t, ["open", "close", "volume"]] *= 50  # an absurd future

    a = features.price_features(features.panels(rows), core=market.SMALL)
    b = features.price_features(features.panels(planted), core=market.SMALL)

    for name in a:
        pd.testing.assert_frame_equal(pd.DataFrame(a[name].loc[:t]), pd.DataFrame(b[name].loc[:t]), obj=name)


def test_tag_events_go_to_the_first_close_they_precede_and_skip_roundups():
    sessions = pd.DatetimeIndex(["2025-03-03", "2025-03-04", "2025-03-05"])  # EST: the close is 21:00 UTC
    tags = market.tags([
        ("news-1", ["AAA"], 0.5, "low", "2025-03-03 20:59", 1),  # 15:59 New York: today's close
        ("news-2", ["AAA", "ZZZ"], -0.6, "high", "2025-03-03 21:01", 2),  # 16:01: tomorrow's; ZZZ not in universe
        ("news-3", ["BBB"], 0.9, "high", "2025-03-04 12:00", 7),  # a roundup: ignored
        ("news-4", ["BBB"], 0.2, "none", "2025-03-01 15:00", 1),  # a Saturday: Monday's close
        ("news-5", ["AAA"], 0.1, "none", "2025-03-05 22:00", 1),  # after the last close: not yet usable
        ("news-6", ["BBB"], 0.3, "low", "2025-03-05 21:00", 1),  # exactly at the close: usable at that close
    ])  # fmt: skip

    events = features.tag_events(tags, sessions, universe=("AAA", "BBB"))

    got = list(zip(events["doc_id"], events["date"].dt.strftime("%m-%d"), events["symbol"]))
    assert got == [("news-1", "03-03", "AAA"), ("news-4", "03-03", "BBB"), ("news-2", "03-04", "AAA"),
                   ("news-6", "03-05", "BBB")]  # fmt: skip


def test_llm_features_decay_sentiment_and_flag_fresh_negative_surprises():
    sessions = pd.bdate_range("2025-03-03", periods=6)
    events = pd.DataFrame({"date": [sessions[0], sessions[3], sessions[1]], "symbol": ["AAA", "AAA", "CCC"],
                           "sentiment": [0.8, -0.6, -0.9], "surprise": ["low", "med", "low"]})  # fmt: skip
    overlay = LlmOverlay(half_life=1, prior_weight=1, negative_window=2, negative_sentiment=-0.5)

    f = features.llm_features(events, sessions, ["AAA", "BBB", "CCC"], overlay=overlay)

    s = f["sentiment"]["AAA"]
    assert s.iloc[0] == pytest.approx(0.8 / 2)  # one article plus the neutral prior
    assert s.iloc[1] == pytest.approx(0.4 / 1.5)  # halved after one session
    assert s.iloc[3] == pytest.approx((0.8 / 8 - 0.6) / (1 / 8 + 1 + 1))
    assert list(f["negative"]["AAA"]) == [False, False, False, True, True, False]
    assert (f["sentiment"]["BBB"] == 0).all() and not f["negative"]["BBB"].any()
    assert not f["negative"]["CCC"].any()  # very negative but unsurprising: not an event
    assert f["count"]["AAA"].sum() == 2 and f["count"]["CCC"].sum() == 1
