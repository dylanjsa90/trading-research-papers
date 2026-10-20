import numpy as np
import pandas as pd
import pytest

from trader.config import CoreStrategy, LlmOverlay, RiskLimits
from trader.strategy import signals

DAYS = pd.bdate_range("2025-03-03", periods=6)
T = DAYS[-1]
NAMES = ["AAA", "BBB", "CCC", "DDD"]
CORE = CoreStrategy(max_names=2, vol_window=5, target_vol=0.15)
LOOSE = RiskLimits(max_weight=1.0, min_price=5, min_dollar_volume=1e6)


def _panel(value):
    """A constant panel; `value` is one number for every name or a {name: number} dict."""
    row = value if isinstance(value, dict) else dict.fromkeys(NAMES, value)
    return pd.DataFrame([row] * len(DAYS), index=DAYS, columns=NAMES)


def _features(momentum, vol=0.2, trend=True, regime=True, close=100.0, dollar_volume=1e9, returns=None):
    if returns is None:
        returns = pd.DataFrame(np.random.default_rng(0).normal(0, 0.01, (len(DAYS), len(NAMES))), index=DAYS,
                               columns=NAMES)  # fmt: skip
    return {
        "momentum": _panel(momentum),
        "vol": _panel(vol),
        "trend": _panel(trend),
        "close": _panel(close),
        "dollar_volume": _panel(dollar_volume),
        "returns": returns,
        "regime": pd.Series(regime, index=DAYS),
    }


def test_core_holds_cash_when_spy_is_below_its_trend():
    f = _features({"AAA": 0.3, "BBB": 0.2, "CCC": 0.1, "DDD": 0.0}, regime=False)
    assert signals.core_weights(f, T, CORE, LOOSE, NAMES).sum() == 0


def test_core_picks_the_top_momentum_names_that_pass_trend_price_and_liquidity():
    f = _features({"AAA": 0.5, "BBB": 0.4, "CCC": 0.3, "DDD": 0.2},
                  trend={"AAA": False, "BBB": True, "CCC": True, "DDD": True},
                  close={"AAA": 100, "BBB": 3, "CCC": 100, "DDD": 100})  # fmt: skip
    w = signals.core_weights(f, T, CORE, LOOSE, NAMES)
    assert sorted(w[w > 0].index) == ["CCC", "DDD"]  # AAA below trend, BBB under the $5 floor


def test_core_weights_are_inverse_vol_scaled_to_the_target_and_capped():
    pattern = [0.02, -0.02, 0.02, -0.02, 0.02, -0.02]
    rets = pd.DataFrame({n: pattern for n in NAMES}, index=DAYS)  # perfectly correlated
    f = _features({"AAA": 0.5, "BBB": 0.4, "CCC": 0.0, "DDD": 0.0}, vol={"AAA": 0.1, "BBB": 0.3, "CCC": 1, "DDD": 1},
                  returns=rets)  # fmt: skip
    est = np.std(pattern[-5:], ddof=1) * np.sqrt(252)  # the book's vol: correlated names, weights summing to 1

    w = signals.core_weights(f, T, CORE, LOOSE, NAMES)

    assert w["AAA"] / w["BBB"] == pytest.approx(3)  # 1/0.1 : 1/0.3
    assert w.sum() == pytest.approx(0.15 / est)
    capped = signals.core_weights(
        f, T, CORE, RiskLimits(max_weight=0.2, min_price=5, min_dollar_volume=1e6), NAMES
    )
    assert capped["AAA"] == pytest.approx(0.2) and capped["BBB"] == pytest.approx(w["BBB"])  # no redistribution


def _llm(sentiment, negative=()):
    return {
        "sentiment": _panel({n: sentiment.get(n, 0.0) for n in NAMES}),
        "negative": _panel({n: n in negative for n in NAMES}),
    }


def test_llm_filter_drops_names_with_fresh_negative_events_to_cash():
    w = pd.Series({"AAA": 0.4, "BBB": 0.4})
    got = signals.apply_llm(w, _llm({}, negative={"AAA"}), T, LlmOverlay(tilt=0), LOOSE)
    assert got.to_dict() == {"AAA": 0.0, "BBB": 0.4}


def test_llm_tilt_shifts_weight_toward_positive_sentiment_at_the_same_gross():
    w = pd.Series({"AAA": 0.4, "BBB": 0.4})
    got = signals.apply_llm(w, _llm({"AAA": 0.5, "BBB": -0.5}), T, LlmOverlay(tilt=0.5), LOOSE)
    assert got["AAA"] == pytest.approx(0.5) and got["BBB"] == pytest.approx(0.3)
    lopsided = signals.apply_llm(
        w, _llm({"AAA": 0.5}), T, LlmOverlay(tilt=0.5), LOOSE
    )  # 0.5 : 0.4 before rescaling
    assert lopsided.sum() == pytest.approx(0.8) and lopsided["AAA"] / lopsided["BBB"] == pytest.approx(1.25)
    capped = signals.apply_llm(w, _llm({"AAA": 0.5, "BBB": -0.5}), T, LlmOverlay(tilt=0.5),
                               RiskLimits(max_weight=0.45))  # fmt: skip
    assert capped["AAA"] == pytest.approx(0.45) and capped["BBB"] == pytest.approx(0.3)


def test_daily_exits_sell_on_regime_off_or_fresh_negative_news_and_otherwise_hold():
    held = pd.Series({"AAA": 0.3, "BBB": 0.5})
    f = _features(0.0)
    assert signals.daily_exits(held, f, None, T).equals(held)
    assert signals.daily_exits(held, f, _llm({}, negative={"BBB"}), T).to_dict() == {"AAA": 0.3, "BBB": 0.0}
    f_off = _features(0.0, regime=False)
    assert signals.daily_exits(held, f_off, None, T).sum() == 0
