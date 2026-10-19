import numpy as np
import pandas as pd
import pytest

from trader import backtest
from trader.config import Costs

FREE = Costs(commission_bps=0, half_spread_bps=0, slippage_bps=0)
DAYS = pd.bdate_range("2025-03-03", periods=4)


def _panels(opens, closes):
    return {"open": pd.DataFrame({"AAA": opens}, index=DAYS[: len(opens)]),
            "close": pd.DataFrame({"AAA": closes}, index=DAYS[: len(closes)])}  # fmt: skip


def _buy_once(t, held, rebalance):
    return pd.Series({"AAA": 1.0}) if t == DAYS[0] else held


def test_targets_decided_at_the_close_fill_at_the_next_open():
    p = _panels([99, 110, 121, 121], [100, 121, 121, 121])
    r = backtest.simulate(p, _buy_once, DAYS, costs=FREE)
    assert r.nav.iloc[0] == 1.0
    assert r.nav.iloc[1] == pytest.approx(121 / 110)  # bought at 110 (the open), not 100 (the decision close)


def test_costs_are_charged_on_traded_notional():
    p = _panels([99, 110, 121, 121], [100, 121, 121, 121])
    r = backtest.simulate(p, _buy_once, DAYS, costs=Costs(commission_bps=10, half_spread_bps=0, slippage_bps=0))
    assert r.nav.iloc[1] == pytest.approx(121 / 110 - 0.001)
    assert r.costs.sum() == pytest.approx(0.001) and r.turnover.iloc[0] == pytest.approx(1.0)
    assert list(r.trades) == [1, 0, 0]  # names traded at each fill


def test_a_target_equal_to_the_held_weight_holds_the_shares():
    p = _panels([99, 110, 130, 90], [100, 121, 125, 100])

    def half_then_hold(t, held, rebalance):
        return pd.Series({"AAA": 0.5}) if t == DAYS[0] else held

    r = backtest.simulate(p, half_then_hold, DAYS, costs=FREE)

    assert r.turnover.iloc[1:].sum() == 0  # overnight drift is not rebalanced away
    assert r.nav.iloc[3] == pytest.approx(0.5 + 0.5 / 110 * 100)


def test_rebalance_flag_follows_the_cadence_and_the_last_close_decides_without_filling():
    days = pd.bdate_range("2025-03-03", periods=7)
    p = {"open": pd.DataFrame({"AAA": 100.0}, index=days), "close": pd.DataFrame({"AAA": 100.0}, index=days)}
    flags = []

    def decide(t, held, rebalance):
        flags.append(rebalance)
        return pd.Series({"AAA": 0.5})

    r = backtest.simulate(p, decide, days, costs=FREE, rebalance_every=3)

    assert flags == [True, False, False, True, False, False, True]
    assert len(r.targets) == 7 and len(r.turnover) == 6


def test_metrics_on_a_known_nav():
    nav = pd.Series([1.0, 1.1, 0.99, 1.089], index=DAYS)
    result = backtest.Result(nav=nav, weights=pd.DataFrame(index=DAYS), targets={},
                             turnover=pd.Series([0.5, 0.0, 0.5], index=DAYS[1:]), costs=pd.Series(dtype=float))  # fmt: skip
    r = np.array([0.1, -0.1, 0.1])

    m = backtest.metrics(result)

    assert m["max_drawdown"] == pytest.approx(0.1)
    assert m["sharpe"] == pytest.approx(r.mean() / r.std(ddof=1) * np.sqrt(252))
    assert m["sortino"] == pytest.approx(r.mean() / np.sqrt(np.mean(np.minimum(r, 0) ** 2)) * np.sqrt(252))
    assert m["cagr"] == pytest.approx(1.089 ** (252 / 3) - 1)
    assert m["turnover"] == pytest.approx(1.0 * 252 / 3)  # one-way, annualised


def test_large_move_hit_rate_is_the_share_of_held_large_moves_that_were_up():
    days = pd.bdate_range("2025-03-03", periods=5)
    close = pd.DataFrame({"AAA": [100, 100, 110, 99, 100]}, index=days)  # +10%, -10%, +1%
    vol = pd.DataFrame({"AAA": 0.02 * np.sqrt(252)}, index=days)  # 2% a day: large means 4% or more
    weights = pd.DataFrame({"AAA": [0.5, 0.5, 0.5, 0.5, 0.5]}, index=days)
    result = backtest.Result(nav=pd.Series(1.0, index=days), weights=weights, targets={},
                             turnover=pd.Series(dtype=float), costs=pd.Series(dtype=float))  # fmt: skip
    assert backtest.metrics(result, close=close, vol=vol)["large_move_hit_rate"] == pytest.approx(0.5)
    bought_late = result.weights.assign(AAA=[0.5, 0.0, 0.5, 0.5, 0.5])  # not held into the +10% day
    result.weights = bought_late
    assert backtest.metrics(result, close=close, vol=vol)["large_move_hit_rate"] == pytest.approx(0.0)
