"""Daily target-weight backtester (§8.2): decide at close(t) from data available then, fill at open(t+1).

Prices are split- and dividend-adjusted, so NAV is a total-return series. Costs are charged per side on traded
notional. A target equal to a name's held weight means "hold the shares", so overnight drift is not traded.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from trader import config


@dataclass
class Result:
    nav: pd.Series  # at each session's close
    weights: pd.DataFrame  # held at each close
    targets: dict  # decision session -> target weights decided at that close
    turnover: pd.Series  # one-way traded notional / NAV, indexed by fill session
    costs: pd.Series  # paid at each fill, in NAV units
    trades: pd.Series = None  # names traded at each fill


def _value(shares, price, t):
    held = shares[shares != 0]
    if price[held.index].isna().any():
        raise ValueError(f"no price on {t:%Y-%m-%d} for held {list(held.index[price[held.index].isna()])}")
    return (held * price[held.index]).reindex(shares.index, fill_value=0.0)


def simulate(p, decide, sessions, costs=config.COSTS, rebalance_every=config.CORE.rebalance_every, nav0=1.0):
    """Run `decide(t, held_weights, rebalance)` at every close in `sessions`; it returns target weights (or None
    to hold). Each target fills at the next session's open; the last session decides but does not fill."""
    opens, closes = p["open"], p["close"]
    sessions = list(sessions)
    shares, cash = pd.Series(0.0, index=closes.columns), nav0
    nav, weights, targets, turnover, paid, trades = {}, {}, {}, {}, {}, {}
    for i, t in enumerate(sessions):
        value = _value(shares, closes.loc[t], t)
        nav[t] = cash + value.sum()
        weights[t] = held = value / nav[t]
        targets[t] = target = decide(t, held, i % rebalance_every == 0)
        if i + 1 == len(sessions):
            break
        fill = sessions[i + 1]
        price = opens.loc[fill]
        nav_open = cash + _value(shares, price, fill).sum()
        target = held if target is None else target.reindex(shares.index, fill_value=0.0)
        keep = (target - held).abs() < 1e-12
        new = (target * nav_open / price).where(~keep, shares).fillna(0.0)
        changed = (new - shares).abs()
        traded = (changed * price).fillna(0.0).sum()
        cost = traded * costs.one_way_bps / 1e4
        cash = nav_open - _value(new, price, fill).sum() - cost
        turnover[fill], paid[fill], trades[fill] = traded / nav_open, cost, int((changed > 0).sum())
        shares = new
    return Result(pd.Series(nav), pd.DataFrame(weights).T, targets, pd.Series(turnover, dtype=float),
                  pd.Series(paid, dtype=float), pd.Series(trades, dtype=int))  # fmt: skip


def _hit_rate(weights, close, vol, start, end):
    """Of the large moves (|return| >= 2 daily sigmas) in names held through the day, the share that were up."""
    names = weights.columns.intersection(close.columns)
    held = weights[names].reindex(close.index).fillna(0.0) > 0
    through = held & held.shift(1, fill_value=False)
    move = close[names].pct_change()
    large = (through & (move.abs() >= 2 * vol[names].shift(1) / np.sqrt(252))).loc[start:end]
    n = large.to_numpy().sum()
    return float((large & (move.loc[start:end] > 0)).to_numpy().sum() / n) if n else np.nan


def metrics(result, close=None, vol=None, start=None, end=None):
    nav = result.nav.loc[start:end]
    r = nav.pct_change().dropna()
    n = len(r)
    downside = float(np.sqrt(np.mean(np.minimum(r, 0) ** 2))) if n else 0.0
    return {
        "cagr": (nav.iloc[-1] / nav.iloc[0]) ** (252 / n) - 1 if n else np.nan,
        "sharpe": r.mean() / r.std() * np.sqrt(252) if n > 1 and r.std() > 0 else np.nan,
        "sortino": r.mean() / downside * np.sqrt(252) if downside > 0 else np.nan,
        "max_drawdown": float((1 - nav / nav.cummax()).max()) if n else np.nan,
        "turnover": result.turnover.loc[start:end].sum() * 252 / n if n else np.nan,
        "large_move_hit_rate": _hit_rate(result.weights, close, vol, start, end) if close is not None else np.nan,
    }
