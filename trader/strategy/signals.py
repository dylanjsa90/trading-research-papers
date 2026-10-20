"""Deterministic target weights: the price-only core (§12.2) and the LLM overlay (§3.3), decided at close(t).

LLM features only ever remove or reweight names the core already chose; they never add a name or raise the
book's gross exposure.
"""

import numpy as np
import pandas as pd

from trader import config


def core_weights(f, t, core=config.CORE, risk=config.RISK, universe=config.UNIVERSE):
    """Trend- and regime-filtered, momentum-ranked, inverse-vol weights scaled toward the target volatility."""
    names = [s for s in universe if s in f["close"].columns]
    if not f["regime"].get(t, False):
        return pd.Series(0.0, index=names)
    row = {k: f[k].loc[t, names] for k in ("trend", "close", "dollar_volume", "momentum", "vol")}
    ok = (row["trend"].astype(bool) & (row["close"] >= risk.min_price)
          & (row["dollar_volume"] >= risk.min_dollar_volume) & row["momentum"].notna() & (row["vol"] > 0))  # fmt: skip
    picks = row["momentum"][ok].nlargest(core.max_names).index
    weights = pd.Series(0.0, index=names)
    if len(picks) == 0:
        return weights
    w = (1 / row["vol"][picks]) / (1 / row["vol"][picks]).sum()
    recent = f["returns"].loc[:t, picks].tail(core.vol_window)
    book_vol = float(np.sqrt(w @ recent.cov() @ w * 252))
    scale = min(risk.max_gross, core.target_vol / book_vol) if book_vol > 0 else risk.max_gross
    weights[picks] = (w * scale).clip(upper=risk.max_weight)
    return weights


def apply_llm(weights, llm, t, overlay=config.OVERLAY, risk=config.RISK):
    """Filter (names with a fresh negative event go to cash), then tilt by decayed sentiment at the same gross."""
    w = weights.copy()
    held = w.index[w > 0]
    w[held[llm["negative"].loc[t, held].to_numpy(bool)]] = 0.0
    gross = w.sum()
    if overlay.tilt and gross > 0:
        tilted = w * (1 + overlay.tilt * llm["sentiment"].loc[t, w.index]).clip(lower=0)
        if tilted.sum() > 0:
            w = (tilted * gross / tilted.sum()).clip(upper=risk.max_weight)
    return w


def daily_exits(held, f, llm, t):
    """Between rebalances: everything to cash when SPY's regime turns off; with LLM features, also sell names
    with a fresh negative event. Otherwise hold as is."""
    if not f["regime"].get(t, False):
        return held * 0.0
    if llm is None:
        return held
    out = held.copy()
    names = held.index[held > 0]
    out[names[llm["negative"].loc[t, names].to_numpy(bool)]] = 0.0
    return out
