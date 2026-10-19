"""The ablation ladder (§8.2): the price-only core, then each LLM layer on top, beside three benchmarks, with
pre- and post-cutoff columns and a per-year walk-forward table.

Parameters are fixed in config before any run and never fitted, so each later period is out of sample. Every
run is checked for look-ahead by planting an absurd future and requiring every feature and decision up to a
rebalance before the cutoff to be unchanged; a leak raises LookAheadError. Every configuration run is logged
to backtest_runs.jsonl, so the number tried is reported beside the results (§8.2.8).

    uv run python -m trader.ablation          # needs data/prices, data/tags and data/news
"""

import hashlib
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from trader import backtest, config
from trader.data import store
from trader.llm import tagger
from trader.strategy import features, signals


class LookAheadError(RuntimeError):
    pass


@dataclass(frozen=True)
class Arm:
    name: str
    model: str | None = None  # tags from this model and prompt version feed the LLM overlay
    prompt_version: str | None = None


ARMS = (
    Arm("price-only"),
    Arm("llm", config.TAGGER_MODEL, tagger.NEWS.prompt_version),
    Arm("llm-anonymised", config.TAGGER_MODEL, tagger.NEWS.prompt_version + "+anon"),
)  # FinBERT (the free baseline) joins when llm/finbert.py exists


@dataclass(frozen=True)
class Settings:
    core: config.CoreStrategy = config.CORE
    risk: config.RiskLimits = config.RISK
    overlay: config.LlmOverlay = config.OVERLAY
    costs: config.Costs = config.COSTS
    universe: tuple = config.UNIVERSE
    regime_symbol: str = config.REGIME_SYMBOL
    warmup: int = 252  # sessions of history before the first decision (the 200-day windows need ~1 year)


DEFAULTS = Settings()


@dataclass
class Report:
    table: pd.DataFrame  # strategy x (period, metric)
    yearly: pd.DataFrame  # strategy x (year, metric): the walk-forward view
    configs_tried: int
    notes: dict = field(default_factory=dict)
    results: dict = field(default_factory=dict)


def _arm_tags(tags, arm):
    return tags[(tags["model"] == arm.model) & (tags["prompt_version"] == arm.prompt_version)]


def prepare(prices, tags, arm, s):
    """(price panels, price features, LLM features or None) for one arm."""
    p = features.panels(prices)
    f = features.price_features(p, s.core, s.regime_symbol)
    if arm.model is None:
        return p, f, None
    sessions = p["close"].index
    events = features.tag_events(_arm_tags(tags, arm), sessions, s.universe, s.overlay)
    return p, f, features.llm_features(events, sessions, list(s.universe), s.overlay)


def strategy(f, llm, s):
    names = [n for n in s.universe if n in f["close"].columns]

    def decide(t, held, rebalance):
        held = held.reindex(names, fill_value=0.0)
        if rebalance or (held.sum() == 0 and f["regime"].get(t, False)):  # re-enter as soon as the regime allows
            w = signals.core_weights(f, t, s.core, s.risk, names)
            return w if llm is None else signals.apply_llm(w, llm, t, s.overlay, s.risk)
        return signals.daily_exits(held, f, llm, t)

    return decide


def _benchmarks(f, s):
    def buy_and_hold(weights):
        def decide(t, held, rebalance):
            return weights if held.sum() == 0 else None

        return decide

    def trend_filter(t, held, rebalance):
        on = bool(f["regime"].get(t, False))
        return None if on == (held.get(s.regime_symbol, 0) > 0) else pd.Series({s.regime_symbol: float(on)})

    names = [n for n in s.universe if n in f["close"].columns]
    return {
        "SPY buy-and-hold": buy_and_hold(pd.Series({s.regime_symbol: 1.0})),
        "equal-weight buy-and-hold": buy_and_hold(pd.Series(1 / len(names), index=names)),
        "SPY trend filter": trend_filter,
    }


def _same(a, b):
    if a is None or b is None:
        return a is b
    return a.equals(b)


def _leak_check(prices, tags, arm, s, prepare, f, llm, result, sessions, at):
    """Plant an absurd future after `at` and require every feature and decision up to close(at) to be unchanged."""
    later = pd.to_datetime(prices["date"]) > at
    planted = prices.copy()
    planted.loc[later, ["open", "close", "volume"]] *= 50
    fake = pd.DataFrame({"doc_id": [f"planted-{n}" for n in s.universe], "model": arm.model,
                         "prompt_version": arm.prompt_version, "symbols": [[n] for n in s.universe],
                         "sentiment": -1.0, "surprise": "high", "event_type": "other", "n_tickers": 1, "seconds": 0.0,
                         "available_at": features.session_closes([at])[0] + pd.Timedelta(minutes=1)})  # fmt: skip
    _, f2, llm2 = prepare(planted, pd.concat([tags, fake], ignore_index=True), arm, s)
    for kind, clean, dirty in (("feature", f, f2), ("LLM feature", llm or {}, llm2 or {})):
        for key in clean:
            if not _same(clean[key].loc[:at], dirty[key].loc[:at]):
                raise LookAheadError(f"{arm.name}: {kind} {key!r} at or before {at:%Y-%m-%d} changed when only "
                                     "later data changed")  # fmt: skip
    upto = [t for t in sessions if t <= at]
    p2 = features.panels(planted)
    rerun = backtest.simulate(p2, strategy(f2, llm2, s), upto, s.costs, s.core.rebalance_every)
    for t in upto:
        if not _same(result.targets[t], rerun.targets[t]):
            raise LookAheadError(f"{arm.name}: the decision at {t:%Y-%m-%d} changed when only data after "
                                 f"{at:%Y-%m-%d} changed")  # fmt: skip


def _log(root, arm, s, cutoff, summary):
    """Append this configuration; return how many distinct configurations have ever been run."""
    cfg = {"arm": asdict(arm), "core": asdict(s.core), "risk": asdict(s.risk), "costs": asdict(s.costs),
           "overlay": asdict(s.overlay) if arm.model else None, "universe": list(s.universe),
           "warmup": s.warmup, "cutoff": f"{cutoff:%Y-%m-%d}"}  # fmt: skip
    digest = hashlib.sha1(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:12]
    path = Path(root or config.DATA_DIR) / "backtest_runs.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as out:
        out.write(json.dumps({"logged_at": pd.Timestamp.now(tz="UTC").isoformat(), "config_hash": digest,
                              "config": cfg, "summary": summary}) + "\n")  # fmt: skip
    return len({json.loads(line)["config_hash"] for line in path.read_text().splitlines()})


def _row(result, f, llm, tags, arm, start, end):
    m = backtest.metrics(result, f["close"], f["vol"], start, end)
    if llm is None:
        return m | {"tag_coverage": np.nan, "llm_gpu_s_per_trade": np.nan}
    covered = (llm["count"].loc[start:end].sum(axis=1) > 0).mean()
    when = _arm_tags(tags, arm)["available_at"].dt.tz_convert("America/New_York").dt.tz_localize(None)
    within = (when >= start) & (when < end + pd.Timedelta(days=1))
    trades = result.trades.loc[start:end].sum()
    seconds = _arm_tags(tags, arm).loc[within, "seconds"].sum()
    return m | {"tag_coverage": covered, "llm_gpu_s_per_trade": seconds / trades if trades else np.nan}


def run(prices, tags, arms=ARMS, settings=DEFAULTS, cutoff=None, root=None, prepare=prepare):
    s = settings
    cutoff = pd.Timestamp(cutoff or config.MODEL_CUTOFFS[config.TAGGER_MODEL])
    sessions = list(pd.to_datetime(sorted(prices["date"].unique()))[s.warmup :])
    periods = {"pre-cutoff": (sessions[0], cutoff), "post-cutoff": (cutoff + pd.Timedelta(days=1), sessions[-1])}
    rebalances = [t for i, t in enumerate(sessions) if i % s.core.rebalance_every == 0 and t <= cutoff]
    at = rebalances[-1] if rebalances else sessions[len(sessions) // 2]
    years = sorted({t.year for t in sessions})
    rows, yearly, notes, results, tried = {}, {}, {}, {}, 0

    def record(name, result, f, llm, arm):
        rows[name] = {
            (p, k): v for p, (a, b) in periods.items() for k, v in _row(result, f, llm, tags, arm, a, b).items()
        }
        yearly[name] = {(str(y), k): v for y in years
                        for k, v in backtest.metrics(result, start=f"{y}-01-01", end=f"{y}-12-31").items()
                        if k in ("cagr", "sharpe", "max_drawdown")}  # fmt: skip
        results[name] = result

    for arm in arms:
        if arm.model and _arm_tags(tags, arm).empty:
            notes[arm.name], rows[arm.name], yearly[arm.name] = "no tags", {}, {}
            continue
        p, f, llm = prepare(prices, tags, arm, s)
        result = backtest.simulate(p, strategy(f, llm, s), sessions, s.costs, s.core.rebalance_every)
        _leak_check(prices, tags, arm, s, prepare, f, llm, result, sessions, at)
        record(arm.name, result, f, llm, arm)
        post = rows[arm.name]
        tried = _log(root, arm, s, cutoff, {k: post[("post-cutoff", k)] for k in ("sharpe", "cagr")})
    p, f, _ = prepare(prices, tags, Arm("benchmarks"), s)
    for name, decide in _benchmarks(f, s).items():
        record(name, backtest.simulate(p, decide, sessions, s.costs, s.core.rebalance_every), f, None, None)
    order = list(rows)  # arms in ladder order (an arm with no tags keeps an empty row), then benchmarks

    def frame(by_name):
        df = pd.DataFrame.from_dict(by_name, orient="index").reindex(order)
        df.columns = pd.MultiIndex.from_tuples(df.columns)
        return df

    return Report(frame(rows), frame(yearly), tried, notes, results)


def load(root=None, now=None):
    """Prices, and tags joined with each article's publisher ticker count, as stored now."""
    now = pd.Timestamp(now or pd.Timestamp.now(tz="UTC"))
    prices = store.as_of("prices", now, root=root)
    cols = ["doc_id", "model", "prompt_version", "symbols", "sentiment", "surprise", "event_type", "seconds",
            "available_at"]  # fmt: skip
    tags = store.as_of("tags", now, root=root, columns=cols)
    news = store.as_of("news", now, root=root, columns=["id", "symbols"])
    counts = pd.DataFrame({"doc_id": "news-" + news["id"].astype(str), "n_tickers": news["symbols"].map(len)})
    return prices, tags.merge(counts, on="doc_id", how="left").fillna({"n_tickers": 1})


def main():
    report = run(*load())
    out = config.DATA_DIR / "backtest"
    out.mkdir(parents=True, exist_ok=True)
    report.table.to_csv(out / "ablation.csv")
    report.yearly.to_csv(out / "walk_forward.csv")
    with pd.option_context("display.width", 200, "display.max_columns", 30, "display.precision", 3):
        for period in report.table.columns.get_level_values(0).unique():
            print(f"\n{period}\n{report.table[period]}")
    print(f"\nnotes: {report.notes}\nconfigurations tried so far: {report.configs_tried}\nwritten to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
