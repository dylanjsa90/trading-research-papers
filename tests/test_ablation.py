from dataclasses import replace

import pandas as pd
import pytest

from tests import market
from trader import ablation
from trader.config import LlmOverlay, RiskLimits

PRICES = market.prices(days=140, symbols=("AAA", "BBB", "CCC", "DDD", "SPY"), seed=3, start="2024-06-03")
SESSIONS = pd.to_datetime(sorted(PRICES["date"].unique()))
CUTOFF = SESSIONS[90]
SETTINGS = ablation.Settings(core=market.SMALL, risk=RiskLimits(max_weight=0.6), universe=("AAA", "BBB", "CCC", "DDD"),
                             warmup=15)  # fmt: skip
ARMS = (ablation.Arm("price-only"), ablation.Arm("llm", "m", "p"), ablation.Arm("llm-anonymised", "m", "p+anon"))
NO_TAGS = market.tags([]).assign(model=[], prompt_version=[])


def _tags(rows):
    return market.tags(rows).assign(model="m", prompt_version="p")


def _run(tmp_path, tags=NO_TAGS, **kwargs):
    return ablation.run(PRICES, tags, kwargs.pop("arms", ARMS), kwargs.pop("settings", SETTINGS), cutoff=CUTOFF,
                        root=tmp_path, **kwargs)  # fmt: skip


def test_table_has_every_arm_and_benchmark_with_pre_and_post_cutoff_columns(tmp_path):
    tags = _tags([("news-1", ["AAA"], 0.6, "low", market.close_time(SESSIONS[100]), 1)])
    report = _run(tmp_path, tags)

    assert list(report.table.index) == ["price-only", "llm", "llm-anonymised", "SPY buy-and-hold",
                                        "equal-weight buy-and-hold", "SPY trend filter"]  # fmt: skip
    assert list(report.table.columns.get_level_values(0).unique()) == ["pre-cutoff", "post-cutoff"]
    assert {"cagr", "sharpe", "sortino", "max_drawdown", "turnover", "large_move_hit_rate", "tag_coverage",
            "llm_gpu_s_per_trade"} <= set(report.table.columns.get_level_values(1))  # fmt: skip
    assert report.table.loc["llm-anonymised"].isna().all() and report.notes["llm-anonymised"] == "no tags"
    assert report.table.loc["llm", ("pre-cutoff", "tag_coverage")] == 0
    assert report.table.loc["llm", ("post-cutoff", "tag_coverage")] > 0
    assert list(report.yearly.columns.get_level_values(0).unique()) == ["2024"]


def test_llm_arm_sells_on_fresh_negative_news_and_otherwise_matches_price_only(tmp_path):
    base = _run(tmp_path).results["price-only"]
    held = base.weights.loc[SESSIONS[100]]
    name = held[held > 0].index[0]
    tags = _tags([("news-1", [name], -0.9, "high", market.close_time(SESSIONS[100]), 1)])

    llm = _run(tmp_path, tags).results["llm"]

    before = base.weights.index[base.weights.index <= SESSIONS[100]]
    pd.testing.assert_frame_equal(llm.weights.loc[before], base.weights.loc[before])
    assert llm.weights.loc[SESSIONS[101], name] == 0 and base.weights.loc[SESSIONS[101], name] > 0


def test_rerun_reproduces_the_table_exactly_and_counts_distinct_configurations(tmp_path):
    tags = _tags([("news-1", ["AAA"], 0.6, "low", market.close_time(SESSIONS[100]), 1)])
    first, again = _run(tmp_path, tags), _run(tmp_path, tags)

    pd.testing.assert_frame_equal(first.table, again.table)
    assert first.configs_tried == again.configs_tried == 2  # price-only and llm; the arm with no tags never ran
    changed = _run(tmp_path, tags, settings=replace(SETTINGS, overlay=LlmOverlay(tilt=0.25)))
    assert changed.configs_tried == 3


def test_a_strategy_that_peeks_ahead_fails_loudly(tmp_path):
    def peeking(prices, tags, arm, settings):
        p, f, llm = ablation.prepare(prices, tags, arm, settings)
        return p, dict(f, momentum=f["close"].shift(-1) / f["close"] - 1), llm  # tomorrow's return

    with pytest.raises(ablation.LookAheadError, match="price-only.*momentum"):
        _run(tmp_path, prepare=peeking)


def test_news_dated_one_session_early_fails_loudly(tmp_path):
    def early(prices, tags, arm, settings):
        moved = tags.assign(available_at=tags["available_at"] - pd.Timedelta(days=1))
        return ablation.prepare(prices, moved, arm, settings)

    with pytest.raises(ablation.LookAheadError, match="llm"):
        _run(
            tmp_path, _tags([("news-1", ["AAA"], 0.6, "low", market.close_time(SESSIONS[100]), 1)]), prepare=early
        )


def test_load_with_a_snapshot_ignores_tags_written_after_it(tmp_path):
    from trader.data import store

    rows = market.prices(days=3, symbols=("AAA",))
    store.upsert("prices", rows, key=["symbol", "date"], partition="symbol", root=tmp_path)
    news = pd.DataFrame({"id": [1, 2], "symbols": [["AAA"], ["AAA", "BBB"]], "month": "2024-01",
                         "available_at": pd.Timestamp("2024-01-02 15:00", tz="UTC")})  # fmt: skip
    store.upsert("news", news, key=["id"], partition="month", root=tmp_path)
    tags = _tags([("news-1", ["AAA"], 0.5, "low", "2024-01-02 15:00", 1), ("news-2", ["AAA"], 0.1, "low",
                  "2024-01-02 15:00", 2)]).drop(columns="n_tickers")  # fmt: skip
    tags["tagged_at"] = pd.to_datetime(["2026-10-19 10:00", "2026-10-19 12:00"], utc=True)
    store.upsert("tags", tags.assign(month="2024-01"), key=["doc_id"], partition="month", root=tmp_path)

    _, got = ablation.load(root=tmp_path, snapshot="2026-10-19 11:00Z")

    assert list(got["doc_id"]) == ["news-1"] and list(got["n_tickers"]) == [1]
