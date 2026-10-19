import pandas as pd
import pytest
import requests

from tests.fakes import FakeClock, FakeSession, response
from trader.data import news, store, web


def _article(id_, created, symbols=("AAPL",), updated=None):
    return {
        "id": id_, "headline": f"h{id_}", "summary": "s", "content": "<p>c</p>", "author": "Benzinga Newsdesk",
        "url": f"https://benzinga.example/{id_}", "source": "benzinga", "symbols": list(symbols),
        "created_at": created, "updated_at": updated or created,
    }  # fmt: skip


def test_articles_frame_is_available_at_creation_and_maps_old_tickers():
    df = news.articles_frame([_article(1, "2021-05-03T13:30:05Z", symbols=["FB", "AAPL", "META"])])
    row = df.iloc[0]
    assert row["available_at"] == pd.Timestamp("2021-05-03 13:30:05", tz="UTC")
    assert list(row["symbols"]) == ["META", "AAPL"]
    assert row["month"] == "2021-05"


def _session(pages):
    return FakeSession({news.NEWS_URL: [response(200, p) for p in pages]})


def _refresh(tmp_path, session, **kwargs):
    clock = FakeClock()
    limiter = web.RateLimiter(1000, clock=clock, sleep=clock.sleep)
    return news.refresh(symbols=["AAPL", "META"], root=tmp_path, session=session, credentials=("PKID", "SECRET"),
                        limiter=limiter, start=pd.Timestamp("2024-01-01", tz="UTC"),
                        end=pd.Timestamp("2024-02-01", tz="UTC"), **kwargs)  # fmt: skip


def _stored_ids(root):
    return sorted(store.as_of("news", pd.Timestamp("2030-01-01", tz="UTC"), root=root)["id"])


def test_refresh_follows_pages_until_the_token_runs_out(tmp_path):
    session = _session(
        [
            {
                "news": [_article(1, "2024-01-02T10:00:00Z"), _article(2, "2024-01-03T10:00:00Z")],
                "next_page_token": "t1",
            },
            {"news": [_article(3, "2024-01-04T10:00:00Z")], "next_page_token": None},
        ]
    )
    assert _refresh(tmp_path, session) == 3
    first, second = session.calls
    assert (first["params"]["start"], first["params"]["end"]) == ("2024-01-01T00:00:00Z", "2024-02-01T00:00:00Z")
    assert set(first["params"]["symbols"].split(",")) == {"AAPL", "META", "FB"}
    assert first["params"]["include_content"] == "true" and first["params"]["sort"] == "asc"
    assert first["headers"] == {"APCA-API-KEY-ID": "PKID", "APCA-API-SECRET-KEY": "SECRET"}
    assert second["params"]["page_token"] == "t1"
    assert _stored_ids(tmp_path) == [1, 2, 3]


def test_interrupted_backfill_keeps_the_pages_already_fetched(tmp_path):
    session = FakeSession(
        {
            news.NEWS_URL: [
                response(200, {"news": [_article(1, "2024-01-02T10:00:00Z")], "next_page_token": "t1"}),
                response(403, {"message": "forbidden"}),
            ]
        }
    )
    with pytest.raises(requests.HTTPError):
        _refresh(tmp_path, session, flush_every=1)
    assert _stored_ids(tmp_path) == [1]


def test_rerun_resumes_from_newest_stored_article_without_duplicates(tmp_path):
    both = ["AAPL", "META"]  # every refreshed symbol already has coverage, so no backfill pass
    first = [{"news": [_article(1, "2024-01-02T10:00:00Z", both), _article(2, "2024-01-03T10:00:00Z", both)]}]
    _refresh(tmp_path, _session(first))
    again = _session([{"news": [_article(2, "2024-01-03T10:00:00Z", both), _article(3, "2024-01-05T10:00:00Z")]}])

    assert _refresh(tmp_path, again) == 1
    assert len(again.calls) == 1
    assert again.calls[0]["params"]["start"] == "2024-01-03T10:00:00Z"
    assert _stored_ids(tmp_path) == [1, 2, 3]


def test_count_day_counts_every_article_on_that_new_york_day():
    session = _session(
        [
            {"news": [_article(i, "2026-10-15T14:00:00Z") for i in range(50)], "next_page_token": "t1"},
            {"news": [_article(i, "2026-10-15T20:00:00Z") for i in range(50, 81)], "next_page_token": None},
        ]
    )
    clock = FakeClock()
    n = news.count_day("2026-10-15", session=session, credentials=("PKID", "SECRET"),
                       limiter=web.RateLimiter(1000, clock=clock, sleep=clock.sleep))  # fmt: skip
    assert n == 81
    params = session.calls[0]["params"]
    assert "symbols" not in params
    assert (params["start"], params["end"]) == ("2026-10-15T04:00:00Z", "2026-10-16T04:00:00Z")  # EDT midnights


def test_symbol_added_later_gets_its_history_backfilled(tmp_path):
    _refresh_symbols = lambda session, symbols: news.refresh(
        symbols=symbols, root=tmp_path, session=session, credentials=("PKID", "SECRET"),
        limiter=web.RateLimiter(1000, clock=FakeClock(), sleep=lambda s: None),
        start=pd.Timestamp("2024-01-01", tz="UTC"), end=pd.Timestamp("2024-02-01", tz="UTC"),
    )  # fmt: skip
    _refresh_symbols(_session([{"news": [_article(1, "2024-01-10T10:00:00Z", symbols=["AAPL"])]}]), ["AAPL"])
    session = _session([{"news": [_article(2, "2024-01-05T10:00:00Z", symbols=["MSFT"])]}, {"news": []}])

    _refresh_symbols(session, ["AAPL", "MSFT"])

    backfill, catch_up = session.calls
    assert (backfill["params"]["symbols"], backfill["params"]["start"], backfill["params"]["end"]) == (
        "MSFT", "2024-01-01T00:00:00Z", "2024-01-10T10:00:00Z")  # fmt: skip
    assert set(catch_up["params"]["symbols"].split(",")) == {"AAPL", "MSFT"}
    assert catch_up["params"]["start"] == "2024-01-10T10:00:00Z"
    assert _stored_ids(tmp_path) == [1, 2]
