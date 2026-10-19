import pandas as pd

from tests.fakes import FakeClock, FakeSession, response
from trader.data import filings, store, web

NY = "America/New_York"
HEADER = """<SEC-HEADER>0000040545-26-000059.hdr.sgml : 20260922
<ACCEPTANCE-DATETIME>20260922081413
<ACCESSION-NUMBER>0000040545-26-000059
<TYPE>8-K
"""


def test_header_acceptance_is_eastern_wall_clock():
    assert filings.parse_header(HEADER) == pd.Timestamp("2026-09-22 08:14:13", tz=NY)
    assert filings.parse_header(HEADER.replace("20260922081413", "20170209170203")) == pd.Timestamp(
        "2017-02-09 17:02:03",
        tz=NY,  # EST
    )


def test_header_without_acceptance_time_is_none():
    assert filings.parse_header("<SEC-HEADER>\n<TYPE>8-K\n") is None


def _block(rows):
    cols = [
        "accessionNumber",
        "filingDate",
        "reportDate",
        "acceptanceDateTime",
        "form",
        "items",
        "primaryDocument",
    ]
    return {c: [r[i] for r in rows] for i, c in enumerate(cols)}


RECENT = _block(
    [
        ("0000320193-26-000080", "2026-07-31", "2026-06-27", "2026-07-31T14:01:02.000Z", "10-Q", "", "aapl.htm"),
        (
            "0000320193-26-000079",
            "2026-07-30",
            "2026-07-30",
            "2026-07-31T00:30:28.000Z",
            "8-K",
            "2.02,9.01",
            "a8k.htm",
        ),
        ("0000320193-26-000078", "2026-07-29", "", "2026-07-29T22:00:00.000Z", "4", "", "form4.xml"),
        ("0000320193-26-000077", "2026-07-28", "", "2026-07-28T22:00:00.000Z", "8-K/A", "5.02", "a8ka.htm"),
    ]
)
OLDER = _block(
    [
        ("0000320193-16-000060", "2016-01-27", "2015-12-26", "2016-01-28T02:32:03.000Z", "10-Q", "", "q.htm"),
        ("0000320193-15-000099", "2015-10-28", "2015-09-26", "2015-10-28T20:00:00.000Z", "10-K", "", "k.htm"),
    ]
)


def test_filings_frame_keeps_only_tracked_forms():
    df = filings.filings_frame("AAPL", 320193, RECENT)
    assert list(df["form"]) == ["10-Q", "8-K"]
    eight_k = df[df["form"] == "8-K"].iloc[0]
    assert eight_k["items"] == "2.02,9.01"
    assert eight_k["url"] == "https://www.sec.gov/Archives/edgar/data/320193/000032019326000079/a8k.htm"


def _header_url(acc):
    return f"https://www.sec.gov/Archives/edgar/data/320193/{acc.replace('-', '')}/{acc}.hdr.sgml"


def _edgar():
    sub = "https://data.sec.gov/submissions/"
    return FakeSession(
        {
            filings.TICKERS_URL: [response(200, {"0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple"}})],
            sub + "CIK0000320193.json": [
                response(
                    200,
                    {
                        "filings": {
                            "recent": RECENT,
                            "files": [
                                {
                                    "name": "CIK0000320193-submissions-001.json",
                                    "filingFrom": "2015-01-01",
                                    "filingTo": "2016-06-30",
                                },
                                {
                                    "name": "CIK0000320193-submissions-002.json",
                                    "filingFrom": "1994-01-01",
                                    "filingTo": "2014-12-31",
                                },  # entirely before HISTORY_START: never fetched
                            ],
                        }
                    },
                )
            ],
            sub + "CIK0000320193-submissions-001.json": [response(200, OLDER)],
            _header_url("0000320193-26-000080"): [
                response(200, text=HEADER.replace("20260922081413", "20260731060102"))
            ],
            _header_url("0000320193-26-000079"): [
                response(200, text=HEADER.replace("20260922081413", "20260730163028"))
            ],
            _header_url("0000320193-16-000060"): [response(404, text="Not Found")],
        }
    )


def _refresh(tmp_path, session):
    clock = FakeClock()
    limiter = web.RateLimiter(1000, clock=clock, sleep=clock.sleep)
    return filings.refresh(symbols=["AAPL"], root=tmp_path, session=session, user_agent="Test test@example.com",
                           limiter=limiter)  # fmt: skip


def _stored(tmp_path):
    df = store.as_of("filings", pd.Timestamp("2030-01-01", tz="UTC"), root=tmp_path)
    return df.set_index("accession")


def test_refresh_stamps_filings_with_header_acceptance_time(tmp_path):
    assert _refresh(tmp_path, _edgar()) == 3  # 10-Q + 8-K from recent, the 2016 10-Q from the older file
    got = _stored(tmp_path)
    assert got.loc["0000320193-26-000079", "available_at"] == pd.Timestamp("2026-07-30 16:30:28", tz=NY)
    assert got.loc["0000320193-26-000080", "available_at"] == pd.Timestamp("2026-07-31 06:01:02", tz=NY)
    assert got.loc["0000320193-26-000079", "accepted_source"] == "sgml-header"


def test_missing_header_falls_back_to_the_later_json_reading(tmp_path):
    _refresh(tmp_path, _edgar())
    row = _stored(tmp_path).loc["0000320193-16-000060"]
    # The JSON value read literally as UTC is the later of its two possible meanings, so it is never early.
    assert row["available_at"] == pd.Timestamp("2016-01-28 02:32:03", tz="UTC")
    assert row["accepted_source"] == "json-latest"


def test_rerun_fetches_no_headers_or_older_files_and_adds_nothing(tmp_path):
    session = _edgar()
    _refresh(tmp_path, session)
    session.calls.clear()

    assert _refresh(tmp_path, session) == 0
    urls = [c["url"] for c in session.calls]
    assert not any(u.endswith((".hdr.sgml", "-submissions-001.json")) for u in urls)
    assert all(c["headers"]["User-Agent"] == "Test test@example.com" for c in session.calls)
