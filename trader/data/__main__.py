"""Refresh the point-in-time data store.

    uv run --env-file .env python -m trader.data [prices|news|filings|all]
    uv run --env-file .env python -m trader.data count-news YYYY-MM-DD

`count-news` prints one New York day's article count across all symbols (Alpaca documents 130+ a day).
"""

import sys

from trader.data import filings, news, prices

SOURCES = {"prices": prices.refresh, "news": news.refresh, "filings": filings.refresh}


def main(argv):
    if len(argv) == 2 and argv[0] == "count-news":
        print(news.count_day(argv[1]))
        return 0
    names = list(SOURCES) if argv in ([], ["all"]) else argv
    if not set(names) <= set(SOURCES):
        print(__doc__, file=sys.stderr)
        return 2
    for name in names:
        print(f"{name}: {SOURCES[name]()} rows added", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
