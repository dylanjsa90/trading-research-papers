"""Run the local LLM tagger. The server is LLM_BASE_URL (default http://localhost:1234/v1, LM Studio).

    uv run --env-file .env python -m trader.llm check
    uv run --env-file .env python -m trader.llm papers [--workers N]
    uv run --env-file .env python -m trader.llm news [--since YYYY-MM-DD] [--anonymised] [--workers N]

`check` confirms the server offers the pinned model and returns valid JSON for a sample article. `papers` is
the Phase 2 shakedown: it writes docs/research/corpus-llm-papers.csv. `news` tags stored articles, by default
only those after the model's training cutoff (the evidence window); --since 2016-01-01 covers the full history.
"""

import argparse
import json
import sys
import time

import pandas as pd
import requests

from trader import config
from trader.data import store
from trader.llm import corpus, local, tagger

SAMPLE = ("Tagged tickers: AAPL\n\nApple Inc (NASDAQ:AAPL) reported quarterly revenue of $95.4 billion, above "
          "the $94.1 billion analysts expected, and raised its dividend by 4%.")  # fmt: skip


def check():
    local.require_model(config.TAGGER_MODEL)
    messages = [{"role": "system", "content": tagger.NEWS.system}, {"role": "user", "content": SAMPLE}]
    start = time.monotonic()
    raw, usage = local.chat_json(messages, tagger.NEWS.schema.model_json_schema(), "check", config.TAGGER_MODEL,
                                 tagger.NEWS.max_tokens)  # fmt: skip
    record = tagger.NEWS.schema.model_validate_json(raw)
    print(f"{config.TAGGER_MODEL} at {config.llm_base_url()}: {time.monotonic() - start:.1f}s, {usage}")
    return record.model_dump()


def news(since, anonymised, workers):
    articles = store.as_of("news", pd.Timestamp.now(tz="UTC"))
    if articles.empty:
        return {"documents": 0, "note": "no stored news yet; run `python -m trader.data news` first"}
    since = pd.Timestamp(since or config.MODEL_CUTOFFS[config.TAGGER_MODEL] + pd.Timedelta(days=1), tz="UTC")
    articles = articles[articles["available_at"] >= since]
    return tagger.tag(tagger.NEWS, tagger.news_documents(articles), anonymised=anonymised, workers=workers)


def main(argv):
    parser = argparse.ArgumentParser(prog="python -m trader.llm", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)  # fmt: skip
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("check")
    commands.add_parser("papers").add_argument("--workers", type=int, default=1)
    tag_news = commands.add_parser("news")
    tag_news.add_argument("--since")
    tag_news.add_argument("--anonymised", action="store_true")
    tag_news.add_argument("--workers", type=int, default=1)
    args = parser.parse_args(argv)
    try:
        if args.command == "check":
            result = check()
        elif args.command == "papers":
            result = corpus.run(workers=args.workers)
        else:
            result = news(args.since, args.anonymised, args.workers)
    except requests.ConnectionError:
        print(f"No LLM server at {config.llm_base_url()}. Start LM Studio's server (see README, 'Local LLM'), "
              "or set LLM_BASE_URL in .env.", file=sys.stderr)  # fmt: skip
        return 1
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
