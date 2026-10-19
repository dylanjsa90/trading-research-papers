"""Tag documents with the local LLM into schema-validated records (Phase 2).

Results are cached in the point-in-time store keyed by (doc_id, model, prompt_version), so backtests are
reproducible and re-runs are free (design rule 5). A record's available_at is its document's: in live use a
document is tagged in the same after-close run that first sees it. Output that still fails validation after
repair turns is quarantined, never guessed at; server trouble aborts the run and keeps what finished (§9).
"""

import json
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup
from pydantic import BaseModel, ValidationError

from trader import config
from trader.data import store
from trader.llm import anonymise as anon
from trader.llm import local
from trader.llm.schemas import Extraction

KEY = ["doc_id", "model", "prompt_version"]


@dataclass(frozen=True)
class Task:
    table: str  # cache table; quarantined documents go to <table>_quarantine
    schema: type[BaseModel]
    system: str
    prompt_version: str  # bump whenever `system` or the document text changes
    max_tokens: int


NEWS_SYSTEM = """\
You read one news article or filing excerpt about US-listed companies and record what the text says, as JSON.

Describe the text only. Never forecast prices, returns or market reactions, and never add facts from outside \
the text. Companies may appear as placeholders such as "Company 1" and dates as [DATE]; report them as given.

- symbols: the tagged tickers the text is materially about (may be empty).
- event_type: the main event: earnings, guidance, m&a, legal_regulatory, product, management, macro, \
analyst_action or other.
- sentiment: how favourably the text presents the news for those companies, from -1 (clearly negative) to 1 \
(clearly positive); 0 if neutral or mixed.
- surprise: how unexpected the text itself says the news is (for example a beat or miss against estimates, or \
an unscheduled announcement): none, low, med or high.
- confidence: how clearly the text supports your event_type and sentiment, from 0 to 1.
- horizon_days: for how many days the information stays relevant to the companies' fundamentals, 1 to 365.
- relevant: false for market wraps, lists of movers, promotions, or texts that mention the companies only in \
passing."""
NEWS = Task("tags", Extraction, NEWS_SYSTEM, "news-v1", 300)


def _plain(html):
    soup = BeautifulSoup(html or "", "html.parser")
    for block in soup.find_all(["p", "br", "div", "li", "tr", "h1", "h2", "h3", "h4"]):
        block.insert_after("\n")
    lines = (" ".join(line.split()) for line in soup.get_text().splitlines())
    return "\n".join(line for line in lines if line)


def news_documents(articles, max_chars=6000):
    """Stored news rows -> documents: headline, summary and plain-text body, truncated (~1.5k tokens)."""
    texts = [
        "\n\n".join(part for part in (h, s, _plain(c)) if part)[:max_chars]
        for h, s, c in zip(articles["headline"], articles["summary"], articles["content"])
    ]
    return pd.DataFrame({"doc_id": "news-" + articles["id"].astype(str), "text": texts,
                         "tickers": articles["symbols"].map(list), "available_at": articles["available_at"]})  # fmt: skip


def _tag_one(task, doc, model, anonymised, session, base_url, attempts, clock):
    text, tickers, mapping = doc.text, getattr(doc, "tickers", None), {}
    if anonymised:
        text, mapping = anon.anonymise(text, list(tickers) if tickers is not None else [])
        if tickers is not None:
            tagged = {config.SYMBOL_ALIASES.get(t, t) for t in tickers}
            tickers = [p for p, symbol in mapping.items() if symbol in tagged]
    prefix = f"Tagged tickers: {', '.join(tickers) or 'none'}\n\n" if tickers is not None else ""
    messages = [{"role": "system", "content": task.system}, {"role": "user", "content": prefix + text}]
    usage, start = {"input_tokens": 0, "output_tokens": 0}, clock()
    for attempt in range(1, attempts + 1):
        raw, used = local.chat_json(messages, task.schema.model_json_schema(), task.table, model, task.max_tokens,
                                    session=session, base_url=base_url)  # fmt: skip
        usage = {k: usage[k] + used[k] for k in usage}
        try:
            fields = task.schema.model_validate_json(raw).model_dump()
        except ValidationError as err:
            error = str(err)
            messages = [*messages, {"role": "assistant", "content": raw},
                        {"role": "user", "content": f"That output failed validation:\n{error}\nReturn corrected JSON."}]  # fmt: skip
            continue
        if anonymised and "symbols" in fields:
            fields["symbols"] = anon.restore(fields["symbols"], mapping)
        return True, fields | {"attempts": attempt}, usage, clock() - start
    return False, {"error": error, "raw": raw, "attempts": attempts}, usage, clock() - start


def tag(task, docs, model=config.TAGGER_MODEL, anonymised=False, root=None, session=None, base_url=None,
        attempts=3, workers=1, flush_every=100, clock=time.monotonic):  # fmt: skip
    """Tag every document in `docs` (doc_id, text, available_at, optional tickers) not already cached or
    quarantined for this model and prompt version. Returns the run summary, also appended to llm_runs.jsonl."""
    if model not in config.MODEL_CUTOFFS:
        raise ValueError(f"no training cutoff recorded for {model}; add it to config.MODEL_CUTOFFS first")
    version = task.prompt_version + ("+anon" if anonymised else "")
    seen = store.keys(task.table, KEY, root) | store.keys(f"{task.table}_quarantine", KEY, root)
    done = {doc_id for doc_id, m, v in seen if (m, v) == (model, version)}
    docs = docs.drop_duplicates("doc_id")
    todo = docs[~docs["doc_id"].isin(done)]
    cutoff = pd.Timestamp(config.MODEL_CUTOFFS[model], tz="UTC") + pd.Timedelta(days=1)
    totals = {"tagged": 0, "quarantined": 0, "input_tokens": 0, "output_tokens": 0}
    buffers = {task.table: [], f"{task.table}_quarantine": []}
    started_at, started = datetime.now(UTC), clock()

    def flush():
        for table, rows in buffers.items():
            if rows:
                store.upsert(table, pd.DataFrame(rows), key=KEY, partition="month", root=root)
                rows.clear()

    def work(doc):
        return _tag_one(task, doc, model, anonymised, session, base_url, attempts, clock)

    if len(todo):
        session = session or requests.Session()
        local.require_model(model, session=session, base_url=base_url)
    pool = ThreadPoolExecutor(workers)
    try:
        rows = list(todo.itertuples(index=False))
        for doc, (ok, fields, usage, seconds) in zip(rows, pool.map(work, rows)):
            base = {"doc_id": doc.doc_id, "model": model, "prompt_version": version}
            meta = usage | {"seconds": round(seconds, 3), "tagged_at": pd.Timestamp.now(tz="UTC"),
                            "available_at": doc.available_at, "month": doc.available_at.strftime("%Y-%m")}  # fmt: skip
            if ok:
                buffers[task.table].append(base | fields | meta | {"post_cutoff": doc.available_at >= cutoff})
            else:
                buffers[f"{task.table}_quarantine"].append(base | fields | meta)
            totals["tagged" if ok else "quarantined"] += 1
            for k in usage:
                totals[k] += usage[k]
            if sum(map(len, buffers.values())) >= flush_every:
                flush()
    finally:
        pool.shutdown(wait=True, cancel_futures=True)
        flush()
        seconds, sent = clock() - started, totals["tagged"] + totals["quarantined"]
        per_1k = {k: round(totals[k] * 1000 / sent, 1) if sent else 0 for k in ("input_tokens", "output_tokens")}
        summary = {"task": task.table, "model": model, "prompt_version": version,
                   "started_at": started_at.isoformat(), "documents": len(docs), "cached": len(docs) - len(todo),
                   **totals, "seconds": round(seconds, 3),
                   # local inference costs GPU time, not dollars
                   "per_1k_docs": per_1k | {"seconds": round(seconds * 1000 / sent, 1) if sent else 0, "usd": 0.0}}  # fmt: skip
        log = Path(root or config.DATA_DIR) / "llm_runs.jsonl"
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("a") as f:
            f.write(json.dumps(summary) + "\n")
    return summary
