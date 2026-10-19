"""Phase 2 shakedown: the tagging pipeline over this repo's LLM, sentiment and agent papers.

Produces docs/research/corpus-llm-papers.csv: how each paper evaluates its method (window, post-cutoff status,
costs, buy-and-hold), the corpus-mining table the research doc asks for (§12). Each paper is sent as its
opening pages plus windows around evaluation keywords, about 8k tokens, so load the model with a context of at
least 16k tokens.
"""

import json
import re
from pathlib import Path

import pandas as pd
import pymupdf

from trader import config
from trader.data import store
from trader.llm import tagger
from trader.llm.schemas import PaperEvaluation

PAPERS_SYSTEM = """\
You read excerpts from one research paper about trading and record how it evaluates its method, as JSON. \
Report only what the paper states; never fill gaps from your own knowledge.

- method: one sentence naming the approach, for example "multi-agent LLM debate producing daily buy/sell signals".
- market: the assets and region traded, for example "US equities", "crypto" or "Chinese A-shares"; \
"not stated" if absent.
- evaluated_models: the language models used in the main results, for example "GPT-4o" or "Llama-3-8B"; \
empty if none.
- test_window: the out-of-sample or backtest period as stated, for example "2023-01 to 2023-12"; \
"not stated" if absent.
- test_after_model_cutoff: "yes" if the paper says its test period comes after the training cutoff of the \
models it evaluates, "no" if it says the period overlaps their training data, otherwise "not_stated".
- costs_modelled: "yes" if the backtest charges transaction costs, commissions or slippage, "no" if the paper \
says it does not, otherwise "not_stated".
- beats_buy_and_hold: "yes" or "no" if the paper compares against buy-and-hold, otherwise "not_compared"."""
PAPERS = tagger.Task("paper_evals", PaperEvaluation, PAPERS_SYSTEM, "papers-v1", 600)

TITLE = re.compile(r"\b(LLMs?|large language|language models?|GPT|ChatGPT|agents?|agentic|sentiment)\b",
                   re.IGNORECASE)  # fmt: skip
KEYWORDS = re.compile(
    r"buy[- ]and[- ]hold|transaction costs?|commissions?|slippage|test(?:ing)? (?:period|set|window)|backtest"
    r"|out-of-sample|(?:knowledge|training) cut-?off|look-?ahead|leakage",
    re.IGNORECASE,
)
FIELDS = list(PaperEvaluation.model_fields)
SEPARATOR = " [...] "


def select_papers(manifest):
    return [r for r in manifest if r.get("status") == "downloaded" and TITLE.search(r["title"])]


def paper_text(path, head=12_000, budget=32_000, window=600):
    """The paper's opening (title, abstract, setup) plus windows around evaluation keywords, within budget."""
    with pymupdf.open(path) as doc:
        text = " ".join(" ".join(page.get_text().split()) for page in doc)
    pieces, used, end = [text[:head]], min(head, len(text)), head
    for match in KEYWORDS.finditer(text, head):
        start, stop = max(match.start() - window, end), min(match.end() + window, len(text))
        if start >= stop:
            continue
        if used + len(SEPARATOR) + stop - start > budget:
            break
        pieces.append(text[start:stop])
        used, end = used + len(SEPARATOR) + stop - start, stop
    return SEPARATOR.join(pieces)


def _cached(table, model, version, root):
    rows = store.as_of(table, pd.Timestamp.now(tz="UTC"), root=root)
    return rows[(rows["model"] == model) & (rows["prompt_version"] == version)] if len(rows) else rows


def run(papers_dir=config.ROOT / "papers", out=config.ROOT / "docs/research/corpus-llm-papers.csv",
        model=config.TAGGER_MODEL, root=None, **tag_options):  # fmt: skip
    """Tag every selected paper not yet cached, then write the CSV from the cache. Returns the run summary."""
    papers = select_papers(json.loads((Path(papers_dir) / "manifest.json").read_text()))
    docs, status = [], {}
    for paper in papers:
        doc_id = f"paper-{paper['rank']:04d}"
        try:
            text = paper_text(Path(papers_dir) / paper["pdf_path"])
        except (OSError, RuntimeError):  # missing or corrupt PDF (pymupdf raises FileDataError)
            status[doc_id] = "unreadable"
            continue
        if not text.strip():
            status[doc_id] = "no_text"  # scanned images
            continue
        docs.append({"doc_id": doc_id, "text": text, "available_at": pd.Timestamp(paper["date"], tz="UTC")})
    summary = tagger.tag(PAPERS, pd.DataFrame(docs, columns=["doc_id", "text", "available_at"]), model=model,
                         root=root, **tag_options)  # fmt: skip

    found = _cached(PAPERS.table, model, PAPERS.prompt_version, root)
    found = found.set_index("doc_id") if len(found) else found
    quarantined = set(_cached(f"{PAPERS.table}_quarantine", model, PAPERS.prompt_version, root).get("doc_id", []))
    rows = []
    for paper in papers:
        doc_id = f"paper-{paper['rank']:04d}"
        row = {"rank": paper["rank"], "title": paper["title"]}
        if doc_id in status or doc_id in quarantined or doc_id not in found.index:
            row |= {"status": status.get(doc_id, "quarantined" if doc_id in quarantined else "missing")}
            row |= dict.fromkeys(FIELDS, "")
        else:
            record = found.loc[doc_id]
            row |= {"status": "ok"} | {f: record[f] for f in FIELDS}
            row["evaluated_models"] = "; ".join(record["evaluated_models"])
        rows.append(row | {"model": model, "prompt_version": PAPERS.prompt_version})
    pd.DataFrame(rows).to_csv(out, index=False)
    return summary
