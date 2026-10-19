import json

import pandas as pd
import pymupdf

from tests.fakes import FakeSession, completion, response
from trader.llm import corpus

BASE = "http://llm.test/v1"
MODEL = "gemma-4-e4b-it"
RECORD = {"method": "LLM agent", "market": "US equities", "evaluated_models": ["GPT-4o", "Llama-3-8B"],
          "test_window": "2024-01 to 2024-06", "test_after_model_cutoff": "not_stated", "costs_modelled": "no",
          "beats_buy_and_hold": "yes"}  # fmt: skip


def _pdf(path, pages):
    doc = pymupdf.open()
    for text in pages:
        doc.new_page().insert_textbox(pymupdf.Rect(40, 40, 560, 800), text, fontsize=8)
    doc.save(path)


def test_select_papers_keeps_downloaded_llm_sentiment_and_agent_titles():
    rows = [
        {"rank": 1, "status": "downloaded", "title": "FinMem: A Performance-Enhanced LLM Trading Agent"},
        {"rank": 2, "status": "downloaded", "title": "GARCH Volatility of Bond Futures"},
        {"rank": 3, "status": "not_found", "title": "News Sentiment and Pairs Trading"},
        {"rank": 4, "status": "downloaded", "title": "Large Language Models for Portfolio Choice"},
    ]
    assert [r["rank"] for r in corpus.select_papers(rows)] == [1, 4]


def test_paper_text_keeps_the_opening_and_windows_around_evaluation_keywords(tmp_path):
    filler = "Lorem ipsum dolor sit amet. " * 40
    _pdf(tmp_path / "p.pdf", ["Abstract. We propose an agent. " + filler, filler,
                              filler + " Our backtest charges transaction\ncosts of 10 bps per trade. " + filler])  # fmt: skip

    text = corpus.paper_text(tmp_path / "p.pdf", head=200, budget=1200, window=150)

    assert text.startswith("Abstract. We propose an agent.")
    assert "transaction costs of 10 bps" in text
    assert len(text) <= 1200


def test_run_tags_the_selected_papers_and_writes_one_csv_row_each(tmp_path):
    papers = tmp_path / "papers"
    papers.mkdir()
    _pdf(papers / "0001_a.pdf", ["Abstract. An LLM agent trades US equities; test period 2024-01 to 2024-06."])
    (papers / "0003_c.pdf").write_bytes(b"not a pdf")
    manifest = [
        {"rank": 1, "status": "downloaded", "title": "An LLM Trading Agent", "pdf_path": "0001_a.pdf", "date": "2026-10-12"},
        {"rank": 2, "status": "downloaded", "title": "Bond GARCH", "pdf_path": "0002_b.pdf", "date": "2026-10-12"},
        {"rank": 3, "status": "downloaded", "title": "Sentiment Agents", "pdf_path": "0003_c.pdf", "date": "2026-10-12"},
    ]  # fmt: skip
    (papers / "manifest.json").write_text(json.dumps(manifest))
    session = FakeSession({f"{BASE}/models": [response(200, {"data": [{"id": MODEL}]})],
                           f"{BASE}/chat/completions": [completion(json.dumps(RECORD))]})  # fmt: skip
    out = tmp_path / "corpus.csv"

    summary = corpus.run(papers_dir=papers, out=out, model=MODEL, root=tmp_path / "data", session=session,
                         base_url=BASE)  # fmt: skip

    assert summary["tagged"] == 1
    rows = pd.read_csv(out, keep_default_na=False).to_dict("records")
    assert rows == [
        {"rank": 1, "title": "An LLM Trading Agent", "status": "ok", "method": "LLM agent", "market": "US equities",
         "evaluated_models": "GPT-4o; Llama-3-8B", "test_window": "2024-01 to 2024-06",
         "test_after_model_cutoff": "not_stated", "costs_modelled": "no", "beats_buy_and_hold": "yes",
         "model": MODEL, "prompt_version": corpus.PAPERS.prompt_version},
        {"rank": 3, "title": "Sentiment Agents", "status": "unreadable", "method": "", "market": "",
         "evaluated_models": "", "test_window": "", "test_after_model_cutoff": "", "costs_modelled": "",
         "beats_buy_and_hold": "", "model": MODEL, "prompt_version": corpus.PAPERS.prompt_version},
    ]  # fmt: skip


def test_paper_text_stops_adding_windows_at_the_budget(tmp_path):
    filler = "Lorem ipsum dolor sit amet. " * 40
    _pdf(tmp_path / "p.pdf", ["Abstract. " + filler, "We charge commissions. " + filler, filler,
                              "We beat buy-and-hold. " + filler])  # fmt: skip

    text = corpus.paper_text(tmp_path / "p.pdf", head=100, budget=500, window=150)

    assert "commissions" in text and "buy-and-hold" not in text and len(text) <= 500
