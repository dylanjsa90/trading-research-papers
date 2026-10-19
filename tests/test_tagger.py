import json

import pandas as pd
import pytest

from tests.fakes import FakeSession, completion, response
from trader.data import news, store
from trader.llm import tagger

BASE = "http://llm.test/v1"
MODEL = "gemma-4-e4b-it"
GOOD = {"symbols": ["AAPL"], "event_type": "earnings", "sentiment": 0.6, "surprise": "med", "confidence": 0.8,
        "horizon_days": 30, "relevant": True}  # fmt: skip
LATER = pd.Timestamp("2030-01-01", tz="UTC")


def _docs(*ids, when="2025-06-02 13:00", text=None, tickers=("AAPL",)):
    return pd.DataFrame({"doc_id": list(ids), "text": text or [f"text {i}" for i in ids],
                         "tickers": [list(tickers)] * len(ids), "available_at": pd.Timestamp(when, tz="UTC")})  # fmt: skip


def _session(*chat_responses):
    replies = [r if not isinstance(r, str) else completion(r) for r in chat_responses]
    return FakeSession({f"{BASE}/models": [response(200, {"data": [{"id": MODEL}]})],
                        f"{BASE}/chat/completions": replies})  # fmt: skip


def _tag(root, docs, session, **kwargs):
    return tagger.tag(tagger.NEWS, docs, model=MODEL, root=root, session=session, base_url=BASE, **kwargs)


def _chats(session):
    return [c for c in session.calls if c["url"].endswith("/chat/completions")]


def test_valid_outputs_are_cached_with_provenance(tmp_path):
    session = _session(json.dumps(GOOD))

    summary = _tag(tmp_path, _docs("n1", "n2"), session)

    assert (summary["tagged"], summary["quarantined"], summary["cached"]) == (2, 0, 0)
    rows = store.as_of("tags", LATER, root=tmp_path)
    assert list(rows["doc_id"]) == ["n1", "n2"]
    assert set(rows["model"]) == {MODEL} and set(rows["prompt_version"]) == {tagger.NEWS.prompt_version}
    assert rows["sentiment"].tolist() == [0.6, 0.6] and list(rows.loc[0, "symbols"]) == ["AAPL"]
    sent = _chats(session)[0]["json"]["messages"]
    assert sent[0]["role"] == "system" and "never" in sent[0]["content"].lower()
    assert sent[1]["content"] == "Tagged tickers: AAPL\n\ntext n1"


def test_rerun_is_a_pure_cache_hit_that_never_contacts_the_server(tmp_path):
    _tag(tmp_path, _docs("n1", "n2"), _session(json.dumps(GOOD)))
    again = _session(json.dumps(GOOD))

    summary = _tag(tmp_path, _docs("n1", "n2"), again)

    assert (summary["tagged"], summary["cached"]) == (0, 2)
    assert again.calls == []


def test_invalid_output_gets_a_repair_turn_quoting_the_validation_error(tmp_path):
    bad = json.dumps(GOOD | {"sentiment": 1.7})
    session = _session(bad, json.dumps(GOOD))

    assert _tag(tmp_path, _docs("n1"), session)["tagged"] == 1

    repair = _chats(session)[1]["json"]["messages"]
    assert repair[-2] == {"role": "assistant", "content": bad}
    assert repair[-1]["role"] == "user" and "sentiment" in repair[-1]["content"]
    assert store.as_of("tags", LATER, root=tmp_path).loc[0, "attempts"] == 2


def test_output_still_invalid_after_all_attempts_is_quarantined_and_skipped_next_run(tmp_path):
    session = _session("not json")

    summary = _tag(tmp_path, _docs("n1"), session, attempts=3)

    assert (summary["tagged"], summary["quarantined"]) == (0, 1)
    assert len(_chats(session)) == 3
    quarantined = store.as_of("tags_quarantine", LATER, root=tmp_path).iloc[0]
    assert quarantined["raw"] == "not json" and "json" in quarantined["error"].lower()
    again = _session(json.dumps(GOOD))
    assert _tag(tmp_path, _docs("n1"), again)["cached"] == 1 and again.calls == []


def test_post_cutoff_label_follows_the_models_training_cutoff(tmp_path):
    docs = pd.concat([_docs("before", when="2025-01-31 23:59"), _docs("after", when="2025-02-01 00:00")])
    _tag(tmp_path, docs, _session(json.dumps(GOOD)))
    rows = store.as_of("tags", LATER, root=tmp_path).set_index("doc_id")
    assert (rows.loc["before", "post_cutoff"], rows.loc["after", "post_cutoff"]) == (False, True)


def test_a_model_without_a_recorded_cutoff_is_refused_before_any_call(tmp_path):
    session = _session(json.dumps(GOOD))
    with pytest.raises(ValueError, match="cutoff"):
        tagger.tag(tagger.NEWS, _docs("n1"), model="mystery-7b", root=tmp_path, session=session, base_url=BASE)
    assert session.calls == []


def test_server_failure_aborts_the_run_but_keeps_finished_documents(tmp_path):
    session = _session(json.dumps(GOOD), response(400, text='{"error": "context length exceeded"}'))

    with pytest.raises(RuntimeError, match="context length"):
        _tag(tmp_path, _docs("n1", "n2", "n3"), session)

    assert list(store.as_of("tags", LATER, root=tmp_path)["doc_id"]) == ["n1"]
    assert store.as_of("tags_quarantine", LATER, root=tmp_path).empty


def test_anonymised_mode_masks_the_input_restores_symbols_and_caches_separately(tmp_path):
    docs = _docs("n1", text=["Apple (NASDAQ:AAPL) beat estimates on March 3, 2025."])
    session = _session(json.dumps(GOOD | {"symbols": ["Company 1"]}))

    _tag(tmp_path, docs, session, anonymised=True)

    sent = _chats(session)[0]["json"]["messages"][1]["content"]
    assert sent == "Tagged tickers: Company 1\n\nCompany 1 (Company 1) beat estimates on [DATE]."
    row = store.as_of("tags", LATER, root=tmp_path).iloc[0]
    assert list(row["symbols"]) == ["AAPL"] and row["prompt_version"] == tagger.NEWS.prompt_version + "+anon"
    assert _tag(tmp_path, docs, _session(json.dumps(GOOD)))["tagged"] == 1  # the normal arm is a different key


def test_each_run_logs_tokens_and_time_per_thousand_documents(tmp_path):
    ticks = iter(range(100))
    _tag(tmp_path, _docs("n1", "n2"), _session(json.dumps(GOOD)), clock=lambda: next(ticks))

    log = json.loads((tmp_path / "llm_runs.jsonl").read_text().splitlines()[-1])

    assert (log["model"], log["tagged"], log["input_tokens"], log["output_tokens"]) == (MODEL, 2, 200, 40)
    assert log["per_1k_docs"]["input_tokens"] == 100_000 and log["per_1k_docs"]["usd"] == 0.0
    assert log["seconds"] > 0 and log["per_1k_docs"]["seconds"] == log["seconds"] * 500


def test_news_documents_join_headline_summary_and_plain_text_body():
    articles = news.articles_frame([{
        "id": 7, "headline": "Apple beats", "summary": "Revenue up.", "content": "<p>Body <b>text</b>.</p>",
        "author": "a", "url": "u", "source": "benzinga", "symbols": ["AAPL", "FB"],
        "created_at": "2025-06-02T13:00:00Z", "updated_at": "2025-06-02T13:00:00Z"}])  # fmt: skip

    docs = tagger.news_documents(articles)

    row = docs.iloc[0]
    assert (row["doc_id"], row["text"]) == ("news-7", "Apple beats\n\nRevenue up.\n\nBody text.")
    assert list(row["tickers"]) == ["AAPL", "META"] and row["available_at"] == articles.loc[0, "available_at"]


def test_anonymised_mode_works_on_documents_without_tagged_tickers(tmp_path):
    docs = _docs("p1", text=["Tesla, 2024."]).drop(columns="tickers")
    session = _session(json.dumps(GOOD | {"symbols": []}))
    _tag(tmp_path, docs, session, anonymised=True)
    assert _chats(session)[0]["json"]["messages"][1]["content"] == "Company 1, [YEAR]."
