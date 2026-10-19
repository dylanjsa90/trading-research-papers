import pytest

from tests.fakes import FakeSession, completion, response
from trader.llm import local

BASE = "http://llm.test/v1"
MESSAGES = [{"role": "user", "content": "hi"}]


def test_require_model_refuses_when_the_pinned_model_is_not_served():
    session = FakeSession({f"{BASE}/models": [response(200, {"data": [{"id": "qwen3-vl-4b"}]})]})
    with pytest.raises(RuntimeError, match="gemma-4-e4b-it.*qwen3-vl-4b"):
        local.require_model("gemma-4-e4b-it", session=session, base_url=BASE)


def test_require_model_passes_when_served():
    session = FakeSession({f"{BASE}/models": [response(200, {"data": [{"id": "gemma-4-e4b-it"}]})]})
    local.require_model("gemma-4-e4b-it", session=session, base_url=BASE)


def test_chat_json_sends_the_schema_at_temperature_zero_and_returns_text_and_usage():
    session = FakeSession({f"{BASE}/chat/completions": [completion('{"a": 1}', 120, 30)]})

    text, usage = local.chat_json(MESSAGES, {"type": "object"}, "record", model="m", max_tokens=50, session=session,
                                  base_url=BASE)  # fmt: skip

    assert (text, usage) == ('{"a": 1}', {"input_tokens": 120, "output_tokens": 30})
    body = session.calls[0]["json"]
    assert (body["model"], body["messages"], body["temperature"], body["max_tokens"]) == ("m", MESSAGES, 0, 50)
    assert body["response_format"] == {
        "type": "json_schema", "json_schema": {"name": "record", "strict": True, "schema": {"type": "object"}}}  # fmt: skip


def test_chat_json_retries_server_errors_then_raises():
    session = FakeSession({f"{BASE}/chat/completions": [response(503, text="model loading")]})
    with pytest.raises(RuntimeError, match="503.*model loading"):
        local.chat_json(MESSAGES, {}, "r", model="m", max_tokens=5, session=session, base_url=BASE, retries=2,
                        sleep=lambda s: None)  # fmt: skip
    assert len(session.calls) == 3


def test_chat_json_does_not_retry_client_errors_such_as_context_overflow():
    error = response(400, text='{"error": "context length 4096 exceeded"}')
    session = FakeSession({f"{BASE}/chat/completions": [error]})
    with pytest.raises(RuntimeError, match="context length"):
        local.chat_json(MESSAGES, {}, "r", model="m", max_tokens=5, session=session, base_url=BASE)
    assert len(session.calls) == 1
