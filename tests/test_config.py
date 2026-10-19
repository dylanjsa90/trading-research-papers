import pytest

from trader import config


def test_require_env_treats_empty_placeholder_as_missing(monkeypatch):
    monkeypatch.setenv("APCA_API_SECRET_KEY", "")
    with pytest.raises(RuntimeError, match="APCA_API_SECRET_KEY"):
        config.require_env("APCA_API_SECRET_KEY")


def _paper_env(monkeypatch, **overrides):
    env = {
        "APCA_API_KEY_ID": "PKTESTKEY",
        "APCA_API_SECRET_KEY": "secret",
        "APCA_API_BASE_URL": "https://paper-api.alpaca.markets",
    } | overrides
    for k, v in env.items():
        monkeypatch.setenv(k, v)


def test_alpaca_credentials_accepts_paper_account(monkeypatch):
    _paper_env(monkeypatch)
    assert config.alpaca_credentials() == ("PKTESTKEY", "secret")


@pytest.mark.parametrize(
    "overrides",
    [
        {"APCA_API_BASE_URL": "https://api.alpaca.markets"},  # live trading endpoint
        {"APCA_API_KEY_ID": "AKLIVEKEY"},  # live key IDs start with AK
    ],
)
def test_alpaca_credentials_refuses_anything_live(monkeypatch, overrides):
    _paper_env(monkeypatch, **overrides)
    with pytest.raises(RuntimeError, match="paper"):
        config.alpaca_credentials()


def test_every_pinned_model_has_a_training_cutoff():
    assert {config.TAGGER_MODEL, config.REVIEW_MODEL} <= set(config.MODEL_CUTOFFS)


def test_llm_base_url_defaults_to_lm_studio_on_localhost(monkeypatch):
    monkeypatch.delenv("LLM_BASE_URL", raising=False)
    assert config.llm_base_url() == "http://localhost:1234/v1"
    monkeypatch.setenv("LLM_BASE_URL", "http://172.17.144.1:1234/v1/")
    assert config.llm_base_url() == "http://172.17.144.1:1234/v1"
