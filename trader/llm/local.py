"""Chat completions with JSON-schema output on a local OpenAI-compatible server (LM Studio, Ollama, llama.cpp).

Server trouble raises, so a run stops with "no new tags" rather than guessing (§9). Only 5xx is retried:
a 4xx such as a context overflow would fail the same way again.
"""

import time

import requests

from trader import config
from trader.data.web import RETRY_STATUS


def _base(base_url):
    return (base_url or config.llm_base_url()).rstrip("/")


def require_model(model, session=None, base_url=None):
    """Refuse to run unless the server offers exactly the pinned model ID (§4.2)."""
    url = f"{_base(base_url)}/models"
    resp = (session or requests.Session()).get(url, timeout=10)
    resp.raise_for_status()
    served = sorted(m["id"] for m in resp.json().get("data", []))
    if model not in served:
        raise RuntimeError(f"{model} is not served at {url}; it offers: {', '.join(served) or 'nothing'}")


def chat_json(messages, schema, name, model, max_tokens, session=None, base_url=None, retries=2, timeout=300,
              sleep=time.sleep):  # fmt: skip
    """One completion constrained to `schema`. Returns (content, {"input_tokens", "output_tokens"})."""
    url = f"{_base(base_url)}/chat/completions"
    body = {
        "model": model, "messages": messages, "temperature": 0, "max_tokens": max_tokens,
        # Thinking burns the token budget before any JSON appears (llama.cpp turns it on for Gemma 4).
        "chat_template_kwargs": {"enable_thinking": False},
        "response_format": {"type": "json_schema", "json_schema": {"name": name, "strict": True, "schema": schema}},
    }  # fmt: skip
    session = session or requests.Session()
    for attempt in range(retries + 1):
        resp = session.post(url, json=body, timeout=timeout)
        if resp.status_code in RETRY_STATUS and attempt < retries:
            sleep(2.0 * 2**attempt)
            continue
        if not resp.ok:
            raise RuntimeError(f"{resp.status_code} from {url}: {resp.text[:500]}")
        data = resp.json()
        usage = data.get("usage") or {}
        return data["choices"][0]["message"]["content"], {
            "input_tokens": usage.get("prompt_tokens", 0), "output_tokens": usage.get("completion_tokens", 0)}  # fmt: skip
