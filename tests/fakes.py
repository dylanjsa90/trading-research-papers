"""Offline stand-ins for HTTP sessions."""

import json

import requests


def response(status, payload=None, text=None):
    r = requests.Response()
    r.status_code = status
    r._content = (text if text is not None else json.dumps(payload or {})).encode()
    r.url = "https://example.test/"
    return r


def completion(content, input_tokens=100, output_tokens=20):
    """An OpenAI-style chat completion, as LM Studio, Ollama and llama.cpp return it."""
    return response(200, {
        "choices": [{"message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": input_tokens, "completion_tokens": output_tokens},
    })  # fmt: skip


class FakeSession:
    """Serves queued responses per URL (the last one repeats) and records every request."""

    def __init__(self, routes):
        self.routes = {url: list(rs) for url, rs in routes.items()}
        self.calls = []

    def get(self, url, params=None, headers=None, timeout=None):
        self.calls.append({"url": url, "params": dict(params or {}), "headers": dict(headers or {})})
        return self._next(url)

    def post(self, url, json=None, headers=None, timeout=None):
        self.calls.append({"url": url, "json": json, "headers": dict(headers or {})})
        return self._next(url)

    def _next(self, url):
        queue = self.routes[url]
        return queue.pop(0) if len(queue) > 1 else queue[0]


class FakeClock:
    def __init__(self):
        self.now, self.slept = 0.0, []

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.slept.append(round(seconds, 9))
        self.now += seconds
