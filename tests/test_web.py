import pytest
import requests

from tests.fakes import FakeClock, FakeSession, response
from trader.data import web


def test_rate_limiter_spaces_burst_calls_and_skips_sleep_after_idle():
    clock = FakeClock()
    limiter = web.RateLimiter(per_second=10, clock=clock, sleep=clock.sleep)
    for _ in range(3):
        limiter.wait()
    assert clock.slept == [0.1, 0.1]
    clock.now += 5
    limiter.wait()
    assert clock.slept == [0.1, 0.1]


def _limiter():
    clock = FakeClock()
    return web.RateLimiter(per_second=1000, clock=clock, sleep=clock.sleep), clock


def test_get_json_retries_throttling_then_returns_payload():
    session = FakeSession({"u": [response(429), response(503), response(200, {"ok": 1})]})
    limiter, clock = _limiter()
    assert web.get_json(session, "u", limiter, sleep=clock.sleep) == {"ok": 1}
    assert len(session.calls) == 3


def test_get_json_gives_up_after_retries():
    session = FakeSession({"u": [response(429)]})
    limiter, clock = _limiter()
    with pytest.raises(requests.HTTPError):
        web.get_json(session, "u", limiter, retries=2, sleep=clock.sleep)
    assert len(session.calls) == 3


def test_get_json_does_not_retry_client_errors():
    session = FakeSession({"u": [response(403)]})
    limiter, clock = _limiter()
    with pytest.raises(requests.HTTPError):
        web.get_json(session, "u", limiter, sleep=clock.sleep)
    assert len(session.calls) == 1
