"""Polite HTTP for the data sources: a minimum spacing between calls, and retries on throttling or 5xx."""

import time

RETRY_STATUS = {429, 500, 502, 503, 504}


class RateLimiter:
    def __init__(self, per_second, clock=time.monotonic, sleep=time.sleep):
        self.interval = 1.0 / per_second
        self._clock, self._sleep = clock, sleep
        self._next = 0.0

    def wait(self):
        now = self._clock()
        if now < self._next:
            self._sleep(self._next - now)
            now = self._next
        self._next = now + self.interval


def fetch(session, url, limiter, params=None, headers=None, retries=3, backoff=2.0, sleep=time.sleep):
    for attempt in range(retries + 1):
        limiter.wait()
        resp = session.get(url, params=params, headers=headers, timeout=30)
        if resp.status_code in RETRY_STATUS and attempt < retries:
            sleep(backoff * 2**attempt)
            continue
        resp.raise_for_status()
        return resp


def get_json(session, url, limiter, **kwargs):
    return fetch(session, url, limiter, **kwargs).json()
