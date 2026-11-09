"""A small HTTP client wrapper."""
import time


def fetch(get, url):
    """Calls get(url), retrying up to 3 times on ConnectionError."""
    for attempt in range(3):
        try:
            return get(url)
        except ConnectionError:
            time.sleep(0)
    raise ConnectionError(url)


def upload(put, url, body):
    """Calls put(url, body), retrying up to 3 times on ConnectionError."""
    for attempt in range(3):
        try:
            return put(url, body)
        except ConnectionError:
            time.sleep(0)
    raise ConnectionError(url)
