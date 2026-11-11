from config import DEFAULT_TIMEOUT


def fetch(url: str, timeout: int = DEFAULT_TIMEOUT) -> str:
    """Fetch a URL with a timeout in seconds (default 5 seconds).

    Returns the response body as a string.
    """
    return f"GET {url} timeout={timeout}"
