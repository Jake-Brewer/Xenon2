import settings as s
assert s.REQUEST_TIMEOUT_SECONDS == 60
assert (s.RETRIES, s.BACKOFF_SECONDS, s.MAX_CONNECTIONS, s.USER_AGENT, s.CHUNK_BYTES) == (3, 2, 8, 'downloader/1.0', 65536)
