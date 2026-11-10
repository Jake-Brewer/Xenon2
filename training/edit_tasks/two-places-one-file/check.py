import pathlib
source = pathlib.Path('client.py').read_text()
assert source.count('MAX_RETRIES =') == 1, 'MAX_RETRIES is defined more than once'
import client
assert client.MAX_RETRIES == 5
def flaky(n):
    calls = []
    def f(*args):
        calls.append(1)
        if len(calls) < n:
            raise ConnectionError()
        return "ok"
    return f, calls
for name, args in (("fetch", ("u",)), ("upload", ("u", "b"))):
    f, calls = flaky(5)
    assert getattr(client, name)(f, *args) == "ok", name
    f, calls = flaky(99)
    try:
        getattr(client, name)(f, *args)
        raise AssertionError(name + " should give up")
    except ConnectionError:
        assert len(calls) == 5, (name, len(calls))
