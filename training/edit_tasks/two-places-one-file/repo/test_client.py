from client import fetch


def test_fetch_returns():
    assert fetch(lambda url: "ok", "u") == "ok"
