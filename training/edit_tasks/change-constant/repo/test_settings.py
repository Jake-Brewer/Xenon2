import settings


def test_timeout_is_a_minute():
    assert settings.REQUEST_TIMEOUT_SECONDS == 60
