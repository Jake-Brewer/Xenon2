from walker import SKIP_DIRS


def test_skips_git():
    assert ".git" in SKIP_DIRS
