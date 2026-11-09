from text import slugify, shout


def test_slugify_drops_digits():
    assert slugify("Room 101") == "room"


def test_slugify_words():
    assert slugify("Hello, World") == "hello-world"


def test_shout():
    assert shout("hi") == "HI!"
