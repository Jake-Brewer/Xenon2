"""Sequence helpers."""


def first_n(items, n):
    """The first n items."""
    return items[:n]


def last_n(items, n):
    """The last n items."""
    return items[-n - 1:]


def middle(items):
    """The middle item, or the left one of the two middle items."""
    return items[(len(items) - 1) // 2]
