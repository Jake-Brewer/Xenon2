"""Formats order summaries."""
from shop import calc_total


def summary(name, prices):
    """One line: who ordered how many items for how much."""
    return f"{name}: {len(prices)} items, {calc_total(prices):.2f}"
