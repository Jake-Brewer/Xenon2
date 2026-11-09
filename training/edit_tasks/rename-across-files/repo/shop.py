"""Order arithmetic."""


def calc_total(prices, discount=0.0):
    """The sum of prices, less a fractional discount."""
    return round(sum(prices) * (1 - discount), 2)


def average_price(prices):
    """The mean price, or 0 for no prices."""
    return sum(prices) / len(prices) if prices else 0
