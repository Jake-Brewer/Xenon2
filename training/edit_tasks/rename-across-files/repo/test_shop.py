from shop import calc_total


def test_total():
    assert calc_total([1.0, 2.0]) == 3.0
