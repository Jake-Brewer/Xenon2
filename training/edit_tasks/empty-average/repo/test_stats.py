from stats import average


def test_average():
    assert average([1, 2, 3]) == 2


def test_average_of_nothing_is_zero():
    assert average([]) == 0
