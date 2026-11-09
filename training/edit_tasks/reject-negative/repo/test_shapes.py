import pytest
from shapes import area


def test_area():
    assert area(2, 3) == 6


def test_negative():
    with pytest.raises(ValueError):
        area(-1, 3)
