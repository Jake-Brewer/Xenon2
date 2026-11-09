import pytest
from shapes import area, perimeter
assert area(2, 3) == 6
assert area(0, 5) == 0
for w, h in ((-1, 3), (3, -1)):
    with pytest.raises(ValueError):
        area(w, h)
assert perimeter(2, 3) == 10
