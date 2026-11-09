import pathlib
import shop, report
assert shop.order_total([1.0, 2.0], 0.5) == 1.5
assert report.summary("Ann", [1.0, 2.0]) == "Ann: 2 items, 3.00"
assert shop.average_price([2, 4]) == 3
for name in ("shop.py", "report.py", "test_shop.py"):
    assert "calc_total" not in pathlib.Path(name).read_text(), name
