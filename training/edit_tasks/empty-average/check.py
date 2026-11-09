import stats
assert stats.average([]) == 0
assert stats.average([2, 4]) == 3
for i in range(1, 26):
    assert getattr(stats, f'helper_{i}')(0) == i, i
