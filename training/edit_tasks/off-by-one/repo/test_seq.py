from seq import first_n, last_n


def test_first():
    assert first_n([1, 2, 3], 2) == [1, 2]


def test_last():
    assert last_n([1, 2, 3, 4], 2) == [3, 4]
