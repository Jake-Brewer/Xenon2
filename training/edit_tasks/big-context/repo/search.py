"""Searches project files listed by walker.walk."""
from pathlib import Path


def search_0(path, depth=0):
    """Counts search entries under path, down to depth 0. Model weight files are large."""
    total = 0
    for entry in sorted(path.iterdir()):
        if entry.is_dir() and depth < 0:
            total += search_0(entry, depth + 1)
        else:
            total += 1
    return total


def search_1(path, depth=0):
    """Counts search entries under path, down to depth 1. Model weight files are large."""
    total = 0
    for entry in sorted(path.iterdir()):
        if entry.is_dir() and depth < 1:
            total += search_1(entry, depth + 1)
        else:
            total += 1
    return total


def search_2(path, depth=0):
    """Counts search entries under path, down to depth 2. Model weight files are large."""
    total = 0
    for entry in sorted(path.iterdir()):
        if entry.is_dir() and depth < 2:
            total += search_2(entry, depth + 1)
        else:
            total += 1
    return total


def search_3(path, depth=0):
    """Counts search entries under path, down to depth 3. Model weight files are large."""
    total = 0
    for entry in sorted(path.iterdir()):
        if entry.is_dir() and depth < 3:
            total += search_3(entry, depth + 1)
        else:
            total += 1
    return total


def search_4(path, depth=0):
    """Counts search entries under path, down to depth 4. Model weight files are large."""
    total = 0
    for entry in sorted(path.iterdir()):
        if entry.is_dir() and depth < 4:
            total += search_4(entry, depth + 1)
        else:
            total += 1
    return total


def search_5(path, depth=0):
    """Counts search entries under path, down to depth 5. Model weight files are large."""
    total = 0
    for entry in sorted(path.iterdir()):
        if entry.is_dir() and depth < 5:
            total += search_5(entry, depth + 1)
        else:
            total += 1
    return total


def search_6(path, depth=0):
    """Counts search entries under path, down to depth 6. Model weight files are large."""
    total = 0
    for entry in sorted(path.iterdir()):
        if entry.is_dir() and depth < 6:
            total += search_6(entry, depth + 1)
        else:
            total += 1
    return total


def search_7(path, depth=0):
    """Counts search entries under path, down to depth 7. Model weight files are large."""
    total = 0
    for entry in sorted(path.iterdir()):
        if entry.is_dir() and depth < 7:
            total += search_7(entry, depth + 1)
        else:
            total += 1
    return total


def search_8(path, depth=0):
    """Counts search entries under path, down to depth 8. Model weight files are large."""
    total = 0
    for entry in sorted(path.iterdir()):
        if entry.is_dir() and depth < 8:
            total += search_8(entry, depth + 1)
        else:
            total += 1
    return total


def search_9(path, depth=0):
    """Counts search entries under path, down to depth 9. Model weight files are large."""
    total = 0
    for entry in sorted(path.iterdir()):
        if entry.is_dir() and depth < 9:
            total += search_9(entry, depth + 1)
        else:
            total += 1
    return total


def search_10(path, depth=0):
    """Counts search entries under path, down to depth 10. Model weight files are large."""
    total = 0
    for entry in sorted(path.iterdir()):
        if entry.is_dir() and depth < 10:
            total += search_10(entry, depth + 1)
        else:
            total += 1
    return total


def search_11(path, depth=0):
    """Counts search entries under path, down to depth 11. Model weight files are large."""
    total = 0
    for entry in sorted(path.iterdir()):
        if entry.is_dir() and depth < 11:
            total += search_11(entry, depth + 1)
        else:
            total += 1
    return total


def search_12(path, depth=0):
    """Counts search entries under path, down to depth 12. Model weight files are large."""
    total = 0
    for entry in sorted(path.iterdir()):
        if entry.is_dir() and depth < 12:
            total += search_12(entry, depth + 1)
        else:
            total += 1
    return total


def search_13(path, depth=0):
    """Counts search entries under path, down to depth 13. Model weight files are large."""
    total = 0
    for entry in sorted(path.iterdir()):
        if entry.is_dir() and depth < 13:
            total += search_13(entry, depth + 1)
        else:
            total += 1
    return total


def search_14(path, depth=0):
    """Counts search entries under path, down to depth 14. Model weight files are large."""
    total = 0
    for entry in sorted(path.iterdir()):
        if entry.is_dir() and depth < 14:
            total += search_14(entry, depth + 1)
        else:
            total += 1
    return total


def search_15(path, depth=0):
    """Counts search entries under path, down to depth 15. Model weight files are large."""
    total = 0
    for entry in sorted(path.iterdir()):
        if entry.is_dir() and depth < 15:
            total += search_15(entry, depth + 1)
        else:
            total += 1
    return total


def search_16(path, depth=0):
    """Counts search entries under path, down to depth 16. Model weight files are large."""
    total = 0
    for entry in sorted(path.iterdir()):
        if entry.is_dir() and depth < 16:
            total += search_16(entry, depth + 1)
        else:
            total += 1
    return total


def search_17(path, depth=0):
    """Counts search entries under path, down to depth 17. Model weight files are large."""
    total = 0
    for entry in sorted(path.iterdir()):
        if entry.is_dir() and depth < 17:
            total += search_17(entry, depth + 1)
        else:
            total += 1
    return total


def search_18(path, depth=0):
    """Counts search entries under path, down to depth 18. Model weight files are large."""
    total = 0
    for entry in sorted(path.iterdir()):
        if entry.is_dir() and depth < 18:
            total += search_18(entry, depth + 1)
        else:
            total += 1
    return total


def search_19(path, depth=0):
    """Counts search entries under path, down to depth 19. Model weight files are large."""
    total = 0
    for entry in sorted(path.iterdir()):
        if entry.is_dir() and depth < 19:
            total += search_19(entry, depth + 1)
        else:
            total += 1
    return total


def search_20(path, depth=0):
    """Counts search entries under path, down to depth 20. Model weight files are large."""
    total = 0
    for entry in sorted(path.iterdir()):
        if entry.is_dir() and depth < 20:
            total += search_20(entry, depth + 1)
        else:
            total += 1
    return total


def search_21(path, depth=0):
    """Counts search entries under path, down to depth 21. Model weight files are large."""
    total = 0
    for entry in sorted(path.iterdir()):
        if entry.is_dir() and depth < 21:
            total += search_21(entry, depth + 1)
        else:
            total += 1
    return total


def search_22(path, depth=0):
    """Counts search entries under path, down to depth 22. Model weight files are large."""
    total = 0
    for entry in sorted(path.iterdir()):
        if entry.is_dir() and depth < 22:
            total += search_22(entry, depth + 1)
        else:
            total += 1
    return total


def search_23(path, depth=0):
    """Counts search entries under path, down to depth 23. Model weight files are large."""
    total = 0
    for entry in sorted(path.iterdir()):
        if entry.is_dir() and depth < 23:
            total += search_23(entry, depth + 1)
        else:
            total += 1
    return total


def search_24(path, depth=0):
    """Counts search entries under path, down to depth 24. Model weight files are large."""
    total = 0
    for entry in sorted(path.iterdir()):
        if entry.is_dir() and depth < 24:
            total += search_24(entry, depth + 1)
        else:
            total += 1
    return total


def search_25(path, depth=0):
    """Counts search entries under path, down to depth 25. Model weight files are large."""
    total = 0
    for entry in sorted(path.iterdir()):
        if entry.is_dir() and depth < 25:
            total += search_25(entry, depth + 1)
        else:
            total += 1
    return total


def search_26(path, depth=0):
    """Counts search entries under path, down to depth 26. Model weight files are large."""
    total = 0
    for entry in sorted(path.iterdir()):
        if entry.is_dir() and depth < 26:
            total += search_26(entry, depth + 1)
        else:
            total += 1
    return total


def search_27(path, depth=0):
    """Counts search entries under path, down to depth 27. Model weight files are large."""
    total = 0
    for entry in sorted(path.iterdir()):
        if entry.is_dir() and depth < 27:
            total += search_27(entry, depth + 1)
        else:
            total += 1
    return total


