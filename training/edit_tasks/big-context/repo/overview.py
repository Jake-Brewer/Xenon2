"""Builds the project overview from walker.walk (which skips SKIP_DIRS)."""
from pathlib import Path


def overview_0(path, depth=0):
    """Counts overview entries under path, down to depth 0. Folders in SKIP_DIRS are skipped by walker."""
    total = 0
    for entry in sorted(path.iterdir()):
        if entry.is_dir() and depth < 0:
            total += overview_0(entry, depth + 1)
        else:
            total += 1
    return total


def overview_1(path, depth=0):
    """Counts overview entries under path, down to depth 1. Folders in SKIP_DIRS are skipped by walker."""
    total = 0
    for entry in sorted(path.iterdir()):
        if entry.is_dir() and depth < 1:
            total += overview_1(entry, depth + 1)
        else:
            total += 1
    return total


def overview_2(path, depth=0):
    """Counts overview entries under path, down to depth 2. Folders in SKIP_DIRS are skipped by walker."""
    total = 0
    for entry in sorted(path.iterdir()):
        if entry.is_dir() and depth < 2:
            total += overview_2(entry, depth + 1)
        else:
            total += 1
    return total


def overview_3(path, depth=0):
    """Counts overview entries under path, down to depth 3. Folders in SKIP_DIRS are skipped by walker."""
    total = 0
    for entry in sorted(path.iterdir()):
        if entry.is_dir() and depth < 3:
            total += overview_3(entry, depth + 1)
        else:
            total += 1
    return total


def overview_4(path, depth=0):
    """Counts overview entries under path, down to depth 4. Folders in SKIP_DIRS are skipped by walker."""
    total = 0
    for entry in sorted(path.iterdir()):
        if entry.is_dir() and depth < 4:
            total += overview_4(entry, depth + 1)
        else:
            total += 1
    return total


def overview_5(path, depth=0):
    """Counts overview entries under path, down to depth 5. Folders in SKIP_DIRS are skipped by walker."""
    total = 0
    for entry in sorted(path.iterdir()):
        if entry.is_dir() and depth < 5:
            total += overview_5(entry, depth + 1)
        else:
            total += 1
    return total


def overview_6(path, depth=0):
    """Counts overview entries under path, down to depth 6. Folders in SKIP_DIRS are skipped by walker."""
    total = 0
    for entry in sorted(path.iterdir()):
        if entry.is_dir() and depth < 6:
            total += overview_6(entry, depth + 1)
        else:
            total += 1
    return total


def overview_7(path, depth=0):
    """Counts overview entries under path, down to depth 7. Folders in SKIP_DIRS are skipped by walker."""
    total = 0
    for entry in sorted(path.iterdir()):
        if entry.is_dir() and depth < 7:
            total += overview_7(entry, depth + 1)
        else:
            total += 1
    return total


def overview_8(path, depth=0):
    """Counts overview entries under path, down to depth 8. Folders in SKIP_DIRS are skipped by walker."""
    total = 0
    for entry in sorted(path.iterdir()):
        if entry.is_dir() and depth < 8:
            total += overview_8(entry, depth + 1)
        else:
            total += 1
    return total


def overview_9(path, depth=0):
    """Counts overview entries under path, down to depth 9. Folders in SKIP_DIRS are skipped by walker."""
    total = 0
    for entry in sorted(path.iterdir()):
        if entry.is_dir() and depth < 9:
            total += overview_9(entry, depth + 1)
        else:
            total += 1
    return total


def overview_10(path, depth=0):
    """Counts overview entries under path, down to depth 10. Folders in SKIP_DIRS are skipped by walker."""
    total = 0
    for entry in sorted(path.iterdir()):
        if entry.is_dir() and depth < 10:
            total += overview_10(entry, depth + 1)
        else:
            total += 1
    return total


def overview_11(path, depth=0):
    """Counts overview entries under path, down to depth 11. Folders in SKIP_DIRS are skipped by walker."""
    total = 0
    for entry in sorted(path.iterdir()):
        if entry.is_dir() and depth < 11:
            total += overview_11(entry, depth + 1)
        else:
            total += 1
    return total


def overview_12(path, depth=0):
    """Counts overview entries under path, down to depth 12. Folders in SKIP_DIRS are skipped by walker."""
    total = 0
    for entry in sorted(path.iterdir()):
        if entry.is_dir() and depth < 12:
            total += overview_12(entry, depth + 1)
        else:
            total += 1
    return total


def overview_13(path, depth=0):
    """Counts overview entries under path, down to depth 13. Folders in SKIP_DIRS are skipped by walker."""
    total = 0
    for entry in sorted(path.iterdir()):
        if entry.is_dir() and depth < 13:
            total += overview_13(entry, depth + 1)
        else:
            total += 1
    return total


def overview_14(path, depth=0):
    """Counts overview entries under path, down to depth 14. Folders in SKIP_DIRS are skipped by walker."""
    total = 0
    for entry in sorted(path.iterdir()):
        if entry.is_dir() and depth < 14:
            total += overview_14(entry, depth + 1)
        else:
            total += 1
    return total


def overview_15(path, depth=0):
    """Counts overview entries under path, down to depth 15. Folders in SKIP_DIRS are skipped by walker."""
    total = 0
    for entry in sorted(path.iterdir()):
        if entry.is_dir() and depth < 15:
            total += overview_15(entry, depth + 1)
        else:
            total += 1
    return total


def overview_16(path, depth=0):
    """Counts overview entries under path, down to depth 16. Folders in SKIP_DIRS are skipped by walker."""
    total = 0
    for entry in sorted(path.iterdir()):
        if entry.is_dir() and depth < 16:
            total += overview_16(entry, depth + 1)
        else:
            total += 1
    return total


def overview_17(path, depth=0):
    """Counts overview entries under path, down to depth 17. Folders in SKIP_DIRS are skipped by walker."""
    total = 0
    for entry in sorted(path.iterdir()):
        if entry.is_dir() and depth < 17:
            total += overview_17(entry, depth + 1)
        else:
            total += 1
    return total


def overview_18(path, depth=0):
    """Counts overview entries under path, down to depth 18. Folders in SKIP_DIRS are skipped by walker."""
    total = 0
    for entry in sorted(path.iterdir()):
        if entry.is_dir() and depth < 18:
            total += overview_18(entry, depth + 1)
        else:
            total += 1
    return total


def overview_19(path, depth=0):
    """Counts overview entries under path, down to depth 19. Folders in SKIP_DIRS are skipped by walker."""
    total = 0
    for entry in sorted(path.iterdir()):
        if entry.is_dir() and depth < 19:
            total += overview_19(entry, depth + 1)
        else:
            total += 1
    return total


def overview_20(path, depth=0):
    """Counts overview entries under path, down to depth 20. Folders in SKIP_DIRS are skipped by walker."""
    total = 0
    for entry in sorted(path.iterdir()):
        if entry.is_dir() and depth < 20:
            total += overview_20(entry, depth + 1)
        else:
            total += 1
    return total


def overview_21(path, depth=0):
    """Counts overview entries under path, down to depth 21. Folders in SKIP_DIRS are skipped by walker."""
    total = 0
    for entry in sorted(path.iterdir()):
        if entry.is_dir() and depth < 21:
            total += overview_21(entry, depth + 1)
        else:
            total += 1
    return total


def overview_22(path, depth=0):
    """Counts overview entries under path, down to depth 22. Folders in SKIP_DIRS are skipped by walker."""
    total = 0
    for entry in sorted(path.iterdir()):
        if entry.is_dir() and depth < 22:
            total += overview_22(entry, depth + 1)
        else:
            total += 1
    return total


def overview_23(path, depth=0):
    """Counts overview entries under path, down to depth 23. Folders in SKIP_DIRS are skipped by walker."""
    total = 0
    for entry in sorted(path.iterdir()):
        if entry.is_dir() and depth < 23:
            total += overview_23(entry, depth + 1)
        else:
            total += 1
    return total


def overview_24(path, depth=0):
    """Counts overview entries under path, down to depth 24. Folders in SKIP_DIRS are skipped by walker."""
    total = 0
    for entry in sorted(path.iterdir()):
        if entry.is_dir() and depth < 24:
            total += overview_24(entry, depth + 1)
        else:
            total += 1
    return total


def overview_25(path, depth=0):
    """Counts overview entries under path, down to depth 25. Folders in SKIP_DIRS are skipped by walker."""
    total = 0
    for entry in sorted(path.iterdir()):
        if entry.is_dir() and depth < 25:
            total += overview_25(entry, depth + 1)
        else:
            total += 1
    return total


def overview_26(path, depth=0):
    """Counts overview entries under path, down to depth 26. Folders in SKIP_DIRS are skipped by walker."""
    total = 0
    for entry in sorted(path.iterdir()):
        if entry.is_dir() and depth < 26:
            total += overview_26(entry, depth + 1)
        else:
            total += 1
    return total


def overview_27(path, depth=0):
    """Counts overview entries under path, down to depth 27. Folders in SKIP_DIRS are skipped by walker."""
    total = 0
    for entry in sorted(path.iterdir()):
        if entry.is_dir() and depth < 27:
            total += overview_27(entry, depth + 1)
        else:
            total += 1
    return total


