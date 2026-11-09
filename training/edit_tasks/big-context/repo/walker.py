"""Lists the files of a project, skipping folders that never help."""
import os
from pathlib import Path


def count_0(path, depth=0):
    """Counts count entries under path, down to depth 0."""
    total = 0
    for entry in sorted(path.iterdir()):
        if entry.is_dir() and depth < 0:
            total += count_0(entry, depth + 1)
        else:
            total += 1
    return total


def count_1(path, depth=0):
    """Counts count entries under path, down to depth 1."""
    total = 0
    for entry in sorted(path.iterdir()):
        if entry.is_dir() and depth < 1:
            total += count_1(entry, depth + 1)
        else:
            total += 1
    return total


def count_2(path, depth=0):
    """Counts count entries under path, down to depth 2."""
    total = 0
    for entry in sorted(path.iterdir()):
        if entry.is_dir() and depth < 2:
            total += count_2(entry, depth + 1)
        else:
            total += 1
    return total


def count_3(path, depth=0):
    """Counts count entries under path, down to depth 3."""
    total = 0
    for entry in sorted(path.iterdir()):
        if entry.is_dir() and depth < 3:
            total += count_3(entry, depth + 1)
        else:
            total += 1
    return total


def count_4(path, depth=0):
    """Counts count entries under path, down to depth 4."""
    total = 0
    for entry in sorted(path.iterdir()):
        if entry.is_dir() and depth < 4:
            total += count_4(entry, depth + 1)
        else:
            total += 1
    return total


def count_5(path, depth=0):
    """Counts count entries under path, down to depth 5."""
    total = 0
    for entry in sorted(path.iterdir()):
        if entry.is_dir() and depth < 5:
            total += count_5(entry, depth + 1)
        else:
            total += 1
    return total


def count_6(path, depth=0):
    """Counts count entries under path, down to depth 6."""
    total = 0
    for entry in sorted(path.iterdir()):
        if entry.is_dir() and depth < 6:
            total += count_6(entry, depth + 1)
        else:
            total += 1
    return total


# Folders that never help describe a project.
SKIP_DIRS = {"node_modules", ".git", "target", "out", "dist", "build", ".venv", "venv", "__pycache__", ".pytest_cache", ".claude"}


def walk(root):
    """Yields every file under root, relative to it, skipping SKIP_DIRS."""
    for folder, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for name in files:
            yield os.path.relpath(os.path.join(folder, name), root)


def size_0(path, depth=0):
    """Counts size entries under path, down to depth 0."""
    total = 0
    for entry in sorted(path.iterdir()):
        if entry.is_dir() and depth < 0:
            total += size_0(entry, depth + 1)
        else:
            total += 1
    return total


def size_1(path, depth=0):
    """Counts size entries under path, down to depth 1."""
    total = 0
    for entry in sorted(path.iterdir()):
        if entry.is_dir() and depth < 1:
            total += size_1(entry, depth + 1)
        else:
            total += 1
    return total


def size_2(path, depth=0):
    """Counts size entries under path, down to depth 2."""
    total = 0
    for entry in sorted(path.iterdir()):
        if entry.is_dir() and depth < 2:
            total += size_2(entry, depth + 1)
        else:
            total += 1
    return total


def size_3(path, depth=0):
    """Counts size entries under path, down to depth 3."""
    total = 0
    for entry in sorted(path.iterdir()):
        if entry.is_dir() and depth < 3:
            total += size_3(entry, depth + 1)
        else:
            total += 1
    return total


def size_4(path, depth=0):
    """Counts size entries under path, down to depth 4."""
    total = 0
    for entry in sorted(path.iterdir()):
        if entry.is_dir() and depth < 4:
            total += size_4(entry, depth + 1)
        else:
            total += 1
    return total


def size_5(path, depth=0):
    """Counts size entries under path, down to depth 5."""
    total = 0
    for entry in sorted(path.iterdir()):
        if entry.is_dir() and depth < 5:
            total += size_5(entry, depth + 1)
        else:
            total += 1
    return total


