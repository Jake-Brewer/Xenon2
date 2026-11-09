from walker import SKIP_DIRS, walk, count_3, size_2
assert 'models' in SKIP_DIRS
assert SKIP_DIRS >= {'node_modules', '.git', 'target', 'out', 'dist', 'build', '.venv', 'venv', '__pycache__', '.pytest_cache', '.claude'}, SKIP_DIRS
