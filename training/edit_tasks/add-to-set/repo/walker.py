"""Lists the files of a project, skipping folders that never help."""
import os

# Folders that never help describe a project.
SKIP_DIRS = {"node_modules", ".git", "target", "out", "dist", "build", ".venv", "venv", "__pycache__", ".pytest_cache", ".claude"}


def walk(root):
    """Yields every file under root, relative to it, skipping SKIP_DIRS."""
    for folder, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for name in files:
            yield os.path.relpath(os.path.join(folder, name), root)
