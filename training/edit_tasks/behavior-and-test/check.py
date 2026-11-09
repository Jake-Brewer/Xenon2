import pathlib, subprocess, sys
from text import slugify, shout
assert slugify("Room 101") == "room-101"
assert slugify("Hello, World") == "hello-world"
assert shout("hi") == "HI!"
tests = pathlib.Path("test_text.py").read_text()
# The test must still test slugify with digits, now expecting them kept, not be deleted.
assert "room-101" in tests, "the test should now expect digits to be kept"
assert "hello-world" in tests and "HI!" in tests, "other tests must stay"
assert subprocess.run([sys.executable, "-m", "pytest", "-q", "test_text.py"]).returncode == 0
