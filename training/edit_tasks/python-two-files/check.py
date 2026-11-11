import pathlib
from config import DEFAULT_TIMEOUT
from client import fetch

assert DEFAULT_TIMEOUT == 30, f'DEFAULT_TIMEOUT should be 30, got {DEFAULT_TIMEOUT}'
assert fetch('http://x') == 'GET http://x timeout=30'

doc = pathlib.Path('client.py').read_text()
assert '30 seconds' in doc, 'docstring should mention 30 seconds'
assert '5 seconds' not in doc, 'old 5-second mention should be gone'
