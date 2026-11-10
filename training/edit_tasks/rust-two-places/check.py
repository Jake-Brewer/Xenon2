import subprocess, sys, pathlib

src = pathlib.Path('lib.rs').read_text()

# MAX_ITEMS must be defined exactly once and used in both functions.
assert src.count('MAX_ITEMS:') == 1, 'MAX_ITEMS should be defined exactly once'
assert src.count('MAX_ITEMS') >= 3, 'MAX_ITEMS should appear at its definition and in both truncate and validate'

# The literal 10 must not appear in the non-test code.
non_test = src.split('#[cfg(test)]')[0]
assert '10' not in non_test.replace('MAX_ITEMS: usize = 10', ''), \
    'the literal 10 should be replaced by MAX_ITEMS in both functions'

result = subprocess.run(['cargo', 'test', '--quiet'], capture_output=True, text=True)
if result.returncode != 0:
    print(result.stdout)
    print(result.stderr)
    sys.exit(1)
