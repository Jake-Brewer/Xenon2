import pathlib

src = pathlib.Path('lib.rs').read_text()

assert src.count('MAX_ITEMS:') == 1, 'MAX_ITEMS should be defined exactly once'
assert src.count('MAX_ITEMS') >= 3, 'MAX_ITEMS should appear at its definition and in both truncate and validate'

non_const = src.replace('MAX_ITEMS: usize = 10', '')
assert '10' not in non_const, 'the literal 10 should be replaced by MAX_ITEMS in both functions'
