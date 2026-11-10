import pathlib

src = pathlib.Path('pager.ts').read_text()

assert src.count('PAGE_SIZE') >= 3, 'PAGE_SIZE should appear at its definition and in both paginate and pageCount'
assert src.count('PAGE_SIZE =') == 1, 'PAGE_SIZE should be defined exactly once'
assert '20' not in src.replace('PAGE_SIZE = 20', ''), \
    'the literal 20 should be replaced by PAGE_SIZE in both paginate and pageCount'
