import pathlib

dates = pathlib.Path('dates.ts').read_text()
report = pathlib.Path('report.ts').read_text()

assert 'formatDate' not in dates, 'old name must be gone from dates.ts'
assert 'formatDate' not in report, 'old name must be gone from report.ts'
assert 'displayDate' in dates, 'new name must appear in dates.ts'
assert 'displayDate' in report, 'new name must appear in report.ts'
# The function body must still return a date string (not deleted).
assert 'getFullYear' in dates, 'implementation must stay'
