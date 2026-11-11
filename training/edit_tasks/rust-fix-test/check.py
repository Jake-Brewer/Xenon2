import pathlib, shutil, subprocess

src = pathlib.Path('src/lib.rs').read_text()
# The test must stay as it was: the fix belongs in clamp.
assert 'clamp_works' in src, 'test must not be deleted'
assert 'assert_eq!(clamp(0, 1, 10), 1' in src, 'assertion for below-min must stay'
assert 'assert_eq!(clamp(11, 1, 10), 10' in src, 'assertion for above-max must stay'

# cargo is often not on PATH on Windows; rustup keeps it here.
cargo = shutil.which('cargo') or str(pathlib.Path.home() / '.cargo' / 'bin' / 'cargo')
if not pathlib.Path(cargo).exists() and not shutil.which('cargo'):
    cargo = str(pathlib.Path.home() / '.rustup/toolchains/stable-x86_64-pc-windows-msvc/bin/cargo.exe')
result = subprocess.run([cargo, 'test', '--lib', '-q'], capture_output=True, text=True)
assert result.returncode == 0, f'cargo test failed:\n{result.stdout}\n{result.stderr}'
