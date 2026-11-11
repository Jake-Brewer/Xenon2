import pathlib, subprocess, sys

src = pathlib.Path('lib.rs').read_text()
# The test must stay intact.
assert 'clamp_works' in src, 'test must not be deleted'
assert 'assert_eq!(clamp(0, 1, 10), 1' in src, 'assertion for below-min must stay'
assert 'assert_eq!(clamp(11, 1, 10), 10' in src, 'assertion for above-max must stay'

# Write a minimal Cargo.toml pointing at lib.rs in the root (not src/).
cargo_toml = pathlib.Path('Cargo.toml')
if not cargo_toml.exists():
    cargo_toml.write_text(
        '[package]\nname = "task"\nversion = "0.1.0"\nedition = "2021"\n\n'
        '[lib]\npath = "lib.rs"\n'
    )
result = subprocess.run(['cargo', 'test', '--lib', '-q'], capture_output=True, text=True)
assert result.returncode == 0, f'cargo test failed:\n{result.stdout}\n{result.stderr}'
