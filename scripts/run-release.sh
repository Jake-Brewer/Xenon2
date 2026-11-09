#!/usr/bin/env bash
# Builds the release app and (re)starts it. Under `npm run tauri dev`, any edit to the app's Rust
# restarts it, so Xenon couldn't change its own Rust code mid-run; a release build has no file
# watcher. The cost: changes to Xenon take effect when this is run again.
#
#   bash scripts/run-release.sh            # build, stop the running app, start the new one
#   XENON_MODEL=rwkv-7-g1l-2.9B-Q5_1.bin bash scripts/run-release.sh
set -euo pipefail
cd "$(dirname "$0")/../app"

if ! command -v cargo >/dev/null; then
  for bin in "$HOME"/.cargo/bin "$HOME"/.rustup/toolchains/stable-*/bin; do
    [ -x "$bin/cargo" ] || [ -x "$bin/cargo.exe" ] && export PATH="$bin:$PATH" && break
  done
fi

exe=src-tauri/target/release/app.exe
[ -f src-tauri/target/release/app ] && exe=src-tauri/target/release/app

# The running exe is locked on Windows, so stop it before the build replaces it.
if command -v powershell >/dev/null; then
  powershell -NoProfile -Command "Get-Process app -ErrorAction SilentlyContinue | Where-Object { \$_.Path -like '*target\\release\\app.exe' } | Stop-Process -Force"
else
  pkill -f "target/release/app" || true
fi

npm run tauri build -- --no-bundle
export XENON_MODEL="${XENON_MODEL:-rwkv-7-g1l-2.9B-Q5_1.bin}"
cd ..
nohup "app/$exe" >/dev/null 2>&1 &
echo "started $exe with $XENON_MODEL"
