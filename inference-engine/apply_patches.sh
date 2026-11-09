#!/usr/bin/env bash
# Applies patches/ to the rwkv.cpp submodule and its ggml, once. Safe to run again: a patch that
# is already in place is skipped. build_cpu.sh and build_cuda.sh run this first.
#   ggml-rwkv-wkv7.patch       adds GGML_OP_RWKV_WKV7 (CPU + CUDA), ported from upstream ggml
#   rwkv-cpp-native-wkv7.patch makes rwkv.cpp's RWKV-7 graph use it instead of a CPU-only custom op
#   rwkv-cpp-state-on-gpu.patch keeps each token's state on the GPU (66 graph splits a token -> 2)
#   rwkv-cpp-state-stays-on-device.patch lets the state stay there between tokens (no 20 MB copies)
set -euo pipefail
cd "$(dirname "$0")"

apply() {
  local dir=$1 patch=$PWD/patches/$2
  if git -C "$dir" apply --reverse --check --ignore-whitespace "$patch" 2>/dev/null; then
    echo "patch already applied: $2"
  else
    git -C "$dir" apply --ignore-whitespace "$patch"
    echo "patch applied: $2"
  fi
}

apply rwkv.cpp/ggml ggml-rwkv-wkv7.patch
apply rwkv.cpp rwkv-cpp-native-wkv7.patch
apply rwkv.cpp rwkv-cpp-state-on-gpu.patch
apply rwkv.cpp rwkv-cpp-state-stays-on-device.patch
