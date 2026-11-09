#!/usr/bin/env bash
# Build rwkv.cpp and the xenon_inference wrapper for CPU (AVX2).
# Use this on a machine with no NVIDIA GPU. The desktop app links build-cuda-app
# when that library exists, and falls back to this build otherwise.
# On the RTX 2060 laptop, run ./build_cuda.sh instead.
set -euo pipefail
cd "$(dirname "$0")"
./apply_patches.sh

cmake -B rwkv.cpp/build-cpu -S rwkv.cpp \
  -DRWKV_CUBLAS=OFF \
  -DCMAKE_BUILD_TYPE=Release
cmake --build rwkv.cpp/build-cpu --target rwkv -j"$(nproc)"

cmake -B build-cpu-app -S . \
  -DCMAKE_BUILD_TYPE=Release \
  -DXENON_RWKV_BUILD_DIR="$(pwd)/rwkv.cpp/build-cpu" \
  -DXENON_HAS_CUDA=OFF
cmake --build build-cpu-app -j"$(nproc)"

echo "CPU build is in $(pwd)/build-cpu-app."
