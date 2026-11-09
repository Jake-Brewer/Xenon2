#!/usr/bin/env bash
# Build rwkv.cpp and the xenon_inference wrapper with CUDA, for an RTX 2060 (sm_75).
# The desktop app links build-cuda-app ahead of the CPU build. See README.md.
set -euo pipefail
cd "$(dirname "$0")"
./apply_patches.sh

if ! command -v nvcc >/dev/null 2>&1; then
  echo "nvcc is not on PATH. Install the CUDA toolkit, then re-run." >&2
  echo "This script compiles for compute capability 7.5 (GeForce RTX 2060)." >&2
  exit 1
fi

cmake -B rwkv.cpp/build-cuda -S rwkv.cpp \
  -DRWKV_CUBLAS=ON \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=75
cmake --build rwkv.cpp/build-cuda --target rwkv -j"$(nproc)"

cmake -B build-cuda-app -S . \
  -DCMAKE_BUILD_TYPE=Release \
  -DXENON_RWKV_BUILD_DIR="$(pwd)/rwkv.cpp/build-cuda" \
  -DXENON_HAS_CUDA=ON
cmake --build build-cuda-app -j"$(nproc)"

echo "CUDA build is in $(pwd)/build-cuda-app (sm_75, RTX 2060)."
