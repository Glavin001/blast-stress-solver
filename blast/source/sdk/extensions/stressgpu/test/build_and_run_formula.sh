#!/bin/bash
# Build and run the bond-stress formula unit tests. Needs nvcc and a GPU.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
blast="$(cd "$here/../../../../.." && pwd)"
cuda="${CUDA_PATH:-${CUDA_HOME:-/usr/local/cuda}}"
arch="${VIBE_CUDA_ARCH:-sm_89}"
out="${TMPDIR:-/tmp}/blast_stress_formula_test"

# The host half is built exactly as the library builds it -- -mfma matters,
# because gcc contracts mul+add by default and that is precisely the
# host/device difference these tests are here to pin down.
"$cuda/bin/nvcc" -std=c++17 -O2 -m64 "-arch=$arch" \
  -Xcompiler -fPIC -Xcompiler -mavx -Xcompiler -mfma \
  -I"$blast/include/extensions/stress" \
  "$here/stress_formula_test.cu" \
  -o "$out" -lcudart
"$out"
