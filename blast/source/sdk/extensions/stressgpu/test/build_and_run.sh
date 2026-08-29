#!/bin/bash
# Build and run the settled-island skip equivalence test. Needs nvcc and a GPU.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
blast="$(cd "$here/../../../../.." && pwd)"
cuda="${CUDA_PATH:-${CUDA_HOME:-/usr/local/cuda}}"
arch="${VIBE_CUDA_ARCH:-sm_89}"
out="${TMPDIR:-/tmp}/blast_gpu_settled_skip_test"

"$cuda/bin/nvcc" -std=c++17 -O2 -m64 "-arch=$arch" -Xcompiler -fPIC \
  -DNVBLAST_ENABLE_CUDA_STRESS \
  -I"$blast/include/extensions/stressgpu" \
  -I"$blast/include/extensions/stress" \
  "$blast/source/sdk/extensions/stressgpu/NvBlastExtStressGpu.cu" \
  "$here/gpu_settled_skip_test.cpp" \
  -o "$out" -lcuda -lcudart
"$out"
