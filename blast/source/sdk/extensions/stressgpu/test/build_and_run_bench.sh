#!/bin/bash
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
blast="$(cd "$here/../../../../.." && pwd)"
cuda="${CUDA_PATH:-${CUDA_HOME:-/usr/local/cuda}}"
arch="${VIBE_CUDA_ARCH:-sm_89}"
out="${TMPDIR:-/tmp}/blast_bond_stress_bench"
"$cuda/bin/nvcc" -std=c++17 -O2 -m64 "-arch=$arch" -Xcompiler -fPIC \
  -Xcompiler -mavx -Xcompiler -mfma \
  -DNVBLAST_ENABLE_CUDA_STRESS \
  -I"$blast/include/extensions/stressgpu" -I"$blast/include/extensions/stress" \
  "$blast/source/sdk/extensions/stressgpu/NvBlastExtStressGpu.cu" \
  "$here/bond_stress_bench.cu" \
  -o "$out" -lcuda -lcudart
"$out" "${1:-200}"
