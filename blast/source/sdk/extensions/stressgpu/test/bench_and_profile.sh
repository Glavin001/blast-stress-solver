#!/usr/bin/env bash
# Build the city-scale solve bench, time it, then profile it with ncu.
#
# Two outputs, and the second is the one that matters: wall/device timings say
# how much, `ncu` says WHY. Every previous attempt at this kernel was steered
# by a code comment; this is the replacement for that.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
blast="$(cd "$here/../../../../.." && pwd)"
cuda="${CUDA_PATH:-${CUDA_HOME:-/usr/local/cuda}}"
arch="${VIBE_CUDA_ARCH:-sm_89}"
out="${TMPDIR:-/tmp}/blast_gpu_solve_bench"

"$cuda/bin/nvcc" -std=c++17 -O3 -m64 "-arch=$arch" -lineinfo -Xcompiler -fPIC \
  -DNVBLAST_ENABLE_CUDA_STRESS \
  -I"$blast/include/extensions/stressgpu" \
  -I"$blast/include/extensions/stress" \
  "$blast/source/sdk/extensions/stressgpu/NvBlastExtStressGpu.cu" \
  "$here/gpu_solve_bench.cpp" \
  -o "$out" -lcuda -lcudart

if [ "${1:-}" = "--run-only" ]; then
  "$out"
  exit 0
fi

echo "=== timings ==="
"$out"

echo
echo "=== ncu: per-kernel time, occupancy, memory/compute throughput ==="
# Fewer solves under the profiler: ncu serialises and replays every launch.
BENCH_SOLVES=3 "$cuda/bin/ncu" \
  --metrics \
gpu__time_duration.sum,\
sm__throughput.avg.pct_of_peak_sustained_elapsed,\
gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed,\
sm__warps_active.avg.pct_of_peak_sustained_active,\
launch__occupancy_limit_registers,\
l1tex__throughput.avg.pct_of_peak_sustained_active,\
lts__throughput.avg.pct_of_peak_sustained_elapsed \
  --csv --target-processes all "$out" 2>&1 | tail -n +1
