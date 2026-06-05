#!/usr/bin/env bash
# Build + run the bond/node reordering-for-locality experiment.
# Compiles the real scalar CGNR kernels in the SAME config the web build ships
# (js_stress_example/scripts/build.js: scalar, no hand-SIMD, -O3).
set -euo pipefail
cd "$(dirname "$0")"
CXX="${CXX:-clang++}"
INC=( -I ../ -I ../../../../include/globals -I ../../../../include/shared/NvFoundation )
DEFS=( -DNDEBUG=1 -DSTRESS_SOLVER_NO_SIMD=1 -DSTRESS_SOLVER_FORCE_SCALAR=1 -DSTRESS_SOLVER_NO_DEVICE_QUERY=1 )
FLAGS=( -std=c++17 -O3 -fno-exceptions -fno-rtti )
echo "Compiler: $($CXX --version | head -1)"
"$CXX" "${FLAGS[@]}" "${DEFS[@]}" "${INC[@]}" reorder_experiment.cpp ../stress.cpp -o /tmp/reorder_experiment
# benign stderr warning for intentionally-static test nodes is dropped
/tmp/reorder_experiment 2>/dev/null
