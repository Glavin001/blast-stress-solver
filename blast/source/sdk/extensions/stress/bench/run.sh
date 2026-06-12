#!/usr/bin/env bash
# Bit-exactness A/B + timing for the settled-bond stress-recompute skip in
# ExtStressSolverImpl::updateBondStress. Builds the Ext stress solver in the SAME scalar config the
# web build ships, with island-aware + skip-settled ON (the only mode the skip fires in), twice:
# default (skip on) and -DSTRESS_NO_BONDSTRESS_SKIP (skip off). The two builds must produce a
# byte-identical hash of the per-frame stress output (and, for the fracture harness, fracture
# decisions + topology), proving the skip changes nothing it shouldn't.
set -euo pipefail
cd "$(dirname "$0")"
BLAST="../../../../.."   # repo blast/ root (bench/ is 5 levels down: bench/stress/extensions/sdk/source)
CXX="${CXX:-clang++}"
SRCS=(
  "$BLAST/rust_stress_example/ffi/stress_bridge.cpp" "$BLAST/rust_stress_example/ffi/ext_stress_bridge.cpp"
  "$BLAST/source/shared/stress_solver/stress.cpp" "$BLAST/source/sdk/extensions/stress/NvBlastExtStressSolver.cpp"
  "$BLAST/source/sdk/common/NvBlastAssert.cpp" "$BLAST/source/sdk/common/NvBlastAtomic.cpp"
  "$BLAST/source/sdk/common/NvBlastTime.cpp" "$BLAST/source/sdk/common/NvBlastTimers.cpp"
  "$BLAST/source/sdk/globals/NvBlastGlobals.cpp" "$BLAST/source/sdk/globals/NvBlastInternalProfiler.cpp"
  "$BLAST/source/sdk/lowlevel/NvBlastActor.cpp" "$BLAST/source/sdk/lowlevel/NvBlastActorSerializationBlock.cpp"
  "$BLAST/source/sdk/lowlevel/NvBlastAsset.cpp" "$BLAST/source/sdk/lowlevel/NvBlastAssetHelper.cpp"
  "$BLAST/source/sdk/lowlevel/NvBlastFamily.cpp" "$BLAST/source/sdk/lowlevel/NvBlastFamilyGraph.cpp"
)
INC=( -I "$BLAST/rust_stress_example/ffi" -I "$BLAST/source/shared/stress_solver" -I "$BLAST/source/shared"
  -I "$BLAST/source/shared/NsFoundation/include" -I "$BLAST/include" -I "$BLAST/include/globals" -I "$BLAST/include/shared"
  -I "$BLAST/include/lowlevel" -I "$BLAST/include/extensions" -I "$BLAST/include/extensions/stress"
  -I "$BLAST/include/shared/NvFoundation" -I "$BLAST/source/sdk/common" -I "$BLAST/source/sdk/globals"
  -I "$BLAST/source/sdk/lowlevel" -I "$BLAST/source/sdk/extensions/stress" )
DEF=( -DNDEBUG=1 -DSTRESS_SOLVER_FORCE_SCALAR=1 -DSTRESS_SOLVER_NO_SIMD=1 -DSTRESS_SOLVER_NO_DEVICE_QUERY=1 )
FLAGS=( -std=c++17 -O3 )

build_run() { # <harness.cpp> <label>
  "$CXX" "${FLAGS[@]}" "${DEF[@]}" "${INC[@]}" "$1" "${SRCS[@]}" -o /tmp/bss_on
  "$CXX" "${FLAGS[@]}" "${DEF[@]}" -DSTRESS_NO_BONDSTRESS_SKIP=1 "${INC[@]}" "$1" "${SRCS[@]}" -o /tmp/bss_off
  echo "── $2 ──"
  local on off
  on=$(/tmp/bss_on); off=$(/tmp/bss_off)
  echo "$on" | sed 's/^/  skip-on  /'
  echo "$off" | grep -E "ms/solve" | sed 's/^/  skip-off /' || true
  if [ "$(echo "$on" | grep hash)" = "$(echo "$off" | grep hash)" ]; then
    echo "  ✅ BIT-EXACT (skip-on hash == skip-off hash)"
  else echo "  ❌ DIVERGED"; exit 1; fi
}

echo "Compiler: $($CXX --version | head -1)"
build_run ext_bondstress_settle_ab.cpp   "settle + drive (no breakage): bit-exact + timing"
build_run ext_bondstress_fracture_ab.cpp "full fracture cycle (partial damage): bit-exact"
