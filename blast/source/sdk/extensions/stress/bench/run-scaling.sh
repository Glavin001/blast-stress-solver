#!/usr/bin/env bash
# Scaling demo for the settled-bond stress skip: builds the Ext solver (island-aware + skip-settled)
# default (skip on) vs -DSTRESS_NO_BONDSTRESS_SKIP (off) and joins the per-frame solve cost across a
# city-size sweep (3 buildings active = localized action) and an activity sweep (fixed 219k-bond city,
# varying how many buildings are active). The saving tracks the settled bonds skipped.
set -euo pipefail
cd "$(dirname "$0")"
BLAST="../../../../.."
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
"$CXX" -std=c++17 -O3 "${DEF[@]}"                          "${INC[@]}" ext_bondstress_scaling.cpp "${SRCS[@]}" -o /tmp/bss_scale_on
"$CXX" -std=c++17 -O3 "${DEF[@]}" -DSTRESS_NO_BONDSTRESS_SKIP=1 "${INC[@]}" ext_bondstress_scaling.cpp "${SRCS[@]}" -o /tmp/bss_scale_off
/tmp/bss_scale_off 2>/dev/null > /tmp/bss_off.txt
/tmp/bss_scale_on  2>/dev/null > /tmp/bss_on.txt
echo "scene      bonds    active  OFFms   ONms   saved  ratio  settled/total"
paste /tmp/bss_off.txt /tmp/bss_on.txt | awk '/^#/{print "  "$1" "$2" "$3" "$4" "$5" "$6" "$7" "$8" "$9; next}
  /^[a-z]/{ if($5+0>0){ printf "%-9s %8s %6s  %6.2f %6.2f %6.2f  %.1fx  %s\n",$1,$3,$4,$5,$11,$5-$11,$5/$11,$12 } }'
