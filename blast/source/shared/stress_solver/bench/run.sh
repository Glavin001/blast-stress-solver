#!/usr/bin/env bash
# Build + run the CGNR scalar-path benchmark natively.
#
# Compiles stress.cpp and the harness with the SAME defines the web build uses
# (js_stress_example/scripts/build.js): scalar CGNR, no hand-SIMD, -O3. Emscripten adds
# -msimd128 (auto-vectorization of the scalar loops); the native -O3 auto-vectorizer
# plays the same role here, so the *relative* kernel breakdown transfers. The goal is
# bottleneck attribution + a bit-exactness regression lock, not wall-clock parity with wasm.
#
# Exit status is non-zero if the solver's solution fingerprints drift from the committed
# golden (bench/golden_fingerprints.txt) — i.e. if a "same output, faster" change wasn't.
set -euo pipefail
cd "$(dirname "$0")"

CXX="${CXX:-clang++}"
INC=( -I ../ -I ../../../../include/globals -I ../../../../include/shared/NvFoundation )
DEFS=( -DNDEBUG=1 -DSTRESS_SOLVER_NO_SIMD=1 -DSTRESS_SOLVER_FORCE_SCALAR=1 -DSTRESS_SOLVER_NO_DEVICE_QUERY=1 )
FLAGS=( -std=c++17 -O3 -fno-exceptions -fno-rtti )
GOLDEN="golden_fingerprints.txt"

echo "Compiler: $($CXX --version | head -1)"
echo "Flags   : ${FLAGS[*]} ${DEFS[*]}"
"$CXX" "${FLAGS[@]}" "${DEFS[@]}" "${INC[@]}" cgnr_bench.cpp ../stress.cpp -o /tmp/cgnr_bench

# The solver prints a benign stderr warning for intentionally-static test nodes; drop it.
/tmp/cgnr_bench 2>/dev/null | tee /tmp/cgnr_bench.out
grep -E "islandAware=.*hash=" /tmp/cgnr_bench.out > /tmp/cgnr_fp.txt || true

# A/B: the same harness with the island grouping cache disabled (rebuild every frame, the
# original behavior). Its fingerprints must equal the default (cached) build's — proof that
# caching is bit-exact, not just "close".
"$CXX" "${FLAGS[@]}" "${DEFS[@]}" -DSTRESS_SOLVER_NO_ISLAND_CACHE=1 "${INC[@]}" cgnr_bench.cpp ../stress.cpp -o /tmp/cgnr_bench_nocache
/tmp/cgnr_bench_nocache 2>/dev/null | grep -E "islandAware=.*hash=" > /tmp/cgnr_fp_nocache.txt || true

echo
echo "================= BIT-EXACTNESS REGRESSION CHECK ================="
status=0
if [[ "${1:-}" == "--update-golden" ]]; then
  cp /tmp/cgnr_fp.txt "$GOLDEN"; echo "golden updated → $GOLDEN"; cat "$GOLDEN"; exit 0
fi
if [[ ! -f "$GOLDEN" ]]; then
  echo "no golden yet; creating $GOLDEN from this run (review & commit it)"; cp /tmp/cgnr_fp.txt "$GOLDEN"; cat "$GOLDEN"
elif diff -q "$GOLDEN" /tmp/cgnr_fp.txt >/dev/null; then
  echo "✅ vs golden       : MATCH — solver output is bit-identical to the committed reference."
else
  echo "❌ vs golden       : DIFFER — output changed!"; diff "$GOLDEN" /tmp/cgnr_fp.txt || true; status=1
fi
if diff -q /tmp/cgnr_fp.txt /tmp/cgnr_fp_nocache.txt >/dev/null; then
  echo "✅ cache vs rebuild : MATCH — the island grouping cache is bit-exact (incl. removeBond)."
else
  echo "❌ cache vs rebuild : DIFFER — caching changed the output!"; diff /tmp/cgnr_fp_nocache.txt /tmp/cgnr_fp.txt || true; status=1
fi
exit $status
