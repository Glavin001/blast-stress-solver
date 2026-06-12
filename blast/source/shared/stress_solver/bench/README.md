# CGNR stress-solver: bottleneck profiling + algorithmic optimization

Tooling and findings for making the **core CGNR solve faster without changing its output**
("CGNR solve (WASM)" is ~62% of `solverUpdate` — 4.32 ms/frame — in the mini-city recording).

This directory is about the **algorithmic / memory-traffic / O(n)** structure of the solve.
Hand-SIMD of the kernels is tracked separately and is orthogonal to (and stacks on top of)
everything here — the wins below are independent of whether the kernels are scalar or SIMD.

> **TL;DR**
> - The solve is **~60% sparse mat-vec** (`rmul`/`lmul`) + **~40% streaming vector ops**
>   (`vmadd`×2, `calculate_error`, `vnmadd`, `length_sq`), and at realistic island sizes it
>   is **compute/auto-vectorization-bound, not memory-bandwidth-bound** (the working set is
>   cache-resident).
> - **Shipped (bit-exact): island-grouping cache** in `stress.cpp`. The per-frame union-find +
>   island assignment + CSR grouping is a pure function of topology, so it is now rebuilt only
>   when the topology changes. **Island-aware steady-state ~20% faster (mean), ~35% faster on
>   settled/converged frames** — same iterations, **bit-identical** solution (verified).
> - **Tried and rejected (measured): reduction fusion.** Folding `|z|²`/`|s|²` into the producing
>   mat-vec is bit-exact but **slower** here — it defeats the `-O3` auto-vectorizer the scalar
>   build relies on, and the "saved" pass was a cache hit, not bandwidth. Kept out of production.

## Why a native C++ harness

The web build compiles the **scalar** CGNR path (`js_stress_example/scripts/build.js`:
`-DSTRESS_SOLVER_FORCE_SCALAR -DSTRESS_SOLVER_NO_SIMD -O3 -msimd128`). The hand-written AVX
kernels are x86-only and excluded; the build leans on the compiler auto-vectorizing the scalar
loops. This harness compiles the **exact same scalar kernels** (`stress.cpp` + the solver
headers, same defines) natively, so we can attribute cost per kernel and prove bit-exactness
**without an emscripten toolchain**. The kernels are pure compute (no syscalls), so the native
scalar build is a faithful proxy for the wasm scalar build's *relative* breakdown.

## Run it

```bash
cd blast/source/shared/stress_solver/bench
./run.sh                       # build + full report + bit-exactness gate (exit!=0 on drift)
./run.sh --update-golden       # re-bless golden_fingerprints.txt after an intended change
CXX=g++ ./run.sh               # other compiler
```

`run.sh` builds the harness, prints the report, then **gates on the solution fingerprint**: a
hash of the solved impulses (+ iteration count + residual) for each scenario must match the
committed `golden_fingerprints.txt`, and the cache-on build must match a cache-off build. Any
"same output, faster" change keeps these green; anything that perturbs the numerics fails it.

## What it measures (three views + a lock)

1. **Isolated kernel microbenchmarks** — clean ns/element throughput per kernel (no timer in
   the loop), at an island-sized building and at city scale.
2. **Instrumented solve** — wraps the real ops so the *real* `CGNR::solve` attributes time per
   kernel across its real iterations/convergence (faithful *shares*; a few % timer overhead).
3. **End-to-end** `StressProcessor::solve` — the production path (prepare + island-aware solve),
   cold first frame + warm steady-state, ms/solve at production settings (maxIter 25, tol 1e-3).
4. **Fingerprint lock** — the bit-exactness regression gate described above.

## Findings — where the CGNR time goes

Per-kernel share of one `CGNR::solve`, consistent at island scale (M=432, N=1116) and city
scale (M=8208, N=20880):

| kernel | % of solve | what it is | streaming passes |
|---|--:|---|---|
| `rmul` (B·p) | ~34% | SpMV: `memset` + scatter-accumulate + 2 cross products / bond | write M, rmw scatter N, pass M |
| `lmul` (Bᵀ·r) | ~27% | SpMV: gather + 2 cross products / bond | pass M, gather N, write N |
| `vmadd` (p,x updates, **2×/iter**) | ~22% | AXPY over N | read+write N (×2) |
| `calculate_error` \|z\|² | ~8% | reduction over N | read N |
| `vnmadd` (r update) | ~5% | AXPY over M | read+write M |
| `length_sq` \|s\|² | ~4% | reduction over M | read M |

Key reads:
- **Mat-vecs ≈ 60%, streaming vector ops ≈ 40%.** The vector ops are *not* negligible — `vmadd`
  alone is ~22% (two AXPYs over N per iteration).
- **It's compute/auto-vectorization-bound, not bandwidth-bound** at these sizes: an island's
  vectors are cache-resident (a 432-node island's `z` is ~35 KB). That single fact is why the
  "obvious" memory-traffic optimization below does **not** help.
- The `AngLin6` element is **32 B but only 24 B is used** (two padding floats for SIMD
  alignment). The scalar path pays 25% extra footprint on every streaming op — but the padding
  is required by the SIMD path, so repacking is a SIMD-path tradeoff, not pursued here.

## Optimization shipped: island-grouping cache (bit-exact)

`StressProcessor::solveIslandAware` rebuilt the **island partition every frame**: union-find over
all bonds, a per-bond island id, and a CSR grouping (counting sort) — plus two `std::vector`
allocations. That work is a **pure function of topology** (bond node indices + which nodes are
static); it is identical every frame until a bond is added/removed.

The grouping is now cached and rebuilt only when `m_islandTopoValid` is false, which `prepare()`
and `removeBond()` set on any topology change. The per-island solves read the cached grouping and
are therefore **bit-identical**; only topology-invariant *setup* is skipped.

**Measured (this harness, city = 28 buildings, 28 islands, 20880 bonds), same-session A/B:**

| island-aware solve | baseline (rebuild/frame) | cached | speedup |
|---|--:|--:|--:|
| warm steady-state, mean | ~1.74 ms | ~1.42 ms | **~1.2×** |
| settled/converged frame ("best") | ~0.83 ms | ~0.54 ms | **~1.5×** |

The converged-frame win is the headline: when islands are settled (the common case in the
recording, with `skipSettled`), almost all the per-frame cost *was* this recomputed setup. The
first frame after a topology change is unchanged (it rebuilds the cache, as before).

Invalidation is **global per topology change** (any `prepare`/`removeBond` rebuilds the whole
grouping), so the win lands on **topology-stable frames** — which dominate once structures settle
and between destruction bursts; a frame in which a bond actually breaks falls back to a full
rebuild (i.e. the original cost, never worse). Finer-grained / incremental invalidation (a broken
bond only dirties islands it touched) is a possible follow-up but is not bit-trivial (removing a
bond can split an island), so it's deferred.

**Bit-exactness:** `run.sh` proves it two ways — the cached build matches the committed golden
(generated from the pre-change code), and matches a cache-disabled build
(`-DSTRESS_SOLVER_NO_ISLAND_CACHE`) including a scenario that removes bonds mid-session to
exercise invalidation. Hashes are byte-identical (`99b1b02517a6b08f` island-aware,
`f937242099bcbc61` with removeBond).

Touched files: `stress.h` (cache members) and `stress.cpp` (`solveIslandAware`, `prepare`,
`removeBond`). No kernel/`cgnr.h` changes, so it does not collide with SIMD work.

## Experiment rejected: reduction fusion (bit-exact but slower)

A natural idea: fold the two per-iteration reductions into the mat-vec that produces their
operand — `|Bᵀr|²` into `lmul`'s `Cᵀ` pass, `|Bp|²` into `rmul`'s inertia pass — removing two
streaming passes/iteration. It is **bit-exact** (same values, same summation order). But the
harness microbenchmarks show it **loses**:

| | separate (2 passes) | fused (1 pass) |
|---|--:|--:|
| `lmul` + `calculate_error` | ~117 µs | ~127 µs ❌ |
| `rmul` + `length_sq` | ~135 µs | ~132 µs ≈ |

End-to-end it was a net regression. Two reasons, both visible in the data: (1) the result vector
is **cache-resident**, so the "saved" re-read was an L1/L2 hit, not bandwidth; (2) the standalone
`calculate_error` is a clean reduction the `-O3` auto-vectorizer handles very well, while fusing
it into the cross-product loop **inhibits vectorization**. The scalar wasm build leans entirely
on that auto-vectorizer, so fusion is the wrong move *for this path*. (Local copies of the fused
kernels live in `cgnr_bench.cpp` purely to keep this A/B reproducible — production is unchanged.)
This could flip under hand-SIMD, where the producer controls vectorization; worth a re-measure if
the kernels are SIMD-ized.

## Is there newer (2025/2026) research that applies?

CGNR (CG on the normal equations, `(BᵀB)x = Bᵀb`) is classical; the modern literature is about
**numerical stability** and **SpMV engineering**, not a faster recurrence:

- **CGLS / LSQR over CGNR.** Applying CG to the explicit normal equations squares the condition
  number (κ(BᵀB)=κ(B)²); CGLS/LSQR are the numerically-stabler factored forms. They change the
  *iterates/solution* (and they're for accuracy, not speed), so they're **out of scope** under a
  no-output-change constraint — noting it for completeness. (Stanford SOL CGLS; Björck, *Stability
  of CG/Lanczos for least squares*, SIMAX.)
- **SpMV is memory-bandwidth + cache-miss bound**, and **reordering** (RCM/bandwidth reduction,
  and 2024–2025 input/output-swapping work, *"Is Sparse Matrix Reordering Effective for SpMV?"*,
  arXiv:2506.10356) speeds SpMV up to ~2.6×. But reordering bonds **changes the scatter
  accumulation order** in `rmul` → not bit-exact → a no-go here unless "within tolerance" is
  acceptable. The island-aware gather already gives each island first-touch-local numbering, so it
  captures much of the locality benefit *for free and bit-exactly*.
- **Preconditioning** (Jacobi/diagonal) cuts iteration count but changes the iterates/solution →
  excluded by the no-output-change rule (and "fewer iterations" is explicitly off the table).

Net: the standard research levers either change the output (reordering, preconditioning, LSQR) or
target ill-conditioning rather than throughput. The bit-exact lever the literature *does* support
— **cut redundant work / memory traffic around an inherently memory-bound kernel** — is exactly
the island-grouping cache (eliminate per-frame recompute), and, for the kernels themselves, SIMD.

## Ranked further opportunities (evidence-based)

1. **Cache the per-island gather too** (`localC`/`localI`/local node renumbering), not just the
   grouping. It's also topology-invariant; only the RHS (velocities) and impulse gather/scatter
   change per frame. Expected to remove most of the *remaining* converged-frame setup (~0.4 ms →
   approaching the ~0.12 ms of actual convergence-check mat-vec). Bigger refactor (per-island
   storage incl. shared static nodes; restructure the interleaved skip-check), so it needs the
   shared-static test case added here — left as the next step, gated by the same fingerprint lock.
2. **SIMD the kernels** (separate effort). The breakdown says `rmul`+`lmul`+`vmadd` ≈ 82% of the
   solve are wide, regular float ops — the textbook SIMD target, and the largest single lever.
3. **Optional, *not* bit-exact (needs sign-off):** bond reordering for SpMV locality (~up to
   2.6× per literature) and/or diagonal preconditioning for fewer iterations. Both change the
   numerics within tolerance; only pursue if "looks the same" is allowed to mean "within solver
   tolerance" rather than "bit-identical".
