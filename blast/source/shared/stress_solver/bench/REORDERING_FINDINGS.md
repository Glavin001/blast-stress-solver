# Experiment: bond/node reordering for cache locality — does it speed up the CGNR solve?

**Question.** The coupling mat-vec (`CouplingMatrixOps::rmul`/`lmul`) scatters/gathers into node
slots via bond-endpoint indices — an irregular access pattern. SpMV literature says reordering a
sparse matrix for locality (Cuthill-McKee / bandwidth reduction) can speed the mat-vec up to
~2.6×. **Does it help *this* solver?**

**TL;DR — No, not for this workload.** Against the order the solver actually uses, reordering wins
**~0%** at every scale, and it costs bit-exactness (the solution drifts within tolerance). The big
SpMV-reordering wins only materialize against a *deliberately bad* (shuffled) order, which this
solver never produces — and the ~2.6× regime needs a DRAM-bound matrix, while these problems are
L1/L3-resident. **Recommendation: do not ship it.**

## How to run

```bash
cd blast/source/shared/stress_solver/bench
./reorder_run.sh            # builds the real scalar CGNR kernels (wasm config) + runs
```

It builds three orderings of the *same* matrix and times `rmul`/`lmul` on each, then measures the
solution drift:
- **natural** — as generated (nodes in (x,y,z) order — already decent locality)
- **shuffled** — random node renumber + random bond order (worst-case locality)
- **CM-local** — Cuthill-McKee node renumber (BFS) + bonds sorted by min endpoint (best case)

## Results (clang 18, `-O3`, scalar path; ns per bond)

| scale | node-vec footprint | shuffled vs natural | **CM-local vs natural** | shuffled→CM headroom |
|---|---|---|---|---|
| **building 6×6×12** (one island the island-aware path solves; M=432) | 14 KB → **L1** | rmul −2% / lmul −2% | **rmul ~0% / lmul ~0%** | ~0% (all in L1) |
| **city, 28 buildings** (≈ recording's whole-graph scale; M=8208) | 256 KB → **L2/L3** | rmul +5% / lmul **+25%** | **rmul −1% / lmul −1%** | rmul 6% / lmul 21% |
| **big-block 40³** (DRAM-scale stress test, not a real scene; M=64000) | 2 MB → **L2/L3** | rmul **+61%** / lmul **+90%** | **rmul −2% / lmul −6%** | rmul 39% / lmul 50% |

Solution drift (CM-local vs natural, island-aware solve, relative L2): **2e‑6 … 7e‑6** — well inside
the 1e‑3 solver tolerance, but **non-zero → not bit-exact** (reordering changes the `rmul`
scatter-accumulation order).

## Why it doesn't help here

1. **The realistic order is already local.** Two structural facts make "natural" as good as CM:
   - the island-aware path's gather renumbers each island's nodes **first-touch (BFS-like)**, which
     *is* a bandwidth-reducing order — the free, bit-exact part of reordering is already captured;
   - building/lattice topology means consecutive bonds naturally touch nearby nodes.
   So CM-on-top buys ~0%. The 21–50% "headroom" the table shows only recovers a **shuffled** order,
   which the solver never has.
2. **The working set is cache-resident.** The web build solves island-**aware** → each island is a
   few-hundred-node, ~tens-of-KB sub-system that lives in L1/L2; even the whole 7.3k-node mini-city
   is ~256 KB → L3. The literature's ~2.6× is for matrices that spill to **DRAM** (tens of MB). At
   2 MB (big-block) a *bad* order already costs +61–90%, but a good order (natural/CM) doesn't —
   confirming the penalty is "bad locality," not "needs reordering."
3. **It's not free.** Reordering changes the FP accumulation order in `rmul`, so the solved impulses
   move (within tolerance). For ~0% upside that's a bad trade — and it would add per-frame
   permute/un-permute of the impulse vector to integrate.

## Recommendation

Skip bond reordering. The bit-exact slice of locality (first-touch island numbering) is already in
place. The real levers remain: **SIMD** the coupling mat-vec (compute-bound, ~50% of the solve) and
the already-shipped **island-grouping cache** (removes redundant per-frame setup). If a future scene
ever pushes a single island past L3 (millions of bonds), revisit — but cap-the-island-size would hit
first.
