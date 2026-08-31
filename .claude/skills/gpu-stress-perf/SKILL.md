---
name: gpu-stress-perf
description: Measure and optimise the CUDA stress solver with gpu_stress_suite — the topology scenarios it covers, the CPU-parity gates that are valid (and the one that silently is not), and the measurement traps that have each produced a confidently wrong answer here. Use before changing NvBlastExtStressGpu.cu and before believing any solver timing.
---

# Measuring and optimising the GPU stress solver

`demos/blast-stress-demo/tests/gpu_stress_suite.cpp` is the harness. It loads the
REAL pack (`assets/mini-city/fractured-downtown.json`), replicates it on a grid, and
runs 14 topologies from one intact building to total dust — checking correctness at
every point, not just speed. No game server, no PhysX scene, no cargo build.

```bash
cmake -S demos/blast-stress-demo -B demos/blast-stress-demo/build -DCMAKE_BUILD_TYPE=Release
cmake --build demos/blast-stress-demo/build -j8        # build IN-TREE: assets resolve
cd demos/blast-stress-demo/build                        # relative to the binary

./gpu_stress_suite --grid 2 --solves 10                 # all topologies
./gpu_stress_suite --grid 2 --histogram --only live-city # island size distribution
./gpu_stress_suite --grid 1 --iters 1000 --compare      # CPU parity gate
./gpu_stress_suite --grid 2 --destroy --ticks 1200 --limit-scale 0.05
./gpu_stress_suite --grid 2 --galerkin                  # AMG hierarchy check
BLAST_GPU_KERNEL_PROFILE=1 ./gpu_stress_suite --only live-city   # per-kernel
BLAST_GPU_GRAPH_STATS=1    ./gpu_stress_suite --destroy --ticks 2000  # host split
```

## Scenarios, and why the localised ones exist

`broken-N` removes bonds UNIFORMLY AT RANDOM. That is not what destruction looks
like, and the difference is not subtle: a live `/city` session at **9.4% of bonds
broken reported 1,353 solver islands**, while `broken-25` — nearly three times the
damage — produces **231**. Scattered single-bond cuts weaken structure; they almost
never disconnect it.

`live-light` / `live-city` / `live-heavy` model it the way the game applies damage:
a **core** radius that pulverises to free chunks, plus a **shell** radius that severs
only bonds STRADDLING the boundary, detaching the interior as a connected cluster.
The shell is the part that creates islands — solid spheres alone gave 8,948 free
chunks but only 137 islands. `live-city` is calibrated against real telemetry
(bonds within 0.6%, islands within 8%) and is the scenario to optimise against.

Blast counts scale with replicated area so damage DENSITY is constant across grids.
Recalibrate with `LIVE_BLASTS` / `LIVE_CORE` / `LIVE_SHELL` if the pack changes.

`--grid` 1..4 spans 74k to 1.19M bonds. Grid 2 is the sweet spot where the working
set fits the 4090's 72 MB L2; beyond it per-bond cost rises.

## The correctness gate — and the one that is silently invalid

**Use the equilibrium residual.** For every dynamic node, do its bonds' wrenches
balance the applied load? Computed host-side identically for both backends, with the
internal length/mass scale recovered by least squares, and restricted to ANCHORED
islands (a free-floating fragment is in freefall; no internal impulse can balance
gravity on it, and including those reports ~46 for a scene behaving perfectly).

**Do NOT gate on componentwise impulse error.** The shipped
`blast_stress_gpu_equivalence` compares impulses at a 2% tolerance, which works at
its 112-bond fixture and **reports 73x on a cold-start, fully converged, correct
solve** at scale. `null(B)` is most of the bond unknowns and the operator is
ill-conditioned, so a residual tolerance bounds the RESIDUAL, not the SOLUTION.

**The equilibrium gate is BLIND to a uniform scale error** — it fits a global scale
factor by least squares. Any change that touches impulse scaling must additionally be
gated on **bonds-broken from `--destroy`** and on **`peak |J|`**, where scenarios that
converge immediately (`broken-90/99`, `shatter-*`) must stay bit-identical.

**Never judge accuracy by bonds-broken alone**: a ~28% stress difference once produced
a 42x breakage difference.

## Measurement traps, each of which produced a wrong answer here

- **Run-to-run variance is several percent**, and worse on a shared GPU. A single
  pair is not a measurement. Use counterbalanced A/B at n>=6 and report medians and
  win counts. `perf-ab-measure` covers the protocol.
- **Profile share hides fixed cost.** Run the same scenario at 32 and 96 iterations
  and solve for the two terms: it revealed that **a third of the solve was fixed
  overhead**, larger than any per-iteration item, which no profile share showed.
- **`peak |J|` scales with `--solves`** because warm start accumulates. Never compare
  it across runs with different solve counts.
- **The `--destroy` end-state residual is only meaningful once the scene has settled.**
  Mid-collapse it swings 1e-2..6e-1 between identical runs, because a collapsing city
  genuinely is far from equilibrium. Use bonds-removed as the work counter.
- **A `.cu`-only edit can relink the OLD kernel.** Confirm the `.o` mtime AND size
  moved.

## Env flags (all default ON unless noted)

`BLAST_GPU_NODE_SPACE` node-space CGLS · `BLAST_GPU_COND_LOOP` + `BLAST_GPU_COND_CHUNK`
device-side early exit · `BLAST_GPU_DELTA_UPLOAD` sparse topology upload ·
`BLAST_GPU_DEFER_REPARTITION` exact split detection · `BLAST_GPU_SKIP_CONVERGED`
within-solve island skip · `BLAST_TOPO_UPLOAD=async|sync` ·
`BLAST_GPU_LOCAL_SPLIT` (**off**, changes physics) · `BLAST_GPU_JACOBI` (**off**,
does not pay) · `BLAST_GPU_VERIFY_PARTITION` (debug).
