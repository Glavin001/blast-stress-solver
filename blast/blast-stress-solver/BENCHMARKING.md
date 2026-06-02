# Benchmarking the destructible pipeline (TS / WASM)

Performance tooling for the JavaScript/TypeScript `blast-stress-solver` runtime — the
web counterpart of the Rust crate's harness
([`../blast-stress-solver-rs/BENCHMARKING.md`](../blast-stress-solver-rs/BENCHMARKING.md)).
Two layers, mirroring the Rust split:

1. **Split-planner micro-benchmark** — the pure topology-diff a fracture runs, in isolation.
   No physics, no stress solve, **no WASM**, so it runs straight from source.
2. **Full-pipeline benchmark** — the whole Rapier step + contact injection + fracture +
   topology edits + resim loop, with per-phase timing. Requires the WASM build.

## 1. Split-planner micro-benchmark (no build required)

When a body fractures into `N` fragments the runtime runs a *split planner* — it decides
which existing Rapier bodies to **reuse** (keep handle + colliders + pose: the cheapest
edit) vs. which fragments need a **new** body, by maximizing node overlap. The original
planner padded this to a single dense Hungarian — `O(max(M,N)^3)` — so one cascade into a
few hundred fragments stalled a frame for tens-to-hundreds of milliseconds.

The shipping planner ([`src/rapier/splitMigrator.ts`](src/rapier/splitMigrator.ts)) solves
it per **connected component** of the body↔child overlap graph, with an argmax fast path
for single-body / single-child components — provably the same optimal assignment, in
near-linear time. The original dense Hungarian is retained as
`planSplitMigrationReference` so the two can be A/B'd and proven equivalent.

```bash
# A/B the shipping planner vs. the reference dense Hungarian across sizes.
npm run bench:split-planner            # full sweep, human table
node scripts/bench-split-planner.mjs --quick    # smaller/faster sweep
node scripts/bench-split-planner.mjs --json      # machine-readable
node scripts/bench-split-planner.mjs --scenario=shatter
```

It bundles the pure-TS planner on the fly with esbuild, so no `npm run build` / emscripten
runtime is needed. Scenarios mirror the Rust `apply_profile` example:

- **shatter (1×N)** — one body lets go into `N` fragments (the common cascade). The shipping
  planner solves it with a single argmax; the reference pads to an `N×N` Hungarian.
- **merge (M→M/2)** — `2N` singleton bodies regroup into `N` two-node children, each pulling
  nodes from two bodies. The reference runs one dense `(2N)²` Hungarian; the shipping planner
  sees `N` tiny independent components.

### What it shows (release Node, ms per plan)

| scenario | N | before (dense) | after (components) | speedup | same optimum? |
|---|--:|--:|--:|--:|:--:|
| shatter | 128 | 9.86 ms | 0.047 ms | ~208× | ✓ |
| shatter | 256 | 68.1 ms | 0.096 ms | ~710× | ✓ |
| shatter | 512 | 519 ms | 0.19 ms | ~2,740× | ✓ |
| merge | 256 | 72.3 ms | 0.36 ms | ~426× | ✓ |
| merge | 512 | 4,045 ms | 0.72 ms | ~5,610× | ✓ |

The shipping planner stays near-linear and keeps going (`N = 4096`: shatter ≈ 1.7 ms,
merge ≈ 7 ms) far past where the reference is even feasible. **Every measured size reaches
the same optimum** — the speedup is free.

### See it live

The [`split-planner-bench.html`](../js_stress_example/split-planner-bench.html) demo charts
this in the browser on a log-log scale, with the old planner rocketing past the 60 fps
frame budget and the new one staying flat. Linked from the demo index as
**⚡ Split-Planner Benchmark**.

### Correctness guards (the speed is free)

- [`src/tests/rapier.splitMigrator.equivalence.test.ts`](src/tests/rapier.splitMigrator.equivalence.test.ts) —
  proves the shipping planner equals the reference Hungarian **and** an independent
  brute-force optimum on cascades, genuine multi-body reparenting, ties, exact matches, a
  400-case randomized sweep, and at large scale; plus determinism and the TS fixed/support
  invariants. (Mirrors the Rust `split_migrator` test module.)
- [`src/tests/rapier.splitPlanner.perf.test.ts`](src/tests/rapier.splitPlanner.perf.test.ts) —
  guards that the `O(N^3)` cliff stays gone (sub-cubic scaling) with deliberately loose,
  non-flaky bounds.

```bash
npx vitest run src/tests/rapier.splitMigrator.equivalence.test.ts \
               src/tests/rapier.splitMigrator.test.ts \
               src/tests/rapier.splitPlanner.perf.test.ts
```

## 2. Full-pipeline benchmark (requires WASM build)

```bash
npm run build            # builds WASM (needs emscripten) + TS
npm run bench            # medium tower, single projectile — per-phase JSON
npm run bench:small      # small tower (3×4)
npm run bench:large      # large tower (8×20)

# Broader scenario matrix (scale, structures, collision patterns, configs) with
# per-phase mean/p50/p95/p99/max tables:
npx vitest run src/tests/rapier.perf.test.ts
```

[`scripts/bench.mjs`](scripts/bench.mjs) drives the full pipeline headlessly via the core's
profiler and prints per-phase timing (Rapier step / stress solve / contact drain / fracture
/ body create / collider rebuild / snapshot / resim). The per-frame profiler sample also
carries `splitPlannerMs` and `splitChildCounts`, so the split-planner cost above is visible
in-context during a real cascade.

### Live in-browser frame profiler

`src/rapier/frameProfiler.ts` turns that same `CoreProfilerSample` stream into a rolling,
chartable breakdown — `FrameProfilerBuffer` (ring buffer + per-phase stats + dominant-cause
detection) and `drawFrameProfilerChart` (a stacked per-phase area chart with the true
`totalMs` line and the 60 fps budget). The **Tower Collapse** demo
([`../js_stress_example/tower-collapse.html`](../js_stress_example/tower-collapse.html))
overlays it live, so a dip below 60 fps is immediately attributable to a phase (physics,
stress solve, fracture, **split planning**, topology edits, snapshots…).

It also has an **A/B toggle** — `setProfiler({ measureReferencePlanner: true })` times the
old dense-Hungarian planner on each frame's real splits (result discarded, the sim keeps the
fast planner) and records `splitPlannerReferenceMs`; the HUD draws the projected old frame
time as a dashed line, so you can watch the spike the optimization removed reappear during a
cascade. The breakdown logic + chart are unit-tested in
[`src/tests/rapier.frameProfiler.test.ts`](src/tests/rapier.frameProfiler.test.ts).

## Optimizations landed (same result, faster)

| Change | Where | Effect |
|---|---|---|
| Connected-component split assignment (argmax fast path; dense Hungarian only per small component) | `splitMigrator.ts` | `O(max(M,N)^3)` → near-linear; 256-fragment cascade plan **68 ms → 0.1 ms**, optimal-equivalent |
| Pooled per-body resim snapshots (reused buffer, no per-frame alloc) | `bodySnapshots.ts` | no `O(bodies)` heap churn per captured frame |
| Per-solve scratch reuse in the C++ CGNR solver | `NvBlastExtStressSolver.cpp` (shared WASM) | no per-frame heap alloc in the stress hot path |

The native AVX/FMA solver path the Rust crate enables is x86-only and **not** applicable to
the WASM build (wasm stays scalar), so it is intentionally not ported here.

## Where the remaining cost is

As on the Rust side, after the planner fix the largest per-frame spikes during heavy
destruction are **Rapier's physics step** (including any resimulation re-run), which is
engine-internal rather than Blast code. The realistic levers are Rapier-level: let settled
debris sleep (`SleepThresholdOptions` — sleeping bodies are skipped by the solver) and cap
fracture/body work per frame via `FracturePolicy`. Reducing the fracture *outcome* to go
faster is intentionally not done — the equivalence guards forbid buying speed with a worse
result.
