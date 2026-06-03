# Benchmarking the destructible pipeline

Performance tooling for `blast-stress-solver`, behind the **`bench-support`** feature
(off by default, never shipped in a release). Everything shares one driver,
`src/bench_harness.rs`, so a criterion number, the spike profiler, and the regression
guard all describe the *same* work: the full Rapier pipeline + contact-force injection +
fracture + topology edits + the demo's resimulation loop.

## Run it

```bash
# 1) Statistically-stable hot-path numbers (warm-up, 100 samples, outlier detection).
cargo bench --bench destruction --features bench-support

#    A/B a change against a saved baseline:
cargo bench --bench destruction --features bench-support -- --save-baseline before
#    …make the change…
cargo bench --bench destruction --features bench-support -- --baseline before

# 2) Per-frame spike / percentile profile over a realistic destruction timeline
#    (mean / p50 / p90 / p99 / MAX, plus where the time goes and a quality fingerprint).
cargo run --release --example frame_profile --features bench-support -- --frames 300

# 3) Anti-cheating regression guard (determinism + genuine-destruction + golden bands).
cargo test --features bench-support --test perf_regression_test -- --nocapture

# Force the scalar solver (disable the native AVX/FMA path) for cross-checking:
BLAST_FORCE_SCALAR=1 cargo bench --bench destruction --features bench-support
```

## Scenarios

Programmatic builders (any size, deterministic): `wall(span, height, layers)`,
`tower(side, stories)`, `bridge(...)`. Committed scene packs (real fractured geometry):
`fractured-tower`, `fractured-bridge`, `brick-building`. Generated large structure:
`high-rise` (930 nodes / 3261 bonds, ~1.57 M kg — regenerate with
`cd ../blast-stress-solver && npm run build:ts` if absent; it is git-ignored).

`Material::{Strong, Weak}` presets pick "holds under gravity" vs "shatters", so each
scenario cleanly exercises either the steady solver cost or the fracture/topology/resim path.

## What the baseline showed (where the time goes)

Measured release, per-frame mean unless noted:

- **Steady, nothing breaking** (the common 120 FPS case): dominated by the C++ CGNR
  stress solve. `idle_skip` (on by default) skips it entirely once forces settle, so it
  is only paid on actively-loaded frames.
- **Cascade frames** (a structure letting go into hundreds of fragments in one frame):
  the spike was dominated by **Rapier topology edits** (`handle_split`) — *not* the
  stress solve.
- **Resimulation**: extra cost appears only on fracture frames (the rollback re-runs the
  physics step). It is bounded by `max_passes` and is **Rapier's `PhysicsPipeline::step`**,
  not Blast code.

## Optimizations landed (same physics, faster)

| Change | Where | Effect |
|---|---|---|
| Native AVX/FMA solver (runtime CPUID-gated, scalar fallback) | `build.rs` | ~7–11% per stress solve on x86_64 |
| Split-planner fast path for 1 parent → N children | `split_migrator.rs` | avoids O(C³) Hungarian on cascades |
| Deterministic split application (ordered bodies/children, handle tie-breaks) | `body_tracker.rs` | reproducible sim + no per-split HashMap allocs |
| Per-solve scratch reuse | `NvBlastExtStressSolver.cpp`, `ext_stress_solver.rs` | no per-frame heap alloc in the hot path |

Measured: tower 256-body cascade spike **47.2 ms → 16.2 ms** (topology edits 32.9 ms →
0.76 ms); high-rise **0.89  ms → 0.63 ms/frame** mean. The discrete fracture outcome
(which bonds break, body/actor counts) is now bit-stable run-to-run.

## Where the remaining cost is (and why it's left)

After the above, the largest per-frame spikes are **Rapier's physics step re-run during
resimulation**. This is engine-internal, not Blast code, so the realistic levers are
Rapier-level: (a) make settled debris sleep (already supported via `SleepThresholdOptions`
— sleeping bodies are skipped by Rapier's solver, which is the "contacts at scale" answer),
and (b) island-local resimulation (re-step only the fractured island) — a future Rapier
integration change. Reducing fracture work to go faster is intentionally **not** done: it
would change the result, which the regression guard forbids.
