# Rapier / resim performance — findings & output-safe levers

Analysis of a real **mini-city** session recording (`blast-sim-recording/v1`,
709 frames / ~18 s, debris growing from 2 → **1355 rigid bodies**), plus
reproducible micro-benchmarks. The goal throughout: **make the large-scene
real-time path faster while keeping the simulation output bit-for-bit identical**
(no smaller world, no fewer solver iterations, no lower resim frequency — those
change behaviour and are out of scope).

The tooling below is built so every candidate optimisation can be **measured**
(did it help?) and **proven faithful** (did the trajectory stay identical?).

## TL;DR

On a large, complex scene the cost is two rapier-side levers:

1. **`world.step()` is the dominant phase (≈46 % of every frame)** and scales
   **super-linearly** with accumulated debris: in the benchmark, 131 → 1027
   bodies (7.8×) drove rapierStep **0.40 ms → 10.55 ms (26×)**.
2. **Resimulation doubles the frame on every fracture.** On a fracture the world
   is rolled back and `world.step()` runs a *second* time over the *whole* world.
   In the recording this fired on **40 % of frames** and accounted for **~28 % of
   all wall-clock time**; a resim frame costs **~2.3× a body-count-matched
   non-resim frame**.

Neither lever requires changing output to attack — the wasted work is
**re-simulating bodies that did not change**.

## Where the time goes (mini-city recording, mean per frame 19.5 ms)

| phase | mean | p95 | max | share |
|---|---:|---:|---:|---:|
| `rapierStepMs` | 8.88 | 21.97 | 25.35 | **45.6 %** |
| `solverUpdateMs` | 5.50 | 11.62 | 13.27 | 28.2 % |
| &nbsp;&nbsp;└ CGNR solve (WASM, SIMD) | 3.66 | 6.99 | 7.64 | 66.6 % *of solver* |
| &nbsp;&nbsp;└ contact inject (JS) | 1.45 | 4.26 | 5.07 | 26.4 % *of solver* |
| `contactDrainMs` | 0.96 | 2.55 | 5.70 | 4.9 % |
| `snapshotCaptureMs` (resim rollback) | 0.77 | 1.84 | 3.01 | 3.9 % |
| `colliderRebuildMs` | 0.48 | 1.05 | **149.5** | one-time hitch (frame 90) |
| `resimMs` (wrapper — 2nd pass) | 5.43 | 18.06 | 24.61 | 27.9 % |

51 % of frames miss 60 FPS, 24 % miss 30 FPS. The 183 ms max is the **one-time**
first-fracture cascade (`colliderRebuild` building all the new colliders at once);
steady-state peaks are 42–54 ms on the 1200 + body frames, `rapierStep`-dominated.

The `contactInjectGrid` per-frame splash-grid rebuild is **~0 ms** — a
previously-landed optimisation. The regression test guards it from coming back.

## The resim mechanism (why a fracture frame is ~2×)

`step()` (in `rapier/destructible-core.ts`) does:

```
captureWorldSnapshot()                 // save all dynamic bodies
world.step()  →  fracture pass         // initial pass
if (fractured) {
  restoreWorldSnapshot()               // roll the WHOLE world back
  captureWorldSnapshot()               // (re-capture)
  world.step()  →  fracture pass       // ← SECOND full world.step over ALL bodies
}
```

The second `world.step()` integrates **every** rigid body again — but a fracture
only changes topology **locally**. Bodies in physically independent islands (not
bonded to, and not in contact with, the fractured region) have identical inputs in
both passes, so they integrate to the **identical** state. Re-stepping them is
pure waste that produces bit-identical output.

The core already ships **read-only instrumentation** for exactly this: islands of
the active-bond graph with static nodes as cut points, and a "settled island"
predicate (`getIslandSettledStats`, the `islandsSkipped` / `islandCount`
recording columns). That measures the skippable fraction with zero behaviour
change — the groundwork for acting on it.

## Output-safe levers, ranked by (impact × safety)

### 1. Island-exact scoped resim — *implemented as opt-in `scopedResim`; output-faithful (sub-mm)*
The second `world.step()` is the most expensive thing on a fracture frame, but only
the fractured region — and whatever is in contact with it — moves differently the
second time. Freeze whole **contact components that are disjoint from the fracture's
contact closure AND settled**: pin them at their initial-step result and sleep them
for the resim step so Rapier's island solver skips them. Rapier auto-wakes a sleeper
the instant an active body *contacts* it, so contact-*gain* coupling is preserved.

The critical detail (and the fix for the first attempt) is **where the closure comes
from**: the **initial pass's** contact graph, read *before* `flushPendingBodies`
migrates colliders. That graph still holds the resting contacts of bodies sitting
*on* the fractured structure, so they land in the closure and are re-stepped —
covering contact-*loss* (support removal), which Rapier never auto-wakes on.

A first attempt (PR #41) classified per-body by velocity and read the contact graph
*after* migration (a clobbered narrow phase that found nothing) — so debris resting
on a collapsing structure was wrongly frozen and the error compounded to **~12.8 m**
on cascades. Island-exact scoping from the pre-migration graph fixes exactly that.

Measured (`rapier.resim-perf.test.ts` §D):

- **Isolated fracture → byte-identical** (`maxPosDelta === 0`).
- **Cascading fracture → ~0.24 mm** (vs ~12.8 m for the old heuristic — a ~50,000×
  reduction). The residual no longer flips discrete fracture decisions, so it does
  not amplify.

Still **opt-in / default off**: near-identical, not provably bit-exact (a re-stepped
body crossing into a frozen island, or a sleep-timer shift, can perturb low bits),
and the **speedup is scene-dependent** — it skips only quiescent, decoupled regions,
so a single collapsing pile (~3 % freezable) gains little, while a scene with many
settled, spatially-separated structures (the mini-city) gains more. The live toggle
+ the "Scoped resim N% frozen" readout in the mini-city sidebar (`getScopedResimStats`)
make the real ceiling visible per scene.

### 2. Drop the dead snapshot re-capture on the final resim pass — *free, easy, safe*
The `captureWorldSnapshot()` after `restoreWorldSnapshot()` exists only to support
*another* resim pass (to roll back again). With `maxResimulationPasses: 1` (the
default, and mini-city's setting) there is never a second pass, so that capture is
**never read** — dead work on every resim frame (~0.8 ms mean, up to 3 ms;
~260–340 ms across the recording's 284 resim frames). Skip it when
`resimCount + 1 >= maxResimulationPasses`. Guarded by the equivalence harness.

### 3. Make the rollback snapshot incremental / scoped — *safe*
`captureDynamicBodySnapshots` saves **all** dynamic bodies every fracture frame.
Only bodies that will be re-stepped (lever 1) need saving; only bodies that moved
need restoring. Scope the snapshot to the same affected set. Pure overhead
removal — output unchanged.

### 4. Sleep / island-skip settled debris — *near-safe (needs tolerance + visual check)*
mini-city ran with `sleepMode: 'off'`, so ~1300 settled rubble bodies are
re-solved every step (and every resim step). Rapier's island solver skips asleep
bodies, and the stress solver's `skipSettled` skips islands whose bodies are all
asleep. This is the lever for the **base** rapierStep cost (lever 1 only removes
the *duplicate* step). It is *near* output-safe: a truly-settled body sleeping is
visually identical, but sleeping is a behaviour change (a sleeping body reacts only
once a contact wakes it). Validate with the equivalence harness at a physical
tolerance (sub-mm) plus a visual A/B, not the exact `=== 0` guard.

> Not pursued (they change output, per the brief): reducing world size, solver
> iterations, resim frequency/passes, debris collision pairs, or debris eviction.

## Tooling delivered (the tests that discover & guard these)

| file | kind | what it does |
|---|---|---|
| `src/rapier/recordingAnalysis.ts` | library | Pure-TS bottleneck analyzer over `decodeSimRecording`: phase breakdown, **body-count scaling regression** (ms/100 bodies), **de-confounded resim multiplier**, budget %, one-time-hitch detection. No WASM. |
| `src/tests/recordingAnalysis.test.ts` | unit | Validates the analyzer math on synthetic recordings (stats, regression slope recovery, 2× resim recovery under a body-count confound, budget %, hitch detection). Runs anywhere. |
| `src/tests/recording.minicity.regression.test.ts` | regression | Runs the analyzer on a committed real session (`fixtures/mini-city.profile.sim.json.gz`, 56 KB) and **locks the bottleneck shape**: rapierStep #1, CGNR #1 in-solver, resim ~40 %/~2×, scaling positive, `contactInjectGrid` ≈ 0. |
| `src/tests/rapier.resim-perf.test.ts` | bench (WASM) | The **output-equivalence harness** (`captureTrajectory` / `compareTrajectories`) with positive (determinism → 0) and negative (different impact → large) controls + profiler-non-interference; resim per-event cost; rapierStep body-count scaling. |
| `scripts/strip-recording.mjs` | tool | Make a committable, profile-only recording (drops the ~95 % body-trajectory blob; keeps timing/columns/resim). |

How a future optimisation uses this:

1. Implement the change behind a flag.
2. `compareTrajectories(before, after)` on a fracturing scenario → assert
   `maxPosDelta === 0` (levers 1–3) or `< tolerance` (lever 4). **Proves faithful.**
3. Compare profiler phase stats before/after → **proves faster.**
4. Re-record the demo and run the mini-city regression to confirm the real-scene
   profile shifted as intended.

## Reproduce

```bash
cd blast/blast-stress-solver
npm run build                                   # WASM + TS
npx vitest run src/tests/recordingAnalysis.test.ts \
              src/tests/recording.minicity.regression.test.ts   # no WASM needed
npx vitest run src/tests/rapier.resim-perf.test.ts              # WASM benches
node scripts/inspect-recording.mjs <file>.sim.json.gz --perf    # ad-hoc breakdown
```
