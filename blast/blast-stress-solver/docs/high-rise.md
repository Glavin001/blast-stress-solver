# High-Rise Apartment — realistic, non-glass destruction

A mid-rise (≈9-storey) reinforced-concrete apartment building that, unlike a uniform-material
structure, does **not** shatter its whole face like glass when hit: walls are frangible infill
hung on a stiff, strongly-anchored skeleton, so the frame refuses to propagate the failure, and
the building sheds a section only when a support is knocked out (Red Faction: Guerrilla style).
The same building runs in the **web demo** and the **Bevy Rust demo** from one shared, generated
scene pack. (See "Destruction model & honest limitations" for what the stress solver alone does
and does not do.)

## Why uniform structures shatter like glass (the problem)

The Blast stress solver uses **global** stress limits — one compression/tension/shear
elastic+fatal set (Pa) for the whole solver. A structure built from one uniform material is
therefore uniformly strong, so an impact's stress propagates roughly evenly and a large
fraction of bonds cross the fatal limit in the same few frames → a glass-like ripple. That
cascade also spawns a flood of rigid bodies → a performance cliff.

Two specific failure modes this design fixes:

- **"The footing rips off and the building slides."** Material strength is expressed only
  through bond **area** (`stress = impulse / area`). If the base joints aren't special-cased
  they're the *weakest* bonds while carrying the entire building weight, so they fail first.
- **"Hitting a wall blows up the whole face."** A uniform infill sheet transmits a point load
  across the whole panel, so a hard hit fails all of it at once.

## The structural model (the fix)

Heterogeneity, not a global strength tweak:

| Layer | What | Fracture | Strength (bond-area multiplier) |
|---|---|---|---|
| **Skeleton** | RC columns + flat floor slabs (a "flat-slab" frame) | coarse (few large chunks) | `column↔slab 6×`, `slab↔slab 3×` |
| **Anchor** | static (mass 0) foundation tiles | box tiles | **`foundation↔column 12×`** — strongest joint |
| **Infill** | drywall wall panels filling perimeter bays | fine (many small chunks) | `infill↔infill 0.1×`, `frame↔infill 0.04×` |

The vertical load path is `foundation → column → slab → column → … → roof` (columns are
interrupted by slabs at each floor, so geometry never overlaps and is fully deterministic).
Infill is a *frangible curtain hung on a stiff frame*: a hit pops out local panels without the
stress wave reaching the skeleton. The ~100-300× skeleton-to-infill strength ratio is what
localizes damage.

### Realistic SI values

- **Masses** are `volume × density` per chunk: reinforced concrete **2400 kg/m³**, drywall
  **800 kg/m³** (heterogeneous — the default building masses ≈ **1.57 M kg**).
- **Global limits are concrete-grade and *decoupled*** (shipped in the scene pack's
  `solver.limits`): compression fatal **30 MPa** (holds gravity), tension **3 MPa**, shear
  **4 MPa** (≈10× weaker — concrete cracks locally under impact, the key non-glass knob), with
  a wide elastic→fatal band for ductility.
- **Wrecking ball**: radius 0.6 m, **2500 kg**, 18 m/s (a believable demolition ball).
### Destruction model & honest limitations (verified in the full Rapier sim)

Destruction relies on the **heterogeneous structure** + the solver's **built-in bond health
(elastic→fatal band)** + **splash-localized** contact forces. Both runtimes share this exact
mechanism (the JS core and the Bevy demo apply contact force the same way: full force at the
hit node + quadratic-falloff splash within ~2 m, `contactForceScale = 30`).

What this **fixes** (the original problem): a uniform building shatters its whole face from one
hit. Here, walls are *infill* hung weakly on a stiff frame, so hitting a wall does **not** blow
out the face — the strong skeleton refuses to propagate the failure. A single heavy ball into a
wall barely dents it; gravity-only is stable; the base never rips off (anchored). Verified.

What it **does not** do, and why: the Blast stress solver is a *global* quasi-static solve, so a
contact force is spatially **all-or-nothing** — below a threshold nothing breaks; once a load
path is severed (e.g. a low column hit hard) the section above progressively collapses. There
is no graceful *per-hit local chipping* from the stress solver alone. That graceful, graded
local destruction is what an **optional contact-damage layer** (per-chunk health + splash,
`DestructibleDamageSystem`) provides — it is wired through the scene pack's `defaults.damage`
block and the web loader but ships **disabled** (out of current scope). A tuned starting point
(`strengthPerVolume 500, kImpact 0.2, contactDamageScale 10, splashRadius 3 m`) is kept in the
block; flipping `enabled: true` (and porting an equivalent to the Rust crate) is the documented
next step when fine-grained local destruction is in scope.

## Composable building kit

The building is authored in TypeScript and is the single source of truth:

- `buildHighRiseScenario(options)` — `src/scenarios/highRiseScenario.ts`. Composes foundation +
  columns + slabs + infill from `subdivideBoxFragments` (deterministic grid "fracture"; no
  Voronoi dependency), proximity-bonds them, and applies `HIGH_RISE_BOND_MULTIPLIERS`.
- Per-fragment `density` (`src/three/fracture.ts` / `scenarioFromFragments.ts`) gives mixed
  materials realistic masses.
- `subdivideBoxFragments` + an optional custom multiplier table in `applyBondStrengthMultipliers`
  (`src/three/fractureBuilders.ts`) are reusable for other composite structures.

## Generate, run, test, sweep

The scene pack is **generated** from the composer and **git-ignored** (regenerated from the
TS build, not committed):

```bash
cd blast/blast-stress-solver
npm install --ignore-scripts
npm run build:ts          # tsup -> dist/, then postbuild:ts generates high-rise.json
# (writes blast-stress-demo-rs/assets/scenes/high-rise.json AND dist/high-rise.json)
```

- **Web demo:** `npm run build && (cd ../js_stress_example && npm run build:web && npm run serve)`,
  then open the **High-Rise Demolition** card. The demo loads the same JSON via
  `loadScenePackFromUrl` and shows bonds-broken-over-time in the HUD.
- **Rust/Bevy demo:** `cd ../blast-stress-demo-rs && BLAST_STRESS_DEMO_SCENARIO=high-rise cargo run`
  (loads the generated JSON at runtime). Headless with scripted shots:
  `BLAST_STRESS_DEMO_SCENARIO=high-rise BLAST_STRESS_DEMO_HEADLESS=1 BLAST_STRESS_DEMO_HEADLESS_FRAMES=300 BLAST_STRESS_DEMO_HEADLESS_SHOT_SCRIPT=high_rise_wrecking_ball cargo run`.
- **Tests:**
  - Pure (no WASM, in `test:invariants`): `npx vitest run src/tests/highRise.scenario.test.ts` —
    structure, anchoring, masses, skeleton≫infill, loader round-trip.
  - Rust headless bands (CI-gating): `cd ../blast-stress-solver-rs && cargo test --features scenarios --test high_rise_scenarios_test` —
    gravity-stable, light-hit→local-infill-only, heavy-hit→bounded-hole, no-glass, deterministic.
- **Sweeps (the "civil-engineering simulator"):**
  - Rust (fast, solver-only): `cargo run --example high_rise_sweep --features scenarios`
  - JS (full Rapier pipeline): `npm run sweep:high-rise` (after `npm run build`).

## Observability

`createBondBreakRecorder(core)` (`src/rapier/metricsRecorder.ts`) samples per frame: active
bonds, bonds broken this frame + cumulative, rigid-body count, and dynamic-chunk COM height —
used by the demo HUD, the JS sweep, and tests to quantify "how many bonds are breaking over
time" (aesthetic *and* performance signal).
