# High-Rise Apartment — realistic, non-glass destruction

A mid-rise (≈9-storey) reinforced-concrete apartment building designed to be destroyed by
a wrecking ball *realistically* — punching local holes and shedding sections incrementally
(Red Faction: Guerrilla style) instead of shattering uniformly like glass. The same building
runs in the **web demo** and the **Bevy Rust demo** from one shared, generated scene pack.

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

These were tuned with the headless sweep (below). Measured response on an infill hit: nothing
below 1 MN, a **local hole** (12-26 bonds, zero skeleton) at 3-10 MN, and a true glass cascade
(>45% of bonds) only at ~300 MN — i.e. ~100× a realistic ball. Columns shrug off everything
below ~30 MN; even a 300 MN strike punches a **local** hole (6.6% of bonds, base stays anchored).

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
