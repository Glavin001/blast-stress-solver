# Destructible Vehicle Demo

Load an arbitrary **GLB model**, auto-decompose it into its parts, and make it
destructible with a **hierarchy of bond strengths** so it comes apart the way a
real built-up vehicle would: the strapped-on payload sheds first, the roll
cage / chassis holds, and a hard enough hit (or a big drop) shatters the whole
thing.

Open `destructible-vehicle.html` (it's the first card on the demo index).

| | |
|---|---|
| Demo page | [`destructible-vehicle.html`](./destructible-vehicle.html) |
| Scene / render / UI | [`destructible-vehicle.ts`](./destructible-vehicle.ts) |
| Reusable decomposition | [`glb-vehicle.ts`](./glb-vehicle.ts) |
| Model | [`assets/buggy.glb`](./assets/buggy.glb) (a junkyard buggy) |
| CLI analyzer / manifest | [`../../scripts/analyze-glb.mjs`](../../scripts/analyze-glb.mjs) |

## What it shows

The brief: *take a GLB, break it into pieces, fracture those pieces, and add
stress solving with different bond connections — wheels strong, axle→body
stronger, chassis strongest — so exterior pieces fall off but the shell holds,
yet a hard enough hit destroys everything.*

This demo does exactly that:

The decomposition pipeline (in `glb-vehicle.ts`) is:

1. **Split into physical pieces.** Each GLB mesh is split into its connected
   components (topological islands), so several physically-separate shapes that an
   artist modelled as one object (e.g. the welded-but-separate tubes of a roll
   cage) become independent pieces — each with its own *tight* collider instead of
   one convex hull spanning the gaps. Wheels are kept whole (a rubbery wheel should
   fall off as a unit, not separate into tire/rim/nuts). The source ground plane is
   dropped automatically.
2. **Classify.** Every piece is tagged with a structural **role** from its parent
   mesh's name, size and position (see `classifyVehiclePart`): *frame*, *wheel*,
   *panel*, *cargo*, *accessory*.
3. **Fracture (volume-scaled).** Large, concave structural pieces (frame / panel)
   are Voronoi-fractured into chunks roughly `fractureCellSize` metres across — the
   chunk count scales with the piece's **volume** (big parts → many chunks, small
   parts → few), with a floor by longest dimension so long thin cage tubes still
   get cut into segments. This is essential: a whole concave roll cage kept intact
   has ONE convex-hull collider that's a nonsensical car-sized blob. Wheels, cargo
   and accessories are *not* fractured (they're cohesive props; a wheel shouldn't
   shatter like glass).
4. **Bond by real contact.** The pieces are bonded with the WASM auto-bonder
   (`createBondsFromTriangles`), which finds where their meshes actually touch and
   gives each bond a real contact area, normal and location — so the assembly comes
   apart along real seams (wheel↔frame at the hub, cargo↔the surface it rests on)
   instead of a centroid star.
5. **Apply the strength hierarchy.** Every bond's strength (its area —
   `stress = force / area`) is scaled by the **role pair**:

  | joint | ×area | meaning |
  |---|---:|---|
  | frame ↔ frame | 8.0 | roll cage / chassis welds — strongest, holds the shell |
  | frame ↔ wheel | 5.0 | hub / axle to the chassis |
  | frame ↔ panel | 3.0 | panels bolted to the frame |
  | panel ↔ panel / wheel | 2.0 | |
  | frame/panel/wheel ↔ **cargo** | 0.2–0.3 | lashed-on payload — weak, sheds first |
  | anything ↔ **accessory** | 0.1–0.18 | loose bits (chain, bucket) — first to go |

  Chunks of the *same* fractured part get strong **internal** bonds so the shell
  holds together until hit hard.

- **Destroy.** Click to shoot, or **Drop from Height**. A light hit knocks the
  barrels/crates off while the cage holds; raise the projectile mass/speed (or
  lower the **Material Scale**) and the skeleton itself lets go.

Pieces are coloured by role so the hierarchy is visible (toggle **Color: Role /
State** to instead see intact-vs-detached state).

## Breaking model

Breaking is **entirely stress-driven** — the NVIDIA Blast stress solver computes
each bond's stress every frame and fractures the ones past their fatal limit.
There are no scripted force/velocity thresholds; destruction is whatever the
physics produces. The three stress sources:

- **Localized contact (impacts).** A projectile or ground hit is a force on one
  region of the body. Even though the car is a *free* (unanchored) body, that hit
  develops real internal bond stress: the struck region has to drag the rest of
  the body along, and the inertial reaction of "the rest of the body" (d'Alembert)
  loads the bonds in between. Hit harder → more stress → more breaks. Contact
  forces are injected into the solver scaled by **Contact Force Scale**.
- **Gravity.** A uniform field on a free body produces *zero* internal stress
  (rigid free-fall), so a parked or airborne car never self-destructs from gravity
  alone — its weight is carried by the localized ground reaction at the wheels,
  which stresses the strong wheel/frame load path (kept under its limit).
- **Centrifugal.** A spinning/tumbling body feeds each node `ω×(ω×r)`, so a car
  that's flung and goes tumbling self-stresses and sheds its loosely-bonded cargo —
  real physics, not a motion heuristic. (On by default here.)

The **hierarchy is the bond geometry**: stress = impulse / area, so each role pair
is given a bond-area multiplier (`frame|frame ×8` … `frame|accessory ×0.12`). The
weak (small-area) cargo/accessory joints reach their limit first and shed; the
strong (large-area) frame welds hold until a catastrophic hit. **Material Scale**
(live) rescales every limit at once — the master toughness knob.

**Warm-up:** fracture is suppressed for the first ~0.5 s after a (re)build
(`fractureWarmupFrames`). A freshly built body starts the iterative solve from a
zero guess, so the first few frames can transiently overshoot the true static
stress; warming up lets the solver converge to equilibrium before any bond is
allowed to break, so a car settling onto the ground doesn't shed parts from the
spawn transient.

Detached debris uses `debrisCollisionMode: 'noDebrisPairs'` — it bounces off the
car and ground but not off other debris (overlapping just-detached chunks colliding
with each other is what caused the early "explosion").

## Controls

Tuning GUI in the sidebar. All breaking is stress-solver-computed, so the knobs
are the real stress parameters:
- **Attachment Strength by Role** — per-role bond-**area** multiplier (frame /
  wheel / panel / cargo / accessory), the hierarchy (needs *Reset* — area is baked
  into the bonds at build). Cargo/accessories start weak so they reach their stress
  limit first and shed.
- **Material Scale (live)** — master toughness; live-rescales every stress limit
  (`core.setMaterialScale`). Lower = the whole car breaks more easily.
- **Contact Force Scale (live)** — how much bond stress an impact injects
  (`core.setContactForceScale`). Higher = a hit breaks parts more easily.
- **Centrifugal (live)** — spin self-fracture; on by default so a flung car sheds
  cargo through computed centrifugal stress.
- **Total Mass**, **Fracture Cell Size**, **Bond Reach** — vehicle build (*Reset*).
- **Projectile** radius / mass / speed — live.
- **Gravity** (*Reset*).
- Shared **Physics / Optimization** controls plus the live frame profiler and
  session recorder.

## The decomposition tool

`scripts/analyze-glb.mjs` is a dependency-free CLI that inspects any `.glb`,
reports each part (size / position / triangles / material), auto-classifies it
into the same roles the demo uses, and writes a hand-editable JSON manifest:

```bash
node scripts/analyze-glb.mjs blast/js_stress_example/assets/buggy.glb --json manifest.json
```

Use it to sanity-check (or correct) the decomposition before/while wiring up a
new model. The heuristics in the CLI are intentionally mirrored by
`classifyVehiclePart()` in `glb-vehicle.ts`, so they agree.

## Swapping in a different model

1. Drop your `.glb` in `assets/` and point `MODEL_URL` in
   `destructible-vehicle.ts` at it.
2. Run the analyzer to see how the parts classify; if the heuristics mislabel a
   part, extend the keyword/geometry rules in `classifyVehiclePart` (and the CLI)
   or rename the mesh in the source model.
3. Tune `ROLE_COLORS`, the `INTER_ROLE_MULTIPLIER` / `INTERNAL_ROLE_MULTIPLIER`
   tables, and `totalMass` to taste.

`glb-vehicle.ts` only uses already-published `blast-stress-solver/three`
exports (`buildScenarioFromFragmentsAsync`, `fractureGeometryAsync`,
`recenterGeometry`), so it builds with esbuild/tsc without rebuilding the WASM.

## Notes & limitations

- The bundled GLB has **no textures** (the source export is geometry-only), so
  flat role-colouring is also the most informative view.
- Colliders are one **convex hull per piece** (built by the runtime from each
  piece's geometry). Splitting + fracturing first is what makes those hulls tight;
  a concave piece kept whole would get one blob hull. Genuinely concave pieces that
  we *don't* fracture (some props) still get a loose hull — a convex-decomposition
  pass (e.g. CoACD/VHACD) would tighten those, at the cost of a heavyweight WASM
  dependency and multi-hull-per-node support in the core.
- Splitting + auto-bonding + fracturing scans/processes a lot of triangles, so the
  first load / Reset of the high-poly model takes a few seconds. Decimating the
  collider/bond geometry (while keeping full-res for rendering) is the obvious
  optimization.
- Voronoi fracture (three-pinata) runs on meshes that may be non-manifold; it's
  guarded with a per-part fallback to "keep whole", so a model that doesn't
  fracture cleanly still works.
- **Impact breaking is stress-driven** (no scripted cuts). A free-floating car
  *does* stress its bonds on a localized hit — the inertial reaction of the rest of
  the body loads the bonds at the impact (d'Alembert), which is what the solver
  computes. Tuning is three knobs: `materialScale` (master toughness, live),
  `contactForceScale` (how much stress a hit injects, live), and the per-role bond
  **area** multipliers (the hierarchy). The earlier build used an `onImpact`
  force-threshold callback to script detachment; that was a workaround for a
  mis-tuned material (limits ~1e9 that no stress could reach) and has been removed.
- **Debris doesn't collide with other debris** (`debrisCollisionMode:
  'noDebrisPairs'`). Fractured/split chunks have overlapping convex hulls; fine
  while welded into one body, but a pile of just-detached overlapping chunks
  resolving their mutual penetration is a violent explosion. `noDebrisPairs` keeps
  debris lively (cargo still bounces off the car and ground) while removing that
  failure mode.

## Headless QA

`scripts/soak-vehicle.mjs` drives the demo in headless Chromium through settle /
light shots / heavy shots / drop and asserts the sim never "explodes" (no body
exceeds ~120 m/s; `window.__vehicleDemo.metrics()` reports max speed, spread and a
count of bodies above 60 m/s). Run it before asking a human to QA in a browser:

```bash
# with the demo built and served (npm start at the repo root, or npm run serve:demos)
cd blast/js_stress_example
npx playwright install chromium   # once
node scripts/soak-vehicle.mjs     # exits non-zero if any scenario blows up
```
