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
  | frame ↔ frame | 6.0 | roll cage / chassis welds — strongest, holds the shell |
  | frame ↔ wheel | 5.0 | hub / axle to the chassis |
  | frame ↔ panel | 3.0 | panels bolted to the frame |
  | panel ↔ panel / wheel | 2.0 | |
  | frame/panel/wheel ↔ **cargo** | 0.2–0.3 | lashed-on payload — weak, sheds first |
  | anything ↔ **accessory** | 0.1–0.18 | loose bits (chain, bucket) — first to go |

  Chunks of the *same* fractured part get strong **internal** bonds so the shell
  holds together until hit hard. The global **`materialScale`** then sets overall
  fragility on top of this hierarchy (lower = more destructible).

## Visuals: render mesh ≠ collision mesh

The car is decomposed (offline, via the GLB pipeline) into tight **CoACD convex
hull** pieces — great *collision* shapes (non-overlapping, so debris doesn't
explode) but ugly faceted blobs if you *render* them. So the demo separates the
two: each node keeps its CoACD hull for **collision**, but is **rendered** with a
slice of the original detailed `buggy.glb` model. At load, every original triangle
is assigned to the node whose collider owns it (`attachDetailedRenderGeometry` in
`glb-vehicle.ts`), so the union of the slices is the real car and each slice rides
with its piece when it detaches. The plumbing is a per-node
`scenario.parameters.colliderGeometries` channel the core prefers for the hull
collider (`destructible-core.ts`), leaving `fragmentGeometries` for the render
mesh. Toggle **Show Debug** to see the tight hull colliders behind the detailed
render.

- **Destroy.** Click to shoot, or **Drop from Height**. A light hit knocks the
  barrels/crates off while the cage holds; raise the projectile mass/speed (or
  lower the **Material Scale**) and the skeleton itself lets go.

Pieces are coloured by role so the hierarchy is visible (toggle **Color: Role /
State** to instead see intact-vs-detached state).

## Breaking model

Breaking is **fully stress-driven** — there is no scripted `onImpact` path. The
key realisation is that a free body is only stress-free in *free-fall*; once it
**rests on the ground**, the ground-contact reaction (which the core already
injects into the stress solver alongside gravity) plus any projectile contact
force create a real internal load path, and the solver's existing
overstressed-bond fracture path breaks bonds wherever the stress exceeds the
(area-encoded) per-role limit. So:

- **At rest:** gravity + the ground reaction hold the intact car solid (stress
  stays under every bond limit).
- **A hit / drop:** the contact force spikes local bond stress past the weak
  joints' limits, so cargo/accessories shed first and the cage holds until a hard
  enough hit overstresses the frame welds too.

`materialScale` sets the single (global) material limit — it's calibrated so the
intact car holds under its own weight yet a hit overstresses the weak joints.
Lower = more fragile. The mechanism is pinned by a headless test,
[`freeBodyGroundStress.test.ts`](../blast-stress-solver/src/tests/freeBodyGroundStress.test.ts)
(a free column breaks under self-weight when resting, stays intact in free-fall).

Two supporting details:
- **Resting cargo** is reduced to a single weak bond (`capRestingBonds` in
  `glb-vehicle.ts`) so it sheds like a resting contact, not a welded seam.
- Detached debris uses **`debrisCollisionMode: 'all'`** — full collision (debris ↔
  car, ↔ ground, and ↔ other debris). This is safe because the offline pipeline now
  emits genuinely **non-overlapping** collider hulls (a `drop-overlaps` pass removes
  the few residual sliver overlaps — harmless since render ≠ collision) **and** the
  core uses a thick ground slab so a deep pile of hundreds of pieces can't tunnel
  through and eject. Pinned by `scripts/shatter-test.mjs`: **cut every bond at once
  → the whole car collapses and settles, no explosion.** Press **Detonate all (F)**
  to see it.

The car is heavy on purpose (`totalMass` ~4000 kg) so a projectile transfers into
local bond stress (pieces break off) instead of shoving the whole car away; the
`materialScale` is set to hold that self-weight at rest yet break progressively
under fire.

## Controls

Tuning GUI in the sidebar:
- **Bond Strength by Role** — per-role attachment strength (frame / wheel / panel /
  cargo / accessory), the main hierarchy knobs (needs *Reset*). Cargo/accessories
  start weak so they shed readily.
- **Total Mass**, **Fracture Cell Size**, **Bond Reach** — vehicle build (*Reset*).
- **Projectile** radius / mass / speed — live.
- **Material Scale** — the global material limit; calibrated so the car is solid
  at rest yet breaks on a hit. Lower = more fragile (*Reset*). **Gravity** (*Reset*).
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
- **Breaking is stress-driven, with one global material.** The Blast stress
  solver has a single material (global stress limits, in Pa); per-bond strength is
  expressed only through bond **area** (`stress = force / area`, and area is also
  the bond's "health" / effective cross-section). That is a clean per-bond
  strength knob — area does not affect the rigid solve, only the break threshold —
  so the role hierarchy lives entirely in the area multipliers. Genuinely
  different *materials* per bond, or tension-free "resting" contacts, would need a
  fork of the vendored solver; the single-bond cargo approximation avoids that.
- **A heavy projectile launches debris fast.** Because breaking is physical, a
  heavy/fast projectile transfers real momentum to the struck part as it detaches,
  so it flies off faster than the old scripted model produced — momentum, not an
  explosion. An *absurd* mass (a 2.5-tonne ram) launches parts past the soak's
  60 m/s "explosion" check, so the soak uses a plausible 600 kg heavy instead.
- **Full debris collision (`'all'`) is on** — every piece collides with the car,
  the ground and other debris. This works because the colliders are genuinely
  non-overlapping (pipeline `drop-overlaps` pass) and the core's ground is a thick
  slab (a thin floor let a deep pile tunnel through and explode — the historical
  reason this was `'noDebrisPairs'`). Verified by `scripts/shatter-test.mjs`.

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
