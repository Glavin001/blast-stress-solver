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

## Controls

- **Total Mass**, **Fracture Cell Size** (~chunk size for large structural parts;
  0 = keep whole), **Bond Reach** (max surface gap auto-bonding treats as contact)
  — vehicle build (needs *Reset*).
- **Projectile** radius / mass / speed — live.
- **Material Scale** — global toughness; **Gravity** (needs *Reset*).
- **Contact Force Scale** — how hard impacts hit the stress graph (needs *Reset*).
- Shared **Physics / Optimization** controls (friction, restitution, debris
  cleanup, …) plus the live frame profiler and session recorder.

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
- **Impact breaking is contact-driven.** A free-floating car barely stresses its
  bonds on impact, so breaking is driven by the core's `onImpact` callback: a hit
  past a role-based force threshold cuts the struck part's bonds (it detaches and
  falls), with a small bounded splash. `materialScale` is kept high so the stress
  solver holds the car rock-solid under gravity; all breaking goes through the
  impact path.
- **Debris collides with the ground only** (`debrisCollisionMode:
  'debrisGroundOnly'`). Fractured/split chunks have overlapping convex hulls;
  fine while welded into one body, but when many detach at once, debris-vs-body
  penetration would otherwise resolve as a violent explosion. Ground-only keeps it
  stable (the trade-off is that shed parts fall through each other rather than
  piling).

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
