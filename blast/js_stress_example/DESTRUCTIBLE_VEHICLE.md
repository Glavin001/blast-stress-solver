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

- **Decompose.** The GLB's meshes become physics nodes. The source ground plane
  is dropped automatically.
- **Classify.** Every part is tagged with a structural **role** from its name,
  size and position (see `classifyVehiclePart`). For the bundled buggy that
  yields: 7 *frame*, 4 *wheel*, 2 *panel*, 19 *cargo*, 3 *accessory*.
- **Bond hierarchy.** Parts are bonded by proximity, then every bond's strength
  (its area — `stress = force / area`) is scaled by the **role pair**:

  | joint | ×area | meaning |
  |---|---:|---|
  | frame ↔ frame | 8.0 | roll cage / chassis welds — strongest, holds the shell |
  | frame ↔ wheel | 5.0 | hub / axle to the chassis |
  | frame ↔ panel | 3.0 | panels bolted to the frame |
  | panel ↔ panel / wheel | 2.0 | |
  | frame/panel/wheel ↔ **cargo** | 0.2–0.3 | lashed-on payload — weak, sheds first |
  | anything ↔ **accessory** | 0.1–0.18 | loose bits (chain, bucket) — first to go |

  Chunks of the *same* part (when you turn fracturing on) get strong **internal**
  bonds so the shell stays together under light hits.

- **Destroy.** Click to shoot, or **Drop from Height**. A light hit knocks the
  barrels/crates off while the cage holds; raise the projectile mass/speed (or
  lower the **Material Scale**) and the skeleton itself lets go.

Parts are coloured by role so the hierarchy is visible (toggle **Color: Role /
State** to instead see intact-vs-detached state).

## Controls

- **Total Mass** / **Shatter Structural Parts** — vehicle build (needs *Reset*).
  "Shatter" Voronoi-fractures the frame/panel/wheel parts into N chunks each so a
  hard hit shatters them (0 = parts stay intact, the robust default).
- **Projectile** radius / mass / speed — live.
- **Material Scale** — global toughness (live); **Gravity** (needs *Reset*).
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
exports (`buildScenarioFromFragments`, `fractureGeometryAsync`,
`recenterGeometry`), so it builds with esbuild/tsc without rebuilding the WASM.

## Notes & limitations

- The bundled GLB has **no textures** (the source export is geometry-only), so
  flat role-colouring is also the most informative view.
- Colliders are per-part **boxes** sized to each part's bounds (matching the
  other fractured demos). Passing a Rapier module to `buildScenarioFromFragments`
  would upgrade these to convex hulls — a natural follow-up for rounder wheels.
- Bond detection scans every part's vertices, so the first load / Reset of the
  high-poly model takes ~1–3 s. Decimating the collider/bond geometry (while
  keeping full-res for rendering) is the obvious optimization.
- Per-part Voronoi fracture (the "Shatter" slider) runs three-pinata on meshes
  that may be non-manifold; it's guarded with a per-part fallback to "keep
  intact", so a model that doesn't fracture cleanly still works.
