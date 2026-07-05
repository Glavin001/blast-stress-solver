# Lazy intact colliders — equivalence findings & hierarchical roadmap

This documents the **collision-dormant intact buildings** optimization (the `lazyIntactColliders`
core option), the precise output-equivalence guarantee it makes (with the experiments that
establish it), and the **hierarchical `collisionTree`** design that is the natural next step.

## The problem
A big mini-city builds ~1 Rapier collider per fragment (≈14,400 for 10×10), all on one fixed
root body. Rapier's broadphase maintains every collider's AABB **every frame, even at idle**.
This is inherent to "N colliders" — even the new (2025) BVH broadphase is O(N) to maintain N
proxies, and Rapier has no "this collider never moves, skip it" flag (see dimforge/rapier#558,
and #730 "separate colliders have worse perf than compound", labelled `C-Performance`/`D-Difficult`).
The engine deliberately pushes this to userland; we have the domain knowledge it lacks
(*these fragments are one intact building that needs no per-fragment colliders until hit*).

## The mechanism (current PR)
- Every real per-fragment collider is created at init (identical handles/order to the eager
  path) but **disabled** — excluded from the broadphase (≈5× cheaper than enabled; measured).
- A building = a connected component of the bond graph (union-find, supports included).
- Before each `world.step`, a **conservative predictive AABB test** enables (`explodeBuilding`)
  any building whose AABB is overlapped by a mover's swept box (`pos → pos+v·dt` + radius + skin).
  Movers are every awake non-fixed body in the world: projectiles, debris (chunk bodies, with a
  fragment-derived radius), **and external bodies the host app created** — vehicles, character
  controllers, thrown props — whose radius is bounded conservatively from their colliders
  (`externalMoverRadius`). It cannot miss (every fragment lives inside the building AABB); a
  false positive is a physics no-op. No proxy collider, no probe step, no rollback.
- Dormant colliders are invisible to **manual world queries** (`world.castRay`, shape casts)
  until something materializes them — hitscan against intact structures needs eager mode or an
  explicit materialize-first step.
- The bond/stress graph is untouched and **collider-independent**, so the hierarchy/LOD can never
  change *which* fractures happen — only the physical collision representation.

Idle headless bench (`scripts/idle-city-bench.mjs`, 10×10 = 14,400 fragments):
`world.step()` **4.3 ms → 0.16 ms**.

## Output-equivalence — what's guaranteed, and the experiments that prove it
Three distinct properties (don't conflate them):

| property | guaranteed? | evidence |
|---|---|---|
| **Physical correctness** (real-geometry contacts, correct fractures, converged solver) | ✅ | descend-before-contact; stress is collider-independent |
| **Determinism** (same input → same output) | ✅ | deterministic predictive test |
| **Bit-identity to the all-colliders reference** | ✅ idle / approach / rigid impacts; ⚠️ *not* through a soft-material mega-shatter | experiments below |

Experiments (single tower, projectile; `maxPosDelta` = max chunk position delta vs eager):

1. **Idle & approach:** `maxPosDelta === 0` for every pre-contact frame.
2. **Mechanism is order-preserving:** *disable→enable every collider at init* ("toggle-init") is
   `0.000e+0` vs eager **through a full 131-body shatter**. So toggling does **not** perturb
   Rapier's contact-solve order — the earlier "handle perturbation" hypothesis was wrong.
3. **The residual is contact warm-start history, not settle-fracture.** Body count after the
   settle phase is 2 in all cases (nothing fractures while settling). Enabling colliders at init
   = `0`; enabling *after* settle = diverges by the same amount → the difference is the persistent
   ground-contact warm-start the eager building accumulates while its colliders sit enabled.
4. **It scales with shatter violence:** rigid `1e10` (mini-city's material) → few bodies →
   `maxPosDelta < 1e-6`; soft `2e6` → 131 bodies → metres.

**Interpretation:** warm-starting only *accelerates solver convergence*; both paths reach the
same physical solution, differing by floating-point rounding that chaos amplifies in large piles.
The all-colliders "reference" is itself non-canonical — its exact rubble also changes under
warm-start tuning, solver-iteration count, a Rapier/SIMD/thread change. So this is **not** a
quality regression; it's chaos sensitivity to a rounding detail we skip to make idle cheap. You
cannot keep that history without keeping the colliders enabled (the very cost we remove). The
live sidebar toggle is the escape hatch if a specific shot needs the all-colliders path.

These properties are pinned by `src/tests/rapier.lazy-colliders.test.ts`.

## Hierarchical collision LOD via `collisionTree` (implemented)
The original descent was binary: a building was fully dormant or fully enabled, so one corner hit
on a tall structure materialised *all* its fragments. The collider frontier now descends a
**hierarchical LOD tree**, so a localized hit only enables the struck region.

**Opt-in input**, orthogonal to the (unchanged) flat `nodes[]`/`bonds[]`:
```ts
type CollisionGroup = { children?: CollisionGroup[]; fragments?: number[] };
scenario.collisionTree?: CollisionGroup[];   // one root per building; absent ⇒ bond-component fallback
```
The runtime builds a tree of `LodNode`s (one root per building), each caching its AABB (= union of
descendant fragment AABBs). `enabled` means "every leaf collider under this node is active". The
**active set is the frontier** (the enabled cut). `predictiveExplodePass` descends per mover:
```
mover swept-AABB vs node AABB → no overlap: prune subtree
                              → overlap, internal node: recurse into children
                              → overlap, leaf: enable that leaf's fragment colliders
```
Non-overlapped subtrees stay dormant; fully-enabled subtrees are pruned. Conservative by
construction (a node AABB encloses all its real colliders) so it never misses.

- **Measured locality:** a 4 t projectile into the **high-rise (930 fragments, 30 m tall)** with a
  spatial tree (`leafMaxFragments: 32`) activates only **58/930 fragments (6 %)** — the struck
  region — versus 100 % for the binary whole-building enable. The descent is O(log) per mover.
- **Same equivalence profile:** correctness + determinism always; bit-identity for idle / approach /
  rigid impacts; warm-start chaos only in soft mega-shatters. The hierarchy is a *performance-
  locality* win, not an equivalence change — orthogonal to the bond graph, it **cannot** alter
  fracture output by construction. (Pinned by the `Hierarchical collision LOD` tests.)
- **Building a tree (two ways):**
  - **Authored / semantic (preferred when the structure is known).** `buildFracturedTowerScenario`
    now emits `scenario.collisionTree` straight from authoring: **building → floor → element (each
    wall / column / slab) → fractured fragments**. A localized hit descends to the struck floor →
    struck wall and wakes just that element (~one wall ≈ **5 %** of the building, measured).
    mini-city's `mergeScenarios` offsets + concatenates each tower's tree into the merged scenario.
  - **Spatial fallback (no metadata).** `buildSpatialCollisionTree(scenario, { leafMaxFragments })`
    — a shape-agnostic balanced median-split per bond-connected component; works for any structure.
    mini-city uses it only if no authored tree is present.
  - Absent any tree, the core uses one flat leaf per building (the original binary behavior).

### Implementation (in `destructible-core.ts` + `collisionTree.ts`)
- `LodNode { children, fragments, aabbMin/Max, enabled, buildingId }`; `lodRoots[]` + `buildingOfNode`.
- `ensureLodRoots` builds from `scenario.collisionTree` (recursively) or the bond-component fallback.
- `predictiveExplodePass` → `descendForMover` recurses per mover; `enableSubtree`/`disableSubtree`
  toggle leaf colliders; `getLazyColliderStats` reports `activeLeafFragments` (the locality metric).
- A fracturing building no longer force-enables its whole self: a splitting fragment gets its
  collider on its new dynamic body via `flushColliderMigrations`, and the still-bonded remainder
  stays dormant until a mover (incl. falling debris) actually approaches it — preserving locality.

## Not pursued (and why)
- **Merged/trimesh/convex-hull intact colliders:** change the contact set → fail "real-geometry
  contacts". Rejected.
- **Per-frame explode budget:** deferring fractures changes output. Rejected — all approached
  buildings explode in-frame.
- **Engine upgrade as the fix:** we already run the post-rewrite BVH broadphase (rapier3d-compat
  0.19.1 ≥ 0.18); the SIMD package (`@dimforge/rapier3d-simd-compat`) measured only ~1.1× on this
  static-collider workload (it delivers 2–5× on *solver-heavy* scenes). Worth adopting separately
  for the destruction phase, but it does not solve idle scaling.
