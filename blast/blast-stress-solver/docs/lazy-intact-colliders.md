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
  It cannot miss (every fragment lives inside the building AABB); a false positive is a physics
  no-op. No proxy collider, no probe step, no rollback.
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

## Next step: hierarchical `collisionTree` (opt-in)
Today's descent is binary: a building is fully dormant or fully exploded. For tall/complex
structures a single corner hit materialises *all* fragments. We have hierarchical authoring
knowledge (building → wall/floor/column → fragment) we currently discard in the merge.

**Opt-in input**, orthogonal to the (unchanged) flat `nodes[]`/`bonds[]`:
```ts
type CollisionGroup = { children?: CollisionGroup[]; fragments?: number[] };
scenario.collisionTree?: CollisionGroup[];   // one root per building; absent ⇒ union-find fallback
```
**Active set = a frontier (cut) through the tree.** Precompute each node's AABB (= union of
descendant fragment AABBs). The predictive test descends **only where movers are**:
```
mover swept-AABB vs building AABBs → for each overlap, vs its element AABBs → enable that
element's fragment colliders (descend to leaves); prune subtrees with no mover overlap.
```
- **Locality:** a corner hit on a 1000-fragment high-rise enables only the struck wall's ~20
  fragments; cost tracks *damage extent*, not structure size. The test is O(log) per mover.
- **Same equivalence profile** as today (it also enables before contact): correctness +
  determinism always; bit-identity for idle/approach/rigid; warm-start chaos only in soft
  mega-shatters. The hierarchy is a *performance-locality* win, not an equivalence change — and
  because it's orthogonal to the bond graph, it **cannot** alter fracture output by construction.
- **Migration:** `buildFracturedTowerScenario` already knows wall/floor/column membership; the
  merge currently throws it away. Emit a `collisionTree` and migrate mini-city, then others.

### Implementation sketch (mostly generalising existing machinery)
- Replace `IntactBuilding` with a tree node carrying `{ children?, fragments?, aabbMin/Max, active }`.
- `predictiveExplodePass` recurses the tree per mover instead of scanning a flat building list.
- `explodeBuilding(node)` enables the node's leaf colliders (or descends one level), reusing
  `setBuildingCollidersEnabled`.
- Stats report frontier depth / active-leaf count per building.

## Not pursued (and why)
- **Merged/trimesh/convex-hull intact colliders:** change the contact set → fail "real-geometry
  contacts". Rejected.
- **Per-frame explode budget:** deferring fractures changes output. Rejected — all approached
  buildings explode in-frame.
- **Engine upgrade as the fix:** we already run the post-rewrite BVH broadphase (rapier3d-compat
  0.19.1 ≥ 0.18); the SIMD package (`@dimforge/rapier3d-simd-compat`) measured only ~1.1× on this
  static-collider workload (it delivers 2–5× on *solver-heavy* scenes). Worth adopting separately
  for the destruction phase, but it does not solve idle scaling.
