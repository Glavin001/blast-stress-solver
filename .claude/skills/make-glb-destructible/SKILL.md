---
name: make-glb-destructible
description: Use when turning a GLB/glTF model (a car, building, prop, etc.) into a destructible Three.js + Rapier physics demo with the blast-stress-solver in this repo — decompose the model into physical pieces, bond them in a strength hierarchy, fracture the concave parts, and make them break/shed realistically. Covers the full pipeline, the threshold-based free-body breaking model, and the gotchas that waste hours (collider explosions, welded concave meshes, dead-feeling tuning sliders, verifying headless before asking a human to QA).
---

# Making a GLB model destructible

A field guide for converting an arbitrary GLB into a stable, tunable destructible
demo. The reference implementation lives in `blast/js_stress_example/`:
`destructible-vehicle.{ts,html}`, `glb-vehicle.ts`, `scripts/soak-vehicle.mjs`, and
the CLI `scripts/analyze-glb.mjs`. **Read those files first** — they're the proven,
working version. This skill is the *why* and the *gotchas*, so you don't re-learn
them the slow way.

## Read this first: three hard truths

These three cost the most time. Internalize them before writing code.

1. **A free-floating body does NOT break via the stress solver.** An impact on a
   free object becomes momentum, not internal stress (verified: 0 parts shed even at
   `materialScale` 1e5 with hard hits and 4 m drops). The stress solver only sheds
   parts on *anchored* structures (walls/towers, where supports create stress). So
   for a vehicle/free prop: keep `materialScale` **high** (solid at rest) and drive
   breaking yourself via **force/violence thresholds**, not bond stress. Corollary:
   **bond "strength"/area values barely affect destruction** — tune *thresholds*.

2. **Overlapping fractured/split colliders explode when many detach at once.** The
   runtime auto-builds one convex hull per node from its geometry; Voronoi chunks and
   split islands overlap each other. Welded into one body that's harmless (no
   self-collision), but the instant many become separate bodies, debris-vs-body
   penetration resolves as a violent blow-apart. Fix with `debrisCollisionMode`
   (see step 9). This is the "front immediately explodes" failure.

3. **One mesh ≠ one physical part.** Artists model several physically-separate shapes
   (e.g. the tubes of a roll cage, or all the wheel-nuts) as a single mesh. One convex
   hull over that mesh spans the gaps — a nonsensical car-sized blob. You must split
   each mesh into its connected components, *and* fracture the welded-but-concave ones.

## The pipeline (in order)

1. **Inspect.** `node scripts/analyze-glb.mjs model.glb --json manifest.json`. Prints
   every part's size/position/triangles/material and an auto-classified role, and
   writes a hand-editable manifest. Zero deps (parses glTF JSON directly). Use it to
   sanity-check the decomposition and spot the ground plane / mislabeled parts before
   touching the demo.

2. **Load + extract parts** (`extractVehicleParts`). GLTFLoader → traverse meshes →
   for each, **clone geometry and bake `mesh.matrixWorld` into it** (world space), then
   record bbox center/size/material color. Call `root.updateMatrixWorld(true)` first.
   **Drop the giant ground plane** the GLB usually contains (exclude any part whose max
   dimension ≫ the vehicle, e.g. > 3× the next-largest or > 20 m).

3. **Split each mesh into connected components** (`splitConnectedComponents`), so
   physically-separate shapes become independent pieces with tight hulls. Weld vertices
   by *quantized position* (handles duplicated verts in exports), union-find the
   triangles, group, merge tiny stray islands into the nearest kept one (cap the count).
   **Role-gate it:** do NOT split wheels (and other cohesive round/rubbery props) — keep
   them whole so a wheel falls off as a unit instead of separating into tire/rim/nuts.

4. **Classify** each part into a role (`classifyVehiclePart`): name keywords first
   (`wheel|tire`, `body|cage|chassis|frame`, `door|panel`, megascans/`aset_|barrel|
   crate|log` → cargo, `chain|rope|bucket` → accessory), then geometry/position
   fallbacks (long central spanning member → frame; low + compact + square → wheel; thin
   slab → panel; small + high → cargo). Classify by the **parent mesh** so all its
   islands share one role.

5. **Fracture large concave structural parts** (frame/panel only) with three-pinata
   Voronoi. A welded roll cage is *topologically one piece*, so splitting can't help —
   its convex hull is a blob. Fracturing into chunks gives each a tight local hull AND
   kills the bond-star (step 6). **Volume-scale the chunk count** (big → many, small →
   few; floor by longest dimension for thin tubes). Gate: only `frame`/`panel` past a
   size threshold; never wheels/cargo. Guard with try/catch → keep-whole fallback
   (artist meshes are often non-manifold).

6. **Bond by real surface contact** — use the WASM auto-bonder
   (`buildScenarioFromFragmentsAsync` with `bondMode:'auto'`,
   `autoBondingOptions:{ mode:'average', maxSeparation }`). This calls
   `createBondsFromTriangles`, which bonds where meshes actually touch (wheel↔hub,
   cargo↔the surface it rests on) with real area/normal/location. **Do not** use
   centroid/proximity bonding — it makes a "star" of bonds radiating from a central
   node, and parts detach in bizarre ways. Use `mode:'average'` (not `'exact'` — separate
   artist meshes don't share exact triangle faces). Tune `maxSeparation` (~0.06–0.2 m):
   too small and wheels/loose parts find no contact (orphans).

7. **Prune the over-bonding (role-aware), apply the hierarchy, guarantee connectivity.**
   Average-mode bonding over-connects a dense fractured model — thin only the *prop*
   attachments (wheel/cargo/accessory) while keeping every chassis bond (gotcha #21).
   Scale each bond's area by a
   role-pair multiplier (frame↔frame strongest, cargo/accessory weakest), then union-find
   the bond graph and stitch any leftover components together so nothing is an orphan
   (an orphan detaches and drops on frame 1). Per-node colors by role for the view.

8. **Build the core** (`buildDestructibleCore`) with **high `materialScale`** (1e12 —
   solid under gravity; at 1e10 it silently fragments into ~28 pieces just settling
   because auto-bond contact areas are small). Render with `createDestructibleThreeBundle`
   using `nodeColors` (role colors) — it renders the *real* per-part geometry, so the car
   keeps its silhouette.

9. **Breaking (threshold-based — the part that actually works on a free body):**
   - **Direct hits:** pass an `onImpact({nodeIndex, force})` callback to
     `buildDestructibleCore` (added in `destructible-core.ts`; fires for projectile/ground
     contacts). When `force` exceeds a per-role threshold, `core.cutNodeBonds(node)` —
     the part detaches and falls *keeping its mesh*, with a small bounded splash.
   - **Inertial shedding:** each frame, for bodies moving/spinning violently, cut the
     bonds of cargo/accessory (and optionally all) nodes on them — so a car that's flung
     and tumbling sheds its payload (which a pure contact path never would).
   - **Debris collision:** `debrisCollisionMode: 'noDebrisPairs'` (debris bounces off the
     car + ground but not other debris) or `'debrisGroundOnly'` (safest). NOT `'all'` —
     that's the explosion. Verify with the soak.

10. **GUI for live tuning:** expose **per-role detach thresholds** and global
    impact/shed sensitivity as **live** sliders. (Wiring sliders to bond *areas* feels
    dead — see truth #1.) Per-role + global multipliers on the thresholds, applied every
    frame, give instant, satisfying tuning.

## Build, run, verify (don't skip the last one)

```bash
# One-time: Emscripten (for the WASM stress solver) — emsdk 3.1.51 at /opt/emsdk
git clone --depth 1 https://github.com/emscripten-core/emsdk.git /opt/emsdk
/opt/emsdk/emsdk install 3.1.51 && /opt/emsdk/emsdk activate 3.1.51
source /opt/emsdk/emsdk_env.sh

cd <repo>
npm run setup:demos                       # installs deps
source /opt/emsdk/emsdk_env.sh
npm --prefix blast/blast-stress-solver run build   # builds WASM + lib (needs emcc)
npm run serve:demos                       # serves http://localhost:8000

# Iterate on a single demo fast (no full rebuild): esbuild just the changed file
cd blast/js_stress_example
npx esbuild glb-vehicle.ts destructible-vehicle.ts --format=esm --outdir=dist

# VERIFY HEADLESS before asking a human to QA:
npx playwright install chromium           # once
node scripts/soak-vehicle.mjs             # settle/shots/drop; FAILS on any blow-up
```

The soak harness drives the sim through scenarios over *time* and asserts no body
exceeds ~120 m/s (and a `metrics()` probe counts bodies > 60 m/s = exploding debris).
**This is the single most important process lesson:** a screenshot at one instant hides
explosions and instability that only show up while the simulation runs.

## Gotchas & tips (the things that caught me)

1. **Verify the simulation headless before asking the human to QA.** Build a Playwright
   soak + a `window.__demo.metrics()` probe (max body speed, count > 60 m/s, spread).
   Single screenshots miss time-dependent blow-ups. This was the #1 wasted-trust moment.
2. **Free bodies don't stress-break** (truth #1). Don't chase materialScale to make hits
   work — drive breaking from `onImpact` + inertial shedding. Anchored demos differ.
3. **`'all'` debris mode → explosion** when overlapping chunks detach together. Use
   `'noDebrisPairs'` / `'debrisGroundOnly'`.
4. **The runtime builds a convex hull per node from `fragmentGeometries`** even if you
   don't pass a Rapier module (`destructible-core.ts` `buildColliderDescForNode`). So a
   disconnected/concave mesh → blob hull. Split + fracture before it ever gets there.
5. **A welded concave mesh won't split** (it's one component) — you must *fracture* it.
   Splitting and fracturing are different tools: split = separate disconnected shapes;
   fracture = break up one connected concave shape.
6. **Don't fracture/split wheels** (cohesive rubbery props). Role-gate both passes.
7. **Use `createBondsFromTriangles` `mode:'average'`**, not proximity and not `'exact'`.
   Proximity = centroid star; exact = no bonds (separate meshes share no faces).
   CAVEAT — `'average'` silently drops EVERY vertical (Y-normal) contact: it never
   bonds one part resting on top of another. Fine for a car (contacts are mostly
   lateral) but FATAL for stacked/coursed structures (walls, towers, masonry), which
   then collapse for lack of course-to-course bonds. For axis-aligned masonry place
   parts TOUCHING (gap 0) and use `'exact'` instead — it bonds all axes incl.
   vertical, by coplanar-overlap area. See repo AGENTS.md "Auto-bonding gotchas".
8. **A big spanning part auto-bonds to everything from a central centroid** → both the
   blob hull and the green bond-star. Fracturing it (gotcha #5) fixes both at once.
9. **maxSeparation is a double-edged sword.** Too small → wheels/loose parts find no
   contact (orphans → stitch fallback → star). Too large → the opposite: the average
   bonder inflates each chunk by `maxSeparation/minSide`, so in a *dense, fractured*
   model every part bonds to dozens of neighbours (a wheel to ~40 frame chunks). That
   over-connection is both the dense web you see in the bond debug view AND real
   over-strength. Keep it modest (~0.08–0.10) and prune (gotcha #21) rather than cranking
   it. Wheels that still won't bond are handled by the connectivity stitch.
10. **`cutNodeBonds` detaches but does NOT set `chunk.detached`**, and the damage
    system's destroy path *disables the collider* (the part vanishes — wrong for a
    vehicle). Use `cutNodeBonds` for "falls off"; track your own shed counter.
11. **Settle grace:** ignore impacts for ~0.5 s after (re)build so spawning/settling
    doesn't shed parts. A height-drop lands after the window.
12. **Coordinate handling:** bake `matrixWorld` → recenter geometry to origin (offset =
    its world centroid) → fragment = `{geometry: local, worldPosition: centroid + global
    offset}`. `fractureGeometry` preserves the input frame and *adds* `worldOffset`, so
    pass only the global offset — don't also add `part.center` (double-count bug).
13. **The GLB usually contains a huge ground plane** — exclude oversized parts.
14. **Geometry-only GLBs have no textures** — flat per-role coloring is fine (and is the
    most informative "show the hierarchy" view anyway).
15. **Headless console truncation red herring:** Playwright truncates long console
    objects; I briefly thought ~half the bonds were missing (a phantom "index bug"). Log
    `JSON.stringify(...)` + explicit counts (`undefMeta`, `maxNodeIdx`) to be sure.
16. **`build:ts` (tsup `clean:true`) wipes the WASM from `dist/`** → demo 404s on
    `stress_solver.wasm`. Run the full `npm run build` (rebuilds WASM via emcc) or restore
    the artifacts. Keep `emsdk_env.sh` sourced.
17. **Demos build two ways:** esbuild (local, bare imports + HTML import map) and tsc
    (prod `scripts/build-demo-site.sh`). Register a new demo in **all** of: tsconfig
    `include`, `package.json` (`build:demo:X` + the `build:web` chain), `demo-index.html`
    card, and add an asset-copy line to `build-demo-site.sh` if you ship a model.
18. **Repo-wide `*.glb` gitignore** (`blast/.gitignore`): un-ignore your demo asset with a
    `!assets/your.glb` negation in the local `.gitignore`, else `git add` silently skips it.
19. **Tuning sliders that feel dead** = they're wired to bond areas (truth #1). Wire them
    to the per-role *thresholds* (live) instead.
20. **Perf:** auto-bonding + split + fracture process a lot of triangles → first
    load/Reset takes a few seconds; high-poly parts make convex-hull colliders chunky.
    Decimating collider/bond geometry (keeping full-res for render) is the obvious win.
21. **Prune the over-bonding ROLE-AWARELY, or you trade one bug for another.** The dense
    web (gotcha #9) makes parts over-strong and the debug view unreadable, so thin it —
    but **only the prop attachments** (wheel/cargo/accessory ↔ anything): per part-pair
    keep the few largest-area contacts, and cap how many other parts each prop bonds to
    (so a wheel attaches at its hub, not to half the car). **Keep every intra-part bond
    AND every structural inter-part bond (frame/panel ↔ frame/panel).** I learned this the
    hard way: pruning the chassis bonds (intra siblings or frame-to-frame) drops the
    frame's rigidity → it jitters apart on settle and *explodes* on heavy hits (sparser
    bonds let detaching chunks cascade into debris-vs-body overlaps). Run the connectivity
    stitch after pruning to reattach anything orphaned. (See `pruneBonds` in
    `glb-vehicle.ts`.) Always re-run the soak after touching bonds — bond changes are the
    most common way to reintroduce the explosion.
22. **Debris mode interacts with bond density.** `'noDebrisPairs'` (debris bounces off the
    car, not other debris) is stable when the chassis is densely bonded, but once bonds
    are sparser a hard hit can still blow up via debris-vs-*body* overlap; `'debrisGroundOnly'`
    (debris hits only the ground) is the density-independent safe choice if you can accept
    parts falling through each other instead of bouncing.
23. **The bond debug lines are node-centroid → node-centroid**, not drawn at the contact
    point. So a bond to a large or central chunk *looks* long even when the surfaces touch
    — judge bonds by count/role-pair stats and where they attach, not just line length.

## Quick-start checklist for a new GLB

- [ ] `analyze-glb.mjs` the model; confirm roles + spot the ground plane.
- [ ] Copy `destructible-vehicle.{ts,html}` + `glb-vehicle.ts` as a starting point; point
      `MODEL_URL` at the new asset; commit the asset (mind the `*.glb` ignore).
- [ ] Tune `classifyVehiclePart` keywords/heuristics for the model's part names.
- [ ] Decide split/fracture role gates (don't split/fracture wheels-equivalent).
- [ ] Set `maxSeparation` so wheels/loose parts bond (check the `bonds by role pair` log).
- [ ] Keep `materialScale` high; confirm it's solid at rest (settle → ~2 bodies).
- [ ] Set per-role detach thresholds; confirm the hierarchy (cargo sheds, frame holds).
- [ ] `node scripts/soak-vehicle.mjs` → must PASS (no explosions) before human QA.
- [ ] Register the demo (tsconfig / package.json / demo-index / build-demo-site).
