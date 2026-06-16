# GLB → High-Quality Destructible Pipeline

Turns an arbitrary **GLB model** into tight, **non-overlapping convex collider
pieces** that a Rapier + Blast-stress-solver demo can render *and* break, without
the "detached chunks explode" failure that overlapping convex hulls cause.

This document is the field guide: the **process** that produced it (so you
understand the *why*), the **install steps**, and how to **run/continue** it on
this branch (`claude/glb-coacd-destructible-pipeline`).

---

## 1. The problem (why this pipeline exists)

The runtime builds **one convex hull collider per node**. While nodes are bonded
into one rigid body those hulls are inert (Rapier ignores same-body collider
pairs), but the instant a piece detaches it **deeply interpenetrates its
neighbours**, and the contact solver resolves that penetration as an explosion.

Root causes we measured on the bundled buggy (`assets/buggy.glb`):

1. **Convex-hull overshoot** — a concave part's single hull bulges into its
   neighbours (up to **1.24 m** penetration).
2. **Open-shell triangle soup** — ~96% of parts are non-watertight, so vertex
   welding and connected-component splitting are wrong, boolean ops fail (no solid
   volume), and CoACD produces thin sliver pieces.

The fix is a build-time pipeline that produces colliders that **don't overlap**:

```
GLB ──▶ clean-glb.mjs ──▶ build_destructible_asset.py ──▶ buggy.pieces.json ──▶ runtime
        (weld+simplify)    (split → CoACD → de-interpenetrate)        (render+collide)
```

### How we got here (the journey, in order)

- **Diagnose first.** `diagnose_overlap.py` quantified the overlap (1498 pairs,
  1.24 m max) — pure-geometry, fcl penetration depth.
- **CoACD** (convex decomposition) replaces each bulging hull with tight hulls.
  Necessary but **not sufficient**: parts still overlap where they interleave.
- **De-interpenetration.** Boolean trim does nothing (open shells = no volume);
  inset can't shrink thin pieces. **Plane-clipping** works: where two cross-part
  pieces overlap, clip the **larger** one's lobe along the contact plane (small
  detail pieces keep their full shape). Clip the larger one → **0 overlap**.
- **Watertight matters.** The thin slivers that clipping *can't* remove (it would
  erase them) come from the open-shell source. A **glTF-Transform weld+simplify**
  pre-pass (the gltfjsx `--transform` idea) welds the soup into manifold parts and
  simplifies away the slivers: **280→131 parts, 2520→527 pieces, 74 mm→11 mm**
  residual overlap, asset **8 MB→1.5 MB**.
- **Runtime calibration.** CoACD pieces have small contact areas → bonds need a
  high `materialScale` to hold at rest; impact breaking is then driven by
  `contactForceScale`. Wheels/props get unbreakable internal bonds (single rigid
  body, tight collider).

Everything is **soak-verified** (`scripts/soak-vehicle.mjs`): holds at rest,
sheds parts on light/heavy shots, stable drops, no explosions.

---

## 2. Prerequisites & install

### Node (the demo + clean-up pre-pass)

```bash
# repo root: vendor libs the served demo needs (three, rapier, pinata)
npm install

# the destructible-vehicle demo package
cd blast/js_stress_example
npm install

# glTF-Transform weld+simplify pre-pass (clean-glb.mjs)
npm install @gltf-transform/core @gltf-transform/functions meshoptimizer
```

### Python 3 (the offline geometry pipeline)

```bash
pip3 install coacd trimesh numpy python-fcl networkx scipy shapely manifold3d
```

| package | used for |
|---|---|
| `coacd` | convex decomposition (SarahWeiii/CoACD) |
| `trimesh` | mesh IO, convex hulls, plane slicing |
| `python-fcl` | exact penetration-depth collision queries |
| `networkx` + `scipy` | trimesh connected-component split |
| `shapely` | trimesh `slice_plane` cap polygons (**required** — without it the clip silently no-ops) |
| `manifold3d` | trimesh boolean backend (used by the `--method boolean` variant) |

### Emscripten — ONLY if you change the C++/Rust core or stress solver

The committed `blast/blast-stress-solver/dist/*.wasm` is enough to run the demo
and tests. To rebuild the solver:

```bash
git clone --depth 1 https://github.com/emscripten-core/emsdk.git ~/emsdk
~/emsdk/emsdk install latest && ~/emsdk/emsdk activate latest
source ~/emsdk/emsdk_env.sh
npm --prefix blast/blast-stress-solver run build      # builds WASM + TS
node blast/blast-stress-solver/scripts/copy-dist.js   # if build:ts cleaned the wasm
```

### Playwright (headless verification)

```bash
cd blast/js_stress_example
npx playwright install chromium   # once
```

---

## 3. Run the pipeline (GLB → asset)

From `blast/js_stress_example/scripts/glb-pipeline/`:

```bash
# (optional) measure overlap on the RAW model
python3 diagnose_overlap.py ../../assets/buggy.glb

# 1) clean: weld the triangle-soup + simplify away thin slivers
node clean-glb.mjs ../../assets/buggy.glb /tmp/buggy-clean.glb \
     --ratio 0.3 --error 0.01 --weld 0.0008

# 2) decompose + de-interpenetrate -> the runtime asset
python3 build_destructible_asset.py /tmp/buggy-clean.glb \
     --threshold 0.1 --method clip --clip-margin 0.001 \
     -o ../../assets/buggy.pieces.json
```

Watch the build output — it reports, and **never fails silently**:

```
[build] CoACD: 131 parts -> 527 hulls in 163s
[build] overlap BEFORE clip: pairs=836  maxDepth=0.933m
[build] clip: 1014 clips skipped (would erase a too-thin piece)
[build] overlap AFTER  clip: pairs=70  maxDepth=0.011m  meanDepth=0.000m
[build] wrote ../../assets/buggy.pieces.json
```

CoACD results are cached in `/tmp/coacd_<glb>_<...>.pkl`, so re-running with
different `--clip-margin` is fast. CoACD itself runs in a **subprocess**
(`_coacd_worker.py`) so a C++ hard-abort on a bad piece becomes a reported
fallback, not a dead build.

### Knobs

| flag / setting | effect |
|---|---|
| `clean-glb --ratio` | simplification target (0.3 = keep 30% of faces). **Lower = fewer thin slivers** (closer to `debris:'all'`-clean) but blockier. |
| `clean-glb --weld` | vertex weld tolerance (m). Bigger = merges more of the soup. |
| `--threshold` (CoACD) | concavity threshold; lower = more, tighter pieces. |
| `--clip-margin` | final gap between pieces (m). 1 mm keeps contact areas large (strong bonds). A gap is only needed to stop touching-pairs re-detecting. |
| `--method clip\|boolean` | `clip` is the robust default. `boolean` is exact (subtract smaller from larger on the watertight CoACD hulls) but CoACD hard-aborts re-decomposing notched meshes — kept for reference. |

`diagnose_overlap.py` and `decompose_coacd.py` are standalone analysis tools
(quantify overlap; CoACD before/after comparison).

---

## 4. Runtime integration

The asset is loaded by `glb-vehicle.ts` → **`buildVehicleScenarioFromAsset()`**:
each piece becomes a node (render + collider); pieces of one part share a `partId`
so they bond internally. `assembleVehicleScenario()` (shared with the old GLB
path) does the auto-bonding, role/geometry-aware bond strength, cargo capping and
connectivity stitching. Key tables in `glb-vehicle.ts`:

- `INTER_ROLE_MULTIPLIER` / `INTERNAL_ROLE_MULTIPLIER` — the bond-strength
  hierarchy. Wheels/accessories use `UNBREAKABLE` internal bonds (single body).
- `thinFactor` in `assembleVehicleScenario` — thin members weaker than thick.
- `classifyVehiclePart` — role from name keywords + geometry/position.

`destructible-vehicle.ts` `CONFIG` is the live tuning surface. Soak-verified
values for the current asset:

```ts
solver.materialScale   = 1e13   // holds the intact car (small CoACD contact areas)
physics.contactForceScale = 200 // amplifies impact force so hits still shed parts
debrisCollisionMode    = 'noDebrisPairs'  // see Known Limitations
```

It loads `./assets/buggy.pieces.json` (committed). The deploy copies the whole
`assets/` dir (`scripts/build-demo-site.sh`), so committing the asset is enough.

The core gained `core.shatterAll()` (fracture every bond → each chunk its own body
with zero imparted velocity) — a real demolition op and the driver for the
acceptance test.

---

## 5. Build, serve, verify

```bash
# build the demo bundle
cd blast/js_stress_example && npm run build:web

# serve the whole repo (maps /vendor/* to three/rapier/pinata/blast dist)
node ../../scripts/serve-demo.mjs --port=8000
# open http://localhost:8000/blast/js_stress_example/destructible-vehicle.html

# headless checks (server must be running):
node scripts/soak-vehicle.mjs    # settle/light/heavy/drop — must report no explosions
node scripts/shatter-test.mjs    # debris=all + shatterAll -> collapse & settle (the strict gate)
node scripts/probe-vehicle.mjs   # sweep projectile mass/speed -> shed/stability
```

Core mechanism test (needs the WASM build):

```bash
cd blast/blast-stress-solver
npx vitest run src/tests/freeBodyGroundStress.test.ts   # free body on ground breaks under stress
```

CI deploys a Vercel preview on every push to an open PR (`.github/workflows/ci.yml`
→ "Deploy Preview"); the URL is posted/updated as a comment on the PR.

---

## 6. Current state, limitations & next steps

**Working:** complete buggy, holds rock-solid at rest, sheds parts on shots/drops,
no explosions. Soak-green.

**Known limitation — `debris: 'noDebrisPairs'`, not `'all'`.** A handful of thin
sliver pieces the source model still produces can't be fully de-interpenetrated
(clipping would erase them → ~70 residual ~11 mm overlaps), which **cascade** into
an explosion under full `'all'` debris collision in a mass shatter. `noDebrisPairs`
(debris bounces off the car + ground, not each other) is robust and lively.

**To reach full `'all'`:** push the clean-up harder — `clean-glb.mjs --ratio 0.15
--error 0.02` (fewer slivers, blockier), and/or drop/merge pieces below a size
threshold in `build_destructible_asset.py`, until `shatter-test.mjs` settles with
`'all'`. Then set `debrisCollisionMode = 'all'` in `destructible-vehicle.ts`.

**Other follow-ups:** drop the now-unused GLB-path imports from
`destructible-vehicle.ts`; widen `bondMaxSeparation` if stitch "star" bonds remain;
consider render mesh ≠ collider (keep original render mesh, CoACD only the
collider) for higher visual fidelity (needs multi-collider-per-node in the core).

**Branch:** `claude/glb-coacd-destructible-pipeline` (PR #72).
