---
name: blast-destruction-diagnostics
description: >-
  Diagnose destruction that looks wrong on screen — nothing happens and then the
  structure detonates, debris sprays instead of falling, pieces jitter or
  explode on first contact, a building collapses with no visible cause. Measures
  the run instead of watching it: interpenetration, bond normals, sliver bonds,
  silent damage accumulation, and a progressive-vs-explosive score from frame
  telemetry. Use when a recorded demo looks bad, before re-recording, or when a
  scene needs a verdict that does not depend on somebody's opinion of a video.
---

# Diagnosing bad-looking destruction

Companion to `blast-structure-authoring`, which covers *authoring* areas and
materials. This skill covers *debugging* a scene that already exists and looks
wrong.

## The rule that matters most

**Check geometry before you touch a material.** Every genuine defect found in
the session that produced this skill was geometric — interpenetration, wrong
bond normals, shard size, a missing bearing surface. Material tuning was
attempted first and wasted hours, because a bad load path re-concentrates
wherever you strengthen it. If the geometry is wrong, no material table fixes
it; you only move the failure somewhere else.

Order of investigation:

1. Interpenetration (`analyze_overlap.py`)
2. Bond normals, sliver bonds, exact contact areas
3. Stability at rest (splits must be 0 under gravity alone)
4. Limit-scale invariance (is stress independent of strength?)
5. Only then: materials and ductility

## Never judge from the video

Watching a recording tells you *that* it looks wrong, not *why*, and it costs a
full render each iteration. Score the run instead:

```sh
blast_stress_demo ... --frame-telemetry /tmp/run.csv
python3 demos/blast-stress-demo/tools/analyze_destruction.py /tmp/run.csv \
  --require-progressive
```

Three numbers decide it:

| metric | meaning | bad |
|---|---|---|
| `deadImpactFraction` | impacts that landed before the first split | > 0.5 — "balls bounce off and nothing happens" |
| `burstFraction` | largest share of damage in one 0.5 s window | > 0.5 — "and then it exploded" |
| `peakFrameShare` | share of damage in the single worst step | > 0.15 — a cascade, not an impact |

Burst metrics are noise below ~30 splits (`MIN_SPLITS_FOR_BURST`); a run with 11
splits reports `LIGHT`, not `EXPLOSIVE`. A healthy run has damage spread over
hundreds of steps with the worst step around 1% of the total.

The two ends are usually the same bug: a dead period means damage is
accumulating invisibly, and the cascade is that stored damage releasing at once.

## Failure taxonomy

### "Nothing happens, then it detonates"

Look at `overstressed_bonds` per second in the telemetry CSV against
`splits_frame`. Bonds sitting overstressed for seconds with zero splits means
damage is draining pools silently; when they empty, the released load
redistributes and cascades.

- **Ductile cladding.** A wide fatal-elastic band on facade joints makes a
  struck panel absorb damage for seconds instead of letting go. Cladding should
  be **brittle** (band ≈ 1.2) so a hit detaches a panel at the impact site and
  reads on screen. Keep the frame ductile (band ≈ 10) so it yields rather than
  snapping. Ductility is independent of strength: the elastic limit decides
  *whether* a joint fails, the band decides *how*.
- **Fragments too large.** Fracturing a full wall face at once yields shards
  metres tall and tonnes heavy — nothing a projectile can locally dislodge, and
  losing one strips a third of a facade. Band the fracture per storey so shard
  size is panel-scale.

### "It explodes / debris sprays outward"

Almost always interpenetration.

```sh
python3 demos/blast-stress-demo/tools/analyze_overlap.py pack.json --require-clean
```

Bonded bodies that overlap are compressed springs: the bond hides the
penetration, and the contact solver ejects both the instant it breaks. Note that
AABB overlap is *expected* for slanted convex shards and means nothing — the
tool resolves true hull overlap by sampling, and treats flush face contact as
zero.

Common sources:
- Authoring a frame *embedded* in a wall so proximity bonding finds contacts.
  Fix the geometry (place flush) rather than keeping the overlap.
- Columns run the full storey height into the slab sitting at the top. Stop them
  at the slab soffit.

### "One joint class sits at safety factor ~1 and won't improve"

Raise its limit and re-measure. If peak stress rises roughly in proportion and
utilisation stays pinned near 1, **the load is redistributing to whatever is
strongest** and tuning that material is futile.

Confirm with a global scale sweep:

```sh
for S in 1 10 100; do blast_stress_demo ... --stress-limit-scale $S; done
```

A correctly-modelled class scales cleanly — stress constant, safety factor ×10,
×100. A class whose stress *grows* with the scale and then saturates is yielding
under gravity at normal scale; the saturated value is the true elastic peak. If
that peak is absurd (tens of MPa on a cladding tie), it is a stress
concentration from the model, not a material shortfall.

The usual cause is bonding an entire continuous interface (e.g. a shard glued up
the whole column face) where the real connection is discrete anchors. A long
stiff bonded interface concentrates enormous stress at its ends.

### "Safety factor is set by one absurd bond"

Check the bond-area distribution per class. Contact patches below ~10 cm² are
geometrically real but structurally meaningless, and since stress is force/area
they appear as singularities. In one measured pack, 9 sliver bonds out of 2470
produced a peak 144× the class mean and set the reported safety factor for the
whole facade. Drop them with a `MIN_BOND_AREA` floor (1e-3 m²); the shards keep
their real bonds.

Remember which statistic a gate reads: `--require-min-safety-factor` uses the
**peak** (worst joint must stand), `--require-max-safety-factor` uses the
**mean** (catches unbreakable over-authoring). A peak-based gate can fail on a
local concentration that never actually breaks anything — cross-check with a
long gravity-only run and confirm `splits=0`.

## Bond normals are load-path physics, not metadata

`normal` must be the **true contact-surface normal**, never the direction
between centroids. The solver splits each bond's load into normal
(compression/tension) and tangential (shear) about this vector, so a wrong
normal books compression as shear.

A shard bearing on a slab ledge sits above and outboard of the slab centre, so
centroid-to-centroid is diagonal — its own weight gets charged against the weak
shear limit instead of the concrete compression limit. That alone put a facade
below safety factor 1 standing still.

- stacked contacts (column/column, column/slab, panel on ledge): `[0,1,0]`
- in-plane cuts (slab to slab): the cut axis
- flush vertical interface: the face outward axis
- fracture seams: the cut-plane normal (for Voronoi, the bisector direction)

Related: compression through a flush **concrete-to-concrete** contact is
governed by the concrete (~12 MPa), not by the connector. Authoring a low
compression limit because the joint is "weak" is wrong — the weakness belongs in
tension and shear.

## Prefer exact contact bonding over proximity

Proximity bonding needs tolerance fudge factors and tempts you into
interpenetrating geometry so it finds contacts. Options, best first:

1. NvBlast's own contact-based generator (`createBondsFromTriangles`, mode
   `exact`) via `generateAutoBondsFromChunks`.
2. Closed-form contact when you author the geometry: for a Voronoi panel built
   by half-plane clipping, shard~shard area is the shared edge length ×
   thickness and shard~frame area is an exact polygon-rectangle intersection.
   This also gives provably exact tiling (cell areas sum to the panel rectangle)
   and full seed determinism.
3. Proximity — last resort.

Verify tiling numerically: total shard volume must equal the analytic panel
volume. If it does, there are no gaps and no overlaps by construction.

### Cross-check closed-form bonds against the library's generator

Closed-form contact is fast and deterministic, but the arithmetic is only
trustworthy once something independent agrees with it:

```sh
node blast/blast-stress-solver/scripts/verify-bonds-against-autobonding.mjs pack.json
```

It runs NvBlast's triangle-based generator over the same geometry and diffs the
bond sets. Read the columns as:

- **GAPPED** — the pack bonds two bodies that are actually apart. Always a bug.
- **MISSING** — a contact the pack does not bond. A real omitted load path
  unless it is below the pack's `MIN_BOND_AREA` floor.
- **AREA** — both find the contact, areas disagree.
- **TOUCHING** — extra bonds whose bodies are flush. **Not a defect.** EXACT
  mode looks for *common surface* and does not reliably report
  exactly-coplanar faces between independently triangulated meshes, e.g. a
  panel bearing on a slab ledge. Do not "fix" these by deleting bonds; the tool
  re-measures the gap so you can tell them apart. AVERAGE mode is not the
  answer either — measured on one pack it covered 45% of authored bonds versus
  EXACT's 82%.

On the pack this skill came from, that diff caught two real bugs closed-form
math had hidden: every column bonded to its footing across a **1 mm gap**
(`BASE_Y = clearance + 2*FOUND_HALF` put the column base above the footing
top), and columns bonded to only **one** of the several slab cells they
straddle, giving that cell the column's whole section area (36% area error, 429
missing contacts). Both were invisible to the stress gates and to the eye.

### Building the WASM runtime

The generator lives in the Emscripten build:

```sh
git clone --depth 1 https://github.com/emscripten-core/emsdk /root/emsdk
cd /root/emsdk && ./emsdk install latest && ./emsdk activate latest
source /root/emsdk/emsdk_env.sh
cd blast/js_stress_example && node scripts/build.js
cd ../blast-stress-solver && node scripts/copy-dist.js
```

Emscripten ≥ 4 needs two flags the older build predates, both already in
`scripts/build.js`: `-sDEFAULT_TO_CXX=1` (the driver is `emcc` over C++ sources,
so the link otherwise fails on `operator new` / `__cxa_throw`) and
`-sINCOMING_MODULE_JS_API=[locateFile,wasmBinary,...]` (the loader supplies
those and newer Emscripten aborts on undeclared Module properties).

Smoke-test with geometry whose answer you know — two unit boxes sharing a face
plus one isolated box should give exactly one bond of area 1.0 with normal
(1,0,0):

```js
const bonds = await generateAutoBondsFromChunks([mk(0), mk(1), mk(8)], { mode: 'exact' });
```

## Determinism

Third-party fracture libraries may not seed their internal cell RNG from your
scene seed, so identical inputs produce different packs. Check before trusting a
"reproducible" pipeline:

```sh
SEED=7 node scripts/export-....mjs && md5sum out.json
SEED=7 node scripts/export-....mjs && md5sum out.json   # must match
```

If it drifts, either freeze a verified artifact or own the fracture so one
seeded RNG drives every choice.

## Acceptance checklist before recording

1. `analyze_overlap.py --require-clean` → 0 m³
2. `verify-bonds-against-autobonding.mjs` → 0 GAPPED, 0 MISSING
3. Shard volume equals analytic panel volume
4. Gravity-only run, long settle → `splits=0`, drift ≈ 0
5. `analyze_destruction.py --require-progressive` → `PROGRESSIVE`
6. Damage monotonic across an energy sweep, with no shattered outliers
7. Same seed twice → identical checksum

Only then render, and inspect extracted frames yourself before showing anyone.
