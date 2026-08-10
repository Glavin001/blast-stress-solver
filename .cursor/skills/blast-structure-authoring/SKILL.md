---
name: blast-structure-authoring
description: >-
  Author destructible structures for the blast-stress-solver (ScenePack v2):
  choose bond areas and materials, calibrate with the gravity load-path report,
  and debug a structure that stands when it should break, breaks when it should
  stand, shatters instead of tearing, or refuses to collapse progressively.
  Use when creating or tuning a scene pack, picking stress limits, reading
  safety factors, or diagnosing destruction that looks wrong.
---

# Authoring destructible structures

This library ships **no material library**. There are infinitely many real
materials, every project's scale and units differ, and a prescriptive enum
would be a treadmill that still fits nobody. What it gives you instead is a
method: two axes, one measurement, and a symptom table.

Every claim below is backed by a test you can run and read. Where a row cites
a test, that test *is* the evidence — read it rather than trusting this file.

- Format spec: [`SCENE_PACK_FORMAT.md`](../../../SCENE_PACK_FORMAT.md)
- Behavior evidence: `demos/blast-stress-demo/tests/material_behavior_test.cpp`
- End-to-end quality (multi-material pack + resim): `demos/blast-stress-demo/tests/destruction_quality_test.cpp`
  (run it with `--sweep` to characterize a structure's response curve)
- Worked example: `blast/blast-stress-solver/scripts/export-reference-building.mjs`

## The two axes

> **Area is geometry. Material is strength. Never trade one for the other.**

| Axis | What it is | Where it lives |
|---|---|---|
| **Area** | The real contact patch, m². Also the bond's damage pool. | `bonds[].area` |
| **Material** | Stress limits, Pa. The only strength knob. | `defaults.solver.materials` + `bonds[].m` |

Bond area is simultaneously the denominator of stress *and* the damage pool, so
"make it stronger by giving it more area" corrupts the stress readout and
scales toughness at the same time. Doubling area halves stress; changing
material leaves stress untouched and moves only capacity
(`testAreaAndMaterialAreIndependentAxes`).

This is not a stylistic preference. An earlier revision of the high-rise pack
authored frame joints at 2×10⁵–10⁶× their geometric area, putting the frame
seven orders of magnitude below its elastic limit — unbreakable under any
impulse the simulation could produce, while every destruction gate still
reported success. The compensations cascaded from there: a 650 t projectile at
32× the density of osmium, and a 99× contact-force multiplier.

## The three material knobs

| Knob | Controls | Evidence |
|---|---|---|
| **Elastic limit** | *Whether* a joint fails: it yields when carried stress exceeds this | `testStrengthSetsFailureThreshold` |
| **Fatal − elastic band** | *How* it fails: narrow = snaps in one tick, wide = yields over many | `testBandWidthControlsBrittleVsDuctile` |
| **Per-mode limits** | Asymmetry: concrete carries ~10× more compression than tension | `testTensionAndCompressionAreIndependent` |

Ductility is the band width, **independent of strength**. Two materials with an
identical elastic limit and a 1.05× vs 500× band behave completely differently:
one snaps, the other survives 400 steps of the same load.

## The measurement

Run any scene with a negligible projectile to get gravity-only loads:

```bash
blast_stress_demo --physics cpu --grid 1 --settle 2 \
  --scene your-pack.json \
  --projectile-mass-scale 0.0001 --projectile-speed-scale 0.0001
```

It prints, before any gate fires:

```
gravity load path (utilisation = peak stress / that bond's material elastic limit):
  column~infill      facade-clip          bonds=48  peakUtil=0.3316  safetyFactor=3.015  peak(c/t/s)=1.66e+05/6.16e+03/1.79e+05 Pa
  infill~slab        drywall-panel        bonds=24  peakUtil=0.3101  safetyFactor=3.225  ...
  column~slab        reinforced-concrete  bonds=20  peakUtil=0.09665 safetyFactor=10.35  ...
  column~foundation  reinforced-concrete  bonds=4   peakUtil=0.03126 safetyFactor=31.99  ...
```

`safetyFactor = 1 / utilisation`. The same table lands in
`metadata.gravityLoadPath`. **`peak(c/t/s)` tells you which mode is binding** —
move that mode's limit, not all six.

### Target bands

| Class | Safety factor | Why |
|---|---|---|
| Base anchor | 30–70 | Never the failure point |
| Frame | 5–20 | Real structural design margin |
| Facade / cladding | 2–4 | The deliberate weak link |

Gate both ends on any run whose numbers you intend to trust:

```
--require-min-safety-factor 2 --require-max-safety-factor 2000
```

The **upper** bound is the one that catches the insidious failure. A joint at
safety factor 10⁵ is not strong — it carries no measurable load and cannot be
broken by anything, so destruction results become vacuous while every damage
gate still passes. The two gates read different statistics on purpose: the
lower bound uses the class **peak** (the worst joint must stand), the upper
bound uses the **mean** (over-authoring is systematic, and peak is defeated by
a single sliver bond in a Voronoi-fractured pack).

## The calibration loop

1. Author geometry; set every `area` from the section it represents.
2. Guess materials. Real-world Pa values are a fine starting point.
3. Run the measurement above.
4. Move the material whose safety factor is outside its band, **in the binding
   mode only**.
5. Repeat. It converges in two or three passes.

The worked example did exactly this, twice. The first pass put the facade clip
at safety factor 1.06 — the building stood, but its cladding was one gust from
letting go. Shear was the binding mode, so only the clip's shear limits moved.
Later, splitting the panels in two halved their bonded edge and dropped the
facade back to 1.08; again shear, again only shear moved. Final: 3.02/3.23. See
the comment block in `export-reference-building.mjs`.

## Symptom table

| Symptom | Most likely cause | What to try |
|---|---|---|
| Collapses under its own weight during warmup | Some class has safety factor < 1 | Read the load-path table — it names the class. Raise that material's binding mode. |
| Nothing breaks no matter what you throw | Over-authored area, or limits too high | Check safety factors. Anything in the thousands carries no load. Fix the **area** back to geometry; do not compensate with a heavier projectile. |
| Needs an absurd projectile to do anything | You are compensating for an authoring error | Check projectile density in `metadata.tuning`. Above ~20,000 kg/m³ you have left physical reality. Fix the structure instead. |
| Shatters into dust instead of tearing in chunks | Fatal−elastic band too narrow | Widen the band. Same elastic limit, higher fatal → yields progressively (`testBandWidthControlsBrittleVsDuctile`). |
| Panels hinge instead of detaching | Band too wide at the facade | Narrow the *facade* band so clips snap; keep the frame band wide. |
| Wrong thing breaks first | Failure follows utilisation, not load | The least-loaded joint fails first if its material is weakest (`testWeakestLinkFailsFirstRegardlessOfLoad`). Compare safety factors, not stresses. |
| Cladding rips the frame down with it | Facade and frame too close in capacity | Widen the gap. Facade 2–4, frame 5–20 (`testFacadeShedsWithoutDroppingFrame`). |
| Holds its weight but explodes on contact | Tension/shear far below compression | Intended for concrete — but if it is too fragile, raise tension/shear specifically (`testTensionAndCompressionAreIndependent`). |
| No progressive collapse — damage stays local | Survivors have too much margin | Redistribution is emergent, never scripted. Cutting one of two legs raised the survivor 0.022 → 0.121 (`testLoadRedistributesOntoSurvivors`). If survivors sit at safety factor 30, they will absorb it. Lower frame margin toward 5–10. |
| Utilisation never changes after damage | Settled-island skip is serving cached values | A **fully supported** structure that fractures does not re-solve until something moves. Real scenes self-correct (detached pieces move). To measure, set `skipSettledIslands = false` and `idleSkip = false`. |
| Numbers differ between Rapier and PhysX | Expected for trajectories, **not** for structure | Asset interpretation is pinned by the conformance digest. If node/bond/material counts differ, that is a loader bug — see `SCENE_PACK_FORMAT.md`. |

## Non-negotiables

These are the knobs that look like tuning and are actually corruption:

- **`contactForceScale` stays 1.** Both engines multiply a contact *force* by
  it (PhysX divides its solved impulse by `dt` first), so 1 is the physically
  correct transfer. Anything else is gain that must be cancelled elsewhere.
- **`stressLimitScale` stays 1** unless you are deliberately sweeping global
  strength. It and `contactForceScale` are opposite ends of the same equation;
  moving both hides the error.
- **Get more destruction by adding energy, not gain.** Scale the projectile's
  radius with mass ∝ r³ so its density stays real.
- **The adapter never overrides a fracture verdict.** If something breaks that
  should not, fix the authoring — there is deliberately no veto to reach for
  (`PHYSICS_ENGINE_CONTRACT.md`, verdict-integrity invariant).

## Worked example

`blast/blast-stress-solver/assets/reference/reference-building.json` — 76
nodes, 180 bonds, 4 materials, 3 floors of a 2×2 bay frame with a ring beam at
every floor. Small enough to read end to end and re-measure in under a second.

Calibrated result (pinned by CTest `blast_stress_reference_building_load_path`):

| Joint class | Material | Safety factor |
|---|---|---|
| `column~infill` | facade-clip | 3.04 |
| `beam~infill` | drywall-panel | 3.20 |
| `infill~infill` | drywall-panel | 3.29 |
| `beam~column` | reinforced-concrete | 6.26 |
| `column~slab` | concrete-slab | 7.71 |
| `beam~beam` | reinforced-concrete | 8.95 |
| `column~column` | reinforced-concrete | 12.5 |
| `slab~slab` | concrete-slab | 13.8 |
| `beam~slab` | concrete-slab | 14.0 |
| `column~foundation` | reinforced-concrete | 32.5 |

Destruction quality, measured through `ExtStressPhysXFrameStepper` with
**resim = 1** and unity contact gain — pinned by CTest
`blast_stress_destruction_quality`:

| Impact | Splits | Bodies | Largest piece | Standing | Reading |
|---|---|---|---|---|---|
| gravity only | 0 | 1 | 76/76 | 100% | stands |
| 400 kg @ 12 m/s | 0 | 1 | 76/76 | 100% | shrugs it off — **not glass** |
| 1.5 t @ 16 m/s | 4 | 25 | 50/76 | 66% | facade shed, frame untouched |
| 4 t @ 20 m/s | 13 | 63 | 5/76 | 5% | frame comes down |
| 40 t @ 45 m/s | 2 | 74 | 2/76 | 5% | everything breaks — **not rigid** |

"Largest piece" is the biggest connected body left. It is the number that
separates destruction from dust. Foundations never move at any energy — they
are the only world-fixed nodes.

### Redundancy widens the band and flattens it — read both numbers

This building was a **tree** before the ring beams: one column per slab
quadrant, redundancy only through the slab diaphragm. Adding a beam ring is the
textbook fix, and measured against the tree it looks like a clear win — the
"standing 0.25–0.85" band goes from a 4.0× spread of impact energy to 5.2×, and
the surviving piece grows from 12–32 chunks to 47–50.

It is not a win, and the reason is a trap worth internalising: **standing
fraction counts chunks, not structure.** 66% standing here is almost exactly
"every frame chunk up, every facade panel shed". Across the whole widened band
`moved(column)` is **0** — the frame is not damaged at all, it is just naked.
The frame's own response went from graded (50% → 39% → 33% as energy rose) to a
step: nothing, then total collapse, over a 300 kg interval.

So when a change widens the partial band, check **which role moved**, not just
how much is standing. `moved(...)` per role is in the sweep output for exactly
this reason. A redundant structure resists damage — that is what redundancy
*is* — so it buys a wider band of *cladding* results and a narrower, sharper
transition in the frame. Whether that is what you want is a design decision,
not a quality metric. The full experiment, including two failed attempts to
recover the gradient, is in the header of `export-reference-building.mjs`.

## Two lessons this example paid for

**Fracture granularity decides whether debris reads as pieces or dust.** The
first version of this building used one node per structural element. A
moderate hit atomized it: 30 bodies out of 31 chunks, largest surviving piece
2 chunks. Elements are now built from several chunks bonded to each other with
full-section monolithic material, so a joint failure releases a *piece* that
holds together. Same materials, same energies — the difference is topology.

**Changing geometry means re-running the calibration.** Splitting the panels in
two halved each panel's bonded edge, which doubled the stress its own weight
put through it, and dropped the facade from safety factor 3.5 to 1.08. The fix
was raising that material's **shear** limits (the binding mode in the report),
not its strength across the board. Geometry and material are independent axes,
but they are not independent *decisions*: change one and re-measure.

Note the coupling in the other direction too — strengthening the facade to hit
its gravity target moved the impact threshold up with it, so the "interesting"
energy level moved from 500 kg to 1.5 t. Expect to re-locate the band after any
material change.

The ring beams paid this a third time, and harder. Adding them moved **eight**
of ten joint classes out of band at once: the beams' own 13.4 t pushed the
anchor under its floor (32.0 → 26), the slab bearing on a beam is a 0.66 m²
patch so it read 35 on the frame material, and the facade — whose geometry only
changed by getting shorter — dropped to 1.52 on a mode nothing had loaded
before (the panel seam, in pure tension). Two of those needed a different
*material assignment* rather than a different value: `column~slab` and
`beam~slab` only sit in the frame band on the **slab** material. Budget a full
re-measure, not a tweak, whenever the load path itself changes shape.
