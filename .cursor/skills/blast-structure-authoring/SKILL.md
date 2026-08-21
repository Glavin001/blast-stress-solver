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

**If a scene already exists and looks wrong on screen, start with
[`blast-destruction-diagnostics`](../blast-destruction-diagnostics/SKILL.md)
instead of this file.** Everything below assumes the geometry is sound. When a
structure detonates, sprays debris, or does nothing and then collapses, the
cause is usually geometric (interpenetration, wrong bond normals, a missing
bearing surface, oversized fragments) and no material table will fix it — you
will only move the failure elsewhere. That skill checks geometry first and
scores runs from telemetry instead of from watching a video.

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

## The fourth knob: chunk crushing (ScenePack v3, opt-in)

The three knobs above decide when a **joint** fails. Crushing decides when a
**chunk itself** is comminuted and leaves the simulation as dust. Both happen in
one impact: a wall separates along its joints, and the small region under the
hit is ground up.

Off unless a material authors a `crush` block, and off is byte-identical to
before it existed — verified by running the crush-enabled reference building
with `--no-crush` against the plain v2 pack (`testDisabledByDefault`).

| Knob | Controls | Evidence |
|---|---|---|
| **`capPressure` + `cohesion` + `frictionSlope`** | *Whether* a chunk comminutes: the Drucker–Prager yield surface | `testOverstressedChunkCrushes` |
| **`crushEnergy`** (J/m³) | *How much* is lost: the specific comminution energy | `testCrushEnergyControlsHowMuchIsLost` |
| **`crushViscosity`** (Pa·s) | *How fast*, once past yield | `testFlowIsQuadraticInOverstress` |
| **`debrisMassFraction`** | How much mass comes back as rigid fragments instead of dust | `testDebrisFractionRespawnsMass` |

Don't dial the cone — derive it. Under an unconfined squeeze at a material's own
compressive strength `fc`, the state is `p = fc/3, q = fc`, so

```
cohesion = fc * (1 - frictionSlope/3)     frictionSlope ~ 1.2 for concrete
capPressure = 2.5 * fc                    confined pore collapse
```

makes a chunk yield at exactly the `fc` its bonds already use. One fewer number
to invent.

### Measure it the same way you measure joints

```
crush utilisation = max( q / (cohesion + frictionSlope*p),  p / capPressure )
crush safety factor = 1 / crush utilisation
```

`getNodeCrushUtilisation` is valid on an intact, motionless structure, so the
crush margin is checkable without exceeding it — exactly like reading joint
safety factors under gravity alone (`testUtilisationIsReadableBeforeYield`). The
demo prints `peakUtil` and `crushSafetyFactor` on every run.

Target: **crush safety factor above ~5 under gravity**. The reference building
sits at 8.4. Below 1 and the structure comminutes standing still.

### What crushing will and will not do

- **Nothing accumulates below yield.** A structure standing under its own weight
  never grinds itself to dust however long it stands
  (`testSettledStructureNeverCrushes`).
- **Flow is quadratic in overstress**, so one material covers both "survives
  ordinary abuse" and "pulverizes under a real hit" with no second threshold.
  Measured on the reference building: ordinary impact → 0 chunks; 2× projectile
  mass → 4.7%; 8× → 6.3%; 16× → 10.9%.
- **No contact needed.** A chunk buried in a collapse, loaded only through its
  bonds, comminutes exactly as a struck one does (`testBondLoadedChunkCrushes`).
- **Tension never crushes.** A chunk in net tension cracks — which is the bond
  model's job — rather than turning to powder. This is also what keeps
  free-floating debris tumbling instead of crumbling (`testConfinementDiscriminates`).
- **Requires `graphReductionLevel = 0`.** Reduction merges chunks into aggregate
  solver nodes, so a per-chunk stress tensor would describe the aggregate.
  Creation fails rather than reporting a plausible wrong number.

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
| Chunks vanish where you wanted them to break apart | Crush too easy for the load | Read `peakUtil`. Above ~10 the chunks are far past yield: raise `crushEnergy` (toughness) before touching the yield surface. |
| Crushing never fires however hard you hit | Chunks are in tension, not compression | Tension never comminutes by design. If the region should be crushed, it needs confinement — check `getNodeStressInvariants`; a negative `p` means it is being torn off, not squeezed. |
| Crushing fires under gravity alone | Crush safety factor below 1 | Read `crushSafetyFactor` on a gravity-only run. Target above ~5; the reference building sits at 8.4. |
| Foundations turn to dust | A support node points at a crushable material | Give footings a material with no `crush` block. A vaporizing footing is never the story (`scene_pack_v3_test`). |
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

`blast/blast-stress-solver/assets/reference/reference-building.json` — 31
nodes, 62 bonds, 4 materials, 3 floors of a 2×2 bay frame. Small enough to read
end to end and re-measure in under a second.

Calibrated result (pinned by CTest `blast_stress_reference_building_load_path`):

| Joint class | Material | Safety factor |
|---|---|---|
| `column~infill` | facade-clip | 3.02 |
| `infill~slab` | drywall-panel | 3.23 |
| `column~column` | reinforced-concrete | 10.4 |
| `column~slab` | reinforced-concrete | 10.4 |
| `slab~slab` | concrete-slab | 14.2 |
| `column~foundation` | reinforced-concrete | 32.0 |

Destruction quality, measured through `ExtStressPhysXFrameStepper` with
**resim = 1** and unity contact gain — pinned by CTest
`blast_stress_destruction_quality`:

| Impact | Splits | Bodies | Largest piece | Standing | Reading |
|---|---|---|---|---|---|
| gravity only | 0 | 1 | 64/64 | 100% | stands |
| 500 kg @ 12 m/s | 0 | 1 | 64/64 | 100% | shrugs it off — **not glass** |
| 1.5 t @ 16 m/s | 9 | 34 | 31/64 | 48% | **partial: a hole, and half the building still up in one piece** |
| 4 t @ 20 m/s | 5 | 40 | 23/64 | 41% | still partial — ductility holds the frame |
| 40 t @ 45 m/s | 1 | 62 | 2/64 | 6% | everything breaks — **not rigid** |

### Ductility is the single biggest quality lever

The frame material ships at `fatal = elastic × 10` (`FRAME_BAND` in the
generator). Elastic limits are untouched by that dial, so **gravity safety
factors are bit-identical at any value** — it changes only *how* the frame
fails, never whether it holds itself up. Measured on this building, standing
fraction after one impact:

| fatal/elastic | 1500 kg | 2500 | 4000 | 8000 | 20000 |
|---|---|---|---|---|---|
| 1.05 (brittle) | 0.06 | 0.06 | 0.06 | 0.06 | 0.06 |
| 2.5 | 0.50 | 0.06 | 0.06 | 0.06 | 0.06 |
| **10 (shipped)** | 0.50 | 0.50 | 0.41 | 0.06 | 0.06 |

A brittle frame has **no partial state at any energy** — it is intact or it is
a rubble field, which is the shattered-glass failure mode however carefully
you tune strength. Widening the band buys a 4× energy range of partial
results, and it does *not* make the structure unbreakable: 20 t still flattens
every variant.

> **This result is about the FRAME. Do not generalize it to cladding.**
> Non-structural skin wants the opposite: a **brittle** band (≈1.2), so a
> struck panel lets go at once, at the impact site. Making the facade ductile
> too produced the worst failure measured on this project — cladding joints sat
> overstressed for ~6 s of simulated time silently draining their damage pool
> with nothing visibly moving, then the pools emptied and the released load
> cascaded in one burst (63% of impacts landed before anything broke; 41% of
> all damage in one 0.5 s window). Ductility belongs in the structure that must
> stay standing, brittleness in the layer meant to come off.
> `export-fractured-city.mjs` ships `FRAME_BAND=10` with `FACADE_BAND=1.2` for
> exactly this reason; see
> [`blast-destruction-diagnostics`](../blast-destruction-diagnostics/SKILL.md)
> § "Nothing happens, then it detonates".

Why it works, mechanically: damage per tick is
`health × Σ(stress − elastic) / (fatal − elastic)`. A wide band means an
overstressed bond loses a *fraction* of its health per tick, so load
redistributes between ticks instead of the whole graph severing at once. It is
the same stress path as everything else — no contact special-case exists.

There is a performance bonus. Less collapse means fewer awake bodies, and the
cost model pays for awake bodies, not chunks. At 2.5 t, the brittle build ends
at 55 bodies / 0.60 ms per frame; the ductile build ends at 33 bodies /
0.11 ms, fully settled.

"Largest piece" is the biggest connected body left. It is the number that
separates destruction from dust: at the interesting level 26 of 64 chunks are
still one object. Foundations never move at any energy — they are the only
world-fixed nodes.

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
