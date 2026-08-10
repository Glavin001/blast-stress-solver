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
  infill~slab        drywall-panel        bonds=12  peakUtil=0.2942  safetyFactor=3.399  peak(c/t/s)=5.5e+04/1.47e+04/4.41e+04 Pa
  column~infill      facade-clip          bonds=24  peakUtil=0.2837  safetyFactor=3.524  ...
  column~slab        reinforced-concrete  bonds=20  peakUtil=0.05837 safetyFactor=17.13  ...
  column~foundation  reinforced-concrete  bonds=4   peakUtil=0.02838 safetyFactor=35.24  ...
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

The worked example did exactly this: the first pass put the facade clip at
safety factor 1.06 — the building stood, but its cladding was one gust from
letting go. The binding mode was shear (5.67×10⁴ Pa against a 0.20×10⁶ limit),
so only the clip's shear limits moved. Second pass: 3.52. See the comment block
in `export-reference-building.mjs`.

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

`blast/blast-stress-solver/assets/reference/reference-building.json` — 31
nodes, 62 bonds, 4 materials, 3 floors of a 2×2 bay frame. Small enough to read
end to end and re-measure in under a second.

Calibrated result (pinned by CTest `blast_stress_reference_building_load_path`):

| Joint class | Material | Safety factor |
|---|---|---|
| `infill~slab` | drywall-panel | 3.40 |
| `column~infill` | facade-clip | 3.52 |
| `column~slab` | reinforced-concrete | 17.1 |
| `slab~slab` | concrete-slab | 32.4 |
| `column~foundation` | reinforced-concrete | 35.2 |

Impact response, measured (`--projectile-waves N`, default 3000 kg ball):

| Impulse (N·s) | Cladding moved | Frame moved | Reading |
|---|---|---|---|
| 6.6×10⁴ | 2/12 | 0/12 | local cladding damage |
| 8.6×10⁴ | 4/12 | 0/12 | cladding sheds, frame intact |
| 1.7×10⁵ | 11/12 | 11/12 | frame comes down |
| 2.9×10⁵ | 12/12 | 12/12 | total collapse |

Cladding fails first and the frame only at ~2.5× that energy — the hierarchy
the materials encode, showing up in behavior. Foundations never move at any
energy: they are the only world-fixed nodes.
