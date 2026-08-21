# ScenePack format

A ScenePack is a **runtime-independent description of a destructible structure**.
One JSON file is loaded, unchanged, by all three integrations:

| Runtime | Loader | Physics engine |
|---|---|---|
| TypeScript | `blast/blast-stress-solver/src/rapier/scenePackLoader.ts` | Rapier |
| Rust | `blast/blast-stress-demo-rs/src/scene_pack.rs` | Rapier |
| C++ | `demos/blast-stress-demo/scene_pack.cpp` | PhysX (CPU or GPU) |

That is the point of the format: the same building, three implementations, so
the API, the behavior, and the performance can be compared without the
structure being a variable. This document is the contract those three loaders
implement. If a loader disagrees with this file, the loader is wrong.

Related: [`PHYSICS_ENGINE_CONTRACT.md`](blast/blast-stress-solver-rs/PHYSICS_ENGINE_CONTRACT.md)
specifies what an *engine* must provide to host a destructible. This file
specifies what an *asset* must contain to be one.

## Units — normative

Every runtime must interpret these identically. Nothing else in the format
makes sense across runtimes if these drift.

| Quantity | Unit | Notes |
|---|---|---|
| Length, position, half-extents | metres | |
| Bond `area` | m² | The **real contact patch**. Never a strength knob — see below. |
| Node `mass` | kg | `0` means *support*: the node is pinned to the world. |
| Node `volume` | m³ | |
| Stress limits | pascals (N/m²) | |
| `gravity` | m/s² | Negative is down. |
| Projectile `speed` | m/s | |
| `ttlMs`, `debrisTtlMs` | milliseconds | |

**Derived quantity, also normative:**

```
utilisation = stress / (that bond's own material ELASTIC limit)
safety factor = 1 / utilisation
```

Utilisation is the number every runtime reports and every gate compares. It is
dimensionless, so "safety factor 3" means the same thing in Rapier and in
PhysX. A runtime that divides by a *global* limit instead of the bond's own
material limit is non-conformant — with heterogeneous materials there is no
correct global divisor.

## The one rule that matters

> **Area is geometry. Material is strength. They are separate axes.**

Bond `area` is simultaneously (a) the denominator of stress and (b) the bond's
damage pool — health is seeded from area and damage subtracts from it. So
inflating area to make a joint "stronger" silently corrupts the stress readout
*and* scales toughness super-linearly.

This is not hypothetical. An earlier revision of the high-rise pack authored
frame joints at 2×10⁵–10⁶× their geometric area. The frame ended up ~7 orders
of magnitude below its elastic limit: mathematically unbreakable under any
impulse the simulation could produce, while every destruction gate still
reported success, because "partial destruction" is satisfied by a structure
that cannot break. The compensations piled up from there — a 650 t projectile
at 32× the density of osmium, a 99× contact-force multiplier.

Author area from geometry. Author strength as a material. If a structure is
too weak or too strong, change the material.

## Version 1

```jsonc
{
  "version": 1,
  "key": "high-rise-10f-local",          // optional, stable id
  "title": "High-Rise Apartment 10F",
  "defaults": {
    "camera":   { "target": {"x":0,"y":12,"z":0}, "distance": 52 },
    "projectile": { "radius": 0.6, "mass": 2500, "speed": 18, "ttlMs": 8000 },
    "solver": {
      "gravity": -9.81,
      "materialScale": 1e10,              // TS-only legacy scale; ignored when `limits` present
      "limits": {                         // OPTIONAL in v1 — see the trap below
        "compressionElastic": 12e6, "compressionFatal": 30e6,
        "tensionElastic":    1.2e6, "tensionFatal":     3e6,
        "shearElastic":      1.6e6, "shearFatal":       4e6
      }
    },
    "physics": {
      "friction": 0.25, "restitution": 0.0,
      "contactForceScale": 1,             // see below — 1 is the correct value
      "debrisCollisionMode": "all", "skipSingleBodies": false
    },
    "damage":       { /* optional per-chunk health layer, TS/Rapier only */ },
    "optimization": { /* optional debris TTL / damping hints */ }
  },
  "scenario": {
    "nodes":         [ { "centroid": {"x":0,"y":0.32,"z":0}, "mass": 0, "volume": 12.6 } ],
    "nodeSizes":     [ { "x": 4.5, "y": 0.6, "z": 4.67 } ],
    "nodeColliders": [ { "kind": "cuboid", "halfExtents": {"x":2.25,"y":0.3,"z":2.33} } ],
    "nodeTypes":     [ "foundation" ],    // optional; parallel to `nodes`
    "bonds": [
      { "node0": 12, "node1": 468,
        "centroid": {"x":-8.75,"y":1.82,"z":-6.24},
        "normal":   {"x":0,"y":-0.5,"z":0.87},
        "area": 0.52461 }
    ]
  },
  "nodeMeshes": null                       // omitted for all-box scenes
}
```

`nodes`, `nodeSizes`, `nodeColliders` and (when present) `nodeTypes` are
**parallel arrays** — same length, same index. Bonds reference node indices.

### v1 trap: `solver.limits` is optional

A pack that omits it inherits whatever placeholder the runtime happens to
default to. In C++ that is 1 MPa elastic / 2 MPa fatal — about a twelfth of
concrete's compressive strength, so the structure is far weaker than the
material it claims to be, and the difference has to be absorbed somewhere else
for the scene to behave. The C++ loader now warns on stderr naming the pack and
the values. **v2 makes materials mandatory.**

### `contactForceScale`

Both Rapier and PhysX multiply a contact **force** by this (PhysX converts its
solved impulse to a force by dividing by `dt` first). The physically correct
value is therefore **1**. Anything above it is gain that must be cancelled
elsewhere — weaker limits, heavier projectiles — and it destroys the
correspondence between a run and its own material model. Treat a value ≠ 1 as a
sensitivity experiment, never as tuning.

## Version 2 — heterogeneous materials

v2 adds one thing: a structure can be made of **more than one material**.

```jsonc
{
  "version": 2,
  "defaults": {
    "solver": {
      "gravity": -9.81,
      "materials": [                       // REQUIRED in v2, >= 1 entry
        { "name": "reinforced-concrete",
          "compressionElastic": 24e6, "compressionFatal": 60e6,
          "tensionElastic":      3e6, "tensionFatal":      8e6,
          "shearElastic":        4e6, "shearFatal":       10e6 },
        { "name": "drywall-track",
          "compressionElastic":  1e6, "compressionFatal":  2e6,
          "tensionElastic":    0.1e6, "tensionFatal":    0.3e6,
          "shearElastic":     0.15e6, "shearFatal":      0.4e6 }
      ]
    }
  },
  "scenario": {
    "bonds": [
      { "node0": 12, "node1": 468, "centroid": {...}, "normal": {...},
        "area": 0.52461,
        "m": 1 }                           // index into defaults.solver.materials
    ]
  }
}
```

### Material semantics

- `name` is **author-defined, free-form**, for reports and debugging. It is not
  an enum and the library ships no material library. There are infinitely many
  real materials and every project's scale differs; what the library provides
  is the *method* for finding values, not the values. Tested examples with
  measured behavior live in the material test matrix — see
  `demos/blast-stress-demo/tests/material_behavior_test.cpp` and
  [the authoring guide](.cursor/skills/blast-structure-authoring/SKILL.md).
- Negative `tension*` / `shear*` limits **inherit** the corresponding
  compression limit. This is resolved once when the table is installed.
- `compressionFatal >= compressionElastic >= 0` is required; violations are a
  hard error.
- **Ductility is the width of the `fatal - elastic` band.** A wide band takes
  partial damage over many frames (reinforced concrete yielding); a narrow band
  snaps in one (glass, a taped drywall seam). This is the main behavioral dial
  and it is independent of raw strength.

### Per-bond assignment

- `m` is the material index. **Omitted means 0**, so a single-material v2 pack
  needs no per-bond field at all.
- An index outside the table is a **hard error**, not a clamp. Silently
  clamping to material 0 would turn an authoring typo into a mysteriously
  strong joint.

### Rules for v2 loaders

| Rule | Behavior |
|---|---|
| `version: 2` without `defaults.solver.materials` | hard error |
| `materials` empty | hard error |
| `bonds[].m` out of range | hard error naming the bond index |
| `compressionFatal < compressionElastic` | hard error naming the material |
| `bonds[].m` absent | material 0 |
| `defaults.solver.limits` present in a v2 pack | ignored; `materials` wins |

### v1 compatibility

v1 packs keep loading everywhere. A loader synthesizes a **one-entry material
table** from `solver.limits` (or from its placeholder default, with the
warning) and puts every bond on index 0. Behavior is identical to before v2
existed, so the Rapier demos and existing assets are unaffected.

## Version 3 — chunk crushing

v2 decides when a **joint** fails. v3 adds the other half: when a **chunk
itself** is comminuted and leaves the simulation as dust.

Both happen in the same impact. A car into a concrete wall separates most of it
along its joints, and grinds up the small region under the hit. Without v3 the
second half is impossible to express: every chunk is indestructible, so a hard
enough hit just pushes intact rigid bodies around.

v3 adds exactly two optional things.

```jsonc
{
  "version": 3,
  "defaults": { "solver": { "materials": [
    { "name": "reinforced-concrete",
      "compressionElastic": 24e6, "compressionFatal": 240e6,
      "tensionElastic": 3e6,  "tensionFatal": 30e6,
      "shearElastic": 4e6,    "shearFatal": 40e6,
      "crush": {                       // OPTIONAL. Absent = this material's
        "capPressure":   60e6,         //   chunks never comminute.
        "cohesion":      14.4e6,
        "frictionSlope": 1.2,
        "crushEnergy":   1.2e8,
        "crushViscosity": 2e5
      } }
  ] } },
  "scenario": {
    "nodes": [
      { "centroid": {...}, "mass": 1800, "volume": 0.75, "m": 0 }  // OPTIONAL `m`
    ]
  }
}
```

**`nodes[].m`** — a node's own material index, exactly like `bonds[].m`. It
selects the chunk's crush properties and nothing else. Absent means 0. Out of
range is a hard error, never a clamp — same rule as bonds, and for the same
reason: a silent clamp turns a typo into a mysteriously indestructible chunk.

A chunk's material is deliberately independent of the materials on the bonds
around it. How hard a column is to grind up is not the same question as how
strong its connections are, and a real building has facade panels that pulverize
hanging off clips that are stronger than the panel.

### The crush model

Each solve builds a per-chunk Cauchy stress tensor by the **Love–Weber (virial)
sum** over the forces acting on the chunk — its bond forces plus any external
contacts — and reduces it to two invariants:

```
p = -trace(sigma)/3        confining pressure, positive in compression
q = sqrt(1.5 * (s:s))      von Mises equivalent, s = sigma + p*I
```

Yield is a **Drucker–Prager cone with a pressure cap**, with tension excluded:

```
crushing requires p > 0
excess = max( q - (cohesion + frictionSlope*p),  p - capPressure )
```

Flow is **Perzyna overstress viscoplasticity**, and damage is the plastic work
per unit volume normalized by the specific comminution energy:

```
D += excess^2 * dt / (crushViscosity * crushEnergy),    pulverized at D >= 1
```

Three properties of that law are what make it behave:

- **Quadratic in overstress.** A chunk barely past yield takes a very long time
  to comminute; a chunk hit hard enough to sit far outside the surface goes
  almost at once. So one material covers "survives ordinary abuse" and
  "pulverizes under a real hit" without a second threshold to author.
- **Needs no strain measurement.** A chunk loaded purely through its bonds —
  buried in a collapse, never touched — comminutes exactly as a struck one does.
- **Nothing accumulates below yield.** A structure standing under its own weight
  has `excess = 0` and never grinds itself to dust, however long it stands.

Tension is excluded because comminution is a compressive phenomenon: a chunk in
net tension fails by cracking, which is what the bond model already represents.
That exclusion is also what keeps free-floating debris intact — it carries no
confining pressure, so it tumbles rather than crumbling.

### Fields and units

| Field | Unit | Meaning |
|---|---|---|
| `capPressure` | Pa | Hydrostatic cap: confined pore collapse. **Required**; must be > 0. |
| `cohesion` | Pa | Drucker–Prager deviatoric intercept at `p = 0`. Default 0. |
| `frictionSlope` | — | `dq/dp` of the cone. Default 0. ~1.2 is concrete-like. |
| `crushEnergy` | J/m³ | Specific comminution energy. **Required**; must be > 0. |
| `crushViscosity` | Pa·s | Perzyna flow viscosity. **Required**; must be > 0. |
| `strainRateExponent` | — | CEB dynamic-increase-factor exponent. Default 0 = no rate hardening. |
| `referenceStrainRate` | 1/s | Strain rate at which the DIF is 1. Default 1. |
| `debrisMassFraction` | — | [0,1] of the chunk's mass respawned as rigid fragments. Default 0. |
| `debrisFragmentCount` | — | Fragments to spawn when `debrisMassFraction > 0`. Default 0. |

Deriving the cone rather than dialling it: under uniaxial compression at a
material's own compressive strength `fc` the stress state is `p = fc/3, q = fc`,
so setting

```
cohesion = fc * (1 - frictionSlope/3)
```

makes a chunk yield at exactly its authored compressive strength when squeezed
unconfined, and demand progressively more under confinement. That is one fewer
number to invent — the cone is pinned to the same `fc` the bonds already use.
`blast/blast-stress-solver/scripts/export-reference-building.mjs` does this.

### The energy bill

`crushEnergy` is charged, not just referenced: the PhysX adapter (setting
`applyCrushResistance`, default on) extracts each damage increment's
`dD * crushEnergy * volume` from the impacting body's kinetic energy as a
resistive impulse, levied on the resimulation pass -- the pass where the
removed material would otherwise refund its stopping impulse and hand the
penetrator a free hole. The same authored number therefore sets both how fast
a material comminutes and how much stopping power comminuting it has, which is
what the physical quantity means. Bond-borne crushes deep inside a structure
charge nobody: their load path is the structure itself and the reaction is
already inside the solve.

### Authoring against it

Crushing has the same shape of measurement as joint strength:

```
crush utilisation = max( q / (cohesion + frictionSlope*p),  p / capPressure )
crush safety factor = 1 / crush utilisation
```

1 means the chunk is exactly at yield. Read it with
`ExtStressPhysXDestructible::getNodeCrushUtilisation` — it is valid on an intact,
motionless structure, so the crush margin can be checked without exceeding it,
exactly as `getBondUtilisations` gives joint margins under gravity alone.

Measured on the reference building
(`assets/reference/reference-building-crush.json`):

| Load | Peak crush utilisation | Chunks comminuted |
|---|---|---|
| Gravity only | 0.12 (safety factor 8.4) | 0 |
| Ordinary impact | 3.3 | 0 |
| 2× projectile mass | 4.7 | 3 of 64 (4.7%) |
| 8× projectile mass | 6.2 | 4 of 64 (6.3%) |
| 16× projectile mass | 8.0 | 7 of 64 (10.9%) |

That progression is the target behaviour: the structure comes apart at its
joints under ordinary loads and only begins losing mass when hit hard, with the
loss staying localized rather than consuming the building.

### Compatibility

v1 and v2 packs load unchanged and behave exactly as before — crushing is
inert unless a material authors a `crush` block. A `crush` block or a `nodes[].m`
in a pack declaring version 1 or 2 is a hard error rather than a silent no-op:
the author is expecting behaviour the declared version does not have.

Verified: the crush-enabled reference building run with crushing disabled
produces physics identical to the v2 reference building.

## Conformance

`blast/blast-stress-solver/assets/conformance/structure-conformance-v2.json` is
the canonical fixture. Each runtime has a test that loads it, computes a
**digest** — node count, bond count, material count, total mass, summed bond
area, per-material bond counts, per-joint-class bond counts — and asserts it
matches `structure-conformance-v2.digest.json`.

That is what makes "the same structure in three runtimes" checkable rather than
aspirational: if any loader drifts (a field silently ignored, an index off by
one, a unit misread), that runtime's own suite fails. The digest is
intentionally structural, not dynamic — it pins *interpretation of the asset*,
not simulation results, which legitimately differ between Rapier and PhysX.

## What each runtime honors

Not every field is meaningful everywhere; a loader may ignore a field it has no
concept of, but must not misinterpret one.

| Field | TS/Rapier | Rust/Rapier | C++/PhysX |
|---|---|---|---|
| `scenario.*`, `bonds[].area`, `bonds[].m` | yes | yes | yes |
| `solver.materials`, `solver.gravity` | yes | yes | yes |
| `nodeTypes` | yes | yes | yes (joint-class reports) |
| `physics.friction/restitution/contactForceScale` | yes | yes | yes |
| `solver.materialScale` | legacy scale | ignored | ignored |
| `damage` | yes (opt-in layer) | ignored | ignored |
| `optimization` | yes | yes | partial |
| `camera`, `projectile` | demo-only | demo-only | demo-only |
| `nodeMeshes` | render only | render only | ignored (boxes/hulls from colliders) |
| `nodes[].m`, `materials[].crush` (v3) | ignored | ignored | yes |
