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
