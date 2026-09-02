---
name: author-structures
description: Author and size a destructible building — the build/audit/sweep loop, the design checks, and the six traps that have each silently produced a wrong result. Use when adding or changing a structure, adding a material, sizing a member, or when a structure stands but should not (or falls but should not).
---

# Authoring a structure

Structures are authored in JavaScript under
`blast/blast-stress-solver/structures/` and emitted as JSON scene packs that
vibe-land-2 loads. The authoring side owns geometry, materials and fracture; the
consuming side owns rendering and simulation.

## The loop

```bash
cd blast/blast-stress-solver/structures

# One structure, with the build gate
node build.mjs parking-garage --emit-vibe-land /root/workspace/vibe-land-2

# What it does under load — one line of JSON, the same verdict the gate uses
cd /root/workspace/vibe-land-2
./target/release/structure-audit parking-garage | jq
```

**Do not tune by repeating that loop by hand.** Sweep instead:

```bash
node sweep.mjs parking-garage deckBeamDepth 1.1,1.3,1.5,1.7
```

```
  setting             bonds  broke  settled   peak   sag m   verdict
  deckBeamDepth=1.1    9143     17      9 s   2.84    6.02   first: tension slab<->slab
  deckBeamDepth=1.3    9069     24     13 s   2.82    9.12   first: tension slab<->slab
  deckBeamDepth=1.5    9194      0      2 s   2.96       -   stands
  deckBeamDepth=1.7    9207    174    never   2.94    9.38   first: compression slab<->slab
```

Four minutes, one command. Those four points once took most of a session at one
edit-build-test cycle each, and the U-shape in them — better to 1.5 m, worse past
it — is **invisible one number at a time**. Assume any member you are sizing has
a non-monotonic response until a sweep says otherwise; in this project they
usually do.

Sweeps two parameters together (not crossed) with
`deckBeamDepth=1.1,1.5 levelHeight=3.7,4.1`, and restores defaults afterwards
unless `--keep`.

`build.mjs --set key=value` is what makes this work, and it needs no
per-structure support: every builder takes a `cfg` object and merges it over its
defaults. It only reaches **top-level config keys** — fracture grading and
per-piece materials are still source edits.

## Check the design before simulating it

`lib/design.mjs`: `slabThicknessFor`, `beamDepthFor`, `secondarySpacingFor`,
`beamSelfWeightShare`, `checkFramedDeck`. Structures call `checkFramedDeck` at
build time so a bad section fails at the source with the required value:

```
Error: parking-garage: the framing does not check out
  - beam: 44.75 MPa over 16 m at 0.9x0.6 m exceeds 13.44 MPa; needs 1.19 m
```

Twenty seconds instead of twenty minutes. Wire it into any new framed structure.

**Two things the formulas will not tell you**, both learned expensively:

- **There is a thickness floor unrelated to bending.** A 250 mm garage deck
  passes its flexural check with two decimal places to spare and was *worse*
  than 300 mm — 9 broken bonds became 165 — because thin slabs make thin bond
  seams and the solver's bending term goes as `6/sqrt(area)`. Treat
  `slabThicknessFor` as a lower bound, not an answer.
- **Strength is not the only check.** `beamDepthFor` returns 1.19 m for the
  garage mains; the authored value is 1.5 m and the difference is *stiffness*. A
  member can pass every stress check and fail on deflection.

## Push work into the build gate

`node build.mjs <name>` runs interpenetration (GJK), ungrounded chunks,
monoliths, the load walk, strength spread and stair arrival — in about 0.3 s. A
simulation costs minutes. **Anything checkable cheaply belongs in the gate.**

The gate also prints the diagnostic that decides how to fix an overstressed
member:

```
self-weight share of what each role delivers: stair 48%, ramp 25%,
  beam 25% (2772 t), slab 23% (5452 t), parapet 10%
```

Above roughly 30%, added section has diminishing returns — the member is
carrying mostly itself, so depth adds weight as fast as it adds capacity.
Change the span, the load, or the material instead. Ignoring this is how the
garage's mains went from 9 broken bonds to 2,235 by being made *stronger*.

## The traps

**1. A failed build leaves the old pack in place.** Nothing in the Rust build
knows about `node build.mjs`, so when it throws, every downstream test passes
happily against the previous structure. `assert_pack_fresh` now panics on this,
but only where it can see the sources. **Always check the build printed `=> ok`
and `wrote …` before trusting any number.** Chain with `&&`, never in separate
commands.

**2. Editing anything in `lib/` invalidates every pack.** The freshness check is
scoped per structure — `lib/`, `build.mjs` and `verify.mjs` always count, plus
the structure's own module — so a shared-library change means a full
`node build.mjs --emit-vibe-land …`, not a single-structure one.

**3. `cellVolume` works backwards from the obvious.** It is a pre-Voronoi *cell*
size that gets cut into up to three shards, so the default for reinforced
concrete is `3.5 × 3 = 10.5` m³. Passing `4.0` makes chunks **finer**, not
coarser. Chunk size is capped by the material's `maxChunkVolume` regardless.

**4. A new material must be registered in three places** or the build throws
mid-fracture: `lib/materials.mjs` (the `SPEC` row, plus `DENSITY`, `MODULUS`,
`RESIDUAL`, `LOOK`), and **both** tables in `lib/fracture.mjs`
(`maxChunkVolume` and the fracture rule). The first two are obvious; the
`fracture.mjs` pair is what actually bit.

**5. A validator that cannot fail is worse than none.** The first version of
`checkFramedDeck` silently passed a 600 mm beam at 45 MPa, because a misspelt
material field returned `undefined` and every comparison against `NaN` is false.
Assert your inputs exist. Test a validator by feeding it something that *should*
fail.

**6. Pack keys are underscored; scene files are hyphenated.** `parking_garage`
vs `parking-garage.json`. Any list keyed by one and matched against the other
needs a count guard, or it silently produces an empty result.

## Adding a structure to the set

- Give it a `_is_stable` test in `destruction/tests/structural_stability.rs`.
  **Check this.** `algedra-tower` existed for a long time with no gate and did
  not stand; a loop over `structure-audit` found it in one pass.
- If it should appear in the walkable scene, add it to `SKYLINE` in
  `neighbourhood.mjs`, and to `STANDS` **only if the audit says it stands**.
  That membership is a measurement, not an opinion.
- Structures verified alone can still fail **together**. The five-building
  `skyline-stable` sheds 943 bonds merged and zero apiece. Use
  `--set exclude=key1,key2` to bisect which.

## Related

`diagnose-structure-failure` for reading an audit card, `city-stack-run` for
serving a scene, `city-physics-tuning` before changing any material constant.
Full write-up: `blast/blast-stress-solver/docs/structure-tooling.md`.
