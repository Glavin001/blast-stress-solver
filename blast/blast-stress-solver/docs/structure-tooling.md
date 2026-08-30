# Tooling for developing structures

Everything here exists because of a specific way a previous session wasted
time. The reasons are recorded next to each tool, because a tool whose reason
is forgotten is the first one someone removes.

## The loop

```bash
cd blast/blast-stress-solver/structures

# 1. build one structure, with the gate
node build.mjs parking-garage --emit-vibe-land /root/workspace/vibe-land-2

# 2. audit it — one line of JSON, the same verdict the gate uses
cd /root/workspace/vibe-land-2
./target/release/structure-audit parking-garage | jq

# or do both, over a range of values, in one command
cd blast/blast-stress-solver/structures
node sweep.mjs parking-garage deckBeamDepth 1.1,1.3,1.5,1.7
```

## `sweep.mjs` — vary one thing, see the shape

```
  setting             bonds  broke  settled   peak   sag m   over  verdict
  deckBeamDepth=1.1    9143     17      9 s   2.84    6.02     90  first: tension slab<->slab y=8.1
  deckBeamDepth=1.3    9069     24     13 s   2.82    9.12    101  first: tension slab<->slab y=16.2
  deckBeamDepth=1.5    9194      0      2 s   2.96       -    111  stands
  deckBeamDepth=1.7    9207    174    never   2.94    9.38    188  first: compression slab<->slab y=0.2
```

That table took four minutes and one command. The same four data points
previously took most of a session, one edit-build-test cycle at a time, and the
U-shape in the middle of it — better up to 1.5 m, worse past it — was invisible
while looking at one number at a time. Four of six attempts to strengthen that
garage made it worse, and this is the shape that explains why.

Sweeps two parameters together (not crossed) with
`deckBeamDepth=1.1,1.5 levelHeight=3.7,4.1`. Restores the defaults when it is
done, unless `--keep`, because a sweep that leaves the last value built is a
stale pack waiting to happen.

Built on `build.mjs --set key=value`, which works for every structure without
per-structure support: they all already take a `cfg` object and merge it over
their defaults.

## `structure-audit` — the audit, as data

`cargo run -p vibe-land-destruction --features cuda-stress --release --bin
structure-audit -- <name> [max-secs]`

The stability suite answers "does this pass", which is the right question once
and the wrong one twelve times running. This prints the numbers themselves as
JSON so something else can tabulate them. Parsing them back out of a panic
message is how a harness starts lying to you.

Exits non-zero only if the pack will not load. A structure that falls over is a
result, not an error.

## Break provenance — *which* bonds broke

The failure card used to say "sheds 3 bonds". Which three was not recorded, and
guessing between a stair and a set of beams cost several ten-minute cycles.

```
minas-tirith: never settles: 126 bonds broken, most recently at 6 s
  peak 2.82 -> 2.02, sag 11.53 m (beam)
  #151291 BROKE at 1 s, tension wall<->beam at y=5m over 0.015 m2, last seen at 2.30x
  #149249 BROKE at 1 s, shear wall<->parapet at y=83m over 0.026 m2, last seen at 2.55x
```

Note the areas: 0.015 and 0.026 m². The tool says "sliver bonds" at a glance,
which is a different diagnosis from anything the overload list suggests.

The joints that break and the joints that are most *overloaded* are routinely
different populations — a bond can sit at 2.4x all run and never go, while the
one that goes was fine until the load shifted onto it. Both are printed now.

Recovered without new FFI: a broken bond is not removed from the stress rows, it
stays and reads exactly zero, so this watches for a bond that was carrying
something and now carries nothing, on a sample where the broken count actually
rose. Heuristic — it can name an unloaded neighbour alongside the real victim —
but the difference between "3 bonds broke" and "3 bonds broke, tension,
stair-to-stair, at y=4 m" is the difference between a guess and a place to look.

## Sag — the criterion that was never measured

`peak_sag` and `sag_role`: furthest any chunk moved *down* from where it was
placed, and what kind of member it was.

Stress and deflection are different failure criteria and only the first was ever
reported. A 1.1 m prestressed garage beam satisfies its bending check with
margin and still cracks the deck above it, because it sags — 116 broken bonds
against 3, from a member that passes on paper. Serviceability usually governs a
long span, and it was invisible.

In the sweep table above, the `sag m` column explains the two failing rows on
its own: 6 m and 9 m of drop.

## Freshness — the stale-pack trap, closed

Packs are authored in JavaScript and emitted as JSON. Nothing in the Rust build
knows about that step, so when `node build.mjs` fails, the previous pack stays
where it was and every test downstream passes against it.

That is not hypothetical: a material was added to the garage without registering
it in the fracture tables, the build threw, and the stability suite reported a
clean run on a structure that did not contain the change. It was caught by
noticing the build had printed nothing — by luck.

`assert_pack_fresh` compares mtimes: if a source that can affect this pack is
newer than the pack, the pack cannot contain it. mtime rather than a content
hash because the failure mode is exactly "the pack was not rewritten".

Scoped per structure — `lib/`, `build.mjs` and `verify.mjs` always count, plus
the structure's own module — so editing `petronas.mjs` does not declare
`villa-savoye.json` stale. A check that cries wolf gets ignored.

Silent when it cannot see the sources, so a checkout without
`blast-stress-solver` beside it still runs. `BLAST_STRUCTURES_DIR` overrides.

## Self-weight share — strength-limited, or mass-limited?

Printed by the build gate:

```
  self-weight share of what each role delivers: stair 48% (7 t), ramp 25%,
    beam 25% (2772 t), slab 23% (5452 t), parapet 10% (190 t)
```

The single most expensive number nobody had. The garage's 16 m beams were
deepened from 1.5 m to 1.7 m to buy bending capacity and broken bonds went from
9 to 2,235, because a beam at that span carries mostly *itself*, so section adds
weight as fast as it adds strength. Four separate attempts to fix that structure
by adding material failed the same way.

Measured as what a role *delivers* downward, not what the load walk accumulates
through it: `carried` is cumulative and grows with chunk count, so summing it
over a role dilutes the ratio into meaninglessness.

A warning sign, not a law. The higher it is, the less extra depth returns; past
roughly a third, prefer changing the span, the load, or the material.

## `lib/design.mjs` — the checks, as functions

`slabThicknessFor`, `beamDepthFor`, `secondarySpacingFor`,
`beamSelfWeightShare`, `checkFramedDeck`.

These existed as throwaway `node -e` one-liners typed fresh each time a member
changed, and afterwards only in commit messages. Now a structure verifies itself
at build time with the same formulas that produced it:

```
Error: parking-garage: the framing does not check out
  - beam: 44.75 MPa over 16 m at 0.9x0.6 m exceeds 13.44 MPa; needs 1.19 m
```

Failing at the source with the required value beats twenty minutes of simulation
resolving into a count of broken bonds. `GARAGE_DESIGN=1` prints the margins on
a passing build.

Two things the formulas will not tell you, both learned the hard way and both
documented in the module:

- **There is a thickness floor that has nothing to do with bending.** A 250 mm
  garage deck passes the flexural check with two decimal places to spare and
  still made things worse than 300 mm, because thin slabs make thin bond seams
  and the solver's section-modulus term goes as `6/sqrt(area)`.
- **Strength is not the only check.** `beamDepthFor` returns 1.19 m for the
  garage's mains. The authored value is 1.5 m, and the difference is stiffness.

## What is still missing

- **No cross-structure regression table.** `structure-audit` over all seven,
  diffed against a committed baseline, would catch a shared-library change
  quietly degrading a building nobody was working on.
- **Sweeps are serial.** Each row is a build plus a simulation. They are
  independent and could run in parallel, at the cost of GPU contention — which
  on this box has historically been fatal rather than slow.
- **`--set` only reaches top-level config keys.** Nested authoring choices
  (fracture grading, per-piece materials) are still source edits.
