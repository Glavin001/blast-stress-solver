# Authored structures

Hand-designed buildings, emitted as v2 ScenePacks.

The exporters in `../scripts/` generate *cities* — procedural archetypes
replicated across a grid, all of them the same concrete. These are the other
case: individual buildings, designed rather than generated, made of several
materials that are meant to be told apart.

```bash
node structures/build.mjs                                     # all three
node structures/build.mjs algedra-tower                       # one
node structures/build.mjs --emit-vibe-land /path/to/vibe-land-2
node structures/verify.mjs path/to/pack.json                  # validate on its own
```

Packs land in `../../blast-stress-demo-rs/assets/scenes/`, and with
`--emit-vibe-land` also in `<vibe-land-2>/destruction/assets/scenes/`, where
they load via `VIBE_CITY_SCENE=<name>.json`. Generation is seeded: same source,
byte-identical output.

## What is here

| Structure | Chunks | What it is for |
|---|---|---|
| `algedra-tower` | 6.4k | Curved balcony bands; the organic-facade case |
| `house-1story` | 312 | Brick on a wood frame; load-bearing masonry |
| `house-2story` | 568 | Stone base over brick; two materials, one frame |
| `villa-savoye` | 540 | Pilotis. The clearest load path in the set — one column drops a corner |
| `park-432` | 6.6k | Slender tube. The only one that TOPPLES rather than pancaking |
| `parking-garage` | 694 | Flat plates, no walls. The only one that PANCAKES |
| `petronas` | 12.8k | Two towers COUPLED by a skybridge; damage travels between them |
| `neighbourhood` | 7.3k | Algedra and both houses in one scene |
| `skyline` | 27.9k | All seven, strung along the spawn axis |

Multi-storey structures have stairs between floors: an open dog-leg flight in
the houses and Villa Savoye, an enclosed core that also braces the frame in the
three towers.

## Looking at them

Rendering lives in vibe-land-2, which owns the triplanar shader, the Poly Haven
texture arrays and the sky/sun rig:

```bash
cd <vibe-land-2>/client && npm run dev
node tools/structure-shot.mjs algedra-tower       # -> docs/structures/<name>/*.png
```

## How it is put together

`lib/pack.mjs` is the builder. A **piece** is a convex polygon extruded along an
axis, which covers everything a building is made of: a box is a rectangle
extruded, a curved balcony segment is a wedge extruded. The format has no
rotation and no scale, so curvature has to be baked into hull points, and
extrusion does exactly that.

- `lib/materials.mjs` — strength (six stress limits, Pa) *and* appearance.
  Ductility (`fatal / elastic`) is the main behavioural dial: glass is banded at
  1.05 so a struck pane lets go at once, steel at 12 so it yields for many
  frames.
- `lib/fracture.mjs` — shard count from **material and surface area**. Glass is
  clamped to 2–4 whatever its size; brick courses into up to 24; steel gets the
  fewest of any solid because it is ductile and bends rather than shattering.
- `lib/contact.mjs` — bearing between two pieces, by separating-axis test on
  real face normals, with the patch measured as a shadow overlap. Bounding
  boxes cannot do this: a roof sheet's AABB runs from eave to ridge, so it reads
  as deeply overlapping the ridge beam it is resting on.
- `lib/gjk.mjs` — exact convex overlap, used by `verify.mjs`.
- `lib/elements.mjs` — walls with openings, roof slopes, facade bands, parapets.

## Rules worth knowing before editing one

**Never fake strength with bond area.** Area is simultaneously the denominator
of stress and the bond's damage pool, so inflating it corrupts the stress
readout *and* scales toughness super-linearly. Area is geometry; material is
strength; separate axes.

**Nothing may share space with anything else.** Running a post up *into* the
plate rather than up to its underside is not a stiffer joint — the contact
finder sees mutual containment, makes no bond at all, and the plate ends up
floating. Every piece stops at the underside of what it carries.

**A beam must be at least as wide as what lands on it.** A column overhanging
its beam puts its corners on the slab panels either side, and that is where the
load then goes — it cost this tower three separate rounds of 300%+ overstress.

**Glazing bears on its sill, with clearance at the head.** Fill the opening
exactly and the masonry above rests on the glass; the static walk duly routes a
wall's weight through a 25 cm² clip.

`verify.mjs` checks all of this and runs automatically inside `build.mjs`, so a
pack that fails is not written. It reports:

- format validity — the same rules `scene_pack.rs` enforces
- collider interpenetration, exactly, by GJK
- grounding — every piece has a path through the bond graph to a support
- rest statics — load flows to ground by hop distance, and no piece-to-piece
  interface exceeds its material's elastic limit standing still
- load path — the most heavily loaded ground element carries a real share of
  the weight, and removing it redistributes load or ungrounds something

The statics is a static load model, not a solver run. It proves a structure is
not overstressed at rest and that its load path exists. It does not predict
collapse dynamics — those appear only once PhysX has the pack.
