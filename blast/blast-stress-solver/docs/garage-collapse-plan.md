# How parking garages actually fail, and what we can model

Written after the garage was made to collapse, to check we are making it
collapse for the RIGHT reasons rather than the first ones that worked.

## What kills parking garages, generally

Not one story. The recurring mechanisms, roughly in order of how often they
appear in collapse investigations:

**1. Punching shear at the slab-column connection.** The signature failure of a
flat plate. The column head pushes through the deck like a die through sheet
metal. It is brittle -- no sag, no cracking anyone notices, no warning -- and it
is why flat plates are the least forgiving garage structure. Pipers Row (1997)
is the famous one, but Harbour Cay Condominium (1981) and 2000 Commonwealth Ave
(1971) are the same mechanism.

**2. Progressive collapse by impact.** The real killer. One bay punches, that
deck falls onto the deck below, and the deck below was designed for parked cars,
not for a falling floor. Each level adds mass, so the failure accelerates
downward. Most of the death toll in garage collapses is this, not the initial
failure.

**3. Deterioration at the connection.** De-icing salt drives chloride into the
slab, rebar corrodes, section is lost precisely where punching capacity lives.
This is what made Pipers Row's connection weak enough to punch under ordinary
load. Slow, invisible, and not something worth modelling for a game.

**4. Overload.** Cars are heavier than they were, occupancy patterns change,
and snow or construction plant lands on a roof deck designed for neither.

**5. Column loss.** Vehicle impact, fire, blast. This is the one the player
causes, and structurally it is the least common in reality.

## What engineers design against, and what those defences look like

- **Brittleness.** Punching gives no warning, so codes push toward ductility:
  shear studs, stirrups, drop panels, capitals. All of them buy warning as much
  as strength.
- **Disproportionate collapse.** A local failure must not take the building.
  Alternate load paths, ties, and continuity.
- **Free fall after a punch.** This is the important one for us. Modern codes
  require INTEGRITY REINFORCEMENT: bottom bars continuous through the column
  cage. After a slab punches, those bars catch it. The deck does not drop free
  -- it hangs from the column in a catenary sag, still attached.

That last point is worth dwelling on, because it is what a collapsed garage
actually looks like: columns standing, their heads poking through the deck, and
the slab draped between them at an angle. Not a pile of rubble on the ground. A
failure that leaves a hanging, sagging deck reads as far more real than one that
drops everything flat.

## What we can and cannot model

**Cannot: fine-grained punching.** A punch is a shear cone through the slab's
thickness on a perimeter about `d/2` from the column face. Our slab is
pre-fractured into chunks that are already larger than that cone, and a bond
either holds or does not. There is no mechanism for a plug to shear out of the
middle of a chunk.

**Can: make the chunk BE the plug.** If the slab chunks immediately around each
column are sized to roughly the punching perimeter, then when those bonds fail
the geometry that drops looks exactly like a punched cone. We are not
simulating the shear surface, we are pre-cutting it -- which is the same trick
the whole fracture model already relies on.

**Can: model what happens after.** Integrity reinforcement is a residual: the
connection loses most of its capacity but retains a tie. That is exactly what
`residualAreaFraction` does, and it is already plumbed. A high residual on
slab-to-column bonds gives the hanging, sagging deck rather than a free drop.

**Can: model progressive collapse.** Debris impact on the deck below already
feeds the stress solver through real PhysX contacts. It should already
pancake; it has never been measured.

## Proposals, ranked by realism gained per unit of work

### 1. Cars as discrete masses (the suggestion, and it is a good one)

~1,500 kg cuboids in parking spots, 30-50% occupancy. Better than the smeared
`liveLoadPa` in three ways:

- **Spatially non-uniform**, which is what real live load is and what makes one
  bay fail before its neighbour. Smeared load makes every bay identical, which
  is why the current garage fails so symmetrically.
- **They fall with the deck**, which is most of the visual payload of a garage
  collapse.
- **They become the impact load** that drives progressive collapse onto the
  deck below -- mechanism 2, which is currently untested.

Cost: 46 x 32 m at ~2.5 x 5 m per spot is roughly 100 spots per level; at 40%
that is ~40 cars per level, 200 for the building. That is a real body count and
needs measuring against the tick budget before committing.

Open question: rest them on the deck (contacts carry the load, most realistic,
and they slide and tumble) or bond them (cheaper, but they become structure).
Resting is the right answer if the contact path carries load correctly, and
`free_island_load_test` says it does.

### 2. Fix the column grid to respect parking geometry

The current grid is 7.6 x 8.0 m, which is square and puts a column in the drive
aisle. Real garages span the aisle: a bay is two rows plus an aisle, 5.0 + 6.5 +
5.0 = 16.5 m, with columns every ~7.5 m along it (three spots between columns).

The building depth is 32 m, which is almost exactly two 16.5 m modules -- so the
plan was designed for this and the grid was not. Changing the Z grid from 8.0 to
16.0 m gives a real garage layout AND doubles the span the deck must carry,
which raises the stress and makes the structure genuinely marginal in the way a
real flat plate is.

This is the change most likely to make failure interesting, because it puts the
building where real ones sit: working hard, not at 6% of capacity.

### 3. Punching-sized chunks at the column head

Grade the deck's fracture so the chunks around each column are about
`column + d` across (roughly 0.8-1.0 m for this slab), and coarser out in the
field. Same chunk budget, detail where the failure happens, and the piece that
drops is shaped like a punching cone.

Uses the existing per-piece `cellVolume`, so it is authoring rather than a
format change.

### 4. Integrity reinforcement at slab-to-column

A high `residualAreaFraction` on the slab-to-column joint specifically, so a
punched connection keeps a tie and the deck hangs rather than dropping free.
Needs per-location material, which is authorable today via `bondMaterialName`.

Expected result: the sagging, still-attached deck of the photographs, instead
of a flat drop.

### 5. Measure progressive collapse

Never tested. Cut one bay's columns on level 1 and watch whether the falling
deck takes level 0 with it. If it does not, the impact path needs looking at;
if it does, that is the most dramatic behaviour in the whole set and it is
already free.

## What I would NOT do

- Model corrosion or time-dependent deterioration. Interesting engineering,
  irrelevant to a player.
- Chase a true punching-shear formulation in the solver. The chunk model cannot
  express the shear surface, and pre-cutting the plug gets the same visual for a
  fraction of the work.
- Tune more material numbers before doing 1 and 2. The garage currently fails
  symmetrically and late because its loading is uniform and its columns are at
  6% of capacity -- both are load and geometry problems, not strength problems.

## Order

2 and 1 first: they change what the structure IS and what it carries, and
everything else is easier to judge once the building is genuinely working for a
living. Then 5 (free measurement), then 3 and 4 (shape the failure).
